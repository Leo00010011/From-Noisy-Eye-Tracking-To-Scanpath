"""F2 — Vendored Mask2Former backbone (detectron2-free).

A standalone image backbone that replaces the single-scale frozen DINOv3 encoder with a
``torchvision ResNet50`` feature extractor feeding a vendored ``MSDeformAttnPixelDecoder``.
It emits three *enhanced* multi-scale feature maps ``[B, 256, Hₗ, Wₗ]`` (no CLS token); the
pixel decoder's internal 6-layer transformer encoder runs the F1 ``DeformableAttention``
primitive (``src/model/blocks.py``) at ``n_levels=3``, so the maps are already cross-scale
fused.

Ported detectron2-free from ``../mask2former/mask2former/modeling/pixel_decoder/msdeformattn.py``
and ``.../transformer_decoder/position_encoding.py``:

* ``@configurable`` / ``from_config`` / ``SEM_SEG_HEADS_REGISTRY`` / ``ShapeSpec`` dropped; the
  ``input_shape`` argument is a plain ``Dict[str, (channels, stride)]``.
* detectron2 ``Conv2d`` + ``get_norm`` → ``nn.Conv2d`` + ``nn.GroupNorm(32, ·)``.
* ``fvcore`` weight-init helpers → ``nn.init.xavier_uniform_``.
* The CUDA ``MSDeformAttn`` op → F1's pure-PyTorch ``grid_sample`` ``DeformableAttention``.
* The ``masks`` / ``valid_ratios`` / ``padding_mask`` machinery is removed — all images are a
  fixed size, so there is no padding; reference points use plain ``linspace(0.5, N-0.5, N)/N``.

Deviation from the constitution (this spec's planning session): the constitution's F2 entry
called for the Mask2Former R50 **COCO-panoptic** checkpoint (backbone + pixel decoder), frozen,
with a Caffe2 key-remap + checksum loader. The user revised this to **torchvision ResNet50
ImageNet weights** for the backbone and a **fresh initialization** for the pixel decoder — so
there is no COCO checkpoint, no key translation, and no weight checksum. Because a *frozen*
randomly-initialized decoder would emit fixed random projections, freezing is now **granular**:
the ResNet50 is frozen (like DINOv3 today), the pixel decoder is trainable.
"""

import copy
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from src.model.blocks import DeformableAttention

logger = logging.getLogger(__name__)


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# ---------------------------------------------------------------------------
# Step 1 — PositionEmbeddingSine (copied verbatim from Mask2Former; already
# detectron2-free / pure torch).
# ---------------------------------------------------------------------------
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 2 — Transformer encoder layer (F1 self-attention).
# ---------------------------------------------------------------------------
class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.0,
                 n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        # F1's grid_sample DeformableAttention replaces Mask2Former's CUDA MSDeformAttn.
        # forward(query, reference_points, value, spatial_shape, level_start_index) — no
        # padding_mask (fixed image size ⇒ no padding).
        self.self_attn = DeformableAttention(embed_dim=d_model, num_heads=n_heads,
                                             num_points=n_points, n_levels=n_levels)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        # value=src: self-attention over the flattened multi-scale memory. F1's value_proj
        # projects it internally.
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points,
                              src, spatial_shapes, level_start_index)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


# ---------------------------------------------------------------------------
# Step 3 — Transformer encoder (reference points, no valid_ratios).
# ---------------------------------------------------------------------------
class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        # No valid_ratios (all-ones, since there is no padding): plain normalized grid centers.
        reference_points_list = []
        for (H, W) in spatial_shapes.tolist():
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / H                # (1, H*W)
            ref_x = ref_x.reshape(-1)[None] / W
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))  # (1, H*W, 2)
        reference_points = torch.cat(reference_points_list, 1)             # (1, ΣHW, 2)
        # (1, ΣHW, L, 2): same reference used for every level (no valid_ratios scaling).
        return reference_points[:, :, None].repeat(1, 1, spatial_shapes.shape[0], 1)

    def forward(self, src, spatial_shapes, level_start_index, pos):
        reference_points = self.get_reference_points(spatial_shapes, src.device)
        reference_points = reference_points.expand(src.shape[0], -1, -1, -1)  # (B, ΣHW, L, 2)
        output = src
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index)
        return output


# ---------------------------------------------------------------------------
# Step 4 — MSDeformAttnTransformerEncoderOnly (drop masks / valid_ratios).
# ---------------------------------------------------------------------------
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 dim_feedforward=1024, dropout=0.0,
                 num_feature_levels=3, enc_n_points=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model, dim_feedforward, dropout,
            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.empty(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier-init the encoder's own linears / input side, but DO NOT overwrite the F1
        # DeformableAttention parameters: F1 re-inits itself (star-pattern sampling_offsets bias,
        # zeroed weights, xavier value/output proj) in its own constructor, and xavier-flooding
        # would destroy that layout. Skip any parameter owned by a `self_attn.` submodule.
        for name, p in self.named_parameters():
            if "self_attn." in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.level_embed)

    def forward(self, srcs, pos_embeds):
        # No masks / valid_ratios: images are a fixed size, so there is no padding.
        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src = src.flatten(2).transpose(1, 2)                       # (B, HW, C)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)                        # (B, ΣHW, C)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, lvl_pos_embed_flatten)
        return memory, spatial_shapes, level_start_index


# ---------------------------------------------------------------------------
# Step 5 — MSDeformAttnPixelDecoder (detectron2-free).
# ---------------------------------------------------------------------------
class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(self, input_shape, *, conv_dim=256, mask_dim=256,
                 transformer_dropout=0.0, transformer_nheads=8,
                 transformer_dim_feedforward=1024, transformer_enc_layers=6,
                 transformer_in_features=("res3", "res4", "res5"),
                 num_points=4, common_stride=4, return_stride4=False):
        """
        Args:
            input_shape: Dict[str, (channels, stride)] for res2..res5.
            conv_dim: fixed at 256 by design (the F1 attention layout and the img_input_proj
                256→model_dim contract F6 expects).
            return_stride4: build+run the FPN + mask_features stride-4 (res2, 64²) branch.
        """
        super().__init__()
        # Transformer levels sorted low→high resolution by stride: res3(8), res4(16), res5(32).
        self.transformer_in_features = sorted(transformer_in_features,
                                              key=lambda k: input_shape[k][1])
        in_ch = [input_shape[k][0] for k in self.transformer_in_features]

        # Input projections ordered low→high resolution (res5, res4, res3): reverse in_ch.
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim))
            for c in in_ch[::-1]])
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim, nhead=transformer_nheads,
            num_encoder_layers=transformer_enc_layers,
            dim_feedforward=transformer_dim_feedforward, dropout=transformer_dropout,
            num_feature_levels=len(in_ch), enc_n_points=num_points)
        self.pe_layer = PositionEmbeddingSine(conv_dim // 2, normalize=True)
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.common_stride = common_stride
        self.return_stride4 = return_stride4

        # Optional stride-4 (res2, 64²) FPN branch + mask_features. Fresh init means building
        # these conditionally is safe (no checkpoint keys to satisfy).
        if return_stride4:
            c_res2 = input_shape["res2"][0]
            self.lateral_res2 = nn.Conv2d(c_res2, conv_dim, kernel_size=1)
            self.output_res2 = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, conv_dim), nn.ReLU())
            self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=1)
            for m in (self.lateral_res2, self.output_res2[0], self.mask_features):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, features):
        srcs, pos = [], []
        for f in self.transformer_in_features[::-1]:              # res5, res4, res3
            x = features[f]
            srcs.append(self.input_proj[len(srcs)](x))
            pos.append(self.pe_layer(x))
        memory, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        b = memory.shape[0]
        sizes = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        splits = torch.split(memory, sizes, dim=1)
        # out ordered coarse→fine: [res5 (8²), res4 (16²), res3 (32²)].
        out = [z.transpose(1, 2).view(b, -1, int(h), int(w))
               for z, (h, w) in zip(splits, spatial_shapes.tolist())]
        if not self.return_stride4:
            return out
        cur = self.lateral_res2(features["res2"]) + F.interpolate(
            out[-1], size=features["res2"].shape[-2:], mode="bilinear", align_corners=False)
        res2_fpn = self.output_res2(cur)                         # (B, 256, 64, 64)
        return out, self.mask_features(res2_fpn), res2_fpn


# ---------------------------------------------------------------------------
# Step 6 — Mask2FormerBackbone (top level + granular freezing).
# ---------------------------------------------------------------------------
def _load_imagenet_weights(name):
    """Fetch torchvision ImageNet weights; warn-and-continue (→ random init) on failure."""
    if name is None:
        return None
    try:
        weights = getattr(ResNet50_Weights, name)
        # Trigger the actual download/verification so a fetch failure is caught here.
        weights.get_state_dict(progress=False)
        return weights
    except Exception as first_err:
        try:
            weights = ResNet50_Weights.IMAGENET1K_V1
            weights.get_state_dict(progress=False)
            logger.warning(
                "Could not load ResNet50 weights '%s' (%s); fell back to IMAGENET1K_V1.",
                name, first_err)
            return weights
        except Exception as second_err:
            logger.warning(
                "Could not load any ImageNet ResNet50 weights (%s / %s); "
                "falling back to random init.", first_err, second_err)
            return None


class Mask2FormerBackbone(nn.Module):
    def __init__(self, conv_dim=256, n_heads=8, n_points=4, transformer_enc_layers=6,
                 transformer_dim_feedforward=1024, transformer_dropout=0.0,
                 transformer_in_features=("res3", "res4", "res5"), return_stride4=False,
                 mask_dim=256, freeze_backbone=True, freeze_pixel_decoder=False,
                 imagenet_weights="IMAGENET1K_V2", device="cpu", dtype=torch.float32):
        super().__init__()
        weights = _load_imagenet_weights(imagenet_weights)
        resnet = torchvision.models.resnet50(weights=weights)
        self.feature_extractor = create_feature_extractor(
            resnet, {"layer1": "res2", "layer2": "res3", "layer3": "res4", "layer4": "res5"})
        # (channels, stride) for res2..res5 at any input size (strides fixed by ResNet50).
        input_shape = {"res2": (256, 4), "res3": (512, 8), "res4": (1024, 16), "res5": (2048, 32)}
        self.pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape, conv_dim=conv_dim, mask_dim=mask_dim,
            transformer_dropout=transformer_dropout, transformer_nheads=n_heads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_enc_layers=transformer_enc_layers,
            transformer_in_features=transformer_in_features,
            num_points=n_points, return_stride4=return_stride4)

        self.embed_dim = conv_dim
        self.return_stride4 = return_stride4
        self.num_levels = 4 if return_stride4 else 3
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
        if freeze_pixel_decoder:
            for p in self.pixel_decoder.parameters():
                p.requires_grad = False
        self.to(device=device, dtype=dtype)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.feature_extractor.eval()   # keep BatchNorm running stats frozen
        return self

    def forward(self, x):
        # x: (B, 3, H, W), assumed pre-normalized by the pipeline (no internal mean/std).
        features = self.feature_extractor(x)          # dict res2..res5, no CLS
        if not self.return_stride4:
            return self.pixel_decoder.forward_features(features)   # list of 3 maps
        out, mask_features, res2_fpn = self.pixel_decoder.forward_features(features)
        # Append the 64² res2 map as the finest level; return mask_features alongside.
        return out + [res2_fpn], mask_features
