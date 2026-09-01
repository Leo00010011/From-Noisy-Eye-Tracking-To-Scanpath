# Plan — F2: Vendored Mask2Former Backbone (detectron2-free)

## Context and Design Decisions

**What F2 delivers and why now.** The migration (TechStack §"Multi-scale Image Backbone Migration")
needs an image-side that produces *enhanced, multi-scale, CLS-free* features so the eye/fixation
decoders (F4) can sample across scales. F1 already generalized the deformable primitive to
`n_levels`; F2 is the first consumer of it at `n_levels=3` and the first concrete backbone. It is
built before F4 (which reworks the decoders) because F4's contract depends on what F2 emits.

**Deviation from the locked decisions — ImageNet R50 + fresh pixel decoder (user-directed).**
TechStack locked-decision 3 called for the Mask2Former R50 **COCO-panoptic** checkpoint (backbone +
pixel decoder), frozen, with a Caffe2/detectron2 key-remap + checksum loader. The user revised this
in planning:
1. Backbone = **torchvision ResNet50 ImageNet** weights.
2. Pixel decoder = **fresh (typical) init** — Mask2Former's own `_reset_parameters` / xavier init,
   no pretrained weights.
3. Therefore **no COCO checkpoint, no C2 key translation, no weight checksum** — the entire
   "F2 weight loader" described in the Roadmap collapses to "torchvision fetches ImageNet weights."
4. Because a **frozen** randomly-initialized pixel decoder would emit fixed random projections,
   freezing becomes **granular**: ResNet50 frozen (matching how DINOv3 is used), pixel decoder
   **trainable**. This is the only coherent reading of "frozen backbone" once the decoder is not
   pretrained.

This also *simplifies* F1's role: F1 no longer needs to load pretrained pixel-decoder weights by
name — it is instantiated fresh at `n_levels=3`. F1's byte-identity guarantee at `n_levels=1` is
irrelevant here and untouched. Roadmap.md / TechStack.md still describe the COCO path and should be
updated; that edit is flagged in the delivery summary, not performed by this spec.

**"Warn and continue" applies to weight fetching.** With no custom checkpoint, the only weight I/O
is torchvision's ImageNet download. Per the user's strictness choice, a failed fetch (offline
machine, torchvision hub error) **logs a warning and falls back to random init** rather than
raising, so bring-up never hard-blocks on network state. Freezing still applies to whatever weights
end up loaded.

**Detectron2-free port (TechStack locked-decision 1).** The Mask2Former pixel decoder is copied and
stripped: `@configurable`/`from_config`/`SEM_SEG_HEADS_REGISTRY`/`ShapeSpec` dropped; detectron2's
`Conv2d`+`get_norm` → `nn.Conv2d` + `nn.GroupNorm(32, ·)`; `fvcore` weight-init helpers → plain
`nn.init.xavier_uniform_`; `PositionEmbeddingSine` copied verbatim (it is already detectron2-free).
The `input_shape: Dict[str, ShapeSpec]` argument becomes `Dict[str, (channels, stride)]` tuples.

**F1 is the only deformable op (locked-decision 5).** The transformer's `self_attn` is F1's
`DeformableAttention(n_levels=3)`, not Mask2Former's `MSDeformAttn` (which needs the CUDA
`MSDeformAttnFunction`). F1's `forward(query, reference_points, value, spatial_shape,
level_start_index)` matches what the encoder layer needs; there is **no `padding_mask`** because all
images are the same size — so the entire `masks`/`valid_ratios` apparatus of the original is
dropped, and `get_reference_points` uses plain `linspace(0.5,N-0.5,N)/N` grids.

**No internal normalization (user decision).** Like `DinoV3Wrapper.forward`, the module assumes its
input is already normalized by the pipeline. Mask2Former's `pixel_mean/std` buffers are **not**
registered; wiring the correct stats is F6's job.

**Optional stride-4 branch (user decision).** Current work is coordinate regression, which needs
only the 3 transformer-enhanced levels (res5/res4/res3). A later heatmap-regression iteration will
want the stride-4 (res2, 64²) map. So the FPN lateral/output convs + `mask_features` 1×1 conv are
built **and** run only when `return_stride4=True`. Fresh init makes conditional param construction
safe (no checkpoint keys to satisfy).

**Level ordering.** The transformer internally orders levels low→high resolution (res5, res4, res3)
and flattens the memory in that order (`level_start_index = [0, 64, 320]`). F2 returns the split
maps in the same **coarse→fine** order `[res5, res4, res3]`, appending res2 (64²) as the finest 4th
level when requested. This monotonic ordering is documented so F3 can flatten deterministically.

**Files.** New: `src/model/ms_deform_backbone.py` (all classes below), `tests/test_ms_deform_backbone.py`.
No existing file is modified in F2 (F1's `blocks.py` is imported, not edited).

---

## Step 1 — `PositionEmbeddingSine` (copy verbatim)

Copy `PositionEmbeddingSine` from
`../mask2former/mask2former/modeling/transformer_decoder/position_encoding.py` into
`ms_deform_backbone.py` unchanged. It is already detectron2-free (pure torch). Used at
`num_pos_feats = conv_dim // 2 = 128`, `normalize=True`.

## Step 2 — Transformer encoder layer (F1 self-attn)

Port `MSDeformAttnTransformerEncoderLayer` with the self-attention swapped for F1:

```python
from src.model.blocks import DeformableAttention

class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.0,
                 n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = DeformableAttention(embed_dim=d_model, num_heads=n_heads,
                                             num_points=n_points, n_levels=n_levels)
        self.dropout1 = nn.Dropout(dropout); self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn); self.activation = F.relu
        self.dropout2 = nn.Dropout(dropout); self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout); self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(t, pos): return t if pos is None else t + pos

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points,
                              src, spatial_shapes, level_start_index)   # F1: no padding_mask
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout3(src2))
        return src
```

Note `value=src` (self-attention over the flattened multi-scale memory); F1's `value_proj` projects
it internally.

## Step 3 — Transformer encoder (reference points, no valid_ratios)

```python
class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__(); self.layers = _get_clones(layer, num_layers)

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        ref_list = []
        for (H, W) in spatial_shapes.tolist():
            ry, rx = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, device=device),
                torch.linspace(0.5, W - 0.5, W, device=device), indexing='ij')
            ry = ry.reshape(-1)[None] / H                # (1, H*W)
            rx = rx.reshape(-1)[None] / W
            ref_list.append(torch.stack((rx, ry), -1))   # (1, H*W, 2)
        ref = torch.cat(ref_list, 1)                     # (1, ΣHW, 2)
        return ref[:, :, None].repeat(1, 1, spatial_shapes.shape[0], 1)  # (1, ΣHW, L, 2)

    def forward(self, src, spatial_shapes, level_start_index, pos):
        ref = self.get_reference_points(spatial_shapes, src.device).expand(src.shape[0], -1, -1, -1)
        out = src
        for layer in self.layers:
            out = layer(out, pos, ref, spatial_shapes, level_start_index)
        return out
```

`_get_clones(m, n) = nn.ModuleList(copy.deepcopy(m) for _ in range(n))` (inline it; do not import
detectron2's).

## Step 4 — `MSDeformAttnTransformerEncoderOnly`

Port the wrapper; drop `masks`/`valid_ratios`. Owns `level_embed = nn.Parameter(torch.empty(3,
d_model))`, `normal_`-initialized.

```python
def forward(self, srcs, pos_embeds):
    src_flatten, pos_flatten, spatial_shapes = [], [], []
    for lvl, (src, pos) in enumerate(zip(srcs, pos_embeds)):
        b, c, h, w = src.shape
        spatial_shapes.append((h, w))
        src = src.flatten(2).transpose(1, 2)                    # (B, HW, C)
        pos = pos.flatten(2).transpose(1, 2) + self.level_embed[lvl].view(1, 1, -1)
        src_flatten.append(src); pos_flatten.append(pos)
    src_flatten = torch.cat(src_flatten, 1)
    pos_flatten = torch.cat(pos_flatten, 1)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)),
                                   spatial_shapes.prod(1).cumsum(0)[:-1]))
    memory = self.encoder(src_flatten, spatial_shapes, level_start_index, pos_flatten)
    return memory, spatial_shapes, level_start_index
```

`_reset_parameters`: `xavier_uniform_` every `p.dim() > 1`, then `normal_(self.level_embed)` — F1's
`DeformableAttention` re-inits itself in its own constructor (star pattern preserved), so **do not**
xavier-overwrite its params: guard the generic xavier loop to skip parameters owned by a
`DeformableAttention` submodule (iterate `self.named_parameters()` and skip names containing
`self_attn.`), or run the xavier loop *before* constructing nothing else — simplest is to let each
`DeformableAttention` keep its own init and only xavier the encoder's linears/`input_proj`. Document
this in a comment.

## Step 5 — `MSDeformAttnPixelDecoder` (detectron2-free)

```python
class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(self, input_shape, *, conv_dim=256, mask_dim=256,
                 transformer_dropout=0.0, transformer_nheads=8,
                 transformer_dim_feedforward=1024, transformer_enc_layers=6,
                 transformer_in_features=("res3","res4","res5"),
                 num_points=4, common_stride=4, return_stride4=False):
        super().__init__()
        self.transformer_in_features = sorted(transformer_in_features,
                                              key=lambda k: input_shape[k][1])  # res3,res4,res5
        in_ch = [input_shape[k][0] for k in self.transformer_in_features]
        # input_proj low->high res (res5,res4,res3): reverse
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, conv_dim, 1), nn.GroupNorm(32, conv_dim))
            for c in in_ch[::-1]])
        for p in self.input_proj:
            nn.init.xavier_uniform_(p[0].weight, gain=1); nn.init.constant_(p[0].bias, 0)
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim, nhead=transformer_nheads,
            num_encoder_layers=transformer_enc_layers,
            dim_feedforward=transformer_dim_feedforward, dropout=transformer_dropout,
            num_feature_levels=len(in_ch), enc_n_points=num_points)
        self.pe_layer = PositionEmbeddingSine(conv_dim // 2, normalize=True)
        self.conv_dim = conv_dim
        self.return_stride4 = return_stride4
        if return_stride4:
            c_res2 = input_shape["res2"][0]
            self.lateral_res2 = nn.Conv2d(c_res2, conv_dim, 1)
            self.output_res2 = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, 3, padding=1),
                nn.GroupNorm(32, conv_dim), nn.ReLU())
            self.mask_features = nn.Conv2d(conv_dim, mask_dim, 1)
            for m in (self.lateral_res2, self.output_res2[0], self.mask_features):
                nn.init.xavier_uniform_(m.weight); nn.init.constant_(m.bias, 0)

    def forward_features(self, features):
        srcs, pos = [], []
        for f in self.transformer_in_features[::-1]:          # res5,res4,res3
            x = features[f]
            srcs.append(self.input_proj[len(srcs)](x)); pos.append(self.pe_layer(x))
        memory, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        b = memory.shape[0]
        sizes = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        splits = torch.split(memory, sizes, dim=1)
        out = [z.transpose(1, 2).view(b, -1, int(h), int(w))
               for z, (h, w) in zip(splits, spatial_shapes.tolist())]  # [res5,res4,res3]
        if not self.return_stride4:
            return out
        cur = self.lateral_res2(features["res2"]) + F.interpolate(
            out[-1], size=features["res2"].shape[-2:], mode="bilinear", align_corners=False)
        res2_fpn = self.output_res2(cur)                      # (B,256,64,64)
        return out, self.mask_features(res2_fpn), res2_fpn
```

## Step 6 — `Mask2FormerBackbone` (top level + freezing)

```python
class Mask2FormerBackbone(nn.Module):
    def __init__(self, conv_dim=256, n_heads=8, n_points=4, transformer_enc_layers=6,
                 transformer_dim_feedforward=1024, transformer_dropout=0.0,
                 transformer_in_features=("res3","res4","res5"), return_stride4=False,
                 mask_dim=256, freeze_backbone=True, freeze_pixel_decoder=False,
                 imagenet_weights="IMAGENET1K_V2", device="cpu", dtype=torch.float32):
        super().__init__()
        weights = _load_imagenet_weights(imagenet_weights)   # try V2 -> V1 -> None, warn on fail
        resnet = torchvision.models.resnet50(weights=weights)
        self.feature_extractor = create_feature_extractor(
            resnet, {"layer1":"res2","layer2":"res3","layer3":"res4","layer4":"res5"})
        input_shape = {"res2":(256,4), "res3":(512,8), "res4":(1024,16), "res5":(2048,32)}
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
            for p in self.feature_extractor.parameters(): p.requires_grad = False
        if freeze_pixel_decoder:
            for p in self.pixel_decoder.parameters(): p.requires_grad = False
        self.to(device=device, dtype=dtype)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.feature_extractor.eval()        # keep BN running stats frozen
        return self

    def forward(self, x):
        feats = self.feature_extractor(x)        # dict res2..res5
        return self.pixel_decoder.forward_features(feats)
```

`_load_imagenet_weights(name)`: `try` `getattr(ResNet50_Weights, name)`; on any exception fall to
`IMAGENET1K_V1`; on failure again return `None` and `logging.getLogger(__name__).warning(...)`.

## Step 7 — Tests (`tests/test_ms_deform_backbone.py`)

Pytest module covering Groups 1–6 of `validation.md`. CPU, `img_size` small where possible (use 256
for the reference shapes, but also a 128 case to prove dynamic shapes). Fixed seeds. Mirror the
style of `tests/test_ms_deformable_attention.py`.

---

## Implementation Order

1. **Step 1** — copy `PositionEmbeddingSine`.
2. **Step 2** — `MSDeformAttnTransformerEncoderLayer` with F1 self-attn.
3. **Step 3** — `MSDeformAttnTransformerEncoder` + `get_reference_points` (no valid_ratios) + inline `_get_clones`.
4. **Step 4** — `MSDeformAttnTransformerEncoderOnly` (drop masks; `level_embed`; guard xavier from F1 params).
5. **Step 5** — `MSDeformAttnPixelDecoder` (input_proj, transformer, optional stride-4 FPN + mask_features).
6. **Step 6** — `Mask2FormerBackbone` (torchvision R50 extractor, granular freezing, `train()` override, weight fallback).
7. **Step 7** — `tests/test_ms_deform_backbone.py` (Groups 1–6).
