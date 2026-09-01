# Requirements — F2: Vendored Mask2Former Backbone (detectron2-free)

## Goal

Provide a standalone, detectron2-free image backbone that replaces the single-scale frozen DINOv3
encoder with a **torchvision ResNet50 → vendored `MSDeformAttnPixelDecoder`** stack, emitting three
*enhanced* multi-scale feature maps `[B, 256, Hₗ, Wₗ]` (no CLS token). The pixel decoder's internal
6-layer MSDeformAttn transformer encoder runs the F1 `DeformableAttention` primitive at
`n_levels=3`, so the maps handed downstream are already cross-scale fused. This module is the
image-side of the Mask2Former migration; F3 later wraps its output in the `MultiScaleFeatures`
bundle and F6 wires it into `MixerModel`/`PipelineBuilder`. It is additive — DINOv3 stays
selectable and untouched.

> **Deviation from the constitution (decided in this spec's planning session).** TechStack
> locked-decision 3 and the Roadmap F2 entry call for loading the **Mask2Former R50 COCO-panoptic
> checkpoint** (backbone + pixel decoder) **frozen**, with a key-remap + checksum weight loader.
> The user has revised this: use **torchvision ResNet50 ImageNet weights** for the backbone and a
> **fresh (typical) initialization** for the pixel decoder. Consequently there is **no COCO
> checkpoint, no detectron2/Caffe2 key translation, and no pixel-decoder weight loading**. Because
> a frozen randomly-initialized decoder would emit fixed random projections, freezing is now
> **granular**: the ResNet50 is frozen (like DINOv3 today), the pixel decoder is **trainable**.
> Roadmap.md / TechStack.md should be updated to reflect this; that edit is out of this spec's
> scope but is flagged in the delivery summary.

## Scope

**In scope**
- A new module file `src/model/ms_deform_backbone.py` containing:
  - `Mask2FormerBackbone` — top-level `nn.Module` (analogous to `DinoV3Wrapper`): torchvision
    ResNet50 feature extractor `{res2..res5}` → `MSDeformAttnPixelDecoder`; `forward(x) -> list of
    maps`.
  - `MSDeformAttnPixelDecoder` — detectron2-free port of the pixel decoder (input projections +
    the transformer encoder + optional stride-4 FPN branch).
  - `MSDeformAttnTransformerEncoderOnly` / `…EncoderLayer` / `…Encoder` — ported, using F1
    `DeformableAttention(n_levels=3)` as the self-attention op.
  - `PositionEmbeddingSine` — copied verbatim from Mask2Former (already detectron2-free).
- ResNet50 weights from torchvision ImageNet; **warn-and-continue** if weights cannot be fetched
  (fall back to random init, log a warning), never raise.
- Granular freezing: `freeze_backbone=True` (default; ResNet50 `requires_grad=False` **and** kept
  in `eval()` so BatchNorm running stats are frozen), `freeze_pixel_decoder=False` (default;
  trainable).
- **Optional** stride-4 (res2, 64²) enhanced feature via an FPN lateral/output-conv branch +
  `mask_features` 1×1 conv, gated by a constructor flag (default **off**; built and run only when
  requested, for a future heatmap-regression iteration).
- Pure-PyTorch throughout: the transformer's deformable op is F1's `grid_sample` path at
  `n_levels=3`. No CUDA custom op, no `MSDeformAttnFunction`, no detectron2 import anywhere in the
  import graph.
- A pytest suite `tests/test_ms_deform_backbone.py`.

**Out of scope (belongs to later features / explicitly excluded)**
- The `MultiScaleFeatures` bundle (`value`, `spatial_shapes`, `level_start_index`,
  `reference_grids`) and the DINOv3 adapter that emits it at `n_levels=1` — **F3**. F2 returns raw
  maps; F3 flattens and packages them.
- Editing `DeformableDecoder` / `DeformableDoubleInputDecoder` — **F4**.
- `MixerModel` / `PipelineBuilder` wiring, `img_input_proj` 256→`model_dim`, the
  `configs/model/image_encoder/` config group, and input **normalization** (the module assumes the
  pipeline pre-normalizes its input, mirroring `DinoV3Wrapper.forward`) — **F6**.
- Loading the Mask2Former COCO-panoptic checkpoint, any Caffe2/detectron2 key remap, and any weight
  checksum — dropped per the deviation above.
- Modifying F1 `DeformableAttention` — consumed as-is at `n_levels=3`.
- The Mask2Former segmentation head, query-based mask decoder, and `valid_ratios`/padding-mask
  machinery (images are a fixed size, so there is no padding).

## Functional Requirements

**FR1 — ResNet50 feature extractor.**
`Mask2FormerBackbone` builds `torchvision.models.resnet50` and extracts the four residual-stage
outputs via `torchvision.models.feature_extraction.create_feature_extractor` with the mapping
`{"layer1":"res2", "layer2":"res3", "layer3":"res4", "layer4":"res5"}`. For an input `x` of shape
`(B, 3, 256, 256)` the returned dict has:

| Key | Shape | Stride |
|---|---|---|
| `res2` | `(B, 256, 64, 64)` | 4 |
| `res3` | `(B, 512, 32, 32)` | 8 |
| `res4` | `(B, 1024, 16, 16)` | 16 |
| `res5` | `(B, 2048, 8, 8)` | 32 |

Weights: `ResNet50_Weights.IMAGENET1K_V2` (falling back to `IMAGENET1K_V1`, then to random init
with a `logging` warning if neither can be loaded — never raise). Input is assumed **already
normalized** by the caller (no internal mean/std subtraction).

**FR2 — Detectron2-free pixel decoder construction.**
`MSDeformAttnPixelDecoder(input_shape, conv_dim=256, mask_dim=256, transformer_dropout=0.0,
transformer_nheads=8, transformer_dim_feedforward=1024, transformer_enc_layers=6,
transformer_in_features=("res3","res4","res5"), num_points=4, common_stride=4,
return_stride4=False)`:
- `input_shape`: `Dict[str, (channels, stride)]` for `res2..res5` (from FR1).
- **Input projections**: one `nn.Sequential(nn.Conv2d(Cₗ, conv_dim, 1), nn.GroupNorm(32, conv_dim))`
  per transformer level, ordered **low→high resolution** (res5, res4, res3), matching Mask2Former.
  Weights `xavier_uniform_`, biases zero.
- **Transformer**: `MSDeformAttnTransformerEncoderOnly(d_model=conv_dim, nhead=transformer_nheads,
  num_encoder_layers=transformer_enc_layers, dim_feedforward=transformer_dim_feedforward,
  dropout=transformer_dropout, num_feature_levels=3, enc_n_points=num_points)`. It owns a
  `level_embed` `nn.Parameter(3, conv_dim)`.
- **Position embedding**: `PositionEmbeddingSine(conv_dim // 2, normalize=True)`.
- `conv_dim` is fixed at **256** by the design (do not parameterize away). `n_levels` fixed at 3.

**FR3 — F1 primitive as the deformable op.**
Each `MSDeformAttnTransformerEncoderLayer.self_attn` is
`DeformableAttention(embed_dim=conv_dim, num_heads=n_heads, num_points=n_points, n_levels=3)` from
`src/model/blocks.py` (F1). For `(conv_dim=256, n_levels=3, n_heads=8, n_points=4)` this yields the
exact Mask2Former parameter layout:

| Param | Shape |
|---|---|
| `sampling_offsets.weight` | `(192, 256)` |
| `attention_weights.weight` | `(96, 256)` |
| `value_proj.weight` | `(256, 256)` |
| `output_proj.weight` | `(256, 256)` |

The layer calls
`self_attn(query=src+pos, reference_points=ref, value=src, spatial_shape=spatial_shapes,
level_start_index=level_start_index)` (no `padding_mask` — F1 does not take one and there is no
padding). `spatial_shapes` is a `(3, 2)` `LongTensor`; `reference_points` is `(B, ΣHₗWₗ, 3, 2)`.

**FR4 — Encoder forward and reference points.**
`MSDeformAttnTransformerEncoderOnly.forward(srcs, pos_embeds) -> (memory, spatial_shapes,
level_start_index)`:
- `srcs`: list of 3 projected maps `(B, 256, Hₗ, Wₗ)` (low→high res); flattened to
  `(B, ΣHₗWₗ, 256)` and concatenated; `pos_embeds` add the per-level `level_embed`.
- `spatial_shapes`: `(3, 2)` `LongTensor` `[(8,8),(16,16),(32,32)]`; `level_start_index`: `(3,)`
  `[0, 64, 320]`.
- Reference points are the normalized per-level grid centers, shape `(B, ΣHₗWₗ, 3, 2)`, built by a
  `get_reference_points(spatial_shapes, device)` that uses `linspace(0.5, N-0.5, N)/N` per axis
  (no `valid_ratios` — all-ones, since there is no padding).
- 6 layers run in sequence; returns `memory (B, ΣHₗWₗ, 256)`.

**FR5 — `forward_features` output contract.**
`MSDeformAttnPixelDecoder.forward_features(features: Dict[str,Tensor])`:
- Projects `res3/res4/res5` (reversed to res5,res4,res3), runs the transformer, splits `memory`
  back per level and reshapes to maps `out = [res5 (B,256,8,8), res4 (B,256,16,16),
  res3 (B,256,32,32)]` (coarse→fine).
- Returns `multi_scale_features: List[Tensor]` = `out` (exactly 3 maps) when `return_stride4=False`.
- When `return_stride4=True`: additionally runs an FPN branch on `res2` — `lateral =
  Conv2d(256, 256, 1)`, top-down add of bilinearly-upsampled `out[-1]`, `output_conv =
  Conv2d(256,256,3,pad=1)+GroupNorm(32,256)+ReLU`, then `mask_features = Conv2d(256, mask_dim, 1)` —
  and returns `(multi_scale_features=[res5,res4,res3], mask_features (B, mask_dim, 64, 64),
  res2_fpn (B,256,64,64))`. The FPN + `mask_features` submodules are **only built** when
  `return_stride4=True` (fresh init means conditional param construction is safe — no checkpoint
  key mismatch).

**FR6 — `Mask2FormerBackbone.forward`.**
`forward(x: (B,3,H,W)) -> List[Tensor]` (or `(List[Tensor], Tensor)` when `return_stride4=True`,
appending the 64² map as the finest level). Steps: `features = feature_extractor(x)` →
`pixel_decoder.forward_features(features)`. Exposes `self.embed_dim = conv_dim = 256` and
`self.num_levels` (3, or 4 when `return_stride4`). Output level ordering is **coarse→fine**:
`[res5, res4, res3(, res2)]`, documented so F3 flattens consistently. No CLS token is ever produced.

**FR7 — Freezing semantics.**
- `freeze_backbone=True` (default): every ResNet50 parameter `requires_grad=False`; the ResNet50
  submodule is set to `.eval()` and kept there (BatchNorm uses frozen running stats), even when the
  parent module is `.train()`. Achieved by overriding `train(mode)` to re-assert
  `feature_extractor.eval()` when frozen.
- `freeze_pixel_decoder=False` (default): pixel-decoder parameters `requires_grad=True` and follow
  the parent's train/eval mode normally.
- Both flags are independent constructor args.

**FR8 — Purity / dependency constraints.**
- Importing `src/model/ms_deform_backbone.py` must **not** import detectron2, fvcore, or any
  compiled `MSDeformAttn` CUDA op. All conv/norm are `torch.nn` (`nn.Conv2d`, `nn.GroupNorm(32,·)`).
- The only sampling path is F1's `grid_sample`; the module is `InferenceRecorder`-compatible
  through F1's existing hooks (a level axis is present, per F1's recorded-tensor contract).

**FR9 — Error / edge conditions.**
- `x` channel dim ≠ 3 ⇒ let torchvision raise its native conv error (not re-wrapped).
- `return_stride4=True` but `res2` missing from `features` ⇒ `KeyError`/`ValueError` (should not
  happen with the FR1 extractor).
- `conv_dim` not divisible by `n_heads` ⇒ `ValueError` from F1 (surfaced at construction).
- Non-square or non-256 input is **allowed** (strides still produce valid `Hₗ,Wₗ`); shapes are
  computed dynamically, never hardcoded to 8/16/32.

## Public API Summary

```python
class Mask2FormerBackbone(nn.Module):
    def __init__(
        self,
        conv_dim: int = 256,
        n_heads: int = 8,
        n_points: int = 4,
        transformer_enc_layers: int = 6,
        transformer_dim_feedforward: int = 1024,
        transformer_dropout: float = 0.0,
        transformer_in_features: tuple = ("res3", "res4", "res5"),
        return_stride4: bool = False,          # build+return the 64² res2 map (future heatmaps)
        mask_dim: int = 256,
        freeze_backbone: bool = True,          # ResNet50 frozen + eval (BN stats frozen)
        freeze_pixel_decoder: bool = False,    # pixel decoder trainable
        imagenet_weights: str = "IMAGENET1K_V2",
        device="cpu",
        dtype=torch.float32,
    ): ...

    def forward(self, x):
        # x: (B, 3, H, W), assumed pre-normalized by the pipeline
        # -> List[Tensor] of 3 maps [B, 256, Hₗ, Wₗ], coarse→fine [res5, res4, res3]
        # -> (List[Tensor(+res2)], mask_features) when return_stride4=True
        ...

    def train(self, mode=True): ...           # re-asserts backbone.eval() when frozen


class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(self, input_shape, *, conv_dim=256, mask_dim=256,
                 transformer_dropout=0.0, transformer_nheads=8,
                 transformer_dim_feedforward=1024, transformer_enc_layers=6,
                 transformer_in_features=("res3","res4","res5"),
                 num_points=4, common_stride=4, return_stride4=False): ...
    def forward_features(self, features: dict): ...


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 dim_feedforward=1024, dropout=0.0,
                 num_feature_levels=3, enc_n_points=4): ...
    def forward(self, srcs, pos_embeds):  # -> (memory, spatial_shapes, level_start_index)
        ...


class PositionEmbeddingSine(nn.Module): ...   # copied verbatim from Mask2Former
```

## Dependencies

| Direction | Item | Notes |
|---|---|---|
| Reads | `torchvision.models.resnet50` + `feature_extraction.create_feature_extractor` | ImageNet weights, `{res2..res5}` |
| Reads | `DeformableAttention` (`src/model/blocks.py`, F1) | Used at `n_levels=3` as the transformer self-attention op |
| Reads | `torch.nn.functional.grid_sample` (via F1), `F.softmax`, `F.interpolate` | Pure PyTorch; no CUDA op |
| Reference (copy) | `../mask2former/.../pixel_decoder/msdeformattn.py`, `.../transformer_decoder/position_encoding.py` | Ported detectron2-free; registry/`@configurable`/`ShapeSpec`/`Conv2d`/`get_norm` dropped |
| Writes | 3 enhanced maps `[B,256,Hₗ,Wₗ]` (+ optional 64² map / `mask_features`) | Consumed by F3 (bundle) then F4/F6 |
| Consumed by (downstream) | F3 (`MultiScaleFeatures` producer) | Flattens F2 maps into `value` + `spatial_shapes` + `level_start_index` |
| Consumed by (downstream) | F6 (`MixerModel`, `img_input_proj` 256→`model_dim`, config group, normalization) | Wiring + input-norm live in F6 |
| Unchanged (guaranteed) | `DinoV3Wrapper`, existing single-scale path | Additive; DINOv3 remains selectable |
