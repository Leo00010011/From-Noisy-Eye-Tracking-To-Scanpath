# F6 — MixerModel + PipelineBuilder + config integration — Requirements

## Goal

Wire the multi-scale Mask2Former image backbone end-to-end so it can actually train and run
inference through `MixerModel`. F1–F4 delivered every piece in isolation — the N-level
`DeformableAttention` op (F1), the vendored `Mask2FormerBackbone` (F2), the backbone-agnostic
`MultiScaleFeatures` bundle + adapters (F3), and the multiscale-capable eye/fixation decoders
(F4) — but nothing constructs the backbone from config or feeds the bundle into the model. F6 is
the final wiring feature: it lets `PipelineBuilder` build either backbone from a new
`configs/model/image_encoder/` config group, teaches `MixerModel.encode` / `decode_fixation` to
consume the F3 bundle (per-level positional encoding via the existing `shared_gaussian` `pos_proj`
plus a learnable `level_embed`, and the F4 decoders' multi-scale dispatch), and guards every
DINOv3-only attribute access behind an explicit backbone type. The change is **strictly
retro-compatible on a dual path**: a DINOv3-backbone run keeps its exact legacy code path — CLS
slicing, `forward_features()` patch-grid PE, single `spatial_shape` tuple, `n_levels=1` decoders —
so its forward is `torch.equal` to pre-F6 ("trains identically"), and old single-scale checkpoints
load with zero missing/unexpected keys.

## Scope

**In scope**
- New config group `configs/model/image_encoder/` with `dinov3.yaml` (the current inline block,
  relocated) and `mask2former.yaml` (F2 backbone flags). `configs/model/mixer_model.yaml` gains a
  nested `defaults:` list selecting `image_encoder: dinov3` by default.
- `PipelineBuilder.build_model` (`src/training/pipeline_builder.py`): branch on
  `config.model.image_encoder.type` (default `"dinov3"`). Build a raw `DinoV3Wrapper` (as today)
  **or** a `Mask2FormerBackbone` wrapped in `Mask2FormerFeatureAdapter`; pass the new
  `image_encoder_type` and `n_image_levels` args to `MixerModel`.
- `MixerModel.__init__` (`src/model/mixer_model.py`): add `image_encoder_type="dinov3"` and
  `n_image_levels=1`; guard DINOv3-only init (`patch_resolution` from `.model.patch_size`, the
  `use_rope` `.model.rope_embed` access) behind the type; build the deformable eye/fixation decoders
  with `n_levels=n_image_levels`; add a learnable `level_embed` (`(n_image_levels, model_dim)`, zero
  init) on the Mask2Former path.
- `MixerModel.encode`: split the image branch into the **legacy DINOv3 path** (unchanged, byte-
  identical) and the **multi-scale path** (bundle → `img_input_proj` on `bundle.value` → per-level PE
  = `pos_proj(reference_grids)` + `level_embed` → deformable eye decoder with
  `spatial_shapes`/`level_start_index`). Store `self.image_spatial_shapes` /
  `self.image_level_start_index` for `decode_fixation`.
- `MixerModel.decode_fixation`: pass the stored `spatial_shapes`/`level_start_index` to the
  deformable fixation decoder (both `None` on the DINOv3 path ⇒ F4 legacy dispatch).
- `img_input_proj` reads `image_encoder.embed_dim` (already the case): 384→`model_dim` for DINOv3,
  256→`model_dim` for Mask2Former, `Identity` when equal.
- Guards: with `image_encoder_type=="mask2former"`, `use_rope`, `head_type in {argmax_regressor,
  heatmap}`, and `input_encoder=="image_features_concat"` raise a clear `ValueError` at construction
  (all depend on a single square patch grid / DINOv3 internals).

**Out of scope (explicitly)**
- **No changes to F1–F4 code** (`DeformableAttention`, `Mask2FormerBackbone`, `ms_features.py`,
  `DeformableDecoder`/`DeformableDoubleInputDecoder`). F6 only *constructs* and *calls* them.
- **The DINOv3 path is not routed through `DinoV3FeatureAdapter`.** Strict dual-path byte-identity
  requires DINOv3 to keep the raw `DinoV3Wrapper` + legacy `encode`/`decode_fixation` code. The F3
  `DinoV3FeatureAdapter` remains the tested contract (exercised by `tests/test_ms_features.py`) and
  is available if the two paths are ever unified — but F6 does not wire it. (Deviation from the naive
  "both through adapters" reading; see plan.md Context.)
- **argmax_regressor and heatmap heads with Mask2Former.** These need a single canonical patch grid;
  supporting them belongs to the deferred stride-4 / `mask_features` heatmap-regression iteration.
  F6 supports coordinate-regression heads only (`linear`, `mlp`, `multi_mlp`, `start_head`).
- **Per-level (`(B,Nq,n_levels,2)`) reference points.** The decoders pass the existing 2-D
  `(B,Nq,2)` gaze/fixation refs; F1 broadcasts them across levels (as in F4).
- **Wiring the stride-4 (res2, 64²) 4th level** as a real decoder level (`return_stride4`). F6 is
  level-count agnostic (`n_image_levels` is read from the adapter, so a 4-level bundle works if the
  flag is set), but the default Mask2Former config keeps `return_stride4: False` (3 levels).
- **DINOv3 dependency pinning, removing hardcoded Windows paths, seeded reproducibility** — separate
  roadmap items, untouched here (`dinov3.yaml` carries the existing `repo_path` verbatim).
- **`rope` support for multi-scale** and non-deformable image cross-attention (`FeatureEnhancer`)
  as the *primary* Mask2Former path. The default config uses `n_eye_decoder=4`,
  `n_feature_enhancer=0`, `use_deformable_fixation_decoder=True` — the deformable route. The
  `FeatureEnhancer`/`GatedFusion` code operates on the flattened CLS-free multi-level tokens without
  modification (no patch grid needed), but is not separately validated beyond a shape smoke test.

## Functional Requirements

**FR1 — New `image_encoder` config group.** `configs/model/image_encoder/dinov3.yaml` holds the
fields currently inline in `mixer_model.yaml` (`enabled`, `repo_path`, `name`, `weights`, `freeze`,
`regularization`, `adapter_hidden_dims`, `image_dim`) plus `type: "dinov3"` and `embed_dim: 384`.
`configs/model/image_encoder/mask2former.yaml` holds `type: "mask2former"`, `enabled: True`,
`embed_dim: 256`, and the F2 backbone flags: `conv_dim: 256`, `n_heads: 8`, `n_points: 4`,
`transformer_enc_layers: 6`, `transformer_dim_feedforward: 1024`, `transformer_dropout: 0.0`,
`transformer_in_features: ["res3","res4","res5"]`, `return_stride4: False`, `mask_dim: 256`,
`freeze_backbone: True`, `freeze_pixel_decoder: False`, `imagenet_weights: "IMAGENET1K_V2"`,
`adapter_hidden_dims: []`. `mixer_model.yaml` gains `defaults: [image_encoder: dinov3, _self_]` and
its inline `image_encoder:` block is removed. The default composed config
(`model: mixer_model`) resolves to the **same** `model.image_encoder.*` values as before F6.

**FR2 — `build_model` branches on backbone type.** `PipelineBuilder.build_model` reads
`image_encoder_type = self.config.model.image_encoder.get('type', 'dinov3')` (the `'dinov3'` default
keeps pre-F6 config snapshots — which lack a `type` key — loading unchanged).
- `type == 'dinov3'`: construct `DinoV3Wrapper(...)` exactly as today (raw, unwrapped);
  `n_image_levels = 1`.
- `type == 'mask2former'`: construct `Mask2FormerBackbone(conv_dim, n_heads, n_points,
  transformer_enc_layers, transformer_dim_feedforward, transformer_dropout, transformer_in_features,
  return_stride4, mask_dim, freeze_backbone, freeze_pixel_decoder, imagenet_weights, device=self.device)`
  and wrap it: `image_encoder = Mask2FormerFeatureAdapter(backbone)`; `n_image_levels =
  image_encoder.num_levels` (3, or 4 when `return_stride4`).

`MixerModel(...)` is constructed with the new kwargs `image_encoder_type=image_encoder_type` and
`n_image_levels=n_image_levels`. When `image_encoder.enabled` is `False`, `image_encoder=None` and
`image_encoder_type` is irrelevant (PathModel-equivalent behaviour, unchanged).

**FR3 — `MixerModel` gains `image_encoder_type` and `n_image_levels`.** Constructor signature adds
`image_encoder_type: str = "dinov3"` and `n_image_levels: int = 1` (both default to the DINOv3
single-scale behaviour, so any existing direct construction is unchanged). Stored as
`self.image_encoder_type` / `self.n_image_levels`.

**FR4 — DINOv3-only init guarded.** In `__init__`, `patch_resolution` (from
`image_encoder.model.patch_size`) is computed **only** when `image_encoder is not None and
image_encoder_type == 'dinov3'`; otherwise `self.patch_resolution = None`. The `shared_gaussian` /
`shared_gaussian_base` encoders' `patch_size=` argument receives the DINOv3 `patch_resolution[0]`
when available, else a nominal `16` (used only by `forward_features()`, which is never called on the
Mask2Former path). The `use_rope` block (which reads `image_encoder.model.rope_embed.*`) is reached
only on the DINOv3 path (guarded by FR8).

**FR5 — Deformable decoders built at `n_image_levels`.** When `use_deformable_eye_decoder`, the
`DeformableDecoder` is constructed with `n_levels=self.n_image_levels` (and `spatial_shape` = the
DINOv3 `patch_resolution` when available, else its `(16,16)` default — a legacy fallback unused on
the multi-scale path). When `use_deformable_fixation_decoder`, the `DeformableDoubleInputDecoder`
likewise gets `n_levels=self.n_image_levels`. At `n_image_levels=1` (DINOv3) the decoder state_dict
keys/shapes are byte-identical to pre-F6 (F1/F4 guarantee).

**FR6 — Learnable `level_embed` on the multi-scale path.** When `image_encoder is not None and
image_encoder_type == 'mask2former'`, `MixerModel` creates
`self.level_embed = nn.Parameter(torch.zeros(n_image_levels, model_dim))` and appends it to
`denoise_modules` (image/encoder side). It is **not** created on the DINOv3 path, so DINOv3
state_dicts are unchanged. Zero-init means it is neutral at the start of training.

**FR7 — `MixerModel.encode` dual image path.**
- **DINOv3 path** (`image_encoder_type == 'dinov3'`): the existing code runs verbatim —
  `image_src = image_encoder(image_src)` (CLS-prefixed), `img_input_proj`, `shared_gaussian`/
  `shared_gaussian_base` PE via `pos_proj.forward_features()` / `img_pos_proj.forward_features()`
  added to `image_src[:, prefix:, :]`, `feature_enhancer` / `eye_decoder` (called with
  `spatial_shapes=None` ⇒ F4 legacy dispatch). `self.image_spatial_shapes = None`,
  `self.image_level_start_index = None`.
- **Mask2Former path** (`image_encoder_type == 'mask2former'`): `bundle = image_encoder(image_src)`
  (a `MultiScaleFeatures`); `image_src = self.img_input_proj(bundle.value)` `(B, S, model_dim)`;
  per-level PE `pe = pos_enc(bundle.reference_grids.unsqueeze(0))` `(1, S, model_dim)` where
  `pos_enc` is `self.pos_proj` for `shared_gaussian` and `self.img_pos_proj` for
  `shared_gaussian_base`; add the level embedding
  `pe = pe + torch.repeat_interleave(self.level_embed, level_sizes, dim=0).unsqueeze(0)` (with
  `level_sizes = bundle.level_sizes()`); `image_src = image_src + pe`. The eye decoder (if present)
  is called with `reference_points=src_coords`, `spatial_shapes=bundle.spatial_shapes`,
  `level_start_index=bundle.level_start_index`. `self.image_spatial_shapes = bundle.spatial_shapes`,
  `self.image_level_start_index = bundle.level_start_index`. `self.final_fenh_norm_image` /
  `mixed_image_features` / `use_enh_img_features` post-processing apply to the flattened CLS-free
  `image_src` unchanged.

**FR8 — Multi-scale guards raise at construction.** With `image_encoder is not None and
image_encoder_type == 'mask2former'`, `MixerModel.__init__` raises `ValueError` when:
`use_rope` is `True` (rope needs the DINOv3 patch grid / `rope_embed`); `head_type` is
`"argmax_regressor"` or `"heatmap"` (need a single square patch grid); or `input_encoder` is
`"image_features_concat"` (indexes `image_src[:, 1:, :]` on a fixed grid). Each message names the
offending option and states it is DINOv3-only in F6.

**FR9 — `MixerModel.decode_fixation` forwards the stored geometry.** The deformable fixation
decoder call passes `spatial_shapes=self.image_spatial_shapes` and
`level_start_index=self.image_level_start_index` (both `None` ⇒ F4 legacy path on DINOv3; the bundle
tensors ⇒ multi-scale path on Mask2Former). `reference_points` construction (start point + tgt
coords) is unchanged. The non-deformable `DoubleInputDecoder` branch is unchanged (DINOv3-only in
practice). The `image_features_concat` visual-token gather (`image_src[:, 1:, :]` +
`patch_resolution`) is unreachable on the Mask2Former path (FR8 rejects it at construction).

**FR10 — Byte-identical DINOv3 forward and checkpoint load.** For a model built with
`image_encoder_type='dinov3'` and otherwise the default config, `MixerModel.forward` (all phases) is
`torch.equal` to the pre-F6 model on identical inputs and a fixed seed, and a pre-F6 checkpoint loads
via `load_state_dict` / `load_encoder` with zero missing/unexpected keys. No `level_embed`,
`image_encoder_type`, or `n_image_levels` leaks a new parameter into the DINOv3 state_dict.

**FR11 — Mask2Former forward produces correct shapes.** For `image_encoder_type='mask2former'`,
default 3 levels, `img_size=256`, batch `B`, gaze length `T`, fixation length `N`, `model_dim=512`:
`encode` sets `self.src` `(B, T, 512)` and `self.image_src` `(B, S, 512)` with `S = 8²+16²+32² =
1344`; `decode_fixation` returns the head dict with `coord`/`reg` last-dim per head type,
`cls` `(B, N+1, 1)`, `dur` `(B, N+1, ·)` (the `+1` is the prepended start token). No `NaN`/`Inf` on a
forward+backward pass; gradients reach the pixel decoder (trainable) and the `level_embed`, and do
**not** reach the frozen ResNet50.

## Public API Summary

```python
class MixerModel(nn.Module):
    def __init__(self, n_encoder, n_decoder, ..., image_encoder=None,
                 image_encoder_type="dinov3",   # NEW: "dinov3" | "mask2former"
                 n_image_levels=1,              # NEW: 1 (DINOv3) or 3/4 (Mask2Former)
                 ...): ...

    def encode(self, src, image_src, src_mask, **kwargs): ...
        # dinov3     -> legacy CLS-prefixed path; self.image_spatial_shapes = None
        # mask2former-> bundle path; PE = pos_proj(reference_grids)+level_embed;
        #               self.image_spatial_shapes / self.image_level_start_index set from bundle

    def decode_fixation(self, tgt, tgt_mask, src_mask, in_tgt=None, **kwargs): ...
        # deformable fixation decoder called with
        #   spatial_shapes=self.image_spatial_shapes, level_start_index=self.image_level_start_index
```

```python
# PipelineBuilder.build_model — MixerModel branch (sketch)
etype = self.config.model.image_encoder.get('type', 'dinov3')
if etype == 'mask2former':
    ie = self.config.model.image_encoder
    backbone = Mask2FormerBackbone(conv_dim=ie.conv_dim, n_heads=ie.n_heads, n_points=ie.n_points,
                                   transformer_enc_layers=ie.transformer_enc_layers,
                                   transformer_dim_feedforward=ie.transformer_dim_feedforward,
                                   transformer_dropout=ie.transformer_dropout,
                                   transformer_in_features=tuple(ie.transformer_in_features),
                                   return_stride4=ie.return_stride4, mask_dim=ie.mask_dim,
                                   freeze_backbone=ie.freeze_backbone,
                                   freeze_pixel_decoder=ie.freeze_pixel_decoder,
                                   imagenet_weights=ie.imagenet_weights, device=self.device)
    image_encoder = Mask2FormerFeatureAdapter(backbone)
    n_image_levels = image_encoder.num_levels
else:  # 'dinov3'
    image_encoder = DinoV3Wrapper(...)   # unchanged
    n_image_levels = 1
```

```yaml
# configs/model/mixer_model.yaml (head)
defaults:
  - image_encoder: dinov3
  - _self_
```

## Dependencies

| Reads from | Provided by | Notes |
|---|---|---|
| `Mask2FormerBackbone(...)` | F2, `src/model/ms_deform_backbone.py` | Torchvision R50 (frozen) + trainable pixel decoder; ImageNet weights, no external checkpoint. |
| `Mask2FormerFeatureAdapter(backbone)` → `MultiScaleFeatures` | F3, `src/model/ms_features.py` | Exposes `.embed_dim` (256), `.num_levels`; `forward(x)` → bundle. |
| `bundle.value / spatial_shapes / level_start_index / reference_grids / level_sizes()` | F3 `MultiScaleFeatures` | Consumed in `encode` (PE) and forwarded to the F4 decoders. |
| `DeformableDecoder(..., n_levels)` / `DeformableDoubleInputDecoder(..., n_levels)` forward `spatial_shapes`/`level_start_index` | F4, `src/model/blocks.py` | Multi-scale dispatch; `None` ⇒ byte-identical legacy path. |
| `DinoV3Wrapper` (raw) | `src/model/dino_wrapper.py` | Unchanged; DINOv3 path only. |
| `pos_proj` / `img_pos_proj` (`GaussianFourierPosEncoder`) | `src/model/pos_encoders.py` | Reused as the coordinate encoder for multi-scale memory PE (called on `reference_grids`, not `forward_features()`). |

| Writes to / touched | Notes |
|---|---|
| `configs/model/image_encoder/dinov3.yaml`, `configs/model/image_encoder/mask2former.yaml` | New group. |
| `configs/model/mixer_model.yaml` | Add `defaults`; remove inline `image_encoder:` block. |
| `src/model/mixer_model.py` | `__init__`, `encode`, `decode_fixation`; additive, DINOv3 byte-identical. |
| `src/training/pipeline_builder.py` | `build_model` MixerModel branch. |
| `tests/test_f6_integration.py` | New CPU-only suite (Mask2Former with `imagenet_weights=None`; DINOv3 byte-identity via a stub encoder). |
```
