# F6 — MixerModel + PipelineBuilder + config integration — Plan

## Context and Design Decisions

**Why F6 exists.** F1–F4 are all done and each is byte-identical / additive by construction, which
means the migration has produced a lot of *capability* and zero *usage*: `PipelineBuilder` still only
knows how to build `DinoV3Wrapper`, and `MixerModel` still hard-codes DINOv3's CLS-prefixed,
single-square-patch conventions. F6 is the join: it constructs the F2 backbone from config, packages
it with the F3 adapter, and feeds the F3 bundle through the F4 decoders. After F6 the whole migration
is reachable from a single config switch (`model/image_encoder=mask2former`).

**Dual path, not unified (locked by the planning session for F6).** The roadmap requires a
DINOv3-backbone run to *train identically* and old checkpoints to load. The tempting alternative —
route DINOv3 through `DinoV3FeatureAdapter` so both backbones share one multi-scale code path — would
change DINOv3's forward numerics: the memory PE would move from `pos_proj.forward_features()` (a grid
`forward_features` builds internally) to `pos_proj(reference_grids)` (the F3 per-token centers), and
the decoder memory would become CLS-free-passed-through instead of CLS-sliced. Weights would still
*load*, but the forward would not be `torch.equal`. Since the constitution's whole migration is built
on byte-identity, F6 keeps DINOv3 on its **exact legacy path** and only Mask2Former on the bundle
path. `MixerModel` selects between them with an explicit `image_encoder_type` string (not
`isinstance`, which is brittle across the raw wrapper vs. the adapter). Consequence: the F3
`DinoV3FeatureAdapter` is *not* wired by F6 — it stays the tested contract for a future unification.

**Why reuse `pos_proj` on `reference_grids` (not a fresh sinusoidal PE).** The model's default
`input_encoder="shared_gaussian"` exists precisely to give gaze coordinates and image patches a
*shared* positional vocabulary (a common random Fourier basis `B`). Encoding the multi-scale memory
tokens with the **same** `pos_proj` — now called on the bundle's continuous `(x, y)` centers instead
of a fixed grid — preserves that shared vocabulary across all image scales, which is the strongest
reason `shared_gaussian` was chosen in the first place. A learnable `level_embed (L, D)` (zero-init,
Deformable-DETR style) is added on top so the three scales are distinguishable; zero-init keeps it
neutral at the start of training. `reference_grids` is exactly the geometry F3 stored for this
purpose ("memory geometry F6 uses to position-encode the multi-level memory", TechStack §F3).

**Why coordinate-regression heads only.** `argmax_regressor` and `heatmap` both index a single
square `patch_resolution` grid (and `heatmap` builds a `TrajectoryHeatmapGenerator` over it). There
is no single canonical grid across three scales; the natural home for image-grid heads is the
deferred stride-4 / `mask_features` heatmap-regression iteration the roadmap already carved out. F6
rejects them at construction on the Mask2Former path rather than silently mis-shaping.

**Constitution constraints honored.** No CUDA op (F1/F2 are pure `grid_sample`); DINOv3 stays
selectable and byte-identical (additive migration); `conv_dim=256 → img_input_proj → model_dim`
contract (TechStack §"Contract changes"); the F3 bundle is the single image-feature interface;
Hydra remains the single source of truth (the new group resolves to the same values by default, so
reproducibility of pre-F6 runs is intact). The pre-F6 `data_path` NameError noted in the roadmap is
already fixed in `load_dataset` (a default is assigned before the `LOCAL_SCRATCH` branch) and is out
of F6's scope.

## Implementation Steps

### Step 1 — Create the `image_encoder` config group

**New file `configs/model/image_encoder/dinov3.yaml`** — the block currently inline in
`mixer_model.yaml`, plus a discriminator and the embed dim:

```yaml
type: "dinov3"
enabled: True
repo_path: "C:\\Users\\ulloa\\OneDrive\\Desktop\\Practicas\\projectes\\dinov3"  # verbatim; VC-path cleanup is a separate roadmap item
name: "dinov3_vits16"
weights: "dinov3_vits16_weights.pth"
freeze: True
regularization: True
adapter_hidden_dims: []
image_dim: 256
embed_dim: 384
```

**New file `configs/model/image_encoder/mask2former.yaml`**:

```yaml
type: "mask2former"
enabled: True
embed_dim: 256
conv_dim: 256
n_heads: 8
n_points: 4
transformer_enc_layers: 6
transformer_dim_feedforward: 1024
transformer_dropout: 0.0
transformer_in_features: ["res3", "res4", "res5"]
return_stride4: False
mask_dim: 256
freeze_backbone: True
freeze_pixel_decoder: False
imagenet_weights: "IMAGENET1K_V2"
adapter_hidden_dims: []
```

**Edit `configs/model/mixer_model.yaml`** — add a nested defaults list at the very top and delete
the inline `image_encoder:` block (lines 58–67):

```yaml
defaults:
  - image_encoder: dinov3
  - _self_

name: "MixerModel"
# ... (unchanged) ...
# (inline image_encoder: block removed — now composed from the group above)
```

Verify with `python -c "import hydra; ..."` / the existing compose path that
`cfg.model.image_encoder.repo_path` etc. still resolve to the pre-F6 values.

### Step 2 — `PipelineBuilder.build_model` backbone branch

**Edit `src/training/pipeline_builder.py`.** Add the imports at the top:

```python
from src.model.ms_deform_backbone import Mask2FormerBackbone
from src.model.ms_features import Mask2FormerFeatureAdapter
```

In the `model_name == 'MixerModel'` branch, replace the current DINOv3-only construction (lines
~463–474) with a type switch that also computes `n_image_levels`:

```python
image_encoder = None
image_dim = None
image_encoder_type = 'dinov3'
n_image_levels = 1
if hasattr(self.config.model, 'image_encoder') and self.config.model.image_encoder.enabled:
    ie = self.config.model.image_encoder
    image_encoder_type = ie.get('type', 'dinov3')          # default keeps pre-F6 snapshots working
    if image_encoder_type == 'mask2former':
        backbone = Mask2FormerBackbone(
            conv_dim=ie.get('conv_dim', 256), n_heads=ie.get('n_heads', 8),
            n_points=ie.get('n_points', 4),
            transformer_enc_layers=ie.get('transformer_enc_layers', 6),
            transformer_dim_feedforward=ie.get('transformer_dim_feedforward', 1024),
            transformer_dropout=ie.get('transformer_dropout', 0.0),
            transformer_in_features=tuple(ie.get('transformer_in_features', ("res3","res4","res5"))),
            return_stride4=ie.get('return_stride4', False), mask_dim=ie.get('mask_dim', 256),
            freeze_backbone=ie.get('freeze_backbone', True),
            freeze_pixel_decoder=ie.get('freeze_pixel_decoder', False),
            imagenet_weights=ie.get('imagenet_weights', 'IMAGENET1K_V2'),
            device=self.device)
        image_encoder = Mask2FormerFeatureAdapter(backbone)
        n_image_levels = image_encoder.num_levels
    else:
        image_encoder = DinoV3Wrapper(
            repo_path=ie.repo_path, model_name=ie.name, freeze=ie.freeze,
            regularization=ie.get('regularization', False), device=self.device, weights=ie.weights)
        image_dim = ie.image_dim
```

Pass the two new kwargs into the `MixerModel(...)` call (anywhere in the kwargs list):

```python
image_encoder_type = image_encoder_type,
n_image_levels = n_image_levels,
```

The `adapter_hidden_dims` kwarg already reads `self.config.model.image_encoder.get('adapter_hidden_dims', ...)`
— it works for both entries unchanged.

### Step 3 — `MixerModel.__init__`: new args, guards, `patch_resolution`, `level_embed`

**Edit `src/model/mixer_model.py`.** Add the constructor params (near `image_encoder`):

```python
image_encoder_type = "dinov3",
n_image_levels = 1,
```

Store them and set defaults early:

```python
self.image_encoder_type = image_encoder_type
self.n_image_levels = n_image_levels
self.image_spatial_shapes = None        # set per-encode()
self.image_level_start_index = None
```

**Guard `patch_resolution` (replace the current lines 131–135):**

```python
self.patch_resolution = None
if image_encoder is not None:
    img_embed_dim = image_encoder.embed_dim
    if image_encoder_type == 'dinov3':
        pr = int(self.img_size / image_encoder.model.patch_size)
        self.patch_resolution = (pr, pr)
```

Define a nominal patch size for the `shared_gaussian` encoders' `patch_size=` arg (only used by
`forward_features()`, i.e. DINOv3):

```python
pos_enc_patch_size = self.patch_resolution[0] if self.patch_resolution is not None else 16
```

and replace every `patch_size = self.patch_resolution[0]` in the `shared_gaussian` /
`shared_gaussian_base` branches with `patch_size = pos_enc_patch_size`.

**FR8 guards** — right after `self.image_encoder = image_encoder` bookkeeping (before building the
decoders/heads), add:

```python
if image_encoder is not None and image_encoder_type == 'mask2former':
    if use_rope:
        raise ValueError("use_rope is DINOv3-only in F6 (needs the patch grid / rope_embed); "
                         "set use_rope=False with the mask2former backbone.")
    if head_type in ('argmax_regressor', 'heatmap'):
        raise ValueError(f"head_type='{head_type}' is DINOv3-only in F6 (needs a single square "
                         "patch grid); use linear/mlp/multi_mlp/start_head with mask2former.")
    if input_encoder == 'image_features_concat':
        raise ValueError("input_encoder='image_features_concat' is DINOv3-only in F6 "
                         "(indexes a fixed patch grid).")
```

**`img_input_proj`** — unchanged: it already builds from `img_embed_dim = image_encoder.embed_dim`,
so the Mask2Former adapter's 256 flows through the same `MLP(256, adapter_hidden_dims, model_dim)`
(or `Identity` if `model_dim == 256`).

**Deformable decoders at `n_levels` (Step overlaps FR5).** In the eye-decoder block, pass
`n_levels=self.n_image_levels` to `DeformableDecoder(...)`; in the fixation-decoder block, pass
`n_levels=self.n_image_levels` to `DeformableDoubleInputDecoder(...)`. Keep `spatial_shape=` as is
(`self.patch_resolution` when set, else the class default `(16,16)`) — it is the legacy fallback and
is unused on the multi-scale path. At `n_image_levels=1` this is byte-identical (F4).

**`level_embed`** — after the decoders/heads, before the denoise head:

```python
if image_encoder is not None and image_encoder_type == 'mask2former':
    self.level_embed = nn.Parameter(torch.zeros(n_image_levels, model_dim, **factory_mode))
    self.denoise_modules.append(self.level_embed)
```

### Step 4 — `MixerModel.encode`: dual image path

**Edit the `if self.image_encoder is not None:` block in `encode` (lines ~683–738).** Wrap the
existing body in the DINOv3 branch and add the Mask2Former branch:

```python
if self.image_encoder is not None:
    if self.image_encoder_type == 'dinov3':
        # ---- LEGACY PATH (verbatim, byte-identical) ----
        image_src = self.image_encoder(image_src)
        image_src = self.img_input_proj(image_src)
        if self.input_encoder == 'shared_gaussian':
            pos_enc = self.pos_proj.forward_features().unsqueeze(0)
            prefix = image_src.size(1) - pos_enc.shape[1]
            image_src[:, prefix:, :] = image_src[:, prefix:, :] + pos_enc
        if self.input_encoder == 'shared_gaussian_base':
            pos_enc = self.img_pos_proj.forward_features().unsqueeze(0)
            prefix = image_src.size(1) - pos_enc.shape[1]
            image_src[:, prefix:, :] = image_src[:, prefix:, :] + pos_enc
        self.image_spatial_shapes = None
        self.image_level_start_index = None
    else:
        # ---- MULTI-SCALE PATH (Mask2Former via F3 bundle) ----
        bundle = self.image_encoder(image_src)                 # MultiScaleFeatures
        image_src = self.img_input_proj(bundle.value)          # (B, S, model_dim)
        pos_enc_mod = self.pos_proj if self.input_encoder == 'shared_gaussian' else self.img_pos_proj
        pe = pos_enc_mod(bundle.reference_grids.unsqueeze(0))   # (1, S, model_dim)
        level_sizes = bundle.level_sizes()
        lvl = torch.repeat_interleave(self.level_embed, 
                                      torch.tensor(level_sizes, device=self.level_embed.device),
                                      dim=0).unsqueeze(0)       # (1, S, model_dim)
        image_src = image_src + pe + lvl
        self.image_spatial_shapes = bundle.spatial_shapes
        self.image_level_start_index = bundle.level_start_index

    # ---- shared tail (both backbones): enhancer / mix / adapter / eye decoder ----
    if self.n_feature_enhancer > 0 and not (self.n_eye_decoder > 0):
        img_enh = image_src
        for mod in self.feature_enhancer:
            src_rope = image_rope = None
            if self.use_rope:                                  # DINOv3-only (FR8 blocks m2f)
                src_rope, image_rope = self.rope_pos(traj_coords=src_coords, patch_res=self.patch_resolution)
            src, img_enh = mod(src, img_enh, src1_mask=src_mask, src2_mask=None,
                               src1_rope=src_rope, src2_rope=image_rope)
        if self.norm_first:
            src = self.final_fenh_norm_src(src)
    if self.mixed_image_features:
        ... (unchanged)
    elif self.use_enh_img_features:
        ... (unchanged)
    else:
        image_src = self.final_fenh_norm_image(image_src)
    if self.n_adapter > 0:
        ... (unchanged)
    if self.n_eye_decoder > 0:
        for mod in self.eye_decoder:
            if self.use_deformable_eye_decoder:
                src = mod(src, image_src, src_mask, reference_points=src_coords,
                          spatial_shapes=self.image_spatial_shapes,
                          level_start_index=self.image_level_start_index)
            else:
                src = mod(src, image_src, src_mask, None)
        if self.norm_first:
            src = self.final_fenh_norm_src(src)
self.src = src
self.image_src = image_src
self.src_coords = src_coords
```

Notes: on the DINOv3 path `self.image_spatial_shapes is None`, so the eye decoder call is the F4
legacy dispatch — byte-identical. `pos_proj(reference_grids.unsqueeze(0))` uses the coordinate-
encoding `forward`, not `forward_features()`, so it works for arbitrary token counts and per-level
grids.

### Step 5 — `MixerModel.decode_fixation`: forward the geometry

**Edit the deformable branch of the decoder loop (lines ~824–834):**

```python
if self.use_deformable_fixation_decoder:
    start_point = torch.full((1, 1, 2), 0.5, device=src.device, dtype=src.dtype).expand(src.size(0), -1, -1)
    if tgt_coords is None:
        reference_points = start_point
    else:
        reference_points = torch.cat([start_point, tgt_coords], dim=1)
    output = mod(output, src, image_src, tgt_mask, mem1_mask=src_mask,
                 reference_points=reference_points,
                 spatial_shapes=self.image_spatial_shapes,
                 level_start_index=self.image_level_start_index)
else:
    output = mod(output, image_src, src, tgt_mask, mem2_mask=src_mask,
                 src_rope=tgt_rope, mem1_rope=image_rope, mem2_rope=src_rope)   # unchanged
```

On DINOv3 both stored tensors are `None` ⇒ F4 legacy dispatch (byte-identical). The
`image_features_concat` visual-token gather earlier in `decode_fixation` is unreachable on the
Mask2Former path (FR8 rejects that `input_encoder` at construction) and is left unchanged for DINOv3.

### Step 6 — Tests

**New file `tests/test_f6_integration.py`** (CPU-only; `imagenet_weights=None` so no network;
DINOv3 byte-identity uses a lightweight stub encoder exposing `.embed_dim`, `.model.patch_size`, and
a CLS-prefixed `forward`). Groups per validation.md. Reuse the F2/F3 test stubs where practical.

## Implementation Order

1. **Step 1** — config group (`dinov3.yaml`, `mask2former.yaml`, `mixer_model.yaml` defaults).
2. **Step 2** — `PipelineBuilder.build_model` backbone branch + new `MixerModel` kwargs.
3. **Step 3** — `MixerModel.__init__` (args, `patch_resolution` guard, FR8 guards, decoder
   `n_levels`, `level_embed`).
4. **Step 4** — `MixerModel.encode` dual path.
5. **Step 5** — `MixerModel.decode_fixation` geometry forwarding.
6. **Step 6** — `tests/test_f6_integration.py`, then update Roadmap/TechStack marking F6 done.
```
