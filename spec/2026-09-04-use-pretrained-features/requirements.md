# Requirements — Use Pretrained (Frozen) Image Features

## Goal

Training the `MixerModel` with the online Mask2Former backbone (`model/image_encoder=mask2former`)
is slow and overfits: every training step re-runs a ResNet50 + a 6-layer MSDeformAttn pixel decoder
over the stimulus image, and enlarging the image to give the deformable attention more spatial
budget made it slower still. Following the common "frozen features" recipe (e.g. ViT-Adapter,
frozen-backbone scanpath work), this feature **precomputes the image features once with a
pretrained-and-frozen Mask2Former backbone and caches them to disk**, then feeds those cached
features to `MixerModel` in place of the live backbone forward. Because the backbone is now truly
pretrained (COCO-tuned ResNet50 + COCO-tuned pixel decoder, both loaded from the user's `.pkl`
files) *and* frozen, the cached features are meaningful and stable, the per-step cost of the image
path drops to a disk read, and the trainable image adapters (`img_input_proj`, positional
encoders, `level_embed`) plus the deformable decoders continue to learn on top of them.

## Scope

**In scope**
- A pretrained-weight loader that populates the existing `Mask2FormerBackbone` from the two
  detectron2-format checkpoints in `pretrained_models/`:
  - `M2F_R50.pkl` — COCO-panoptic-tuned ResNet50, **detectron2 naming** (`stem.*`, `stages.resN.*`,
    `.norm.`, `shortcut`) → remapped to torchvision naming (`conv1`, `bn1`, `layerN`, `downsample`).
  - `M2F_R50_MSDeformAttnPixelDecoder.pkl` — the pixel decoder, keys already 1:1 with the vendored
    `MSDeformAttnPixelDecoder` under a `pixel_decoder.` prefix.
- A **precompute script** that builds the backbone (`return_stride4=True`, `transformer_enc_layers=6`,
  everything frozen), loads both checkpoints, runs it once over every **unique** stimulus image, and
  writes a keyed HDF5 feature cache.
- The cache stores, per unique image, **both** feature products needed downstream: the 3-level
  deformable memory `ms_value [S, 256]` (flattened `[res5, res4, res3]`, coarse→fine) **and** the
  stride-4 `mask_features [256, 64, 64]` (reserved for a future heatmap-regression head).
- A **feature dataset** (`PrecomputedFeatureDataset`) that replaces `DeduplicatedMemoryDataset`:
  instead of loading/resizing images it returns the cached `ms_value` for each sample's unique
  image, keyed by the identical first-seen unique-image ordering.
- A **minimal, drop-in model adaptation**: a zero-parameter `PrecomputedFeatureAdapter` (an
  `nn.Module` that reconstructs a `MultiScaleFeatures` bundle from a cached `ms_value` batch + the
  fixed geometry) is installed as `MixerModel.image_encoder`. Because it satisfies the exact F3
  bundle contract that the online `Mask2FormerFeatureAdapter` satisfies, **`MixerModel` needs no
  edits** — `self.image_encoder(image_src)` returns a bundle whether it runs a backbone or replays
  a cached tensor.
- `PipelineBuilder` wiring: a `mask2former_precomputed` image-encoder config that builds the stub
  adapter (no backbone), and a data flag that swaps the image dataset/dataloader.
- Additive throughout: the online DINOv3 and online Mask2Former paths stay byte-identical and
  selectable; no existing checkpoint or training-HDF5 layout changes.

**Explicitly out of scope**
- Wiring `mask_features` into an actual heatmap head or the deformable decoders as a 4th level — it
  is cached "just in case" and left unconsumed by the model in this feature (the F6 `heatmap`/4th-level
  deferrals stand).
- Fine-tuning or unfreezing any part of the backbone (ResNet50 or pixel decoder). The backbone is
  frozen for both precompute and training; the pixel decoder no longer trains.
- Precomputing DINOv3 features, or a generic multi-backbone cache. This feature targets the
  Mask2Former path only.
- Changing input normalization: the M2F R50 was trained with **RGB ImageNet** normalization
  (`PIXEL_MEAN=[123.675,116.280,103.530]`, `PIXEL_STD=[58.395,57.120,57.375]`, i.e. the 0–1 ImageNet
  mean/std already applied by `PipelineBuilder.make_transform`), so the existing transform is reused
  unchanged.
- The EVE real-noise inference path.

## Functional Requirements

### Pretrained weight loading

**FR1 — detectron2→torchvision ResNet50 remap.** `remap_detectron2_resnet50(state_dict) -> dict`
converts a detectron2 R50 state dict into torchvision `resnet50` naming:
- `stem.conv1.weight` → `conv1.weight`; `stem.conv1.norm.{weight,bias,running_mean,running_var}` →
  `bn1.{...}`.
- `stages.res{2,3,4,5}` → `layer{1,2,3,4}`; within a block `conv{i}.weight` → `conv{i}.weight`,
  `conv{i}.norm.*` → `bn{i}.*` (i∈{1,2,3}); `shortcut.weight` → `downsample.0.weight`,
  `shortcut.norm.*` → `downsample.1.*`.
- Any key not matching a known pattern is dropped with a collected warning (not silently). The
  function returns only remapped keys; it does not load them.

**FR2 — pixel-decoder key alignment.** `remap_pixel_decoder(state_dict, prefix="pixel_decoder.")
-> dict` prepends `prefix` to every key of `M2F_R50_MSDeformAttnPixelDecoder.pkl` (which are already
`input_proj.*`, `transformer.level_embed`, `transformer.encoder.layers.{i}.self_attn.*`,
`transformer.encoder.layers.{i}.{norm1,linear1,linear2,norm2}.*`, and — when present —
`mask_features.*`, `adapter_*`/`layer_*` FPN convs) so they line up with
`Mask2FormerBackbone.pixel_decoder.*`.

**FR3 — combined loader.** `load_pretrained_mask2former(backbone, r50_path, pixel_decoder_path) ->
LoadReport` loads both remapped dicts into `backbone` via `load_state_dict(strict=False)` and returns
a report with `missing_keys`, `unexpected_keys`, and counts. It **raises `RuntimeError`** if:
  - any `feature_extractor.*` parameter (the ResNet50) is left in `missing_keys` after the R50 load
    (i.e. a systematically broken remap), or
  - any `pixel_decoder.transformer.*` or `pixel_decoder.input_proj.*` parameter is missing after the
    decoder load.
  It tolerates (reports but does not raise on) genuinely-absent optional keys: torchvision's `fc.*`
  (dropped by `create_feature_extractor`), and — when the backbone is built without a given branch —
  the pixel decoder's stride-4 FPN/`mask_features` keys.
  BatchNorm caveat: detectron2 `FrozenBatchNorm2d` and torchvision `BatchNorm2d` (in `eval()`,
  frozen) apply the same affine transform of the running stats; the `weight/bias/running_mean/
  running_var` transfer directly. A ≤1e-5 eps discrepancy is accepted (documented, not corrected).

### Precompute script

**FR4 — coverage and ordering.** The script builds `CocoFreeView` on the CocoFreeView data root,
applies the **same filter** as training
(`data.filter_by_idx(FreeViewInMemory(...).data_store['filtered_idx'])`), and enumerates unique
stimulus images in **first-seen order** — byte-for-byte the ordering
`DeduplicatedMemoryDataset.build_index` produces (`path_to_id` assigns ids in iteration order). Unique
image `u` in the cache MUST be the image `DeduplicatedMemoryDataset` assigns `unique_id == u`.

**FR5 — backbone construction for precompute.** The backbone is
`Mask2FormerBackbone(conv_dim=256, n_heads=8, n_points=4, transformer_enc_layers=6,
transformer_dim_feedforward=1024, transformer_dropout=0.0,
transformer_in_features=("res3","res4","res5"), return_stride4=True, mask_dim=256,
freeze_backbone=True, freeze_pixel_decoder=True, imagenet_weights="IMAGENET1K_V2")`, then both
checkpoints are loaded (FR3), then `.eval()`. `torch.inference_mode()` wraps the forward. Images are
preprocessed by `PipelineBuilder.make_transform(resize_size=img_size)` (RGB ImageNet norm), reading
the uint8 unique-image bank (`all_images_{img_size}.pth`) if present, else decoding from disk.

**FR6 — feature products.** For each unique image the script computes
`([res5, res4, res3, res2_fpn], mask_features) = backbone(x)` and stores:
  - `ms_value[u] = torch.cat([m.flatten(2).transpose(1,2) for m in (res5,res4,res3)], dim=1)` →
    `(S, 256)` float32, `S = Σ Hₗ·Wₗ`. This equals `Mask2FormerFeatureAdapter(backbone).forward(x)
    .value[0]` (identical concat), so it is the exact online-path memory.
  - `mask_features[u]` → `(mask_dim=256, H4, W4)` float32 (stride-4; at img_size 256, `64×64`).
  - `res2_fpn` is **discarded** (the 4th deformable level is out of scope).

**FR7 — HDF5 layout.** Written to `data/Coco FreeView/image_features_{img_size}.h5` (overridable),
single group `/features`, mode `"w"`:

| Dataset | Shape | dtype | Notes |
|---|---|---|---|
| `ms_value` | `(U, S, 256)` | float32 | deformable memory, `[res5,res4,res3]` coarse→fine, row-major within a level |
| `mask_features` | `(U, 256, H4, W4)` | float32 | stride-4 final map (heatmap-reserved) |
| `image_path` | `(U,)` | vlen utf8 | unique image path, first-seen order — the keying invariant |

Group attrs: `img_size`, `S`, `spatial_shapes` (flattened `[8,8,16,16,32,32]`),
`level_start_index` (`[0,64,320]`), `embed_dim` (256), `num_levels` (3), `mask_dim` (256),
`mask_feature_shape` (`[H4,W4]`), `normalization` (`"imagenet_rgb"`),
`r50_checkpoint`, `pixel_decoder_checkpoint`, `imagenet_weights`, `transformer_enc_layers` (6),
`num_unique` (U), `created_at`. Chunked per-image (`chunks=(1, S, 256)` / `(1,256,H4,W4)`) so a
dataset reads one image without loading the whole array.

### Feature adapter (the model-side change)

**FR8 — `PrecomputedFeatureAdapter(nn.Module)`** in `src/model/ms_features.py`:
  - `__init__(spatial_shapes, embed_dim=256)` where `spatial_shapes` is a `(L,2)` int64 tensor;
    stores `embed_dim`, `num_levels = L`, and pre-builds `level_start_index` (via
    `build_level_start_index`) and `reference_grids` (via `build_reference_grids`) as **registered
    buffers** so `.to(device)` moves them.
  - `forward(value) -> MultiScaleFeatures`: `value` is `(B, S, embed_dim)` (the cached `ms_value`
    batch); returns `MultiScaleFeatures(value=value, spatial_shapes=..., level_start_index=...,
    reference_grids=...)` with the buffers broadcast/consistent with `value.device`. Adds **zero
    parameters**. Raises `MultiScaleFeatures`'s own `ValueError` if `value`'s `S` disagrees with
    `spatial_shapes`.
  - Contract: for the same image, `PrecomputedFeatureAdapter(shapes).forward(cached_value)` and
    `Mask2FormerFeatureAdapter(backbone).forward(img)` produce bundles whose `value`,
    `spatial_shapes`, `level_start_index`, and `reference_grids` are equal (float32, `atol=0`), so
    `MixerModel.encode` behaves identically.

**FR9 — MixerModel untouched.** No change to `mixer_model.py`. With the stub installed as
`image_encoder` (`image_encoder_type='mask2former'`, `n_image_levels=3`, `embed_dim=256`), the
existing multi-scale `encode`/`decode_fixation` path runs unchanged: `img_input_proj`, `pos_proj`/
`img_pos_proj` PE, `level_embed`, and the deformable eye/fixation decoders are all built and trained
exactly as on the online m2f path. The FR8-guards (no `use_rope`/`argmax`/`heatmap`/
`image_features_concat`) apply identically.

### Dataset and pipeline wiring

**FR10 — `PrecomputedFeatureDataset(Dataset)`** in `src/data/image_feature_cache.py`, interface-
compatible with `DeduplicatedMemoryDataset` for `CoupledDataloader`:
  - `__init__(data: CocoFreeView, cache_path, preload=False)`: rebuilds the first-seen unique index
    from `data` (same logic as `DeduplicatedMemoryDataset.build_index`), opens the HDF5 cache, and
    **verifies** `image_path[u]` in the cache equals the rebuilt unique path for every `u` (raises
    `ValueError` on any mismatch — the order invariant is not bypassable). `preload=True` loads
    `ms_value` fully into a RAM tensor; default lazily reads per `__getitem__` (per-worker file
    handle, opened on first access).
  - `__getitem__(idx) -> (feature, idx, unique_idx)`: `feature = ms_value[unique_idx]` as a
    `(S, 256)` float32 tensor; the tuple shape matches `DeduplicatedMemoryDataset` so
    `CoupledDataloader` sets `batch['image_src'] = feature_batch` and `batch['image_idx'] =
    unique_idx_batch` with no change to `CoupledDataloader`.
  - `__len__ == len(data)`.
  - `mask_features` is **not** returned by default (unused by the model); an optional
    `return_mask_features=False` flag is reserved for the future heatmap path.

**FR11 — config + builder.**
  - New `configs/model/image_encoder/mask2former_precomputed.yaml`: `type: "mask2former"`,
    `enabled: True`, `precomputed: True`, `embed_dim: 256`, `num_levels: 3`,
    `spatial_shapes: [[8,8],[16,16],[32,32]]`, `feature_cache_path: "data/Coco FreeView/image_features_256.h5"`,
    `adapter_hidden_dims: []`. It carries **no** backbone-construction flags (no ResNet50 is built).
  - `PipelineBuilder.build_model`: when `image_encoder.get('precomputed', False)` is truthy on a
    `mask2former` encoder, build `PrecomputedFeatureAdapter(spatial_shapes=..., embed_dim=256)`
    **instead of** `Mask2FormerBackbone`+`Mask2FormerFeatureAdapter`, set `n_image_levels =
    adapter.num_levels`, and pass `image_encoder_type='mask2former'`. No `Mask2FormerBackbone` is
    instantiated (no weight download, no ResNet50).
  - `load_dataset` / `build_dataloader`: when `data.load.use_precomputed_features` is set, construct
    `PrecomputedFeatureDataset(self.data, cache_path)` in place of `DeduplicatedMemoryDataset` and
    feed it to `CoupledDataloader` exactly as today (`Subset` over train/val/test idx).

**FR12 — failure modes.** A missing cache file raises `FileNotFoundError` with the expected path and
a hint to run the precompute script. A cache whose `img_size`/`spatial_shapes`/`embed_dim` attrs
disagree with the requested config raises `ValueError` (no silent shape coercion). Requesting
`precomputed: True` together with `use_precomputed_features` unset (or vice-versa) raises a
`ValueError` at build time (the two must agree — features on both sides or neither).

## Public API Summary

```python
# src/model/m2f_pretrained.py
def remap_detectron2_resnet50(state_dict: dict) -> dict: ...
def remap_pixel_decoder(state_dict: dict, prefix: str = "pixel_decoder.") -> dict: ...
@dataclass
class LoadReport:
    missing_keys: list[str]; unexpected_keys: list[str]
    n_resnet_loaded: int; n_pixdec_loaded: int
def load_pretrained_mask2former(backbone, r50_path: str,
                                pixel_decoder_path: str) -> LoadReport: ...

# src/model/ms_features.py  (additive)
class PrecomputedFeatureAdapter(nn.Module):
    def __init__(self, spatial_shapes: torch.Tensor, embed_dim: int = 256): ...
    def forward(self, value: torch.Tensor) -> MultiScaleFeatures: ...
    # attrs: embed_dim, num_levels ; buffers: level_start_index, reference_grids

# src/data/image_feature_cache.py
class ImageFeatureCache:                      # HDF5 writer/reader, group "/features", mode "w"
    @staticmethod
    def write(path, ms_value, mask_features, image_paths, attrs): ...
    def __init__(self, path): ...             # read side: .ms_value, .mask_features, .image_path, .attrs
class PrecomputedFeatureDataset(Dataset):
    def __init__(self, data: CocoFreeView, cache_path: str,
                 preload: bool = False, return_mask_features: bool = False): ...
    def __getitem__(self, idx) -> tuple: ...  # (feature (S,256) float32, idx, unique_idx)

# scripts/build_image_feature_cache.py       # CLI driver (argparse), no @hydra.main
def main(): ...
```

## Dependencies

| Reads from | Writes to |
|---|---|
| `pretrained_models/M2F_R50.pkl`, `pretrained_models/M2F_R50_MSDeformAttnPixelDecoder.pkl` | `data/Coco FreeView/image_features_{img_size}.h5` (`/features`) |
| `data/Coco FreeView/all_images_{img_size}.pth` (uint8 unique-image bank) or raw images | — |
| `src/model/ms_deform_backbone.py` (`Mask2FormerBackbone`, unchanged) | — |
| `src/model/ms_features.py` (`MultiScaleFeatures`, geometry helpers; adds `PrecomputedFeatureAdapter`) | same file (additive) |
| `src/data/datasets.py` (`DeduplicatedMemoryDataset.build_index` ordering, `CoupledDataloader`) | — |
| `src/data/parsers.py` (`CocoFreeView.filter_by_idx`, `get_img_path`) | — |
| `src/training/pipeline_builder.py` (`build_model`, `load_dataset`, `build_dataloader`) | same file (additive branches) |
| `configs/model/image_encoder/mask2former.yaml` (reference) | `configs/model/image_encoder/mask2former_precomputed.yaml` (new) |
