# F3 — Multi-scale feature contract / backbone adapter

## Goal

Introduce a single, backbone-agnostic image-feature contract — the `MultiScaleFeatures`
bundle — that both image backbones produce and that every downstream consumer (the F1
`DeformableAttention` op, the F4 decoders, and the F6 `MixerModel`) reads. Today `MixerModel`
is welded to DINOv3 specifics: it strips a CLS prefix with `mem[:, 1:, :]`, assumes a single
square `patch_resolution` tuple, and reaches into `image_encoder.model.patch_size` /
`image_encoder.model.rope_embed`. F3 replaces that implicit, single-scale, CLS-prefixed
convention with an explicit, N-level, CLS-free bundle carrying the flattened multi-level memory
plus the geometry needed to sample and position-encode it. F3 ships the bundle and the two
producing adapters (Mask2Former → 3 levels, DINOv3 → 1 level with CLS stripped once at the
boundary); the migration of `MixerModel`/decoders to *consume* it is F4/F6.

## Scope

**In scope**
- A `MultiScaleFeatures` bundle (dataclass) holding `value`, `spatial_shapes`,
  `level_start_index`, and `reference_grids`, with construction-time validation and a
  device/dtype `.to()` helper.
- Two geometry helpers, `build_reference_grids` and `build_level_start_index`, shared by both
  adapters (and reusable by F6).
- `Mask2FormerFeatureAdapter` — wraps the F2 `Mask2FormerBackbone` and packages its
  coarse→fine `[B,256,Hₗ,Wₗ]` maps into the bundle (3 levels; tolerates the 4-level
  `return_stride4` output).
- `DinoV3FeatureAdapter` — wraps the existing `DinoV3Wrapper`, strips the CLS prefix once, and
  emits a 1-level bundle.
- A unit-test suite `tests/test_ms_features.py` (CPU-only, fixed seeds, no network) using a
  stub DINOv3 backbone so the DINOv3 path is testable without `torch.hub`.

**Out of scope (explicitly)**
- **Any edit to `MixerModel`, `DeformableDecoder`, or `DeformableDoubleInputDecoder`.**
  Consuming the bundle is F4 (decoders) and F6 (`MixerModel`/`PipelineBuilder`). F3 does not
  remove `mem[:, 1:, :]`, `patch_resolution`, or `.rope_embed` access from `MixerModel` — it
  only builds the contract that makes that removal possible.
- **Any edit to `Mask2FormerBackbone` or `DinoV3Wrapper`.** Adapters wrap them; the backbones
  stay pure/additive and DINOv3 stays selectable.
- The `img_input_proj` 256/384 → `model_dim` projection (F6). The bundle carries the backbone's
  **native** channel dim `D` (256 for Mask2Former, 384 for DINOv3 ViT-S/16); F3 does not unify it.
- Wiring the stride-4 (res2, 64²) map or `mask_features` as a real decoder level (deferred to
  the heatmap-regression iteration). The Mask2Former adapter *tolerates* a 4-level backbone
  output but **discards** `mask_features`.
- New Hydra config groups (F6) and any change to the training loop, HDF5 caches, or datasets.

## Functional Requirements

**FR1 — Bundle fields and shapes.** `MultiScaleFeatures` is a dataclass with exactly four
tensor fields, `S := Σₗ Hₗ·Wₗ` and `L := n_levels`:
- `value` — `(B, S, D)`, floating dtype. Flattened multi-level memory, levels concatenated in
  the producer's order (coarse→fine for Mask2Former; the single level for DINOv3). Within each
  level, tokens are row-major (`W` fastest), i.e. exactly `map.flatten(2).transpose(1, 2)`.
- `spatial_shapes` — `(L, 2)`, `torch.int64`, row `l` is `(Hₗ, Wₗ)`.
- `level_start_index` — `(L,)`, `torch.int64`, `[0, H₀W₀, H₀W₀+H₁W₁, …]` (last level's start
  excluded from the tail per the F1/Deformable-DETR convention).
- `reference_grids` — `(S, 2)`, floating dtype, the normalized `(x, y)` **center** of every
  memory token in the same order as `value`, each within its own level's `[0,1]` range
  (`linspace(0.5, N-0.5, N)/N`). No batch and no level axis; consumers broadcast.

**FR2 — Bundle construction validation.** `__post_init__` raises `ValueError` when any invariant
is violated: `value.dim() != 3`; `spatial_shapes.shape != (L, 2)`; `reference_grids.shape != (S, 2)`;
`Σₗ Hₗ·Wₗ != value.shape[1]`; `level_start_index` not equal to the value derived from
`spatial_shapes`; `spatial_shapes.dtype`/`level_start_index.dtype` not integer. It does **not**
coerce dtypes silently. Validation is O(L) (no per-token work).

**FR3 — Bundle properties.** Read-only properties: `num_levels -> int` (= `spatial_shapes.shape[0]`),
`embed_dim -> int` (= `value.shape[-1]`), `batch_size -> int` (= `value.shape[0]`),
`seq_len -> int` (= `value.shape[1]`), and `level_sizes() -> list[int]` (= `(Hₗ·Wₗ)` per level).

**FR4 — Bundle `.to()`.** `to(device=None, dtype=None) -> MultiScaleFeatures` returns a **new**
bundle. `device` moves all four tensors. `dtype` casts **only** `value` and `reference_grids`;
`spatial_shapes` and `level_start_index` remain `int64` regardless (casting index tensors to a
float dtype would corrupt them). Passing neither returns a bundle whose tensors are unchanged
references.

**FR5 — `build_reference_grids`.**
`build_reference_grids(spatial_shapes, device='cpu', dtype=torch.float32) -> Tensor (S, 2)`
produces per-token normalized centers matching FR1. For a level `(H, W)`, token index
`h*W + w` maps to `(x=(w+0.5)/W, y=(h+0.5)/H)`. Levels are concatenated in `spatial_shapes` row
order. This is byte-identical (up to the level-broadcast repeat and the leading batch axis) to
`MSDeformAttnTransformerEncoder.get_reference_points` in `src/model/ms_deform_backbone.py`, so
the bundle's geometry equals the geometry the F2 pixel decoder used internally.

**FR6 — `build_level_start_index`.**
`build_level_start_index(spatial_shapes) -> Tensor (L,) int64` returns
`cat([[0], cumprod_sizes.cumsum(0)[:-1]])`, on `spatial_shapes.device`, matching the F1
`DeformableAttention` derivation exactly.

**FR7 — `Mask2FormerFeatureAdapter`.** Constructed with an already-built `Mask2FormerBackbone`,
held as a submodule. Exposes `embed_dim` (= `backbone.embed_dim`, 256) and `num_levels`
(= `backbone.num_levels`). `forward(x: (B,3,H,W)) -> MultiScaleFeatures`:
1. Call `backbone(x)`. If `backbone.return_stride4`, the backbone returns `(maps, mask_features)`;
   the adapter takes `maps` and **discards `mask_features`**. Otherwise it returns `maps`
   directly.
2. `maps` is a list of `(B, 256, Hₗ, Wₗ)` in coarse→fine order. Flatten each via
   `.flatten(2).transpose(1, 2)` → `(B, HₗWₗ, 256)`, concat over dim 1 → `value`.
3. Build `spatial_shapes` from the maps' `(H, W)`, `level_start_index` via FR6,
   `reference_grids` via FR5 (on `value.device`, `value.dtype`), and return the bundle.
The resulting `spatial_shapes`/`level_start_index` are identical to the tensors the backbone's
pixel decoder produced internally, and `value` equals the concatenated flattened maps
element-for-element (round-trip identity — reshaping the transformer memory to maps and back is
a no-op).

**FR8 — `DinoV3FeatureAdapter`.** Constructed with an already-built `DinoV3Wrapper` (held as a
submodule) and an optional `num_prefix_tokens: int = 1`. Exposes `embed_dim`
(= `backbone.embed_dim`, 384) and `num_levels = 1`. Reads `backbone.model.patch_size` **once at
init** (the only place DINOv3-internal attributes are touched) into `self.patch_size`.
`forward(x: (B,3,H,W)) -> MultiScaleFeatures`:
1. `tokens = backbone(x)` → `(B, num_prefix_tokens + H'·W', D)` (CLS-prefixed).
2. `value = tokens[:, num_prefix_tokens:, :]` — strip the prefix **once here** (the sole
   `mem[:, 1:, :]` in the codebase after F4/F6).
3. `H' = x.shape[-2] // patch_size`, `W' = x.shape[-1] // patch_size`. Raise `ValueError` if
   `H'·W' != value.shape[1]` (guards a wrong `patch_size` or an unexpected prefix count — a
   "phantom CLS" check).
4. `spatial_shapes = [[H', W']]`, `level_start_index = [0]`, `reference_grids` via FR5; return
   the 1-level bundle.

**FR9 — F1 consumability.** A bundle produced by either adapter is directly consumable by
`DeformableAttention(embed_dim=D, num_heads, num_points, n_levels=bundle.num_levels)`:
`attn(query, reference_points, bundle.value, bundle.spatial_shapes, bundle.level_start_index)`
runs with no shape error and returns `(B, Nq, D)`. (F3 asserts consumability with a matching-D
attention module; the real decoders swap to `model_dim` in F4/F6.)

**FR10 — Module semantics.** Both adapters are `nn.Module`s owning their backbone as a
submodule, so `.to()`, `.train()/.eval()`, `.state_dict()`, `.parameters()`, and F1's
`InferenceRecorder` hooks propagate to the backbone unchanged (adapter parameter keys are
prefixed `backbone.…`). The adapters add **no parameters of their own** (`spatial_shapes`,
`level_start_index`, and `reference_grids` are computed per forward, not registered buffers, so
dynamic input sizes are supported). Freezing/eval behavior is entirely the wrapped backbone's.

**FR11 — Additivity.** F3 creates one new module file and one new test file and modifies **no**
existing file. Existing single-scale DINOv3 and F2 backbone code paths are untouched.

## Public API Summary

```python
# src/model/ms_features.py
from dataclasses import dataclass
import torch
import torch.nn as nn

def build_reference_grids(spatial_shapes: torch.Tensor,
                          device="cpu", dtype=torch.float32) -> torch.Tensor: ...   # (S, 2)

def build_level_start_index(spatial_shapes: torch.Tensor) -> torch.Tensor: ...      # (L,) int64

@dataclass
class MultiScaleFeatures:
    value: torch.Tensor             # (B, S, D)  float
    spatial_shapes: torch.Tensor    # (L, 2)     int64  rows (Hₗ, Wₗ)
    level_start_index: torch.Tensor # (L,)       int64
    reference_grids: torch.Tensor   # (S, 2)     float  per-token (x, y) centers in [0,1]

    def __post_init__(self): ...                       # FR2 validation
    @property
    def num_levels(self) -> int: ...
    @property
    def embed_dim(self) -> int: ...
    @property
    def batch_size(self) -> int: ...
    @property
    def seq_len(self) -> int: ...
    def level_sizes(self) -> list[int]: ...
    def to(self, device=None, dtype=None) -> "MultiScaleFeatures": ...   # FR4

class Mask2FormerFeatureAdapter(nn.Module):
    def __init__(self, backbone): ...          # Mask2FormerBackbone
    embed_dim: int
    num_levels: int
    def forward(self, x: torch.Tensor) -> MultiScaleFeatures: ...

class DinoV3FeatureAdapter(nn.Module):
    def __init__(self, backbone, num_prefix_tokens: int = 1): ...   # DinoV3Wrapper
    embed_dim: int
    num_levels: int   # == 1
    def forward(self, x: torch.Tensor) -> MultiScaleFeatures: ...
```

## Dependencies

| Direction | Component | Interaction |
|---|---|---|
| Reads | `Mask2FormerBackbone.forward` (`src/model/ms_deform_backbone.py`) | 3 coarse→fine maps `[B,256,Hₗ,Wₗ]` (or `(maps, mask_features)` when `return_stride4`) |
| Reads | `DinoV3Wrapper.forward` (`src/model/dino_wrapper.py`) | `(B, 1+H'W', 384)` CLS-prefixed tokens; `.model.patch_size` (init-time) |
| Reuses | `MSDeformAttnTransformerEncoder.get_reference_points` semantics | `build_reference_grids` reproduces the same grid geometry |
| Produced-for (later) | F1 `DeformableAttention` (`src/model/blocks.py`) | bundle feeds `value`/`spatial_shapes`/`level_start_index` |
| Produced-for (later) | F4 `DeformableDecoder`, `DeformableDoubleInputDecoder` | bundle replaces the `(H,W)` tuple + `mem[:,1:,:]` slice |
| Produced-for (later) | F6 `MixerModel.encode`/`decode_fixation` | one bundle interface for both backbones; `reference_grids` for per-level positional encoding |
| Writes | *(none)* | no HDF5, no config, no dataset changes |
