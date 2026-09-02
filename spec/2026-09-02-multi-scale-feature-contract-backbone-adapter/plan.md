# F3 — Implementation Plan

## Context and Design Decisions

**Why a bundle at all.** The current `MixerModel` hard-codes the DINOv3 single-scale, CLS-prefixed
convention in three places: `mem[:, 1:, :]` slices inside the deformable decoders
(`blocks.py:973/983/1114`), a single square `self.patch_resolution` tuple derived from
`image_encoder.model.patch_size` (`mixer_model.py:133-134`), and direct `image_encoder.model.rope_embed`
access (`mixer_model.py:229-239`). The F2 backbone emits **three** CLS-free maps at different
resolutions, so none of those assumptions survive. Rather than sprinkle backbone-type branches
through `encode`/`decode_fixation` and both decoders, we define one contract — `MultiScaleFeatures`
— that carries the flattened multi-level memory plus the geometry (`spatial_shapes`,
`level_start_index`, `reference_grids`) that the F1 op and any consumer need. Every consumer then
speaks "bundle", and each backbone's quirks (CLS prefix, patch grid) are handled exactly once, in
its adapter.

**What F3 is and is not.** Per the locked scope (planning session), F3 is *only* the contract and
its two producers. It touches neither `MixerModel` nor the decoders — those consume the bundle in
F6 and F4 respectively (roadmap dependency `{F2, F3, F4} → F6`). Keeping F3 additive means the
existing single-scale path keeps running untouched while F4/F6 are built, and old checkpoints stay
loadable (the byte-identity guarantees F1 established are not disturbed).

**Adapters wrap, backbones stay pure.** The F2 backbone "modifies no existing file" and DINOv3
stays selectable; we preserve that by adding *adapter* `nn.Module`s that own a backbone as a
submodule and repackage its output, rather than adding a bundle mode to the backbones themselves.
This also confines DINOv3-internal attribute access (`.model.patch_size`) to `DinoV3FeatureAdapter`
— realizing the roadmap's "kills the `image_encoder.model.patch_size` access" at the boundary.

**Native channel dim, no projection.** The bundle carries `D` = the backbone's own channel dim
(256 for Mask2Former, 384 for DINOv3). Unifying to `model_dim` via `img_input_proj` is F6's job
(TechStack: "256 (`conv_dim`) → `img_input_proj` → 512"); doing it here would prejudge F6 and
duplicate a projection the bundle should be agnostic to.

**`reference_grids` are per-token centers, not query reference points.** The deformable *query*
reference points come from gaze/fixation coordinates (F4/F6), not from grid centers.
`reference_grids` is the geometry of the *memory* — the normalized center of each token — which F6
uses to position-encode the multi-level memory (replacing the DINOv3-only
`pos_proj.forward_features()` patch grid) and which any consumer can broadcast to the
`(B, Nq, L, 2)` form if it wants grid-anchored refs. We store the compact `(S, 2)` form (no batch,
no level axis) to match how `MSDeformAttnTransformerEncoder.get_reference_points` builds its base
grid before the level-repeat, keeping F3 geometry byte-consistent with F2's internal geometry.

**Order and flattening must be exact.** F1 `DeformableAttention` validates that `Σ Hₗ·Wₗ ==
value.shape[1]` and that `level_start_index` is consistent, and it splits `value` per level using
`spatial_shapes`. So the bundle's `value` token order, `spatial_shapes` row order, and
`reference_grids` order must all agree. We fix the convention: **levels in the producer's list
order** (coarse→fine for Mask2Former), **tokens row-major (`W` fastest)** within a level — exactly
`map.flatten(2).transpose(1, 2)`, which is what F2's pixel decoder already uses internally, so the
adapter's re-flatten of the output maps is an identity round-trip.

## Implementation Steps

### Step 1 — Geometry helpers (`src/model/ms_features.py`, new file)

Create the module and the two shared helpers first (no dependencies on the rest).

```python
import torch

def build_level_start_index(spatial_shapes: torch.Tensor) -> torch.Tensor:
    # spatial_shapes: (L, 2) int. Returns (L,) int64: [0, H0W0, H0W0+H1W1, ...]
    sizes = spatial_shapes[:, 0] * spatial_shapes[:, 1]           # (L,)
    return torch.cat([spatial_shapes.new_zeros((1,)), sizes.cumsum(0)[:-1]]).to(torch.int64)

def build_reference_grids(spatial_shapes, device="cpu", dtype=torch.float32) -> torch.Tensor:
    grids = []
    for (H, W) in spatial_shapes.tolist():
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=device, dtype=dtype) / H,
            torch.linspace(0.5, W - 0.5, W, device=device, dtype=dtype) / W,
            indexing="ij")                                        # each (H, W)
        grids.append(torch.stack((ref_x.reshape(-1), ref_y.reshape(-1)), dim=-1))  # (H*W, 2)
    return torch.cat(grids, dim=0)                                # (S, 2)
```

Note the `(ref_x, ref_y)` stack order — x first — matching F1's `reference_points` `(…, 2)`
convention `(x, y)`. `reshape(-1)` on the `(H, W)` grid is row-major (`W` fastest), matching
`flatten(2)`.

### Step 2 — `MultiScaleFeatures` dataclass (same file)

```python
from dataclasses import dataclass

@dataclass
class MultiScaleFeatures:
    value: torch.Tensor
    spatial_shapes: torch.Tensor
    level_start_index: torch.Tensor
    reference_grids: torch.Tensor

    def __post_init__(self):
        if self.value.dim() != 3:
            raise ValueError(f"value must be (B, S, D); got {tuple(self.value.shape)}")
        L = self.spatial_shapes.shape[0]
        if self.spatial_shapes.shape != (L, 2):
            raise ValueError(...)
        if not torch.is_floating_point(self.value):
            raise ValueError("value must be a floating tensor")
        for name, t in (("spatial_shapes", self.spatial_shapes),
                        ("level_start_index", self.level_start_index)):
            if t.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must be an integer tensor; got {t.dtype}")
        S = int((self.spatial_shapes[:, 0] * self.spatial_shapes[:, 1]).sum())
        if self.value.shape[1] != S:
            raise ValueError(f"Σ Hₗ·Wₗ ({S}) != value length ({self.value.shape[1]})")
        if tuple(self.reference_grids.shape) != (S, 2):
            raise ValueError(...)
        expected = build_level_start_index(self.spatial_shapes)
        if self.level_start_index.tolist() != expected.tolist():
            raise ValueError(f"level_start_index {self.level_start_index.tolist()} "
                             f"inconsistent with spatial_shapes (expected {expected.tolist()})")

    @property
    def num_levels(self): return self.spatial_shapes.shape[0]
    @property
    def embed_dim(self): return self.value.shape[-1]
    @property
    def batch_size(self): return self.value.shape[0]
    @property
    def seq_len(self): return self.value.shape[1]
    def level_sizes(self): return (self.spatial_shapes[:, 0] * self.spatial_shapes[:, 1]).tolist()

    def to(self, device=None, dtype=None) -> "MultiScaleFeatures":
        # dtype casts ONLY the float tensors; index tensors stay int64.
        return MultiScaleFeatures(
            value=self.value.to(device=device, dtype=dtype),
            spatial_shapes=self.spatial_shapes.to(device=device),
            level_start_index=self.level_start_index.to(device=device),
            reference_grids=self.reference_grids.to(device=device, dtype=dtype),
        )
```

`.to(device=..., dtype=None)` is well-defined: `Tensor.to(device=d, dtype=None)` moves device only.

### Step 3 — `Mask2FormerFeatureAdapter` (same file)

```python
import torch.nn as nn

class Mask2FormerFeatureAdapter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim          # 256
        self.num_levels = backbone.num_levels        # 3 (or 4 if return_stride4)

    def forward(self, x) -> MultiScaleFeatures:
        out = self.backbone(x)
        maps = out[0] if self.backbone.return_stride4 else out   # discard mask_features
        flats, shapes = [], []
        for m in maps:                               # coarse→fine [B, 256, Hₗ, Wₗ]
            B, C, H, W = m.shape
            flats.append(m.flatten(2).transpose(1, 2))           # (B, H*W, C)
            shapes.append((H, W))
        value = torch.cat(flats, dim=1)              # (B, S, 256)
        spatial_shapes = torch.as_tensor(shapes, dtype=torch.int64, device=value.device)
        return MultiScaleFeatures(
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=build_level_start_index(spatial_shapes),
            reference_grids=build_reference_grids(spatial_shapes, value.device, value.dtype),
        )
```

`return_stride4` support is defensive (the flag is off by default and wiring the 4th level is
deferred); `mask_features` is intentionally dropped.

### Step 4 — `DinoV3FeatureAdapter` (same file)

```python
class DinoV3FeatureAdapter(nn.Module):
    def __init__(self, backbone, num_prefix_tokens: int = 1):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim          # 384
        self.num_levels = 1
        self.num_prefix_tokens = num_prefix_tokens
        self.patch_size = backbone.model.patch_size  # sole DINOv3-internal access, init-time

    def forward(self, x) -> MultiScaleFeatures:
        tokens = self.backbone(x)                    # (B, prefix + H'W', D), CLS-prefixed
        value = tokens[:, self.num_prefix_tokens:, :]           # strip CLS ONCE
        Hs, Ws = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        if Hs * Ws != value.shape[1]:
            raise ValueError(f"patch grid {Hs}x{Ws}={Hs*Ws} != token count {value.shape[1]} "
                             f"(check patch_size / num_prefix_tokens)")
        spatial_shapes = torch.as_tensor([[Hs, Ws]], dtype=torch.int64, device=value.device)
        return MultiScaleFeatures(
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=build_level_start_index(spatial_shapes),   # [0]
            reference_grids=build_reference_grids(spatial_shapes, value.device, value.dtype),
        )
```

### Step 5 — Tests (`tests/test_ms_features.py`, new file)

- A `FakeDino(nn.Module)` stub exposing `.embed_dim` and `.model.patch_size` and a `forward`
  returning `(B, 1 + (H//p)*(W//p), D)` random tokens — lets the DINOv3 adapter be tested with no
  `torch.hub`.
- Mask2Former adapter tests build `Mask2FormerBackbone(imagenet_weights=None)` (random init, no
  network), mirroring `tests/test_ms_deform_backbone.py`.
- Cover the six groups in `validation.md` (bundle validation, geometry helpers, both adapters, F1
  consumability, module integrity). CPU-only, `torch.manual_seed` fixed.

## Implementation Order

1. **Step 1** — `build_level_start_index`, `build_reference_grids` (foundation, no deps).
2. **Step 2** — `MultiScaleFeatures` dataclass + validation + `.to()` (depends on Step 1 for the
   `level_start_index` consistency check).
3. **Step 3** — `Mask2FormerFeatureAdapter` (depends on Steps 1–2 and the F2 backbone).
4. **Step 4** — `DinoV3FeatureAdapter` (depends on Steps 1–2 and `DinoV3Wrapper`).
5. **Step 5** — `tests/test_ms_features.py` (depends on all of the above).
