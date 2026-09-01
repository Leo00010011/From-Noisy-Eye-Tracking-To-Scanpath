# Requirements — F1: Multi-scale Deformable Attention Primitive

## Goal

Generalize the single-scale `DeformableAttention` module (`src/model/blocks.py:680`) so it can
sample over *N* feature levels instead of one, while remaining **byte-identical** at `n_levels=1`.
This is the linchpin of the Mask2Former backbone migration: the extended parameter layout must
match Mask2Former's `MSDeformAttn` (`sampling_offsets` = `n_heads·n_levels·n_points·2`,
`attention_weights` = `n_heads·n_levels·n_points`) so that F2 can load the pretrained
pixel-decoder weights by name and shape, and the `n_levels=1` collapse must reproduce today's
shapes and numerics exactly so that existing single-scale checkpoints (e.g. HP-search runs) load
with zero missing/unexpected keys and produce identical activations. The op stays pure-PyTorch
(`grid_sample`, no CUDA custom op), and keeps the star-pattern init, `geometric_sigma` jitter,
KV-cache, and `InferenceRecorder` hooks.

## Scope

**In scope**
- Extend `DeformableAttention` in place (class name preserved) to accept an `n_levels` constructor
  argument (default `1`) that resizes `sampling_offsets` and `attention_weights` to the
  multi-scale param layout.
- A multi-level `forward` that: (a) splits the flattened value map per level, (b) grid-samples each
  level, (c) softmaxes attention weights jointly over `n_levels·n_points` and reduces over both
  levels and points.
- Backward-compatible `forward` signature: the existing keyword `spatial_shape` accepts either the
  legacy `(H, W)` tuple (1 level) **or** a `(n_levels, 2)` `LongTensor`; `reference_points` accepts
  either `(B, Nq, 2)` (broadcast across levels) or `(B, Nq, n_levels, 2)`; a new optional
  `level_start_index` argument is accepted for parity with Mask2Former and derived via `cumsum`
  when absent.
- Preserve KV-cache (`enable_memory_kv_cache` / `clear_kv_cache` / `disable_*`), recorder hooks
  (`sampling_offsets`, `attention_weights`, `sampling_locations`, `reference_points`),
  `geometric_sigma` jitter, and `normalize_grid_init` behavior across all levels.
- Star-pattern (`_reset_parameters`) generalized to `(n_heads, n_levels, n_points, 2)`, collapsing
  to today's exact bias vector at `n_levels=1`.

**Out of scope (belongs to later features)**
- Editing `DeformableDecoder` / `DeformableDoubleInputDecoder` to *drive* multiple levels (F4). F1
  leaves their call sites untouched; they keep passing a `(H, W)` tuple and `(B, Nq, 2)` ref points.
- The `MultiScaleFeatures` bundle, `spatial_shapes`/`level_start_index` plumbing from a backbone,
  and `reference_grids` construction (F3).
- The vendored ResNet50 + pixel decoder and the COCO weight loader (F2).
- Any config-group or `MixerModel`/`PipelineBuilder` wiring (F6).
- A CUDA custom op or a copy of Mask2Former's `ms_deform_attn_core_pytorch` — F1 is self-contained
  pure PyTorch and is **not** validated against the Mask2Former reference kernel (deferred to F2).

## Functional Requirements

**FR1 — Constructor gains `n_levels`.**
`DeformableAttention.__init__` accepts `n_levels: int = 1` (inserted without disturbing existing
positional/keyword defaults). `sampling_offsets` becomes
`nn.Linear(embed_dim, n_heads·n_levels·n_points·2)` and `attention_weights` becomes
`nn.Linear(embed_dim, n_heads·n_levels·n_points)`. `value_proj` and `output_proj` are unchanged
(`embed_dim → embed_dim`). At `n_levels=1` these shapes equal today's exactly.

**FR2 — Init byte-identity at `n_levels=1`.**
`_reset_parameters` builds the star-pattern bias as `(n_heads, n_levels, n_points, 2)`:
per-head unit vector (normalized when `normalize_grid_init=True`), repeated across levels and
points, scaled by `point_index + 1`, flattened to the `sampling_offsets.bias` vector. For any fixed
seed, a `DeformableAttention(..., n_levels=1)` produces a `state_dict` whose every tensor is
element-wise equal to today's `DeformableAttention` with the same args. `value_proj`/`output_proj`
init is left as today (weights xavier-uniform, biases at PyTorch default — **not** zeroed), so
fresh-init identity holds; Mask2Former checkpoint biases override on load regardless.

**FR3 — `forward` accepts legacy and multi-scale inputs.**
Signature: `forward(query, reference_points, value, spatial_shape, level_start_index=None)`.
- `query`: `(B, Nq, embed_dim)` float.
- `value`: `(B, ΣHₗWₗ, embed_dim)` float (single map `(B, H·W, embed_dim)` when 1 level).
- `spatial_shape`: `(H, W)` tuple/list (⇒ 1 level) **or** `(n_levels, 2)` int tensor with rows
  `(Hₗ, Wₗ)`. When a tensor is given, `n_levels` must equal `self.n_levels` and
  `Σ Hₗ·Wₗ == value.shape[1]`, else `ValueError`.
- `reference_points`: `(B, Nq, 2)` (broadcast to every level) **or** `(B, Nq, n_levels, 2)`,
  normalized `[0,1]`. Last dim must be `2` (box refs, last-dim `4`, are out of scope ⇒ `ValueError`).
- `level_start_index`: optional `(n_levels,)` int tensor `[0, H₀W₀, H₀W₀+H₁W₁, …]`; when `None`,
  derived from `spatial_shape` via `cumsum`. When provided it must be consistent with
  `spatial_shape` (used only to split `value`; a mismatch that would over/under-run raises).
- Returns `(B, Nq, embed_dim)` float.

**FR4 — Multi-level sampling numerics.**
Per level `l`: reshape that level's value slice to `(B·n_heads, head_dim, Hₗ, Wₗ)`; compute
`sampling_locations[...,l,...] = reference_points[...,l,:] + sampling_offsets[...,l,:,:] /
[Wₗ, Hₗ]`; map `[0,1]→[-1,1]`; `grid_sample(mode='bilinear', padding_mode='zeros',
align_corners=False)`. Attention weights are softmaxed **jointly** over the flattened
`n_levels·n_points` axis (matching Mask2Former), then the weighted sum runs over both levels and
points. At `n_levels=1` this reduces exactly to today's points-only softmax and single-map sample.

**FR5 — `geometric_sigma` jitter.**
When `self.training and geometric_sigma > 0`, add `randn_like(sampling_offsets)·geometric_sigma`
to the full `(B, Nq, n_heads, n_levels, n_points, 2)` offset tensor before normalization —
identical semantics to today, now spanning all levels.

**FR6 — KV-cache over the flattened value.**
`enable_memory_kv_cache()` caches the projected value **once**, keyed by the whole flattened
memory. The cached artifact is the per-level list of reshaped maps (or the projected
`(B, ΣHₗWₗ, n_heads, head_dim)` tensor re-split each call); `clear_kv_cache()` / `disable_*` reset
it. Cache correctness: two forwards with cache enabled and identical `value` produce identical
output to cache disabled. Splitting from cache must use the same `spatial_shape` as the cached
build (documented precondition: memory geometry is fixed while the cache is warm).

**FR7 — Recorder hooks unchanged in name, generalized in shape.**
When `_module_recording_enabled(self)`, record `sampling_offsets`
`(B, Nq, n_heads, n_levels, n_points, 2)`, `attention_weights`
`(B, Nq, n_heads, n_levels, n_points)` (post-softmax), `sampling_locations` (same shape as
offsets), and `reference_points` (as broadcast, `(B, Nq, n_levels, 2)`). Keys are unchanged so
existing recorder consumers keep working; at `n_levels=1` the extra singleton axis is present but
squeezable.

**FR8 — Retro-compatibility of existing decoders.**
`DeformableDecoder` and `DeformableDoubleInputDecoder` are **not modified** by F1. Their existing
calls — `cross_attn(query=src, reference_points=<(B,Nq,2)>, value=mem, spatial_shape=(16,16))` —
continue to run unchanged and produce identical outputs to the pre-F1 code. (They still slice the
CLS token via `mem[:,1:,:]`; that is F3's concern, not F1's.)

**FR9 — Error conditions.**
- `embed_dim % num_heads != 0` ⇒ `ValueError` (unchanged).
- `spatial_shape` tensor whose row count ≠ `self.n_levels` ⇒ `ValueError`.
- `Σ Hₗ·Wₗ != value.shape[1]` ⇒ `ValueError`.
- `reference_points` last dim ∉ {2} or `n_levels` axis ≠ `self.n_levels` ⇒ `ValueError`.

## Public API Summary

```python
class DeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        n_levels: int = 1,                 # NEW — 1 ⇒ byte-identical to today
        attn_dropout: float = 0.0,
        geometric_sigma: float = 0.0,
        normalize_grid_init: bool = True,
        device="cpu",
        dtype=torch.float32,
    ): ...

    def forward(
        self,
        query,                             # (B, Nq, embed_dim)
        reference_points,                  # (B, Nq, 2) or (B, Nq, n_levels, 2), in [0,1]
        value,                             # (B, ΣHₗWₗ, embed_dim)
        spatial_shape,                     # (H, W) tuple  OR  (n_levels, 2) LongTensor
        level_start_index=None,            # (n_levels,) LongTensor; derived if None
    ):                                     # -> (B, Nq, embed_dim)
        ...

    # unchanged public surface
    def enable_memory_kv_cache(self): ...
    def disable_memory_kv_cache(self): ...
    def clear_kv_cache(self): ...
    def disable_kv_cache(self): ...
```

## Dependencies

| Direction | Item | Notes |
|---|---|---|
| Reads | `query`, `reference_points`, `value`, `spatial_shape`, `level_start_index` (forward args) | Geometry supplied by caller; F1 does not build reference grids |
| Reads | `_module_recording_enabled`, `record_module_value` (`src/model/blocks.py`) | Existing recorder helpers, reused unchanged |
| Reads | `torch.nn.functional.grid_sample`, `F.softmax` | Pure-PyTorch sampling; no CUDA op |
| Writes | `DeformableAttention.state_dict` param shapes | New layout consumed by F2's weight loader |
| Consumed by (downstream) | F2 (`MSDeformAttnPixelDecoder` internal attention at `n_levels=3`) | Must load pretrained `sampling_offsets`/`attention_weights`/`value_proj`/`output_proj` by shape |
| Consumed by (downstream) | F4 (`DeformableDecoder`, `DeformableDoubleInputDecoder`) | Opt into tensor `spatial_shape` + per-level ref points later |
| Unchanged (guaranteed) | Existing `DeformableDecoder` / `DeformableDoubleInputDecoder` call sites | Legacy tuple + 2-D ref-point path stays byte-identical |
