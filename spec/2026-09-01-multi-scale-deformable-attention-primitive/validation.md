# Validation — F1: Multi-scale Deformable Attention Primitive

Tests live in `tests/test_ms_deformable_attention.py` (pytest, CPU, fixed seeds, small dummy
tensors). Reference dims used below unless stated: `embed_dim=32`, `num_heads=4`, `num_points=4`,
`B=2`, `Nq=5`. Tolerances are for `float32`.

## Code Correctness

### Group 1 — `n_levels=1` byte-identity (the retro-compat contract)

- [ ] **Param shapes match today.** A `DeformableAttention(embed_dim=32, num_heads=4, num_points=4,
  n_levels=1)` has `sampling_offsets.weight.shape == (4·1·4·2, 32) == (32, 32)`,
  `sampling_offsets.bias.shape == (32,)`, `attention_weights.weight.shape == (16, 32)`,
  `attention_weights.bias.shape == (16,)`. Fail ⇒ layout regression.
- [ ] **`state_dict` byte-identity vs. a pinned reference.** Construct two modules under the same
  `torch.manual_seed(0)` — one from the F1 class at `n_levels=1`, one from a saved snapshot of the
  pre-F1 class (or an inline re-implementation of the old `_reset_parameters`). Every tensor in the
  two `state_dict`s is equal: `torch.equal(a[k], b[k])` for all keys `k`, and key sets are equal.
  Fail ⇒ init diverged.
- [ ] **Forward output identity at 1 level.** With identical weights and the same
  `query`/`reference_points (B,Nq,2)`/`value (B,H·W,32)`/`spatial_shape=(H,W)` inputs
  (`H=W=8`), the F1 forward output equals the pre-F1 forward output to `atol=0, rtol=0`
  (`torch.equal`) in eval mode. Fail ⇒ numeric regression on the single-scale path.
- [ ] **Old-checkpoint load with zero missing/unexpected keys.** `load_state_dict(old_sd,
  strict=True)` on an F1 `n_levels=1` module succeeds (empty missing/unexpected). Fail ⇒ HP-search
  checkpoints would break.
- [ ] **Softmax reduces to points-only at 1 level.** Recorded `attention_weights` (shape
  `(B,Nq,4,1,4)`) sums to `1.0` along the last axis per `(b,q,head)` (`atol=1e-6`); squeezing the
  level axis reproduces today's `(B,Nq,4,4)` normalization.

### Group 2 — Multi-level forward correctness

- [ ] **Shapes at `n_levels=3`.** Build `n_levels=3`; feed `spatial_shape=tensor([[8,8],[4,4],[2,2]])`,
  `value` of length `64+16+4=84`, `reference_points (B,Nq,3,2)`. Output shape is `(B,Nq,32)`; no
  exception. Fail ⇒ multi-level plumbing broken.
- [ ] **Value split alignment.** Monkeypatch / inspect `value_levels`: level lengths are
  `[64,16,4]` and each reshaped map has shape `(B·4, 8, Hₗ, Wₗ)` with `head_dim=8`. Fail ⇒
  `level_start_index`/split off-by-one.
- [ ] **Joint softmax over levels·points.** Recorded post-softmax `attention_weights`
  `(B,Nq,4,3,4)` sums to `1.0` over the combined `(level,point)` axes per `(b,q,head)`
  (`atol=1e-6`). Fail ⇒ softmax applied per-level instead of jointly.
- [ ] **Single-level equivalence embedded in multi-level.** Construct an `n_levels=3` module, then
  force it to behave like 1 level: set all three levels' value slices to the **same** map and all
  three ref-point levels equal, and zero `sampling_offsets` (bias too). Output equals a bilinear
  read of that shared map at the reference points (all points collapse to the ref location), i.e.
  each query returns `output_proj(value_proj(interp(map, ref)))` — matches an independent manual
  `grid_sample` to `atol=1e-5`. Fail ⇒ level reduction incorrect.
- [ ] **`level_start_index` optional.** Calling with `level_start_index=None` and with an explicitly
  passed correct `[0,64,80]` tensor produces identical output (`atol=0`). Fail ⇒ derivation bug.
- [ ] **Gradients flow to all levels.** After `output.sum().backward()`, `sampling_offsets.weight.grad`,
  `attention_weights.weight.grad`, `value_proj.weight.grad`, `output_proj.weight.grad` are all
  non-`None` and finite. Fail ⇒ a level detached from the graph.

### Group 3 — Input polymorphism and error conditions

- [ ] **Tuple `spatial_shape` ⇒ 1 level.** `n_levels=1` module accepts `spatial_shape=(8,8)`
  (tuple) and `spatial_shape=torch.tensor([[8,8]])` interchangeably with identical output
  (`atol=0`).
- [ ] **2-D ref points broadcast.** `n_levels=3` module given `reference_points (B,Nq,2)` yields the
  same output as the same points manually expanded to `(B,Nq,3,2)` (`atol=0`). Fail ⇒ broadcast
  path diverges.
- [ ] **Level-count mismatch raises.** `n_levels=3` module with a `(2,2)`-row `spatial_shape` tensor
  raises `ValueError`. Fail ⇒ silent wrong-shape read.
- [ ] **Value-length mismatch raises.** `Σ Hₗ·Wₗ != value.shape[1]` raises `ValueError`.
- [ ] **Bad ref-point last dim raises.** `reference_points` with last dim `4` raises `ValueError`
  (box refs out of scope).
- [ ] **`embed_dim % num_heads != 0` raises** at construction (unchanged behavior).

### Group 4 — Decoder integration (F1 leaves decoders untouched)

- [ ] **`DeformableDecoder` forward unchanged.** A `DeformableDecoder(spatial_shape=(8,8))` runs its
  existing `forward(src, mem, reference_points=(B,Nq,2))` without modification and returns shape
  `(B, Nq, model_dim)`; output equals the pre-F1 result to `torch.equal` under fixed seed. Fail ⇒
  F1 broke the legacy call path.
- [ ] **`DeformableDoubleInputDecoder` forward unchanged.** Same for
  `DeformableDoubleInputDecoder(spatial_shape=(8,8))` `forward(src, mem1, mem2, reference_points)`
  — shape `(B, Nq, model_dim)`, byte-identical to pre-F1.
- [ ] **KV-cache parity.** With `enable_memory_kv_cache()`, two successive forwards on identical
  `value` produce identical output to the cache-disabled path (`atol=0`); `clear_kv_cache()` then a
  forward with a *different* `value` reflects the new memory. Fail ⇒ stale cache.
- [ ] **Recorder keys present and shaped.** With `InferenceRecorder` enabled (or
  `_module_recording_enabled` stubbed true), a forward records `sampling_offsets`
  `(B,Nq,4,L,4,2)`, `attention_weights` `(B,Nq,4,L,4)`, `sampling_locations` `(B,Nq,4,L,4,2)`,
  `reference_points` `(B,Nq,L,2)` for both `L=1` and `L=3`. Fail ⇒ recorder contract broken.

## Data Validity

These are lightweight numerical sanity checks (pytest or a scratch cell), not dataset checks — F1
is a pure module with no HDF5 footprint.

- [ ] **Star-pattern geometry.** With `normalize_grid_init=True`, the reshaped
  `sampling_offsets.bias` `(heads,levels,points,2)` has, per head, point `i` at radius `≈ i+1`
  along that head's axis; the `num_heads` head directions are evenly spaced on the circle (angular
  gaps equal to `atol=1e-5`). Confirms init matches Deformable-DETR intent across all levels.
- [ ] **Sampling locations stay near reference at init.** Immediately after construction (offsets
  bias is the star pattern but `sampling_offsets.weight==0`), for a mid-image reference point
  `(0.5,0.5)` and `Hₗ=Wₗ=8`, all `sampling_locations` lie within `[0,1]²` (no off-map sampling for a
  centered query at the coarsest reasonable resolution). Flags pathological normalizer bugs.
- [ ] **`geometric_sigma` is train-only.** In `eval()` mode with `geometric_sigma=0.5`, two forwards
  on identical inputs are bit-identical; in `train()` mode they differ. Confirms jitter gating.
- [ ] **Output magnitude sanity.** For unit-scale random inputs, the forward output has finite,
  non-NaN values with per-element std within a plausible band (e.g. `< 10×` input std); guards
  against a normalizer or reduction that explodes activations.

## Data Architecture Integrity

F1 introduces no new HDF5 groups, keys, or `exp_key`-keyed artifacts, so the usual keying-invariant
checks do not apply. The architectural invariants F1 *must* preserve are instead:

- [ ] **Param-name stability.** The module's `state_dict` keys are exactly
  `{sampling_offsets.weight, sampling_offsets.bias, attention_weights.weight, attention_weights.bias,
  value_proj.weight, value_proj.bias, output_proj.weight, output_proj.bias}` — unchanged from
  today. No renamed, added, or dropped parameters. Guarantees old checkpoints map by name.
- [ ] **Mask2Former layout match (forward-looking).** For `n_levels=3, num_heads=8, num_points=4,
  embed_dim=256`, the four param shapes equal those of `mask2former`'s
  `MSDeformAttn(d_model=256, n_levels=3, n_heads=8, n_points=4)`:
  `sampling_offsets: (768, 256)`, `attention_weights: (384, 256)`, `value_proj: (256,256)`,
  `output_proj: (256,256)`. This is the contract F2's weight loader relies on; assert shape parity
  now so a later layout drift is caught here, not in F2.
- [ ] **`n_levels=1` collapse is total.** No code path, cache artifact, or recorded tensor at
  `n_levels=1` carries information that would alter a byte-for-byte reload of an existing
  single-scale checkpoint or its forward output (covered cumulatively by Group 1 — restated here as
  the standing invariant the migration depends on).
