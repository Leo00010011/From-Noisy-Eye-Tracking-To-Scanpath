# F4 — Multiscale-capable eye & fixation decoders — Requirements

## Goal

Generalize the two deformable cross-attention decoders — `DeformableDecoder` (the eye decoder)
and `DeformableDoubleInputDecoder` (the fixation decoder) in `src/model/blocks.py` — from a single
fixed image scale to *N* feature levels, so they can consume the multi-scale memory produced by the
F2 backbone and packaged by the F3 `MultiScaleFeatures` bundle. This is the decoder-side half of the
Mask2Former migration: F1 already made the inner `DeformableAttention` op N-level capable, and F2/F3
produce and package the multi-scale features; F4 opts the decoders into that op's tensor
`spatial_shape` + `level_start_index` form. The change is **retro-compatible**: at a single level,
with the legacy CLS-prefixed memory and the fixed `spatial_shape` tuple, every forward output is
**byte-identical** to today's, so existing single-scale checkpoints (HP-search runs) load with zero
missing/unexpected keys and the unmodified `MixerModel` keeps training identically until F6 wires the
bundle end-to-end.

## Scope

**In scope**
- `DeformableDecoder` (`src/model/blocks.py:903`): add an `n_levels` constructor arg (default `1`),
  build the inner `DeformableAttention` at that level count, and make `forward` polymorphic — accept
  either the legacy CLS-prefixed `mem` + fixed `self.spatial_shape` (single level) **or** an already
  CLS-free flattened multi-level `value` + `spatial_shapes`/`level_start_index` tensors.
- `DeformableDoubleInputDecoder` (`src/model/blocks.py:990`): the same generalization on its second
  (deformable) cross-attention. Its first cross-attention (`MultiHeadedAttention` over the gaze
  memory `mem1`) is untouched.
- Fix the flagged latent bug in `DeformableDoubleInputDecoder.forward`'s **non-`norm_first`** branch
  (`blocks.py:1125`): it calls `__cross_attention2` with `attn_mask`/`src_rope`/`mem2_rope` kwargs the
  method does not accept (a `TypeError` if ever reached) and, unlike the `norm_first` branch, does not
  strip CLS. Bring it to full behavioral parity with the `norm_first` branch.
- New test suite `tests/test_ms_decoders.py`.

**Out of scope (explicitly)**
- **No edits to `src/model/mixer_model.py`, `src/training/pipeline_builder.py`, or any config.**
  Wiring the F3 bundle through `MixerModel.encode` / `decode_fixation`, the `img_input_proj`
  256→`model_dim` change, per-level positional encoding + `level_embed`, and the backbone config group
  are **F6**. F4 only makes the decoders *capable*; the unmodified `MixerModel` continues to call the
  legacy single-scale path.
- No change to `DeformableAttention` (F1, done) — F4 only constructs it with `n_levels>1` and passes
  the tensor `spatial_shape` form.
- No change to the non-deformable decoders (`TransformerDecoder`, `DoubleInputDecoder`) or the
  `MultiHeadedAttention` first cross-attention.
- Per-level (`(B,Nq,n_levels,2)`) reference points are **not** constructed here — the decoders pass
  the existing 2-D `(B,Nq,2)` reference points straight through, and F1's op broadcasts them across
  levels. (Distinct per-level refs remain available via F1 but are an F6/later concern.)
- The stride-4 (res2, 64²) 4th level remains behind F2's `return_stride4` flag; F4 is level-count
  agnostic (`n_levels` is a plain int) but does not itself wire a 4th level.

## Functional Requirements

**FR1 — `DeformableDecoder` gains `n_levels`.** The constructor signature adds `n_levels: int = 1`
(placed after `num_points`). The inner `self.cross_attn = DeformableAttention(...)` is constructed with
`n_levels=n_levels`. All other constructor args, their defaults, and their order are unchanged.
`self.spatial_shape` (default `(16, 16)`) is retained as the legacy single-level fallback.

**FR2 — `DeformableDecoder.forward` is polymorphic.** New signature:
`forward(src, mem, tgt_mask=None, reference_points=None, spatial_shapes=None, level_start_index=None)`.
- **Legacy path** (`spatial_shapes is None`): the memory `mem` is treated as CLS-prefixed; CLS is
  stripped via `mem[:, 1:, :]`, and the inner op is called with the fixed `self.spatial_shape` tuple
  and `level_start_index=None`. Requires `self.n_levels == 1`.
- **Multi-scale path** (`spatial_shapes is not None`): `mem` is the already CLS-free flattened
  multi-level `value` `(B, ΣHₗWₗ, D)` (from the F3 bundle); it is passed through **without** slicing,
  and `spatial_shapes` `(L,2)` + `level_start_index` `(L,)` are forwarded to the inner op.
- `reference_points` `(B, Nq, 2)` is passed unchanged in both paths (F1 broadcasts across levels).
- Recorder hooks (`self_attention_res`, `cross_attention_res`, `ffn_res`) fire identically in both
  branches.

**FR3 — Byte-identical single-scale legacy path.** With `n_levels=1`, `spatial_shapes=None`, and the
same inputs, `DeformableDecoder.forward` produces output `torch.equal` to the pre-F4 class (both the
`norm_first` and non-`norm_first` branches). The module's `state_dict` keys and tensor shapes are
unchanged at `n_levels=1`, so a pre-F4 checkpoint loads with zero missing/unexpected keys.

**FR4 — `DeformableDoubleInputDecoder` gains `n_levels`.** The constructor signature adds
`n_levels: int = 1` (placed after `num_points`). Only `self.second_cross_attn = DeformableAttention(...)`
is constructed with `n_levels=n_levels`; `self.first_cross_attn` (a `MultiHeadedAttention`) is
unchanged. `self.spatial_shape` (default `(16, 16)`) is retained as the legacy fallback.

**FR5 — `DeformableDoubleInputDecoder.forward` is polymorphic on the second cross-attention.** New
signature:
`forward(src, mem1, mem2, tgt_mask=None, mem1_mask=None, mem2_mask=None, reference_points=None, spatial_shapes=None, level_start_index=None)`.
`mem1` (gaze memory) flows through the first cross-attention unchanged. `mem2` (image memory) feeds
the deformable second cross-attention with the same legacy/multi-scale dispatch as FR2:
`spatial_shapes is None` ⇒ strip CLS from `mem2` and use `self.spatial_shape`; otherwise pass `mem2`
through with `spatial_shapes`/`level_start_index`.

**FR6 — `__cross_attention2` signature fix.** `__cross_attention2(self, src, value, reference_points=None, spatial_shapes=None, level_start_index=None)`
forwards `spatial_shape=spatial_shapes if spatial_shapes is not None else self.spatial_shape` and
`level_start_index=level_start_index` to `self.second_cross_attn`. It **no longer** accepts or forwards
`attn_mask`, `src_rope`, or `mem2_rope` (the deformable op has no rope/mask hooks).

**FR7 — Non-`norm_first` branch parity (bug fix).** In `DeformableDoubleInputDecoder.forward`, the
non-`norm_first` branch calls `__cross_attention2` with exactly the same value/reference/shape
arguments as the `norm_first` branch (CLS stripped in the legacy path; `spatial_shapes`/
`level_start_index` forwarded in the multi-scale path) and **no** unaccepted kwargs. After the fix the
two branches differ only in pre- vs post-LayerNorm placement, matching every other decoder in the file.

**FR8 — Byte-identical single-scale legacy path (double-input).** With `n_levels=1`,
`spatial_shapes=None`, `norm_first=True`, and the same inputs, `DeformableDoubleInputDecoder.forward`
output is `torch.equal` to the pre-F4 class. `state_dict` keys/shapes are unchanged at `n_levels=1`.
(The non-`norm_first` branch was previously non-functional dead code, so it has no byte-identity
obligation — FR7 makes it *correct*, validated numerically against the `norm_first` branch on shared
inputs where applicable.)

**FR9 — KV / memory cache carry-through.** `enable_memory_kv_cache` / `disable_memory_kv_cache` /
`clear_kv_cache` continue to delegate to the inner ops unchanged. With `n_levels>1`, the second
cross-attention's per-level value cache (F1) is populated on the first decode step and reused on
subsequent steps; results with the cache warm match results with it cold (memory geometry fixed during
autoregressive decode).

**FR10 — Error conditions.** All raised as the inner op's `ValueError` (F4 adds no new validation
beyond the dispatch guard below):
- Passing a multi-level `spatial_shapes` (L>1) to a decoder built with `n_levels=1` raises F1's
  "spatial_shape has L levels but module has n_levels=N".
- Legacy path (`spatial_shapes is None`) on a decoder built with `n_levels>1` raises a `ValueError`
  from the decoder ("legacy single-scale path requires n_levels==1"), rather than silently
  mis-slicing CLS.
- `ΣHₗWₗ ≠ value.shape[1]`, `reference_points` last dim ≠ 2, and inconsistent `level_start_index`
  all surface F1's existing `ValueError`s unchanged.

## Public API Summary

```python
class DeformableDecoder(nn.Module):
    def __init__(self, model_dim=1024, total_dim=1024, n_heads=8, ff_dim=2048, dropout_p=0,
                 activation=F.relu, eps=1e-5, norm_first=False, num_points=4,
                 n_levels=1,                     # NEW
                 spatial_shape=(16, 16), geometric_sigma=0, attn_dropout=0.0,
                 normalize_grid_init=True, device='cpu', dtype=torch.float32): ...

    def forward(self, src, mem, tgt_mask=None, reference_points=None,
                spatial_shapes=None, level_start_index=None):   # NEW kwargs
        # spatial_shapes is None  -> legacy: mem is CLS-prefixed, uses self.spatial_shape (n_levels==1)
        # spatial_shapes not None -> mem is CLS-free (B, ΣHₗWₗ, D); multi-level
        ...

class DeformableDoubleInputDecoder(nn.Module):
    def __init__(self, model_dim=1024, total_dim=1024, n_heads=8, ff_dim=2048, dropout_p=0,
                 activation=F.relu, eps=1e-5, norm_first=False, use_kv_cache=False,
                 spatial_shape=(16, 16), num_points=4,
                 n_levels=1,                     # NEW
                 attn_dropout=0.0, normalize_grid_init=True, device='cpu', dtype=torch.float32): ...

    def forward(self, src, mem1, mem2, tgt_mask=None, mem1_mask=None, mem2_mask=None,
                reference_points=None,
                spatial_shapes=None, level_start_index=None):   # NEW kwargs
        # mem2 dispatch identical to DeformableDecoder.forward; mem1 path unchanged
        ...
```

## Dependencies

| Reads from | Provided by | Notes |
|---|---|---|
| `DeformableAttention(embed_dim, num_heads, num_points, n_levels, ...)` | F1, `src/model/blocks.py` | Inner op; already N-level capable and byte-identical at `n_levels=1`. |
| `value (B, ΣHₗWₗ, D)`, `spatial_shapes (L,2)`, `level_start_index (L,)` | F3 `MultiScaleFeatures`, `src/model/ms_features.py` | Supplied **by F6** at the call site (unpacked from the bundle). F4 never imports the dataclass — decoders take unpacked tensors, avoiding the `blocks ← ms_features` import cycle. |
| `reference_points (B, Nq, 2)` | Caller (`MixerModel`, unchanged in F4) | Gaze/fixation coords; broadcast across levels inside F1. |

| Writes to / touched | Notes |
|---|---|
| `src/model/blocks.py` | `DeformableDecoder`, `DeformableDoubleInputDecoder` only. Additive; single-scale path byte-identical. |
| `tests/test_ms_decoders.py` | New CPU-only suite. |

**Not touched (deferred to F6):** `src/model/mixer_model.py`, `src/training/pipeline_builder.py`,
`configs/model/` — the `MixerModel` keeps calling the legacy path until F6.
