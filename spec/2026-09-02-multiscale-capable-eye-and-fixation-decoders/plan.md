# F4 — Multiscale-capable eye & fixation decoders — Plan

## Context and Design Decisions

**Why the decoders change at all.** F1 already generalized the inner `DeformableAttention` op to *N*
levels: its `forward(query, reference_points, value, spatial_shape, level_start_index=None)` accepts a
`(n_levels, 2)` tensor `spatial_shape` and softmaxes jointly over `n_levels · n_points`. But the two
enclosing decoders still hard-code the single-scale contract: they slice a CLS token out of the memory
(`mem[:, 1:, :]`) and pass the fixed `self.spatial_shape = (16, 16)` tuple. F4 is the surgical change
that opts them into F1's tensor form so the F2/F3 multi-scale memory can flow through.

**Why unpacked tensors, not the bundle object** (locked this session). `ms_features.py` imports the
backbones, which import `blocks.py`; a `blocks.py` decoder importing `MultiScaleFeatures` would close
an import cycle. Passing unpacked `value` / `spatial_shapes` / `level_start_index` — exactly F1's
forward args — keeps `blocks.py` free of any `ms_features` dependency, and F6 unpacks the bundle at
the `MixerModel` call site. This mirrors the F1 decision to keep the low-level op independent of the
high-level contract.

**Why polymorphic forward, not a new method or a new class** (mirrors F1). The whole migration rides
on the single-scale path staying byte-identical so HP-search checkpoints keep loading and the
unmodified `MixerModel` keeps training until F6. A `spatial_shapes=None` sentinel selects the legacy
branch: CLS-stripped `mem`, fixed tuple, `n_levels==1`. When `spatial_shapes` is supplied, `mem` is
already the CLS-free flattened multi-level `value` (F3 strips CLS once at the bundle boundary — see
TechStack §"Multi-scale feature contract"), so the decoder passes it straight through. This is the
same "legacy tuple vs. tensor" dispatch F1 uses one level down, kept consistent for readability.

**Why CLS is stripped only on the legacy path.** Constitution invariant (TechStack "Contract changes"
table): DINOv3 emits a CLS prefix and today's decoders strip it; the Mask2Former path is CLS-free and
F3 strips DINOv3's CLS once at the bundle boundary. So in the multi-scale path the memory arriving at
the decoder is *already* CLS-free — stripping again would drop a real feature token. The
`spatial_shapes is None` sentinel is exactly the "am I on the legacy DINOv3 single-scale path?"
question, so it also gates the CLS slice.

**Why fix the dead non-`norm_first` branch now** (locked this session, full parity). Operative runs
use `norm_first=True`, so `DeformableDoubleInputDecoder`'s non-`norm_first` branch is currently dead —
and broken: it forwards `attn_mask`/`src_rope`/`mem2_rope` kwargs `__cross_attention2` doesn't accept
(instant `TypeError` if reached) and never strips CLS. Since F4 rewrites `__cross_attention2`'s
signature anyway, leaving the branch inconsistent would bake a latent trap into the migration. F4
brings it to behavioral parity with the `norm_first` branch (same value/ref/shape handling), so the
two differ only in norm placement.

**Retro-compat contract (the load-bearing invariant).** At `n_levels=1`:
- The inner `DeformableAttention` param layout, init, and numerics are byte-identical to pre-F1 (F1's
  guarantee), so adding the `n_levels=1` constructor default changes **no** `state_dict` key or shape.
- The legacy forward branch reproduces today's exact call
  (`cross_attn(query, ref, value=mem[:,1:,:], spatial_shape=(16,16), level_start_index=None)`), so the
  forward output is `torch.equal` to pre-F4.
- Therefore existing checkpoints load with zero missing/unexpected keys and the unmodified `MixerModel`
  (which never passes `spatial_shapes`) trains identically.

**Constraints honored.** Pure-PyTorch `grid_sample` inside F1 (no CUDA op, InferenceRecorder-safe);
additive to `blocks.py` only (no `mixer_model.py`/config edits — F6); DINOv3 single-scale path stays
selectable and unchanged.

---

## Step 1 — `DeformableDecoder`: constructor `n_levels`

File: `src/model/blocks.py`, `DeformableDecoder.__init__` (currently `:903`).

Add `n_levels: int = 1` to the signature (immediately after `num_points=4`). Store `self.n_levels =
n_levels`. Construct the inner op with the level count:

```python
self.cross_attn = DeformableAttention(embed_dim=model_dim,
                                      num_heads=n_heads,
                                      num_points=num_points,
                                      n_levels=n_levels,          # NEW
                                      geometric_sigma=geometric_sigma,
                                      attn_dropout=attn_dropout,
                                      normalize_grid_init=normalize_grid_init,
                                      **factory_kwargs)
```

`self.spatial_shape` is kept as-is (legacy fallback). No other constructor line changes.

## Step 2 — `DeformableDecoder`: polymorphic `__cross_attention` + `forward`

File: `src/model/blocks.py`, same class.

Rewrite the private helper to take the value and shape explicitly:

```python
def __cross_attention(self, src, value, reference_points=None,
                      spatial_shapes=None, level_start_index=None):
    shape = spatial_shapes if spatial_shapes is not None else self.spatial_shape
    return self.dropout2(self.cross_attn(query=src, reference_points=reference_points,
                                         value=value, spatial_shape=shape,
                                         level_start_index=level_start_index))
```

Rewrite `forward` to dispatch the CLS slice once, before either norm branch:

```python
def forward(self, src, mem, tgt_mask=None, reference_points=None,
            spatial_shapes=None, level_start_index=None):
    if spatial_shapes is None:                 # legacy single-scale (CLS-prefixed memory)
        if self.n_levels != 1:
            raise ValueError("legacy single-scale path requires n_levels==1")
        value = mem[:, 1:, :]
    else:                                      # F3 multi-scale bundle (CLS-free)
        value = mem
    x = src
    if self.norm_first:
        temp = self.__self_attention(self.norm1(x), attn_mask=tgt_mask)
        record ... "self_attention_res"
        x = x + temp
        temp = self.__cross_attention(self.norm2(x), value, reference_points=reference_points,
                                      spatial_shapes=spatial_shapes,
                                      level_start_index=level_start_index)
        record ... "cross_attention_res"
        x = x + temp
        temp = self.__feed_forward(self.norm3(x)); record "ffn_res"; x = x + temp
    else:
        x = self.norm1(x + self.__self_attention(x, attn_mask=tgt_mask))
        x = self.norm2(x + self.__cross_attention(x, value, reference_points=reference_points,
                                                  spatial_shapes=spatial_shapes,
                                                  level_start_index=level_start_index))
        x = self.norm3(x + self.__feed_forward(x))
    return x
```

Note the name mangling: inside the class the call is `self._DeformableDecoder__cross_attention` via
`self.__cross_attention`; keep the private double-underscore name. The legacy branch reproduces the
exact prior call (`value = mem[:,1:,:]`, `shape = self.spatial_shape`, `level_start_index=None`) ⇒
byte-identical (FR3).

## Step 3 — `DeformableDoubleInputDecoder`: constructor `n_levels`

File: `src/model/blocks.py`, `DeformableDoubleInputDecoder.__init__` (currently `:990`).

Add `n_levels: int = 1` after `num_points=4`; store `self.n_levels = n_levels`. Construct **only** the
second cross-attention with the level count (the first cross-attention stays a plain
`MultiHeadedAttention`):

```python
self.second_cross_attn = DeformableAttention(embed_dim=model_dim,
                                             num_heads=n_heads,
                                             num_points=num_points,
                                             n_levels=n_levels,   # NEW
                                             attn_dropout=attn_dropout,
                                             normalize_grid_init=normalize_grid_init,
                                             **factory_kwargs)
```

## Step 4 — `DeformableDoubleInputDecoder`: fix `__cross_attention2` signature

File: `src/model/blocks.py`, same class. Replace the helper (drops the bogus rope/mask kwargs the
non-`norm_first` branch used to pass):

```python
def __cross_attention2(self, src, value, reference_points=None,
                       spatial_shapes=None, level_start_index=None):
    shape = spatial_shapes if spatial_shapes is not None else self.spatial_shape
    return self.second_cross_attn_dropout(
        self.second_cross_attn(query=src, reference_points=reference_points,
                               value=value, spatial_shape=shape,
                               level_start_index=level_start_index))
```

`__self_attention`, `__cross_attention1` (the `MultiHeadedAttention` over `mem1`), and
`__feed_forward` are unchanged.

## Step 5 — `DeformableDoubleInputDecoder.forward`: dispatch + non-`norm_first` parity fix

File: `src/model/blocks.py`, same class. New signature adds `spatial_shapes`/`level_start_index`;
strip CLS once up front; both branches call `__cross_attention2` identically (FR5–FR8):

```python
def forward(self, src, mem1, mem2, tgt_mask=None, mem1_mask=None, mem2_mask=None,
            reference_points=None, spatial_shapes=None, level_start_index=None):
    if spatial_shapes is None:                 # legacy single-scale (CLS-prefixed image memory)
        if self.n_levels != 1:
            raise ValueError("legacy single-scale path requires n_levels==1")
        value2 = mem2[:, 1:, :]
    else:                                      # F3 multi-scale bundle (CLS-free)
        value2 = mem2
    x = src
    if self.norm_first:
        temp = self.__self_attention(self.self_attn_norm(x), attn_mask=tgt_mask, src_rope=None)
        record "self_attention_res"; x = x + temp
        temp = self.__cross_attention1(self.first_cross_attn_norm(x), mem1, attn_mask=mem1_mask,
                                       src_rope=None, mem1_rope=None)
        record "first_cross_res"; x = x + temp
        temp = self.__cross_attention2(self.second_cross_attn_norm(x), value2,
                                       reference_points=reference_points,
                                       spatial_shapes=spatial_shapes,
                                       level_start_index=level_start_index)
        record "second_cross_res"; x = x + temp
        temp = self.__feed_forward(self.linear_norm(x)); record "ffn_res"; x = x + temp
    else:
        x = self.self_attn_norm(x + self.__self_attention(x, attn_mask=tgt_mask, src_rope=None))
        x = self.first_cross_attn_norm(x + self.__cross_attention1(x, mem1, attn_mask=mem1_mask,
                                                                   src_rope=None, mem1_rope=None))
        x = self.second_cross_attn_norm(x + self.__cross_attention2(
                x, value2, reference_points=reference_points,
                spatial_shapes=spatial_shapes, level_start_index=level_start_index))   # FIXED
        x = self.linear_norm(x + self.__feed_forward(x))
    return x
```

The **only** behavioral change to the operative (`norm_first`) path is that the CLS slice now happens
via `value2` up front instead of inline `mem2[:,1:,:]` — numerically identical (FR8). The
non-`norm_first` branch previously (a) passed `mem2` un-stripped and (b) forwarded three kwargs
`__cross_attention2` rejected; both are now fixed to parity (FR7).

## Step 6 — Tests: `tests/test_ms_decoders.py`

New CPU-only suite (fixed seeds, no network), covering:
- Byte-identity vs. a pinned pre-F4 reference for both decoders at `n_levels=1`, both norm modes
  (FR3, FR8). Build a reference by capturing output on the current code before edits, or re-derive by
  constructing the inner op directly; assert `torch.equal`.
- `state_dict` key/shape parity at `n_levels=1`; a pre-F4-shaped `state_dict` loads with zero
  missing/unexpected keys.
- Multi-scale forward at `n_levels=3` with a synthetic bundle
  (`spatial_shapes=[[8,8],[16,16],[32,32]]`, `value (B,1344,D)`, `level_start_index=[0,64,320]`,
  `reference_points (B,Nq,2)`): output shape `(B, Nq, D)`, finite.
- Error paths (FR10): multi-level `spatial_shapes` into an `n_levels=1` decoder; legacy call
  (`spatial_shapes=None`) into an `n_levels=3` decoder; `ΣHₗWₗ` mismatch.
- `norm_first` vs non-`norm_first` `DeformableDoubleInputDecoder`: on shared inputs both run without
  error and produce finite output of the right shape (the non-`norm_first` branch no longer raises).
- KV/memory-cache carry-through (FR9): cold vs warm second-cross-attn value cache at `n_levels=3`
  gives matching output.
- InferenceRecorder hooks fire with a level axis at `n_levels=3` and squeeze to the legacy shape at
  `n_levels=1`.

---

## Implementation Order

1. **Step 1** — `DeformableDecoder.__init__`: add `n_levels`, build inner op at that level count.
2. **Step 2** — `DeformableDecoder`: polymorphic `__cross_attention` + `forward` (CLS dispatch,
   forward `spatial_shapes`/`level_start_index`).
3. **Step 3** — `DeformableDoubleInputDecoder.__init__`: add `n_levels`, build `second_cross_attn` at
   that level count.
4. **Step 4** — `DeformableDoubleInputDecoder.__cross_attention2`: new signature, drop bogus kwargs.
5. **Step 5** — `DeformableDoubleInputDecoder.forward`: CLS dispatch + non-`norm_first` parity fix.
6. **Step 6** — `tests/test_ms_decoders.py`.

(1–2 are independent of 3–5; 6 follows all.)
