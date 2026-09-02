# F4 — Validation

CPU-only, fixed seeds (`torch.manual_seed(0)`), no network, no HDF5. Reference dims unless stated:
`model_dim = D = 32`, `n_heads = 4`, `num_points = 4`, batch `B = 2`, queries `Nq = 5`. Single-scale
legacy memory: `H*W = 16*16 = 256`, CLS-prefixed ⇒ `mem (B, 257, D)`, `self.spatial_shape = (16,16)`.
Synthetic 3-level bundle: `spatial_shapes = [[8,8],[16,16],[32,32]]` ⇒ `ΣHₗWₗ = 64+256+1024 = 1344`,
`level_start_index = [0, 64, 320]`, CLS-free `value (B, 1344, D)`. Reference points `(B, Nq, 2)` in
`[0,1]`. "pre-F4 reference" = the decoder output captured from the class as it stood before this
feature (pin the tensor, or reconstruct by calling the inner `DeformableAttention` directly with the
legacy args).

## Code Correctness

### Group 1 — Retro-compat byte-identity (`DeformableDecoder`)
- [ ] `DeformableDecoder(model_dim=D, n_heads=n_heads, num_points=4)` builds with `n_levels == 1` by
      default; `self.cross_attn.n_levels == 1`.
- [ ] `state_dict()` keys and every tensor shape at `n_levels=1` are identical to the pre-F4 class
      (diff the two key→shape maps; expect empty diff).
- [ ] A pre-F4-saved `state_dict` loads via `load_state_dict(..., strict=True)` with **zero** missing
      and **zero** unexpected keys.
- [ ] `norm_first=True`: `forward(src, mem, tgt_mask, reference_points)` (legacy call, no
      `spatial_shapes`) output is `torch.equal` to the pre-F4 reference (seeded, eval mode).
- [ ] `norm_first=False`: same `torch.equal` byte-identity against the pre-F4 reference.
- [ ] Legacy path strips CLS: the op receives `mem[:,1:,:]` (length 256), not the full 257 — assert by
      recording `value` length or by shape mismatch if the full memory were passed with `(16,16)`.

### Group 2 — Multi-scale path (`DeformableDecoder`)
- [ ] `DeformableDecoder(..., n_levels=3)` builds with `self.cross_attn.n_levels == 3`;
      `sampling_offsets` weight shape is `(n_heads·3·4·2, D)`, `attention_weights` `(n_heads·3·4, D)`.
- [ ] `forward(src, value(B,1344,D), reference_points=(B,Nq,2), spatial_shapes=[[8,8],[16,16],[32,32]],
      level_start_index=[0,64,320])` returns `(B, Nq, D)`, all-finite, `dtype == src.dtype`.
- [ ] `value` is passed through **without** CLS slicing on the multi-scale path: a `value` of length
      exactly `1344` succeeds; a length-`1345` value (as if CLS still present) raises F1's
      `ΣHₗWₗ ≠ value.shape[1]` `ValueError`.
- [ ] `level_start_index=None` on the multi-scale path is derived by F1 and gives output `torch.equal`
      to passing the explicit `[0,64,320]`.

### Group 3 — Retro-compat + multi-scale (`DeformableDoubleInputDecoder`)
- [ ] Default `n_levels == 1`; only `second_cross_attn.n_levels` tracks `n_levels` — `first_cross_attn`
      is a `MultiHeadedAttention` with no `n_levels` attribute.
- [ ] `state_dict` keys/shapes at `n_levels=1` identical to pre-F4; pre-F4 `state_dict` loads with zero
      missing/unexpected keys.
- [ ] `norm_first=True` legacy call
      `forward(src, mem1, mem2(B,257,D), tgt_mask, mem1_mask, reference_points=ref)` output is
      `torch.equal` to the pre-F4 reference.
- [ ] `n_levels=3` multi-scale call
      `forward(src, mem1, value(B,1344,D), reference_points=ref, spatial_shapes=..., level_start_index=...)`
      returns `(B, Nq, D)`, finite. `mem1` still flows through the first cross-attention unchanged
      (drop/replace `mem1` → output changes; drop/replace `value` → output changes).

### Group 4 — Non-`norm_first` branch parity (bug fix, FR7)
- [ ] `DeformableDoubleInputDecoder(norm_first=False)` `forward(...)` **runs without raising** on the
      legacy call — pre-F4 it raised `TypeError` (unexpected `attn_mask`/`src_rope`/`mem2_rope` kwargs).
- [ ] Non-`norm_first` legacy path strips CLS (`value2 = mem2[:,1:,:]`): a length-257 `mem2` succeeds;
      the op never sees length 257 against a `(16,16)` shape (would be a `ΣHₗWₗ` mismatch if unstripped).
- [ ] `__cross_attention2` no longer accepts `attn_mask`/`src_rope`/`mem2_rope`: calling it with any
      of them raises `TypeError` (kwargs removed from the signature).
- [ ] Non-`norm_first` and `norm_first` variants, given the same weights and inputs, both produce
      finite `(B,Nq,D)` output; they differ only by norm placement (not asserted equal — different
      residual structure — but both must be finite and correctly shaped).

### Group 5 — Error conditions (FR10)
- [ ] Passing `spatial_shapes=[[8,8],[16,16],[32,32]]` (3 levels) to a decoder built with `n_levels=1`
      raises `ValueError` ("spatial_shape has 3 levels but module has n_levels=1", from F1).
- [ ] Legacy call (`spatial_shapes=None`) on a decoder built with `n_levels=3` raises the decoder's
      `ValueError` ("legacy single-scale path requires n_levels==1") — **not** a silent CLS mis-slice.
- [ ] `reference_points` with last dim ≠ 2 raises F1's `ValueError` (box refs out of scope).
- [ ] A `level_start_index` inconsistent with `spatial_shapes` (e.g. `[0,64,300]`) raises F1's
      `ValueError`, surfaced unchanged through the decoder.

### Group 6 — KV / memory cache (FR9)
- [ ] `DeformableDoubleInputDecoder(..., n_levels=3)`: after `enable_memory_kv_cache()`, two
      successive `forward` calls with the **same** `value`/`spatial_shapes` give `torch.equal` outputs,
      and the second populates `second_cross_attn.value_cache` as a length-3 per-level list.
- [ ] Output with the cache warm equals output with the cache disabled/cleared (cold) at `n_levels=3`
      within `atol=0` (`torch.equal`) — geometry fixed during decode.
- [ ] `disable_memory_kv_cache()` / `clear_kv_cache()` reset `value_cache` to `None`.

### Group 7 — InferenceRecorder carry-through
- [ ] With recording enabled at `n_levels=3`, `DeformableDecoder` records `cross_attention_res`
      `(B,Nq,D)` and the inner op records `sampling_offsets`/`sampling_locations` `(B,Nq,n_heads,3,4,2)`,
      `attention_weights` `(B,Nq,n_heads,3,4)`, `reference_points` `(B,Nq,3,2)`.
- [ ] At `n_levels=1` the recorded inner tensors have a squeezable singleton level axis (`...,1,...`),
      so existing recorder consumers keep working.
- [ ] `self_attention_res`, `cross_attention_res`/`second_cross_res`, `ffn_res` all fire in both norm
      branches of both decoders.

## Data Validity

These run on real model tensors (a small `MixerModel`-shaped decoder stack, or the F3 adapters feeding
a decoder directly) rather than pure synthetic noise — sanity that the multi-scale wiring is
geometrically meaningful, not just shape-correct.

- [ ] **Feed an F3 bundle straight into an `n_levels=3` decoder.** Build a
      `Mask2FormerFeatureAdapter(Mask2FormerBackbone(imagenet_weights=None))`, run
      `bundle = adapter(torch.randn(2,3,256,256))`, then
      `DeformableDecoder(model_dim=256, n_levels=3)(src, bundle.value, reference_points=ref,
      spatial_shapes=bundle.spatial_shapes, level_start_index=bundle.level_start_index)` — output
      `(2, Nq, 256)`, finite. Confirms the F3 contract plugs into F4 with no glue.
- [ ] **Reference points at cell centers sample near the corresponding token.** With attention weights
      forced to a single point (or `num_points=1`, offsets≈0), a query whose `reference_points` sits at
      a known token's center returns a value close to that token's projected feature (cross-check the
      grid_sample against a manual gather), within `atol=1e-4`.
- [ ] **Single-scale vs. this-feature single-scale agree on real memory.** Run the pre-F4 decoder and
      the F4 decoder (`n_levels=1`, legacy call) on the same DINOv3-style CLS-prefixed memory; per-token
      max abs difference `== 0`.

## Data Architecture Integrity

The "keying invariants" for F4 are the **contract-boundary invariants** between the decoders and F1/F3
(there is no HDF5 or `exp_key` surface here — F4 is a pure in-memory module change).

- [ ] **CLS is stripped exactly once, on exactly one path.** In the multi-scale path the decoder does
      **not** slice `mem` (grep the two `forward`s: `[:, 1:, :]` appears only under the
      `spatial_shapes is None` branch). Feeding a CLS-free `value` of the correct length succeeds;
      there is no double-strip.
- [ ] **`n_levels` is single-sourced from the constructor to the inner op.** The decoder never
      hard-codes a level count; `decoder.cross_attn.n_levels == decoder.n_levels` (and likewise
      `second_cross_attn.n_levels`) for `n_levels ∈ {1, 3, 4}`.
- [ ] **Legacy/multi-scale dispatch cannot be silently bypassed.** The `spatial_shapes is None`
      sentinel is the sole selector; a decoder built with `n_levels>1` **cannot** be driven down the
      legacy CLS-stripping branch (it raises), and a decoder built with `n_levels=1` **cannot** be fed a
      multi-level `spatial_shapes` (F1 raises). Both mis-pairings are covered in Group 5.
- [ ] **No import cycle introduced.** `import src.model.blocks` does not import `src.model.ms_features`
      (grep the module: no `ms_features` / `MultiScaleFeatures` reference); the decoders take unpacked
      tensors only.
- [ ] **Byte-identity gate for the migration.** The Group 1 / Group 3 `torch.equal` and zero
      missing/unexpected-key checks are the invariant the rest of the migration depends on — treat any
      failure as a blocking regression, not a tolerance to loosen.
