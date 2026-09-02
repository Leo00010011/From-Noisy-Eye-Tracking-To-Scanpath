# F3 — Validation

CPU-only, fixed seeds, no network. Mask2Former tests use `Mask2FormerBackbone(imagenet_weights=None)`
(random init); DINOv3 tests use a `FakeDino` stub returning CLS-prefixed random tokens. Reference
image `x = torch.randn(2, 3, 256, 256)` unless stated. At `img_size=256` the Mask2Former backbone
emits levels `[(8,8), (16,16), (32,32)]` (coarse→fine), so `S = 64 + 256 + 1024 = 1344`,
`level_start_index = [0, 64, 320]`, `D = 256`.

## Code Correctness

### Group 1 — Bundle construction and validation
- [ ] A well-formed `MultiScaleFeatures` (`value (2,1344,256)`, `spatial_shapes [[8,8],[16,16],[32,32]]`,
      `level_start_index [0,64,320]`, `reference_grids (1344,2)`) constructs without error.
- [ ] `num_levels == 3`, `embed_dim == 256`, `batch_size == 2`, `seq_len == 1344`,
      `level_sizes() == [64, 256, 1024]`.
- [ ] `value.dim() != 3` (e.g. `(2,1344)`) raises `ValueError`.
- [ ] `Σ Hₗ·Wₗ != value.shape[1]` (e.g. `value` length 1000) raises `ValueError`.
- [ ] `reference_grids.shape != (S, 2)` (e.g. `(1344,3)` or `(1000,2)`) raises `ValueError`.
- [ ] `spatial_shapes.shape != (L, 2)` raises `ValueError`.
- [ ] A `level_start_index` inconsistent with `spatial_shapes` (e.g. `[0,64,300]`) raises
      `ValueError`.
- [ ] A float `spatial_shapes` or float `level_start_index` raises `ValueError` (dtype not coerced
      silently).
- [ ] A non-floating `value` (e.g. int64) raises `ValueError`.

### Group 2 — Geometry helpers
- [ ] `build_level_start_index(tensor([[8,8],[16,16],[32,32]]))` equals `tensor([0,64,320])`,
      dtype `int64`.
- [ ] `build_reference_grids(tensor([[2,2]]))` equals `[[0.25,0.25],[0.75,0.25],[0.25,0.75],[0.75,0.75]]`
      (x fastest, `(x,y)` order) within `atol=1e-6`.
- [ ] `build_reference_grids` output shape is `(S, 2)`, dtype `float32`, all values in `(0, 1)`.
- [ ] Ordering cross-check: for a `(8,8)` level, `reference_grids[i]` equals
      `((i%8 + 0.5)/8, (i//8 + 0.5)/8)` for all `i` in `[0,64)`.
- [ ] `build_reference_grids` matches `MSDeformAttnTransformerEncoder.get_reference_points`:
      for `spatial_shapes=[[8,8],[16,16],[32,32]]`, `get_reference_points(...)[0, :, 0, :]`
      (batch 0, level-0 slice, which is level-independent) equals `build_reference_grids(...)`
      within `atol=1e-6`.

### Group 3 — Mask2Former adapter
- [ ] `Mask2FormerFeatureAdapter(backbone).embed_dim == 256` and `.num_levels == 3`.
- [ ] `forward(x)` returns a `MultiScaleFeatures` with `value.shape == (2,1344,256)`,
      `spatial_shapes.tolist() == [[8,8],[16,16],[32,32]]`, `level_start_index.tolist() == [0,64,320]`,
      `reference_grids.shape == (1344,2)`.
- [ ] **Round-trip identity:** the adapter's `value` equals the concatenation of the backbone's
      returned maps each flattened via `m.flatten(2).transpose(1,2)` (`torch.equal`).
- [ ] Adapter `spatial_shapes`/`level_start_index` equal the tensors the backbone's pixel decoder
      produced internally (patch `MSDeformAttnTransformerEncoderOnly.forward` to capture them, or
      reconstruct from the map shapes) — `torch.equal`.
- [ ] Dynamic size: at `x = randn(2,3,128,128)` levels are `[(4,4),(8,8),(16,16)]`, `S = 336`, and
      the bundle validates.
- [ ] `return_stride4=True`: adapter reports `num_levels == 4`, `value.shape == (2, 1344+4096, 256)`,
      the `mask_features` tensor is **not** present in the bundle, and the bundle validates.

### Group 4 — DINOv3 adapter
- [ ] With `FakeDino(embed_dim=384, patch_size=16)` on `x=(2,3,256,256)`:
      `forward(x)` returns `value.shape == (2,256,384)` (CLS stripped — one token fewer than the
      stub's `(2,257,384)` output), `spatial_shapes.tolist() == [[16,16]]`,
      `level_start_index.tolist() == [0]`, `num_levels == 1`, `embed_dim == 384`.
- [ ] The stripped token is the prefix: `bundle.value` equals `stub_output[:, 1:, :]`
      (`torch.equal`).
- [ ] Phantom-prefix guard: a stub returning `(2, 256, 384)` (no CLS) while `patch_size=16`
      yields `H'·W'=256 != 255` after stripping → `ValueError`.
- [ ] `num_prefix_tokens=0` (register-token-free variant) strips nothing and validates
      (`value.shape == (2,256,384)` when the stub returns 256 tokens).
- [ ] Non-square input `x=(2,3,256,128)` gives `spatial_shapes == [[16,8]]`, `S=128`, validates.

### Group 5 — F1 consumability
- [ ] A Mask2Former bundle feeds `DeformableAttention(embed_dim=256, num_heads=8, num_points=4,
      n_levels=3)`: with `query=(2,5,256)` and `reference_points=(2,5,2)` in `[0,1]`,
      `attn(query, ref, bundle.value, bundle.spatial_shapes, bundle.level_start_index)` returns
      `(2,5,256)` with no error.
- [ ] The same call with `reference_points = bundle.reference_grids[None, :5]` (grid-anchored refs)
      also runs and returns `(2,5,256)`.
- [ ] A DINOv3 bundle feeds `DeformableAttention(embed_dim=384, …, n_levels=1)` and returns
      `(B, Nq, 384)`.
- [ ] Passing a bundle whose `spatial_shapes` has 3 rows to an `n_levels=1` attention raises the
      F1 `ValueError` ("spatial_shape has 3 levels but module has n_levels=1") — confirms the
      contract is enforced end-to-end.

### Group 6 — Module integrity
- [ ] `.to(dtype=torch.float64)` on a bundle casts `value` and `reference_grids` to float64 but
      leaves `spatial_shapes` and `level_start_index` `int64`.
- [ ] `.to()` returns a new object; the original tensors are unchanged.
- [ ] Adapters register the backbone as a submodule: `Mask2FormerFeatureAdapter(bb).state_dict()`
      keys are all prefixed `backbone.`, and the adapter adds **zero** parameters of its own
      (`sum(p.numel() for p in adapter.parameters()) == sum(p.numel() for p in bb.parameters())`).
- [ ] `adapter.eval()` propagates to the backbone (`adapter.backbone.feature_extractor.training is
      False`); `adapter.train()` keeps the frozen ResNet in eval (F2's `train()` override still
      applies through the submodule).
- [ ] `InferenceRecorder` attached to the adapter captures the backbone's F1 deformable tensors
      with a level axis of 3 (recorder hooks propagate through the submodule).

## Data Validity

- [ ] **Reference grid coverage.** For every level, `reference_grids` restricted to that level
      spans `((0.5/W, 0.5/H), ((W-0.5)/W, (H-0.5)/H))` — min/max per axis match to `atol=1e-6`;
      no value is `≤0` or `≥1`.
- [ ] **Grid spacing.** Within the `(32,32)` level, consecutive x-centers differ by `1/32`
      (`atol=1e-6`); the y-center increments by `1/32` every 32 tokens.
- [ ] **Value finiteness.** With a random-init backbone the bundle `value` is all-finite
      (`torch.isfinite(value).all()`), confirming the flatten/concat introduces no NaN/Inf.
- [ ] **Level energy sanity.** The per-level mean absolute activation of `value` is within the same
      order of magnitude across the three levels (the pixel decoder's GroupNorm keeps scales
      comparable) — a coarse guard that no level is accidentally zeroed by a mis-slice.

## Data Architecture Integrity

These replace the HDF5/`exp_key` invariants (there is no cache here) with the F3 ordering/geometry
invariants that downstream sampling relies on.

- [ ] **No phantom prefix (CLS) token.** For the DINOv3 adapter, `value.shape[1]` equals exactly
      `H'·W'` (the patch count) — never `H'·W' + prefix`. The prefix count is stripped once, at the
      adapter, and the FR8 guard raises if the arithmetic does not close.
- [ ] **Order is the single source of truth.** `value` token order, `spatial_shapes` row order,
      and `reference_grids` order are mutually consistent by construction and enforced by
      `__post_init__`; there is no separate index array that could drift. Verified by feeding the
      bundle to F1 (which independently re-derives level splits from `spatial_shapes`) and getting
      a valid result (Group 5).
- [ ] **`level_start_index` is not bypassable.** Constructing a bundle with a hand-supplied
      `level_start_index` that disagrees with `spatial_shapes` raises (`__post_init__`), so a
      consumer can never receive an internally inconsistent bundle (mirrors the F1
      `level_start_index` consistency check).
- [ ] **Round-trip fidelity.** The Mask2Former adapter's flatten→concat is the exact inverse of the
      pixel decoder's split→reshape, so `value` is element-for-element the transformer memory
      (Group 3 round-trip test) — the bundle does not reorder or re-scale the backbone's output.
- [ ] **Native channel dim preserved.** `bundle.embed_dim` equals the backbone's `embed_dim`
      (256 / 384); F3 never projects, so no information is added or lost before F6's `img_input_proj`.
