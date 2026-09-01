# Validation — F2: Vendored Mask2Former Backbone (detectron2-free)

Tolerances: exact for shapes/dtypes/param counts; `atol=1e-5` for float cross-checks unless noted.
All tests run on CPU with fixed seeds (`torch.manual_seed(0)`). Reference input unless stated:
`x = torch.randn(2, 3, 256, 256)`.

## Code Correctness

### Group 1 — ResNet50 feature extraction
- [ ] `Mask2FormerBackbone().feature_extractor(x)` returns a dict with keys exactly
  `{"res2","res3","res4","res5"}`.
- [ ] Shapes at 256²: `res2 (2,256,64,64)`, `res3 (2,512,32,32)`, `res4 (2,1024,16,16)`,
  `res5 (2,2048,8,8)`; dtype `float32`.
- [ ] At `img_size=128`: `res2 (2,256,32,32)`, `res5 (2,2048,4,4)` — proves shapes are dynamic, not
  hardcoded.
- [ ] ImageNet weights loaded: with default `IMAGENET1K_V2`, the first conv weight is **not** a
  freshly-seeded random tensor (compare to `resnet50(weights=None)` conv1 under the same seed — they
  must differ). If the environment is offline, this test is `skip`ped with a reason, not failed.

### Group 2 — Pixel decoder forward (3 levels)
- [ ] `Mask2FormerBackbone(return_stride4=False)(x)` returns a `list` of length 3.
- [ ] Map shapes (coarse→fine): `[(2,256,8,8), (2,256,16,16), (2,256,32,32)]`; dtype `float32`;
  all finite (`torch.isfinite(...).all()`).
- [ ] The transformer's `self_attn` is an F1 `DeformableAttention` with `n_levels == 3`.
- [ ] F1 param shapes inside every encoder layer match the Mask2Former reference for
  `(d=256,L=3,H=8,P=4)`: `sampling_offsets.weight (192,256)`, `attention_weights.weight (96,256)`,
  `value_proj.weight (256,256)`, `output_proj.weight (256,256)`. (Asserts the corrected reference,
  **not** validation.md's F1-spec `768/384` typo.)
- [ ] `MSDeformAttnTransformerEncoderOnly` runs exactly `transformer_enc_layers` (6) layers.
- [ ] `spatial_shapes` computed internally == `tensor([[8,8],[16,16],[32,32]])` and
  `level_start_index == tensor([0, 64, 320])`.
- [ ] `level_embed` is a `Parameter` of shape `(3, 256)`.

### Group 3 — Optional stride-4 branch
- [ ] `return_stride4=False`: the backbone has **no** `lateral_res2`/`output_res2`/`mask_features`
  submodules (`hasattr` is `False`); `forward` returns a bare list of 3.
- [ ] `return_stride4=True`: `forward(x)` returns `(maps, mask_features)` where the appended finest
  level (or `res2_fpn`) has shape `(2,256,64,64)` and `mask_features` has shape
  `(2, mask_dim, 64, 64)` = `(2,256,64,64)`; all finite. `num_levels == 4`.
- [ ] `return_stride4=True` builds strictly more parameters than `return_stride4=False`
  (the FPN + mask_features convs); the 3-level maps are otherwise shape-identical.

### Group 4 — Freezing semantics
- [ ] Default (`freeze_backbone=True`, `freeze_pixel_decoder=False`): every
  `feature_extractor` parameter has `requires_grad == False`; every `pixel_decoder` parameter has
  `requires_grad == True`.
- [ ] After `backbone.train()`, `feature_extractor.training == False` (BN stats frozen) while
  `pixel_decoder.training == True`.
- [ ] Backward test: `backbone(x)[-1].sum().backward()` yields `None`/zero grad on a ResNet conv
  weight and a **non-zero** grad on a pixel-decoder `input_proj` conv weight.
- [ ] BN running-mean invariance: two `forward` passes in `.train()` mode over different inputs
  leave `feature_extractor`'s first BN `running_mean` unchanged (frozen).
- [ ] `freeze_backbone=False`: ResNet params become `requires_grad == True`.

### Group 5 — Purity / no forbidden dependencies
- [ ] Importing `src.model.ms_deform_backbone` does not import `detectron2`, `fvcore`, or any
  `MSDeformAttn` CUDA extension (assert those module names are absent from `sys.modules` after a
  fresh import in a subprocess, or that `import detectron2` is never triggered).
- [ ] Every normalization layer in `pixel_decoder` is `nn.GroupNorm` or `nn.LayerNorm`; every conv
  is `nn.Conv2d` (no detectron2 `Conv2d`).
- [ ] With `InferenceRecorder` enabled on the backbone, a forward pass records F1's tensors with a
  level axis of size 3 (`sampling_offsets (…,3,4,2)`, `attention_weights (…,3,4)`) — proves
  recorder-compatibility carries through.

### Group 6 — Input contract & determinism
- [ ] Batch independence: `forward` on `x[:1]` equals the first row of `forward(x)` within
  `atol=1e-5` (in `eval()` mode, dropout off).
- [ ] Determinism: with the backbone in `eval()`, two forwards on identical `x` are bit-identical.
- [ ] Non-square input `(2,3,256,192)` runs without error and yields `res5 (2,2048,8,6)`,
  decoder maps with matching `Hₗ,Wₗ`.
- [ ] The module never returns a CLS token: output maps are 4-D `(B,C,H,W)`, never a
  `(B, HW+1, C)` sequence.

## Data Validity

Sanity checks on the actual features produced (notebook cells acceptable; each states its expected
outcome).

- [ ] **Feature magnitude sane.** For a real stimulus image (loaded and pipeline-normalized), the
  enhanced maps have finite, non-degenerate statistics: per-map std `> 1e-3` (not collapsed to a
  constant) and no `NaN`/`Inf`. Expected: all three levels pass.
- [ ] **Positional encoding matches source.** `PositionEmbeddingSine(128, normalize=True)(x)` output
  is element-wise equal (`atol=1e-6`) to the Mask2Former original run on the same input — confirms
  the copy is faithful.
- [ ] **ImageNet features are meaningful, not random.** Cosine similarity between the pooled `res5`
  of two augmentations (e.g. horizontal flip) of the same image is higher than between two different
  images — a smoke check that the ImageNet backbone carries semantic signal (frozen decoder init is
  random, so run this on the **backbone** `res5`, pre-decoder). Expected: same-image similarity
  meaningfully higher.
- [ ] **Cross-scale fusion is active.** Zeroing one input level's `input_proj` output measurably
  changes all three returned maps (the transformer mixes scales), whereas for a hypothetical
  no-attention baseline it would not. Expected: non-trivial change on every level.
- [ ] **Stride-4 map resolution.** When `return_stride4=True`, the 64² map's spatial resolution is
  exactly 4× the res5 map per axis (8→64), confirming the FPN upsampling path is wired to res2.

## Architectural Integrity

Adapted from the template's "Data Architecture Integrity" — F2 has no HDF5 cache or `exp_key`, so
these check the *contract invariants* the downstream features (F3/F4/F6) rely on.

- [ ] **F1-layout compatibility.** The instantiated F1 `sampling_offsets`/`attention_weights`
  shapes are `(192,256)`/`(96,256)` — the corrected Mask2Former layout — so a future checkpoint (if
  the COCO path is ever revived) would load by shape. No `768/384` shapes appear anywhere.
- [ ] **CLS-free boundary.** No code path in `ms_deform_backbone.py` produces or slices a CLS token;
  the DINOv3 `mem[:,1:,:]` convention is absent. (Grep-level assertion + shape test.)
- [ ] **`conv_dim` fixed at 256.** The channel dim of every returned map is 256 regardless of other
  args; changing `conv_dim` is possible in the constructor but the default and the ImageNet-adapter
  path are 256, matching the `img_input_proj` 256→`model_dim` contract F6 expects.
- [ ] **Level ordering is monotonic coarse→fine.** Returned maps satisfy strictly increasing
  `Hₗ` (8 < 16 < 32 [< 64]); F3 relies on this to build `spatial_shapes`/`level_start_index`.
- [ ] **No `valid_ratios`/padding assumption leaks.** The encoder builds reference points from
  `spatial_shapes` alone; there is no mask argument on any public method — so F3/F4 need not supply
  one.
- [ ] **Additivity / retro-compat.** `DinoV3Wrapper` and the existing single-scale path are
  unchanged (F2 modifies no existing file); a repo-wide test collection still passes and DINOv3
  imports independently of `ms_deform_backbone`.
- [ ] **Deviation is recorded.** The spec's Goal/Context explicitly documents that COCO weights and
  the checksum loader were dropped in favor of ImageNet R50 + fresh decoder, so a later reader does
  not expect a weight loader that isn't there.
