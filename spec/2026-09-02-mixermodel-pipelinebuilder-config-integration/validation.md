# F6 — MixerModel + PipelineBuilder + config integration — Validation

CPU-only unless noted. Mask2Former is built with `imagenet_weights=None` (random ResNet, no
network). DINOv3 byte-identity uses a lightweight stub encoder (`embed_dim=384`,
`.model.patch_size=16`, CLS-prefixed `forward` returning `(B, 1+16², 384)`) so no DINOv3 clone /
weights are needed. Reuse F2/F3 fixtures where practical. Tolerances are stated per case; "byte-
identical" means `torch.equal` (bitwise), everything else uses the stated `atol`.

## Code Correctness

### Group 1 — Config composition
- [ ] `configs/model/image_encoder/dinov3.yaml` and `mask2former.yaml` exist and each carry a `type`
  key (`"dinov3"` / `"mask2former"`) and `embed_dim` (`384` / `256`).
- [ ] Composing `main.yaml` (default `model: mixer_model`) yields `cfg.model.image_encoder.type ==
  "dinov3"` and the same `repo_path`, `name`, `weights`, `freeze`, `regularization`,
  `adapter_hidden_dims`, `image_dim` values as the pre-F6 inline block (field-by-field equality).
- [ ] Overriding `model/image_encoder=mask2former` yields `cfg.model.image_encoder.type ==
  "mask2former"`, `conv_dim == 256`, `transformer_in_features == ["res3","res4","res5"]`,
  `return_stride4 == False`, `freeze_backbone == True`, `freeze_pixel_decoder == False`.
- [ ] `mixer_model.yaml` no longer contains an inline `image_encoder:` mapping (grep returns only the
  `defaults:` reference), and its `defaults` list is `[{image_encoder: dinov3}, _self_]`.

### Group 2 — `build_model` backbone construction
- [ ] With `type: dinov3`, `build_model` returns a `MixerModel` whose `.image_encoder` is a
  `DinoV3Wrapper` (raw, not an adapter), `.image_encoder_type == "dinov3"`, `.n_image_levels == 1`.
- [ ] With `type: mask2former`, `.image_encoder` is a `Mask2FormerFeatureAdapter`,
  `.image_encoder_type == "mask2former"`, `.n_image_levels == 3`; `.image_encoder.embed_dim == 256`.
- [ ] With `return_stride4: True`, `.n_image_levels == 4` and the built decoders report
  `n_levels == 4` (eye + fixation).
- [ ] A pre-F6 config snapshot (an `OmegaConf` dict whose `model.image_encoder` has **no** `type`
  key) drives `build_model` down the DINOv3 branch without error (the `.get('type','dinov3')`
  default).
- [ ] `image_encoder.enabled: False` ⇒ `.image_encoder is None`, no backbone constructed, model runs
  PathModel-equivalently (unchanged from pre-F6).

### Group 3 — DINOv3 byte-identity & checkpoint load
- [ ] Build two `MixerModel`s from the **same** seed and default config (stub DINOv3 encoder): one
  with the pre-F6 code (reference tensor captured before the change, or a pinned expected-output
  fixture) and one post-F6. `encode` outputs `self.src`, `self.image_src` are `torch.equal`.
- [ ] `decode_fixation` output dict tensors (`coord`/`reg`, `dur`, `cls`) are `torch.equal` pre- vs
  post-F6 on identical `tgt` and a fixed seed, for `norm_first=True` (the operative config) — checked
  in `Fixation` and `Combined` phases.
- [ ] Post-F6 DINOv3 model `state_dict()` has **no** `level_embed` key and the same key set as a
  pre-F6 model; a pre-F6 checkpoint loads via `load_state_dict(strict=True)` with zero missing /
  unexpected keys.
- [ ] `load_encoder(pre_f6_ckpt)` on a post-F6 DINOv3 model reports zero "NOT loaded" warnings for
  every `denoise_modules` key.
- [ ] `self.image_spatial_shapes is None` and `self.image_level_start_index is None` after `encode`
  on the DINOv3 path (proves the eye/fixation decoders took the F4 legacy dispatch).

### Group 4 — Mask2Former forward (shapes, PE, level_embed)
- [ ] `encode(src (B,T,3), image (B,3,256,256), src_mask)` on the Mask2Former path sets
  `self.image_src` shape `(B, 1344, 512)` where `1344 = 8² + 16² + 32²`, and `self.src` shape
  `(B, T, 512)`; both float32, no `NaN`/`Inf`.
- [ ] `self.image_spatial_shapes.tolist() == [[8,8],[16,16],[32,32]]` and
  `self.image_level_start_index.tolist() == [0, 64, 320]` (int64).
- [ ] The per-level PE is added exactly once: with `level_embed` and `pos_proj` temporarily zeroed /
  identity-patched, `self.image_src == img_input_proj(bundle.value)` (no double PE); with only
  `pos_proj` output forced to a known constant `c`, `self.image_src - img_input_proj(bundle.value)`
  equals `c` broadcast (up to `level_embed`), `atol=1e-6`.
- [ ] `level_embed` broadcast is correct: the first `64` memory tokens carry `level_embed[0]`, the
  next `256` carry `level_embed[1]`, the last `1024` carry `level_embed[2]` (test by setting
  `level_embed` to `[[1..],[2..],[3..]]`, zeroing `pos_proj`, and checking the additive component per
  token range), `atol=1e-6`.
- [ ] `decode_fixation` returns `coord (B, N+1, 2)`, `dur (B, N+1, 1 or 2)`, `cls (B, N+1, 1)` for
  `head_type="multi_mlp"`; and `reg (B, N+1, 3)`, `cls (B, N+1, 1)` for `head_type="linear"`. No
  `NaN`/`Inf`.
- [ ] A non-square input `(B,3,256,192)` produces a valid bundle (res5 `8×6`), `encode` runs, and
  `image_spatial_shapes` reflects the rectangular levels — proving nothing hard-codes `256²`.
- [ ] `img_input_proj` on the Mask2Former path is an `MLP`/`Linear` `256→512` (or `Identity` when
  `model_dim==256`); its input feature dim equals `image_encoder.embed_dim == 256`.

### Group 5 — Guards / error conditions
- [ ] Constructing a Mask2Former `MixerModel` with `use_rope=True` raises `ValueError` naming
  `use_rope`.
- [ ] `head_type="argmax_regressor"` and `head_type="heatmap"` each raise `ValueError` naming the head
  on the Mask2Former path; both build fine on the DINOv3 path.
- [ ] `input_encoder="image_features_concat"` raises `ValueError` on the Mask2Former path.
- [ ] Passing a 3-level bundle to a decoder mistakenly built with `n_levels=1` surfaces F1's
  "spatial_shape has 3 levels but module has n_levels=1" (guards the `n_image_levels` plumbing).
- [ ] DINOv3 path with `use_rope=True`, `head_type="heatmap"`, and `image_features_concat` all still
  construct and run (no F6 guard fires on DINOv3).

### Group 6 — Gradient flow / freezing (Mask2Former)
- [ ] After a forward+backward on a Combined-phase loss, `level_embed.grad is not None` and nonzero.
- [ ] Every frozen ResNet50 parameter (`image_encoder.backbone.feature_extractor.*`) has
  `requires_grad == False` and `.grad is None`; the pixel decoder params
  (`image_encoder.backbone.pixel_decoder.*`, excluding the frozen extractor) have `requires_grad ==
  True` and receive gradient.
- [ ] `model.train()` leaves the wrapped ResNet50 in `eval()` (BN running stats frozen) — assert
  `image_encoder.backbone.feature_extractor.training == False` after `model.train()` (F2's `train()`
  override propagating through the adapter submodule).
- [ ] `get_parameter_groups(lr)` places the deformable `sampling_offsets` (eye + fixation) in the
  `10×` group and never includes a frozen ResNet param in either group.

### Group 7 — KV / memory cache & scheduled-sampling parity (Mask2Former)
- [ ] With `use_kv_cache=True`, autoregressive `decode_fixation` over `K` steps (memory geometry
  fixed) produces per-step outputs matching a cold, non-cached full-sequence decode of the same
  inputs, `atol=1e-5`.
- [ ] `enable_memory_kv_cache()` / `disable_memory_kv_cache()` / `clear_kv_cache()` run without error
  on the Mask2Former model and the second cross-attn's per-level value cache warms on step 0
  (delegation to F4/F1 intact).
- [ ] A scheduled-sampling forward (`self.scheduled_sampling` set, ratio > 0) completes on the
  Mask2Former path: `encode` is called once, `decode_fixation` reads the stored
  `image_spatial_shapes`/`image_level_start_index` on every decode step (assert they are unchanged
  between steps).

## Data Validity

These run on a real (or fixture) batch and sanity-check the produced features, not just shapes.
Notebook cells acceptable; each states its expected outcome.

- [ ] **PE preserves the shared vocabulary.** On the Mask2Former path with `shared_gaussian`, the
  memory PE `pos_proj(reference_grids)` and the gaze PE `pos_proj(src[:,:,:2])` come from the **same**
  `pos_proj.B` basis (assert `id(self.pos_proj)` is the one used for both; and encoding the same
  coordinate through each yields the same vector, `atol=1e-6`). Confirms the "shared positional
  vocabulary" design intent, not a second PE.
- [ ] **Coarse→fine token layout matches geometry.** `bundle.reference_grids` for level 0 (res5,
  8×8) spans the full `(0,1)²` in 8 steps and level 2 (res3, 32×32) in 32 steps; the first token of
  each level sits near `(1/(2W), 1/(2H))`. Spot-check 3 tokens per level against
  `build_reference_grids`, `atol=1e-6`.
- [ ] **Trained-signal sanity (short run).** A 1–2 epoch Combined-phase run on a small COCO subset
  with `model/image_encoder=mask2former` produces a decreasing training loss and a finite
  `reg_error_val` at the first validation (not `NaN`, not stuck at the PAD_TOKEN_ID collapse value)
  — a smoke check that the wired path actually learns.
- [ ] **DINOv3 parity end-to-end.** The default DINOv3 config trains one step with loss and
  `reg_error_val` identical (`atol=1e-6`) to a pre-F6 checkout on the same seed + batch — the
  practical restatement of Group 3.

## Data Architecture Integrity

- [ ] **Single image-feature interface.** On the Mask2Former path, the only object crossing the
  backbone→model boundary is a `MultiScaleFeatures` bundle (assert `isinstance`), and `MixerModel`
  reads only its public fields/props (`value`, `spatial_shapes`, `level_start_index`,
  `reference_grids`, `level_sizes()`) — no reach into `image_encoder.backbone` internals from
  `encode`/`decode_fixation` (grep the two methods for `.backbone` / `.pixel_decoder` /
  `.feature_extractor` → none).
- [ ] **No DINOv3 attribute access on the Mask2Former path.** Grep `encode`/`decode_fixation`/`__init__`
  for `.model.patch_size`, `.model.rope_embed`, `forward_features(`, and `mem[:, 1:` / `[:,1:,:]`;
  every occurrence is reachable **only** when `image_encoder_type == 'dinov3'` (static read + a
  runtime test that monkeypatches those attributes to raise and confirms the Mask2Former forward
  never triggers them).
- [ ] **`spatial_shapes` / `level_start_index` consistency is enforced end-to-end.** The tensors
  stored on the model are exactly the bundle's (identity, not a rebuilt copy), so F3's
  `__post_init__` invariants (Σ Hₗ·Wₗ == value length; `level_start_index` == cumsum) hold for what
  the decoders receive — assert equality with `build_level_start_index(spatial_shapes)`.
- [ ] **DINOv3 state_dict is a strict superset-free match.** The post-F6 DINOv3 model introduces zero
  new persistent tensors vs. pre-F6 (`level_embed` absent, `image_spatial_shapes` /
  `image_level_start_index` are plain attributes not `nn.Parameter`/buffers) — assert they are not in
  `state_dict()` and do not appear in `named_parameters()` / `named_buffers()`.
- [ ] **Config → build determinism.** Re-composing the config and re-running `build_model` twice with
  the same seed yields models with identical parameter shapes and key sets for both backbones (no
  hidden nondeterminism in level count / decoder construction).
```
