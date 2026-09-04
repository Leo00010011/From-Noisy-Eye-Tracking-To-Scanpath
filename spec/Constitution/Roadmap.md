# Roadmap

## Done

- ✓ **Use pretrained (frozen) image features** — precompute the Mask2Former image features once
  with a **pretrained-and-frozen** backbone and cache them to disk, then feed the cache to
  `MixerModel` in place of the live backbone forward (frozen-features recipe: the online m2f path
  re-ran ResNet50 + a 6-layer pixel decoder every step and overfit). New loader
  `src/model/m2f_pretrained.py` populates the F2 `Mask2FormerBackbone` from the two detectron2-format
  checkpoints in `pretrained_models/`: `remap_detectron2_resnet50` (COCO-tuned R50 `stem`/`stages.resN`/
  `.norm.`/`shortcut` → torchvision `conv1`/`bn1`/`layerN`/`downsample`, values unchanged),
  `remap_pixel_decoder` (prefix `pixel_decoder.` — the pkl keys are already 1:1 with the vendored
  decoder), and `load_pretrained_mask2former` (→ `LoadReport`; **raises** if any ResNet50 or core
  pixel-decoder *parameter* is unloaded — guarding on `named_parameters()` so BN `num_batches_tracked`
  buffers, absent from detectron2 `FrozenBatchNorm2d`, don't false-trigger; tolerates `fc.*` and the
  optional stride-4 FPN keys). Precompute driver `scripts/build_image_feature_cache.py` builds the
  backbone (`return_stride4=True`, `transformer_enc_layers=6`, everything frozen), loads both pkls, runs
  once over every **unique** stimulus (first-seen order, same filter as training), and writes
  `data/Coco FreeView/image_features_{img_size}.h5` (group `/features`, mode `"w"`): `ms_value (U,S,256)`
  (flattened `[res5,res4,res3]` coarse→fine — the exact online deformable memory) + `mask_features
  (U,256,64,64)` (stride-4, heatmap-reserved, unconsumed) + `image_path (U,)` (the keying invariant),
  chunked per-image. Model side: zero-parameter `PrecomputedFeatureAdapter` (`src/model/ms_features.py`)
  reconstructs a `MultiScaleFeatures` bundle from a cached `ms_value` batch + fixed geometry (non-persistent
  buffers), installed as `MixerModel.image_encoder` — **`MixerModel` and `ms_deform_backbone.py` need
  ZERO edits** (the `self.image_encoder(image_src)` seam already returns a bundle). Data side:
  `ImageFeatureCache` + `PrecomputedFeatureDataset` (`src/data/image_feature_cache.py`) drop in for
  `DeduplicatedMemoryDataset` under `CoupledDataloader`, verifying `image_path[u]` against the rebuilt
  first-seen path **unconditionally** (order invariant not bypassable). New config group
  `configs/model/image_encoder/mask2former_precomputed.yaml` (`precomputed: True`, no backbone flags);
  `PipelineBuilder` gains three additive branches (`build_model` stub install, `load_dataset`/
  `build_dataloader` feature-dataset swap) + a `_validate_feature_cache` FR12 check (precomputed on both
  sides or neither). Additive/dual-path: online DINOv3 and online Mask2Former stay byte-identical and
  selectable. 31-test suite `tests/test_image_feature_cache.py` (all pass; Groups 1–2 run against the
  real pkls). Data-validity confirmed on the real data: **U=4317** unique images, S=1344,
  `spatial_shapes=[[8,8],[16,16],[32,32]]`, features finite/non-constant, pretrained≠random. **Note:**
  the full cache is ~24 GB with `mask_features` (float32; `ms_value` alone ≈5.9 GB) — validation.md's
  "~9 GB" prose understates it; its own byte formula gives ~24 GB. The full CPU precompute write is left
  to a GPU run. Spec: `spec/2026-09-04-use-pretrained-features/`.
- ✓ **Image-reliance diagnostic suite** — additive, diagnosis-only measurement of *"does the
  Mask2Former-backbone `MixerModel` actually use the image?"* (the `train_ms.sh` run tied DINOv3
  on accuracy, so the dual-path design lets us test whether the image path is inert). New module
  `src/eval/image_reliance.py` holds pure, unit-tested primitives — `sampling_in_range_fraction`
  (fraction of deformable sampling locations inside `[0,1]`), `residual_norms` /
  `extract_residuals` / `extract_sampling_locations` (harvest the already-recorded
  `first_cross_res`/`second_cross_res`/`cross_attention_res`/`sampling_locations` from the
  `norm_first`+deformable hooks), `shuffle_images_in_batch` (cyclic-roll derangement),
  `per_sample_reg_error` (mirrors `eval_reg` per row) — plus two pass drivers: **Pass A**
  (`run_recording_pass`, recorder ON) captures cross-attention residual norms + in-range fractions
  for the eye and fixation decoders via the predict-then-clean-forward pattern (model's own
  predictions as `tgt`); **Pass B** (`run_perturbation_pass`, recorder OFF) recomputes per-sample
  regression error with images shuffled within each batch (gaze untouched). Writers
  `write_reliance_store` (per-sample HDF5, group `/reliance`, mode `"w"`, Pass B aligned to Pass A
  **by `sample_idx`**) and `write_summary` (aggregated JSON + printed table with the three headline
  scalars: image/gaze residual ratio, perturbation delta, per-level in-range fraction, each with an
  interpretation string). Thin driver `src/notebooks/save_image_reliance.py` loads a checkpoint via
  `model_io.load_pipeline`/`load_test_data`/`load_model` on the CocoFreeView **test** split. **No
  model code touched** — `blocks.py`/`mixer_model.py`/`inference_recorder.py` unchanged; no training
  HDF5 layout change. 30-test suite `tests/test_image_reliance.py` (CPU-only, synthetic + stub
  recorder + one tiny real `MixerModel` for the module-name assertion). Spec:
  `spec/2026-09-03-image-reliance-diagnostic-suite/`.
- ✓ **Optuna + W&B hyperparameter search** — standalone driver `scripts/hp_search.py` tunes the `MixerModel` on a reduced-budget, from-scratch **Combined** phase (`configs/exp/hp_search.yaml`, `E=40` with schedule invariants: `scheduler.warmup+stable+decay == E`, `scheduled_sampling.warmup+active <= E`). Each Optuna trial Hydra-`compose`s the config with 15 sampled hyperparameters (`weight_decay`, `n_encoder`/`n_decoder`/`n_eye_decoder`, the 9 mixer dropouts >0, `cls_weight`, `dur_weight` — bounds in `configs/hp_search.yaml`), trains via `train(builder, trial=...)`, and reports the best `reg_error_val` (`MetricsStorage.best_metric_value`) as the minimise objective. `TPESampler(seed)` + SQLite storage (`load_if_exists=True`) make the study reproducible and resumable; `MedianPruner` prunes weak trials via `trial.report()` at each validation; `study.optimize(..., catch=(Exception,))` keeps the study alive past a FAILED trial (e.g. CUDA OOM). One W&B run per trial (grouped by study) logs per-epoch `train/*` and per-validation `val/*` curves; the driver owns `wandb.init`/`finish`, `train()` only logs to the active run (and `finish()`es on the prune path). Per-trial artifacts (`metrics.json`, `model.pth`, `split.pth`, `config.yaml`) under `outputs/hp_search/<study>/trial_<n>/`; study-level `trials.csv` + `best_params.yaml`. `train()` is backward-compatible (`trial=None`, `training.wandb` absent → inert via `.get`); `python train.py` behaviour is byte-for-byte unchanged. 43-test suite `tests/test_hp_search.py`; `optuna`/`wandb` added to `requirements.txt`. Spec: `spec/2026-07-30-optuna-wandb-hyperparameter-search/`
- ✓ **EVE real-noise scanpath inference** — `EyeNetGazeCache` projects EyeNet's per-frame normalized gaze predictions (`predictions.csv`) into a screen-space, `exp_key`-keyed HDF5 cache via `EveBundle.project_normalized_gaze` (left/right eye intercepts averaged in pixel space → 89.6 px / 2.32 DVA median error vs ray-derived ground truth). `EveRealNoiseDataset` / `EveRealNoiseImgDataset` feed a trained `MixerModel` autoregressively over real degraded gaze; `RealNoiseInferenceStore` persists predicted scanpaths keyed by `exp_key`. Inference-only — no `clean_x`, no accuracy metrics, `PipelineBuilder` untouched. Filtering is on `eyenet_split` (the operative partition; EVE's split is metadata only). New files under `src/data/eve_real_noise*.py`, `configs/data/eve_real.yaml`, `scripts/build_eyenet_gaze_cache.py`, `src/notebooks/save_predictions_eve_real.py`; 52-test suite `tests/test_eve_real_noise.py`. All validation.md reference numbers reproduced on the production bundle. Spec: `spec/2026-07-27-eve-real-noise-scanpath-inference/`
- **Spec-driven workflow bootstrapped** — Mission, TechStack, and Roadmap documents in place under `spec/Constitution/`
- **Notebook infrastructure** — `review_utils`, `generate_review_notebooks`, `eval_batch` for offline analysis and scanpath visualisation
- **Inference recorder** — `InferenceRecorder` hooks into model submodules to capture intermediate tensors for debugging; gated by `training.inference_recorder.enabled`
- **KV cache for autoregressive inference** — `use_kv_cache` flag in `DoubleInputDecoder`; disabled during training, usable at inference time
- **Curriculum noise** — `AddCurriculumNoise` with step-based schedule; wired into `PipelineBuilder` as `curriculum_noise`
- **RoPE positional embeddings** — `RopePositionEmbedding` for cross-attention between gaze trajectory and image patches; active when `use_rope=True`
- **Deformable attention** — `DeformableDecoder` and `DeformableDoubleInputDecoder` in both the eye decoder and fixation decoder; default in `mixer_model.yaml`
- **Shared Gaussian Fourier encoders** — `shared_gaussian` and `shared_gaussian_base` input encoder modes; shared random basis matrix `B` creates a common positional vocabulary across gaze and image modalities
- **Multiple head types** — `linear`, `mlp`, `multi_mlp`, `argmax_regressor`, `heatmap`, `start_head`; all wired in `PipelineBuilder.build_model()`
- **MixerModel — full image-conditioned architecture** — encoder + feature enhancer / eye decoder + double-input decoder + denoise head; phase-aware `requires_grad` toggling via `denoise_modules` / `fixation_modules` lists
- **Phase-based training** — Denoise / Fixation / Combined phases with per-phase loss, decisive metric, and epoch budget; `auto_best_denoise` checkpoint selection from `outputs/`
- **Scheduled sampling** — `ScheduledSampling` replaces teacher-forced inputs with model predictions during Fixation/Combined phases; probability schedule configurable
- **DINOv3 image encoder** — `DinoV3Wrapper` loads frozen ViT-S/16 from a local clone; `MLP` adapter projects patch tokens to `model_dim`
- **Hydra configuration system** — composable config groups (`model/`, `scheduler/`, `loss/`, `head_type/`, `data/split_strategy/`, `exp/`); per-run output directory with full config snapshot at `.hydra/config.yaml`
- **Loss functions** — `EntireRegLossFunction`, `SeparatedRegLossFunction`, `CombinedLossFunction`, `DenoiseRegLoss`, `PenaltyReducedFocalLoss`, `EndBinaryCrossEntropy`, `EndSoftMax`
- **LR schedulers** — `one_cycle`, `multistep_lr`, `warmup_stable_decay`
- **Split strategies** — `random`, `stimuly_disjoint`, `disjoint` (subject + stimuli disjoint)
- **PathModel baseline** — gaze-only encoder–decoder transformer; `linear` and `shared_gaussian` input encoders, three head types
- **Data transform pipeline** — `ExtractRandomPeriod`, `Normalize`, `LogNormalizeDuration`, `QuantileNormalizeDuration`, `SaveCleanX`, `StandarizeTime`, `AddGaussianNoiseToFixations`, `AddHeatmaps`; composable via `config.data.transforms.transform_list`
- **FreeViewInMemory + CoupledDataloader** — in-memory HDF5 loading with per-item transforms; `DeduplicatedMemoryDataset` deduplicates repeated stimulus images; `CoupledDataloader` synchronises gaze and image iteration
- **Physics-informed noise simulation** — `AddRandomCenterCorrelatedRadialNoise` (correlated radial with drifting center), `AddIsotropicGaussianNoise`, `DiscretizationNoise`; all configurable via Hydra
- **HDF5 preprocessing pipeline** — raw CocoFreeView → downsampled gaze + fixation arrays stored in `dataset.hdf5`; `CocoFreeView` parser with disjoint split helpers
- **Training evaluation metrics** — `eval_reg` (Euclidean coord error, duration MAE), `eval_denoise` (MSE on denoised coords), end-of-sequence accuracy / precision / recall

---

## In Progress

- **Spec-driven development workflow** — Constitution documents written; first sprint-level feature spec (EVE real-noise scanpath inference) delivered end-to-end under `spec/2026-07-27-eve-real-noise-scanpath-inference/`
- **Multi-scale image backbone migration (Mask2Former)** — replacing the single-scale frozen DINOv3 image encoder with a vendored, detectron2-free **ResNet50 + MSDeformAttn pixel decoder** (from the Mask2Former clone at `../mask2former/`), and making the eye-decoder and fixation-decoder deformable cross-attentions **multi-scale**. Factored into six independently testable features **F1–F6** (see the dedicated Backlog subsection below and the TechStack "Multi-scale Image Backbone Migration" section for full contracts). **F1, F2, F3, F4, and F6 are ✓ DONE — the migration is complete: the Mask2Former backbone is reachable end-to-end via `model/image_encoder=mask2former`.** F1 (`DeformableAttention`) is now N-level capable and byte-identical at `n_levels=1`; its as-built API and the multi-scale shape reference live in TechStack §"Deformable param layout (F1 as-built)". F2 (`Mask2FormerBackbone` in `src/model/ms_deform_backbone.py`) is the vendored, detectron2-free **torchvision ResNet50 (ImageNet, frozen) + freshly-initialized, trainable pixel decoder** consuming F1 at `n_levels=3`; it emits 3 CLS-free multi-scale maps `[B,256,Hₗ,Wₗ]`. F3 (`src/model/ms_features.py`) is the `MultiScaleFeatures` bundle + two backbone adapters that decouple `MixerModel` from backbone specifics. F4 (multiscale-capable eye & fixation decoders) consumes F1 at *N* levels. F6 (`MixerModel` + `PipelineBuilder` + the `configs/model/image_encoder/` group) constructs the F2 backbone from config, wraps it in the F3 adapter, feeds the bundle through the F4 decoders with per-level PE (shared `shared_gaussian` basis) + a `level_embed`, and guards every DINOv3-only access behind an explicit `image_encoder_type` — **DINOv3 stays byte-identical and old checkpoints load** (dual path, not unified). The backbone is **ImageNet-pretrained ResNet50 frozen + trainable pixel decoder** — no external segmentation checkpoint is loaded. **DINOv3 remains selectable** — the new backbone is additive, gated by a config group, so the existing single-scale path and its checkpoints keep working throughout. Remaining migration follow-ups are non-blocking (stride-4 4th level, path unification — see "Open items").

---

## Backlog

Priority order within each group.

### Blockers for Publication

- **Publication-level evaluation metrics** — DTW (Dynamic Time Warping), multi-match (Jarodzka et al.), and ScanMatch are listed in the Mission success criteria but not implemented. These require operating in pixel space after `Normalize.inverse()`. Without them the results section cannot be written.
- **Seeded reproducibility enforcement** — `torch.manual_seed`, `numpy.random.seed`, `random.seed`, and `torch.backends.cudnn.deterministic` are not set in the training entry point. The Mission claims runs are re-runnable from a config + seed; this is not currently true.
- **Formal baseline comparison infrastructure** — no script runs PathModel and MixerModel on the same split with the same seed and produces a comparison table. Required for the ablation section.
- **Comparison against SOTA scanpath models** — Mission scopes comparisons against Gazeformer, ScanDy, IORE (or equivalent); no integration or evaluation harness exists yet.

### Architecture Migration: Multi-scale Image Backbone (Mask2Former)

Six features, in dependency order. **Locked decisions** (from the planning session): vendor a
detectron2-free minimal port; extend the existing `grid_sample` deformable op (retro-compatible,
with a sibling-class escape hatch); backbone = **torchvision ResNet50 (ImageNet) frozen** +
**freshly-initialized, trainable pixel decoder** (no external checkpoint); keep the full pixel
decoder (its internal 6-layer MSDeform transformer encoder) with **3 feature levels**; **no CUDA
custom op** (pure-PyTorch `grid_sample`, InferenceRecorder-compatible). Each feature is a separate
implementation session. Full interface contracts live in TechStack.

Dependency graph: `F1 → {F2, F4}`; `F3 → F6`; `{F2, F3, F4} → F6`. Suggested order **F1 → F2 →
F3 → F4 → F6** (F3 parallelizable with F2).

- ✓ **F1 — Multi-scale deformable attention primitive** — DONE. `DeformableAttention`
  (`src/model/blocks.py`) generalized in place (class name preserved) to *N* levels via a new
  `n_levels: int = 1` constructor arg. `sampling_offsets` → `Linear(d, n_heads·n_levels·n_points·2)`,
  `attention_weights` → `Linear(d, n_heads·n_levels·n_points)`, softmax **jointly** over the flattened
  `n_levels·n_points` axis — the same `MSDeformAttn` param layout and star-pattern init Mask2Former
  uses (verified against `../mask2former/.../ops/modules/ms_deform_attn.py`), which is the op F2's
  pixel decoder instantiates fresh at `n_levels=3`. `forward(query, reference_points, value, spatial_shape,
  level_start_index=None)` is polymorphic: `spatial_shape` accepts a legacy `(H,W)` tuple (1 level) or
  a `(n_levels,2)` LongTensor; `reference_points` accepts `(B,Nq,2)` (broadcast) or `(B,Nq,n_levels,2)`;
  `level_start_index` is derived via `cumsum` when absent and consistency-checked when supplied. At
  `n_levels=1` the param layout, init, and forward output are **byte-identical** to the pre-F1 class
  (`torch.equal`), so existing single-scale checkpoints (HP-search runs) load with zero
  missing/unexpected keys. Pure `grid_sample` (per-level loop, no CUDA op); star-pattern init,
  `geometric_sigma` jitter, KV-cache (now a per-level value list), and recorder hooks preserved
  (recorded tensors gain a level axis: `sampling_offsets/locations (B,Nq,H,L,P,2)`,
  `attention_weights (B,Nq,H,L,P)`, `reference_points (B,Nq,L,2)`). `DeformableDecoder` /
  `DeformableDoubleInputDecoder` are untouched (F4's job) and their legacy calls stay byte-identical.
  Escape hatch (sibling `MultiScaleDeformableAttention`) not needed — the single class carries both
  paths. 28-test suite `tests/test_ms_deformable_attention.py`. Spec:
  `spec/2026-09-01-multi-scale-deformable-attention-primitive/`. **Note:** that spec's validation.md
  layout reference (`768`/`384`) is a typo; the correct shapes for `(d=256,L=3,H=8,P=4)` are
  `sampling_offsets (192,256)`, `attention_weights (96,256)` — what F2's pixel-decoder attention
  builds.
- ✓ **F2 — Vendored Mask2Former backbone (detectron2-free)** — DONE. `Mask2FormerBackbone`
  (`src/model/ms_deform_backbone.py`): torchvision ResNet50 → `{res2..res5}` (via
  `create_feature_extractor`, `{layer1..layer4}`→`{res2..res5}`) feeding a ported
  `MSDeformAttnPixelDecoder` (detectron2 pieces inlined: `PositionEmbeddingSine` copied verbatim,
  `nn.Conv2d`+`nn.GroupNorm(32,·)`, `xavier_uniform_` init, no registry/`@configurable`/`ShapeSpec`;
  `input_shape` is a plain `Dict[str,(channels,stride)]`). Internal 6-layer transformer encoder's
  `self_attn` = F1 `DeformableAttention` at `n_levels=3` (`sampling_offsets (192,256)`,
  `attention_weights (96,256)`), softmax joint over `n_levels·n_points`; the `masks`/`valid_ratios`/
  `padding_mask` machinery is dropped (fixed image size ⇒ no padding), reference points are plain
  `linspace(0.5,N-0.5,N)/N` grids. `_reset_parameters` skips `self_attn.` params so F1's star-pattern
  init survives the generic xavier flood. `forward(x)` returns **3 enhanced multi-scale maps**
  `[B,256,Hₗ,Wₗ]` in **coarse→fine** order `[res5(8²),res4(16²),res3(32²)]` at `img_size=256`,
  **no CLS**; shapes are dynamic (128²→res5 4², non-square 256×192→res5 8×6). ResNet50 uses
  **torchvision ImageNet weights** (`IMAGENET1K_V2`→`V1`→random-init, warn-and-continue on fetch
  failure, never raises). Pixel decoder is **freshly initialized**. Freezing is **granular**:
  ResNet50 frozen + kept in `eval()` (BN running stats frozen) via a `train()` override, pixel
  decoder **trainable** (independent `freeze_backbone`/`freeze_pixel_decoder` flags). Optional
  stride-4 (res2, 64²) FPN + `mask_features` branch (`return_stride4`, default off) is **built only**
  when requested → `forward` then returns `([res5,res4,res3,res2_fpn], mask_features)`, `num_levels=4`.
  Input assumed pipeline-normalized (no internal mean/std, mirroring `DinoV3Wrapper`). Pure PyTorch —
  no detectron2/fvcore/CUDA `MSDeformAttn` in the import graph (subprocess-asserted); F1 recorder
  hooks carry through with a level axis of 3. **Modifies no existing file** (additive; DINOv3 stays
  selectable). 31-test suite `tests/test_ms_deform_backbone.py`. Spec:
  `spec/2026-09-01-vendored-mask2former-backbone/`. **Deviation from the original F2 plan (locked
  decision 3):** the Mask2Former R50 **COCO-panoptic** checkpoint + Caffe2 key-remap + checksum
  loader were dropped in favor of ImageNet R50 + fresh decoder (user-directed, planning session), so
  there is no external checkpoint, no weight translation, and no checksum. **Next up: F3** (the
  `MultiScaleFeatures` bundle; F4 parallelizable).
- ✓ **F3 — Multi-scale feature contract / backbone adapter** — DONE. New file
  `src/model/ms_features.py`: the `MultiScaleFeatures` dataclass
  (`value:[B,ΣHₗWₗ,D]` float, `spatial_shapes:(L,2)` int64, `level_start_index:(L,)` int64,
  `reference_grids:(S,2)` float per-token `(x,y)` centers), two geometry helpers
  (`build_level_start_index`, `build_reference_grids` — the latter byte-identical to
  `MSDeformAttnTransformerEncoder.get_reference_points` up to the level-repeat/batch axis), and two
  producing `nn.Module` adapters. `Mask2FormerFeatureAdapter` wraps the F2 backbone and repackages
  its coarse→fine `[B,256,Hₗ,Wₗ]` maps via `m.flatten(2).transpose(1,2)` + concat (round-trip
  identity with the pixel decoder's split→reshape; tolerates the 4-level `return_stride4` output and
  **discards** `mask_features`). `DinoV3FeatureAdapter` wraps `DinoV3Wrapper`, strips the CLS prefix
  **once** (`tokens[:, num_prefix_tokens:, :]`, `num_prefix_tokens=1` default), reads
  `backbone.model.patch_size` **once at init** (the sole DINOv3-internal access after F4/F6), and
  emits a 1-level bundle with a phantom-prefix guard. `__post_init__` raises `ValueError` on every
  shape/dtype/consistency violation (no silent coercion); `.to(device,dtype)` casts only the float
  tensors, index tensors stay int64. Adapters add **zero** params (backbone held as a submodule,
  keys prefixed `backbone.`; geometry computed per forward, not buffered → dynamic sizes work);
  `.eval()/.train()`, `state_dict`, and F1 recorder hooks propagate to the backbone. Bundle is
  directly F1-consumable (`DeformableAttention(embed_dim=D, n_levels=bundle.num_levels)`).
  **Modifies no existing file** (additive; DINOv3 stays selectable, F1's byte-identity intact).
  40-test suite `tests/test_ms_features.py` (CPU-only, `imagenet_weights=None`, `FakeDino` stub — no
  network). Spec: `spec/2026-09-02-multi-scale-feature-contract-backbone-adapter/`. F4 (decoders)
  now DONE; **next up: F6** (MixerModel + PipelineBuilder + config).
- ✓ **F4 — Multiscale-capable eye & fixation decoders** — DONE. `DeformableDecoder` and
  `DeformableDoubleInputDecoder` (`src/model/blocks.py`) each gain an `n_levels: int = 1` constructor
  arg (after `num_points`) that is single-sourced into the inner F1 op (`cross_attn` /
  `second_cross_attn`); the first cross-attention (`MultiHeadedAttention` over the gaze memory) is
  untouched. Both `forward`s are **polymorphic** on new `spatial_shapes=None`/`level_start_index=None`
  kwargs: `spatial_shapes is None` selects the **legacy single-scale path** (CLS-prefixed memory
  sliced `mem[:,1:,:]`, fixed `self.spatial_shape=(16,16)` tuple, requires `n_levels==1`, raises
  otherwise); a supplied `spatial_shapes (L,2)` selects the **multi-scale path** (memory is the
  already-CLS-free F3 `value (B,ΣHₗWₗ,D)`, passed through unsliced, `spatial_shapes`/
  `level_start_index` forwarded to F1). CLS is stripped **once, on exactly one branch**;
  `reference_points (B,Nq,2)` pass straight through and F1 broadcasts across levels. **Byte-identical
  at `n_levels=1`**: `state_dict` keys/shapes unchanged, pre-F4 checkpoints load with zero
  missing/unexpected keys, and the legacy forward is `torch.equal` to pre-F4 (both norm modes) — so
  the unmodified `MixerModel` keeps training identically until F6. Fixes the flagged latent bug in
  `DeformableDoubleInputDecoder`'s non-`norm_first` branch (previously forwarded
  `attn_mask`/`src_rope`/`mem2_rope` kwargs `__cross_attention2` rejected and never stripped CLS):
  `__cross_attention2`'s signature drops those kwargs and both branches now call it identically, so
  they differ only in norm placement. Error paths surface F1's `ValueError`s unchanged
  (level-count/value-length/ref-dim/`level_start_index` mismatches) plus the decoder's own "legacy
  single-scale path requires n_levels==1". KV/memory cache and InferenceRecorder hooks (level axis at
  `n_levels>1`, squeezable singleton at 1) carry through. **No import cycle** — decoders take unpacked
  tensors, so `blocks.py` never imports `ms_features`; F6 unpacks the bundle at the call site.
  **Modifies no other file** (additive; DINOv3 single-scale path unchanged). 34-test suite
  `tests/test_ms_decoders.py` (+ two F1 decoder-integration tests updated for the new
  `level_start_index` kwarg). Spec: `spec/2026-09-02-multiscale-capable-eye-and-fixation-decoders/`.
  **Note:** validation.md Group 7 asks recorder stages to fire in *both* norm branches, but the plan
  keeps the non-`norm_first` branch record-free (matching pre-F4, and the operative path is
  `norm_first=True`); the suite asserts recording on the `norm_first` path only. **Next up: F6.**
- ✓ **F6 — MixerModel + PipelineBuilder + config integration** — DONE. New config group
  `configs/model/image_encoder/` (`dinov3.yaml` = the relocated inline block + `type`/`embed_dim`;
  `mask2former.yaml` = F2 backbone flags); `mixer_model.yaml` gains `defaults: [image_encoder:
  dinov3, _self_]` and drops its inline `image_encoder:` mapping — the default composed
  `model.image_encoder.*` is **field-for-field identical** to pre-F6. `PipelineBuilder.build_model`
  branches on `image_encoder.get('type','dinov3')` (the default keeps pre-F6 snapshots loading):
  `dinov3` → raw `DinoV3Wrapper` (`n_image_levels=1`); `mask2former` → `Mask2FormerBackbone` wrapped
  in `Mask2FormerFeatureAdapter` (`n_image_levels=adapter.num_levels`, 3 or 4). `MixerModel` gains
  `image_encoder_type="dinov3"` / `n_image_levels=1`; `patch_resolution` + the `use_rope`
  `rope_embed` access are **DINOv3-guarded** (nominal `patch_size=16` feeds the `shared_gaussian`
  encoders on the m2f path, only ever consumed by `forward_features()`); the deformable eye/fixation
  decoders are built at `n_levels=n_image_levels`; a zero-init `level_embed (n_image_levels,
  model_dim)` is added to `denoise_modules` **only** on the m2f path. **Dual path, not unified:**
  DINOv3 keeps its exact legacy `encode`/`decode_fixation` (CLS-slice, `forward_features()` patch PE,
  `spatial_shapes=None` ⇒ F4 legacy dispatch) so its forward is byte-identical and old checkpoints
  load with zero missing/unexpected keys (no `level_embed`/`image_*` leak into the state_dict); the
  F3 `DinoV3FeatureAdapter` is **not** wired (reserved for a future unification). The m2f path:
  `bundle = image_encoder(img)` → `img_input_proj(bundle.value)` (256→`model_dim`) → per-level PE =
  `pos_proj(reference_grids)` (**shared `shared_gaussian` basis**) + `repeat_interleave(level_embed,
  level_sizes)` → eye/fixation decoders fed `spatial_shapes`/`level_start_index` (stored on the model
  for `decode_fixation`). FR8 guards raise at construction on the m2f path for `use_rope`,
  `head_type∈{argmax_regressor,heatmap}`, `input_encoder=="image_features_concat"` (all need a single
  square patch grid / DINOv3 internals). Frozen ResNet50 gets no gradient and stays `eval()` through
  `model.train()`; the pixel decoder + `level_embed` train; `sampling_offsets` stay in the 10× LR
  group. 38-test suite `tests/test_f6_integration.py` (CPU-only, `imagenet_weights=None`, DINOv3 via
  a deterministic stub). Spec: `spec/2026-09-02-mixermodel-pipelinebuilder-config-integration/`.
  **The Mask2Former backbone is now reachable end-to-end from a single config switch
  (`model/image_encoder=mask2former`).** Deferred (unchanged): wiring the stride-4 res2 map as a real
  4th decoder level, and routing DINOv3 through the adapter to unify the two paths.
- **Open items (non-blocking)** — the stride-4 (res2, 64²) map is produced behind F2's
  `return_stride4` flag, but wiring it as a real 4th decoder level (F4/F6) is deferred to the
  heatmap-regression iteration; how aggressively to share the `shared_gaussian` basis across scales;
  whether to later revive external pretrained pixel-decoder weights (COCO/ADE) as an ablation
  (would require reintroducing a key-remap loader).

### Engineering Correctness

- **Fix latent `NameError` in `pipeline_builder.py`** — `data_path` is only assigned inside the `LOCAL_SCRATCH` branch (lines 184–186) but is consumed unconditionally at line 199. Crashes silently when `LOCAL_SCRATCH` is not set and the dataset lives at the default path. Add a `data_path = os.path.join('data', 'Coco FreeView')` default before the conditional.
- **Remove hardcoded Windows paths from version-controlled configs** — `configs/model/mixer_model.yaml` contains `repo_path: "C:\\Users\\ulloa\\..."`. Move to a gitignored local override file (e.g., `configs/local.yaml`) or resolve via an environment variable.
- **Remove dead code** — `FreeViewBatch` in `src/data/datasets.py` is never instantiated by `PipelineBuilder`; remove it. Commented-out blocks in `mixer_model.py` (`ResidualRegressor`, `GatedFusion` param group) and `pipeline_builder.py` (`test_segment_is_inside`, DINO param group) should be deleted.
- **Named constants for magic numbers** — `320`, `512`, `1/16`, `4.13`, `5.5`, `PAD_TOKEN_ID = 0.5` and others appear inline across multiple files. Centralise in a `src/constants.py` or surface through config.

### Reproducibility & Experiment Management

- **Experiment tracking** — ✓ Weights & Biases integrated for the hyperparameter search (`scripts/hp_search.py`, one run per Optuna trial, per-epoch metric curves, sampled-param config). Remaining: wire W&B into the default `train.py` path (currently W&B is gated behind `training.wandb.enabled`, only flipped on by the search driver) and add cross-seed ablation dashboards.
- **DINOv3 dependency pinning** — the image encoder is loaded from an unpinned local git clone. Pin to a specific commit hash and add a checksum for the weights file so results are reproducible across machines.

### Code Quality

- **Minimal test suite** — add at least: (1) a forward-pass smoke test for both `PathModel` and `MixerModel` with dummy tensors, (2) unit tests for the noise transforms (check output shape and value range), (3) a test for `eval_reg` and `eval_denoise` against hand-computed values.
- **Replace `print()` with `logging`** — all diagnostic output currently goes to stdout via `print()`. Replace with `logging.getLogger(__name__)` so verbosity is configurable and output is filterable.
- **Type hints** — add function signatures to `PipelineBuilder`, model `forward`/`encode`/`decode_*`, and transform `__call__` methods. Not strictly necessary but cuts debugging time significantly.

### Research Extensions (Post-Baseline)

- **Real WebGazer validation** — validate the noise simulation pipeline against a small set of real paired WebGazer / lab-grade recordings to confirm the simulated distribution matches real noise characteristics.
- **Noise model ablation** — compare isotropic Gaussian vs. correlated radial vs. mixture-of-Gaussians to confirm the physics-informed noise model outperforms simpler alternatives.
- **Image encoder ablation** — compare DINOv3 ViT-S/16 frozen vs. fine-tuned vs. no image encoder (PathModel-equivalent) to quantify the contribution of image features.
- **Input encoder ablation** — `shared_gaussian` vs. `linear` on both PathModel and MixerModel to validate the positional encoding choice.

---

## Known Issues / Blockers

| Issue | Severity | Location |
|---|---|---|
| `data_path` potentially unbound if `LOCAL_SCRATCH` not set | **Bug** | `pipeline_builder.py:184–199` |
| Hardcoded Windows path in version-controlled config | **Reproducibility** | `configs/model/mixer_model.yaml:52` |
| DTW / multi-match not implemented | **Publication blocker** | `src/eval/` — missing entirely |
| Random seeds not enforced | **Reproducibility** | Training entry point (`src/training/pipeline.py`) |
| `FreeViewBatch` is dead code | **Maintenance** | `src/data/datasets.py:65–183` |
| `torch.compile` incompatible with `InferenceRecorder` | **Known limitation** | `pipeline.py:33–36` — already guarded, document it |
