# Roadmap

## Done

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

- **Spec-driven development workflow** — Constitution documents written; sprint-level feature specs not yet started

---

## Backlog

Priority order within each group.

### Blockers for Publication

- **Publication-level evaluation metrics** — DTW (Dynamic Time Warping), multi-match (Jarodzka et al.), and ScanMatch are listed in the Mission success criteria but not implemented. These require operating in pixel space after `Normalize.inverse()`. Without them the results section cannot be written.
- **Seeded reproducibility enforcement** — `torch.manual_seed`, `numpy.random.seed`, `random.seed`, and `torch.backends.cudnn.deterministic` are not set in the training entry point. The Mission claims runs are re-runnable from a config + seed; this is not currently true.
- **Formal baseline comparison infrastructure** — no script runs PathModel and MixerModel on the same split with the same seed and produces a comparison table. Required for the ablation section.
- **Comparison against SOTA scanpath models** — Mission scopes comparisons against Gazeformer, ScanDy, IORE (or equivalent); no integration or evaluation harness exists yet.

### Engineering Correctness

- **Fix latent `NameError` in `pipeline_builder.py`** — `data_path` is only assigned inside the `LOCAL_SCRATCH` branch (lines 184–186) but is consumed unconditionally at line 199. Crashes silently when `LOCAL_SCRATCH` is not set and the dataset lives at the default path. Add a `data_path = os.path.join('data', 'Coco FreeView')` default before the conditional.
- **Remove hardcoded Windows paths from version-controlled configs** — `configs/model/mixer_model.yaml` contains `repo_path: "C:\\Users\\ulloa\\..."`. Move to a gitignored local override file (e.g., `configs/local.yaml`) or resolve via an environment variable.
- **Remove dead code** — `FreeViewBatch` in `src/data/datasets.py` is never instantiated by `PipelineBuilder`; remove it. Commented-out blocks in `mixer_model.py` (`ResidualRegressor`, `GatedFusion` param group) and `pipeline_builder.py` (`test_segment_is_inside`, DINO param group) should be deleted.
- **Named constants for magic numbers** — `320`, `512`, `1/16`, `4.13`, `5.5`, `PAD_TOKEN_ID = 0.5` and others appear inline across multiple files. Centralise in a `src/constants.py` or surface through config.

### Reproducibility & Experiment Management

- **Experiment tracking** — integrate Weights & Biases (or MLflow) to tag runs with config diffs, track metric curves, and compare ablations across seeds. TensorBoard alone is insufficient at the ablation scale needed for publication.
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
