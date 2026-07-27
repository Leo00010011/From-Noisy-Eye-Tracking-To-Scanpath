# Requirements: EVE Real-Noise Scanpath Inference

## Goal

Every result this project has produced so far rests on *simulated* WebGazer noise applied to clean
CocoFreeView trajectories. To claim the model works on real degraded gaze, it must be run on a real
noisy eye-tracking signal. A ResNet18 gaze-estimation network was trained on EVE and its per-frame
predictions — 3D unit gaze vectors in the normalized (Zhang et al.) camera frame — are stored in
`predictions.csv` from the `EyeNet Pipeline` project. This feature converts those gaze vectors into a
screen-space noisy gaze trajectory using the `evedataset` package's canonical projection path, pairs
each trajectory with its resized stimulus image, runs a trained `MixerModel` checkpoint
autoregressively, and persists both the projected inputs and the predicted scanpaths to an
`exp_key`-keyed HDF5 artifact for offline study.

The recovery model was trained on CocoFreeView and has never seen any EVE data, so EVE's own
train/val/test partition carries no leakage meaning for it and is not used to select data. What does
matter is **EyeNet's** partition: the model must be tested on EyeNet output from data EyeNet itself
did not train on. All 734 experiments in `predictions.csv` are EyeNet val or test rows, so the whole
artifact is unseen-by-EyeNet — and the split label is retained per experiment so val and test can be
analysed separately.

Ground-truth EVE scanpaths are deliberately **not** consumed: access to them is still in progress in
the EVE repo. This feature therefore computes and stores outputs only — it computes no accuracy
metrics and makes no claims about quality. The artifact it produces is the input that a later
evaluation feature will join against ground-truth scanpaths once they are available.

---

## Scope

**In scope**

- `src/data/eve_real_noise.py` (new) — `EyeNetGazeCache` (screen-space projection cache),
  `EveRealNoiseDataset` (gaze), `EveRealNoiseImgDataset` (stimulus).
- `src/data/eve_real_noise_store.py` (new) — `RealNoiseInferenceStore`, the keyed HDF5 writer/reader
  for model outputs.
- `scripts/build_eyenet_gaze_cache.py` (new) — one-shot CLI that turns `predictions.csv` +
  `bundle.h5` into `eyenet_gaze_cache.h5`.
- `src/notebooks/save_predictions_eve_real.py` (new) — inference driver, mirroring the existing
  `save_predictions_eve.py`.
- `configs/data/eve_real.yaml` (new) — Hydra data group for this path.
- `tests/test_eve_real_noise.py` (new).
- Stimulus resize from the source 1920×1080 to the model's training resolution (256), via the
  existing `PipelineBuilder.make_transform`.

**Out of scope**

- Any modification to the `evedataset` package or the `EveDataset` repo. Everything this feature
  needs is already exposed by `EveBundle` (see Dependencies). No new EVE-side capability is
  requested.
- Ground-truth EVE scanpaths (`/samples/scanpath`), the Tobii `eye_track` signal, and therefore the
  Denoise phase's `clean_x` supervision. `clean_x` is not produced.
- Any accuracy, DTW, multi-match, or ScanMatch metric. No numbers comparing prediction to truth.
- Training or fine-tuning on EVE real-noise data. Inference only, `model.eval()`, `torch.no_grad()`.
- Changes to `MixerModel`, `PathModel`, loss functions, `seq2seq_padded_collate_fn`, or
  `CoupledDataloader`.
- Changes to the existing `dataset_type: "eve"` path (`_load_eve_dataset`, `configs/data/eve.yaml`),
  which consumes the WebGazer-replica `/webgazer` group and is left byte-for-byte untouched.
- The `right`/`left` eye ablation study, the `basler` camera, and video/Wikipedia EVE stimuli.

---

## Background: What the Input Data Actually Is

Measured against the production bundle at
`eve_shared/EveDataset/bundle/bundle.h5` and the 11.9 MB `predictions.csv`. These numbers are load-bearing
for the requirements below.

| Property | Value |
|---|---|
| CSV rows | 70,858 |
| CSV columns | `split, exp_key, frame, patch, pred_x, pred_y, pred_z, target_x, target_y, target_z, angular_error_deg` |
| Distinct `exp_key` | 734 |
| `patch` values | `left` (35,429 rows), `right` (35,429 rows) — every frame has both |
| `frame` range | 40 … 89 (center camera, 30 Hz, 90 frames total) |
| Frames per (exp_key, eye) | mean 48.3, median 50, min 5, max 50 |
| Duplicate `(exp_key, frame, patch)` | 0 |
| `‖(pred_x, pred_y, pred_z)‖` | 1.0 ± 1.4e-7 — already unit vectors, normalized frame |
| `angular_error_deg` | mean 4.09°, median 3.62°, p75 5.37°, max 38.7° |

**Coverage against the bundle.** All 734 `exp_key`s are present in `/samples`, in
`/gaze_norm/center`, and in `/gaze_ray/center` — so `project_normalized_gaze` is callable for every
row with no coverage gating needed. The CSV frame set for an experiment is exactly the subset of
`gaze_norm/center/{eye}_validity == True` frames with `frame >= 40` (verified on 200 experiments);
consequently the projection's own `validity` mask never removes a CSV frame (verified: 0 frames
dropped across all 734 experiments).

**Two split labels exist and they mean different things.** The CSV `split` column records the
*ResNet18's* partition; `bundle.samples_df["split"]` records *EVE's*. They disagree on 100% of rows:
every CSV `val` row (45,482) sits in an EVE **train** experiment and every CSV `test` row (25,376) in
an EVE **val** experiment.

The **CSV split is the operative one** and is renamed `eyenet_split` throughout to keep that
unambiguous. The recovery model was trained on CocoFreeView and has never seen EVE, so EVE's
partition implies nothing about leakage for it. What must be controlled is EyeNet's: predictions on
EyeNet's own training data would be optimistically clean and would not represent the noise regime the
recovery model has to survive. All 734 experiments are EyeNet val or test, so the entire artifact is
already unseen-by-EyeNet; the label is kept so val and test can be reported separately.

| `eyenet_split` | Experiments | Subjects | EVE subjects | Valid frames |
|---|---|---|---|---|
| `val` | 474 | 8 | `train05, train06, train08, train24, train26, train29, train31, train39` | 22,741 |
| `test` | 260 | 5 | `val01 … val05` | 12,688 |

Every `exp_key` carries exactly one `eyenet_split` (verified: 0 experiments span both), and the two
groups are subject-disjoint. The EVE split is stored alongside as descriptive metadata and is never
used to filter.

**Frame 40 is stimulus onset, not a global constant.** 75% of experiments start at frame 40 but the
per-experiment minimum ranges up to 60. Frame indices are taken verbatim from the CSV; onset is never
re-derived.

**No center-camera timestamps exist in the bundle.** `/samples/eye_track` is `(3096, 3, 180)` — the
basler camera at 60 Hz, on an absolute millisecond clock (row 2 spans ~2944 units over 180 frames).
The center camera runs at a fixed 30 Hz, so timestamps are synthesized (FR4).

---

## Functional Requirements

### FR1 — CSV parsing

**FR1.1** `load_eyenet_predictions(csv_path) -> pd.DataFrame` reads the CSV with `exp_key` and
`patch` as `str`, `frame` as `int32`, and the six vector columns plus `angular_error_deg` as
`float32`.

**FR1.2** It raises `ValueError` if any of the 11 expected columns is absent, if
`df.duplicated(["exp_key", "frame", "patch"]).any()`, if `frame` is outside `[0, 90)`, or if `patch`
contains a value other than `left`/`right`.

**FR1.3** It emits a `warnings.warn` (never raises) if `‖(pred_x, pred_y, pred_z)‖` deviates from 1.0
by more than `1e-3` for any row, reporting the count. Vectors are **not** re-normalized here —
`project_normalized_gaze` normalizes after the rotation.

### FR2 — Per-experiment scatter into a 90-frame buffer

**FR2.1** `project_normalized_gaze` requires the prediction array's frame count to equal the camera
frame count (90) exactly; a mismatch raises `ValueError` inside `evedataset`. For each
`(exp_key, eye)` the sparse CSV frames are therefore scattered into a dense `(90, 3)` float64 buffer.

**FR2.2** Frames absent from the CSV are filled with the sentinel unit vector `[0.0, 0.0, -1.0]`
(straight ahead in the normalized frame). The fill value is never read back: a separate
`(90,)` bool `frame_mask`, `True` exactly at the CSV frame indices, gates every downstream use.

**FR2.3** The same scatter is applied to `(target_x, target_y, target_z)` to produce the ray-derived
ground-truth direction buffer, and to `angular_error_deg` (fill `NaN`).

### FR3 — Screen-space projection and eye combination

**FR3.1** For each `(exp_key, eye)`, call
`bundle.project_normalized_gaze(exp_key, pred_buffer, eye=eye, spherical=False)`. It returns
`hit_px (90,2) float32`, `hit_mm`, `depth`, `direction_screen`, and `validity (90,) bool`.

**FR3.2** Per-eye validity is `eye_valid = frame_mask & result["validity"]`.

**FR3.3** The combined signal is the **per-frame mean of the two eyes' screen intercepts**:

| Condition | `gaze_px[t]` | `validity[t]` |
|---|---|---|
| `left_valid[t] and right_valid[t]` | `(left_px[t] + right_px[t]) / 2` | `True` |
| exactly one eye valid | that eye's `hit_px[t]` | `True` |
| neither valid | `NaN` | `False` |

Averaging is done in screen-pixel space, not vector space, so it composes with the per-eye origin
each ray already carries. On the production data this reduces the median per-frame distance to the
ray-derived ground truth from 149 px (single eye) to **89.6 px = 2.32 DVA** at 38.55 px/DVA.

**FR3.4** The identical combination is applied to the `target_*` projections to produce
`gt_gaze_px (90,2)`, stored as a clean reference. It is stored for offline study only and is not fed
to the model.

**FR3.5** Coordinates are stored **unclamped**. On the production data 0.0% of combined predicted
frames and 0.065% of ground-truth frames fall outside `[0,1920)×[0,1080)`, so clamping would be a
no-op that hides genuine off-screen intercepts. Off-screen frames remain valid and are counted into
the `n_offscreen` build statistic.

### FR4 — Timestamps

**FR4.1** The `t` row of the gaze trajectory is synthesized as
`t_ms = frame_idx * (1000.0 / 30.0)` — the center camera's fixed 30 Hz rate. `CENTER_FPS = 30.0` and
`CENTER_FRAME_COUNT = 90` are module constants.

**FR4.2** `StandarizeTime` (first transform in the list) subtracts `t[0]`, so the absolute origin is
irrelevant; only the 33.333 ms inter-frame spacing matters. Because CSV frames are contiguous within
an experiment's valid window, this reproduces true relative timing exactly under the constant-rate
assumption.

**FR4.3** This is an approximation, and it is recorded as such: the cache stores
`timestamp_source = "synthesized_30hz"` as an HDF5 attribute on the `/eyenet_gaze` group so a future
feature can detect artifacts built before real center timestamps were available.

### FR5 — `EyeNetGazeCache` HDF5 layout

Stored in its own file (default `data/eve_real_noise/eyenet_gaze_cache.h5`) under the
`/eyenet_gaze` group. `N` = number of cached experiments (734 on the current CSV).

| Dataset | Shape | dtype | Description |
|---|---|---|---|
| `exp_keys` | `(N,)` | vlen UTF-8 | **Primary key**, verified on load |
| `eyenet_split` | `(N,)` | vlen UTF-8 | ResNet18's split from the CSV — **the operative filter** (`val` / `test`) |
| `eve_split` | `(N,)` | vlen UTF-8 | EVE split from `bundle.samples_df` — descriptive metadata, never used to filter |
| `stimulus_name` | `(N,)` | vlen UTF-8 | From `bundle.samples_df`, used for image dedup |
| `gaze_px` | `(N, 90, 2)` | float32 | Combined predicted screen intercept; `NaN` where invalid |
| `validity` | `(N, 90)` | bool | Combined per-frame validity (FR3.3) |
| `gt_gaze_px` | `(N, 90, 2)` | float32 | Combined ray-derived ground truth; `NaN` where invalid |
| `left_px` / `right_px` | `(N, 90, 2)` | float32 | Per-eye intercepts; `NaN` where invalid |
| `left_validity` / `right_validity` | `(N, 90)` | bool | Per-eye validity |
| `angular_error_deg` | `(N, 90, 2)` | float32 | CSV column, `[..., 0]`=left, `[..., 1]`=right; `NaN` where absent |

Group attributes: `timestamp_source` (str), `center_fps` (float), `source_csv` (str),
`bundle_dir` (str), `built_at` (ISO-8601 str), `n_offscreen` (int).

Per-eye arrays are stored in addition to the combined signal (a ~0.5 MB total cost) because the
Data Validity checks in `validation.md` require left/right agreement statistics, and because a
future eye-ablation would otherwise force a full rebuild.

### FR6 — `EyeNetGazeCache` API and error conditions

**FR6.1** `EyeNetGazeCache.build(csv_path, bundle, cache_path=DEFAULT) -> (cache, skipped)` —
`skipped` is a `list[tuple[str, str]]` of `(exp_key, reason)`. An experiment is skipped, never
crashed on, when: it is absent from `bundle.samples_df` (`reason="not_in_bundle"`), when
`bundle.has_gaze_norm(exp_key)` or `bundle.has_gaze_ray(exp_key)` is `False`
(`"no_gaze_norm"` / `"no_gaze_ray"`), or when `project_normalized_gaze` raises `KeyError`/`ValueError`
(`"projection_failed: {msg}"`). On the current data `skipped` is expected to be empty.

**FR6.2** `cache.save(cache_path)` opens the file with `h5py.File(path, "a")` and writes only the
`/eyenet_gaze` group, deleting and recreating it if present. It never touches another group.

**FR6.3** `EyeNetGazeCache.load(cache_path) -> EyeNetGazeCache` raises `FileNotFoundError` if the
file is absent and `ValueError` if the `/eyenet_gaze` group is missing, if any expected dataset is
missing, or if `exp_keys` contains a duplicate. It builds an internal `exp_key -> row index` dict;
no consumer ever indexes by integer position.

**FR6.4** `cache.verify(bundle) -> list[tuple[str, str]]` returns `(exp_key, reason)` for every
cached key absent from `bundle.samples_df`, for every key whose cached `eve_split` disagrees with the
bundle's, and for every key whose `eyenet_split` is not in `("val", "test")` — the last of which would
mean EyeNet training data leaked into the artifact. It returns, never raises: it is a QA hook, not a
load guard.

**FR6.5** Accessors, all addressed by key and raising `KeyError` on an unknown one:
`get_gaze(exp_key)`, `get_validity(exp_key)`, `get_gt_gaze(exp_key)`, `get_eye_gaze(exp_key, eye)`,
`get_angular_error(exp_key)`. `exp_keys` and `splits_df` (columns `exp_key`, `eyenet_split`,
`eve_split`, `stimulus_name`) are read-only properties.

### FR7 — `EveRealNoiseDataset`

**FR7.1** Signature:

```python
EveRealNoiseDataset(
    cache: EyeNetGazeCache,
    bundle: EveBundle,
    eyenet_split: str | None = None,   # "val" | "test" | None (= both)
    max_fixations: int = 20,
    min_valid_frames: int = 5,
    transforms: list = (),
    log: bool = False,
)
```

The filter parameter is named `eyenet_split`, not `split`, so a caller cannot accidentally pass an
EVE split label. A value outside `("val", "test", None)` raises `ValueError` listing the accepted
values — passing `"train"` (an EVE-only concept here) fails loudly rather than yielding an empty
dataset.

**FR7.2** At `__init__` it materializes every accepted row into RAM (no HDF5 handle is retained), in
the cache's stored `exp_key` order filtered by `eyenet_split`. Per row it builds:

- `x` `(3, T_valid)` float64 — rows `[x_px, y_px, t_ms]`, columns are the frames where
  `cache.get_validity(exp_key)` is `True`, in ascending frame order. `t_ms` per FR4.1.
- `y` `(3, max_fixations)` float64 — the **placeholder target** (FR7.4).
- `fixation_mask` `(T_valid,)` uint8 — all zeros.

**FR7.3** A row is skipped when `T_valid < min_valid_frames`. On the production data all 734
experiments satisfy `T_valid >= 5`, so 0 rows are expected to be skipped at the default.

**FR7.4 — The placeholder target.** No ground-truth scanpath is available, but
`seq2seq_padded_collate_fn` requires `item['y']` and `eval_autoregressive` derives its decode-step
budget from `tgt_mask.size(1)`. `y` is therefore filled entirely with `PAD_TOKEN_ID` (0.5) and shaped
`(3, max_fixations)`, yielding a `tgt_mask` of width `max_fixations + 1` and exactly
`max_fixations + 1` autoregressive decode steps. The placeholder is **never** read as supervision:
inference runs under `torch.no_grad()` with no loss, and `eval_autoregressive` sets
`inputs['tgt'] = None` on the first step and feeds back only the model's own predictions thereafter.
Sequence termination is decided offline from the stored EOS logits (FR9.3), not from `y`.

**FR7.5** `self.data_store` contains keys `x`, `y`, `fixation_mask`, `exp_keys`, `eyenet_splits`,
`eve_splits`, `frame_indices` (`list[np.ndarray]`, the source frame index of each column of `x`).
`__getitem__(i)` applies `self.transforms` in order and returns
`{'x': ..., 'y': ..., 'sample_idx': i}` plus any of `in_tgt`/`down_offset`/`heatmaps` a transform
added — matching `EveScanpathDataset.__getitem__` exactly. `clean_x` is never emitted.

**FR7.6** `dataset.exp_key_at(i) -> str` maps a positional sample index back to its key. Every
artifact written downstream is keyed through this method; no consumer relies on positional
correspondence between the dataset and the output store.

### FR8 — `EveRealNoiseImgDataset` and the 1080p → 256 resize

**FR8.1** Signature mirrors `EveImgDataset`:

```python
EveRealNoiseImgDataset(
    cache: EyeNetGazeCache,
    bundle: EveBundle,
    eyenet_split: str | None = None,
    max_fixations: int = 20,      # unused; accepted so the two datasets share a filter signature
    min_valid_frames: int = 5,
    resize_size: int = 256,
    transform=None,
)
```

**FR8.2** It applies the **identical** accept/skip filter as `EveRealNoiseDataset` (same
`eyenet_split`, same `min_valid_frames`), so positional index `i` refers to the same `exp_key` in
both. This invariant is asserted at construction of the pair (FR8.5) rather than assumed.

**FR8.3** Images are deduplicated by `stimulus_name`. Each unique stimulus is read once via
`bundle.get_stimulus(exp_key)` → `(1080, 1920, 3)` uint8 RGB, ingested through
`v2.Compose([ToImage(), Resize((resize_size, resize_size), antialias=True), ToDtype(torch.uint8, scale=False)])`
into an `image_bank` of shape `(N_unique, 3, resize_size, resize_size)` uint8.

**FR8.4 — Resize semantics.** `Resize((256, 256))` on a 1920×1080 source is a **non-uniform** squash,
not an aspect-preserving fit. This is correct and deliberate: `NormalizeCoords` divides x by 1920 and
y by 1080 independently, so normalized gaze coordinates in `[0,1]²` map linearly onto the squashed
image's `[0,1]²`. Preserving aspect ratio would break that correspondence. It also matches how
CocoFreeView images are ingested by `DeduplicatedMemoryDataset`.

**FR8.5** `resize_size` must equal the `img_size` the checkpoint was trained at — **256**, matching
`configs/data/default.yaml` and `configs/data/eve.yaml`, which both already use 256.
`save_predictions_eve_real.py` reads `img_size` from the merged config and raises `ValueError` if
`configs/data/eve_real.yaml`'s `load.img_size` disagrees with the checkpoint's own
`.hydra/config.yaml` value, rather than silently feeding the model a resolution it never saw. The
check is on equality with the checkpoint, not on the literal 256, so a future model trained at
another resolution is handled by editing one config value.

**FR8.6** `__getitem__(i)` returns `(img_tensor, i, unique_img_idx)` — the 3-tuple
`CoupledDataloader` expects — with `self.runtime_transform` applied when set.

### FR9 — `RealNoiseInferenceStore` HDF5 layout

Written to `outputs/eve_real_noise/{run_name}.h5`, group `/inference`. `N` = samples,
`K = max_fixations + 1` decode steps, `T = max_valid_frames` across the run.

| Dataset | Shape | dtype | Description |
|---|---|---|---|
| `exp_keys` | `(N,)` | vlen UTF-8 | **Primary key**, verified on load |
| `eyenet_split` | `(N,)` | vlen UTF-8 | `val` / `test` — carried through so results can be reported per split |
| `eve_split` | `(N,)` | vlen UTF-8 | Descriptive metadata |
| `pred_scanpath` | `(N, K, 3)` | float32 | Predicted fixations `[x_px, y_px, dur_ms]`, pixel space |
| `eos_logit` | `(N, K)` | float32 | Raw end-of-sequence logits; apply `sigmoid` for probability |
| `pred_len` | `(N,)` | int32 | First `k` with `sigmoid(eos_logit[k]) > 0.5`, else `K` (FR9.3) |
| `src_px` | `(N, T, 3)` | float32 | Model input inverted to pixel space `[x_px, y_px, t_ms]`; `NaN` padding |
| `src_len` | `(N,)` | int32 | Valid length of `src_px` per row |
| `frame_indices` | `(N, T)` | int32 | Source center-camera frame of each `src_px` column; `-1` padding |
| `denoise_px` | `(N, T, 2)` | float32 | Denoise-head output in pixel space; **written only if the checkpoint has a denoise head**; `NaN` padding |

Group attributes: `run_name`, `checkpoint_path`, `img_size` (int), `max_fixations` (int),
`gaze_cache_path` (str), `bundle_dir` (str), `created_at` (ISO-8601), `has_denoise` (bool).

**FR9.1** `RealNoiseInferenceStore.save(path, run_name, records, attrs)` writes with
`h5py.File(path, "w")`. `records` is a `list[dict]`; every dict must carry `exp_key`. Raises
`ValueError` if `exp_key` is missing from any record or if the list contains a duplicate `exp_key`.

**FR9.2** `RealNoiseInferenceStore.load(path)` raises `FileNotFoundError` if absent, `ValueError` if
`/inference` or `exp_keys` is missing or contains duplicates. Accessors `get(exp_key) -> dict` and
`get_scanpath(exp_key) -> (pred_len, 3) float32` (already trimmed by `pred_len`) raise `KeyError` on
an unknown key. `df` is a read-only `pd.DataFrame` with `exp_key`, `eyenet_split`, `eve_split`,
`pred_len`, `src_len`.

**FR9.3** `pred_len` is a **stored convenience, not a truncation**: `pred_scanpath` always retains all
`K` decode steps so an offline study can re-threshold the EOS head. The 0.5 probability threshold is
recorded as the `eos_threshold` group attribute.

**FR9.4** All coordinates in the store are in **1920×1080 screen pixels**, recovered by applying
`invert_transforms` (which calls each `Normalize.inverse()`), never by hand-multiplying. Durations
are in milliseconds.

### FR10 — `configs/data/eve_real.yaml`

A new Hydra data group. It sets `dataset_type: "eve_real"`, and reuses the `eve.yaml` transform list
minus everything tied to `clean_x`:

- `transform_list: [StandarizeTime, NormalizeCoords, NormalizeTime, NormalizeFixationCoords, NormalizeDuration]`
- `has_precomputed_clean_x: false` — no Tobii reference in this path.
- `NormalizeCoords` / `NormalizeFixationCoords`: `image_W: 1920`, `image_H: 1080`.
- `NormalizeTime.period_duration: 3000` — 90 frames at 30 Hz.
- `NormalizeDuration.period_duration: 1200` — matches the EVE scanpath pipeline's outlier cutoff.
- `load.img_size: 256`, `load.batch_size: 32`, `load.use_img_dataset: true`,
  `load.num_workers: 0` (the datasets are fully in RAM; workers add fork cost for no gain at N=734).
- `gaze_cache_path: "data/eve_real_noise/eyenet_gaze_cache.h5"`, `bundle_dir: "MUST_OVERRIDE"`,
  `max_fixations: 20`, `min_valid_frames: 5`.
- `eyenet_split: null` — `null` runs both EyeNet val and test in one pass; set to `"val"` or
  `"test"` to restrict. There is deliberately no EVE-split key.

**FR10.1** `PipelineBuilder` is **not** modified. `dataset_type: "eve_real"` is consumed only by
`save_predictions_eve_real.py`, which constructs the datasets and the `CoupledDataloader` directly.
Keeping this out of `PipelineBuilder` avoids touching the training entry point for an inference-only
feature, and avoids the risk of the existing `eve` branch regressing.

### FR11 — Inference driver

**FR11.1** `save_predictions_eve_real.py` loads a checkpoint's `.hydra/config.yaml`, merges
`configs/data/eve_real.yaml` over its `data` group, overrides `data.bundle_dir`, and validates
`img_size` per FR8.5.

**FR11.2** It builds the model via `PipelineBuilder.build_model()`, loads `model.pth` with the
`_orig_mod.` prefix stripped, calls `model.set_phase('Fixation')`, `model.eval()`, and moves to CUDA
when available.

**FR11.3** Batches come from `CoupledDataloader(gaze_dataset, Subset(img_dataset, arange(N)),
shuffle=False, drop_last_batch=False, num_workers=0, persistent_workers=False)`. `shuffle=False` and
`drop_last_batch=False` are **required**: shuffling would break the `sample_idx → exp_key` mapping
used to key the output, and dropping the last partial batch would silently lose up to
`batch_size - 1` experiments.

**FR11.4** Per batch: `eval_autoregressive(model, inp, only_last=True)`, then
`model.decode_denoise(**inp)` when `callable(getattr(model, 'decode_denoise', None))`, then
`invert_transforms(inp, output, dl, remove_outliers=True)`. Records are keyed via
`gaze_dataset.exp_key_at(int(inp['sample_idx'][i]))`.

**FR11.5** It raises `ValueError` if the number of records written differs from
`len(gaze_dataset)`, or if any `exp_key` is emitted twice.

---

## Public API Summary

```python
# src/data/eve_real_noise.py

CENTER_FPS: float = 30.0
CENTER_FRAME_COUNT: int = 90
STIMULUS_W: int = 1920
STIMULUS_H: int = 1080
DEFAULT_CACHE_PATH: str = "data/eve_real_noise/eyenet_gaze_cache.h5"
_FILL_DIRECTION: np.ndarray = np.array([0.0, 0.0, -1.0])


def load_eyenet_predictions(csv_path: str | Path) -> pd.DataFrame: ...


class EyeNetGazeCache:
    @classmethod
    def build(cls, csv_path, bundle, cache_path=DEFAULT_CACHE_PATH
              ) -> tuple["EyeNetGazeCache", list[tuple[str, str]]]: ...
    @classmethod
    def load(cls, cache_path=DEFAULT_CACHE_PATH) -> "EyeNetGazeCache": ...
    def save(self, cache_path=DEFAULT_CACHE_PATH) -> None: ...
    def verify(self, bundle) -> list[tuple[str, str]]: ...

    @property
    def exp_keys(self) -> list[str]: ...
    @property
    def splits_df(self) -> pd.DataFrame: ...   # exp_key, eyenet_split, eve_split, stimulus_name

    def get_gaze(self, exp_key: str) -> np.ndarray: ...              # (90, 2) float32, NaN-gapped
    def get_validity(self, exp_key: str) -> np.ndarray: ...          # (90,) bool
    def get_gt_gaze(self, exp_key: str) -> np.ndarray: ...           # (90, 2) float32
    def get_eye_gaze(self, exp_key: str, eye: str) -> dict: ...      # px (90,2), validity (90,)
    def get_angular_error(self, exp_key: str) -> np.ndarray: ...     # (90, 2) float32


class EveRealNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, cache, bundle, eyenet_split=None, max_fixations=20,
                 min_valid_frames=5, transforms=(), log=False): ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> dict: ...       # {'x', 'y', 'sample_idx', ...}
    def exp_key_at(self, i: int) -> str: ...
    def eyenet_split_at(self, i: int) -> str: ...
    def eve_split_at(self, i: int) -> str: ...
    def frame_indices_at(self, i: int) -> np.ndarray: ...


class EveRealNoiseImgDataset(torch.utils.data.Dataset):
    def __init__(self, cache, bundle, eyenet_split=None, max_fixations=20,
                 min_valid_frames=5, resize_size=256, transform=None): ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]: ...
    def exp_key_at(self, i: int) -> str: ...


# src/data/eve_real_noise_store.py

class RealNoiseInferenceStore:
    @classmethod
    def save(cls, path, run_name: str, records: list[dict], attrs: dict) -> None: ...
    @classmethod
    def load(cls, path) -> "RealNoiseInferenceStore": ...

    @property
    def df(self) -> pd.DataFrame: ...
    @property
    def has_denoise(self) -> bool: ...
    def get(self, exp_key: str) -> dict: ...
    def get_scanpath(self, exp_key: str) -> np.ndarray: ...   # (pred_len, 3) float32, px + ms
```

---

## Dependencies

| Direction | Artifact | Used for |
|---|---|---|
| Read | `EyeNet Pipeline/predictions.csv` | 70,858 per-frame normalized gaze vectors, 734 experiments |
| Read | `evedataset.EveBundle.load(bundle_dir)` | Bundle handle; must carry `/gaze_norm/center` **and** `/gaze_ray/center` (`include_gaze_vector_data=True`, `include_gaze_ray_data=True`) |
| Read | `EveBundle.project_normalized_gaze(exp_key, pred, eye, spherical=False)` | Canonical normalized-frame → screen-pixel projection (FR3.1) |
| Read | `EveBundle.has_gaze_norm` / `has_gaze_ray` | Coverage gating (FR6.1) |
| Read | `EveBundle.get_stimulus(exp_key)` | `(1080, 1920, 3)` uint8 stimulus (FR8.3) |
| Read | `EveBundle.samples_df` | `stimulus_name` for image dedup, plus `eve_split` as metadata (FR5) |
| Read | `PipelineBuilder.make_transform(resize_size)` | ImageNet normalization for the resized stimulus |
| Read | `PipelineBuilder.build_model()` | Model construction from the checkpoint config |
| Read | `src.data.datasets.CoupledDataloader`, `seq2seq_padded_collate_fn`, `PAD_TOKEN_ID` | Batching (unmodified) |
| Read | `src.eval.eval_utils.eval_autoregressive`, `invert_transforms` | Autoregressive decode and pixel-space recovery |
| Write | `data/eve_real_noise/eyenet_gaze_cache.h5` `/eyenet_gaze` | Projected screen-space gaze cache (FR5) |
| Write | `outputs/eve_real_noise/{run_name}.h5` `/inference` | Model outputs (FR9) |
| Not touched | `evedataset` package, `EveDataset` repo | No EVE-side change is required by this feature |
| Not touched | `PipelineBuilder`, `configs/data/eve.yaml`, `MixerModel`, transforms | Existing `dataset_type: "eve"` path is unchanged |
