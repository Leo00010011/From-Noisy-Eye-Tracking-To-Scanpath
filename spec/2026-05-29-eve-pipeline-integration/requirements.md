# Requirements: EVE Pipeline Integration

## Goal

Enable `MixerModel` and `PathModel` to train and evaluate on real EVE WebGazer data by wiring
the `EveScanpathDataset` / `EveImgDataset` classes (implemented in the EVE repo as F4.5) into
`PipelineBuilder`, and by extending `EveScanpathDataset` to supply Tobii gaze as `clean_x` for
the Denoise training phase.

This spec covers changes in **two repositories**:

| Repo | Changes |
|---|---|
| `EveDataset` (EVE repo) | Extend `_compute_sample` and `EveScanpathDataset` to compute and store `clean_x` from `bundle.get_eye_track(exp_key)` |
| `From-Noisy-Eye-Tracking-To-Scanpath` (pipeline repo) | Add `has_precomputed_clean_x` flag to `_build_transforms`; add `configs/data/eve.yaml` |

---

## Scope

**In scope:**
- `src/evedataset/dataset.py` in the EVE repo — `_compute_sample` and `EveScanpathDataset` extension for `clean_x`
- `src/training/pipeline_builder.py` in the pipeline repo — `has_precomputed_clean_x` flag in `_build_transforms`
- `configs/data/eve.yaml` — new Hydra config file for EVE training
- `tests/test_eve_pipeline_integration.py` in the pipeline repo — smoke tests for the integration

**Out of scope:**
- Any changes to the model, loss function, or training loop
- Changes to `CoupledDataloader`, `seq2seq_padded_collate_fn`, or any existing transform class
- `ExtractRandomPeriod` (deliberately omitted for EVE — see design decision below)
- Noise simulation transforms (`AddIsotropicGaussianNoise`, `AddRandomCenterCorrelatedRadialNoise`, `DiscretizationNoise`) — omitted; `x` is already real WebGazer noise
- Cluster deployment documentation (a separate task)

---

## Background: Why EVE Needs Adaptation

### Real WebGazer signal, no simulation

For COCO FreeView, the pipeline takes clean lab-grade gaze and simulates WebGazer noise. The transform
list therefore begins with `SaveCleanX` (to snapshot gaze before noise), then applies noise transforms.

For EVE, the `x` signal IS the real WebGazer prediction — already noisy. No noise transforms should be
applied. The `clean_x` reference for the Denoise phase comes from the Tobii hardware gaze signal stored
in the bundle.

### Variable-length valid sequences, no period extraction

COCO stores the full gaze recording (typically 5–10 seconds). `ExtractRandomPeriod` crops a fixed
2600 ms window at a random temporal offset to create training diversity.

EVE's WebGazer signal is limited to **post-stimulus valid frames only** — the `EveScanpathDataset`
(F4.5) already pre-filters to valid frames, so `x.shape[1]` (T_valid) ranges from 0 to 90. A 2600 ms
window requires `ceil(2600/33.333) = 78` frames. Any sample with fewer valid frames would cause
`extract_random_period` to crash with `ValueError: low >= high` in the `randint` call.

**Decision: Skip `ExtractRandomPeriod` entirely for EVE.** The full valid sequence is passed to the
model. Variable-length padding (already handled by `seq2seq_padded_collate_fn`) accommodates sequences
from a few frames up to 90 frames.

### Denoise phase: Tobii gaze as `clean_x`

The Denoise phase uses `clean_x` — the clean reference for the noisy `x` input. For EVE, this is the
per-frame Tobii gaze aligned to WebGazer timestamps.

`EveBundle.get_eye_track(exp_key)` returns `{"gaze": (3, T_basler) float32, "validity": (T_basler,) bool}`
where `gaze[0]` = x_px, `gaze[1]` = y_px, `gaze[2]` = timestamp_seconds (all in 1920×1080 screen space).

`T_basler = 180` for basler-mode bundles (60 Hz). For each valid WebGazer frame at timestamp `t_k`
seconds, the nearest Tobii frame is found by minimising `|tobii_timestamps - t_k|`.

---

## FR1 — Changes to `EveScanpathDataset` in the EVE repo

### FR1.1 — Extended `_compute_sample` signature

```python
def _compute_sample(
    gaze_pred: np.ndarray,          # (3, 90) float32 — rows [x_px, y_px, t_sec]
    validity: np.ndarray,           # (90,) bool
    scanpath: np.ndarray,           # (4, F) float32 — rows [start_t_sec, dur_ms, x_px, y_px]
    eye_track: np.ndarray,          # (3, T_basler) float32 — rows [x_px, y_px, t_sec]; NaN where invalid
    eye_track_validity: np.ndarray, # (T_basler,) bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, fixation_mask, clean_x).
    
    x:            (3, T_valid) float64 — rows [x_px, y_px, t_ms]
    y:            (3, F) float64       — rows [x_px, y_px, dur_ms]
    fixation_mask: (T_valid,) uint8   — 0=saccade, k=fixation k (1-indexed)
    clean_x:       (3, T_valid) float64 — rows [tobii_x_px, tobii_y_px, t_ms]
                   NaN-free: any frame where Tobii was invalid uses the nearest filled value.
    """
```

The existing computation of `x`, `y`, and `fixation_mask` remains unchanged (from F4.5 spec).
`clean_x` is computed as follows:

**Step 1 — Interpolate Tobii invalids:**
```python
from src.scanpath_tools import interpolate_invalids

# eye_track shape: (3, T_basler), rows [x_px, y_px, t_sec]
# eye_track_validity shape: (T_basler,)
tobii, tobii_validity = interpolate_invalids(eye_track, eye_track_validity, max_run=2)
# tobii still has NaN for runs > max_run and for leading invalids.
# Apply forward-fill for any remaining leading/interior NaN:
tobii = _forward_backward_fill(tobii, tobii_validity)
# After fill, tobii[0:2] is NaN-free iff eye_track_validity.sum() > 0.
```

Helper function `_forward_backward_fill(eye_track, validity)` (module-level private):
```python
def _forward_backward_fill(eye_track: np.ndarray, validity: np.ndarray) -> np.ndarray:
    """Fill NaN entries in eye_track[0:2] using forward-fill then backward-fill.
    
    eye_track: (3, T) — rows [x, y, t]; t column is never modified.
    validity:  (T,) bool — True for valid frames.
    Returns a copy with NaN in x and y rows replaced by nearest valid neighbor.
    """
    out = eye_track.copy()
    for row in (0, 1):
        series = out[row].copy()
        # forward fill
        last_valid = None
        for i in range(len(series)):
            if validity[i]:
                last_valid = series[i]
            elif last_valid is not None:
                series[i] = last_valid
        # backward fill (for leading NaN)
        last_valid = None
        for i in range(len(series) - 1, -1, -1):
            if validity[i]:
                last_valid = series[i]
            elif last_valid is not None:
                series[i] = last_valid
        out[row] = series
    return out
```

**Step 2 — Align Tobii to WebGazer timestamps:**
```python
tobii_timestamps = tobii[2, :]   # shape (T_basler,), in seconds
valid_frames = np.where(validity)[0]   # WebGazer valid frame indices

clean_x = np.empty((3, len(valid_frames)), dtype=np.float64)
for k, frame_idx in enumerate(valid_frames):
    t_k = float(gaze_pred[2, frame_idx])   # seconds
    nearest_j = int(np.argmin(np.abs(tobii_timestamps - t_k)))
    clean_x[0, k] = tobii[0, nearest_j]   # x_px
    clean_x[1, k] = tobii[1, nearest_j]   # y_px
    clean_x[2, k] = t_k * 1000.0          # same timestamp as x[2, k] (ms)
```

### FR1.2 — Extended skip condition in `EveScanpathDataset.__init__`

Add a third skip condition after the existing `x.shape[1] == 0` and `y.shape[1] == 0` checks:

```python
if eye_track_validity.sum() == 0:
    # Cannot compute clean_x — no valid Tobii frames at all.
    skipped += 1
    continue
```

This matches the bundle's contract: rows with all-invalid Tobii have `eye_track_validity` all-False.

### FR1.3 — Load eye_track data in `EveScanpathDataset.__init__`

Inside the pair-loading loop, add a `bundle.get_eye_track(exp_key)` call before computing the
sample:

```python
for exp_key, run_key in pairs:
    wg         = bundle.get_webgazer(exp_key, run_key)
    scanpath   = bundle.get_scanpath(exp_key)          # (4, F)
    eye_track_data = bundle.get_eye_track(exp_key)     # {"gaze": (3, T), "validity": (T,)}
    gaze_pred  = wg["gaze_pred"]                        # (3, 90) float32
    validity   = wg["validity"]                         # (90,) bool
    et_gaze    = eye_track_data["gaze"]                 # (3, T_basler) float32
    et_validity = eye_track_data["validity"]            # (T_basler,) bool

    if et_validity.sum() == 0:    # new skip condition
        skipped += 1
        continue

    x, y, mask, clean_x = _compute_sample(gaze_pred, validity, scanpath, et_gaze, et_validity)

    if x.shape[1] == 0 or y.shape[1] == 0:
        skipped += 1
        continue
    ...
```

### FR1.4 — Store `clean_x` in `data_store`

Extend `self.data_store` to include `clean_x`:

```python
self.data_store = {
    'x':             xs,
    'y':             ys,
    'fixation_mask': masks,
    'clean_x':       clean_xs,    # new
    'exp_keys':      exp_keys,
    'run_keys':      run_keys,
}
```

### FR1.5 — Return `clean_x` in `__getitem__`

The existing optional-key loop already handles `clean_x`:
```python
for k in ('clean_x', 'in_tgt', 'down_offset', 'heatmaps'):
    if k in inp:
        out[k] = inp[k]
```

Since `clean_x` is now stored in `data_store`, it must also be added to `inp` before the optional-key
loop:

```python
def __getitem__(self, i: int) -> dict:
    x             = self.data_store['x'][i].copy()
    y             = self.data_store['y'][i].copy()
    fixation_mask = self.data_store['fixation_mask'][i].copy()
    clean_x       = self.data_store['clean_x'][i].copy()   # new
    inp = {'x': x, 'y': y, 'fixation_mask': fixation_mask, 'clean_x': clean_x}
    for t in self.transforms:
        inp = t(inp)
    out = {'x': inp['x'], 'y': inp['y'], 'sample_idx': i}
    for k in ('clean_x', 'in_tgt', 'down_offset', 'heatmaps'):
        if k in inp:
            out[k] = inp[k]
    return out
```

---

## FR2 — `has_precomputed_clean_x` Flag in `PipelineBuilder`

### FR2.1 — Purpose

When `EveScanpathDataset` supplies `clean_x` directly (from Tobii gaze), the transform pipeline must
still normalize `clean_x[:2]` (coordinate rows) the same way it normalizes `x[:2]`. In the COCO path,
this is triggered by `has_save_clean_x = True` which is set when `SaveCleanX` appears in the
transform list.

For EVE, `SaveCleanX` must NOT appear in the transform list (it would overwrite the Tobii `clean_x`
with the noisy `x`). Instead, a config flag signals that `clean_x` is precomputed and needs
normalization.

### FR2.2 — Config key

Add `has_precomputed_clean_x: true` to `configs/data/eve.yaml`. It defaults to `false` and has no
effect on the COCO path.

### FR2.3 — Implementation in `_build_transforms()`

At the very start of `_build_transforms()`, before the transform-list loop, read the flag:

```python
def _build_transforms(self) -> list:
    transforms = []
    has_save_clean_x = getattr(self.config.data, 'has_precomputed_clean_x', False)
    has_in_tgt = False
    if hasattr(self.config.data, 'transforms'):
        for transform_str in self.config.data.transforms.transform_list:
            ...
            elif transform_str == 'SaveCleanX':
                transforms.append(SaveCleanX())
                has_save_clean_x = True          # also set here for COCO
            elif transform_str == 'NormalizeCoords':
                transforms.append(build_normalize_coords(transform_config))
                if has_save_clean_x:
                    transforms.append(build_normalize_coords(transform_config, key='clean_x'))
            elif transform_str == 'NormalizeTime':
                transforms.append(build_normalize_time(transform_config))
                if has_save_clean_x:
                    transforms.append(build_normalize_time(transform_config, key='clean_x'))
            ...
```

The only change to existing code is initialising `has_save_clean_x` from the config flag instead
of hardcoding `False`. This is backward-compatible: COCO configs do not set `has_precomputed_clean_x`,
so the default `False` preserves existing behaviour.

**Note:** `NormalizeTime` for `clean_x` normalises `clean_x[2, :]` (the timestamp row). Since
`StandarizeTime` does not modify `clean_x[2]`, the normalised value will be `raw_ms / period_duration`
rather than `(raw_ms - t_0) / period_duration`. This is a known limitation — it does not affect
training because `DenoiseRegLoss` uses only `clean_x[:, :, :2]` (coordinate columns).

---

## FR3 — `configs/data/eve.yaml`

New Hydra config file. It deliberately omits `split_strategy` (no random splits for EVE — splits are
pre-defined in the bundle) and omits all noise transforms.

```yaml
# configs/data/eve.yaml
# EVE dataset configuration.
# Notable differences from configs/data/default.yaml:
#   - No split_strategy (pre-defined splits in bundle.samples_df["split"])
#   - No noise transforms (x is real WebGazer signal, not simulated noise)
#   - No ExtractRandomPeriod (valid WebGazer sequences are already short and variable-length)
#   - has_precomputed_clean_x = true (Tobii gaze supplied by EveScanpathDataset)
#   - image_W/image_H = 1920/1080 (EVE screen resolution)

dataset_type: "eve"

# Path to the directory containing bundle.h5 and stimuli/.
# Must be overridden at run time — do not commit a machine-specific path here.
bundle_dir: "MUST_OVERRIDE"

# WebGazer variant selection passed to EveScanpathDataset and EveImgDataset.
# dict form: filter rows by equality on these columns, then take lowest mean_error_px.
run_key_filter:
  max_distance: 50

# Signal to PipelineBuilder that clean_x is precomputed by the dataset.
# This causes NormalizeCoords and NormalizeTime to also normalise clean_x.
has_precomputed_clean_x: true

load:
  batch_size: 64
  use_img_dataset: true
  num_workers: 4
  persistent_workers: true
  prefetch_factor: 2
  pin_memory: true
  img_size: 256
  log: true

transforms:
  transform_list:
    - StandarizeTime
    - NormalizeCoords
    - NormalizeTime
    - NormalizeFixationCoords
    - NormalizeDuration

  StandarizeTime: {}

  NormalizeCoords:
    key: "x"
    mode: "coords"
    image_H: 1080
    image_W: 1920

  # period_duration = 90 * (1000/30) = 3000 ms — maximum possible valid-frame window.
  # Using 3000 instead of 2600 prevents normalised timestamps from exceeding 1.0 for
  # recordings with a full 90-frame valid window.
  NormalizeTime:
    key: "x"
    mode: "time"
    period_duration: 3000

  NormalizeFixationCoords:
    key: "y"
    mode: "coords"
    image_H: 1080
    image_W: 1920

  NormalizeDuration:
    key: "y"
    mode: "time"
    period_duration: 1200
```

---

## FR4 — Usage: Switching from COCO to EVE

To use EVE data, override the `data` config group in the experiment config or at the command line:

**Via a new experiment YAML** (e.g., `configs/exp/eve_experiment.yaml`):
```yaml
defaults:
  - override /data: eve
```

**Via Hydra command-line override:**
```bash
python src/training/pipeline.py data=eve data.bundle_dir=/path/to/bundle
```

The `PipelineBuilder` modifications from F4.5 (`load_dataset`, `make_splits`, `build_dataloader`,
`clear_dataframe`) handle the EVE mode once `dataset_type == "eve"` is detected. No additional
changes to the training entry point are required.

---

## FR5 — Installation Prerequisite

The `evedataset` package from the EVE repo must be installed before running:

```bash
pip install -e /path/to/EveDataset
```

This exposes `from src.evedataset import EveBundle, EveScanpathDataset, EveImgDataset` which is
imported lazily inside `_load_eve_dataset()` in `PipelineBuilder` (so COCO training does not require
the package to be installed).

---

## Data Format Contract

After the EVE transform pipeline, the batch tensors passed to the model have exactly the same format
as COCO batches:

| Key | Shape | Notes |
|---|---|---|
| `src` | `(B, T, 3)` float32 | WebGazer gaze [x_norm, y_norm, t_norm]; padded to batch max |
| `tgt` | `(B, F+1, 3)` float32 | Scanpath [x_norm, y_norm, dur_norm]; +1 for start token |
| `src_mask` | `(B, T)` bool or None | True for valid positions |
| `tgt_mask` | `(B, F+1)` bool | True for valid positions |
| `clean_x` | `(B, T, 3)` float32 | Tobii gaze [x_norm, y_norm, t_raw_norm]; coordinates in [0,1] |
| `image_src` | `(B, 3, img_size, img_size)` float32 | Stimulus image after ImageNet normalisation |
| `fixation_len` | `(B,)` int | Number of fixations per sample |

**Coordinate ranges after normalization:**
- `src[:, :, :2]`: `[0, 1]` (divided by `[1920, 1080]`)
- `tgt[:, :, :2]`: `[0, 1]` (divided by `[1920, 1080]`)
- `clean_x[:, :, :2]`: `[0, 1]` (divided by `[1920, 1080]`)
- `src[:, :, 2]`: `[0, ~1]` (divided by 3000 ms; may exceed 1 for very short sequences where first timestamp is > 0 after fill)
- `tgt[:, :, 2]`: `[0, 1]` (divided by 1200 ms; fixation durations are filtered to ≤ 1200 ms by the scanpath pipeline)

---

## Known Limitations and Non-goals

1. **No temporal augmentation for EVE:** Skipping `ExtractRandomPeriod` means each sample is used
   once per epoch at full length. For COCO, random windowing provides augmentation. Future work could
   add a custom EVE-aware period transform that handles variable-length sequences gracefully.

2. **`clean_x[2]` (timestamp column) is not standardized:** `StandarizeTime` only modifies `x[2]`.
   `clean_x[2]` will contain raw ms timestamps after `NormalizeTime` normalises them by
   `period_duration`. This is harmless because `DenoiseRegLoss` uses only `clean_x[:, :, :2]`.

3. **Tobii fill for long invalid runs:** Frames where Tobii was invalid for more than 2 consecutive
   frames use the nearest valid neighbor (forward-fill or backward-fill). Training on these positions
   pulls the model toward filled, approximate values. Since Tobii validity is ~94%, this affects a
   small fraction of frames.

4. **No test-split ground truth for some EVE subjects:** The bundle marks such rows `valid=False` in
   `samples_df`, so they are excluded by `EveScanpathDataset` automatically. The test dataset will
   contain only subjects whose Tobii data was released.
