# Plan: EVE Pipeline Integration

## Context and Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Period extraction | Skip `ExtractRandomPeriod` | Valid WebGazer sequences are ≤90 frames (≤3 s). A 2600 ms window would require 78 frames and crash for shorter recordings. Full sequence is passed directly. |
| Noise transforms | Omit all noise transforms | `x` is real WebGazer signal; adding simulation noise on top of real noise is incorrect and would corrupt the denoising target. |
| `clean_x` source | Tobii gaze from `bundle.get_eye_track()` | This is the ground-truth hardware signal aligned to the same timestamps as WebGazer. It enables the Denoise training phase without simulation. |
| Tobii alignment | Nearest-frame lookup by absolute timestamp | WebGazer (30 Hz) and Tobii/basler (60 Hz) are not synchronised; nearest-frame interpolation is the standard approach for mismatched rates. |
| Invalid Tobii frames | Forward-fill then backward-fill, then skip if all-invalid | Matches the fill logic in `interpolate_invalids` from `scanpath_tools.py`. Keeps `clean_x` NaN-free for all accepted samples. |
| `clean_x[2]` (timestamp) | Set to same raw ms as `x[2]`; normalisation accuracy not enforced | `DenoiseRegLoss` uses only `clean_x[:,:,:2]`. Timestamp column accuracy is irrelevant for training. |
| `has_precomputed_clean_x` flag | New config key; initialises `has_save_clean_x = True` before the transform loop | Avoids inserting `SaveCleanX` (which would overwrite the Tobii clean_x with the noisy x). Minimal change to `_build_transforms`. |
| COCO path preservation | All changes guarded by `has_precomputed_clean_x` default `False` | Zero risk of regression on existing COCO training runs. |

---

## Pre-conditions (verify before coding)

Before writing any code, verify the following with a real bundle file:

1. **`bundle.get_eye_track(exp_key)` API exists and returns correct data:**  
   Open a bundle, call `bundle.get_eye_track("train01_step007")`. Confirm:
   - Returns a dict with keys `"gaze"` and `"validity"`.
   - `gaze.shape == (3, 180)` for a basler-mode bundle.
   - `gaze[2, :]` contains timestamps in seconds that fall within the stimulus window.

2. **WebGazer and Tobii timestamps overlap:**  
   For one experiment, compare `gaze_pred[2, valid_frames]` (WebGazer timestamps in seconds) with
   `eye_track["gaze"][2, :]` (Tobii timestamps in seconds). Confirm that for at least 80% of valid
   WebGazer frames, there exists a Tobii frame within `1/(2*60)` seconds (half a Tobii frame period =
   8.3 ms). A larger mismatch indicates a timestamp alignment problem.

3. **`clean_x` sanity check:**  
   After running `_compute_sample` with the extended signature on one experiment, confirm:
   - `clean_x.shape == (3, validity.sum())` (same T_valid as x).
   - `clean_x[0:2]` contains no NaN.
   - `clean_x[0:2]` values are in `[0, 1920]` x `[0, 1080]`.
   - Plotting `clean_x[0:2]` and `x[0:2]` overlaid: the Tobii trace should be smoother and the
     WebGazer trace noisier. Both should cover the same spatial region of the stimulus.

Document these checks in a comment block at the top of the modified `dataset.py`.

---

## Step 1 — Extend `_compute_sample` in `src/evedataset/dataset.py` (EVE repo)

Add `_forward_backward_fill` as a module-level private helper, then update `_compute_sample` to
accept `eye_track` and `eye_track_validity` and return a fourth element `clean_x`.

```python
def _forward_backward_fill(
    eye_track: np.ndarray,   # (3, T) float32 or float64
    validity: np.ndarray,    # (T,) bool
) -> np.ndarray:
    """NaN-fill eye_track[0] and [1] using forward-fill then backward-fill.

    eye_track[2] (timestamps) is never modified.
    Returns a copy; does not mutate the input.
    """
    out = eye_track.copy()
    for row in (0, 1):
        series = out[row].astype(np.float64)
        # forward fill
        last_valid_val = None
        for i in range(len(series)):
            if validity[i]:
                last_valid_val = series[i]
            elif last_valid_val is not None:
                series[i] = last_valid_val
        # backward fill (handles leading NaN)
        last_valid_val = None
        for i in range(len(series) - 1, -1, -1):
            if validity[i]:
                last_valid_val = series[i]
            elif last_valid_val is not None:
                series[i] = last_valid_val
        out[row] = series
    return out
```

Update `_compute_sample`:

```python
def _compute_sample(
    gaze_pred: np.ndarray,          # (3, 90) float32
    validity: np.ndarray,           # (90,) bool
    scanpath: np.ndarray,           # (4, F) float32
    eye_track: np.ndarray,          # (3, T_basler) float32
    eye_track_validity: np.ndarray, # (T_basler,) bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # ---- existing x, y, fixation_mask computation (unchanged from F4.5) ----
    valid_frames = np.where(validity)[0]

    x = np.empty((3, len(valid_frames)), dtype=np.float64)
    x[0] = gaze_pred[0, valid_frames]
    x[1] = gaze_pred[1, valid_frames]
    x[2] = gaze_pred[2, valid_frames] * 1000.0

    y = np.array([
        scanpath[2].astype(np.float64),
        scanpath[3].astype(np.float64),
        scanpath[1].astype(np.float64),
    ])

    fixation_mask = np.zeros(len(valid_frames), dtype=np.uint8)
    F = scanpath.shape[1]
    for k, frame_idx in enumerate(valid_frames):
        t_k = float(gaze_pred[2, frame_idx])
        for j in range(F):
            start_j = float(scanpath[0, j])
            end_j   = start_j + float(scanpath[1, j]) / 1000.0
            if start_j <= t_k < end_j:
                fixation_mask[k] = j + 1
                break

    # ---- new: clean_x from Tobii ----
    # Interpolate short invalid runs (max_run=2 matches the basler scanpath pipeline).
    tobii, tobii_validity = interpolate_invalids(
        eye_track.astype(np.float64), eye_track_validity, max_run=2)
    # Fill remaining NaN (long runs, leading invalids) with nearest valid neighbor.
    tobii = _forward_backward_fill(tobii, tobii_validity)

    tobii_timestamps = tobii[2, :]   # seconds
    clean_x = np.empty((3, len(valid_frames)), dtype=np.float64)
    for k, frame_idx in enumerate(valid_frames):
        t_k       = float(gaze_pred[2, frame_idx])    # seconds
        nearest_j = int(np.argmin(np.abs(tobii_timestamps - t_k)))
        clean_x[0, k] = tobii[0, nearest_j]           # x_px
        clean_x[1, k] = tobii[1, nearest_j]           # y_px
        clean_x[2, k] = t_k * 1000.0                  # ms — same as x[2, k]

    return x, y, fixation_mask, clean_x
```

**Import required at the top of `dataset.py`:**
```python
from src.scanpath_tools import interpolate_invalids
```

---

## Step 2 — Update `EveScanpathDataset.__init__` (EVE repo)

Inside the pair-loading loop:

```python
for exp_key, run_key in pairs:
    wg              = bundle.get_webgazer(exp_key, run_key)
    scanpath        = bundle.get_scanpath(exp_key)
    eye_track_data  = bundle.get_eye_track(exp_key)         # NEW
    gaze_pred       = wg["gaze_pred"]
    validity        = wg["validity"]
    et_gaze         = eye_track_data["gaze"]                 # (3, T_basler) float32
    et_validity     = eye_track_data["validity"]             # (T_basler,) bool

    # Skip if no valid Tobii frames — cannot compute clean_x  # NEW
    if et_validity.sum() == 0:
        skipped += 1
        continue

    x, y, mask, clean_x = _compute_sample(
        gaze_pred, validity, scanpath, et_gaze, et_validity)  # CHANGED signature

    if x.shape[1] == 0 or y.shape[1] == 0:
        skipped += 1
        continue

    xs.append(x); ys.append(y); masks.append(mask)
    clean_xs.append(clean_x)                                 # NEW
    exp_keys.append(exp_key); run_keys.append(run_key)
```

At the top of `__init__`, initialise `clean_xs`:
```python
xs, ys, masks, clean_xs, exp_keys, run_keys = [], [], [], [], [], []
```

Update `data_store`:
```python
self.data_store = {
    'x':             xs,
    'y':             ys,
    'fixation_mask': masks,
    'clean_x':       clean_xs,    # NEW
    'exp_keys':      exp_keys,
    'run_keys':      run_keys,
}
```

---

## Step 3 — Update `EveScanpathDataset.__getitem__` (EVE repo)

```python
def __getitem__(self, i: int) -> dict:
    x             = self.data_store['x'][i].copy()
    y             = self.data_store['y'][i].copy()
    fixation_mask = self.data_store['fixation_mask'][i].copy()
    clean_x       = self.data_store['clean_x'][i].copy()    # NEW
    inp = {'x': x, 'y': y, 'fixation_mask': fixation_mask, 'clean_x': clean_x}  # CHANGED
    for t in self.transforms:
        inp = t(inp)
    out = {'x': inp['x'], 'y': inp['y'], 'sample_idx': i}
    for k in ('clean_x', 'in_tgt', 'down_offset', 'heatmaps'):
        if k in inp:
            out[k] = inp[k]
    return out
```

The `clean_x` key is always present in `inp` (from `data_store`), so it will always appear in `out`.

---

## Step 4 — Update `_build_transforms()` in `pipeline_builder.py` (pipeline repo)

First, extract the transform-building logic into `_build_transforms()` as described in
`pipeline_builder_integration.md` from F4.5. Then apply this one-line modification:

```python
def _build_transforms(self) -> list:
    transforms = []
    # CHANGED: initialise from config instead of hardcoding False
    has_save_clean_x = bool(getattr(self.config.data, 'has_precomputed_clean_x', False))
    has_in_tgt = False
    if hasattr(self.config.data, 'transforms'):
        for transform_str in self.config.data.transforms.transform_list:
            transform_config = self.config.data.transforms.get(transform_str)
            if transform_str == 'ExtractRandomPeriod':
                transforms.append(build_extract_random_period(transform_config))
            ...
            elif transform_str == 'SaveCleanX':
                transforms.append(SaveCleanX())
                has_save_clean_x = True          # also supports COCO path
            elif transform_str == 'NormalizeCoords':
                transforms.append(build_normalize_coords(transform_config))
                if has_save_clean_x:
                    transforms.append(build_normalize_coords(transform_config, key='clean_x'))
            elif transform_str == 'NormalizeTime':
                transforms.append(build_normalize_time(transform_config))
                if has_save_clean_x:
                    transforms.append(build_normalize_time(transform_config, key='clean_x'))
            ...
    else:
        transforms = [...]
    return transforms
```

**This is the only change to `_build_transforms()`** relative to the refactored version described
in `pipeline_builder_integration.md`. All other logic (EVE mode detection, split handling, dataloader
construction) was already specified in F4.5.

---

## Step 5 — Create `configs/data/eve.yaml` (pipeline repo)

Write the file exactly as specified in FR3 of `requirements.md`. No other config files need to change.

To use it, pass `data=eve data.bundle_dir=/path/to/bundle` on the Hydra command line, or add
`- override /data: eve` to an experiment config's defaults list.

---

## Step 6 — Write `tests/test_eve_pipeline_integration.py` (pipeline repo)

See `validation.md` for the full test list. Fixtures:

```python
@pytest.fixture(scope="session")
def bundle(mini_bundle_dir):
    from src.evedataset import EveBundle
    return EveBundle.load(mini_bundle_dir)

@pytest.fixture(scope="session")
def eve_dataset(bundle):
    from src.evedataset import EveScanpathDataset
    return EveScanpathDataset(bundle, split="train", run_key_filter={"max_distance": 50})

@pytest.fixture(scope="session")
def eve_img_dataset(bundle):
    from src.evedataset import EveImgDataset
    return EveImgDataset(bundle, split="train", run_key_filter={"max_distance": 50}, resize_size=64)
```

---

## Implementation Order

1. **Pre-condition verification** (Step 0): open a real bundle in a notebook, check timestamps overlap and `get_eye_track` API.
2. **`_forward_backward_fill`** (Step 1a): write and unit-test with a hand-crafted array.
3. **`_compute_sample` extension** (Step 1b): update signature and add `clean_x` block.
4. **`EveScanpathDataset.__init__` and `__getitem__`** (Steps 2–3): wire `clean_x` into the dataset.
5. **Re-run F4.5 tests** (`tests/test_eve_pipeline_dataset.py`): confirm no regression from the signature change.
6. **`_build_transforms()` refactor and flag** (Step 4): extract transform builder, add `has_precomputed_clean_x`.
7. **`configs/data/eve.yaml`** (Step 5): create config file.
8. **Integration tests** (Step 6): write and run.

Do NOT modify:
- `CoupledDataloader`, `seq2seq_padded_collate_fn`, or any transform class.
- `DenoiseRegLoss` or any loss function.
- `StandarizeTime` — it does not need to handle `clean_x`.
- Any model class.

---

## Notes on `_build_transforms` Refactoring

The F4.5 `pipeline_builder_integration.md` describes extracting the transform-building loop into
`_build_transforms()`. If that refactoring has not yet been done, it must be completed as part of
this feature (Step 4 above). The EVE path (`_load_eve_dataset`) calls `_build_transforms()`, so the
refactoring is a prerequisite.

If F4.5 pipeline builder changes have already been applied (i.e., `_build_transforms` already exists
as a private method), Step 4 reduces to the single-line initialisation change shown above.

---

## Key Files

| File | Repo | Action |
|---|---|---|
| `src/evedataset/dataset.py` | EVE repo | Add `_forward_backward_fill`; extend `_compute_sample`; update `__init__` and `__getitem__` |
| `src/training/pipeline_builder.py` | Pipeline repo | Add `has_precomputed_clean_x` flag; refactor `_build_transforms` if not done |
| `configs/data/eve.yaml` | Pipeline repo | Create |
| `tests/test_eve_pipeline_integration.py` | Pipeline repo | Create |
