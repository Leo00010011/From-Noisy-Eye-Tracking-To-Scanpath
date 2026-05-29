# Validation: EVE Pipeline Integration

Tests live in two places:
- **EVE repo**: extend existing `tests/test_eve_pipeline_dataset.py` with new groups covering `clean_x`.
- **Pipeline repo**: new `tests/test_eve_pipeline_integration.py` covering the config and full stack.

---

## Group 1 — `_forward_backward_fill` (EVE repo, unit tests)

- [ ] A fully-valid Tobii array (no NaN) passes through unchanged (`np.allclose`).
- [ ] An array with one interior NaN (validity[3] = False, others True) — after fill, validity[3]
  value equals linear interpolation between validity[2] and validity[4] (confirming `interpolate_invalids`
  handled it, and `_forward_backward_fill` left it as already filled).
- [ ] An array with leading NaN (validity[0:3] = False, rest True) — after fill, validity[0:3]
  values equal validity[3] (backward fill applied).
- [ ] An array with trailing NaN (validity[-3:] = False, rest True) — after fill, validity[-3:]
  values equal validity[-4] (forward fill applied).
- [ ] An all-invalid array (validity = all False) — all rows remain NaN (nothing to fill from).
  This case is handled by skipping the sample before calling `_forward_backward_fill`.
- [ ] `eye_track[2]` (timestamp row) is never modified by `_forward_backward_fill`.
- [ ] Input array is not mutated (check that the original is unchanged after the call).

---

## Group 2 — `_compute_sample` with `clean_x` (EVE repo, unit tests)

Use one real experiment from the test bundle: `"train01_step007"`, run_key with `max_distance=50`.

- [ ] `_compute_sample` returns a 4-tuple `(x, y, fixation_mask, clean_x)`.
- [ ] `clean_x.shape == (3, x.shape[1])` — same T_valid as `x`.
- [ ] `clean_x.dtype == np.float64`.
- [ ] `clean_x[0:2]` contains no NaN values.
- [ ] `clean_x[0]` values are in `[0, 1920]` — EVE pixel space x.
- [ ] `clean_x[1]` values are in `[0, 1080]` — EVE pixel space y.
- [ ] `clean_x[2, k] == x[2, k]` for all k — timestamp rows are identical.
- [ ] For at least 80% of positions k, the nearest Tobii frame to WebGazer timestamp `x[2, k]` is
  within 16.7 ms (half a 30 Hz WebGazer period). This sanity check can be computed manually:
  ```python
  diffs = [abs(tobii_ts - x[2,k]/1000) for k, ... ]
  assert np.mean(np.array(diffs) < 0.0167) >= 0.8
  ```
- [ ] The spatial distance between `clean_x[0:2, k]` and `x[0:2, k]` is non-zero for most k
  (Tobii ≠ WebGazer — confirms they are different signals, not the same copied twice).
- [ ] **Edge case**: an experiment where `eye_track_validity.sum() == 0` → `clean_x` is not
  computed (the sample is skipped before calling `_compute_sample`). Verify this by calling
  `_compute_sample` with a synthetic all-False validity array and checking that the caller skips it.

---

## Group 3 — `EveScanpathDataset` with `clean_x` (EVE repo)

- [ ] `EveScanpathDataset(bundle, split="train", run_key_filter={"max_distance": 50}, transforms=[])`
  constructs without error.
- [ ] `"clean_x"` is a key in `dataset.data_store`.
- [ ] `len(dataset.data_store["clean_x"]) == len(dataset)`.
- [ ] `dataset[0]` returns a dict containing key `"clean_x"`.
- [ ] `dataset[0]["clean_x"]` is a 2D numpy array of dtype float64 with shape `(3, T)` where
  `T == dataset[0]["x"].shape[1]` — same T as x.
- [ ] `dataset[0]["clean_x"][0:2]` contains no NaN.
- [ ] Mutating `dataset.data_store["clean_x"][0]` does NOT affect `dataset[0]["clean_x"]` — the
  `.copy()` in `__getitem__` is effective.
- [ ] An experiment with `eye_track_validity` all-False is NOT included in the dataset (length is
  reduced by the number of such experiments in the split).

---

## Group 4 — `clean_x` survives transforms (EVE repo)

Construct `EveScanpathDataset` with the full EVE transform pipeline:
```python
transforms = [
    StandarizeTime(),
    Normalize(key='x', mode='coords', max_value=torch.tensor([1920, 1080])),
    Normalize(key='x', mode='time',   max_value=3000),
    Normalize(key='clean_x', mode='coords', max_value=torch.tensor([1920, 1080])),
    Normalize(key='clean_x', mode='time',   max_value=3000),
    Normalize(key='y', mode='coords', max_value=torch.tensor([1920, 1080])),
    Normalize(key='y', mode='time',   max_value=1200),
]
```

- [ ] `dataset[0]` does not raise.
- [ ] After transforms, `dataset[0]["clean_x"][0:2]` values are in `[0, 1]`.
- [ ] After transforms, `dataset[0]["x"][0:2]` values are in `[0, 1]`.
- [ ] After `StandarizeTime`, `dataset[0]["x"][2, 0] == 0.0` (relative to first frame).
- [ ] `dataset[0]["clean_x"]` still has no NaN.

---

## Group 5 — `has_precomputed_clean_x` flag in `PipelineBuilder` (pipeline repo)

Load a minimal OmegaConf config with `data.has_precomputed_clean_x = true` and a transform list
of `['NormalizeCoords', 'NormalizeTime', 'NormalizeFixationCoords', 'NormalizeDuration']`.

- [ ] `builder._build_transforms()` returns a transform list that includes two `Normalize` instances
  with `key='clean_x'` (one for coords, one for time).
- [ ] Setting `data.has_precomputed_clean_x = false` (or omitting the key) returns a transform list
  with NO `Normalize` instance with `key='clean_x'`.
- [ ] A config with `SaveCleanX` in the transform list AND `has_precomputed_clean_x = false` still
  produces `clean_x` normalisation transforms (regression: the original COCO path must still work).
- [ ] A config with both `SaveCleanX` AND `has_precomputed_clean_x = true` does NOT produce duplicate
  `clean_x` normalisation (the flag sets the initial value to True; `SaveCleanX` also sets it to True —
  only one set of normalisation transforms should be emitted).

---

## Group 6 — `configs/data/eve.yaml` correctness (pipeline repo)

Load `configs/data/eve.yaml` with `OmegaConf.load()` or via a minimal Hydra setup.

- [ ] `cfg.data.dataset_type == "eve"`.
- [ ] `cfg.data.has_precomputed_clean_x == True`.
- [ ] `cfg.data.transforms.transform_list` does NOT contain `"ExtractRandomPeriod"`.
- [ ] `cfg.data.transforms.transform_list` does NOT contain `"SaveCleanX"`.
- [ ] `cfg.data.transforms.transform_list` does NOT contain any noise transform
  (`"AddIsotropicGaussianNoise"`, `"AddRandomCenterCorrelatedRadialNoise"`, `"DiscretizationNoise"`).
- [ ] `cfg.data.transforms.NormalizeCoords.image_W == 1920`.
- [ ] `cfg.data.transforms.NormalizeCoords.image_H == 1080`.
- [ ] `cfg.data.transforms.NormalizeTime.period_duration == 3000`.
- [ ] `cfg.data.transforms.NormalizeDuration.period_duration == 1200`.

---

## Group 7 — Full EVE pipeline stack (pipeline repo, integration test)

This group requires a real mini-bundle with at least 2 train experiments and 1 val experiment, each
with valid Tobii data.

- [ ] Construct `PipelineBuilder` with `data=eve data.bundle_dir=/path/to/mini_bundle`.
  Call `load_dataset()` — it must not raise.
- [ ] `builder.PathDataset` is a `EveScanpathDataset` instance.
- [ ] `builder.train_dataset`, `builder.val_dataset`, `builder.test_dataset` are all
  `EveScanpathDataset` instances with non-zero length for train and val.
- [ ] `builder.train_img_dataset`, `builder.val_img_dataset` are `EveImgDataset` instances.
- [ ] Call `make_splits()` — returns three tensors `(train_idx, val_idx, test_idx)`.
  `train_idx` equals `torch.arange(len(builder.train_dataset))`.
- [ ] Call `build_dataloader(train_idx, val_idx, test_idx)` — returns three `CoupledDataloader`
  instances without error.
- [ ] Iterate one batch from the train dataloader — it must not raise.
- [ ] `batch["src"]` shape is `(B, L, 3)` with dtype `float32`.
- [ ] `batch["tgt"]` shape is `(B, F_max+1, 3)` with dtype `float32`.
- [ ] `"clean_x"` is present in the batch dict.
- [ ] `batch["clean_x"]` shape is `(B, L, 3)` with dtype `float32`.
- [ ] `batch["clean_x"][:, :, 0:2]` values are all in `[0, 1]` (after NormalizeCoords).
- [ ] No NaN values in `batch["src"]`, `batch["tgt"]`, or `batch["clean_x"]`.
- [ ] `batch["image_src"]` shape is `(B, 3, img_size, img_size)`.

---

## Group 8 — Regression: COCO path unaffected (pipeline repo)

Construct `PipelineBuilder` with the COCO default config (`data=default`) and confirm:

- [ ] `builder._build_transforms()` does not produce any `Normalize(key='clean_x')` transform
  when neither `SaveCleanX` nor `has_precomputed_clean_x` is set.
- [ ] The full COCO `load_dataset()` → `make_splits()` → `build_dataloader()` cycle completes
  without error and produces the same types as before this feature (`FreeViewInMemory`,
  `DeduplicatedMemoryDataset`, `CoupledDataloader`).
- [ ] A COCO batch obtained with `SaveCleanX` in the transform list contains `"clean_x"` with
  values equal to the pre-noise gaze (i.e., the original COCO Denoise path is intact).

---

## Manual Data Validity Checks (required before marking complete)

These are not automated tests — run them in a notebook once before merging.

### Check A — `clean_x` vs WebGazer overlay

For 3 train experiments with `calibration_valid=True`:
1. Call `dataset[i]` (with no transforms beyond coord normalisation) to get `x` and `clean_x`.
2. Plot `x[0:2]` (WebGazer) and `clean_x[0:2]` (Tobii) on the same axis, colour-coded.
3. **Expected:** The Tobii trace is smooth and clustered; the WebGazer trace is noisy and offset.
   Both traces should cover the same approximate region of the `[0,1]²` normalised space.
4. **Failure mode:** If the two traces occupy completely different regions, the timestamp alignment
   or coordinate space mapping is wrong.

### Check B — Denoise loss does not NaN

Run 10 training batches with the Denoise phase active (`training.Phases: ["Denoise"]`):
1. Inspect the `denoise_loss` value in the training log after each step.
2. **Expected:** Loss decreases from step 1 to step 10, no NaN values.
3. **Failure mode:** NaN loss indicates that `clean_x` contains NaN (fill logic failed) or that
   the loss function received a tensor with wrong dtype.

### Check C — Fixation prediction after transform pipeline

For 5 samples, verify that after the full transform pipeline:
- `y[2, :]` (duration_norm) are all in `[0, 1]` (duration in [0, 1200 ms] divided by 1200).
- `y[0:2, :]` (fixation coord_norm) are all in `[0, 1]`.
- `x[2, :]` (time_norm) are monotonically non-decreasing and start near 0.
