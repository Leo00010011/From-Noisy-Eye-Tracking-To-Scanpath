# Validation: EVE Real-Noise Scanpath Inference

Reference numbers below were measured on the production bundle at
`eve_shared/EveDataset/bundle/bundle.h5` and the 70,858-row `predictions.csv` before implementation
started. They are targets, not post-hoc descriptions — a deviation is a bug until explained.

| Quantity | Expected |
|---|---|
| Cached experiments | 734 |
| `eyenet_split` breakdown | `val`: 474 experiments / 8 subjects / 22,741 frames; `test`: 260 / 5 / 12,688 |
| Experiments spanning both EyeNet splits | 0 |
| Valid frames total | 35,429 |
| Valid frames per experiment | mean 48.27, median 50, min 5, max 50 |
| Median `‖gaze_px − gt_gaze_px‖` (combined) | 89.6 px (2.32 DVA) |
| Mean / p90 combined error | 109.9 px / 210.7 px |
| Median single-eye error | ≈ 149 px |
| Median left–right disagreement | 180.4 px |
| Frames dropped by projection validity | 0 |
| Off-screen combined predicted frames | 0.0% |
| Off-screen ground-truth frames | 0.065% |
| CSV `target_*` projection vs `get_screen_intercept` | ≤ 1.3e-4 px |

---

## Code Correctness

### Group 1 — CSV parsing and eye combination (pure functions, no I/O)

- [ ] `load_eyenet_predictions` on a well-formed 12-row synthetic CSV returns a DataFrame with 11
      columns, `frame` dtype `int32`, and the 7 numeric columns dtype `float32`.
- [ ] Dropping the `pred_z` column raises `ValueError` whose message contains `"pred_z"`.
- [ ] Duplicating one `(exp_key, frame, patch)` row raises `ValueError` containing `"duplicate"`.
- [ ] A row with `patch="center"` raises `ValueError` containing `"center"`.
- [ ] A row with `frame=90` raises `ValueError`; `frame=89` does not.
- [ ] A vector of norm 1.05 triggers exactly one `warnings.warn` and does **not** raise; the returned
      `pred_z` value is unchanged (no re-normalization).
- [ ] `_combine_eyes` with both eyes valid at frame `t` returns the exact arithmetic mean:
      `left=(100,200)`, `right=(300,400)` → `(200.0, 300.0)`, `validity[t] is True`,
      `np.allclose(..., atol=1e-6)`.
- [ ] `_combine_eyes` with only the left eye valid returns the left value verbatim and
      `validity[t] is True`; only right → right value; neither → `np.isnan(out[t]).all()` and
      `validity[t] is False`.
- [ ] `_combine_eyes` output dtype is `float32` and shape `(90, 2)`; validity shape `(90,)` dtype
      `bool`.

### Group 2 — Projection correctness against the real bundle

Skipped when the production bundle is absent.

- [ ] For 20 randomly chosen `exp_key`s, projecting the CSV `target_*` buffer through
      `project_normalized_gaze(..., spherical=False)` matches
      `bundle.get_screen_intercept(exp_key, eye)["hit_px"]` on all CSV frames to
      `atol=1e-3` px. Failure here means the scatter or the eye argument is wrong, and every
      downstream number is meaningless.
- [ ] For the same 20 keys, `frame_mask & result["validity"] == frame_mask` — the projection drops no
      CSV frame. Expected: exact equality for all 734 experiments.
- [ ] The sentinel-filled frames (`_FILL_DIRECTION` positions) never appear in `gaze_px`:
      `np.isnan(gaze_px[~validity]).all()` is `True`.
- [ ] Changing `_FILL_DIRECTION` to `[0, 0, 1]` leaves `gaze_px[validity]` bit-identical — proving
      the fill value is genuinely unread.
- [ ] `‖gaze_px − gt_gaze_px‖` over all valid frames of the 20 keys has median in `[70, 110]` px.
      Outside that band, suspect the eye-averaging or the wrong `eye` patch.

### Group 3 — `EyeNetGazeCache` build, save, and roundtrip

- [ ] `build` on the 3-experiment synthetic CSV + `FakeBundle` returns `skipped == []` and a cache
      with `len(cache.exp_keys) == 3`, keys sorted ascending.
- [ ] `save` then `load` roundtrips every dataset bit-exactly:
      `np.array_equal(loaded.get_gaze(k), built.get_gaze(k), equal_nan=True)` for all 3 keys, and
      likewise for `validity` (dtype `bool`), `gt_gaze_px`, `left_px`, `right_px`,
      `angular_error_deg`.
- [ ] Array shapes and dtypes after load: `gaze_px (3, 90, 2) float32`, `validity (3, 90) bool`,
      `angular_error_deg (3, 90, 2) float32`, `exp_keys (3,)` decoded to `str` (not `bytes`).
- [ ] Group attributes survive the roundtrip: `timestamp_source == "synthesized_30hz"`,
      `center_fps == 30.0`, and `n_offscreen` is an `int`.
- [ ] `splits_df` has columns `['exp_key', 'eyenet_split', 'eve_split', 'stimulus_name']`;
      `eyenet_split` values come from the CSV and `eve_split` from the fake bundle, and the two
      differ for at least one row in the fixture — proving they are read from different sources and
      not aliased.
- [ ] A synthetic CSV where one `exp_key` carries both `split="val"` and `split="test"` rows raises
      `ValueError` during `build` naming the key (a key must have exactly one EyeNet split).
- [ ] An `exp_key` present in the CSV but absent from `FakeBundle.samples_df` appears in `skipped`
      as `(key, "not_in_bundle")` and is absent from `cache.exp_keys` — it does **not** raise.
- [ ] A `FakeBundle` whose `has_gaze_ray` returns `False` for one key yields
      `(key, "no_gaze_ray")` in `skipped`.
- [ ] A `FakeBundle` whose `project_normalized_gaze` raises `KeyError` for one key yields a
      `skipped` entry whose reason starts with `"projection_failed"`.
- [ ] `load` on a nonexistent path raises `FileNotFoundError`.
- [ ] `load` on an HDF5 file containing only an unrelated `/other` group raises `ValueError`
      mentioning `"eyenet_gaze"`.
- [ ] An HDF5 file whose `exp_keys` contains a duplicate raises `ValueError` on `load`.
- [ ] `get_gaze("nope")` raises `KeyError` whose message contains `"nope"`.
- [ ] `get_eye_gaze(k, "middle")` raises `ValueError` before any key lookup.
- [ ] `verify(bundle)` returns `[]` for a consistent cache, and `[(key, ...)]` — not an exception —
      for a cache row whose `eve_split` was hand-edited to disagree with the bundle.
- [ ] `verify(bundle)` flags a cache row whose `eyenet_split` was hand-edited to `"train"`, since
      that would mean EyeNet training data entered the artifact.

### Group 4 — HDF5 isolation

- [ ] Write a decoy group `/decoy` with a `(5,)` int32 dataset into the cache file, then call
      `cache.save(same_path)`. Reopen: `/decoy` still exists with identical contents, and
      `/eyenet_gaze` holds the new data. Confirms the append-mode + single-group-delete convention.
- [ ] Calling `save` twice in a row leaves exactly one `/eyenet_gaze` group with `N` rows (not `2N`)
      — the delete-then-recreate is idempotent.
- [ ] `RealNoiseInferenceStore.save` uses mode `"w"` and therefore *does* replace the whole file;
      assert that a pre-existing `/decoy` group in the **output** file is gone, documenting the
      deliberate difference from the cache's append semantics.

### Group 5 — Dataset construction and the index invariant

- [ ] `EveRealNoiseDataset` on the 3-key fake cache produces `len == 3`; `x` has shape
      `(3, T_valid)` dtype `float64`; `y` has shape `(3, max_fixations)` and
      `np.all(y == PAD_TOKEN_ID)`.
- [ ] `x[2]` equals `frames * (1000/30)` exactly: for contiguous frames 40…44,
      `np.allclose(x[2], [1333.333, 1366.667, 1400.0, 1433.333, 1466.667], atol=1e-3)`.
- [ ] `x[:2]` contains no `NaN` for any sample. A cache row with `validity[t] == True` but
      `gaze_px[t] == NaN` raises `ValueError` naming the `exp_key`.
- [ ] A cache row with 4 valid frames is skipped at `min_valid_frames=5` and kept at
      `min_valid_frames=4`.
- [ ] With `eyenet_split="val"`, only rows whose cached `eyenet_split` is `"val"` are present, in
      cache order; `eyenet_split="test"` likewise; `None` yields both. The three lengths sum
      correctly (`len(val) + len(test) == len(None)`).
- [ ] `eyenet_split="train"` raises `ValueError` whose message names the accepted values — an EVE
      split label is rejected rather than silently producing an empty dataset.
- [ ] Filtering selects on `eyenet_split`, not `eve_split`: build a fixture where the two disagree
      for every row, request `eyenet_split="val"`, and assert the returned `exp_key`s are exactly the
      CSV-val ones. On the production cache this is the difference between 474 experiments and 0.
- [ ] `__getitem__` with an empty transform list returns exactly the keys
      `{'x', 'y', 'sample_idx'}` — `clean_x` is absent, confirming the Denoise path is not silently
      half-wired.
- [ ] **Index invariant**: for every `i`, `gaze_ds.exp_key_at(i) == img_ds.exp_key_at(i)`, and
      `len(gaze_ds) == len(img_ds)`, for `eyenet_split` in `{None, "val", "test"}` and
      `min_valid_frames` in `{1, 5, 40}`. This is the single check that positional coupling between
      the two datasets cannot drift.
- [ ] `EveRealNoiseImgDataset.__getitem__(0)` returns a 3-tuple; the image tensor has shape
      `(3, 256, 256)`, dtype `uint8` when `transform=None` and `float32` after
      `PipelineBuilder.make_transform(256)`.
- [ ] Two experiments sharing a `stimulus_name` map to the same `unique_img_idx`, and
      `image_bank.shape[0] == n_unique_stimulus_names` (< number of samples).
- [ ] `seq2seq_padded_collate_fn([ds[0], ds[1]])` returns `src (2, T_max, 3)`,
      `tgt (2, max_fixations, 3)`, `tgt_mask (2, max_fixations + 1)` — confirming the decode-step
      budget is `max_fixations + 1` as FR7.4 claims.

### Group 6 — `RealNoiseInferenceStore`

- [ ] `save` → `load` roundtrips `pred_scanpath (N, K, 3) float32`, `eos_logit (N, K) float32`,
      `pred_len (N,) int32`, `src_px (N, T, 3) float32`, `src_len (N,) int32`,
      `frame_indices (N, T) int32` bit-exactly (`equal_nan=True` for the float arrays).
- [ ] Records of differing `src_len` are padded to `T = max(src_len)` with `NaN` in `src_px` and
      `-1` in `frame_indices`; `get(exp_key)["src_px"]` is trimmed to that row's `src_len` and
      contains no `NaN`.
- [ ] A record missing `exp_key` raises `ValueError`; two records with the same `exp_key` raise
      `ValueError` containing the key.
- [ ] Records with differing `pred_scanpath.shape[0]` raise `ValueError` — a mixed `max_fixations`
      run is rejected rather than silently truncated.
- [ ] A record list where only some entries carry `denoise_px` raises `ValueError`; where none do,
      `has_denoise is False` and the dataset is absent from the file; where all do,
      `has_denoise is True`.
- [ ] `get_scanpath(exp_key)` returns shape `(pred_len, 3)`, and `pred_scanpath` in the file still
      has all `K` rows (FR9.3 — no destructive truncation).
- [ ] `get("nope")` raises `KeyError`; `load` on a file with duplicate `exp_keys` raises
      `ValueError`.
- [ ] `store.df` has exactly the columns
      `['exp_key', 'eyenet_split', 'eve_split', 'pred_len', 'src_len']` and `len(df) == N`.

### Group 7 — End-to-end inference driver

Run against the production bundle and a real checkpoint.

- [ ] `save_predictions_eve_real.py` on a checkpoint whose `.hydra/config.yaml` has
      `img_size: 512` raises `ValueError` mentioning both `512` and `256`, before any model weight is
      loaded (FR8.5). On a checkpoint with `img_size: 256` it proceeds.
- [ ] Running with `eyenet_split="val"` and then `eyenet_split="test"` writes two stores whose
      `exp_key` sets are disjoint and whose union equals the `eyenet_split=None` run's set
      (474 + 260 = 734 on the production cache).
- [ ] The number of records written equals `len(gaze_ds)`; no `exp_key` appears twice (FR11.5).
- [ ] Set of `exp_keys` in the output store == set of `exp_keys` the dataset accepted from the cache.
      A mismatch means `drop_last_batch` or `shuffle` regressed.
- [ ] Running the driver twice with the same checkpoint and seed produces byte-identical
      `pred_scanpath` arrays.
- [ ] With a checkpoint that has no denoise head, the run completes and `has_denoise is False`; with
      one that does, `denoise_px` is present with shape `(N, T, 2)`.

---

## Data Validity

These are notebook / build-report checks on the real artifacts, not pytest assertions. Each states
the outcome that counts as passing.

### Cache-level

- [ ] **Coverage.** `len(cache.exp_keys) == 734` and `skipped == []`. Any skip is a coverage
      regression: report it by reason before proceeding.
- [ ] **Frame counts.** Valid frames per experiment: mean 48.27 ± 4.76, median 50, min 5, max 50;
      total 35,429. A total below ~35,000 means the frame mask is over-restrictive.
- [ ] **Error magnitude.** Median `‖gaze_px − gt_gaze_px‖` = 89.6 px, mean 109.9 px, p90 210.7 px.
      Dividing the median by 38.55 px/DVA gives 2.32 DVA. Cross-check: the CSV's own mean
      `angular_error_deg` is 4.09°, and the single-eye screen error is ≈149 px ≈ 3.87 DVA — the
      angular and screen-space measures must agree to within ~10% for the single-eye case, which is
      the strongest available confirmation that the projection geometry is right.
- [ ] **Averaging actually helps.** Median combined error (89.6 px) is materially below median
      single-eye error (≈149 px). If they are equal, `_combine_eyes` is falling through to a single
      eye.
- [ ] **Left–right disagreement.** Median `‖left_px − right_px‖` = 180.4 px, of the same order as the
      per-eye error — consistent with two largely independent noisy estimates. A disagreement near
      0 would mean the same eye was projected twice.
- [ ] **Spatial distribution.** Combined `gaze_px` percentiles: x `[453, 980, 1471]` and y
      `[190, 483, 876]` at the 0.5 / 50 / 99.5 marks. The medians should sit near screen center
      (960, 540); a systematic offset indicates a bad camera→screen rotation.
- [ ] **Off-screen rate.** 0.0% of valid combined predicted frames outside `[0,1920)×[0,1080)`;
      0.065% for ground truth. A predicted rate above ~1% means the projection or the eye origin is
      wrong.
- [ ] **Split composition.** `cache.splits_df["eyenet_split"].value_counts()` == `{val: 474,
      test: 260}`, subject-disjoint (8 subjects in val, 5 in test, no overlap). No row has
      `eyenet_split == "train"` — the artifact must contain zero EyeNet training data, which is the
      whole point of using it as a test set for the recovery model.
- [ ] **The two split labels are independent.** `eyenet_split` disagrees with `eve_split` on ~100% of
      rows (CSV `val` ↔ EVE `train`, CSV `test` ↔ EVE `val`). This is expected and confirms the two
      are read from different sources. If they ever agree on most rows, one has been overwritten by
      the other.
- [ ] **Per-split error parity.** Report median `‖gaze_px − gt_gaze_px‖` separately for `val` and
      `test`. A large gap between them would mean EyeNet generalizes unevenly across its two held-out
      groups, which changes how the recovery model's results should be read per split.
- [ ] **Independent clean reference.** For 20 experiments, compare `gt_gaze_px` against the Tobii
      point-of-gaze in `bundle.get_eye_track(exp_key)` (basler, 60 Hz — take frame `2 * center_frame`).
      These are two independently derived ground truths (ray-intercept vs. Tobii PoG); agreement
      within a few tens of pixels validates the whole geometric chain. A large systematic gap is a
      finding worth investigating before any model output is interpreted.

### Stimulus and inference-level

- [ ] **Resize sanity.** For 5 experiments, plot the 256×256 resized stimulus with the valid
      `gaze_px` overlaid after scaling by `(256/1920, 256/1080)`. Points must land on plausible image
      content, not systematically drift toward a corner. Note the two scale factors differ — that is
      the intended non-uniform squash of FR8.4, not a bug.
- [ ] **Trajectory plausibility.** For 10 experiments, plot the noisy `gaze_px` trajectory. It should
      look like a jittery but connected path, not a point cloud and not a straight line.
- [ ] **Predicted scanpath plausibility** (the qualitative criterion from Mission → Success
      Criteria): predicted fixations lie inside `[0,1920)×[0,1080)`, do not collapse to a single
      point, and do not diverge off-image. Report the fraction of predicted fixations outside the
      screen — a large fraction means the model is being fed a distribution it never saw.
- [ ] **Predicted lengths.** Histogram of `pred_len`. Values pinned at `K` for nearly every sample
      mean the EOS head never fires on this distribution; values of 1 everywhere mean it fires
      immediately. Either extreme is a finding to report, not a failure to hide — but note that with
      only ~1.67 s of post-stimulus signal (50 frames at 30 Hz), short scanpaths are expected.
- [ ] **Duration distribution.** Predicted `dur_ms` should sit mostly in 100–1200 ms, matching the
      EVE scanpath pipeline's own filters. Note how many were clipped by `remove_outliers=True`.
- [ ] **Input-length effect.** Scatter `pred_len` against `src_len`. A strong dependence is expected
      and benign; a flat line means the model is ignoring the gaze input, which would undercut the
      whole premise.

---

## Data Architecture Integrity

The EVE Mission's third correctness requirement — no positional coupling between independently
written artifacts.

- [ ] **`exp_key` is the only address.** Grep the new modules: no consumer indexes `gaze_px`,
      `validity`, or `pred_scanpath` by a raw integer derived from anywhere other than the
      `exp_key → row` dict. `cache.get_*` and `store.get*` are the only read paths.
- [ ] **`exp_keys` roundtrip.** After `save`/`load`, `loaded.exp_keys == built.exp_keys` as a list of
      `str` in identical order, and `loaded._idx` maps each key to the same row index.
- [ ] **No phantom keys.** Every `exp_key` in the inference store exists in the gaze cache, and every
      `exp_key` in the gaze cache exists in `bundle.samples_df`. Assert both set inclusions
      explicitly after a production run; a phantom key must raise, never warn.
- [ ] **Order verification is not bypassable.** Hand-edit a saved cache to permute `exp_keys` while
      leaving `gaze_px` in the original order, then `load` and call `get_gaze(k)`. The returned array
      must correspond to the permuted key — i.e. the accessor genuinely resolves through `_idx` and
      does not shortcut to a cached positional index. Document that this specific corruption is
      *undetectable* from the cache alone (the arrays carry no independent key), which is exactly why
      `verify(bundle)` and the split cross-check exist.
- [ ] **The two split columns are not cross-wired.** Assert `cache.splits_df["eve_split"]` equals
      `bundle.samples_df.set_index("exp_key").loc[keys, "split"]` element-wise, and that
      `cache.splits_df["eyenet_split"]` equals the CSV's per-key `split` element-wise. Because the two
      vocabularies overlap (`val`, `test` appear in both), a swap would not raise anywhere — it would
      silently select 0 experiments for `eyenet_split="val"` (no EVE-val key is EyeNet-val) or, worse,
      select the wrong 260. This assertion is the only thing standing between that swap and a
      published number.
- [ ] **Dataset ↔ image-dataset correspondence.** The Group 5 index invariant is re-asserted at
      runtime in `save_predictions_eve_real.py` (an `assert` over all `i`), not only in tests — the
      two datasets are constructed independently and a filter-argument typo would otherwise pair
      every gaze trajectory with the wrong stimulus.
- [ ] **Frame provenance survives.** `frame_indices` in the inference store lets any row of `src_px`
      be traced back to its center-camera frame and thus to its CSV row. Spot-check 5 rows:
      `frame_indices[i, :src_len[i]]` equals `np.where(cache.get_validity(exp_key))[0]`.
