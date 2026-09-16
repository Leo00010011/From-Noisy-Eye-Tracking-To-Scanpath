# EyeNet gaze coverage for the EVE head-to-head

*Written 2026-09-16. Covers the eval repo's `spec/handoff/prediction_contract.md`.*

## TL;DR

The contract asks for a prediction on each of the **1062** `split == "test"` cells in
`fixations.json`. Our model's input is the EyeNet gaze of that trial. The current
`data/eve_real_noise/eyenet_gaze_cache.h5` has EyeNet gaze for only **332** of those cells.

| | cells |
|---|---|
| test cells in `fixations.json` | 1062 |
| have EyeNet gaze in the cache (all with ≥ 5 valid frames) | **332** (90 from `val02–05`, 242 from 8 `train*` participants) |
| **missing EyeNet gaze** | **730** (26 participants) |

The missing trials are listed in
[`eve_eyenet_missing_exp_keys.csv`](eve_eyenet_missing_exp_keys.csv), one row per cell,
with columns `exp_key, subject_eve, subject_dense, name, webgazer_has_valid_frames`.

`src/notebooks/save_prediction_contract_eve.py` refuses to write the handoff file while
more than `MAX_EMPTY_CELLS = 6` cells lack gaze. Once the cache is rebuilt with full
coverage, rerun it. No code change is needed.

## Why they are missing

EyeNet (`../EyeNet Pipeline`) splits EVE **by participant** (`src/eyenet/splits.py`):

- EVE `val*` participants → EyeNet **test**
- EVE `train*` participants → shuffled into EyeNet **train** / **val**

`scripts/export_predictions.py` only exports `SPLITS = ("val", "test")`. The 26
participants below were in EyeNet **train**, so `predictions.csv` has no rows for them:

| participant | cells | participant | cells | participant | cells |
|---|---|---|---|---|---|
| train01 | 18 | train13 | 22 | train27 | 27 |
| train02 | 29 | train14 | 16 | train28 | 30 |
| train04 | 26 | train16 | 25 | train30 | 27 |
| train07 | 28 | train18 | 30 | train32 | 27 |
| train09 | 31 | train19 | 29 | train33 | 40 |
| train10 | 43 | train20 | 24 | train34 | 27 |
| train11 | 28 | train21 | 31 | train35 | 35 |
| train12 | 20 | train22 | 25 | train36 | 29 |
|  |  | train25 | 29 | train37 | 34 |

The participants that do have gaze are EyeNet-val `train05, 06, 08, 24, 26, 29, 31, 39`
and EyeNet-test `val02–05`.

## ⚠ Leakage: do not just export the current checkpoint on its train split

Running the existing EyeNet checkpoint on these 26 participants yields **in-sample**
gaze: the network was fitted on exactly those webcam frames. Its angular error would be
lower than on the 332 held-out cells. Our model would then receive cleaner input than the
real-noise setting claims, which biases the head-to-head in our favour.

Options, in order of preference:

1. **Cross-fit EyeNet across participants (recommended).** Split the EVE-`train`
   participants into K folds (e.g. K = 5). Train one EyeNet per fold on the other folds
   and export predictions for the held-out fold. Every trial then gets out-of-sample gaze
   with the same training recipe. Label the exported rows so provenance survives (e.g.
   `split = "test"` plus a fold id in the run notes). The cache stores the label as
   `eyenet_split` and the prediction driver does not filter on it.
2. **In-sample export, flagged.** Export `train` too and state in `run_notes.md` that
   730 of the 1062 cells carry in-sample EyeNet gaze. Report the median angular error
   of the two groups side by side. It is cheap, but a reviewer will object.
3. **Score the 332-cell subset only.** This needs a change in the eval repo, which
   currently rejects anything other than the full 1062-key set, and ISP-SENet would
   have to be rescored on the same subset.

## Cluster steps (option 1 or 2)

1. **Export EyeNet predictions** for the `exp_key`s in `eve_eyenet_missing_exp_keys.csv`
   (or for all trials of those 26 participants) in the same CSV schema as
   `predictions.csv`: `split, exp_key, frame, patch, pred_x..z, target_x..z,
   angular_error_deg`, with right-eye vectors unflipped. `export_predictions.py`
   only needs a loader over those participants.
2. **Merge** with the existing `predictions.csv`. Each `exp_key` must carry exactly one
   `split` value, and `(exp_key, frame, patch)` must stay unique. `load_eyenet_predictions`
   raises on either violation.
3. **Rebuild the cache against the full bundle.** The local bundle
   (`../eve_shared/EveDataset/bundle`) has no `gaze_norm` / `gaze_ray` for these trials,
   so the projection can only run where the full bundle lives:
   ```bash
   python scripts/build_eyenet_gaze_cache.py \
       --csv merged_predictions.csv \
       --bundle-dir /mnt/scratch/leonardo.ulloa/5519804/data/bundle \
       --out data/eve_real_noise/eyenet_gaze_cache.h5
   ```
   It exits 1 if any experiment was skipped. Check the skip reasons.
4. **Bring back** `data/eve_real_noise/eyenet_gaze_cache.h5`, or run step 5 on the
   cluster.
5. **Produce the handoff file.** Set `BUNDLE_DIR` / `CKPT_PATH` / `RUN_NAME` in
   `src/notebooks/save_prediction_contract_eve.py` and run it. It prints
   `1062 test cells; N have EyeNet gaze`, so expect N ≥ 1056. It writes
   `outputs/eve_prediction_contract/<run>/pred_seed0.json` and `run_notes.md`. Then run:
   ```bash
   python scripts/validate_prediction.py data/eve_bridge/fixations.json \
       outputs/eve_prediction_contract/<run>/pred_seed0.json
   ```

## The six `train18` cells

`train18_step019, 021, 052, 061, 065, 067` have **no valid WebGazer frames** in any
replica. They are among the 730, so whether EyeNet can predict them is unknown until
step 1 runs. If they also have no usable EyeNet gaze (no face crops, or skipped at
projection), the driver emits an **empty scanpath** (`status: "no_eyenet_gaze"`). The
contract allows this, and the run notes list the cells. `MAX_EMPTY_CELLS = 6` reserves
room for exactly these six.

## What else was verified

- Every test cell maps to **exactly one** bundle trial via
  `(splitext(name)[0], to_eve[subject])`: 1062 distinct `exp_key`s, and each trial's
  participant matches.
- The **per-trial stimulus matters**: the three participants of a given name see
  different 1920×1080 renders (e.g. `train30_step048` and `val03_step051` have a mean absolute
  pixel difference of ≈38/255). The driver therefore uses `EveRealNoiseImgDataset(...,
  dedup_by="exp_key")`. The older `save_predictions_eve_real.py` still dedups by
  `stimulus_name` (the default). There, a trial's gaze can be paired with another trial's
  render of the same image, which is probably a latent bug in that path too.
- `fixations.json` sha256 matches the contract
  (`46c6926f…c5ac9b`). The copy lives in `data/eve_bridge/`, which is git-ignored.
