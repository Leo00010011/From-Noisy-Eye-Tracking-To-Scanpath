# Plan: EVE Real-Noise Scanpath Inference

## Context and Design Decisions

### Why a new code path instead of extending the existing `eve` one

`configs/data/eve.yaml` and `PipelineBuilder._load_eve_dataset` already read EVE data — but they read
the `/webgazer` group, which holds the *WebGazer-replica* signal produced by the EVE repo's
`PredictionCache` sweep. That signal is a different noise source with a different provenance, and
`EveScanpathDataset` hard-requires a ground-truth scanpath (`sp.shape[1] > 0`) and a Tobii
`eye_track` (for `clean_x`). This feature has neither: scanpath access is still in progress in the
EVE repo, and the Denoise phase is out of scope.

Extending `EveScanpathDataset` would mean either modifying the `evedataset` package (explicitly
forbidden — the EVE Mission's *Architectural Boundary* says consuming code lives downstream) or
threading optional-target logic through a class that four other scripts depend on. A separate,
self-contained pair of dataset classes in this repo is smaller, cannot regress the existing path, and
is deleted cleanly if the approach is abandoned.

For the same reason **`PipelineBuilder` is not touched at all**. This is an inference-only feature;
the training entry point gains nothing from knowing about it, and the `data_path` `NameError` and
other known fragilities in that file (Roadmap → *Engineering Correctness*) make it the wrong place to
add a branch.

### Why the projection is not reimplemented here

`EveBundle.project_normalized_gaze` is the EVE repo's designated single canonical projection path —
the *Expose GazeRayCache* feature states explicitly that "no consumer reimplements the geometry." It
applies the per-frame head-pose `R` from `/gaze_norm/center`, the constant camera→screen rotation
from `/gaze_ray/center`, and the prediction-independent eye origin, then intersects the screen plane
`z = 0`. Reimplementing it here would duplicate ~40 lines of geometry whose correctness was
established by tests in another repo, and would silently drift when that repo's convention changes.

The choice has a free correctness dividend: feeding the CSV's `target_*` vectors through the same
call reproduces `EveBundle.get_screen_intercept` to ≤1.3e-4 px, which becomes the cache's build-time
self-test (validation Group 2).

### Why a cache rather than computing on the fly

Projection is not free — 1,468 `project_normalized_gaze` calls, each opening `bundle.h5` several
times (the package deliberately holds no persistent handle, per its portability rule). More
importantly, the projected screen-space gaze is the *durable* artifact: it is what a future evaluation
feature joins against ground-truth scanpaths, and what a QA notebook plots. Persisting it keyed by
`exp_key` follows the EVE constitution's third correctness requirement — *data architecture
integrity*: "every cached artifact must be addressable by a stable, explicit primary key rather than
by integer position." The `exp_keys` dataset is the verification guard; every accessor resolves
through an `exp_key → row` dict.

### Why left and right eyes are averaged

The CSV carries both eyes for every frame. Averaging the two *screen intercepts* (not the two
direction vectors — each ray carries its own eye origin, so averaging must happen after projection)
cuts the median per-frame distance to the ray-derived ground truth from 149 px to 89.6 px, i.e. from
3.9 DVA to 2.32 DVA. The per-eye arrays are still stored, at ~0.5 MB total, so the averaging decision
stays auditable and a future eye ablation does not force a rebuild.

For reference, the combined signal's 2.32 DVA median error sits below WebGazer's reported ~4.17°
mean, so this is a *cleaner* real signal than the noise regime the model was trained to handle —
worth stating plainly when the outputs are eventually interpreted.

### Why the target is a placeholder, and why that is safe

`seq2seq_padded_collate_fn` requires `item['y']`, and `eval_autoregressive` takes its decode-step
budget from `tgt_mask.size(1)`. With no ground truth available, `y` is a `(3, max_fixations)` block of
`PAD_TOKEN_ID`. This is safe because inference never reads it: `eval_autoregressive` sets
`inputs['tgt'] = None` before the first decode step and feeds back only the model's own output
thereafter, and there is no loss function in the loop. The placeholder's *only* effect is fixing the
number of decode steps at `max_fixations + 1`. Sequence length is then a purely offline decision made
from the stored EOS logits, which is why `pred_scanpath` keeps all `K` steps rather than truncating.

### Why timestamps are synthesized

The production bundle's `cam_type` is `basler`: `/samples/eye_track` is `(3096, 3, 180)` at 60 Hz.
There is no center-camera (90-frame, 30 Hz) timestamp anywhere in the bundle. Since the center camera
runs at a fixed 30 Hz and `StandarizeTime` subtracts the first timestamp anyway, `t_ms = frame * 33.333`
reproduces true relative timing exactly under the constant-rate assumption — and relative timing is
all `NormalizeTime` consumes. The approximation is recorded as the `timestamp_source` HDF5 attribute
so it is detectable later, rather than being an undocumented assumption baked into float arrays.

### Why EyeNet's split is the operative one and EVE's is metadata

Two split labels are in play and it would be easy to pick the wrong one. The recovery model was
trained on CocoFreeView and has never seen EVE, so EVE's train/val/test partition carries no leakage
meaning for it — filtering by it would be arbitrary. EyeNet's partition is what matters: its
predictions on its own training data would be unrepresentatively clean, and testing the recovery
model on those would overstate how it handles real degraded gaze.

All 734 experiments in the CSV are EyeNet val (474 experiments, 8 subjects, 22,741 frames) or test
(260 experiments, 5 subjects, 12,688 frames), so the artifact is entirely unseen-by-EyeNet and no
filtering is needed for correctness. The label is kept per experiment so val and test can be reported
separately, and the two groups are subject-disjoint, so that separation is meaningful.

To make the wrong choice unrepresentable in code, the field is named `eyenet_split` everywhere, the
EVE one is named `eve_split`, and the dataset filter parameter is `eyenet_split` — there is no
parameter that accepts an EVE split. Passing `"train"` raises rather than silently returning an empty
dataset.

### Why coordinates are not clamped

Measured on all 734 experiments: 0.0% of combined predicted frames and 0.065% of ground-truth frames
land outside the screen rectangle. Clamping would be a no-op that nonetheless destroys the ability to
tell a genuinely off-screen intercept from a boundary fixation. The `n_offscreen` build statistic
records the count instead.

### Constitution constraints that apply

- *TechStack → Data Flow*: gaze arrays are `(3, T)` with rows `[x, y, t]`; fixations are `(3, N)` with
  rows `[x, y, duration]`; coordinates are `(x, y)` normalized to `[0,1]`. Both new datasets emit
  exactly this, so `seq2seq_padded_collate_fn` and the model need no changes.
- *TechStack → Evaluation Protocol*: "Before computing pixel-space metrics, invert the `Normalize`
  transforms using their `.inverse()` methods." Every pixel value written to the output store goes
  through `invert_transforms`; nothing is hand-multiplied by 1920/1080.
- *TechStack → Gotchas*: `PAD_TOKEN_ID = 0.5`. The placeholder target reuses it rather than
  introducing a second sentinel.
- *Roadmap → Research Extensions*: "Real WebGazer validation — validate the noise simulation pipeline
  against real paired recordings." This feature builds the substrate for that item; it does not
  itself close it, because it computes no comparison metrics.
- *EVE Mission → Architectural Boundary*: consuming code lives downstream. Nothing here is proposed
  for the `evedataset` package.

### Nothing is needed from the EVE repo

Worth stating explicitly, since the request invited it: every capability this feature needs already
ships in `evedataset` — `project_normalized_gaze`, `has_gaze_norm`, `has_gaze_ray`, `get_stimulus`,
`samples_df`. The one thing that would *improve* the result is a real center-camera timestamp
accessor (FR4.3), but the 30 Hz synthesis is exact under a constant frame rate, so this is a
nice-to-have, not a blocker. The only hard prerequisite is that the bundle be exported with both
`include_gaze_vector_data=True` and `include_gaze_ray_data=True` — the production bundle already is.

---

## Implementation Steps

### Step 1 — `src/data/eve_real_noise.py`: constants and CSV loader

Create the module. Add constants `CENTER_FPS = 30.0`, `CENTER_FRAME_COUNT = 90`, `STIMULUS_W = 1920`,
`STIMULUS_H = 1080`, `DEFAULT_CACHE_PATH`, `_FILL_DIRECTION = np.array([0.0, 0.0, -1.0])`,
`_EYES = ("left", "right")`, `_GROUP = "/eyenet_gaze"`, `_REQUIRED_COLUMNS` (the 11 CSV columns).

```python
def load_eyenet_predictions(csv_path):
    df = pd.read_csv(csv_path, dtype={"exp_key": str, "patch": str, "split": str})
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing: raise ValueError(f"predictions CSV missing columns: {sorted(missing)}")
    bad_patch = set(df["patch"].unique()) - set(_EYES)
    if bad_patch: raise ValueError(f"unexpected patch values: {sorted(bad_patch)}")
    if df.duplicated(["exp_key", "frame", "patch"]).any():
        n = int(df.duplicated(["exp_key", "frame", "patch"]).sum())
        raise ValueError(f"{n} duplicate (exp_key, frame, patch) rows")
    if df["frame"].min() < 0 or df["frame"].max() >= CENTER_FRAME_COUNT:
        raise ValueError(f"frame outside [0, {CENTER_FRAME_COUNT})")
    norms = np.linalg.norm(df[["pred_x","pred_y","pred_z"]].to_numpy(np.float64), axis=1)
    n_bad = int((np.abs(norms - 1.0) > 1e-3).sum())
    if n_bad:
        warnings.warn(f"{n_bad} prediction vectors deviate from unit norm by > 1e-3")
    return df.astype({"frame": "int32", ...float32 for the 7 numeric columns...})
```

No re-normalization — `project_normalized_gaze` normalizes after applying `R` (FR1.3).

### Step 2 — `src/data/eve_real_noise.py`: the scatter + projection helper

Module-level private function, unit-testable without HDF5 by injecting a fake bundle.

```python
def _project_experiment(sub_df, exp_key, bundle):
    """sub_df: all CSV rows for one exp_key (both eyes).
    Returns dict of the per-experiment arrays listed in FR5, or raises."""
    out = {}
    for eye in _EYES:
        s = sub_df[sub_df["patch"] == eye].sort_values("frame")
        fr = s["frame"].to_numpy(np.int64)
        mask = np.zeros(CENTER_FRAME_COUNT, dtype=bool); mask[fr] = True

        pred = np.tile(_FILL_DIRECTION, (CENTER_FRAME_COUNT, 1))
        tgt  = np.tile(_FILL_DIRECTION, (CENTER_FRAME_COUNT, 1))
        pred[fr] = s[["pred_x","pred_y","pred_z"]].to_numpy(np.float64)
        tgt[fr]  = s[["target_x","target_y","target_z"]].to_numpy(np.float64)

        p = bundle.project_normalized_gaze(exp_key, pred, eye=eye, spherical=False)
        t = bundle.project_normalized_gaze(exp_key, tgt,  eye=eye, spherical=False)
        valid = mask & p["validity"]                        # FR3.2

        out[f"{eye}_px"]       = np.where(valid[:,None], p["hit_px"], np.nan).astype(np.float32)
        out[f"{eye}_gt_px"]    = np.where(valid[:,None], t["hit_px"], np.nan).astype(np.float32)
        out[f"{eye}_validity"] = valid
        ae = np.full(CENTER_FRAME_COUNT, np.nan, np.float32)
        ae[fr] = s["angular_error_deg"].to_numpy(np.float32)
        out[f"{eye}_ae"] = ae

    out["gaze_px"], out["validity"] = _combine_eyes(
        out["left_px"], out["left_validity"], out["right_px"], out["right_validity"])
    out["gt_gaze_px"], _ = _combine_eyes(
        out["left_gt_px"], out["left_validity"], out["right_gt_px"], out["right_validity"])
    return out
```

```python
def _combine_eyes(lpx, lval, rpx, rval):
    """FR3.3 — mean where both eyes valid, single eye where only one, NaN where neither."""
    both = lval & rval
    out  = np.full_like(lpx, np.nan)
    out[both]        = (lpx[both] + rpx[both]) / 2.0
    out[lval & ~rval] = lpx[lval & ~rval]
    out[rval & ~lval] = rpx[rval & ~lval]
    return out.astype(np.float32), (lval | rval)
```

`_combine_eyes` is a pure function over four arrays — the primary unit-test surface (validation
Group 1).

### Step 3 — `src/data/eve_real_noise.py`: `EyeNetGazeCache`

`__init__(self, arrays: dict, attrs: dict)` stores the stacked arrays and builds
`self._idx = {key: i for i, key in enumerate(exp_keys)}`. All accessors resolve through `_idx` and
raise `KeyError` with the offending key in the message.

`build(csv_path, bundle, cache_path)`:

1. `df = load_eyenet_predictions(csv_path)`.
2. `samples = bundle.samples_df.set_index("exp_key")`.
3. For each `exp_key` in `sorted(df["exp_key"].unique())`:
   - skip `("not_in_bundle")` if absent from `samples`;
   - skip `("no_gaze_norm")` / `("no_gaze_ray")` on the `has_*` guards;
   - `try: rec = _project_experiment(...) except (KeyError, ValueError) as e: skip("projection_failed: {e}")`;
   - append `rec`, plus `eyenet_split = sub_df["split"].iloc[0]` (assert `sub_df["split"].nunique() == 1`
     — a key spanning both EyeNet splits is a corrupt CSV and must raise, not be silently resolved),
     `eve_split = samples.loc[k, "split"]`, `stimulus_name = samples.loc[k, "stimulus_name"]`.
4. Stack into `(N, 90, ...)` arrays; compute `n_offscreen` over `gaze_px` where `validity`.
5. Build `attrs` (FR5), `cache = cls(arrays, attrs)`, `cache.save(cache_path)`, return
   `(cache, skipped)`.

`save(cache_path)` — `Path(cache_path).parent.mkdir(parents=True, exist_ok=True)`, then
`h5py.File(path, "a")`, `if _GROUP in f: del f[_GROUP]`, `g = f.require_group(_GROUP)`, write the ten
datasets with `h5py.string_dtype()` for the four string columns, then `g.attrs.update(attrs)`.
Append mode + single-group write is the convention every EVE cache class follows.

`load(cache_path)` — `FileNotFoundError` / `ValueError` per FR6.3, including the duplicate-`exp_keys`
check.

`verify(bundle)` — FR6.4, returns a list, never raises.

Accessors per FR6.5. `get_eye_gaze(exp_key, eye)` raises `ValueError` for an eye outside
`("left", "right")` before the key lookup.

### Step 4 — `src/data/eve_real_noise.py`: the shared row filter

Both datasets must accept exactly the same rows, so the filter is one function used by both
(FR8.2):

```python
_EYENET_SPLITS = ("val", "test")

def _accepted_rows(cache, eyenet_split, min_valid_frames):
    """Return list[(row_index_in_cache, exp_key)] in cache order."""
    if eyenet_split is not None and eyenet_split not in _EYENET_SPLITS:
        raise ValueError(
            f"eyenet_split must be one of {_EYENET_SPLITS} or None, got {eyenet_split!r}. "
            "EVE split labels ('train'/'val'/'test' from the bundle) are not accepted here — "
            "the recovery model never saw EVE, so only EyeNet's split is meaningful."
        )
    out = []
    for i, k in enumerate(cache.exp_keys):
        if eyenet_split is not None and cache.splits_df.at[i, "eyenet_split"] != eyenet_split:
            continue
        if int(cache.get_validity(k).sum()) < min_valid_frames:
            continue
        out.append((i, k))
    return out
```

Note the collision hazard the error message guards against: EVE's split vocabulary
(`train`/`val`/`test`) overlaps EyeNet's (`val`/`test`), so `eyenet_split="val"` would silently
"work" while meaning something different from what a caller thinking in EVE terms intended. The
naming is the primary defence; the raise catches only the unambiguous `"train"` case.

### Step 5 — `src/data/eve_real_noise.py`: `EveRealNoiseDataset`

`__init__` calls `_accepted_rows`, then per accepted key:

```python
val   = cache.get_validity(k)                 # (90,) bool
frames = np.where(val)[0]                     # ascending
gaze  = cache.get_gaze(k)[frames]             # (T, 2) float32, NaN-free by construction
x = np.empty((3, len(frames)), dtype=np.float64)
x[0], x[1] = gaze[:, 0], gaze[:, 1]
x[2] = frames * (1000.0 / CENTER_FPS)          # FR4.1
y = np.full((3, max_fixations), PAD_TOKEN_ID, dtype=np.float64)   # FR7.4
fixation_mask = np.zeros(len(frames), dtype=np.uint8)
```

Assert `not np.isnan(x[:2]).any()` — a NaN here means `validity` and `gaze_px` disagree, i.e. a
corrupt cache; raise `ValueError` naming the key rather than letting NaN reach the model.

`data_store` per FR7.5; `__getitem__` mirrors `EveScanpathDataset.__getitem__` byte for byte except
that `clean_x` is absent from both `inp` and the optional-key loop. `exp_key_at(i)` returns
`self.data_store["exp_keys"][i]`.

`log=True` prints
`f"EveRealNoiseDataset eyenet_split={eyenet_split}: {n} samples ({skipped} skipped of {len(cache.exp_keys)})"`.

### Step 6 — `src/data/eve_real_noise.py`: `EveRealNoiseImgDataset`

Same `_accepted_rows` call with the same arguments. Dedup by `stimulus_name` exactly as
`EveImgDataset` does:

```python
ingest = v2.Compose([v2.ToImage(),
                     v2.Resize((resize_size, resize_size), antialias=True),
                     v2.ToDtype(torch.uint8, scale=False)])
self.image_bank = torch.empty((N_unique, 3, resize_size, resize_size), dtype=torch.uint8)
for uid, exp_key in enumerate(first_key_per_stimulus):
    self.image_bank[uid] = ingest(bundle.get_stimulus(exp_key))   # (1080,1920,3) uint8 -> (3,256,256)
```

`get_stimulus` returns HWC uint8; `v2.ToImage()` handles the HWC→CHW conversion, so no manual
transpose. `__getitem__` returns `(self.runtime_transform(bank[uid]) if transform else bank[uid], i, uid)`.

`resize_size` defaults to 256 — the resolution the current checkpoints were trained at, matching
`configs/data/default.yaml` and `configs/data/eve.yaml`. It is never hardcoded in the driver; it flows
from `cfg.data.load.img_size` and is cross-checked against the checkpoint (Step 10).

### Step 7 — `src/data/eve_real_noise_store.py`: `RealNoiseInferenceStore`

Depends on nothing from Steps 1–6 (it takes plain dicts), so it can be written and tested
independently.

`save(path, run_name, records, attrs)`:
1. Validate: every record has `exp_key`; no duplicates → `ValueError` (FR9.1).
2. `T = max(r["src_len"] for r in records)`, `K = records[0]["pred_scanpath"].shape[0]`; raise
   `ValueError` if any record's `K` differs.
3. Allocate NaN-filled `(N, T, ·)` and `-1`-filled `frame_indices`; copy each record's valid prefix.
4. `h5py.File(path, "w")`, group `/inference`, write per FR9, `g.attrs.update(attrs)`.
5. `denoise_px` written only if every record carries it; if some do and some do not, raise
   `ValueError` — a mixed run means two different checkpoints were merged.

`load(path)` + accessors per FR9.2. `get_scanpath(exp_key)` slices `pred_scanpath[i, :pred_len[i]]`.

### Step 8 — `configs/data/eve_real.yaml`

Per FR10. Header comment states the three ways it differs from `eve.yaml`: no `clean_x`, no
`run_key_filter` (there is no WebGazer variant to select), and `eyenet_split` in place of any EVE
split notion. `load.img_size` stays at 256, the same value `eve.yaml` and `default.yaml` already use.

### Step 9 — `scripts/build_eyenet_gaze_cache.py`

```
python scripts/build_eyenet_gaze_cache.py \
    --csv "../EyeNet Pipeline/predictions.csv" \
    --bundle-dir /path/to/bundle \
    --out data/eve_real_noise/eyenet_gaze_cache.h5
```

`argparse`, then `EveBundle.load` → `EyeNetGazeCache.build` → print the build report: number cached,
`skipped` grouped by reason, `n_offscreen`, mean/median frames per experiment, the
`eyenet_split` breakdown (expected `val: 474, test: 260`), and the median `‖gaze_px − gt_gaze_px‖`
(the headline sanity number, expected ≈ 89.6 px). Exit code 1 if `skipped` is non-empty, so a silent
coverage regression is not mistaken for success.

### Step 10 — `src/notebooks/save_predictions_eve_real.py`

Mirrors `save_predictions_eve.py`'s structure and header docstring.

```python
def load_model_and_data(ckpt_path, bundle_dir, cache_path, eyenet_split=None):
    cfg = OmegaConf.load(Path(ckpt_path) / ".hydra" / "config.yaml")
    ckpt_img_size = cfg.data.load.img_size if "load" in cfg.data else cfg.data.img_size
    real = OmegaConf.load("configs/data/eve_real.yaml")
    if int(real.load.img_size) != int(ckpt_img_size):          # FR8.5
        raise ValueError(f"img_size mismatch: config {real.load.img_size} vs checkpoint {ckpt_img_size}")
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": OmegaConf.to_container(real, resolve=True)}))
    cfg.data.bundle_dir = bundle_dir

    bundle = EveBundle.load(bundle_dir)
    cache  = EyeNetGazeCache.load(cache_path)
    transforms = PipelineBuilder(cfg)._build_transforms()
    gaze_ds = EveRealNoiseDataset(cache, bundle, eyenet_split=eyenet_split,
                                  max_fixations=cfg.data.max_fixations,
                                  min_valid_frames=cfg.data.min_valid_frames,
                                  transforms=transforms, log=True)
    img_ds  = EveRealNoiseImgDataset(cache, bundle, eyenet_split=eyenet_split,
                                     min_valid_frames=cfg.data.min_valid_frames,
                                     resize_size=cfg.data.load.img_size,
                                     transform=PipelineBuilder.make_transform(cfg.data.load.img_size))
    assert len(gaze_ds) == len(img_ds)
    assert all(gaze_ds.exp_key_at(i) == img_ds.exp_key_at(i) for i in range(len(gaze_ds)))
    dl = CoupledDataloader(gaze_ds, Subset(img_ds, torch.arange(len(img_ds))),
                           batch_size=cfg.data.load.batch_size, shuffle=False,      # FR11.3
                           num_workers=0, persistent_workers=False,
                           pin_memory=False, drop_last_batch=False)
    return cfg, gaze_ds, dl
```

Note `_build_transforms()` is called on a `PipelineBuilder` instance only to reuse its transform
construction; no dataset loading happens through it, so FR10.1 (no `PipelineBuilder` change) holds.

Inference loop per FR11.4:

```python
for batch in tqdm(dl):
    inp = move_data_to_device(batch, device)
    out = eval_autoregressive(model, inp, only_last=True)
    if has_denoise:
        out.update(model.decode_denoise(**inp))
    inp_px, out_px = invert_transforms(inp, out, dl, remove_outliers=True)
    for i in range(inp["src"].size(0)):
        idx = int(inp["sample_idx"][i])
        key = gaze_ds.exp_key_at(idx)
        T   = int(inp["src_mask"][i].sum()) if inp["src_mask"] is not None else inp["src"].size(1)
        records.append({
            "exp_key":      key,
            "eyenet_split": gaze_ds.eyenet_split_at(idx),
            "eve_split":    gaze_ds.eve_split_at(idx),
            "pred_scanpath": out_px["reg"][i].cpu().numpy(),        # (K, 3) px + ms
            "eos_logit":     out_px["cls"][i].squeeze(-1).cpu().numpy(),
            "src_px":        _invert_src_to_px(inp["src"][i, :T], transforms),
            "src_len":       T,
            "frame_indices": gaze_ds.frame_indices_at(idx)[:T],
            **({"denoise_px": out_px["denoise"][i, :T, :2].cpu().numpy()} if has_denoise else {}),
        })
```

`src_px` needs its own inversion: `invert_transforms` inverts `tgt`/`clean_x`/`denoise` but not
`src`. Add a small local helper that walks `reversed(transforms)` and applies
`transform.inverse(src, None, 'x')` for every transform whose `key == 'x'` — the same pattern
`invert_transforms_clean_x` uses. It lives in the driver script, not in `eval_utils.py`, so no shared
evaluation code changes.

Finally compute `pred_len` from `sigmoid(eos_logit) > 0.5`, assert the FR11.5 count/duplicate
conditions, and call `RealNoiseInferenceStore.save(...)`.

### Step 11 — `tests/test_eve_real_noise.py`

Implements validation.md Groups 1–5. Fixtures: a 3-experiment synthetic CSV written to `tmp_path`,
and a `FakeBundle` exposing `samples_df`, `has_gaze_norm`, `has_gaze_ray`, `get_stimulus`, and a
`project_normalized_gaze` that returns a deterministic analytic result — so Groups 1, 3, 4 run with
no bundle on disk. Groups 2 and 6 are marked
`@pytest.mark.skipif(not Path(BUNDLE_DIR).exists(), reason="production bundle not available")` and
exercise the real bundle when present.

---

## Implementation Order

1. **Step 1** — constants + `load_eyenet_predictions` (no dependencies)
2. **Step 2** — `_combine_eyes` + `_project_experiment` (needs Step 1's constants)
3. **Step 3** — `EyeNetGazeCache` build/save/load/verify/accessors (needs Step 2)
4. **Step 4** — `_accepted_rows` shared filter (needs Step 3's accessors)
5. **Step 5** — `EveRealNoiseDataset` (needs Step 4)
6. **Step 6** — `EveRealNoiseImgDataset` (needs Step 4)
7. **Step 7** — `RealNoiseInferenceStore` (independent; can be done any time after Step 1)
8. **Step 8** — `configs/data/eve_real.yaml` (independent)
9. **Step 9** — `scripts/build_eyenet_gaze_cache.py` (needs Step 3)
10. **Step 10** — `src/notebooks/save_predictions_eve_real.py` (needs Steps 5, 6, 7, 8)
11. **Step 11** — `tests/test_eve_real_noise.py` (needs Steps 1–7; write Group 1 tests alongside
    Step 2 rather than at the end)

Run Step 9 against the production bundle before starting Step 10 — the build report's median
`‖gaze_px − gt_gaze_px‖ ≈ 89.6 px` is the cheapest confirmation that the projection is wired
correctly, and a wrong number there invalidates everything downstream.
