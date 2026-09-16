# %% [markdown]
# # Save prediction.json (EVE head-to-head)
# Runs a trained MixerModel on the EyeNet real-noise gaze of every EVE test cell of
# the few-shot-scanpath evaluation repo and writes the contract file it scores
# (`pred_seed0.json`) plus `run_notes.md`. Contract: the eval repo's
# `spec/handoff/prediction_contract.md`; rules are in src/eval/prediction_contract.py.
#
# Only EyeNet gaze is used. A cell whose trial has no EyeNet prediction in the cache
# (or too few valid frames) gets an empty scanpath and is listed in the run notes;
# MAX_EMPTY_CELLS stops a file with more gaps than expected from being written.
# See spec/handoff/eve_eyenet_coverage.md for the cells still missing EyeNet gaze.
#
# The model is deterministic (regressed coordinates/durations, EOS-threshold stop),
# so one file (seed 0) is produced.
#
# Usage (edit the configuration block below, then run):
#   python src/notebooks/save_prediction_contract_eve.py

# %%
import os
import sys
import gc
import random
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)

from src.data.datasets import CoupledDataloader
from src.data.eve_real_noise import EyeNetGazeCache, EveRealNoiseDataset, EveRealNoiseImgDataset
from src.eval import prediction_contract as pc
from src.eval.eval_utils import invert_transforms, eval_autoregressive
from src.training.pipeline_builder import PipelineBuilder
from src.training.training_utils import move_data_to_device


# ── Configuration ─────────────────────────────────────────────────────────────

BUNDLE_DIR = "/mnt/scratch/leonardo.ulloa/5519804/data/bundle"
CACHE_PATH = "data/eve_real_noise/eyenet_gaze_cache.h5"
# Copied by hand from few-shot-scanpath/data/eve_bridge/ (git-ignored there).
GT_PATH = "data/eve_bridge/fixations.json"
SUBJECT_MAP_PATH = "data/eve_bridge/subject_id_map.json"

CKPT_PATH = os.path.join("outputs", "2026-07-24", "16-17-19")
RUN_NAME = "mixer_eyenet"
MODEL_NAME = "Ours (MixerModel, EyeNet gaze)"
OUT_DIR = os.path.join("outputs", "eve_prediction_contract", RUN_NAME)

SEED = 0
# Known unavoidable gaps: 6 train18 trials have no usable gaze in any source.
# Raise this only for debugging runs on a partial cache — never for a handoff file.
MAX_EMPTY_CELLS = 6


# ── Helpers ───────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return sha + (" (dirty working tree)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_model_and_data(ckpt_path, bundle, cache, exp_keys):
    cfg = OmegaConf.load(os.path.join(ckpt_path, ".hydra", "config.yaml"))
    ckpt_img_size = int(cfg.data.load.img_size if "load" in cfg.data else cfg.data.img_size)

    real = OmegaConf.load(os.path.join("configs", "data", "eve_real.yaml"))
    if int(real.load.img_size) != ckpt_img_size:
        raise ValueError(
            f"img_size mismatch: eve_real.yaml load.img_size={int(real.load.img_size)} "
            f"vs checkpoint img_size={ckpt_img_size}."
        )
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": OmegaConf.to_container(real, resolve=True)}))
    cfg.data.bundle_dir = str(bundle.bundle_dir)

    transforms = PipelineBuilder(cfg)._build_transforms()
    gaze_ds = EveRealNoiseDataset(
        cache, bundle, eyenet_split=None, exp_keys=exp_keys,
        max_fixations=cfg.data.max_fixations,
        min_valid_frames=cfg.data.min_valid_frames,
        transforms=transforms, log=True,
    )
    # One image per trial: EVE renders each photograph at a per-trial scale/offset,
    # and predictions live in that trial's screen coordinates.
    img_ds = EveRealNoiseImgDataset(
        cache, bundle, eyenet_split=None, exp_keys=exp_keys, dedup_by="exp_key",
        min_valid_frames=cfg.data.min_valid_frames,
        resize_size=cfg.data.load.img_size,
        transform=PipelineBuilder.make_transform(cfg.data.load.img_size),
    )
    assert len(gaze_ds) == len(img_ds), f"{len(gaze_ds)} gaze vs {len(img_ds)} img samples"
    assert all(gaze_ds.exp_key_at(i) == img_ds.exp_key_at(i) for i in range(len(gaze_ds))), \
        "gaze/img datasets disagree on exp_key ordering"

    dl = CoupledDataloader(
        gaze_ds, Subset(img_ds, torch.arange(len(img_ds))),
        batch_size=cfg.data.load.batch_size, shuffle=False,
        num_workers=1, persistent_workers=False,
        pin_memory=False, drop_last_batch=False,
    )

    model, _ = PipelineBuilder(cfg).build_model()
    ckpt = torch.load(os.path.join(ckpt_path, "model.pth"), map_location="cpu")
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    return cfg, model, gaze_ds, dl


def write_run_notes(path, *, cfg, empty, n_capped, cache, validator_line):
    reasons = {}
    for rec in empty:
        reasons.setdefault(rec["status"], []).append(rec)
    empty_lines = "\n".join(
        f"  - `{reason}`: {len(recs)} cells — "
        + ", ".join(f"{r['exp_key']} ({r['subject_eve']})" for r in recs[:12])
        + (" …" if len(recs) > 12 else "")
        for reason, recs in sorted(reasons.items())
    ) or "  - none"
    model_pth = os.path.join(CKPT_PATH, "model.pth")
    text = f"""# Run notes — {MODEL_NAME}

Generated {datetime.now(timezone.utc).isoformat()} by
`src/notebooks/save_prediction_contract_eve.py`.

## Model
- Name: {MODEL_NAME} (`{RUN_NAME}`), `{cfg.model.get('name', 'MixerModel')}`,
  image encoder `{cfg.model.image_encoder.get('type', 'dinov3')}`.
- Checkpoint: `{model_pth}`, sha256 `{pc.sha256_file(model_pth)}`.
- Model repo commit: `{git_commit()}`.

## Determinism
- **Deterministic.** Coordinates and durations are regressed and the scanpath stops
  at the first step with `sigmoid(eos) > {pc.EOS_THRESHOLD}`; nothing is sampled.
  One file (`pred_seed0.json`, seed {SEED}) is provided, not three copies.

## Conditioning
- **Not subject-conditioned** in the few-shot sense: no support scanpaths are used,
  and nothing from `split == "train"` or from any test record's `X/Y/T` is read.
  The retrieval block (MRR, R@k) is not meaningful for this arm.
- **Different input regime from ISP-SENet:** the model receives the *same trial's*
  webcam-estimated gaze (EyeNet ResNet18 prediction from the EVE webcam frames,
  projected to the screen) together with that trial's stimulus. It never sees
  the Tobii ground truth, but it does observe a noisy measurement of the very
  viewing behaviour being scored.
- EyeNet gaze source: `{cache.attrs.get('source_csv', '?')}`, cache built
  {cache.attrs.get('built_at', '?')}. Cells whose trial came from an EyeNet
  *training* participant would carry in-sample (optimistic) gaze; see
  `spec/handoff/eve_eyenet_coverage.md`.

## Coordinates
- Model output: normalized `[0,1]` screen coordinates of the trial's 1920×1080 screen
  (the input stimulus is that trial's full screen render, squashed to
  {int(cfg.data.load.img_size)}×{int(cfg.data.load.img_size)}).
- Inverted to 1920×1080 pixels by the pipeline's `Normalize` inverse
  (`invert_transforms`), then `x/3.75`, `y/2.8125`, clipped to `[0,512]`/`[0,384]`.

## Stimulus per cell
- Cell `(name, subject)` → EVE trial `exp_key` with
  `stimulus_name == splitext(name)[0]` and `subject == to_eve[subject]` (exactly one
  trial per cell, asserted). The stimulus is `EveBundle.get_stimulus(exp_key)` — the
  per-trial screen render, never a per-name image. Each record carries `exp_key`.

## Duration and length
- Duration: regressed (inverse `NormalizeDuration`, period 1200 ms; predictions above
  1200 ms clamped by `invert_transforms(remove_outliers=True)`), rounded to int ms,
  negatives clipped to 0.
- Length: EOS stop token (first step with `sigmoid(eos) > {pc.EOS_THRESHOLD}`), decode
  budget {int(cfg.data.max_fixations) + 1} steps, capped at {pc.MAX_LENGTH}
  ({n_capped} cells hit the cap).
- Empty scanpaths ({len(empty)} cells), by reason:
{empty_lines}

## Validator output
```
{validator_line}
```
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ── Main ──────────────────────────────────────────────────────────────────────

seed_everything(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

gt = pc.load_ground_truth(GT_PATH)
cells = pc.contract_cells(gt, pc.load_subject_map(SUBJECT_MAP_PATH))
assert len(cells) == pc.N_TEST_CELLS, len(cells)

from evedataset import EveBundle
bundle = EveBundle.load(BUNDLE_DIR)
cache = EyeNetGazeCache.load(CACHE_PATH)
cell_to_exp = pc.resolve_exp_keys(cells, bundle.samples_df)
exp_to_cell = {cell_to_exp[c.key]: c for c in cells}
print(f"{len(cells)} test cells; {sum(k in set(cache.exp_keys) for k in exp_to_cell)} have EyeNet gaze")

cfg, model, gaze_ds, dl = load_model_and_data(CKPT_PATH, bundle, cache, list(exp_to_cell))
model.set_phase("Fixation")
model.to(device)
model.eval()

records = {}
n_capped = 0
with torch.no_grad():
    for batch in tqdm(dl, desc="Predicting"):
        inp = move_data_to_device(batch, device)
        out = eval_autoregressive(model, inp, only_last=True)
        inp_px, out_px = invert_transforms(inp, out, dl, remove_outliers=True)
        for i in range(inp["src"].size(0)):
            key = gaze_ds.exp_key_at(int(inp["sample_idx"][i]))
            length = pc.predicted_length(out_px["cls"][i].squeeze(-1).cpu().numpy())
            n_capped += length > pc.MAX_LENGTH
            records[key] = pc.make_record(exp_to_cell[key], key,
                                          out_px["reg"][i].cpu().numpy(), length)

cached = set(cache.exp_keys)
empty = []
for key, cell in exp_to_cell.items():
    if key not in records:
        reason = "no_eyenet_gaze" if key not in cached else "too_few_valid_frames"
        records[key] = pc.make_record(cell, key, None, 0, status=reason)
        empty.append(records[key])

if MAX_EMPTY_CELLS is not None and len(empty) > MAX_EMPTY_CELLS:
    raise RuntimeError(
        f"{len(empty)} cells have no EyeNet gaze (limit {MAX_EMPTY_CELLS}). Rebuild the "
        "gaze cache with full coverage (spec/handoff/eve_eyenet_coverage.md) or raise "
        "MAX_EMPTY_CELLS for a debugging run."
    )

out_list = [records[cell_to_exp[c.key]] for c in cells]
stats = pc.validate_records(out_list, cells)
pred_path = os.path.join(OUT_DIR, f"pred_seed{SEED}.json")
pc.write_prediction_json(pred_path, out_list)
line = pc.format_stats(pred_path, stats)
print(line)
write_run_notes(os.path.join(OUT_DIR, "run_notes.md"), cfg=cfg, empty=empty,
                n_capped=n_capped, cache=cache, validator_line=line)
print(f"Wrote {pred_path} and run_notes.md")
print("Validate with the eval repo's stdlib checker before handing off:")
print(f"  python scripts/validate_prediction.py {GT_PATH} {pred_path}")

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
