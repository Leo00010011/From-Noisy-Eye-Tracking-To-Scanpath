# %% [markdown]
# # Save Predictions (EVE real noise)
# Runs autoregressive fixation prediction (and optional denoising) on the EyeNet
# real-noise gaze cache, then persists inputs and outputs to an exp_key-keyed HDF5
# artifact (RealNoiseInferenceStore) for offline study.
#
# No ground-truth scanpath is consumed and no accuracy metric is computed — this
# script produces the substrate that a later evaluation feature joins against ground
# truth once EVE scanpaths are available.
#
# Usage (edit the configuration block below, then run):
#   python src/notebooks/save_predictions_eve_real.py
#
# Mirrors src/notebooks/save_predictions_eve.py but reads configs/data/eve_real.yaml,
# has no clean_x, and keys every output row by exp_key rather than positional index.

# %%
import os
import sys
import gc

import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)

from src.data.datasets import CoupledDataloader
from src.data.eve_real_noise import EyeNetGazeCache, EveRealNoiseDataset, EveRealNoiseImgDataset
from src.data.eve_real_noise_store import RealNoiseInferenceStore
from src.eval.eval_utils import invert_transforms, eval_autoregressive
from src.training.pipeline_builder import PipelineBuilder
from src.training.training_utils import move_data_to_device


# ── Configuration ─────────────────────────────────────────────────────────────

BUNDLE_DIR = "/mnt/beegfs/home/leonardo.ulloa/projects/bundle"
CACHE_PATH = "data/eve_real_noise/eyenet_gaze_cache.h5"
OUT_DIR = os.path.join("outputs", "eve_real_noise")
EYENET_SPLIT = None   # None (both), "val", or "test"

ckpt_paths = [
    os.path.join("outputs", "2026-06-15", "19-18-05"),
]
names = [
    "eve_real_duration_dist",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ckpt_img_size(cfg) -> int:
    return int(cfg.data.load.img_size if "load" in cfg.data else cfg.data.img_size)


def _invert_src_to_px(src_row: torch.Tensor, transforms) -> np.ndarray:
    """Invert a single normalised src row (T, 3) back to pixel/ms space.

    invert_transforms handles tgt/clean_x/denoise but not src, so walk the transforms
    in reverse and apply every key=='x' inverse (Normalize coords + time). StandarizeTime
    carries key=='x' but no inverse and is skipped by the hasattr guard.
    """
    s = src_row.unsqueeze(0).clone()
    for t in reversed(transforms):
        if getattr(t, "key", None) == "x" and hasattr(t, "inverse"):
            s = t.inverse(s, None, "x")
    return s.squeeze(0).cpu().numpy()


def load_model_and_data(ckpt_path: str, bundle_dir: str, cache_path: str, eyenet_split=None):
    cfg = OmegaConf.load(os.path.join(ckpt_path, ".hydra", "config.yaml"))
    ckpt_img_size = _ckpt_img_size(cfg)

    real = OmegaConf.load(os.path.join("configs", "data", "eve_real.yaml"))
    if int(real.load.img_size) != int(ckpt_img_size):                     # FR8.5
        raise ValueError(
            f"img_size mismatch: eve_real.yaml load.img_size={int(real.load.img_size)} "
            f"vs checkpoint img_size={int(ckpt_img_size)}. The stimulus must be fed at "
            "the resolution the checkpoint was trained on."
        )
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": OmegaConf.to_container(real, resolve=True)}))
    cfg.data.bundle_dir = bundle_dir

    from evedataset import EveBundle
    bundle = EveBundle.load(bundle_dir)
    cache = EyeNetGazeCache.load(cache_path)

    transforms = PipelineBuilder(cfg)._build_transforms()
    gaze_ds = EveRealNoiseDataset(
        cache, bundle, eyenet_split=eyenet_split,
        max_fixations=cfg.data.max_fixations,
        min_valid_frames=cfg.data.min_valid_frames,
        transforms=transforms, log=True,
    )
    img_ds = EveRealNoiseImgDataset(
        cache, bundle, eyenet_split=eyenet_split,
        min_valid_frames=cfg.data.min_valid_frames,
        resize_size=cfg.data.load.img_size,
        transform=PipelineBuilder.make_transform(cfg.data.load.img_size),
    )

    # Runtime re-assertion of the Group 5 index invariant (Data Architecture Integrity).
    assert len(gaze_ds) == len(img_ds), f"{len(gaze_ds)} gaze vs {len(img_ds)} img samples"
    assert all(gaze_ds.exp_key_at(i) == img_ds.exp_key_at(i) for i in range(len(gaze_ds))), \
        "gaze/img datasets disagree on exp_key ordering"

    dl = CoupledDataloader(
        gaze_ds, Subset(img_ds, torch.arange(len(img_ds))),
        batch_size=cfg.data.load.batch_size, shuffle=False,               # FR11.3
        num_workers=0, persistent_workers=False,
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

    return cfg, model, gaze_ds, dl, bundle


# ── Main ──────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

for ckpt_path, name in zip(ckpt_paths, names):
    print(f"\n>>> Model: {name}  ({ckpt_path})")
    cfg, model, gaze_ds, dl, bundle = load_model_and_data(
        ckpt_path, BUNDLE_DIR, CACHE_PATH, eyenet_split=EYENET_SPLIT
    )
    model.set_phase("Fixation")
    model.to(device)
    model.eval()

    has_denoise = callable(getattr(model, "decode_denoise", None))
    transforms = gaze_ds.transforms
    records = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Saving"):
            inp = move_data_to_device(batch, device)
            out = eval_autoregressive(model, inp, only_last=True)
            if has_denoise:
                out.update(model.decode_denoise(**inp))
            inp_px, out_px = invert_transforms(inp, out, dl, remove_outliers=True)

            for i in range(inp["src"].size(0)):
                idx = int(inp["sample_idx"][i])
                key = gaze_ds.exp_key_at(idx)
                T = int(inp["src_mask"][i].sum()) if inp["src_mask"] is not None else inp["src"].size(1)
                rec = {
                    "exp_key": key,
                    "eyenet_split": gaze_ds.eyenet_split_at(idx),
                    "eve_split": gaze_ds.eve_split_at(idx),
                    "pred_scanpath": out_px["reg"][i].cpu().numpy(),          # (K, 3) px + ms
                    "eos_logit": out_px["cls"][i].squeeze(-1).cpu().numpy(),  # (K,)
                    "src_px": _invert_src_to_px(inp["src"][i, :T], transforms),
                    "src_len": T,
                    "frame_indices": gaze_ds.frame_indices_at(idx)[:T],
                }
                if has_denoise:
                    rec["denoise_px"] = out_px["denoise"][i, :T, :2].cpu().numpy()
                records.append(rec)

    if len(records) != len(gaze_ds):                                          # FR11.5
        raise ValueError(f"wrote {len(records)} records but dataset has {len(gaze_ds)} samples")
    keys = [r["exp_key"] for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("an exp_key was emitted twice")

    attrs = {
        "checkpoint_path": ckpt_path,
        "img_size": int(cfg.data.load.img_size),
        "max_fixations": int(cfg.data.max_fixations),
        "gaze_cache_path": CACHE_PATH,
        "bundle_dir": BUNDLE_DIR,
    }
    safe_name = name.replace(" ", "_").replace("/", "_")
    if EYENET_SPLIT is not None:
        safe_name = f"{safe_name}_{EYENET_SPLIT}"
    out_path = os.path.join(OUT_DIR, f"{safe_name}.h5")
    RealNoiseInferenceStore.save(out_path, run_name=name, records=records, attrs=attrs)
    print(f"Saved {len(records)} records -> {out_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

print("\nDone.")
print("Load offline with:")
print("  store = RealNoiseInferenceStore.load('<path>.h5')")
print("  sp = store.get_scanpath(store.exp_keys[0])")
