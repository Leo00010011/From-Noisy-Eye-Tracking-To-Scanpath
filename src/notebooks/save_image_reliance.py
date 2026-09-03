# %% [markdown]
# # Image-Reliance Diagnostic Suite (CocoFreeView test split)
#
# Answers *"does the trained MixerModel actually use the image backbone?"* for a
# Mask2Former (or DINOv3) checkpoint, with three converging measurements over two passes:
#
#   Pass A (recorder ON) — cross-attention residual magnitudes + deformable sampling-in-range
#     fractions for the eye decoder and the fixation decoder.
#   Pass B (recorder OFF) — input-perturbation test: re-run with the images shuffled within
#     each batch (gaze untouched); if the regression error barely moves, the image carries no signal.
#
# Diagnosis only — records and reports, never modifies the model, retrains, or fixes anything.
# All needed tensors are already recorded by the existing norm_first+deformable hooks.
#
# Usage (edit the configuration block, then run):
#   python src/notebooks/save_image_reliance.py
#
# Outputs:
#   outputs/image_reliance/{run_name}_reliance.h5   — per-sample residuals / in-range / perturbation
#   outputs/image_reliance/{run_name}_summary.json  — aggregated headline metrics + interpretation

# %%
import os
import sys
import gc
from datetime import datetime, timezone

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)

from src.model.model_io import load_pipeline, load_test_data
from src.training.inference_recorder import InferenceRecorder
from src.eval.image_reliance import (
    run_recording_pass,
    run_perturbation_pass,
    write_reliance_store,
    write_summary,
)


# ── Configuration ─────────────────────────────────────────────────────────────

CKPT_PATH = os.path.join("outputs","2026-09-02","19-45-06")   # <-- checkpoint run directory

RUN_NAME = "mask2former_ms"
OUT_DIR = os.path.join("outputs", "image_reliance")
SAVE_FULL_RESIDUALS = True     # also persist full fixation cross residuals (float16) for offline PCA
EPS_IGNORE = 1e-3              # normalised-units threshold: |shuffled - clean| < eps ⇒ "unchanged"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Load ──────────────────────────────────────────────────────────────────────

def _load_checkpoint_model(pipe, ckpt_path):
    """Build THIS checkpoint's model and load its weights.

    Mirrors ``model_io.load_model`` (strip the ``_orig_mod.`` prefix, ``strict=False``) with two
    robustness fixes for runs whose config has ``training.pretrained_model`` set:
      * that field makes ``build_model`` rebuild the architecture from the *referenced* run's
        config (and return a ``(model, splits)`` tuple) — wrong architecture + a tuple
        ``model_io.load_model`` crashes on. We null the field first so ``build_model`` constructs
        this run's own architecture (the field only seeds weights at train time; we load the final
        weights below anyway), and unpack defensively.
    """
    if pipe.config.training.get("pretrained_model", None) is not None:
        from omegaconf import open_dict
        with open_dict(pipe.config):
            pipe.config.training.pretrained_model = None
    built = pipe.build_model()
    model = built[0] if isinstance(built, tuple) else built
    ckpt = torch.load(os.path.join(ckpt_path, "model.pth"), map_location="cpu")
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Info: {len(missing)} missing keys and {len(unexpected)} unexpected keys.")
    if missing:
        print(f"  Missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected (first 5): {unexpected[:5]}")
    return model


def main(ckpt_path=CKPT_PATH, run_name=RUN_NAME, out_dir=OUT_DIR,
         save_full=SAVE_FULL_RESIDUALS, eps_ignore=EPS_IGNORE, device=DEVICE):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n>>> Image-reliance diagnostic — {run_name}  ({ckpt_path})")

    pipe = load_pipeline(ckpt_path)                    # FR1
    pipe.load_dataset()
    _, _, test_dl = load_test_data(pipe, ckpt_path, return_dataloaders=True)   # test loader only
    model = _load_checkpoint_model(pipe, ckpt_path)
    model.set_phase("Fixation")
    model.to(device)
    model.eval()

    cfg = pipe.config
    ie_type = "dinov3"
    if hasattr(cfg.model, "image_encoder"):
        ie_type = cfg.model.image_encoder.get("type", "dinov3")
    if ie_type != "mask2former":                        # FR2
        print(f"  [WARN] image_encoder.type='{ie_type}' (not 'mask2former'); the suite still "
              "runs — only the in-range test's n_levels depends on the backbone.")

    # ── Pass A: residuals + sampling-in-range (recorder ON) ─────────────────────
    recorder = InferenceRecorder(output_dir=out_dir, enabled=True)
    a_records, support = run_recording_pass(
        model, test_dl, device, recorder, save_full_residuals=save_full)

    # ── Pass B: input-perturbation (recorder OFF) ───────────────────────────────
    model.set_inference_recorder(None)
    batch_size = int(cfg.data.load.batch_size if "load" in cfg.data else cfg.data.batch_size)
    if batch_size < 2:                                  # FR18
        print("  [WARN] batch_size < 2: the perturbation pass will write all-NaN columns.")
    b_records = run_perturbation_pass(model, test_dl, device, eps_ignore=eps_ignore)

    # ── Coverage checks (Data Architecture Integrity) ───────────────────────────
    n_dataset = len(test_dl.path_dataset) if hasattr(test_dl, "path_dataset") else len(test_dl.dataset)
    if len(a_records) != n_dataset:
        print(f"  [WARN] Pass A wrote {len(a_records)} records but dataset has {n_dataset} samples")
    a_ids = [r["sample_idx"] for r in a_records]
    if len(set(a_ids)) != len(a_ids):
        raise ValueError("a sample_idx was emitted twice in Pass A")

    # ── Attrs (FR14) ────────────────────────────────────────────────────────────
    img_size = int(cfg.data.load.img_size if "load" in cfg.data else cfg.data.img_size)
    ss = getattr(model, "image_spatial_shapes", None)
    lsi = getattr(model, "image_level_start_index", None)
    attrs = {
        "run_name": run_name,
        "checkpoint_path": ckpt_path,
        "img_size": img_size,
        "image_encoder_type": str(getattr(model, "image_encoder_type", ie_type)),
        "n_image_levels": int(getattr(model, "n_image_levels", 1)),
        "spatial_shapes": (np.asarray(ss.cpu()).flatten().tolist() if ss is not None else []),
        "level_start_index": (np.asarray(lsi.cpu()).tolist() if lsi is not None else []),
        "K1": int(support["K1"]) if support.get("K1") is not None else 0,
        "model_dim": int(getattr(model, "model_dim", 0)),
        "n_decoder": int(support["n_decoder"]),
        "n_eye_decoder": int(support["n_eye_decoder"]),
        "target_mode": "pred",
        "split": "test",
        "eps_ignore": eps_ignore,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    safe = run_name.replace(" ", "_").replace("/", "_")
    store_path = os.path.join(out_dir, f"{safe}_reliance.h5")
    summary_path = os.path.join(out_dir, f"{safe}_summary.json")

    write_reliance_store(store_path, a_records, b_records, support, attrs)
    print(f"Saved per-sample store -> {store_path}")
    write_summary(summary_path, a_records, b_records, support, attrs)
    print(f"Saved summary -> {summary_path}")

    model.set_inference_recorder(None)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\nDone.")


if __name__ == "__main__":
    main()
