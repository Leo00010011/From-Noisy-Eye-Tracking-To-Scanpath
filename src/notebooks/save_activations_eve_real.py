# %% [markdown]
# # Save Activations (EVE real noise)
# Superset of ``save_predictions_eve_real.py``: runs the same autoregressive fixation
# prediction on the EyeNet real-noise cache AND captures the decoders' internal
# activations — most importantly the **cross-attention residuals** — so we can measure
# which features each decoder actually uses.
#
# Motivating question: *is the model ignoring the image features?* In the fixation
# decoder (``DeformableDoubleInputDecoder``) every layer adds two cross-attention
# contributions into the residual stream:
#   - ``first_cross_res``  — scanpath tokens attending to the **gaze** encoding
#   - ``second_cross_res`` — scanpath tokens attending to the **image** patches
# Comparing the L2 magnitude these two add to the residual stream, per layer and per
# fixation, directly quantifies how much the decoder leans on image vs gaze. The same
# is captured on the encoder side for the ``eye_decoder`` (gaze tokens attending to the
# image, ``cross_attention_res``).
#
# How the residuals are obtained: ``src/model/blocks.py`` already records these tensors
# via ``record_module_value`` whenever an ``InferenceRecorder`` is attached AND active —
# but ONLY inside the ``norm_first=True`` branch of the *deformable* decoder blocks. The
# default architecture (``use_deformable_*_decoder=True``, ``norm_first=True``) satisfies
# this; the script detects and warns if a checkpoint does not.
#
# Rather than record across the ragged autoregressive loop (growing sequence, list
# accumulation), we first predict the scanpath with the recorder OFF, then do ONE clean
# causal forward feeding the model's own predicted scanpath back as ``tgt`` with the
# recorder ON. That yields clean fixed-shape ``(B, K+1, D)`` residuals whose position i
# corresponds to predicting fixation i (position 0 = start token).
#
# Usage (edit the configuration block below, then run):
#   python src/notebooks/save_activations_eve_real.py
#
# Outputs, per model:
#   outputs/eve_real_noise/{name}.h5             — standard RealNoiseInferenceStore
#   outputs/eve_real_noise/{name}_activations.h5 — exp_key-keyed residual store (below)

# %%
import os
import sys
import gc
from datetime import datetime, timezone

import h5py
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
from src.training.inference_recorder import InferenceRecorder
from src.training.pipeline_builder import PipelineBuilder
from src.training.training_utils import move_data_to_device


# ── Configuration ─────────────────────────────────────────────────────────────

BUNDLE_DIR = "/mnt/scratch/leonardo.ulloa/5519804/data/bundle"
CACHE_PATH = "data/eve_real_noise/eyenet_gaze_cache.h5"
OUT_DIR = os.path.join("outputs", "eve_real_noise")
EYENET_SPLIT = None   # None (both), "val", or "test"

# Also persist the full cross-attention residual tensors (fixation decoder only:
# gaze_cross + image_cross), stored as float16. Per-position L2 norms are always saved
# regardless; the full tensors let you inspect per-dimension / do PCA offline.
#   size ~= N * n_decoder * (K+1) * model_dim * 2 (residuals) * 2 bytes
SAVE_FULL_RESIDUALS = True

ckpt_paths = [
    os.path.join("outputs", "2026-07-24", "16-17-19"),
]
names = [
    "eve_real_duration_dist",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ckpt_img_size(cfg) -> int:
    return int(cfg.data.load.img_size if "load" in cfg.data else cfg.data.img_size)


def _invert_src_to_px(src_row: torch.Tensor, transforms) -> np.ndarray:
    """Invert a single normalised src row (T, 3) back to pixel/ms space (see the
    original save_predictions_eve_real.py for the rationale)."""
    s = src_row.unsqueeze(0).clone()
    for t in reversed(transforms):
        if getattr(t, "key", None) == "x" and hasattr(t, "inverse"):
            s = t.inverse(s, None, "x")
    return s.squeeze(0).cpu().numpy()


def _invert_denoise_to_px(denoise_row: torch.Tensor, transforms) -> np.ndarray:
    """Invert a single denoise-head row (T, 2) of normalised coords back to pixels."""
    d = denoise_row.unsqueeze(0).clone()
    for t in reversed(transforms):
        if (getattr(t, "key", None) == "x" and getattr(t, "mode", None) == "coords"
                and hasattr(t, "inverse")):
            d = t.inverse(d, None, "x")
    return d.squeeze(0).cpu().numpy()


def _row_norm(t: torch.Tensor) -> np.ndarray:
    """L2 norm over the feature dimension: (L, D) -> (L,)."""
    return t.float().norm(dim=-1).cpu().numpy()


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

    assert len(gaze_ds) == len(img_ds), f"{len(gaze_ds)} gaze vs {len(img_ds)} img samples"
    assert all(gaze_ds.exp_key_at(i) == img_ds.exp_key_at(i) for i in range(len(gaze_ds))), \
        "gaze/img datasets disagree on exp_key ordering"

    dl = CoupledDataloader(
        gaze_ds, Subset(img_ds, torch.arange(len(img_ds))),
        batch_size=cfg.data.load.batch_size, shuffle=False,               # FR11.3
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

    return cfg, model, gaze_ds, dl, bundle


def probe_recording_support(model):
    """Introspect which residual streams this checkpoint can record.

    ``blocks.py`` records residuals only inside the ``norm_first=True`` branch of the
    *deformable* decoder blocks, so a non-deformable or post-norm checkpoint yields no
    residuals. Returns a dict describing what is capturable.
    """
    norm_first = bool(getattr(model, "norm_first", False))
    fix_deform = bool(getattr(model, "use_deformable_fixation_decoder", False))
    eye_deform = bool(getattr(model, "use_deformable_eye_decoder", False))
    n_decoder = int(getattr(model, "n_decoder", 0))
    n_eye_decoder = int(getattr(model, "n_eye_decoder", 0))
    has_eye = hasattr(model, "eye_decoder") and n_eye_decoder > 0

    fix_ok = norm_first and fix_deform and n_decoder > 0
    eye_ok = norm_first and eye_deform and has_eye

    info = {
        "norm_first": norm_first,
        "fix_deform": fix_deform,
        "eye_deform": eye_deform,
        "n_decoder": n_decoder,
        "n_eye_decoder": n_eye_decoder,
        "fix_ok": fix_ok,
        "eye_ok": eye_ok,
    }
    if not norm_first:
        print("  [WARN] checkpoint has norm_first=False; blocks.py records residuals "
              "only in the norm_first branch — NO residuals will be captured.")
    if not fix_deform:
        print("  [WARN] fixation decoder is non-deformable; only DeformableDoubleInput"
              "Decoder records residuals — fixation residuals unavailable.")
    if has_eye and not eye_deform:
        print("  [WARN] eye_decoder is non-deformable; eye residuals unavailable.")
    return info


def extract_residuals(recorder, module_prefix, n_layers, value_names):
    """Pull recorded residuals for layers ``{module_prefix}.{i}`` from a single
    single-pass recording. Returns {value_name: [tensor_layer0, tensor_layer1, ...]}.

    Each value is a single tensor (single forward, no list accumulation); if a list is
    seen (defensive), the last element is used.
    """
    acts = recorder.current_payload["activations"]
    out = {name: [] for name in value_names}
    for layer in range(n_layers):
        bucket = acts.get(f"{module_prefix}.{layer}", {})
        for name in value_names:
            val = bucket.get(name)
            if isinstance(val, list):
                val = val[-1]
            out[name].append(val)
    return out


# ── Activation store writer (self-contained; exp_key-keyed) ─────────────────────

def save_activation_store(path, records, support, K1, model_dim, attrs, save_full):
    """Write an exp_key-keyed HDF5 of residual norms (+ optional full cross residuals).

    records: list of dicts, each with
      exp_key, eyenet_split, eve_split, pred_len, src_len,
      dec_norms{value_name: (n_decoder, K1)},          # fixation decoder, fixed length
      eye_norms{value_name: (n_eye_decoder, T)},       # eye decoder, variable T
      [full{value_name: (n_decoder, K1, D)}]           # optional, float16
    """
    N = len(records)
    n_dec = support["n_decoder"]
    n_eye = support["n_eye_decoder"]
    str_dt = h5py.string_dtype()

    exp_keys = np.array([str(r["exp_key"]) for r in records], dtype=object)
    eyenet = np.array([str(r["eyenet_split"]) for r in records], dtype=object)
    eve = np.array([str(r["eve_split"]) for r in records], dtype=object)
    pred_len = np.array([int(r["pred_len"]) for r in records], np.int32)
    src_len = np.array([int(r["src_len"]) for r in records], np.int32)

    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.require_group("/activations")
        g.create_dataset("exp_keys", data=exp_keys, dtype=str_dt)
        g.create_dataset("eyenet_split", data=eyenet, dtype=str_dt)
        g.create_dataset("eve_split", data=eve, dtype=str_dt)
        g.create_dataset("pred_len", data=pred_len)
        g.create_dataset("src_len", data=src_len)

        # Fixation decoder residual norms — (N, n_decoder, K1), fixed length.
        if support["fix_ok"]:
            for vname in FIX_NORM_KEYS:
                arr = np.full((N, n_dec, K1), np.nan, np.float32)
                for i, r in enumerate(records):
                    arr[i] = r["dec_norms"][vname]
                g.create_dataset(f"dec_{vname}_norm", data=arr)

        # Eye decoder residual norms — (N, n_eye, T_max), NaN-padded per src_len.
        if support["eye_ok"]:
            T_max = max(int(r["src_len"]) for r in records)
            for vname in EYE_NORM_KEYS:
                arr = np.full((N, n_eye, T_max), np.nan, np.float32)
                for i, r in enumerate(records):
                    L = int(r["src_len"])
                    arr[i, :, :L] = r["eye_norms"][vname]
                g.create_dataset(f"eye_{vname}_norm", data=arr)

        # Optional full fixation cross-attention residuals — (N, n_decoder, K1, D) fp16.
        if save_full and support["fix_ok"]:
            for vname in ("first_cross_res", "second_cross_res"):
                arr = np.full((N, n_dec, K1, model_dim), np.nan, np.float16)
                for i, r in enumerate(records):
                    arr[i] = r["full"][vname]
                g.create_dataset(f"dec_{vname}", data=arr)

        merged = dict(attrs)
        merged.update({
            "n_decoder": n_dec,
            "n_eye_decoder": n_eye,
            "K1": int(K1),
            "model_dim": int(model_dim),
            "fix_residuals_saved": bool(support["fix_ok"]),
            "eye_residuals_saved": bool(support["eye_ok"]),
            "full_residuals_saved": bool(save_full and support["fix_ok"]),
            # Names decoded for downstream clarity (call-site ordering in mixer_model).
            "fix_first_cross": "gaze",   # scanpath tokens attend to gaze encoding
            "fix_second_cross": "image",  # scanpath tokens attend to image patches
            "eye_cross": "image",         # gaze tokens attend to image patches
        })
        g.attrs.update(merged)


# blocks.py value names -> what they mean for the fixation decoder (mem1=gaze, mem2=img)
FIX_NORM_KEYS = ("self_attention_res", "first_cross_res", "second_cross_res", "ffn_res")
EYE_NORM_KEYS = ("self_attention_res", "cross_attention_res", "ffn_res")


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
    if hasattr(model, "disable_kv_cache"):
        model.disable_kv_cache()   # analysis pass is a full causal forward, not incremental

    support = probe_recording_support(model)
    capture = support["fix_ok"] or support["eye_ok"]
    model_dim = int(getattr(model, "model_dim", 0))

    recorder = InferenceRecorder(output_dir=OUT_DIR, enabled=True)
    model.set_inference_recorder(recorder)
    recorder.clear()   # attached but inactive during the autoregressive prediction

    has_denoise = hasattr(model, "denoise_head")   # decode_denoise() returns {} without it
    transforms = gaze_ds.transforms
    records = []       # for RealNoiseInferenceStore (predictions)
    act_records = []   # for the activation store
    K1 = None

    with torch.no_grad():
        for batch in tqdm(dl, desc="Saving"):
            inp = move_data_to_device(batch, device)

            # 1) Autoregressive prediction — recorder inactive (no residual overhead).
            out = eval_autoregressive(model, inp, only_last=True)
            pred_reg_norm = out["reg"].clone()   # (B, K, 3) normalised; clone before inversion

            denoise_norm = model.decode_denoise(**inp)["denoise"] if has_denoise else None
            inp_px, out_px = invert_transforms(inp, out, dl, remove_outliers=True)

            # 2) Clean single-pass forward feeding the model's own scanpath, recorder ON.
            if capture:
                inp2 = dict(inp)
                inp2["tgt"] = pred_reg_norm
                inp2["tgt_mask"] = None
                inp2["in_tgt"] = None
                recorder.start_batch(epoch=0, phase="Fixation", split="eve_real", batch_index=0)
                model.encode(**inp2)
                model.decode_fixation(**inp2)

                if K1 is None:
                    K1 = pred_reg_norm.size(1) + 1   # + start token
                fix_res = (extract_residuals(recorder, "decoder", support["n_decoder"], FIX_NORM_KEYS)
                           if support["fix_ok"] else None)
                eye_res = (extract_residuals(recorder, "eye_decoder", support["n_eye_decoder"], EYE_NORM_KEYS)
                           if support["eye_ok"] else None)

            # 3) Per-sample records.
            for i in range(inp["src"].size(0)):
                idx = int(inp["sample_idx"][i])
                key = gaze_ds.exp_key_at(idx)
                T = int(inp["src_mask"][i].sum()) if inp["src_mask"] is not None else inp["src"].size(1)

                eos = out_px["cls"][i].squeeze(-1).cpu().numpy()
                prob = 1.0 / (1.0 + np.exp(-eos))
                fired = np.where(prob > 0.5)[0]
                pred_len = int(fired[0]) if fired.size else len(eos)

                rec = {
                    "exp_key": key,
                    "eyenet_split": gaze_ds.eyenet_split_at(idx),
                    "eve_split": gaze_ds.eve_split_at(idx),
                    "pred_scanpath": out_px["reg"][i].cpu().numpy(),
                    "eos_logit": eos,
                    "src_px": _invert_src_to_px(inp["src"][i, :T], transforms),
                    "src_len": T,
                    "frame_indices": gaze_ds.frame_indices_at(idx)[:T],
                }
                if has_denoise:
                    rec["denoise_px"] = _invert_denoise_to_px(denoise_norm[i, :T, :2], transforms)
                records.append(rec)

                if capture:
                    a = {
                        "exp_key": key,
                        "eyenet_split": gaze_ds.eyenet_split_at(idx),
                        "eve_split": gaze_ds.eve_split_at(idx),
                        "pred_len": pred_len,
                        "src_len": T,
                    }
                    if support["fix_ok"]:
                        a["dec_norms"] = {
                            v: np.stack([_row_norm(fix_res[v][l][i]) for l in range(support["n_decoder"])])
                            for v in FIX_NORM_KEYS
                        }
                        if SAVE_FULL_RESIDUALS:
                            a["full"] = {
                                v: np.stack([
                                    fix_res[v][l][i].float().cpu().numpy().astype(np.float16)
                                    for l in range(support["n_decoder"])
                                ])
                                for v in ("first_cross_res", "second_cross_res")
                            }
                    if support["eye_ok"]:
                        a["eye_norms"] = {
                            v: np.stack([_row_norm(eye_res[v][l][i, :T]) for l in range(support["n_eye_decoder"])])
                            for v in EYE_NORM_KEYS
                        }
                    act_records.append(a)

            if capture:
                recorder.clear()

    # ── Save predictions (RealNoiseInferenceStore) ──────────────────────────────
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
    print(f"Saved {len(records)} prediction records -> {out_path}")

    # ── Save activations ────────────────────────────────────────────────────────
    if capture and act_records:
        act_attrs = dict(attrs)
        act_attrs.update({
            "run_name": name,
            "checkpoint_path": ckpt_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        act_path = os.path.join(OUT_DIR, f"{safe_name}_activations.h5")
        save_activation_store(
            act_path, act_records, support, K1=K1, model_dim=model_dim,
            attrs=act_attrs, save_full=SAVE_FULL_RESIDUALS,
        )
        print(f"Saved {len(act_records)} activation records -> {act_path}")

        # ── Headline summary: gaze vs image residual magnitude ──────────────────
        if support["fix_ok"]:
            print("\n  Fixation decoder — mean residual L2 norm over active positions "
                  "(pos 0..pred_len), across all samples:")
            print("    layer |   self   |   gaze(1st) |  image(2nd) |  ffn   | image/gaze")
            for l in range(support["n_decoder"]):
                means = {}
                for v in FIX_NORM_KEYS:
                    vals = []
                    for r in act_records:
                        pl = int(r["pred_len"])
                        vals.append(r["dec_norms"][v][l, :pl + 1])
                    means[v] = float(np.concatenate(vals).mean()) if vals else float("nan")
                ratio = means["second_cross_res"] / max(means["first_cross_res"], 1e-9)
                print(f"    {l:5d} | {means['self_attention_res']:8.3f} | "
                      f"{means['first_cross_res']:11.3f} | {means['second_cross_res']:11.3f} | "
                      f"{means['ffn_res']:6.3f} | {ratio:9.3f}")
            print("  (image/gaze << 1 supports the 'image is being ignored' hypothesis.)")

        if support["eye_ok"]:
            print("\n  Eye decoder (gaze->image) — mean residual L2 norm over valid gaze frames:")
            print("    layer |   self   |  image(cross) |  ffn")
            for l in range(support["n_eye_decoder"]):
                means = {}
                for v in EYE_NORM_KEYS:
                    vals = [r["eye_norms"][v][l, :int(r["src_len"])] for r in act_records]
                    means[v] = float(np.concatenate(vals).mean()) if vals else float("nan")
                print(f"    {l:5d} | {means['self_attention_res']:8.3f} | "
                      f"{means['cross_attention_res']:13.3f} | {means['ffn_res']:6.3f}")
    elif not capture:
        print("  [WARN] no residuals captured for this checkpoint (see warnings above).")

    model.set_inference_recorder(None)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

print("\nDone.")
print("Load predictions:  store = RealNoiseInferenceStore.load('<name>.h5')")
print("Load activations:  with h5py.File('<name>_activations.h5') as f: g = f['/activations']")
print("  Key residual norms: g['dec_first_cross_res_norm'] (gaze), "
      "g['dec_second_cross_res_norm'] (image) — shape (N, n_decoder, K+1).")
