"""Image-reliance diagnostic suite — measurement primitives + pass drivers.

Answers *"does the trained MixerModel actually use the image backbone?"* with three
converging, diagnosis-only measurements over the CocoFreeView **test** split:

1. **Cross-attention residual magnitudes** — how much each decoder's *image*
   cross-attention adds to the residual stream, relative to the *gaze* cross-attention
   (fixation decoder) or the running stream (eye decoder). ``image/gaze << 1`` ⇒ image ignored.
2. **Input-perturbation test** — re-run the forward with the images shuffled within each
   batch (gaze untouched). If the regression error barely moves, the image carries no signal.
3. **Sampling-coordinates-in-range test** — the fraction of deformable sampling locations
   that land inside the ``[0,1]`` feature map. A low fraction means ``grid_sample`` returns
   zero-padded vectors and attention weight is wasted — a mechanism that would *cause*
   image neglect.

Nothing here modifies the model, the recording hooks, or any training artifact — all needed
tensors (``first_cross_res``, ``second_cross_res``, ``cross_attention_res``,
``sampling_locations``) are already recorded by the existing ``norm_first``+deformable hooks
in ``src/model/blocks.py``. See ``src/notebooks/save_image_reliance.py`` for the driver and
``spec/2026-09-03-image-reliance-diagnostic-suite/`` for the contract.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.eval.eval_utils import eval_autoregressive
from src.training.training_utils import move_data_to_device


# blocks.py value names -> meaning for each decoder (fixation: mem1=gaze, mem2=image).
FIX_NORM_KEYS = ("self_attention_res", "first_cross_res", "second_cross_res", "ffn_res")
EYE_NORM_KEYS = ("self_attention_res", "cross_attention_res", "ffn_res")


# ─────────────────────────────────────────────────────────────────────────────
# Recording-support probe
# ─────────────────────────────────────────────────────────────────────────────
def probe_recording_support(model) -> dict:
    """Introspect which residual streams this checkpoint can record.

    ``blocks.py`` records residuals AND sampling locations only inside the ``norm_first=True``
    branch of the *deformable* decoder blocks, so a non-deformable or post-norm checkpoint
    yields nothing. Ported from ``save_activations_eve_real.py`` (FR3).
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


# ─────────────────────────────────────────────────────────────────────────────
# Residual extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_residuals(recorder, module_prefix, n_layers, value_names):
    """Pull recorded residuals for layers ``{module_prefix}.{i}`` from a single-pass recording.

    Returns ``{value_name: [tensor_layer0, tensor_layer1, ...]}`` where each tensor is the
    whole batch ``(B, Nq, D)``. If a list is seen (defensive), the last element is used.
    Ported from ``save_activations_eve_real.py`` (FR5).
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


def residual_norms(res, sample_i, value_names):
    """Per-decode-step L2 norm over the feature dim for one sample.

    ``res`` from :func:`extract_residuals`; returns ``{value_name: (n_layers, Nq)}`` for
    ``res[value_name][layer][sample_i]`` shaped ``(Nq, D)`` (FR5).
    """
    n_layers = len(res[value_names[0]])
    return {
        v: np.stack([
            res[v][l][sample_i].float().norm(dim=-1).cpu().numpy()
            for l in range(n_layers)
        ])
        for v in value_names
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sampling-location extraction + in-range fraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_sampling_locations(recorder, module_prefix, attn_attr, n_layers):
    """Return the per-layer ``sampling_locations`` tensors for a deformable sub-module.

    Each entry is ``(B, Nq, n_heads, n_levels, n_points, 2)`` in ``[0,1]`` space. The recorded
    module name is the dotted ``named_modules()`` path, e.g. ``decoder.{l}.second_cross_attn``
    or ``eye_decoder.{l}.cross_attn`` (FR6). If a list is seen (defensive), the last element
    is used.
    """
    acts = recorder.current_payload["activations"]

    def _last(x):
        return x[-1] if isinstance(x, list) else x

    return [
        _last(acts[f"{module_prefix}.{l}.{attn_attr}"]["sampling_locations"])
        for l in range(n_layers)
    ]


def sampling_in_range_fraction(sampling_locations, query_mask=None, n_levels=None):
    """Fraction of deformable sampling locations landing inside the ``[0,1]`` feature map.

    ``sampling_locations`` is ``(B, Nq, H, L, P, 2)`` in ``[0,1]`` space. A location is
    in-range iff ``0 <= x <= 1 AND 0 <= y <= 1`` (inclusive). The boolean is averaged over the
    query (Nq), head (H) and point (P) axes, per (batch, level) → returns ``(B, L)`` float in
    ``[0,1]``. ``query_mask`` (``(B, Nq)``, True = valid) excludes padded queries (FR6).

    Raises ``ValueError`` if the last dim ≠ 2, the tensor is not 6-D, or (when ``n_levels`` is
    supplied) the level axis ≠ ``n_levels`` (FR16).
    """
    if sampling_locations.shape[-1] != 2:
        raise ValueError(
            f"sampling_locations last dim must be 2 (got {sampling_locations.shape[-1]}); "
            "box references are out of scope"
        )
    if sampling_locations.dim() != 6:
        raise ValueError(
            f"sampling_locations must be 6-D (B, Nq, H, L, P, 2); got shape "
            f"{tuple(sampling_locations.shape)}"
        )
    if n_levels is not None and sampling_locations.shape[3] != n_levels:
        raise ValueError(
            f"sampling_locations level axis ({sampling_locations.shape[3]}) must equal "
            f"n_levels ({n_levels})"
        )

    x = sampling_locations[..., 0]                       # (B, Nq, H, L, P)
    y = sampling_locations[..., 1]
    inside = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)   # bool (B, Nq, H, L, P)

    if query_mask is not None:
        m = query_mask.bool()[:, :, None, None, None]    # (B, Nq, 1, 1, 1)
        num = (inside & m).sum(dim=(1, 2, 4))            # (B, L)
        den = m.expand_as(inside).sum(dim=(1, 2, 4))     # (B, L)
        return num.float() / den.float().clamp(min=1)
    return inside.float().mean(dim=(1, 2, 4))            # (B, L)


# ─────────────────────────────────────────────────────────────────────────────
# Image shuffle + per-sample regression error
# ─────────────────────────────────────────────────────────────────────────────
def shuffle_images_in_batch(image_tensor):
    """Cyclic roll by 1 along the batch axis — a guaranteed derangement (no fixed point).

    Returns ``(permuted_images, perm_index)`` where ``perm_index[i]`` is the source sample
    whose image sample ``i`` received. Requires ``B >= 2`` (FR10).
    """
    B = image_tensor.shape[0]
    if B < 2:
        raise ValueError(f"shuffle requires batch size >= 2, got {B}")
    perm = torch.roll(torch.arange(B, device=image_tensor.device), shifts=1)
    return image_tensor[perm], perm


def per_sample_reg_error(pred_reg, tgt, tgt_mask):
    """Per-sample regression error, mirroring ``eval_metrics.eval_reg`` reduced per row (FR11).

    ``pred_reg`` ``(B, K1, 3)``, ``tgt`` ``(B, K, 3)`` with ``K = K1 - 1``; ``tgt_mask``
    ``(B, K1)`` (True = valid), or ``None`` ⇒ all positions valid. Returns
    ``(coord_err (B,), dur_err (B,))`` as numpy: coord = mean Euclidean distance over valid
    positions, dur = masked MAE on channel 2. Mean over samples equals ``eval_reg`` when all
    rows share the same valid count.
    """
    if tgt_mask is None:
        tgt_mask = torch.ones(pred_reg.size(0), pred_reg.size(1),
                              dtype=torch.bool, device=pred_reg.device)
    y_mask = tgt_mask.bool().unsqueeze(-1)[:, 1:, :]                 # (B, K, 1)
    zero = torch.tensor(0.0, device=pred_reg.device)
    diff = torch.where(y_mask, pred_reg[:, :-1, :3] - tgt, zero)     # (B, K, 3)
    diff_xy = diff[:, :, :2]
    reg_err = torch.sqrt(torch.sum(diff_xy ** 2, dim=-1))           # (B, K)
    dur_err = torch.abs(diff[:, :, 2])                              # (B, K)
    count = y_mask.squeeze(-1).sum(dim=1).clamp(min=1)             # (B,)
    reg_row = (reg_err.sum(dim=1) / count).cpu().numpy()
    dur_row = (dur_err.sum(dim=1) / count).cpu().numpy()
    return reg_row, dur_row


# ─────────────────────────────────────────────────────────────────────────────
# Driver helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pred_len(out, i):
    """First decode step with ``sigmoid(eos) > 0.5``, else the full length (FR8)."""
    eos = out["cls"][i].squeeze(-1).float().cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-eos))
    fired = np.where(prob > 0.5)[0]
    return int(fired[0]) if fired.size else int(len(eos))


def _stim_name(dataloader, sample_idx):
    """Best-effort stimulus label for a gaze ``sample_idx`` (empty string when unavailable)."""
    try:
        img_ds = getattr(dataloader, "dataset", None)
        img_ds = getattr(img_ds, "dataset", img_ds)          # unwrap a Subset
        if hasattr(img_ds, "unique_paths") and hasattr(img_ds, "indices"):
            path = img_ds.unique_paths[int(img_ds.indices[sample_idx])]
            return os.path.basename(str(path))
    except Exception:
        pass
    return ""


def _image_key(inp):
    """The batch key carrying the image tensor consumed by ``encode`` (CoupledDataloader)."""
    if "image_src" in inp:
        return "image_src"
    raise KeyError("no image tensor ('image_src') in the batch; cannot run the perturbation pass")


# ─────────────────────────────────────────────────────────────────────────────
# Pass A — residuals + sampling-in-range (one recorder-on forward per batch)
# ─────────────────────────────────────────────────────────────────────────────
def run_recording_pass(model, dataloader, device, recorder, *, save_full_residuals=True):
    """Pass A: for each test batch, predict autoregressively (recorder off), then run one clean
    causal forward feeding the model's own predictions back as ``tgt`` with the recorder on,
    and harvest cross-attention residual norms + deformable sampling-in-range fractions
    (FR4–FR8). Returns ``(per_sample_records, support)`` with ``support["K1"]`` set."""
    support = probe_recording_support(model)
    n_levels = int(getattr(model, "n_image_levels", 1))
    model.set_inference_recorder(recorder)
    recorder.clear()   # attached but inactive during the autoregressive prediction

    records = []
    K1 = None
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Pass A (residuals + in-range)"):
            inp = move_data_to_device(batch, device)

            # 1) Autoregressive prediction — recorder inactive (no residual overhead).
            out = eval_autoregressive(model, inp, only_last=True)
            pred = out["reg"].clone()                       # (B, K, 3) normalised

            fix = eye = fix_loc = eye_loc = None
            capture = support["fix_ok"] or support["eye_ok"]
            if capture:
                # 2) Clean single-pass forward feeding the model's own scanpath, recorder ON.
                inp2 = dict(inp)
                inp2["tgt"] = pred
                inp2["tgt_mask"] = None
                inp2["in_tgt"] = None
                recorder.start_batch(epoch=0, phase="Fixation",
                                     split="cocofreeview_test", batch_index=0)
                model.encode(**inp2)
                model.decode_fixation(**inp2)
                if K1 is None:
                    K1 = pred.size(1) + 1                   # + start token (decoder query length)
                if support["fix_ok"]:
                    fix = extract_residuals(recorder, "decoder", support["n_decoder"], FIX_NORM_KEYS)
                    fix_loc = extract_sampling_locations(
                        recorder, "decoder", "second_cross_attn", support["n_decoder"])
                if support["eye_ok"]:
                    eye = extract_residuals(recorder, "eye_decoder", support["n_eye_decoder"], EYE_NORM_KEYS)
                    eye_loc = extract_sampling_locations(
                        recorder, "eye_decoder", "cross_attn", support["n_eye_decoder"])

            src_mask = inp.get("src_mask")
            for i in range(inp["src"].size(0)):
                idx = int(inp["sample_idx"][i])
                T = int(src_mask[i].sum()) if src_mask is not None else inp["src"].size(1)
                rec = {
                    "sample_idx": idx,
                    "src_len": T,
                    "stimulus_name": _stim_name(dataloader, idx),
                    "pred_len": _pred_len(out, i),
                }
                if support["fix_ok"]:
                    rec["dec_norms"] = residual_norms(fix, i, FIX_NORM_KEYS)        # {v:(n_dec,K1)}
                    rec["dec_inrange"] = np.stack([
                        sampling_in_range_fraction(fix_loc[l], n_levels=n_levels)[i].cpu().numpy()
                        for l in range(support["n_decoder"])
                    ])                                                              # (n_dec, L)
                    if save_full_residuals:
                        rec["full"] = {
                            v: np.stack([
                                fix[v][l][i].float().cpu().numpy().astype(np.float16)
                                for l in range(support["n_decoder"])
                            ])
                            for v in ("first_cross_res", "second_cross_res")
                        }                                                           # {v:(n_dec,K1,D)}
                if support["eye_ok"]:
                    all_eye = residual_norms(eye, i, EYE_NORM_KEYS)
                    rec["eye_norms"] = {v: all_eye[v][:, :T] for v in EYE_NORM_KEYS}  # trim to T
                    qmask = src_mask[i:i + 1] if src_mask is not None else None
                    rec["eye_inrange"] = np.stack([
                        sampling_in_range_fraction(eye_loc[l][i:i + 1], qmask, n_levels=n_levels)[0].cpu().numpy()
                        for l in range(support["n_eye_decoder"])
                    ])                                                              # (n_eye, L)
                records.append(rec)

            if capture:
                recorder.clear()

    if not (support["fix_ok"] or support["eye_ok"]):
        print("  [WARN] no residual/in-range streams captured (see probe warnings); "
              "Pass A wrote prediction/length metadata only.")
    return records, {**support, "K1": K1}


# ─────────────────────────────────────────────────────────────────────────────
# Pass B — input-perturbation (recorder off)
# ─────────────────────────────────────────────────────────────────────────────
def run_perturbation_pass(model, dataloader, device, *, eps_ignore=1e-3):
    """Pass B: per-sample normalised regression error with the true image vs. the image tensor
    shuffled within the batch (gaze/masks/tgt identical). Recorder is untouched (off). Returns
    per-sample records (FR9–FR11)."""
    records = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Pass B (perturbation)"):
            inp = move_data_to_device(batch, device)
            B = inp["src"].size(0)

            out_c = eval_autoregressive(model, inp, only_last=True)
            rc, dc = per_sample_reg_error(out_c["reg"], inp["tgt"], inp.get("tgt_mask"))

            if B >= 2:
                img_key = _image_key(inp)
                perm_img, perm = shuffle_images_in_batch(inp[img_key])
                inp_s = dict(inp)
                inp_s[img_key] = perm_img
                out_s = eval_autoregressive(model, inp_s, only_last=True)
                rs, ds = per_sample_reg_error(out_s["reg"], inp["tgt"], inp.get("tgt_mask"))
                perm = perm.cpu().numpy()
            else:
                print("  [WARN] trailing batch of size 1 cannot be shuffled; "
                      "writing NaN perturbation columns for it.")
                rs = np.full(B, np.nan, np.float32)
                ds = np.full(B, np.nan, np.float32)
                perm = np.full(B, -1, np.int64)

            for i in range(B):
                records.append({
                    "sample_idx": int(inp["sample_idx"][i]),
                    "reg_error_clean": float(rc[i]),
                    "reg_error_shuffled": float(rs[i]),
                    "dur_error_clean": float(dc[i]),
                    "dur_error_shuffled": float(ds[i]),
                    "perm_index": int(perm[i]),
                })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Persistence — per-sample HDF5 store
# ─────────────────────────────────────────────────────────────────────────────
def write_reliance_store(path, pass_a_records, pass_b_records, support, attrs):
    """Write the per-sample HDF5 store (mode ``"w"``, single group ``/reliance``).

    Pass A supplies residual norms, in-range fractions and (optionally) full cross residuals;
    Pass B supplies the perturbation columns, aligned to Pass A **by ``sample_idx``** (dict
    lookup, never row position). Eye arrays are NaN-padded to ``T_max``; ``dec_*`` are dense.
    See the HDF5 layout block in requirements.md (FR12/FR14).
    """
    N = len(pass_a_records)
    n_dec = int(support["n_decoder"])
    n_eye = int(support["n_eye_decoder"])
    fix_ok = bool(support["fix_ok"])
    eye_ok = bool(support["eye_ok"])
    save_full = fix_ok and any("full" in r for r in pass_a_records)
    str_dt = h5py.string_dtype()

    sample_idx = np.array([int(r["sample_idx"]) for r in pass_a_records], np.int32)
    stim = np.array([str(r.get("stimulus_name", "")) for r in pass_a_records], dtype=object)
    pred_len = np.array([int(r["pred_len"]) for r in pass_a_records], np.int32)
    src_len = np.array([int(r["src_len"]) for r in pass_a_records], np.int32)

    # Pass B lookup by sample_idx (alignment cannot be bypassed by row position).
    b_by_idx = {int(r["sample_idx"]): r for r in (pass_b_records or [])}

    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.require_group("/reliance")
        g.create_dataset("sample_idx", data=sample_idx)
        g.create_dataset("stimulus_name", data=stim, dtype=str_dt)
        g.create_dataset("pred_len", data=pred_len)
        g.create_dataset("src_len", data=src_len)

        K1 = model_dim = n_lvl = None

        # ── Pass A: fixation-decoder residual norms — (N, n_dec, K1_max), NaN-padded.
        # K1 (= pred.size(1)+1) is the fixation-decoder query length, which depends on each
        # batch's max fixation count, so it varies across batches — pad to the global max.
        if fix_ok:
            K1 = max(int(r["dec_norms"][FIX_NORM_KEYS[0]].shape[1]) for r in pass_a_records)
            for v in FIX_NORM_KEYS:
                arr = np.full((N, n_dec, K1), np.nan, np.float32)
                for i, r in enumerate(pass_a_records):
                    k = r["dec_norms"][v].shape[1]
                    arr[i, :, :k] = r["dec_norms"][v]
                g.create_dataset(f"dec_{v}_norm", data=arr)

        # ── Pass A: eye-decoder residual norms — (N, n_eye, T_max), NaN-padded per src_len.
        if eye_ok:
            T_max = max(int(r["src_len"]) for r in pass_a_records)
            for v in EYE_NORM_KEYS:
                arr = np.full((N, n_eye, T_max), np.nan, np.float32)
                for i, r in enumerate(pass_a_records):
                    L = int(r["src_len"])
                    arr[i, :, :L] = r["eye_norms"][v]
                g.create_dataset(f"eye_{v}_norm", data=arr)

        # ── Pass A: sampling-in-range fractions.
        if fix_ok:
            n_lvl = int(pass_a_records[0]["dec_inrange"].shape[1])
            arr = np.stack([r["dec_inrange"] for r in pass_a_records]).astype(np.float32)
            g.create_dataset("dec_inrange", data=arr)                       # (N, n_dec, n_lvl)
        if eye_ok:
            n_lvl = int(pass_a_records[0]["eye_inrange"].shape[1])
            arr = np.stack([r["eye_inrange"] for r in pass_a_records]).astype(np.float32)
            g.create_dataset("eye_inrange", data=arr)                       # (N, n_eye, n_lvl)

        # ── Pass A: optional full fixation cross residuals — (N, n_dec, K1, D) fp16.
        if save_full:
            model_dim = int(pass_a_records[0]["full"]["first_cross_res"].shape[-1])
            for v in ("first_cross_res", "second_cross_res"):
                arr = np.full((N, n_dec, K1, model_dim), np.nan, np.float16)
                for i, r in enumerate(pass_a_records):
                    k = r["full"][v].shape[1]
                    arr[i, :, :k, :] = r["full"][v]
                g.create_dataset(f"dec_{v}", data=arr)

        # ── Pass B: perturbation columns, aligned by sample_idx.
        reg_c = np.full(N, np.nan, np.float32)
        reg_s = np.full(N, np.nan, np.float32)
        dur_c = np.full(N, np.nan, np.float32)
        dur_s = np.full(N, np.nan, np.float32)
        perm = np.full(N, -1, np.int32)
        for i, idx in enumerate(sample_idx):
            b = b_by_idx.get(int(idx))
            if b is not None:
                reg_c[i] = b["reg_error_clean"]
                reg_s[i] = b["reg_error_shuffled"]
                dur_c[i] = b["dur_error_clean"]
                dur_s[i] = b["dur_error_shuffled"]
                perm[i] = b["perm_index"]
        g.create_dataset("reg_error_clean", data=reg_c)
        g.create_dataset("reg_error_shuffled", data=reg_s)
        g.create_dataset("dur_error_clean", data=dur_c)
        g.create_dataset("dur_error_shuffled", data=dur_s)
        g.create_dataset("perm_index", data=perm)

        # ── Group attrs (FR14).
        merged = dict(attrs)
        merged.update({
            "n_decoder": n_dec,
            "n_eye_decoder": n_eye,
            "fix_residuals_saved": fix_ok,
            "eye_residuals_saved": eye_ok,
            "full_residuals_saved": bool(save_full),
            "inrange_saved": bool(fix_ok or eye_ok),
            "target_mode": attrs.get("target_mode", "pred"),
            "split": attrs.get("split", "test"),
            # Decode legend.
            "fix_first_cross": "gaze",
            "fix_second_cross": "image",
            "eye_cross": "image",
        })
        if K1 is not None:
            merged["K1"] = K1          # the padded K1_max actually written (batch-dependent)
        if model_dim is not None:
            merged.setdefault("model_dim", model_dim)
        # Drop any None-valued attrs (h5py cannot store None).
        for k in list(merged.keys()):
            if merged[k] is None:
                merged[k] = ""
        g.attrs.update(merged)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence — aggregated summary JSON + printed table
# ─────────────────────────────────────────────────────────────────────────────
def _nanmean(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.nanmean(a)) if a.size and not np.all(np.isnan(a)) else float("nan")


def write_summary(path, pass_a_records, pass_b_records, support, attrs):
    """Compute the FR13 headline scalars + interpretation strings, write JSON, print a table,
    and return the summary dict."""
    fix_ok = bool(support["fix_ok"])
    eye_ok = bool(support["eye_ok"])
    n_dec = int(support["n_decoder"])
    n_eye = int(support["n_eye_decoder"])
    eps_ignore = float(attrs.get("eps_ignore", 1e-3))

    summary = {
        "run_name": attrs.get("run_name", ""),
        "checkpoint_path": attrs.get("checkpoint_path", ""),
        "n_samples": len(pass_a_records),
        "eps_ignore": eps_ignore,
        "residuals": {},
        "perturbation": {},
        "in_range": {},
    }

    # ── Test 1: residual reliance ─────────────────────────────────────────────
    if fix_ok and pass_a_records:
        fix_layers = []
        for l in range(n_dec):
            # Aggregate over active decode steps only (0..pred_len; position 0 = start token).
            first = np.concatenate([r["dec_norms"]["first_cross_res"][l][:int(r["pred_len"]) + 1]
                                    for r in pass_a_records])
            second = np.concatenate([r["dec_norms"]["second_cross_res"][l][:int(r["pred_len"]) + 1]
                                     for r in pass_a_records])
            ratio = second / np.clip(first, 1e-9, None)
            fix_layers.append({
                "layer": l,
                "gaze_mean": float(first.mean()),
                "image_mean": float(second.mean()),
                "image_over_gaze_ratio": float(ratio.mean()),
            })
        overall = float(np.mean([d["image_over_gaze_ratio"] for d in fix_layers]))
        summary["residuals"]["fixation_decoder"] = fix_layers
        summary["residuals"]["fixation_mean_image_over_gaze"] = overall
        summary["residuals"]["interpretation"] = (
            f"image/gaze ratio {overall:.3f} << 1 ⇒ image contribution negligible"
            if overall < 0.1 else
            f"image/gaze ratio {overall:.3f} ⇒ image contributes to the fixation decoder"
        )
    if eye_ok and pass_a_records:
        eye_layers = []
        for l in range(n_eye):
            cross = np.concatenate([r["eye_norms"]["cross_attention_res"][l] for r in pass_a_records])
            selfa = np.concatenate([r["eye_norms"]["self_attention_res"][l] for r in pass_a_records])
            eye_layers.append({
                "layer": l,
                "image_cross_mean": float(cross.mean()),
                "self_mean": float(selfa.mean()),
                "image_over_self_ratio": float(cross.mean() / max(selfa.mean(), 1e-9)),
            })
        summary["residuals"]["eye_decoder"] = eye_layers

    # ── Test 2: perturbation ──────────────────────────────────────────────────
    if pass_b_records:
        clean = np.array([r["reg_error_clean"] for r in pass_b_records], np.float64)
        shuf = np.array([r["reg_error_shuffled"] for r in pass_b_records], np.float64)
        valid = ~np.isnan(shuf) & ~np.isnan(clean)
        mean_clean = _nanmean(clean)
        mean_shuf = _nanmean(shuf)
        abs_delta = mean_shuf - mean_clean
        rel_delta = abs_delta / mean_clean if mean_clean not in (0.0, float("nan")) else float("nan")
        if valid.any():
            frac_unchanged = float(np.mean(np.abs(shuf[valid] - clean[valid]) < eps_ignore))
        else:
            frac_unchanged = float("nan")
        summary["perturbation"] = {
            "mean_reg_error_clean": mean_clean,
            "mean_reg_error_shuffled": mean_shuf,
            "abs_delta": float(abs_delta),
            "rel_delta": float(rel_delta),
            "frac_samples_below_eps": frac_unchanged,
            "n_valid": int(valid.sum()),
            "interpretation": (
                f"shuffling images moved reg error by {abs_delta:+.4f} "
                f"({frac_unchanged:.1%} of samples changed < {eps_ignore}) ⇒ "
                + ("image carries no signal" if abs(abs_delta) < eps_ignore
                   else "image carries signal")
            ),
        }

    # ── Test 3: in-range ──────────────────────────────────────────────────────
    if fix_ok and pass_a_records:
        arr = np.stack([r["dec_inrange"] for r in pass_a_records])   # (N, n_dec, n_lvl)
        per_level = arr.mean(axis=(0, 1))                            # (n_lvl,)
        summary["in_range"]["fixation_decoder_per_level"] = [float(x) for x in per_level]
        summary["in_range"]["fixation_interpretation"] = _inrange_interp(per_level)
    if eye_ok and pass_a_records:
        arr = np.stack([r["eye_inrange"] for r in pass_a_records])
        per_level = arr.mean(axis=(0, 1))
        summary["in_range"]["eye_decoder_per_level"] = [float(x) for x in per_level]
        summary["in_range"]["eye_interpretation"] = _inrange_interp(per_level)

    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        json.dump(summary, fp, indent=2)

    _print_summary(summary, support)
    return summary


def _inrange_interp(per_level):
    lo = float(np.min(per_level))
    if lo < 0.5:
        return (f"min per-level in-range {lo:.2f} < 0.5 ⇒ deformable attention samples off-map "
                "(grid_sample zero-pads; attention weight wasted) — a mechanistic cause of neglect")
    return f"min per-level in-range {lo:.2f} ⇒ sampling is healthy (on-map)"


def _print_summary(summary, support):
    print("\n" + "=" * 72)
    print(f"IMAGE-RELIANCE SUMMARY — {summary.get('run_name', '')}  (N={summary['n_samples']})")
    print("=" * 72)

    res = summary.get("residuals", {})
    if "fixation_decoder" in res:
        print("\n[Test 1] Fixation decoder residual norms (mean over samples × decode-steps):")
        print("  layer |  gaze(1st) | image(2nd) | image/gaze")
        for d in res["fixation_decoder"]:
            print(f"  {d['layer']:5d} | {d['gaze_mean']:10.4f} | {d['image_mean']:10.4f} "
                  f"| {d['image_over_gaze_ratio']:10.4f}")
        print(f"  -> {res.get('interpretation', '')}")
    if "eye_decoder" in res:
        print("\n  Eye decoder (gaze->image) residual norms over valid frames:")
        print("  layer |  self     | image(cross) | image/self")
        for d in res["eye_decoder"]:
            print(f"  {d['layer']:5d} | {d['self_mean']:9.4f} | {d['image_cross_mean']:12.4f} "
                  f"| {d['image_over_self_ratio']:10.4f}")

    pert = summary.get("perturbation", {})
    if pert:
        print("\n[Test 2] Input-perturbation (image shuffled within batch):")
        print(f"  mean reg_error clean    : {pert['mean_reg_error_clean']:.4f}")
        print(f"  mean reg_error shuffled : {pert['mean_reg_error_shuffled']:.4f}")
        print(f"  abs delta / rel delta   : {pert['abs_delta']:+.4f} / {pert['rel_delta']:+.2%}")
        print(f"  frac samples < eps      : {pert['frac_samples_below_eps']:.2%}")
        print(f"  -> {pert.get('interpretation', '')}")

    ir = summary.get("in_range", {})
    if "fixation_decoder_per_level" in ir:
        vals = ", ".join(f"{x:.3f}" for x in ir["fixation_decoder_per_level"])
        print(f"\n[Test 3] Fixation in-range per level: [{vals}]")
        print(f"  -> {ir.get('fixation_interpretation', '')}")
    if "eye_decoder_per_level" in ir:
        vals = ", ".join(f"{x:.3f}" for x in ir["eye_decoder_per_level"])
        print(f"  Eye in-range per level: [{vals}]")
        print(f"  -> {ir.get('eye_interpretation', '')}")
    print("=" * 72 + "\n")
