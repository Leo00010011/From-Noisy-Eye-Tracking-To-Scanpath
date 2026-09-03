"""Image-reliance diagnostic suite — unit tests for ``src/eval/image_reliance.py``.

Groups mirror ``spec/2026-09-03-image-reliance-diagnostic-suite/validation.md``. Groups 1–8
are CPU-only, synthetic tensors + a stub recorder payload (no checkpoint, no GPU, no network);
Group 5 adds one tiny real ``MixerModel`` (n_image_levels=1, DINOv3 stub) to assert the
recorded module-name format matches ``InferenceRecorder.attach``.
"""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.eval_metrics import eval_reg
from src.eval.image_reliance import (
    EYE_NORM_KEYS,
    FIX_NORM_KEYS,
    extract_residuals,
    extract_sampling_locations,
    per_sample_reg_error,
    probe_recording_support,
    residual_norms,
    sampling_in_range_fraction,
    shuffle_images_in_batch,
    write_reliance_store,
    write_summary,
)


class StubRecorder:
    """Minimal recorder exposing only ``current_payload['activations']``."""

    def __init__(self, activations):
        self.current_payload = {"activations": activations}


# ===========================================================================
# Group 1 — In-range fraction math (FR6/FR16)
# ===========================================================================
def test_g1_all_inside():
    sl = torch.full((2, 5, 4, 3, 4, 2), 0.5)
    out = sampling_in_range_fraction(sl)
    assert out.shape == (2, 3)
    assert torch.equal(out, torch.ones(2, 3))


def test_g1_all_outside():
    sl = torch.full((2, 5, 4, 3, 4, 2), 1.5)
    out = sampling_in_range_fraction(sl)
    assert torch.equal(out, torch.zeros(2, 3))


def test_g1_mixed_per_level():
    # (B=1, Nq=2, H=1, L=3, P=1, 2): level0 inside, level1 outside, level2 half.
    sl = torch.zeros(1, 2, 1, 3, 1, 2)
    sl[:, :, :, 0] = 0.5           # level 0 fully inside
    sl[:, :, :, 1] = 2.0           # level 1 fully outside
    sl[:, 0, :, 2] = 0.5           # level 2 query 0 inside
    sl[:, 1, :, 2] = 2.0           # level 2 query 1 outside
    out = sampling_in_range_fraction(sl)
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0, 0.5]), atol=1e-6)


def test_g1_boundary_inclusive():
    sl = torch.zeros(1, 1, 1, 1, 4, 2)
    sl[..., 0, :] = 0.0            # inside (inclusive lower)
    sl[..., 1, :] = 1.0            # inside (inclusive upper)
    sl[..., 2, :] = -1e-6          # outside
    sl[..., 3, :] = 1 + 1e-6       # outside
    out = sampling_in_range_fraction(sl)
    assert torch.allclose(out[0], torch.tensor([0.5]), atol=1e-6)


def test_g1_query_mask_excludes_padded():
    sl = torch.full((1, 3, 1, 1, 1, 2), 5.0)   # everything off-map
    sl[:, 0] = 0.5                              # query 0 inside
    mask = torch.tensor([[True, False, False]])
    out = sampling_in_range_fraction(sl, query_mask=mask)
    assert torch.allclose(out, torch.ones(1, 1), atol=1e-6)


def test_g1_raises_last_dim_not_2():
    with pytest.raises(ValueError):
        sampling_in_range_fraction(torch.zeros(2, 5, 4, 3, 4, 3))


def test_g1_raises_level_axis_mismatch():
    sl = torch.full((2, 5, 4, 3, 4, 2), 0.5)   # 3 levels
    with pytest.raises(ValueError):
        sampling_in_range_fraction(sl, n_levels=2)


# ===========================================================================
# Group 2 — Image shuffle is a guaranteed derangement (FR10)
# ===========================================================================
@pytest.mark.parametrize("B", [2, 3, 8])
def test_g2_no_fixed_point(B):
    imgs = torch.randn(B, 3, 8, 8)
    perm_imgs, perm = shuffle_images_in_batch(imgs)
    assert not bool((perm == torch.arange(B)).any())
    assert torch.equal(torch.sort(perm).values, torch.arange(B))
    assert torch.equal(perm_imgs, imgs[perm])


def test_g2_b1_raises():
    with pytest.raises(ValueError):
        shuffle_images_in_batch(torch.randn(1, 3, 8, 8))


# ===========================================================================
# Group 3 — Per-sample regression error matches eval_reg (FR11)
# ===========================================================================
def test_g3_mean_matches_eval_reg():
    torch.manual_seed(0)
    B, K1 = 4, 6
    reg = torch.rand(B, K1, 3)
    tgt = torch.rand(B, K1 - 1, 3)
    mask = torch.ones(B, K1, dtype=torch.bool)     # equal counts per row
    agg_reg, agg_dur = eval_reg(reg.clone(), tgt.clone(), mask.clone())
    per_reg, per_dur = per_sample_reg_error(reg, tgt, mask)
    assert np.isclose(per_reg.mean(), agg_reg, atol=1e-5)
    assert np.isclose(per_dur.mean(), agg_dur, atol=1e-5)


def test_g3_masked_positions_do_not_contribute():
    torch.manual_seed(1)
    B, K1 = 2, 5
    reg = torch.rand(B, K1, 3)
    tgt = torch.rand(B, K1 - 1, 3)
    mask = torch.ones(B, K1, dtype=torch.bool)
    mask[:, -1] = False                            # last target position is padding
    before, _ = per_sample_reg_error(reg, tgt, mask)
    reg2 = reg.clone()
    reg2[:, -2] += 100.0                           # reg col -2 aligns to masked target -1
    after, _ = per_sample_reg_error(reg2, tgt, mask)
    assert np.allclose(before, after)


def test_g3_duration_masked_mae_hand_computed():
    # B=2, K1=2 -> K=1; only channel 2 matters for duration.
    reg = torch.zeros(2, 2, 3)
    tgt = torch.zeros(2, 1, 3)
    reg[0, 0, 2] = 0.7                             # pred dur
    tgt[0, 0, 2] = 0.2                             # gt dur -> |0.5|
    reg[1, 0, 2] = 0.1
    tgt[1, 0, 2] = 0.4                             # -> |0.3|
    mask = torch.ones(2, 2, dtype=torch.bool)
    _, dur = per_sample_reg_error(reg, tgt, mask)
    assert np.allclose(dur, [0.5, 0.3], atol=1e-6)


# ===========================================================================
# Group 4 — Residual extraction shapes (FR5)
# ===========================================================================
def _fix_activations(B=2, K1=4, D=8, n=3, seed=0):
    torch.manual_seed(seed)
    acts = {}
    for l in range(n):
        acts[f"decoder.{l}"] = {v: torch.randn(B, K1, D) for v in FIX_NORM_KEYS}
    return acts


def test_g4_residual_norms_shapes_and_values():
    B, K1, D, n = 2, 4, 8, 3
    acts = _fix_activations(B, K1, D, n)
    rec = StubRecorder(acts)
    res = extract_residuals(rec, "decoder", n, FIX_NORM_KEYS)
    norms = residual_norms(res, sample_i=1, value_names=FIX_NORM_KEYS)
    for v in FIX_NORM_KEYS:
        assert norms[v].shape == (n, K1)
        expected = np.stack([
            acts[f"decoder.{l}"][v][1].norm(dim=-1).numpy() for l in range(n)
        ])
        assert np.allclose(norms[v], expected, atol=1e-5)


def test_g4_list_bucket_uses_last():
    B, K1, D = 2, 4, 8
    a_first = torch.randn(B, K1, D)
    a_last = torch.randn(B, K1, D)
    acts = {"decoder.0": {"self_attention_res": [a_first, a_last]}}
    rec = StubRecorder(acts)
    res = extract_residuals(rec, "decoder", 1, ("self_attention_res",))
    assert torch.equal(res["self_attention_res"][0], a_last)


def test_g4_missing_eye_keys_skipped():
    acts = {"decoder.0": {v: torch.randn(1, 3, 4) for v in FIX_NORM_KEYS}}
    rec = StubRecorder(acts)
    res = extract_residuals(rec, "eye_decoder", 2, EYE_NORM_KEYS)  # none present
    for v in EYE_NORM_KEYS:
        assert res[v] == [None, None]


# ===========================================================================
# Group 5 — Sampling-location extraction & module names
# ===========================================================================
def test_g5_extract_sampling_locations_stub():
    B, Nq, H, L, P = 2, 5, 4, 3, 4
    n = 2
    acts = {}
    fix_tensors, eye_tensors = [], []
    for l in range(n):
        ft = torch.randn(B, Nq, H, L, P, 2)
        et = torch.randn(B, Nq, H, L, P, 2)
        acts[f"decoder.{l}.second_cross_attn"] = {"sampling_locations": ft}
        acts[f"eye_decoder.{l}.cross_attn"] = {"sampling_locations": et}
        fix_tensors.append(ft)
        eye_tensors.append(et)
    rec = StubRecorder(acts)
    fix = extract_sampling_locations(rec, "decoder", "second_cross_attn", n)
    eye = extract_sampling_locations(rec, "eye_decoder", "cross_attn", n)
    assert len(fix) == n and len(eye) == n
    for l in range(n):
        assert torch.equal(fix[l], fix_tensors[l])
        assert torch.equal(eye[l], eye_tensors[l])


# --- tiny real MixerModel for the module-name integration assertion ----------
class _DinoInner(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size


class DummyDino(nn.Module):
    def __init__(self, embed_dim=384, patch_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.model = _DinoInner(patch_size)
        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        p = self.model.patch_size
        patches = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)
        tokens = self.proj(patches)
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)


def _tiny_dino_model():
    from src.model.mixer_model import MixerModel
    torch.manual_seed(0)
    return MixerModel(
        n_encoder=2, n_decoder=2, n_eye_decoder=2, n_feature_enhancer=0,
        model_dim=512, total_dim=512, n_heads=8, ff_dim=256,
        max_pos_enc=90, max_pos_dec=26, input_encoder="shared_gaussian",
        norm_first=True, mlp_head_hidden_dim=[128], pos_enc_hidden_dim=64,
        num_freq_bands=8, pos_enc_sigma=1.0, use_deformable_eye_decoder=True,
        use_deformable_fixation_decoder=True, pred_dur_pdf=False,
        phases=["Fixation", "Combined"], activation=F.gelu, device="cpu",
        image_encoder=DummyDino(), image_encoder_type="dinov3",
        n_image_levels=1, head_type="linear",
    )


def test_g5_module_names_match_recorder_attach(tmp_path):
    from src.training.inference_recorder import InferenceRecorder

    model = _tiny_dino_model()
    model.set_phase("Fixation")
    model.eval()
    recorder = InferenceRecorder(output_dir=str(tmp_path / "rec"), enabled=True)
    model.set_inference_recorder(recorder)

    B, T, N = 2, 5, 4
    torch.manual_seed(3)
    batch = dict(src=torch.rand(B, T, 3), tgt=torch.rand(B, N, 3),
                 image_src=torch.rand(B, 3, 256, 256), src_mask=None, tgt_mask=None)
    recorder.start_batch(epoch=0, phase="Fixation", split="test", batch_index=0)
    with torch.no_grad():
        model.encode(**batch)
        model.decode_fixation(**batch)

    acts = recorder.current_payload["activations"]
    for l in range(model.n_decoder):
        assert f"decoder.{l}.second_cross_attn" in acts
        assert "sampling_locations" in acts[f"decoder.{l}.second_cross_attn"]
    for l in range(model.n_eye_decoder):
        assert f"eye_decoder.{l}.cross_attn" in acts
        assert "sampling_locations" in acts[f"eye_decoder.{l}.cross_attn"]
    # residual streams recorded on the block modules themselves
    for l in range(model.n_decoder):
        assert "second_cross_res" in acts[f"decoder.{l}"]
        assert "first_cross_res" in acts[f"decoder.{l}"]


# ===========================================================================
# Group 6 — Recording-support probe (FR3/FR17)
# ===========================================================================
def _fake_model(norm_first, fix_deform, eye_deform, n_decoder=2, n_eye_decoder=2):
    ns = SimpleNamespace(
        norm_first=norm_first,
        use_deformable_fixation_decoder=fix_deform,
        use_deformable_eye_decoder=eye_deform,
        n_decoder=n_decoder,
        n_eye_decoder=n_eye_decoder,
    )
    if n_eye_decoder > 0:
        ns.eye_decoder = object()
    return ns


def test_g6_both_ok():
    s = probe_recording_support(_fake_model(True, True, True))
    assert s["fix_ok"] and s["eye_ok"]


def test_g6_post_norm_disables_all(capsys):
    s = probe_recording_support(_fake_model(False, True, True))
    assert not s["fix_ok"] and not s["eye_ok"]
    assert "norm_first=False" in capsys.readouterr().out


def test_g6_non_deformable_fixation():
    s = probe_recording_support(_fake_model(True, False, True))
    assert not s["fix_ok"] and s["eye_ok"]


# ===========================================================================
# Group 7 — HDF5 writer roundtrip (FR12/FR14)
# ===========================================================================
def _synthetic_pass_a(N=3, n_dec=2, n_eye=2, K1=4, n_lvl=3, D=8, src_lens=(5, 3, 4),
                      full=True, fix_ok=True, eye_ok=True, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(N):
        rec = {"sample_idx": 10 + i, "src_len": int(src_lens[i]),
               "stimulus_name": f"img_{i}.jpg", "pred_len": i + 1}
        if fix_ok:
            rec["dec_norms"] = {v: rng.random((n_dec, K1)).astype(np.float32) for v in FIX_NORM_KEYS}
            rec["dec_inrange"] = rng.random((n_dec, n_lvl)).astype(np.float32)
            if full:
                rec["full"] = {v: rng.random((n_dec, K1, D)).astype(np.float16)
                               for v in ("first_cross_res", "second_cross_res")}
        if eye_ok:
            L = int(src_lens[i])
            rec["eye_norms"] = {v: rng.random((n_eye, L)).astype(np.float32) for v in EYE_NORM_KEYS}
            rec["eye_inrange"] = rng.random((n_eye, n_lvl)).astype(np.float32)
        records.append(rec)
    return records


def _support(n_dec=2, n_eye=2, fix_ok=True, eye_ok=True):
    return {"n_decoder": n_dec, "n_eye_decoder": n_eye, "fix_ok": fix_ok, "eye_ok": eye_ok,
            "norm_first": True, "fix_deform": True, "eye_deform": True}


def _attrs():
    return {"run_name": "t", "checkpoint_path": "ck", "img_size": 256,
            "image_encoder_type": "mask2former", "n_image_levels": 3,
            "spatial_shapes": [8, 8, 16, 16, 32, 32], "level_start_index": [0, 64, 320],
            "K1": 4, "model_dim": 8, "n_decoder": 2, "n_eye_decoder": 2,
            "target_mode": "pred", "split": "test", "eps_ignore": 1e-3, "created_at": "now"}


def test_g7_roundtrip(tmp_path):
    a = _synthetic_pass_a()
    b = [{"sample_idx": r["sample_idx"], "reg_error_clean": 0.5 + i,
          "reg_error_shuffled": 0.5 + i + 0.1, "dur_error_clean": 0.2,
          "dur_error_shuffled": 0.25, "perm_index": (i + 1) % len(a)}
         for i, r in enumerate(a)]
    p = tmp_path / "r.h5"
    write_reliance_store(p, a, b, _support(), _attrs())
    with h5py.File(p, "r") as f:
        g = f["/reliance"]
        assert np.array_equal(g["sample_idx"][:], np.array([10, 11, 12], np.int32))
        assert np.allclose(g["dec_second_cross_res_norm"][:],
                           np.stack([r["dec_norms"]["second_cross_res"] for r in a]))
        assert np.allclose(g["dec_inrange"][:], np.stack([r["dec_inrange"] for r in a]))
        assert np.allclose(g["eye_inrange"][:], np.stack([r["eye_inrange"] for r in a]))
        assert np.array_equal(g["dec_first_cross_res"][:],
                              np.stack([r["full"]["first_cross_res"] for r in a]))
        assert np.allclose(g["reg_error_clean"][:], [0.5, 1.5, 2.5])
        assert np.allclose(g["reg_error_shuffled"][:], [0.6, 1.6, 2.6])
        assert g.attrs["fix_second_cross"] == "image"
        assert g.attrs["eye_cross"] == "image"
        assert g.attrs["fix_first_cross"] == "gaze"


def test_g7_attrs_cover_fr14(tmp_path):
    a = _synthetic_pass_a()
    p = tmp_path / "r.h5"
    write_reliance_store(p, a, [], _support(), _attrs())
    required = {"run_name", "checkpoint_path", "img_size", "image_encoder_type",
               "n_image_levels", "spatial_shapes", "level_start_index", "K1", "model_dim",
               "n_decoder", "n_eye_decoder", "target_mode", "split", "eps_ignore", "created_at",
               "fix_residuals_saved", "eye_residuals_saved", "full_residuals_saved",
               "inrange_saved", "fix_first_cross", "fix_second_cross", "eye_cross"}
    with h5py.File(p, "r") as f:
        assert required.issubset(set(f["/reliance"].attrs.keys()))


def test_g7_eye_nan_padded(tmp_path):
    a = _synthetic_pass_a(src_lens=(5, 3, 4))
    p = tmp_path / "r.h5"
    write_reliance_store(p, a, [], _support(), _attrs())
    with h5py.File(p, "r") as f:
        eye = f["/reliance"]["eye_self_attention_res_norm"][:]   # (N, n_eye, T_max=5)
        assert eye.shape[-1] == 5
        assert np.isnan(eye[1, :, 3:]).all()                     # row 1 src_len=3 -> cols 3,4 NaN
        assert not np.isnan(eye[1, :, :3]).any()
        dec = f["/reliance"]["dec_self_attention_res_norm"][:]
        assert not np.isnan(dec).any()                           # dense


def test_g7_variable_k1_padded(tmp_path):
    # Fixation-decoder K1 differs across batches (variable max-fixation count) -> NaN-pad to K1_max.
    n_dec, n_lvl, D = 2, 3, 8
    rng = np.random.default_rng(4)
    a = []
    for i, K1 in enumerate((14, 15, 14)):
        a.append({
            "sample_idx": 20 + i, "src_len": 4, "stimulus_name": "", "pred_len": 2,
            "dec_norms": {v: rng.random((n_dec, K1)).astype(np.float32) for v in FIX_NORM_KEYS},
            "dec_inrange": rng.random((n_dec, n_lvl)).astype(np.float32),
            "full": {v: rng.random((n_dec, K1, D)).astype(np.float16)
                     for v in ("first_cross_res", "second_cross_res")},
        })
    p = tmp_path / "r.h5"
    write_reliance_store(p, a, [], _support(eye_ok=False), _attrs())
    with h5py.File(p, "r") as f:
        g = f["/reliance"]
        arr = g["dec_first_cross_res_norm"][:]           # (3, n_dec, 15)
        assert arr.shape == (3, n_dec, 15)
        assert not np.isnan(arr[0, :, :14]).any()        # row 0 K1=14 filled
        assert np.isnan(arr[0, :, 14]).all()             # ...col 14 padded
        assert not np.isnan(arr[1]).any()                # row 1 K1=15 fully filled
        assert int(g.attrs["K1"]) == 15
        full = g["dec_first_cross_res"][:]               # (3, n_dec, 15, D)
        assert np.isnan(full[0, :, 14, :]).all()


def test_g7_pass_a_only(tmp_path):
    a = _synthetic_pass_a()
    p = tmp_path / "r.h5"
    write_reliance_store(p, a, [], _support(), _attrs())
    with h5py.File(p, "r") as f:
        g = f["/reliance"]
        assert np.isnan(g["reg_error_clean"][:]).all()
        assert np.array_equal(g["perm_index"][:], np.full(3, -1, np.int32))


def test_g7_skipped_streams(tmp_path):
    a = _synthetic_pass_a(fix_ok=False, eye_ok=True, full=False)
    p = tmp_path / "r.h5"
    write_reliance_store(p, a, [], _support(fix_ok=False), _attrs())
    with h5py.File(p, "r") as f:
        g = f["/reliance"]
        assert "dec_second_cross_res_norm" not in g
        assert "dec_inrange" not in g
        assert "eye_inrange" in g
        assert g.attrs["fix_residuals_saved"] == np.False_ or g.attrs["fix_residuals_saved"] is False


# ===========================================================================
# Group 8 — Summary aggregation (FR13)
# ===========================================================================
def _summary_records(ratio=0.02, N=4, n_dec=2, n_eye=2, K1=4, n_lvl=3, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(N):
        first = rng.random((n_dec, K1)).astype(np.float32) + 0.5
        second = (first * ratio).astype(np.float32)
        recs.append({
            "sample_idx": i, "src_len": 4, "stimulus_name": "", "pred_len": 2,
            "dec_norms": {"first_cross_res": first, "second_cross_res": second,
                          "self_attention_res": first, "ffn_res": first},
            "dec_inrange": np.full((n_dec, n_lvl), 0.9, np.float32),
            "eye_norms": {"cross_attention_res": second[:n_eye], "self_attention_res": first[:n_eye],
                          "ffn_res": first[:n_eye]},
            "eye_inrange": np.full((n_eye, n_lvl), 0.4, np.float32),
        })
    return recs


def test_g8_image_gaze_ratio(tmp_path):
    a = _summary_records(ratio=0.02)
    s = write_summary(tmp_path / "s.json", a, [], _support(), _attrs())
    ratios = [d["image_over_gaze_ratio"] for d in s["residuals"]["fixation_decoder"]]
    assert all(np.isclose(r, 0.02, atol=1e-6) for r in ratios)
    assert "<< 1" in s["residuals"]["interpretation"]


def test_g8_perturbation_delta_and_eps(tmp_path):
    a = _summary_records()
    # 4 samples: two unchanged (< eps), one changed, one NaN (excluded).
    b = [
        {"sample_idx": 0, "reg_error_clean": 1.0, "reg_error_shuffled": 1.0000, "dur_error_clean": 0.1, "dur_error_shuffled": 0.1, "perm_index": 1},
        {"sample_idx": 1, "reg_error_clean": 1.0, "reg_error_shuffled": 1.00005, "dur_error_clean": 0.1, "dur_error_shuffled": 0.1, "perm_index": 2},
        {"sample_idx": 2, "reg_error_clean": 1.0, "reg_error_shuffled": 2.0, "dur_error_clean": 0.1, "dur_error_shuffled": 0.2, "perm_index": 3},
        {"sample_idx": 3, "reg_error_clean": 1.0, "reg_error_shuffled": np.nan, "dur_error_clean": 0.1, "dur_error_shuffled": np.nan, "perm_index": -1},
    ]
    s = write_summary(tmp_path / "s2.json", a, b, _support(), _attrs())
    pert = s["perturbation"]
    assert pert["n_valid"] == 3
    assert np.isclose(pert["mean_reg_error_clean"], 1.0, atol=1e-6)
    assert np.isclose(pert["mean_reg_error_shuffled"], (1.0 + 1.00005 + 2.0) / 3, atol=1e-6)
    # two of three valid samples changed by < eps (1e-3)
    assert np.isclose(pert["frac_samples_below_eps"], 2.0 / 3.0, atol=1e-6)


def test_g8_json_valid_with_blocks(tmp_path):
    a = _summary_records()
    b = [{"sample_idx": i, "reg_error_clean": 1.0, "reg_error_shuffled": 1.1,
          "dur_error_clean": 0.1, "dur_error_shuffled": 0.1, "perm_index": (i + 1) % 4}
         for i in range(4)]
    p = tmp_path / "s.json"
    write_summary(p, a, b, _support(), _attrs())
    with open(p) as f:
        loaded = json.load(f)
    assert "residuals" in loaded and "perturbation" in loaded and "in_range" in loaded
    assert "interpretation" in loaded["residuals"]
    assert "interpretation" in loaded["perturbation"]
    assert "fixation_interpretation" in loaded["in_range"]
