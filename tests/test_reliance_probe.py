"""Validation-time image-reliance probe (``src/eval/reliance_probe.py``).

CPU-only, synthetic. Covers: the probe never changes the forward output, the residual/cross-attn
ratio and gate statistics match hand-computed values, each query position is counted once under
the autoregressive prefix re-run (with and without KV cache), padding masks are honoured, and the
end-to-end ``validate()`` wiring on a tiny ``MixerModel`` with scheduled sampling.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.reliance_probe import RelianceProbe
from src.model.blocks import DeformableDecoder, DeformableDoubleInputDecoder

D, N_HEADS, FF, HW = 32, 4, 64, 16 * 16


class _Holder(nn.Module):
    """Mimics MixerModel's ``decoder`` / ``eye_decoder`` ModuleList naming."""

    def __init__(self, gated=False, norm_first=True, use_kv_cache=False):
        super().__init__()
        self.decoder = nn.ModuleList([DeformableDoubleInputDecoder(
            model_dim=D, total_dim=D, n_heads=N_HEADS, ff_dim=FF, norm_first=norm_first,
            use_kv_cache=use_kv_cache, image_gated_fusion=gated, spatial_shape=(16, 16))])
        self.eye_decoder = nn.ModuleList([DeformableDecoder(
            model_dim=D, total_dim=D, n_heads=N_HEADS, ff_dim=FF, norm_first=norm_first,
            spatial_shape=(16, 16))])
        self.eval()


def _inputs(B=2, Nq=5, T=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(src=torch.randn(B, Nq, D, generator=g),
                mem1=torch.randn(B, T, D, generator=g),
                mem2=torch.randn(B, 1 + HW, D, generator=g),
                reference_points=torch.rand(B, Nq, 2, generator=g))


def _manual_norm_first(dec, src, mem1, mem2, reference_points):
    """norm_first forward returning the pre-add residual and outputs of both cross-attentions."""
    x = src
    x = x + dec.self_attn(dec.self_attn_norm(x))
    gaze = dec.first_cross_attn(dec.first_cross_attn_norm(x), mem1)
    x_gaze = x
    x = x + gaze
    img = dec.second_cross_attn(query=dec.second_cross_attn_norm(x), reference_points=reference_points,
                                value=mem2[:, 1:, :], spatial_shape=(16, 16))
    return x_gaze, gaze, x, img


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("norm_first", [True, False])
def test_probe_does_not_change_output(gated, norm_first):
    torch.manual_seed(0)
    m = _Holder(gated=gated, norm_first=norm_first)
    inp = _inputs()
    with torch.no_grad():
        ref = m.decoder[0](**inp)
        ref_eye = m.eye_decoder[0](inp["src"], inp["mem2"], reference_points=inp["reference_points"])
        probe = RelianceProbe(m)
        probe.active = True
        probe.start_batch()
        out = m.decoder[0](**inp)
        out_eye = m.eye_decoder[0](inp["src"], inp["mem2"], reference_points=inp["reference_points"])
    assert torch.equal(ref, out) and torch.equal(ref_eye, out_eye)
    s = probe.summary()
    assert "img_res_ratio_dec0" in s and "gaze_res_ratio_dec0" in s and "img_res_ratio_eye0" in s
    assert ("gate_img_share_dec0" in s) == gated


def test_inactive_probe_records_nothing():
    m = _Holder()
    probe = RelianceProbe(m)
    with torch.no_grad():
        m.decoder[0](**_inputs())
    assert probe.summary() == {}


def test_residual_ratio_matches_hand_computation():
    torch.manual_seed(1)
    m = _Holder()
    dec = m.decoder[0]
    inp = _inputs()
    probe = RelianceProbe(m)
    probe.active = True
    probe.start_batch()
    with torch.no_grad():
        dec(**inp)
        x_gaze, gaze, x_img, img = _manual_norm_first(dec, **inp)
    s = probe.summary()
    exp_img = (x_img.norm(dim=-1).mean() / img.norm(dim=-1).mean()).item()
    exp_gaze = (x_gaze.norm(dim=-1).mean() / gaze.norm(dim=-1).mean()).item()
    assert s["img_res_ratio_dec0"] == pytest.approx(exp_img, rel=1e-5)
    assert s["gaze_res_ratio_dec0"] == pytest.approx(exp_gaze, rel=1e-5)
    assert s["img_res_ratio_dec_mean"] == pytest.approx(exp_img, rel=1e-5)


def test_gate_statistics_match_fc_gate_output():
    torch.manual_seed(2)
    m = _Holder(gated=True)
    dec = m.decoder[0]
    nn.init.normal_(dec.image_gate.fc_gate.weight, std=0.5)
    inp = _inputs()
    captured = {}
    dec.image_gate.register_forward_hook(lambda mod, args, out: captured.update(img=args[0], dec=args[1]))
    probe = RelianceProbe(m)
    probe.active = True
    probe.start_batch()
    with torch.no_grad():
        dec(**inp)
        logit = dec.image_gate.fc_gate(torch.cat((captured["img"], captured["dec"]), -1))
    g = torch.sigmoid(logit)
    s = probe.summary()
    assert s["gate_logit_dec0"] == pytest.approx(logit.mean().item(), rel=1e-5, abs=1e-6)
    assert s["gate_dec0"] == pytest.approx(g.mean().item(), rel=1e-5)
    assert s["gate_sat_img_dec0"] == pytest.approx((g > 0.9).float().mean().item(), abs=1e-6)
    assert s["gate_sat_dec_dec0"] == pytest.approx((g < 0.1).float().mean().item(), abs=1e-6)
    a = (g * captured["img"]).norm(dim=-1).mean()
    b = ((1 - g) * captured["dec"]).norm(dim=-1).mean()
    assert s["gate_img_share_dec0"] == pytest.approx((a / (a + b)).item(), rel=1e-5)
    assert 0.0 <= s["gate_img_share_dec0"] <= 1.0


@pytest.mark.parametrize("use_kv_cache", [False, True])
def test_autoregressive_counts_each_position_once(use_kv_cache):
    """Prefix re-runs (no cache) or one-token steps (cache) == one full causal call."""
    torch.manual_seed(3)
    m = _Holder(use_kv_cache=use_kv_cache)
    dec = m.decoder[0]
    inp = _inputs(Nq=4)
    probe = RelianceProbe(m)
    probe.active = True

    dec.disable_kv_cache()
    probe.start_batch()
    with torch.no_grad():
        dec(**inp)
    full = probe.summary()

    probe.reset()
    probe.start_batch()
    if use_kv_cache:
        dec.use_kv_cache = True
        dec.self_attn.use_kv_cache = True
    with torch.no_grad():
        for t in range(1, 5):
            if use_kv_cache:
                step = {k: v[:, t - 1:t] if k in ("src", "reference_points") else v for k, v in inp.items()}
            else:
                step = {k: v[:, :t] if k in ("src", "reference_points") else v for k, v in inp.items()}
            dec(**step)
    dec.clear_kv_cache()
    assert float(probe.sums["dec0/img_n"]) == 2 * 4
    for key, value in full.items():
        assert probe.summary()[key] == pytest.approx(value, rel=1e-4), key


def test_masks_exclude_padding():
    torch.manual_seed(4)
    m = _Holder()
    inp = _inputs(B=2, Nq=5)
    probe = RelianceProbe(m)
    probe.active = True
    tgt_mask = torch.tensor([[True] * 5, [True, True, False, False, False]])
    src_mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    probe.start_batch(src_mask=src_mask, tgt_mask=tgt_mask)
    with torch.no_grad():
        m.decoder[0](**inp)
        m.eye_decoder[0](inp["src"], inp["mem2"], reference_points=inp["reference_points"])
    assert float(probe.sums["dec0/img_n"]) == 7
    assert float(probe.sums["eye0/img_n"]) == 8

    # changing padded tokens must not move the stats
    before = probe.summary()
    probe.reset()
    probe.start_batch(src_mask=src_mask, tgt_mask=tgt_mask)
    inp2 = dict(inp)
    inp2["src"] = inp["src"].clone()
    inp2["src"][1, 2:] = 100.0   # causal: rows 1, positions >=2 do not affect earlier positions
    with torch.no_grad():
        m.decoder[0](**inp2)
    after = probe.summary()
    assert after["img_res_ratio_dec0"] == pytest.approx(before["img_res_ratio_dec0"], rel=1e-5)


# ---------------------------------------------------------------------------
# validate() integration on a tiny MixerModel
# ---------------------------------------------------------------------------
class _FakePathDataset:
    transforms = []


class _FakeLoader:
    def __init__(self, batches):
        self._batches = batches
        self.path_dataset = _FakePathDataset()

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


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
        tokens = self.proj(F.unfold(x, kernel_size=p, stride=p).transpose(1, 2))
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1)


def _tiny_mixer(gated):
    from src.model.mixer_model import MixerModel
    torch.manual_seed(0)
    return MixerModel(
        n_encoder=1, n_decoder=2, n_eye_decoder=2, n_feature_enhancer=0,
        model_dim=512, total_dim=512, n_heads=8, ff_dim=256,
        max_pos_enc=90, max_pos_dec=26, input_encoder="shared_gaussian",
        norm_first=True, mlp_head_hidden_dim=[128], pos_enc_hidden_dim=64,
        num_freq_bands=8, pos_enc_sigma=1.0, use_deformable_eye_decoder=True,
        use_deformable_fixation_decoder=True, pred_dur_pdf=False,
        phases=["Fixation", "Combined"], activation=F.gelu, device="cpu",
        image_encoder=DummyDino(), image_encoder_type="dinov3",
        n_image_levels=1, head_type="linear", image_gated_fusion=gated,
    )


def _batch(B=2, T=6, N=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(src=torch.rand(B, T, 3, generator=g), tgt=torch.rand(B, N, 3, generator=g),
                image_src=torch.rand(B, 3, 256, 256, generator=g),
                src_mask=None, tgt_mask=torch.ones(B, N + 1, dtype=torch.bool),
                fixation_len=torch.tensor([N] * B))


def test_validate_populates_reliance_metrics():
    from src.training.training_utils import MetricsStorage, ScheduledSampling, validate
    from src.model.loss_functions import EntireRegLossFunction

    m = _tiny_mixer(gated=True)
    m.set_phase("Fixation")
    m.set_scheduled_sampling(ScheduledSampling(active_epochs=1, warmup_epochs=0, device="cpu"))
    probe = RelianceProbe(m)
    assert len(probe) == 4
    loss = EntireRegLossFunction()
    metrics = MetricsStorage(decisive_metric="reg_error_val").metrics
    out = validate(m, loss, _FakeLoader([_batch(seed=0), _batch(seed=1)]), epoch=0, device="cpu",
                   metrics=metrics, log=True, reliance_probe=probe)
    assert not probe.active and m.training
    for key in ("img_res_ratio_dec_mean", "gaze_res_ratio_dec_mean", "img_res_ratio_eye_mean",
                "gate_logit_dec_mean", "gate_dec_mean", "gate_img_share_dec_mean",
                "img_res_ratio_dec1", "gate_img_share_dec0"):
        assert key in out and metrics[key] == [out[key]]
    # 2 batches x B=2 x K1=4 query positions, each counted once despite the prefix re-runs
    assert float(probe.sums["dec0/img_n"]) == 16
    assert float(probe.sums["eye0/img_n"]) == 2 * 2 * 6
    # no probe -> empty summary, metrics untouched by reliance keys
    metrics2 = MetricsStorage(decisive_metric="reg_error_val").metrics
    assert validate(m, loss, _FakeLoader([_batch()]), epoch=0, device="cpu",
                    metrics=metrics2, log=False) == {}
    assert "img_res_ratio_dec_mean" not in metrics2
