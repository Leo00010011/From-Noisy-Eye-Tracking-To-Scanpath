"""Revive PathModel Training (with Scheduled Sampling) — validation suite.

CPU-only. No HDF5 / DINOv3 / network access is required: unit groups use synthetic
tensors and a stub sampler, the real-ScheduledSampling group drives a tiny PathModel,
and the MixerModel regression tripwire builds a small model with a deterministic DINOv3
stub (no weights needed).

Groups mirror ``spec/2026-09-08-revive-path-model-training/validation.md``.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hydra import initialize, compose

from src.model.path_model import PathModel
from src.model.model_io import save_splits
from src.data.datasets import seq2seq_padded_collate_fn
from src.training.training_utils import ScheduledSampling

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Builders / helpers
# ---------------------------------------------------------------------------
def make_path_model(**over):
    kw = dict(
        n_encoder=2, n_decoder=2, model_dim=32, total_dim=32, n_heads=4,
        ff_dim=64, head_type='multi_mlp', mlp_head_hidden_dim=[16, 8],
        max_pos_enc=15, max_pos_dec=26,
        input_encoder='shared_gaussian', norm_first=True, device='cpu',
    )
    kw.update(over)
    return PathModel(**kw)


def gaze_batch(lengths_x=(7, 6), lengths_y=(4, 3)):
    """Build a gaze-only batch dict via the real collate fn (no image_src)."""
    batch = []
    for tx, ny in zip(lengths_x, lengths_y):
        batch.append({
            'x': np.random.rand(3, tx).astype(np.float32),
            'y': np.random.rand(3, ny).astype(np.float32),
        })
    return seq2seq_padded_collate_fn(batch)


class StubSampler:
    """Minimal stand-in for ScheduledSampling with a call counter."""
    def __init__(self, ratio):
        self._ratio = ratio
        self.model = None
        self.set_model_calls = 0
        self.call_count = 0
        self.sentinel = object()

    def set_model(self, model):
        self.model = model
        self.set_model_calls += 1

    def get_current_ratio(self):
        return self._ratio

    def __call__(self, **kwargs):
        self.call_count += 1
        return self.sentinel


def _compose(overrides=None):
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="main", overrides=overrides or [])


# ===========================================================================
# Group 1 — PathModel seams (unit, no sampler)
# ===========================================================================
def test_g1_attributes_initialised():
    m = make_path_model()
    assert m.phase is None
    assert m.scheduled_sampling is None


def test_g1_set_phase_records_and_no_freeze():
    m = make_path_model()
    before = [p.requires_grad for p in m.parameters()]
    ret = m.set_phase('Fixation')
    assert ret is None
    assert m.phase == 'Fixation'
    after = [p.requires_grad for p in m.parameters()]
    assert before == after  # no requires_grad toggling


def test_g1_decode_denoise_empty():
    m = make_path_model()
    assert m.decode_denoise(src=torch.randn(2, 5, 32)) == {}


def test_g1_teacher_forced_forward_no_sampler():
    torch.manual_seed(0)
    m = make_path_model()
    m.set_phase('Fixation')
    src = torch.rand(2, 7, 3)
    tgt = torch.rand(2, 4, 3)
    tgt_mask = torch.ones(2, 5, dtype=torch.bool)
    out = m(src=src, src_mask=None, tgt=tgt, tgt_mask=tgt_mask)
    assert set(out.keys()) == {'coord', 'dur', 'cls'}
    assert out['coord'].shape == (2, 5, 2)
    assert out['dur'].shape == (2, 5, 1)
    assert out['cls'].shape == (2, 5, 1)


# ===========================================================================
# Group 2 — Sampler routing (unit, stub sampler)
# ===========================================================================
def test_g2_set_scheduled_sampling():
    m = make_path_model()
    stub = StubSampler(0.5)
    m.set_scheduled_sampling(stub)
    assert m.scheduled_sampling is stub
    assert stub.set_model_calls == 1
    assert stub.model is m
    assert hasattr(m, 'set_scheduled_sampling')


def test_g2_ratio_zero_skips_sampler():
    torch.manual_seed(0)
    m = make_path_model()
    m.set_phase('Fixation')
    stub = StubSampler(0.0)
    m.set_scheduled_sampling(stub)
    out = m(src=torch.rand(2, 7, 3), src_mask=None,
            tgt=torch.rand(2, 4, 3), tgt_mask=torch.ones(2, 5, dtype=torch.bool))
    assert stub.call_count == 0
    assert set(out.keys()) == {'coord', 'dur', 'cls'}


def test_g2_ratio_positive_routes_to_sampler():
    m = make_path_model()
    m.set_phase('Fixation')
    stub = StubSampler(0.5)
    m.set_scheduled_sampling(stub)
    out = m(src=torch.rand(2, 7, 3), src_mask=None,
            tgt=torch.rand(2, 4, 3), tgt_mask=torch.ones(2, 5, dtype=torch.bool))
    assert stub.call_count == 1
    assert out is stub.sentinel


def test_g2_pass_sampler_never_reencodes(monkeypatch):
    m = make_path_model()
    m.set_phase('Fixation')
    stub = StubSampler(0.5)
    m.set_scheduled_sampling(stub)
    # Prime self.memory as the sampler's own encode would have.
    m.encode(src=torch.rand(2, 7, 3), src_mask=None)
    calls = {'n': 0}
    real_encode = m.encode

    def spy(**kwargs):
        calls['n'] += 1
        return real_encode(**kwargs)
    monkeypatch.setattr(m, 'encode', spy)
    out = m(pass_sampler=True, src=torch.rand(2, 7, 3), src_mask=None,
            tgt=torch.rand(2, 4, 3), tgt_mask=torch.ones(2, 5, dtype=torch.bool))
    assert calls['n'] == 0            # never re-encodes under pass_sampler
    assert stub.call_count == 0       # never routes back into the sampler
    assert set(out.keys()) == {'coord', 'dur', 'cls'}


# ===========================================================================
# Group 3 — Real ScheduledSampling drive (unit)
# ===========================================================================
def _real_sampler_input(B=2):
    b = gaze_batch(lengths_x=(7, 6), lengths_y=(4, 3))
    return b


def test_g3_train_mode_autoregressive():
    torch.manual_seed(0)
    m = make_path_model()
    sampler = ScheduledSampling(active_epochs=1, warmup_epochs=0, device='cpu',
                                steps_per_epoch=1, max_prob=0.85)
    m.set_scheduled_sampling(sampler)
    m.set_phase('Fixation')
    sampler.step()
    assert sampler.get_current_ratio() > 0
    m.train()
    inp = _real_sampler_input()
    out = m(**inp)
    assert set(out.keys()) == {'coord', 'dur', 'cls'}
    assert out['coord'].size(1) == inp['tgt_mask'].size(1)
    assert not hasattr(m, 'enable_memory_kv_cache')  # hasattr guard skips it


def test_g3_eval_mode_ratio_one():
    torch.manual_seed(0)
    m = make_path_model()
    sampler = ScheduledSampling(active_epochs=1, warmup_epochs=0, device='cpu',
                                steps_per_epoch=1, max_prob=0.85)
    m.set_scheduled_sampling(sampler)
    m.set_phase('Fixation')
    m.eval()
    assert sampler.get_current_ratio() == 1
    inp = _real_sampler_input()
    out = m(**inp)
    assert out['coord'].size(1) == inp['tgt_mask'].size(1)


# ===========================================================================
# Group 4 — PipelineBuilder / build_model
# ===========================================================================
def _path_builder(overrides=None):
    from src.training.pipeline_builder import PipelineBuilder
    cfg = _compose((overrides or []) + ["exp=path_model_training", "model.device=cpu"])
    return PipelineBuilder(cfg)


def test_g4_build_model_returns_pathmodel_and_none_splits():
    builder = _path_builder()
    model, splits = builder.build_model()
    assert isinstance(model, PathModel)
    assert splits is None


def test_g4_build_scheduled_sampling_and_set():
    builder = _path_builder()
    model, _ = builder.build_model()
    sampler = builder.build_scheduled_sampling(steps_per_epoch=1)
    assert isinstance(sampler, ScheduledSampling)
    model.set_scheduled_sampling(sampler)  # the original bug (FR2/FR12) — no AttributeError


def test_g4_build_loss_is_separated_reg():
    from src.model.loss_functions import SeparatedRegLossFunction, CombinedLossFunction
    builder = _path_builder()
    loss_fn = builder.build_loss_fn()
    assert isinstance(loss_fn, SeparatedRegLossFunction)
    assert not isinstance(loss_fn, CombinedLossFunction)


# ===========================================================================
# Group 5 — Split reuse (FR8)
# ===========================================================================
def test_g5_reuse_split_exact(tmp_path):
    train_idx = torch.tensor([3, 1, 4, 1, 5])
    val_idx = torch.tensor([9, 2, 6])
    test_idx = torch.tensor([5, 3, 5])
    save_splits(train_idx, val_idx, test_idx, str(tmp_path / "split.pth"))
    builder = _path_builder([f"+training.reuse_split_from={tmp_path.as_posix()}"])
    _, splits = builder.build_model()
    assert splits is not None
    tr, va, te = splits
    assert torch.equal(tr, train_idx)
    assert torch.equal(va, val_idx)
    assert torch.equal(te, test_idx)


def test_g5_reuse_missing_split_raises(tmp_path):
    builder = _path_builder([f"+training.reuse_split_from={tmp_path.as_posix()}"])
    with pytest.raises(Exception, match="Split not found"):
        builder.build_model()


def test_g5_reuse_bypasses_make_splits(tmp_path, monkeypatch):
    train_idx = torch.tensor([0, 1, 2])
    val_idx = torch.tensor([3, 4])
    test_idx = torch.tensor([5])
    save_splits(train_idx, val_idx, test_idx, str(tmp_path / "split.pth"))
    builder = _path_builder([f"+training.reuse_split_from={tmp_path.as_posix()}"])

    def boom(*a, **k):
        raise AssertionError("make_splits must not be called when a split is reused")
    monkeypatch.setattr(builder, 'make_splits', boom)
    _, splits = builder.build_model()
    assert splits is not None and torch.equal(splits[0], train_idx)


# ===========================================================================
# Group 6 — Split-file roundtrip (FR8, no dataset needed)
# ===========================================================================
def test_g6_split_roundtrip_exact_and_stable(tmp_path):
    from src.model.model_io import load_test_data
    train_idx = torch.tensor([10, 20, 30])
    val_idx = torch.tensor([40, 50])
    test_idx = torch.tensor([60])
    save_splits(train_idx, val_idx, test_idx, str(tmp_path / "split.pth"))
    builder = _path_builder()
    a = load_test_data(builder, tmp_path.as_posix(), return_dataloaders=False)
    b = load_test_data(builder, tmp_path.as_posix(), return_dataloaders=False)
    for got in (a, b):
        assert torch.equal(got[0], train_idx)
        assert torch.equal(got[1], val_idx)
        assert torch.equal(got[2], test_idx)
        assert got[0].dtype == train_idx.dtype  # integer, no dtype drift


# ===========================================================================
# Data Validity — scheduled-sampling schedule monotonicity
# ===========================================================================
def test_dv_schedule_monotonic_and_saturates():
    warmup, active, max_prob = 0, 4, 0.85
    steps = 3
    sampler = ScheduledSampling(active_epochs=active, warmup_epochs=warmup, device='cpu',
                                steps_per_epoch=steps, max_prob=max_prob)
    ratios = []
    for _ in range((warmup + active + 1) * steps):
        sampler.step()
        ratios.append(sampler.use_model_prob)
    assert all(b >= a - 1e-9 for a, b in zip(ratios, ratios[1:]))  # non-decreasing
    assert ratios[-1] == pytest.approx(max_prob)
    assert max(ratios) <= max_prob + 1e-9


# ===========================================================================
# End-to-end smoke — a short train/eval loop over synthetic gaze-only data
# ===========================================================================
def test_e2e_smoke_train_and_eval_wellformed():
    torch.manual_seed(0)
    np.random.seed(0)
    m = make_path_model()
    m.set_phase('Fixation')
    sampler = ScheduledSampling(active_epochs=1, warmup_epochs=0, device='cpu',
                                steps_per_epoch=2, max_prob=0.85)
    m.set_scheduled_sampling(sampler)
    loss_fn = _path_builder().build_loss_fn()  # SeparatedRegLossFunction, config-wired
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    batches = [gaze_batch(lengths_x=(7, 6), lengths_y=(4, 3)) for _ in range(2)]

    m.train()
    for _epoch in range(2):
        for b in batches:
            opt.zero_grad()
            out = m(**b)
            loss, _info = loss_fn(b, out)
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
            sampler.step()

    # Eval-mode autoregression: coordinates well-formed and pred_len sane.
    m.eval()
    with torch.no_grad():
        vb = gaze_batch(lengths_x=(7, 6), lengths_y=(4, 3))
        out = m(**vb)
    budget = vb['tgt_mask'].size(1)
    assert out['coord'].size(1) == budget
    pred_len = (torch.sigmoid(out['cls']) > 0.5).float().argmax(dim=1)  # first True per row
    # argmax returns 0 when no True; treat "no EOS" as full budget
    has_eos = (torch.sigmoid(out['cls']) > 0.5).any(dim=1).squeeze(-1)
    pred_len = torch.where(has_eos.unsqueeze(-1), pred_len, torch.tensor(budget))
    assert (pred_len >= 0).all() and (pred_len <= budget).all()


# ===========================================================================
# Data Architecture Integrity — MixerModel regression tripwire (FR12)
# ===========================================================================
def test_di_mixermodel_forward_deterministic():
    """Two identically-seeded MixerModels (DINOv3 stub path) produce byte-identical
    coord on a teacher-forced forward — the path_model.py + build_model edits do not
    perturb the MixerModel path."""
    import torch.nn as nn
    from src.model.mixer_model import MixerModel

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

    common = dict(
        n_encoder=2, n_decoder=2, n_eye_decoder=2, n_feature_enhancer=0,
        model_dim=512, total_dim=512, n_heads=8, ff_dim=256,
        max_pos_enc=90, max_pos_dec=26, input_encoder="shared_gaussian",
        norm_first=True, mlp_head_hidden_dim=[128], pos_enc_hidden_dim=64,
        num_freq_bands=8, pos_enc_sigma=1.0, use_deformable_eye_decoder=True,
        use_deformable_fixation_decoder=True, pred_dur_pdf=False,
        phases=["Fixation", "Combined"], activation=F.gelu, device="cpu",
    )

    def build(seed):
        torch.manual_seed(seed)
        dino = DummyDino()
        return MixerModel(image_encoder=dino, image_encoder_type="dinov3",
                          n_image_levels=1, head_type="linear", **common)

    m1 = build(0)
    m2 = build(0)
    m1.set_phase('Fixation')
    m2.set_phase('Fixation')
    m1.eval()
    m2.eval()
    torch.manual_seed(123)
    src = torch.rand(2, 5, 3)
    tgt = torch.rand(2, 4, 3)
    image = torch.rand(2, 3, 256, 256)
    inp = dict(src=src, tgt=tgt, image_src=image, src_mask=None, tgt_mask=None)
    with torch.no_grad():
        o1 = m1(**inp)
        o2 = m2(**inp)
    assert torch.equal(o1['reg'], o2['reg'])
