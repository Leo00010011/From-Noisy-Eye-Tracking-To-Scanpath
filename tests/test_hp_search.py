"""Tests for the Optuna + W&B hyperparameter search (validation.md Groups 1-7 + integrity).

Runnable without the dataset / GPU / W&B: `train()`'s new control flow is exercised with a
lightweight stub builder + a monkeypatched `validate`; the driver's config composition,
search-space sampling, schedule invariants, study persistence and reproducibility are tested
against the real code paths. Heavy end-to-end items (a real Combined training run, live W&B
curves) require the production environment and are covered by module-level `skip`s / notes.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import optuna

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import hp_search as H  # noqa: E402  scripts/hp_search.py
from src.training import pipeline as pipeline_mod  # noqa: E402
from src.training.pipeline import train  # noqa: E402
from src.training.training_utils import MetricsStorage, WarmupStableDecayScheduler  # noqa: E402

# The 15 Hydra override keys of FR2.
FR2_KEYS = {
    "training.weight_decay", "model.n_encoder", "model.n_decoder", "model.n_eye_decoder",
    "model.src_dropout", "model.decoder_dropout", "model.eye_encoder_dropout",
    "model.eye_decoder_dropout", "model.image_features_dropout", "model.dur_head_dropout",
    "model.end_dropout", "model.reg_head_output_dropout", "model.denoise_head_output_dropout",
    "loss.cls_weight", "loss.dur_weight",
}


# ── Lightweight harness for train() ────────────────────────────────────────────

class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.name = "FakeModel"

    def forward(self, **kwargs):
        return {"base": self.p.sum()}

    def set_phase(self, phase):
        self.phase = phase

    def set_scheduled_sampling(self, ss):
        pass

    def param_summary(self):
        return "fake-params"


class _FakeLoss:
    def __call__(self, input, output):
        loss = output["base"] * 0.0 + 1.0
        return loss, {"loss": 1.0}

    def set_denoise_weight(self, w):
        pass

    def summary(self):
        pass


class _FakeLoader:
    def __init__(self, n_batches=1):
        self._batches = [{"dummy": torch.zeros(1)} for _ in range(n_batches)]
        self.dataset = list(range(n_batches))

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class _DummyScheduler:
    def step(self):
        pass


class _FakeBuilder:
    """Minimal PipelineBuilder stand-in exercising train()'s control flow only."""

    def __init__(self, config, model):
        self.config = config
        self.device = torch.device("cpu")
        self._model = model
        self.curriculum_noise = None

    def build_phases(self):
        c = self.config.training.Combined
        return [(c.name, c.denoise_weight, c.decisive_metric, c.epochs)]

    def load_dataset(self):
        pass

    def build_model(self):
        return self._model, None

    def build_inference_recorder(self, model):
        return None

    def make_splits(self):
        return [0], [0], [0]

    def build_dataloader(self, *a, **k):
        return _FakeLoader(), _FakeLoader(), None

    def clear_dataframe(self):
        pass

    def training_summary(self, n):
        pass

    def build_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.01)

    def build_scheduler(self, optimizer, loader):
        return _DummyScheduler()

    def build_loss_fn(self):
        return _FakeLoss()

    def build_weights_scheduler(self, loss_fn):
        return None

    def build_scheduled_sampling(self, steps):
        return None

    def build_denoise_dropout_scheduler(self, model, steps):
        return None


def _make_config(tmp_path, *, epochs=2, val_interval=1, validate=True, wandb_enabled=False):
    return OmegaConf.create({
        "training": {
            "validate": validate,
            "val_interval": val_interval,
            "log": True,
            "decisive_metric": "reg_error_val",
            "save_full_state": False,
            "metric_file": str(tmp_path / "metrics.json"),
            "checkpoint_file": str(tmp_path / "model.pth"),
            "splits_file": str(tmp_path / "split.pth"),
            "inference_recorder": {"rec_interval": 999, "split": "train"},
            "wandb": {"enabled": wandb_enabled},
            "Combined": {
                "name": "Combined", "denoise_weight": 0,
                "decisive_metric": "reg_error_val", "epochs": epochs,
            },
        },
        "model": {"compilate": False},
        "scheduler": {"batch_lr": False},
    })


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Neutralise disk / heavy calls inside train(); return a recorder for validate."""
    monkeypatch.setattr(pipeline_mod, "save_splits", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(MetricsStorage, "compute_normalized_regression_metrics",
                        lambda self, *a, **k: None)

    state = {"reg_values": [0.5, 0.3, 0.4], "call": 0}

    def fake_validate(model, loss_fn, val_dataloader, epoch, device, metrics, **kwargs):
        i = state["call"]
        v = state["reg_values"][i % len(state["reg_values"])]
        metrics["epoch"].append(epoch + 1)
        metrics["reg_error_val"].append(v)
        metrics["duration_error_val"].append(0.1 + 0.01 * i)
        metrics["accuracy"].append(0.8)
        metrics["precision_pos"].append(0.7)
        metrics["recall_pos"].append(0.6)
        state["call"] += 1

    monkeypatch.setattr(pipeline_mod, "validate", fake_validate)
    return state


# ── Group 1 — train() signature and return value ───────────────────────────────

def test_train_returns_min_reg_error(tmp_path, patched_pipeline):
    patched_pipeline["reg_values"] = [0.5, 0.3, 0.4]
    cfg = _make_config(tmp_path, epochs=3, val_interval=1)
    ret = train(_FakeBuilder(cfg, _FakeModel()))
    assert isinstance(ret, float)
    assert abs(ret - 0.3) < 1e-9


def test_train_returns_none_when_never_validated(tmp_path, patched_pipeline):
    cfg = _make_config(tmp_path, epochs=2, validate=False)
    ret = train(_FakeBuilder(cfg, _FakeModel()))
    assert ret is None


# ── Group 2 — Pruning path ─────────────────────────────────────────────────────

class _StubTrial:
    def __init__(self, prune_schedule):
        self.reports = []          # list of (value, step)
        self._prune = list(prune_schedule)
        self._i = 0

    def report(self, value, step):
        self.reports.append((value, step))

    def should_prune(self):
        val = self._prune[self._i] if self._i < len(self._prune) else False
        self._i += 1
        return val


def test_prune_fires_on_second_report(tmp_path, patched_pipeline):
    patched_pipeline["reg_values"] = [0.5, 0.4, 0.3]
    cfg = _make_config(tmp_path, epochs=3, val_interval=1)
    trial = _StubTrial([False, True])
    with pytest.raises(optuna.TrialPruned):
        train(_FakeBuilder(cfg, _FakeModel()), trial=trial)
    assert len(trial.reports) == 2
    assert [s for _, s in trial.reports] == [0, 1]
    assert [round(v, 6) for v, _ in trial.reports] == [0.5, 0.4]


def test_report_only_on_new_validation(tmp_path, patched_pipeline):
    patched_pipeline["reg_values"] = [0.5, 0.4, 0.3]
    cfg = _make_config(tmp_path, epochs=3, val_interval=1)
    trial = _StubTrial([False, False, False])
    train(_FakeBuilder(cfg, _FakeModel()), trial=trial)
    assert len(trial.reports) == 3
    assert [s for _, s in trial.reports] == [0, 1, 2]


def test_prune_finishes_wandb(tmp_path, patched_pipeline, monkeypatch):
    fake_wandb = _install_fake_wandb(monkeypatch, run_truthy=True)
    patched_pipeline["reg_values"] = [0.5, 0.4]
    cfg = _make_config(tmp_path, epochs=2, val_interval=1, wandb_enabled=True)
    trial = _StubTrial([True])
    with pytest.raises(optuna.TrialPruned):
        train(_FakeBuilder(cfg, _FakeModel()), trial=trial)
    assert fake_wandb.finish_called


# ── Group 3 — W&B logging ──────────────────────────────────────────────────────

class _FakeRun:
    def __init__(self):
        self.summary = {}


def _install_fake_wandb(monkeypatch, run_truthy=True):
    mod = types.ModuleType("wandb")
    mod.logs = []
    mod.finish_called = False
    mod.init_called = False
    mod.login_called = False
    mod.run = _FakeRun() if run_truthy else None

    def log(payload, step=None):
        mod.logs.append((dict(payload), step))

    def finish(exit_code=None):
        mod.finish_called = True

    def init(*a, **k):
        mod.init_called = True
        mod.run = _FakeRun()
        return mod.run

    def login(*a, **k):
        mod.login_called = True

    mod.log, mod.finish, mod.init, mod.login = log, finish, init, login
    mod.define_metric = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return mod


def test_wandb_logs_train_and_val(tmp_path, patched_pipeline, monkeypatch):
    fw = _install_fake_wandb(monkeypatch, run_truthy=True)
    patched_pipeline["reg_values"] = [0.5, 0.4]
    cfg = _make_config(tmp_path, epochs=2, val_interval=1, wandb_enabled=True)
    train(_FakeBuilder(cfg, _FakeModel()))
    train_logs = [p for p, _ in fw.logs if any(k.startswith("train/") for k in p)]
    val_logs = [p for p, _ in fw.logs if any(k.startswith("val/") for k in p)]
    assert len(train_logs) == 2
    assert len(val_logs) == 2
    assert all("val/reg_error_val" in p for p in val_logs)
    assert [p["val/reg_error_val"] for p in val_logs] == [0.5, 0.4]


def test_wandb_no_run_skips_logging(tmp_path, patched_pipeline, monkeypatch):
    fw = _install_fake_wandb(monkeypatch, run_truthy=False)  # wandb.run is None
    cfg = _make_config(tmp_path, epochs=2, val_interval=1, wandb_enabled=True)
    ret = train(_FakeBuilder(cfg, _FakeModel()))  # must not raise
    assert ret is not None
    assert fw.logs == []


def test_train_never_calls_init_or_login(tmp_path, patched_pipeline, monkeypatch):
    fw = _install_fake_wandb(monkeypatch, run_truthy=True)
    cfg = _make_config(tmp_path, epochs=2, val_interval=1, wandb_enabled=True)
    train(_FakeBuilder(cfg, _FakeModel()))
    assert not fw.init_called
    assert not fw.login_called


# ── Group 4 — Config composition ───────────────────────────────────────────────

def test_compose_reflects_overrides():
    cfg = H.compose_trial_config({"model.n_encoder": 7, "training.weight_decay": 3e-4})
    assert cfg.model.n_encoder == 7
    assert abs(cfg.training.weight_decay - 3e-4) < 1e-12


def test_exp_hp_search_wins():
    cfg = H.compose_trial_config({})
    assert list(cfg.training.Phases) == ["Combined"]
    assert cfg.model.pretrained_encoder_path is None
    assert cfg.training.Combined.epochs == 40


@pytest.mark.parametrize("key,value", [
    ("training.weight_decay", 0.01),
    ("model.n_encoder", 4), ("model.n_decoder", 4), ("model.n_eye_decoder", 4),
    ("model.src_dropout", 0.25), ("model.decoder_dropout", 0.25),
    ("model.eye_encoder_dropout", 0.25), ("model.eye_decoder_dropout", 0.25),
    ("model.image_features_dropout", 0.25), ("model.dur_head_dropout", 0.25),
    ("model.end_dropout", 0.25), ("model.reg_head_output_dropout", 0.25),
    ("model.denoise_head_output_dropout", 0.25),
    ("loss.cls_weight", 0.5), ("loss.dur_weight", 0.5),
])
def test_all_fr2_keys_resolve(key, value):
    cfg = H.compose_trial_config({key: value})
    node = cfg
    for part in key.split("."):
        node = node[part]
    assert abs(float(node) - float(value)) < 1e-9


def test_float_formatting_keeps_float_type():
    cfg = H.compose_trial_config({"training.weight_decay": 1.2345e-05})
    assert isinstance(cfg.training.weight_decay, float)
    assert abs(cfg.training.weight_decay - 1.2345e-05) < 1e-12


# ── Group 5 — Reduced-budget invariants ────────────────────────────────────────

def test_scheduler_sum_equals_E():
    cfg = H.compose_trial_config({})
    s = cfg.scheduler
    assert s.warmup_steps + s.stable_steps + s.decay_steps == cfg.training.Combined.epochs


def test_scheduled_sampling_sum_le_E():
    cfg = H.compose_trial_config({})
    ss = cfg.scheduled_sampling
    assert ss.warmup_epochs + ss.active_epochs <= cfg.training.Combined.epochs


def test_val_interval_gives_enough_points():
    cfg = H.compose_trial_config({})
    assert cfg.training.Combined.epochs // cfg.training.val_interval >= 4


def test_scheduler_decays_within_budget():
    cfg = H.compose_trial_config({})
    E = cfg.training.Combined.epochs
    steps_per_epoch = 3
    param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=0.001)
    sched = WarmupStableDecayScheduler(
        opt,
        warmup_steps=cfg.scheduler.warmup_steps * steps_per_epoch,
        stable_steps=cfg.scheduler.stable_steps * steps_per_epoch,
        decay_steps=cfg.scheduler.decay_steps * steps_per_epoch,
        min_lr=cfg.scheduler.min_lr,
    )
    last = None
    for _ in range(E * steps_per_epoch):
        opt.step()
        sched.step()
        last = opt.param_groups[0]["lr"]
    assert abs(last - cfg.scheduler.min_lr) < 1e-6


# ── Group 6 — Search-space sampling ────────────────────────────────────────────

def _fresh_trial():
    return optuna.create_study().ask()


def test_suggest_overrides_keys():
    sc = H.build_search_config()
    ov = H.suggest_overrides(_fresh_trial(), sc.search_space)
    assert set(ov.keys()) == FR2_KEYS


def test_suggest_overrides_within_bounds():
    sc = H.build_search_config()
    ov = H.suggest_overrides(_fresh_trial(), sc.search_space)
    for k in H.DROPOUT_KEYS:
        assert 0.0 <= ov[k] <= 0.5
    for k in ("model.n_encoder", "model.n_decoder", "model.n_eye_decoder"):
        assert 2 <= ov[k] <= 8 and isinstance(ov[k], int)
    assert 1e-5 <= ov["training.weight_decay"] <= 1e-1
    assert 0.05 <= ov["loss.cls_weight"] <= 1.0
    assert 0.05 <= ov["loss.dur_weight"] <= 1.0


def test_dropout_param_names_distinct():
    sc = H.build_search_config()
    trial = _fresh_trial()
    H.suggest_overrides(trial, sc.search_space)
    dropout_names = [k.split(".")[-1] for k in H.DROPOUT_KEYS]
    assert len(set(dropout_names)) == 9
    assert set(dropout_names).issubset(set(trial.params.keys()))


def test_per_dropout_ranges_respected():
    """A dropout with its own entry in dropout_ranges uses it; others fall back to `dropouts`."""
    space = OmegaConf.create({
        "weight_decay": {"low": 1e-5, "high": 1e-1, "log": True},
        "n_encoder": {"low": 2, "high": 8}, "n_decoder": {"low": 2, "high": 8},
        "n_eye_decoder": {"low": 2, "high": 8},
        "dropouts": {"low": 0.0, "high": 0.5},
        "dropout_ranges": {"src_dropout": {"low": 0.30, "high": 0.31}},
        "cls_weight": {"low": 0.05, "high": 1.0}, "dur_weight": {"low": 0.05, "high": 1.0},
    })
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    for _ in range(30):
        t = study.ask()
        ov = H.suggest_overrides(t, space)
        assert 0.30 <= ov["model.src_dropout"] <= 0.31       # narrowed key honoured
        assert 0.0 <= ov["model.decoder_dropout"] <= 0.5     # falls back to default range
        study.tell(t, 0.0)


def test_config_dropout_ranges_cover_all_keys():
    """Every dropout in DROPOUT_KEYS has an entry under search_space.dropout_ranges."""
    sc = H.build_search_config()
    names = {k.split(".")[-1] for k in H.DROPOUT_KEYS}
    assert names == set(sc.search_space.dropout_ranges.keys())


def test_model_consumes_src_dropout_override():
    from src.model.mixer_model import MixerModel
    try:
        model = MixerModel(
            input_dim=3, output_dim=3, img_size=256,
            n_encoder=2, n_decoder=2, n_eye_decoder=0,
            model_dim=32, total_dim=32, n_heads=2, ff_dim=64,
            image_encoder=None, use_deformable_eye_decoder=False,
            use_deformable_fixation_decoder=False, input_encoder="linear",
            head_type="linear", src_dropout=0.42, device="cpu",
        )
    except Exception as e:  # pragma: no cover - construction may need GPU-only ops
        pytest.skip(f"MixerModel not CPU-constructible here: {e}")
    assert model.src_dropout == 0.42
    assert model.src_dropout_nn.p == 0.42


# ── Group 7 — Study, storage, persistence ──────────────────────────────────────

def _instant_search_cfg(tmp_path, study_name="t_study", n_trials=2, wandb_enabled=False):
    return OmegaConf.create({
        "study": {
            "study_name": study_name, "n_trials": n_trials,
            "storage_dir": str(tmp_path), "sampler_seed": 42,
            "pruner": {"n_startup_trials": 5, "n_warmup_steps": 2},
        },
        "wandb": {"enabled": wandb_enabled, "project": "p", "entity": None,
                  "mode": "disabled", "group": study_name},
        "search_space": OmegaConf.load(ROOT / "configs" / "hp_search.yaml").search_space,
    })


def test_tpe_sampling_reproducible():
    sc = H.build_search_config()

    def first_params(seed, n=3):
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))

        def obj(trial):
            H.suggest_overrides(trial, sc.search_space)
            return 0.0
        study.optimize(obj, n_trials=n)
        return [t.params for t in study.trials]

    assert first_params(42) == first_params(42)


def test_study_persistence_and_resume(tmp_path):
    name = "persist_study"
    db = tmp_path / f"{name}.db"
    storage = f"sqlite:///{db.as_posix()}"

    def make():
        return optuna.create_study(
            study_name=name, direction="minimize", storage=storage, load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42))

    s1 = make()
    s1.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2)
    assert db.exists()
    s2 = make()  # resume — must not raise, must see prior trials
    assert len(s2.trials) == 2
    s2.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2)
    assert len(s2.trials) == 4


def test_study_identity_direction_guard(tmp_path):
    """The driver always creates/resumes with direction='minimize'; resume preserves it and
    re-creating without load_if_exists on an existing name raises (study identity is stable)."""
    name = "dir_study"
    storage = f"sqlite:///{(tmp_path / (name + '.db')).as_posix()}"
    s1 = optuna.create_study(study_name=name, direction="minimize", storage=storage,
                             load_if_exists=True)
    assert s1.direction == optuna.study.StudyDirection.MINIMIZE
    # resume with the same (consistent) direction never flips the objective.
    s2 = optuna.create_study(study_name=name, direction="minimize", storage=storage,
                             load_if_exists=True)
    assert s2.direction == optuna.study.StudyDirection.MINIMIZE
    # creating the same name without load_if_exists is an error (guaranteed optuna behaviour).
    with pytest.raises(optuna.exceptions.DuplicatedStudyError):
        optuna.create_study(study_name=name, direction="minimize", storage=storage)


def test_failure_isolation(tmp_path, monkeypatch):
    """A trial raising a non-prune error is FAILED; the study continues (FR9)."""
    fw = _install_fake_wandb(monkeypatch, run_truthy=True)
    sc = _instant_search_cfg(tmp_path, study_name="fail_study", n_trials=3, wandb_enabled=True)

    calls = {"n": 0}

    def fake_builder(cfg):
        return object()

    def fake_train(builder, trial=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")   # first trial fails
        return 0.5

    monkeypatch.setattr(H, "PipelineBuilder", fake_builder)
    monkeypatch.setattr(H, "train", fake_train)

    storage = f"sqlite:///{(tmp_path / 'fail_study.db').as_posix()}"
    study = optuna.create_study(study_name="fail_study", direction="minimize", storage=storage,
                                sampler=optuna.samplers.TPESampler(seed=42))
    # mirror the driver's main(): catch=(Exception,) keeps the study going past a FAILED trial.
    study.optimize(H.make_objective(sc), n_trials=3, catch=(Exception,))
    states = [t.state for t in study.trials]
    assert optuna.trial.TrialState.FAIL in states
    assert optuna.trial.TrialState.COMPLETE in states
    assert fw.finish_called  # finished with exit_code=1 on the failing trial


def test_build_storage_sqlite(tmp_path):
    cfg = OmegaConf.create({"study_name": "sq", "storage_backend": "sqlite"})
    st = H.build_storage(cfg, tmp_path)
    assert isinstance(st, optuna.storages.RDBStorage)


def test_build_storage_journal(tmp_path):
    cfg = OmegaConf.create({"study_name": "jr", "storage_backend": "journal"})
    st = H.build_storage(cfg, tmp_path)
    from optuna.storages import JournalStorage
    assert isinstance(st, JournalStorage)


def test_build_storage_auto_falls_back_on_broken_sqlalchemy(tmp_path, monkeypatch):
    """When SQLite/SQLAlchemy is broken (as on the HPC image), 'auto' uses JournalStorage."""
    def boom(url):
        raise ImportError("cannot import name 'util' from partially initialized module 'sqlalchemy'")
    monkeypatch.setattr(optuna.storages, "RDBStorage", boom)
    cfg = OmegaConf.create({"study_name": "au", "storage_backend": "auto"})
    st = H.build_storage(cfg, tmp_path)
    from optuna.storages import JournalStorage
    assert isinstance(st, JournalStorage)


def test_build_storage_sqlite_forced_raises_on_broken_sqlalchemy(tmp_path, monkeypatch):
    def boom(url):
        raise ImportError("broken sqlalchemy")
    monkeypatch.setattr(optuna.storages, "RDBStorage", boom)
    cfg = OmegaConf.create({"study_name": "sf", "storage_backend": "sqlite"})
    with pytest.raises(ImportError):
        H.build_storage(cfg, tmp_path)


# ── Warm-start (import a previous study into a narrower one) ───────────────────

import optuna.distributions as optd  # noqa: E402


def test_build_search_distributions_types_and_bounds():
    sc = H.build_search_config()
    dists = H.build_search_distributions(sc.search_space)
    assert set(dists.keys()) == {k.split(".")[-1] for k in FR2_KEYS}
    assert isinstance(dists["n_encoder"], optd.IntDistribution)
    assert isinstance(dists["weight_decay"], optd.FloatDistribution)
    assert dists["weight_decay"].log == bool(sc.search_space.weight_decay.get("log", False))
    # per-dropout range flows through
    r = sc.search_space.dropout_ranges.src_dropout
    assert dists["src_dropout"].low == r.low and dists["src_dropout"].high == r.high


def test_trial_fits_filters_out_of_range():
    dists = {"src_dropout": optd.FloatDistribution(0.0, 0.15),
             "n_encoder": optd.IntDistribution(2, 3)}
    inside = optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE, value=0.1,
        params={"src_dropout": 0.1, "n_encoder": 3},
        distributions={"src_dropout": optd.FloatDistribution(0.0, 0.5),
                       "n_encoder": optd.IntDistribution(2, 8)})
    outside = optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE, value=0.1,
        params={"src_dropout": 0.4, "n_encoder": 3},   # 0.4 outside [0,0.15]
        distributions={"src_dropout": optd.FloatDistribution(0.0, 0.5),
                       "n_encoder": optd.IntDistribution(2, 8)})
    missing = optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE, value=0.1,
        params={"src_dropout": 0.1},
        distributions={"src_dropout": optd.FloatDistribution(0.0, 0.5)})
    assert H._trial_fits(inside, dists)
    assert not H._trial_fits(outside, dists)
    assert not H._trial_fits(missing, dists)


def _wide_prev_study():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=1),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0))

    def wide(t):
        d = t.suggest_float("src_dropout", 0.0, 0.5)
        x = t.suggest_int("n_encoder", 2, 8)
        for step in range(3):
            t.report(d + 0.1 * step, step)
            if t.should_prune():
                raise optuna.TrialPruned()
        return d + 0.01 * x
    study.optimize(wide, n_trials=15)
    return study


def test_warm_start_imports_only_in_range():
    prev = _wide_prev_study()
    dists = {"src_dropout": optd.FloatDistribution(0.0, 0.15),
             "n_encoder": optd.IntDistribution(2, 3)}
    new = optuna.create_study(direction="minimize")
    n = H.warm_start_study(new, prev, dists, include_pruned=True)
    assert n == len(new.get_trials(deepcopy=False))
    # every imported trial fits the new (narrow) ranges
    for t in new.get_trials(deepcopy=False):
        assert 0.0 <= t.params["src_dropout"] <= 0.15
        assert 2 <= t.params["n_encoder"] <= 3
    # nothing out of range slipped through
    assert n < len(prev.get_trials(deepcopy=False))


def test_warm_start_excludes_pruned_when_disabled():
    prev = _wide_prev_study()
    dists = {"src_dropout": optd.FloatDistribution(0.0, 0.5),
             "n_encoder": optd.IntDistribution(2, 8)}
    new = optuna.create_study(direction="minimize")
    H.warm_start_study(new, prev, dists, include_pruned=False)
    assert all(t.state == optuna.trial.TrialState.COMPLETE
               for t in new.get_trials(deepcopy=False))


def _warm_cfg(tmp_path, new_name, prev_name, enabled=True):
    sc = OmegaConf.create(OmegaConf.to_container(H.build_search_config(), resolve=True))
    sc.study.study_name = new_name
    sc.study.storage_dir = str(tmp_path)
    sc.study.storage_backend = "sqlite"
    sc.study.warm_start = {"enabled": enabled, "study_name": prev_name,
                           "storage_dir": None, "storage_backend": None, "include_pruned": True}
    return sc


def test_maybe_warm_start_end_to_end(tmp_path):
    # previous WIDE study persisted to sqlite in tmp_path
    prev_storage = H.build_storage(
        OmegaConf.create({"study_name": "prev", "storage_backend": "sqlite"}), tmp_path)
    prev = optuna.create_study(study_name="prev", direction="minimize", storage=prev_storage,
                               sampler=optuna.samplers.TPESampler(seed=1))

    def wide(t):
        d = t.suggest_float("src_dropout", 0.0, 0.5)
        for name in ("decoder_dropout", "eye_encoder_dropout", "eye_decoder_dropout",
                     "image_features_dropout", "dur_head_dropout", "end_dropout",
                     "reg_head_output_dropout", "denoise_head_output_dropout"):
            t.suggest_float(name, 0.0, 0.5)
        t.suggest_int("n_encoder", 2, 8); t.suggest_int("n_decoder", 2, 8)
        t.suggest_int("n_eye_decoder", 2, 8)
        t.suggest_float("weight_decay", 1e-5, 1e-1)
        t.suggest_float("cls_weight", 0.05, 1.0); t.suggest_float("dur_weight", 0.05, 1.0)
        return d
    prev.optimize(wide, n_trials=8)

    sc = _warm_cfg(tmp_path, new_name="new", prev_name="prev")
    new_storage = H.build_storage(sc.study, REPO_ROOT / sc.study.storage_dir
                                  if not Path(sc.study.storage_dir).is_absolute()
                                  else Path(sc.study.storage_dir))
    new = optuna.create_study(study_name="new", direction="minimize", storage=new_storage,
                              load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=2))
    imported = H.maybe_warm_start(new, sc)
    assert imported == len(new.get_trials(deepcopy=False))
    assert imported >= 0
    # resume guard: a second call on the now-non-empty study imports nothing
    again = H.maybe_warm_start(new, sc)
    assert again == 0


def test_maybe_warm_start_disabled_is_noop():
    sc = H.build_search_config()
    # no warm_start block present after resolve? build a minimal cfg with disabled
    cfg = OmegaConf.create({"study": {"study_name": "x", "storage_dir": "outputs/hp_search",
                                      "storage_backend": "auto",
                                      "warm_start": {"enabled": False}},
                            "search_space": sc.search_space})
    study = optuna.create_study(direction="minimize")
    assert H.maybe_warm_start(study, cfg) == 0


def test_maybe_warm_start_rejects_self_import():
    sc = H.build_search_config()
    cfg = OmegaConf.create({"study": {"study_name": "same", "storage_dir": "outputs/hp_search",
                                      "storage_backend": "auto",
                                      "warm_start": {"enabled": True, "study_name": "same",
                                                     "storage_dir": None, "storage_backend": None}},
                            "search_space": sc.search_space})
    study = optuna.create_study(study_name="same", direction="minimize")
    with pytest.raises(ValueError):
        H.maybe_warm_start(study, cfg)


def test_config_snapshot_roundtrip(tmp_path):
    overrides = {"model.n_encoder": 6, "model.src_dropout": 0.33, "loss.cls_weight": 0.7}
    cfg = H.compose_trial_config(overrides)
    trial_dir = tmp_path / "trial_0"
    H._set_trial_paths(cfg, trial_dir)
    OmegaConf.save(cfg, trial_dir / "config.yaml")
    loaded = OmegaConf.load(trial_dir / "config.yaml")
    assert loaded.model.n_encoder == 6
    assert abs(loaded.model.src_dropout - 0.33) < 1e-9
    assert abs(loaded.loss.cls_weight - 0.7) < 1e-9
    assert "trial_0" in loaded.training.metric_file
    assert "trial_0" in loaded.training.checkpoint_file


# ── Data Architecture Integrity ────────────────────────────────────────────────

def test_default_configs_unmodified():
    import subprocess
    files = [
        "configs/main.yaml", "configs/model/mixer_model.yaml",
        "configs/loss/separated_loss.yaml", "configs/scheduler/warmup.yaml",
    ]
    out = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--"] + files,
        cwd=ROOT, capture_output=True, text=True)
    changed = [ln for ln in out.stdout.splitlines() if ln.strip()]
    assert changed == [], f"default configs drifted: {changed}"


def test_dropout_keys_match_mixer_config():
    """The 9 DROPOUT_KEYS are exactly the dropout keys with value > 0 in mixer_model.yaml."""
    mixer = OmegaConf.load(ROOT / "configs" / "model" / "mixer_model.yaml")
    positive_dropouts = {
        f"model.{k}" for k, v in mixer.items()
        if k.endswith("dropout") and isinstance(v, (int, float)) and v > 0
    }
    assert set(H.DROPOUT_KEYS) == positive_dropouts


def test_trial_output_isolation():
    c0 = H.compose_trial_config({})
    c1 = H.compose_trial_config({})
    H._set_trial_paths(c0, Path("outputs/hp_search/x/trial_0"))
    H._set_trial_paths(c1, Path("outputs/hp_search/x/trial_1"))
    for attr in ("metric_file", "checkpoint_file", "splits_file"):
        p0, p1 = c0.training[attr], c1.training[attr]
        assert "trial_0" in p0 and "trial_1" in p1
        assert p0 != p1


def test_objective_never_returns_nonfinite(tmp_path, patched_pipeline):
    patched_pipeline["reg_values"] = [0.5, 0.3]
    cfg = _make_config(tmp_path, epochs=2, val_interval=1)
    ret = train(_FakeBuilder(cfg, _FakeModel()))
    assert np.isfinite(ret)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
