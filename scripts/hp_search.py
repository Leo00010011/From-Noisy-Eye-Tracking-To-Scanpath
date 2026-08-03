"""Optuna + W&B hyperparameter search driver for the MixerModel Combined phase.

Runs (or resumes) an Optuna study defined in ``configs/hp_search.yaml``. Each trial samples
15 hyperparameters, composes a training config via Hydra's ``compose`` API (with
``exp=hp_search`` for the reduced Combined-only budget), trains through the existing
``train(builder, trial=...)`` entry point, and reports the best ``reg_error_val`` as the
objective to minimise. The study persists to SQLite so a killed HPC job resumes cleanly.

Usage:
    py scripts/hp_search.py
"""
import math
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict
import optuna
import optuna.distributions as optd

# make `src` importable when run as `py scripts/hp_search.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.pipeline_builder import PipelineBuilder
from src.training.pipeline import train

REPO_ROOT = Path(__file__).resolve().parents[1]

# The 9 dropout keys currently > 0 in configs/model/mixer_model.yaml, each sampled independently.
DROPOUT_KEYS = [
    "model.src_dropout", "model.decoder_dropout", "model.eye_encoder_dropout",
    "model.eye_decoder_dropout", "model.image_features_dropout", "model.dur_head_dropout",
    "model.end_dropout", "model.reg_head_output_dropout", "model.denoise_head_output_dropout",
]


def build_search_config(path="configs/hp_search.yaml"):
    """Load the search-meta config (study / W&B / search-space) via OmegaConf."""
    return OmegaConf.load(REPO_ROOT / path)


def suggest_overrides(trial, search_space):
    """Sample all 15 hyperparameters; return {hydra_dotted_key: value}."""
    space = search_space
    ov = {}
    wd = space.weight_decay
    ov["training.weight_decay"] = trial.suggest_float("weight_decay", wd.low, wd.high, log=wd.log)
    for key, name in (("model.n_encoder", "n_encoder"),
                      ("model.n_decoder", "n_decoder"),
                      ("model.n_eye_decoder", "n_eye_decoder")):
        rng = space[name]
        ov[key] = trial.suggest_int(name, rng.low, rng.high)
    d_default = space.dropouts
    d_ranges = space.get("dropout_ranges", None)   # optional per-dropout overrides
    for key in DROPOUT_KEYS:
        pname = key.split(".")[-1]                 # unique Optuna param name per dropout
        rng = d_ranges[pname] if (d_ranges is not None and pname in d_ranges) else d_default
        ov[key] = trial.suggest_float(pname, rng.low, rng.high)
    ov["loss.cls_weight"] = trial.suggest_float("cls_weight", space.cls_weight.low, space.cls_weight.high)
    ov["loss.dur_weight"] = trial.suggest_float("dur_weight", space.dur_weight.low, space.dur_weight.high)
    return ov


def build_search_distributions(space):
    """Optuna distribution per sampled param, mirroring suggest_overrides' ranges/types.

    Used to warm-start a new (narrower) study from a previous one: it defines the new search
    space so imported trials can be filtered to what still fits and re-tagged with the new
    distributions.
    """
    dists = {}
    wd = space.weight_decay
    dists["weight_decay"] = optd.FloatDistribution(
        float(wd.low), float(wd.high), log=bool(wd.get("log", False)))
    for name in ("n_encoder", "n_decoder", "n_eye_decoder"):
        rng = space[name]
        dists[name] = optd.IntDistribution(int(rng.low), int(rng.high))
    d_default = space.dropouts
    d_ranges = space.get("dropout_ranges", None)
    for key in DROPOUT_KEYS:
        pname = key.split(".")[-1]
        rng = d_ranges[pname] if (d_ranges is not None and pname in d_ranges) else d_default
        dists[pname] = optd.FloatDistribution(float(rng.low), float(rng.high))
    for name in ("cls_weight", "dur_weight"):
        rng = space[name]
        dists[name] = optd.FloatDistribution(float(rng.low), float(rng.high))
    return dists


def _trial_fits(trial, dists):
    """True iff the trial carries every param and each value lies inside the new distribution."""
    for name, dist in dists.items():
        if name not in trial.params:
            return False
        v = trial.params[name]
        if isinstance(dist, optd.IntDistribution) and not float(v).is_integer():
            return False
        if v < dist.low or v > dist.high:
            return False
    return True


def warm_start_study(new_study, prev_study, dists, include_pruned=True):
    """Import COMPLETE (and optionally PRUNED) trials from prev_study into new_study, keeping
    only those whose params fall inside the new distributions. Both the TPE sampler and the
    MedianPruner read their state from the study's trials, so this warms up both. Trials with
    any out-of-range param are dropped so Optuna is never handed an observation outside the new
    space. Returns the number of imported trials."""
    usable = {optuna.trial.TrialState.COMPLETE}
    if include_pruned:
        usable.add(optuna.trial.TrialState.PRUNED)
    prev_trials = prev_study.get_trials(deepcopy=False)
    to_add, sk_range, sk_state, err = [], 0, 0, 0
    for t in prev_trials:
        if t.state not in usable:
            sk_state += 1
            continue
        if t.state == optuna.trial.TrialState.COMPLETE and (t.value is None or not math.isfinite(t.value)):
            sk_state += 1
            continue
        if not _trial_fits(t, dists):
            sk_range += 1
            continue
        try:
            ft = optuna.trial.create_trial(
                state=t.state,
                value=t.value if t.state == optuna.trial.TrialState.COMPLETE else None,
                params={n: t.params[n] for n in dists},
                distributions={n: dists[n] for n in dists},
                intermediate_values=dict(t.intermediate_values),
            )
            to_add.append(ft)
        except Exception as e:                       # malformed historical trial → skip, keep going
            err += 1
            print(f"[hp_search] warm start: skipped a trial ({type(e).__name__}: {e})")
    if to_add:
        new_study.add_trials(to_add)
    print(f"[hp_search] warm start from '{prev_study.study_name}': imported {len(to_add)} "
          f"(skipped {sk_range} out-of-range, {sk_state} unusable-state, {err} errored) "
          f"of {len(prev_trials)} previous trials.")
    return len(to_add)


def _fmt(v):
    # ensure floats round-trip through Hydra override parsing (no sci-notation surprises)
    return repr(v) if isinstance(v, float) else str(v)


def compose_trial_config(overrides):
    """Hydra-compose main.yaml with exp=hp_search plus the sampled overrides."""
    ov_list = ["exp=hp_search"] + [f"{k}={_fmt(v)}" for k, v in overrides.items()]
    # hydra.initialize is a context manager that may only be active once; enter it fresh per call.
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="../configs"):
        return hydra.compose(config_name="main", overrides=ov_list)


def _set_trial_paths(cfg, trial_dir):
    """Replicate train.py::add_metric_and_checkpoint_paths for a per-trial directory."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.training.metric_file = str(trial_dir / "metrics.json")
        cfg.training.checkpoint_file = str(trial_dir / "model.pth")
        cfg.training.splits_file = str(trial_dir / "split.pth")
        if "inference_recorder" in cfg.training:
            cfg.training.inference_recorder.output_dir = str(trial_dir / "inference_records")


def build_storage(study_cfg, storage_dir):
    """Return an Optuna storage. Prefers SQLite; falls back to file-based JournalStorage when
    SQLAlchemy is unavailable/broken (common on HPC images) so the study is still resumable.

    ``study.storage_backend`` in configs/hp_search.yaml selects the behaviour:
    "auto" (default) tries SQLite then falls back; "sqlite" forces SQLite (raises on failure);
    "journal" forces the SQLAlchemy-free journal backend.
    """
    backend = study_cfg.get("storage_backend", "auto")
    name = study_cfg.study_name
    if backend in ("sqlite", "auto"):
        db_path = Path(storage_dir) / f"{name}.db"
        url = f"sqlite:///{db_path.as_posix()}"
        try:
            storage = optuna.storages.RDBStorage(url)   # imports SQLAlchemy under the hood
            print(f"[hp_search] Using SQLite storage: {url}")
            return storage
        except Exception as e:                            # broken/missing SQLAlchemy, etc.
            if backend == "sqlite":
                raise
            print(f"[hp_search] SQLite storage unavailable ({type(e).__name__}: {e}); "
                  f"falling back to JournalStorage.")
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    log_path = Path(storage_dir) / f"{name}.log"
    print(f"[hp_search] Using Journal storage: {log_path}")
    return JournalStorage(JournalFileBackend(str(log_path)))


def make_objective(search_cfg):
    """Return an ``objective(trial) -> float`` that runs one trial end to end."""
    study_name = search_cfg.study.study_name
    out_root = REPO_ROOT / search_cfg.study.storage_dir / study_name
    wcfg = search_cfg.wandb

    def objective(trial):
        overrides = suggest_overrides(trial, search_cfg.search_space)
        cfg = compose_trial_config(overrides)
        trial_dir = out_root / f"trial_{trial.number}"
        _set_trial_paths(cfg, trial_dir)
        with open_dict(cfg):
            cfg.training.wandb.enabled = bool(wcfg.enabled)
        OmegaConf.save(cfg, trial_dir / "config.yaml")   # reproducibility snapshot (FR7)

        run = None
        if wcfg.enabled:
            import wandb
            run = wandb.init(
                project=wcfg.project, entity=wcfg.get("entity", None),
                mode=wcfg.get("mode", "online"),
                group=wcfg.get("group", study_name),
                name=f"trial_{trial.number}", reinit=True,
                config={"trial_number": trial.number, **{k: trial.params[k] for k in trial.params}},
            )
            # Use `epoch` as the x-axis for all train/val curves (train() logs it as a field).
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
        try:
            builder = PipelineBuilder(cfg)
            best = train(builder, trial=trial)
            if best is None:
                raise optuna.TrialPruned()          # never validated → treat as unusable
            if run is not None:
                run.summary["best_reg_error_val"] = best
                import wandb
                wandb.finish()
            return best
        except optuna.TrialPruned:
            raise                                    # train() may already have finished the run
        except Exception:
            if run is not None:
                import wandb
                wandb.finish(exit_code=1)
            raise                                    # let Optuna mark FAILED, continue study
    return objective


def maybe_warm_start(study, search_cfg):
    """If study.warm_start.enabled, seed the (fresh) new study from a previous study's trials,
    filtered to the current search space. No-op when the study already has trials (resume) so a
    re-submitted job never double-imports. Returns the number of trials imported."""
    ws = search_cfg.study.get("warm_start", None)
    if ws is None or not ws.get("enabled", False):
        return 0
    existing = len(study.get_trials(deepcopy=False))
    if existing > 0:
        print(f"[hp_search] warm start requested but study '{study.study_name}' already has "
              f"{existing} trials; skipping import (this is a resume).")
        return 0
    prev_name = ws.get("study_name", None)
    if not prev_name:
        raise ValueError("study.warm_start.enabled is true but study.warm_start.study_name is unset.")
    prev_dir = REPO_ROOT / (ws.get("storage_dir", None) or search_cfg.study.storage_dir)
    prev_backend = ws.get("storage_backend", None) or search_cfg.study.get("storage_backend", "auto")
    same_store = (prev_dir == REPO_ROOT / search_cfg.study.storage_dir
                  and prev_backend == search_cfg.study.get("storage_backend", "auto"))
    if prev_name == study.study_name and same_store:
        raise ValueError("warm_start.study_name must differ from study.study_name (or use a "
                         "different storage) — importing a study into itself is not allowed.")
    prev_storage = build_storage(
        OmegaConf.create({"study_name": prev_name, "storage_backend": prev_backend}), prev_dir)
    try:
        prev_study = optuna.load_study(study_name=prev_name, storage=prev_storage)
    except Exception as e:
        raise RuntimeError(
            f"Could not load previous study '{prev_name}' from {prev_dir} "
            f"(backend={prev_backend}): {e}. Check that warm_start.study_name and "
            f"warm_start.storage_backend match how the previous study was stored.") from e
    dists = build_search_distributions(search_cfg.search_space)
    return warm_start_study(study, prev_study, dists,
                            include_pruned=bool(ws.get("include_pruned", True)))


def main():
    search_cfg = build_search_config()
    study_name = search_cfg.study.study_name
    out_root = REPO_ROOT / search_cfg.study.storage_dir / study_name
    out_root.mkdir(parents=True, exist_ok=True)
    storage_dir = REPO_ROOT / search_cfg.study.storage_dir
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage = build_storage(search_cfg.study, storage_dir)
    pr = search_cfg.study.pruner
    study = optuna.create_study(
        study_name=study_name, direction="minimize", storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=search_cfg.study.sampler_seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=pr.n_startup_trials, n_warmup_steps=pr.n_warmup_steps),
    )
    # Warm-start the sampler + pruner from a previous study (if configured) before optimizing.
    maybe_warm_start(study, search_cfg)
    # catch=(Exception,) so a single failing trial (e.g. CUDA OOM from a large n_* corner) is
    # recorded FAILED and the study proceeds to the next trial (FR9). optuna handles
    # optuna.TrialPruned separately (→ PRUNED) regardless of `catch`.
    study.optimize(make_objective(search_cfg), n_trials=search_cfg.study.n_trials,
                   gc_after_trial=True, catch=(Exception,))

    study.trials_dataframe().to_csv(out_root / "trials.csv", index=False)
    best = study.best_trial
    OmegaConf.save(OmegaConf.create(
        {"best_trial_number": best.number, "best_reg_error_val": best.value, "params": best.params}),
        out_root / "best_params.yaml")
    print(f"Best trial {best.number}: reg_error_val={best.value:.4f}")
    print(OmegaConf.to_yaml(OmegaConf.create(best.params)))


if __name__ == "__main__":
    main()
