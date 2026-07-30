"""Optuna + W&B hyperparameter search driver for the MixerModel Combined phase.

Runs (or resumes) an Optuna study defined in ``configs/hp_search.yaml``. Each trial samples
15 hyperparameters, composes a training config via Hydra's ``compose`` API (with
``exp=hp_search`` for the reduced Combined-only budget), trains through the existing
``train(builder, trial=...)`` entry point, and reports the best ``reg_error_val`` as the
objective to minimise. The study persists to SQLite so a killed HPC job resumes cleanly.

Usage:
    py scripts/hp_search.py
"""
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict
import optuna

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
    d = space.dropouts
    for key in DROPOUT_KEYS:
        pname = key.split(".")[-1]                 # unique Optuna param name per dropout
        ov[key] = trial.suggest_float(pname, d.low, d.high)
    ov["loss.cls_weight"] = trial.suggest_float("cls_weight", space.cls_weight.low, space.cls_weight.high)
    ov["loss.dur_weight"] = trial.suggest_float("dur_weight", space.dur_weight.low, space.dur_weight.high)
    return ov


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


def main():
    search_cfg = build_search_config()
    study_name = search_cfg.study.study_name
    out_root = REPO_ROOT / search_cfg.study.storage_dir / study_name
    out_root.mkdir(parents=True, exist_ok=True)
    db_path = REPO_ROOT / search_cfg.study.storage_dir / f"{study_name}.db"
    storage = f"sqlite:///{db_path.as_posix()}"
    pr = search_cfg.study.pruner
    study = optuna.create_study(
        study_name=study_name, direction="minimize", storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=search_cfg.study.sampler_seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=pr.n_startup_trials, n_warmup_steps=pr.n_warmup_steps),
    )
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
