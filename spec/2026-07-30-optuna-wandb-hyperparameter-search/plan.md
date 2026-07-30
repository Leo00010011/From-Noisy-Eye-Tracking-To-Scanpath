# Plan — Optuna + W&B Hyperparameter Search

## Context and Design Decisions

**Why a standalone driver instead of `hydra-optuna-sweeper`.** The Roadmap requires W&B
experiment tracking with per-trial metric curves *and* Optuna pruning. The official Hydra
sweeper plugin makes both awkward: it owns the process-per-run lifecycle, which complicates
initialising/finishing one W&B run per trial and calling `trial.report()` mid-training for
`MedianPruner`. A standalone driver (`scripts/hp_search.py`) that uses Hydra's `compose` API keeps
the whole search in one process, gives direct access to the `optuna.Trial` object inside the
training loop, and lets us own the W&B run lifecycle explicitly. This matches the constitution's
"all runs fully specified by a config" principle because the trial config is still composed from
the same Hydra groups and snapshotted to disk (FR7).

**Why Combined-only, from scratch.** The user is currently running only the Combined stage. Two
of the requested search dimensions — `n_encoder` and `n_eye_decoder` — change the encoder/eye
architecture. `MixerModel.load_encoder` cherry-picks encoder weights **by module name** from a
fixed checkpoint; loading a depth-4 encoder into a depth-7 model would silently mis-populate or
fail. So the search sets `pretrained_encoder_path: null` and trains Combined from random init.
This makes trials self-consistent and comparable. The existing `configs/exp/only_combined.yaml`
already encodes `Phases: ["Combined"] + pretrained_encoder_path: null`; the new
`configs/exp/hp_search.yaml` is its reduced-epoch sibling.

**Why the reduced-budget invariants matter.** `WarmupStableDecayScheduler` is constructed in
`PipelineBuilder.build_scheduler` as
`warmup_steps = scheduler.warmup_steps * steps_per_epoch` (and likewise stable/decay), so the
config's `warmup/stable/decay_steps` are **epoch counts**. If we shrink the Combined phase to 40
epochs but leave the scheduler at `10/90/20` (=120 epochs), the LR never leaves warmup+stable and
never decays — the run is mis-scheduled. Likewise `ScheduledSampling` (the "scheduler
pretraining" the user referred to) is driven by `scheduled_sampling.warmup_epochs/active_epochs`
(20/50 = 70 epochs); at 40 epochs it would never reach its `max_prob`. The exp file therefore
scales all three schedules together with two invariants (FR3): scheduler epoch-sum `== E`,
scheduled-sampling epoch-sum `<= E`.

**Why the objective is `MetricsStorage.best_metric_value`.** `reg_error_val` is already the
`decisive_metric` and `MetricsStorage.update_best()` already tracks its running minimum and
drives best-checkpoint saving. Reusing it means the objective equals the metric that selects the
saved checkpoint — the trial's reported score and its persisted model agree. No new metric code.

**Why W&B logging lives inside `train()` but the run lifecycle lives in the driver.** The metric
curves must be logged *during* the loop (per epoch / per validation), which only `train()` can
do. But `wandb.init`/`finish` and the run's config/name belong to the trial orchestration. So the
driver `wandb.init`s before calling `train()`, `train()` logs to the active run when
`training.wandb.enabled`, and the driver `finish`es afterward. `train()` only force-finishes on
the prune path so a pruned run closes cleanly before the exception unwinds. This keeps
`train.py`'s normal path (flag absent/false) byte-for-byte behavioural-identical.

Constitution constraints honoured: reproducibility via Hydra compose + per-trial `config.yaml`
snapshot (never edited post-run); TPESampler + SQLite storage seeded for a deterministic,
resumable study; no change to datasets, losses, or default configs.

---

## Step 1 — Add dependencies

**File:** `requirements.txt` (modify)

Append:
```
optuna==4.1.0
wandb==0.18.7
```
(Pin to whatever is already installed on the node if different — verify with `pip show optuna
wandb` before finalising; the point is a pinned, reproducible line, consistent with the rest of
the file's `==` pins.)

---

## Step 2 — Reduced-budget experiment config

**File:** `configs/exp/hp_search.yaml` (new)

`@package _global_` override that shrinks the Combined-only recipe. Default `E = 40`.

```yaml
# @package _global_
training:
  decisive_metric: "reg_error_val"
  pretrained_model: null
  use_scheduled_sampling: true
  Phases: ["Combined"]
  num_epochs: 40                 # E — keeps one_cycle path consistent if ever used
  val_interval: 5                # → 8 validation points inside E=40 (pruning signal)
  Combined:
    name: "Combined"
    denoise_weight: 0
    decisive_metric: ${training.decisive_metric}
    epochs: 40                   # E, reduced from 160
  wandb:
    enabled: false               # driver flips this to true per trial; default keeps train.py inert

model:
  pretrained_encoder_path: null  # train Combined from scratch (n_encoder is searched)

scheduler:
  warmup_steps: 5                # epochs; 5 + 25 + 10 == E == 40
  stable_steps: 25
  decay_steps: 10

scheduled_sampling:
  warmup_epochs: 5               # 5 + 25 == 30 <= E == 40
  active_epochs: 25
```

Invariants to preserve if `E` changes (documented inline as comments):
`scheduler.warmup_steps + stable_steps + decay_steps == training.Combined.epochs`, and
`scheduled_sampling.warmup_epochs + active_epochs <= training.Combined.epochs`.

Note: `training.wandb.enabled` is added here so the key exists in the composed config; the driver
sets it to `true` and fills `training.wandb.*` runtime fields (project/mode) via overrides or by
mutating the composed `DictConfig` with `open_dict`.

---

## Step 3 — Search-meta config

**File:** `configs/hp_search.yaml` (new)

Loaded by the driver via `OmegaConf.load` (NOT part of the Hydra `main` composition — it holds
study/W&B/search-space settings, not training params). Content is exactly the block in
requirements.md "Public API Summary". Keep `search_space.dropouts` as a single `{low, high}` pair
reused for all 9 dropout keys (they share a range per FR2), while `n_encoder`/`n_decoder`/
`n_eye_decoder` and the two loss weights get their own entries.

---

## Step 4 — Make `train()` return the objective and support trial/W&B

**File:** `src/training/pipeline.py` (modify `train`)

Signature and additions (pseudocode of the deltas only — the existing body is unchanged except at
the marked hooks):

```python
def train(builder, trial=None):
    ...
    wandb_enabled = bool(builder.config.training.get("wandb", {}).get("enabled", False))
    wb = None
    if wandb_enabled:
        import wandb as wb                      # lazy; optional dependency
    val_report_step = 0                          # monotonic counter for trial.report
    ...
    for phase, denoise_weight, decisive_metric, epochs in phases:
        ...
        for epoch in range(epochs):
            ...
            loss_info = metrics_storage.finalize_epoch()
            ...
            if wandb_enabled and wb.run is not None:
                wb.log({f"train/{k}": v for k, v in loss_info.items()}, step=global_epoch)

            if needs_validate and ((epoch + 1) % val_interval == 0):
                prev_len = len(metrics_storage.metrics["reg_error_val"])
                validate(...)                    # unchanged call
                ...
                # --- W&B validation logging ---
                if wandb_enabled and wb.run is not None:
                    log = {}
                    for key in ("reg_error_val", "duration_error_val", "accuracy",
                                "precision_pos", "recall_pos"):
                        seq = metrics_storage.metrics.get(key, [])
                        if seq:
                            log[f"val/{key}"] = seq[-1]
                    wb.log(log, step=global_epoch)
                # --- Optuna pruning ---
                new_len = len(metrics_storage.metrics["reg_error_val"])
                if trial is not None and new_len > prev_len:
                    current = metrics_storage.metrics["reg_error_val"][-1]
                    trial.report(current, val_report_step)
                    val_report_step += 1
                    if trial.should_prune():
                        if wandb_enabled and wb.run is not None:
                            wb.finish()
                        import optuna
                        raise optuna.TrialPruned()
                metrics_storage.save_metrics()
                is_best = metrics_storage.update_best()
                if is_best:
                    save_checkpoint(...)         # unchanged
            global_epoch += 1
    print("Training finished!")
    return metrics_storage.best_metric_value if metrics_storage.metrics["reg_error_val"] else None
```

Key points:
- `builder.config.training.get("wandb", {})` — OmegaConf `DictConfig.get` returns a nested node or
  the default; guard for the key being absent so `train.py` (no `wandb` key) stays inert.
- Pruning report happens only when a new `reg_error_val` was actually appended (`new_len >
  prev_len`) — `validate()` only appends when `coord_error_acum > 0`.
- `wb.run is not None` guards against logging when the driver did not open a run.
- The function now returns a float (or `None`); `train.py::main` ignores it (FR10).
- Do **not** import `optuna`/`wandb` at module top level — both are lazy so normal training does
  not require them installed.

---

## Step 5 — The Optuna driver

**File:** `scripts/hp_search.py` (new)

Depends on Steps 2–4. Structure:

```python
import os, sys
from pathlib import Path
import hydra
from omegaconf import OmegaConf, open_dict
import optuna

# make `src` importable when run as `python scripts/hp_search.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.pipeline_builder import PipelineBuilder
from src.training.pipeline import train

REPO_ROOT = Path(__file__).resolve().parents[1]

DROPOUT_KEYS = [
    "model.src_dropout", "model.decoder_dropout", "model.eye_encoder_dropout",
    "model.eye_decoder_dropout", "model.image_features_dropout", "model.dur_head_dropout",
    "model.end_dropout", "model.reg_head_output_dropout", "model.denoise_head_output_dropout",
]

def build_search_config(path="configs/hp_search.yaml"):
    return OmegaConf.load(REPO_ROOT / path)

def suggest_overrides(trial, space):
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
    ov_list = ["exp=hp_search"] + [f"{k}={_fmt(v)}" for k, v in overrides.items()]
    with hydra.initialize(version_base=None, config_path="../configs"):
        return hydra.compose(config_name="main", overrides=ov_list)

def _set_trial_paths(cfg, trial_dir):
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.training.metric_file = str(trial_dir / "metrics.json")
        cfg.training.checkpoint_file = str(trial_dir / "model.pth")
        cfg.training.splits_file = str(trial_dir / "split.pth")
        if "inference_recorder" in cfg.training:
            cfg.training.inference_recorder.output_dir = str(trial_dir / "inference_records")

def make_objective(search_cfg):
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
                import wandb; wandb.finish()
            return best
        except optuna.TrialPruned:
            raise                                    # train() may already have finished the run
        except Exception:
            if run is not None:
                import wandb; wandb.finish(exit_code=1)
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
    study.optimize(make_objective(search_cfg), n_trials=search_cfg.study.n_trials,
                   gc_after_trial=True)

    study.trials_dataframe().to_csv(out_root / "trials.csv", index=False)
    best = study.best_trial
    OmegaConf.save(OmegaConf.create(
        {"best_trial_number": best.number, "best_reg_error_val": best.value, "params": best.params}),
        out_root / "best_params.yaml")
    print(f"Best trial {best.number}: reg_error_val={best.value:.4f}")
    print(OmegaConf.to_yaml(OmegaConf.create(best.params)))

if __name__ == "__main__":
    main()
```

Notes / gotchas:
- `hydra.initialize` must be entered fresh per `compose` call (it is a context manager that can
  only be active once); calling it inside `compose_trial_config` per trial is correct. If a
  `GlobalHydra already initialized` error surfaces (e.g. leftover state), call
  `hydra.core.global_hydra.GlobalHydra.instance().clear()` before `initialize`.
- W&B is imported lazily so the driver's import section does not hard-require it when
  `wandb.enabled=false`.
- `trial.params` is populated by the `suggest_*` calls and is the clean record for W&B config and
  `best_params.yaml`.
- CUDA OOM from an unlucky `(n_encoder, n_decoder, n_eye_decoder)` corner is caught by the generic
  `except Exception` → FAILED, study proceeds (FR9).

---

## Step 6 — Documentation

**File:** `README.md` (modify — small addition near the existing training-invocation section)

Add a short "Hyperparameter search" subsection:
```
py scripts/hp_search.py            # runs/resumes the Optuna study defined in configs/hp_search.yaml
```
Document: edit `configs/hp_search.yaml` to change `n_trials`, ranges, or W&B project; results land
in `outputs/hp_search/<study_name>/` and in the W&B project; the study is resumable (re-run the
same command). Note the W&B key is expected to already be present on the node.

---

## Implementation Order

1. **Step 1** — add `optuna`, `wandb` to `requirements.txt` (verify installed versions on node).
2. **Step 4** — modify `train()` to return `best_reg_error_val` and add trial/W&B hooks
   (independent of the new configs; verify `python train.py` still runs unchanged).
3. **Step 2** — `configs/exp/hp_search.yaml` (reduced Combined-only budget + schedule invariants).
4. **Step 3** — `configs/hp_search.yaml` (study / W&B / search-space meta-config).
5. **Step 5** — `scripts/hp_search.py` (driver: sample → compose → train → report → persist).
6. **Step 6** — README subsection.
7. Validate per `validation.md`.
