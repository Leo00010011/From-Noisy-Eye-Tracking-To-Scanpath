# Validation — Optuna + W&B Hyperparameter Search

## Code Correctness

### Group 1 — `train()` signature and return value
- [ ] Calling `train(builder)` (no `trial`, `training.wandb.enabled` absent) runs to completion
      and returns a `float` equal to `min(metrics['reg_error_val'])` for that run; assert
      `abs(returned - min(metrics_storage.metrics['reg_error_val'])) < 1e-9`.
- [ ] With validation disabled (`training.validate=False`) or zero validations, `train()` returns
      `None` (guarded by `metrics['reg_error_val']` being empty). No exception.
- [ ] `python train.py exp=only_combined training.num_epochs=1 training.Combined.epochs=1` runs
      end to end with **no** `optuna`/`wandb` import triggered (grep the run: the lazy imports are
      not reached because `trial=None` and `wandb.enabled` is false). Behaviour identical to a
      pre-change run of the same command (same metrics.json keys, same checkpoint written).

### Group 2 — Pruning path
- [ ] With a stub `trial` whose `should_prune()` returns `True` on the 2nd report, `train()` raises
      `optuna.TrialPruned` after exactly 2 `trial.report(...)` calls, and the reported values equal
      the last two `reg_error_val` entries in order (steps `0, 1`).
- [ ] `trial.report` is called **only** on validation epochs where a new `reg_error_val` was
      appended: run 3 epochs with `val_interval=1` and assert `trial.report` call-count == 3 and
      `step` args are `0,1,2` (monotonic, no gaps, no double-report).
- [ ] When pruning fires with `wandb.enabled=True`, `wandb.finish()` is called before the
      `TrialPruned` propagates (assert via a mock wandb: `finish` invoked, then exception raised).

### Group 3 — W&B logging
- [ ] With a mock `wandb` module (`wandb.run` truthy), running `train()` for 2 epochs with
      `val_interval=1` logs: 2 `train/*` payloads (one per epoch, `step=global_epoch`) and 2
      `val/*` payloads including key `val/reg_error_val`. Assert each `val/*` value equals the last
      appended `reg_error_val`/`duration_error_val`/`accuracy` for that epoch.
- [ ] When `wandb.run is None` (init never called) but `training.wandb.enabled=True`, `train()`
      does not raise and simply skips logging (guard on `wb.run is not None`).
- [ ] `train()` never calls `wandb.init` or `wandb.login` itself (assert those are not invoked on
      the mock) — the driver owns them.

### Group 4 — Config composition (`compose_trial_config`)
- [ ] `compose_trial_config({"model.n_encoder": 7, "training.weight_decay": 3e-4})` returns a
      config with `cfg.model.n_encoder == 7` and `abs(cfg.training.weight_decay - 3e-4) < 1e-12`.
- [ ] `exp=hp_search` wins over the default `exp=whole_model_pretraining`: assert
      `cfg.training.Phases == ["Combined"]`, `cfg.model.pretrained_encoder_path is None`,
      `cfg.training.Combined.epochs == 40`.
- [ ] All 15 override keys from FR2 resolve to existing config nodes (no Hydra
      `ConfigAttributeError` / "Could not append" on any key). Parametrised test over the 15 keys
      with a mid-range value each; composition succeeds and the value is present at the dotted path.
- [ ] Float formatting: a sampled `weight_decay = 1.2345e-05` composes to a float
      `cfg.training.weight_decay` (type `float`, not `str`) within `1e-12` of the original.

### Group 5 — Reduced-budget invariants (`configs/exp/hp_search.yaml`)
- [ ] Compose with `exp=hp_search` and assert
      `scheduler.warmup_steps + scheduler.stable_steps + scheduler.decay_steps ==
       training.Combined.epochs` (== `E`, default 40).
- [ ] Assert `scheduled_sampling.warmup_epochs + scheduled_sampling.active_epochs <=
      training.Combined.epochs`.
- [ ] Assert `training.val_interval` divides into `training.Combined.epochs` giving `>= 4`
      validation points (`Combined.epochs // val_interval >= 4`) so pruning has enough reports.
- [ ] Build a `WarmupStableDecayScheduler` from the composed config for a small dummy dataloader
      and confirm the LR completes warmup→stable→decay within `E * steps_per_epoch` steps (final
      LR ≈ `scheduler.min_lr`), i.e. the schedule actually decays inside the reduced budget.

### Group 6 — Search-space sampling (`suggest_overrides`)
- [ ] With a recorded/fixed-seed `optuna.Trial`, `suggest_overrides` returns a dict of exactly 15
      keys covering every FR2 override key (assert set equality against the FR2 key list).
- [ ] Every sampled value lies within its configured bound: dropouts in `[0.0, 0.5]`, `n_*` ints in
      `[2, 8]`, `weight_decay` in `[1e-5, 1e-1]`, `cls_weight`/`dur_weight` in `[0.05, 1.0]`.
- [ ] The 9 dropout Optuna param **names** are distinct (`src_dropout`, `decoder_dropout`, …) so
      Optuna treats them as independent dimensions (assert `len(set(names)) == 9`).
- [ ] Model actually consumes the overrides: build a `MixerModel` from a composed config with
      `model.src_dropout=0.42` and assert the corresponding dropout module's `p == 0.42` (or the
      stored attribute equals 0.42), proving the tuned key is not inert.

### Group 7 — Study, storage, persistence (driver `main`)
- [ ] Run the driver with `n_trials=2` and a dummy 1-epoch budget: it creates
      `outputs/hp_search/<study>.db`, and `outputs/hp_search/<study>/{trial_0,trial_1}/` each
      containing `metrics.json`, `model.pth`, `split.pth`, `config.yaml`.
- [ ] `trials.csv` and `best_params.yaml` are written at study end; `best_params.yaml` contains
      `best_trial_number`, `best_reg_error_val`, and a `params` block with all 15 hyperparameters.
- [ ] **Resume**: re-running the driver with the same `study_name` and `n_trials=2` on the
      existing DB does not error and reuses the same study object (`len(study.trials)` grows;
      previous trials retain their values). `load_if_exists=True` path verified.
- [ ] **Failure isolation**: inject a trial that raises `RuntimeError` (e.g. an intentionally
      impossible `n_encoder`); assert that trial's state is `FAILED`, the study continues, and the
      W&B run (if enabled) was finished with `exit_code=1`.
- [ ] Per-trial `config.yaml` snapshot round-trips: `OmegaConf.load(trial_dir/'config.yaml')`
      equals the config passed to `PipelineBuilder` for that trial (same `model.n_encoder`, same
      dropouts, same `loss.cls_weight`).

## Data Validity

- [ ] **Objective agrees with checkpoint**: for a completed trial, the returned objective equals
      the `reg_error_val` at the epoch whose checkpoint was saved as best (`save_best_only=True`).
      Cross-check the value against the minimum in that trial's `metrics.json`.
- [ ] **Baseline reachability**: a trial forced to the current defaults (`n_encoder=4`,
      `n_decoder=4`, `n_eye_decoder=4`, dropouts at their mixer_model.yaml values, `weight_decay
      =0.01`, `cls_weight=0.2`, `dur_weight=0.33`) produces a `reg_error_val` in the same ballpark
      as a known-good short Combined run (within a loose tolerance, e.g. same order of magnitude) —
      confirms the reduced budget still learns and the search can recover the current config.
- [ ] **Pruning actually helps**: over a ≥10-trial run, assert at least one trial reaches state
      `PRUNED` (given `n_warmup_steps=2`, `n_startup_trials=5`) — i.e. the pruner is wired and
      firing, not silently disabled.
- [ ] **W&B curves present**: open one trial's W&B run and confirm `val/reg_error_val` has one
      point per validation epoch and is monotonically consistent with the trial's `metrics.json`
      `reg_error_val` list (same length, same values within float tolerance).
- [ ] **Metric range sanity**: every logged `reg_error_val` is a finite positive float in
      normalised space (roughly `(0, 2]`); no `NaN`/`inf` reach the objective (a `NaN` objective
      would corrupt TPE — assert the driver never returns non-finite; if a run diverges to `NaN`
      it should surface as FAILED, not as a fake best).

## Data Architecture Integrity

- [ ] **No default-config drift**: `git diff` shows `configs/main.yaml`,
      `configs/model/mixer_model.yaml`, `configs/loss/separated_loss.yaml`, and
      `configs/scheduler/warmup.yaml` are **unmodified**. The reduced budget lives only in the new
      `configs/exp/hp_search.yaml`; `python train.py` with no exp override is unchanged.
- [ ] **Search keys ↔ config keys are exhaustive and exact**: every key in `DROPOUT_KEYS` plus the
      4 non-dropout tuned keys corresponds to an existing key in the composed config, and the 9
      dropout keys are exactly the set of dropout keys with value `> 0` in
      `configs/model/mixer_model.yaml` at spec time (no missing, no extra — cross-check against the
      FR2 table).
- [ ] **Trial output isolation**: two trials never write to the same `metric_file` /
      `checkpoint_file` / `splits_file`; assert the three paths contain `trial_<number>` and differ
      between trial 0 and trial 1 (no cross-trial overwrite of `metrics.json` or `model.pth`).
- [ ] **Study identity**: the SQLite study's `study_name` and `direction` match the config
      (`minimize`); re-creating with a different direction on the same name would raise — assert the
      driver uses `load_if_exists=True` with a consistent direction so resume never silently flips
      the objective.
- [ ] **Reproducibility of sampling**: two fresh studies created with the same `sampler_seed` and
      the same (mocked, instant) objective propose the **same** first-N parameter sets — confirms
      `TPESampler(seed=...)` makes the search reproducible as the constitution requires.
