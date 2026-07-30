# Requirements — Optuna + W&B Hyperparameter Search

## Goal

Add a reproducible hyperparameter-search harness that tunes the `MixerModel` on the
**Combined** training phase using [Optuna](https://optuna.org) as the optimizer and
[Weights & Biases](https://wandb.ai) as the experiment tracker. Each Optuna trial trains a
`MixerModel` for a **reduced** epoch budget and reports the validation coordinate error
(`reg_error_val`) as the single objective to **minimise**. The harness persists the study to
SQLite (resumable on the HPC node), prunes weak trials with `MedianPruner`, logs every trial's
per-epoch metric curves and sampled hyperparameters to W&B, and stores each trial's
`metrics.json` and checkpoint on disk. This delivers the "Experiment tracking (W&B)" backlog
item from the Roadmap and gives a principled way to pick the model configuration for the
publication baseline.

## Scope

**In scope**
- A standalone Optuna driver script (`scripts/hp_search.py`) that:
  - builds each trial's training config via Hydra's `compose` API,
  - applies the trial's sampled values as Hydra overrides,
  - runs training through the existing `train(builder, trial=...)` entry point,
  - returns the best `reg_error_val` seen during the trial as the objective value.
- A search-meta config file (`configs/hp_search.yaml`) holding the study settings (n_trials,
  storage URI, sampler seed, pruner params), the W&B settings (project / entity / mode), and the
  **search-space bounds** for every tuned hyperparameter.
- A reduced-budget experiment override (`configs/exp/hp_search.yaml`) that runs **only** the
  Combined phase from scratch (`pretrained_encoder_path: null`) with reduced epoch counts,
  reduced LR-scheduler epochs, and reduced scheduled-sampling ("scheduler pretraining") epochs,
  all kept mutually consistent.
- Minimal, backward-compatible changes to `src/training/pipeline.py::train` so it (a) accepts an
  optional Optuna `trial`, (b) reports intermediate `reg_error_val` for pruning, (c) logs to W&B
  when enabled, and (d) **returns** the best `reg_error_val`.
- Persisting each trial's `metrics.json`, checkpoint, and split file under a per-trial output
  directory, plus a study-level trials summary CSV and a best-params YAML.
- Adding `optuna` and `wandb` to `requirements.txt`.

**Out of scope (explicitly)**
- Tuning `Denoise` or `Fixation` phases, or loading a pretrained denoise encoder during the
  search. Searching `n_encoder` changes encoder depth, which is incompatible with cherry-picking
  a fixed-depth encoder checkpoint via `MixerModel.load_encoder`, so the search trains Combined
  **from scratch** (`pretrained_encoder_path: null`). Tuning the full `[Fixation, Combined]`
  pipeline or the denoise pretraining is deferred.
- Multi-GPU / distributed or parallel-worker Optuna execution. The harness runs trials
  sequentially on a single node (the SQLite storage still permits a future second worker, but
  that is not a requirement here).
- W&B **Sweeps** as the controller. Optuna is the sole optimizer; W&B is used only for logging
  and visualisation (one W&B run per Optuna trial, grouped by study).
- Changing the training loop's numerical behaviour, the loss functions, the datasets, or any
  existing config default. `train.py` / `python train.py` must behave exactly as before.
- Pixel-space publication metrics (DTW, multi-match) — those remain a separate backlog item.

## Functional Requirements

### FR1 — Objective and direction
The objective value returned per trial is the **minimum** `reg_error_val` observed across all
validation points of that trial (i.e. `MetricsStorage.best_metric_value` at the end of
training). The Optuna study is created with `direction="minimize"`. `reg_error_val` is the mean
Euclidean coordinate error in normalised `[0,1]` space on the validation split, exactly as
already computed by `validate()` in `src/training/training_utils.py` — no new metric is defined.

### FR2 — Tuned hyperparameters and search space
The following 15 hyperparameters are sampled per trial. Bounds live in `configs/hp_search.yaml`
under `search_space` and are the defaults below (all inclusive; a value equal to the current
default must be reachable inside each range).

| # | Hydra override key | Optuna suggest | Default range | Current value |
|---|---|---|---|---|
| 1 | `training.weight_decay` | `suggest_float(log=True)` | `[1e-5, 1e-1]` | `0.01` |
| 2 | `model.n_encoder` | `suggest_int` | `[2, 8]` | `4` |
| 3 | `model.n_decoder` | `suggest_int` | `[2, 8]` | `4` |
| 4 | `model.n_eye_decoder` | `suggest_int` | `[2, 8]` | `4` |
| 5 | `model.src_dropout` | `suggest_float` | `[0.0, 0.5]` | `0.2` |
| 6 | `model.decoder_dropout` | `suggest_float` | `[0.0, 0.5]` | `0.1` |
| 7 | `model.eye_encoder_dropout` | `suggest_float` | `[0.0, 0.5]` | `0.1` |
| 8 | `model.eye_decoder_dropout` | `suggest_float` | `[0.0, 0.5]` | `0.1` |
| 9 | `model.image_features_dropout` | `suggest_float` | `[0.0, 0.5]` | `0.1` |
| 10 | `model.dur_head_dropout` | `suggest_float` | `[0.0, 0.5]` | `0.1` |
| 11 | `model.end_dropout` | `suggest_float` | `[0.0, 0.5]` | `0.1` |
| 12 | `model.reg_head_output_dropout` | `suggest_float` | `[0.0, 0.5]` | `0.1` |
| 13 | `model.denoise_head_output_dropout` | `suggest_float` | `[0.0, 0.5]` | `0.1` |
| 14 | `loss.cls_weight` | `suggest_float` | `[0.05, 1.0]` | `0.2` |
| 15 | `loss.dur_weight` | `suggest_float` | `[0.05, 1.0]` | `0.33` |

Notes:
- The 9 dropouts (#5–#13) are the ones **currently > 0** in `configs/model/mixer_model.yaml`.
  Each is sampled **independently** (one Optuna param per dropout). Dropouts already at `0`
  (`src_word_dropout_prob`, `tgt_dropout`, `adapter_dropout`, `reg_head_dropout`,
  `enh_features_dropout`, `word_dropout_prob`, `denoise_head_hidden_dropout`, `mixer_dropout`,
  `dropout_p`, `geometric_sigma`) are **not** tuned.
- All 15 keys must be valid Hydra dotted-override paths against the composed config (`model.*`
  resolves to `configs/model/mixer_model.yaml`, `loss.*` to `configs/loss/separated_loss.yaml`
  merged with the `loss:` block in `main.yaml`, `training.weight_decay` to `main.yaml`).

### FR3 — Reduced training budget (`configs/exp/hp_search.yaml`)
The search overrides the default `exp` (which is `whole_model_pretraining`, running
`[Fixation, Combined]` for ~200 epochs). `configs/exp/hp_search.yaml` (a `@package _global_`
override) MUST set:
- `training.Phases: ["Combined"]` (Combined stage only).
- `model.pretrained_encoder_path: null` (train from scratch; see out-of-scope rationale).
- `training.Combined.epochs: E` where `E` is a reduced budget (default `E = 40`, down from 160).
- `training.num_epochs: E` (kept consistent; used by the `one_cycle` scheduler path only).
- LR-scheduler epochs consistent with `E`: because
  `WarmupStableDecayScheduler` receives `warmup_steps = scheduler.warmup_steps * steps_per_epoch`
  (and likewise stable/decay), the config values are **epoch counts** and MUST satisfy
  `scheduler.warmup_steps + scheduler.stable_steps + scheduler.decay_steps == E`
  (default `5 + 25 + 10 = 40`).
- Scheduled-sampling ("scheduler pretraining") epochs consistent with `E`:
  `scheduled_sampling.warmup_epochs + scheduled_sampling.active_epochs <= E`
  (default `warmup_epochs = 5`, `active_epochs = 25`).
- `training.val_interval` small enough to produce several validation points inside `E` (so
  pruning and the metric curve are meaningful): default `val_interval = 5` (→ 8 validations at
  `E = 40`).

These reduced counts are the defaults; `E`, the scheduler split, and `val_interval` are
adjustable in the file. Any change to `E` MUST preserve the two invariants above (scheduler sum
`== E`; scheduled-sampling sum `<= E`). The exp file overrides `scheduler.*` under its
`@package _global_` scope (the active scheduler is `warmup_stable_decay`).

### FR4 — `train()` signature and behaviour change
`src/training/pipeline.py::train` is modified to:
```python
def train(builder: PipelineBuilder, trial=None) -> float | None:
    ...
    return best_reg_error_val  # metrics_storage.best_metric_value, or None if never validated
```
Requirements:
- **Backward compatible**: called as `train(builder)` (no `trial`, W&B disabled) it behaves
  exactly as today except it now *returns* a float; `train.py`'s `main()` ignores the return.
- **Pruning**: when `trial is not None`, after every `validate(...)` call, report the latest
  `reg_error_val` to Optuna via `trial.report(value, step)` where `step` is a monotonically
  increasing validation counter (0,1,2,…). If `trial.should_prune()` returns `True`, finish the
  W&B run (if any) and raise `optuna.TrialPruned`. Pruning must only be attempted on validation
  epochs where `reg_error_val` was actually appended (guard on list length growth).
- **Return value**: the best (minimum) `reg_error_val` across the trial, read from
  `metrics_storage.best_metric_value`. If the trial ran but never validated (should not happen
  given FR3), return `None` and the driver treats it as a failed trial.
- No change to loss computation, checkpoint selection, or split handling.

### FR5 — W&B logging
W&B logging is gated by a new config flag `training.wandb.enabled` (default `False`, so
`train.py` is unaffected). When `True`, `train()`:
- Assumes the W&B run has already been initialised by the caller (the driver calls
  `wandb.init(...)` before `train`), and uses the active `wandb.run`. `train()` does **not** call
  `wandb.login` (the node already has the key) and does **not** call `wandb.finish` on the normal
  path (the driver owns the run lifecycle); it only calls `wandb.finish` right before raising
  `optuna.TrialPruned` so a pruned run is closed cleanly.
- After each training epoch, logs the aggregated train metrics (the `agg_loss_info` dict returned
  by `finalize_epoch`) with the epoch as `step`.
- After each `validate(...)`, logs the newly appended validation metrics — at minimum
  `reg_error_val`, `duration_error_val`, `accuracy`, and the losses — for that epoch.
- Import of `wandb` is done lazily inside the `enabled` branch so the dependency is optional for
  normal training.

### FR6 — Optuna study, storage, sampler, pruner
The driver creates (or resumes) the study with:
- `direction="minimize"`, `study_name` from config (default `"mixer_hp_search"`).
- `storage="sqlite:///<outputs>/hp_search/<study_name>.db"` (path from config), and
  `load_if_exists=True` so a killed HPC job resumes the same study.
- `sampler=optuna.samplers.TPESampler(seed=<config seed>)` for reproducibility.
- `pruner=optuna.pruners.MedianPruner(n_startup_trials=..., n_warmup_steps=...)` with values from
  config (defaults `n_startup_trials=5`, `n_warmup_steps=2` → do not prune before 2 validation
  reports).
- `study.optimize(objective, n_trials=<config n_trials>, gc_after_trial=True)`.

### FR7 — Per-trial isolation and artifact storage
Each trial's outputs live under `<outputs>/hp_search/<study_name>/trial_<number>/` containing:
- `metrics.json` — written by `MetricsStorage` (`config.training.metric_file` set to this path).
- `model.pth` — best checkpoint (`config.training.checkpoint_file`).
- `split.pth` — split indices (`config.training.splits_file`).
- `config.yaml` — the fully resolved OmegaConf config used for the trial (snapshot for
  reproducibility; the driver must write it because it bypasses `hydra.main`).
- `inference_records/` — only if the recorder is enabled (it is not, by default).

The driver replicates `train.py::add_metric_and_checkpoint_paths` (which normally sets these
paths from the Hydra run dir) because `scripts/hp_search.py` does not go through `@hydra.main`.

At study end, the driver writes to `<outputs>/hp_search/<study_name>/`:
- `trials.csv` — `study.trials_dataframe()` (all trials, params, values, states).
- `best_params.yaml` — the best trial's number, objective value, and full sampled param dict.

### FR8 — Config composition correctness
The objective builds the trial config via Hydra `compose`:
```python
with hydra.initialize(version_base=None, config_path="../configs"):
    cfg = hydra.compose(config_name="main", overrides=[
        "exp=hp_search",
        *[f"{key}={value}" for key, value in sampled.items()],
    ])
```
Requirements:
- The composed config MUST reflect every sampled override (verified: e.g. `cfg.model.n_encoder`
  equals the value passed to `suggest_int`).
- `exp=hp_search` MUST win over the default `exp=whole_model_pretraining` for `Phases`,
  `pretrained_encoder_path`, epoch counts, scheduler epochs, and scheduled-sampling epochs.
- Float overrides are formatted so Hydra/OmegaConf parses them as floats (avoid scientific
  notation ambiguity for `weight_decay`; format with enough precision, e.g. `repr(value)`).
- `hydra.initialize` uses a `config_path` **relative to `scripts/hp_search.py`** (`"../configs"`).

### FR9 — Failure and resume semantics
- A trial that raises `optuna.TrialPruned` is recorded as `PRUNED` (the objective re-raises it).
- A trial that raises any other exception (e.g. CUDA OOM from a large `n_*` combination) is
  recorded as `FAILED`; the driver must let Optuna capture it (do **not** swallow into a fake
  objective value) so the study continues to subsequent trials. The W&B run for that trial is
  finished with `exit_code=1`.
- Re-running `scripts/hp_search.py` with the same `study_name` and storage resumes the existing
  study and only runs the remaining `n_trials` budget (Optuna's standard behaviour with
  `load_if_exists=True`).

### FR10 — No regression to default training
`python train.py` and `python train.py exp=<existing>` MUST produce identical behaviour to
before this feature (W&B disabled, `trial=None`, `train()` return value ignored). The only
change visible to that path is the added (ignored) return value and the new optional config key
`training.wandb` (absent → treated as disabled via `.get`).

## Public API Summary

```python
# src/training/pipeline.py
def train(builder: PipelineBuilder, trial: "optuna.Trial | None" = None) -> "float | None":
    """Train per configured phases. If `trial` is given, report reg_error_val for pruning.
    If config.training.wandb.enabled, log epoch/val metrics to the active wandb run.
    Returns the best (minimum) reg_error_val observed, or None if never validated."""

# scripts/hp_search.py
def build_search_config(path: str = "configs/hp_search.yaml") -> DictConfig: ...
def suggest_overrides(trial: "optuna.Trial", search_space: DictConfig) -> dict[str, object]:
    """Sample all 15 hyperparameters; return {hydra_key: value}."""
def compose_trial_config(overrides: dict[str, object]) -> DictConfig:
    """Hydra-compose main.yaml with exp=hp_search plus sampled overrides."""
def make_objective(search_cfg: DictConfig): 
    """Return objective(trial) -> float that runs one trial end to end."""
def main() -> None:
    """Create/resume the study, run study.optimize, write trials.csv + best_params.yaml."""
```

```yaml
# configs/hp_search.yaml (search-meta config, loaded by the driver via OmegaConf, NOT composed)
study:
  study_name: "mixer_hp_search"
  n_trials: 50
  storage_dir: "outputs/hp_search"        # sqlite:///<storage_dir>/<study_name>.db
  sampler_seed: 42
  pruner:
    n_startup_trials: 5
    n_warmup_steps: 2
wandb:
  enabled: true
  project: "noisy-eye-scanpath"
  entity: null                            # null → default entity for the node's API key
  mode: "online"                          # "online" | "offline" | "disabled"
  group: "mixer_hp_search"                # defaults to study_name
search_space:
  weight_decay:   {low: 1.0e-5, high: 1.0e-1, log: true}
  n_encoder:      {low: 2, high: 8}
  n_decoder:      {low: 2, high: 8}
  n_eye_decoder:  {low: 2, high: 8}
  dropouts:       {low: 0.0, high: 0.5}   # applied to each of the 9 dropout keys
  cls_weight:     {low: 0.05, high: 1.0}
  dur_weight:     {low: 0.05, high: 1.0}
```

## Dependencies

| Reads from | Purpose |
|---|---|
| `configs/main.yaml` (+ groups) | Base training config composed per trial |
| `configs/exp/hp_search.yaml` (new) | Reduced-budget Combined-only override |
| `configs/hp_search.yaml` (new) | Study, W&B, and search-space settings |
| `src/training/pipeline_builder.py::PipelineBuilder` | Builds model/data/optim from composed config |
| `src/training/training_utils.py::MetricsStorage`, `validate` | Source of `reg_error_val` |
| WANDB API key on the node (env / `~/.netrc`) | W&B auth, assumed pre-configured |

| Writes to | Purpose |
|---|---|
| `outputs/hp_search/<study_name>.db` | SQLite Optuna study (resumable) |
| `outputs/hp_search/<study_name>/trial_<n>/` | Per-trial `metrics.json`, `model.pth`, `split.pth`, `config.yaml` |
| `outputs/hp_search/<study_name>/trials.csv` | All-trials summary |
| `outputs/hp_search/<study_name>/best_params.yaml` | Best trial params + objective |
| W&B project `<wandb.project>` | One run per trial with metric curves + params |
| `requirements.txt` | Add `optuna`, `wandb` |
