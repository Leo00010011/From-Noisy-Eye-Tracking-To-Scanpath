# Plan — Revive PathModel Training (with Scheduled Sampling)

## Context and Design Decisions

**Why the model won't train today.** `train()` calls `model.set_scheduled_sampling(scheduled_sampling)`
at `src/training/pipeline.py:62` for *every* model, but only `MixerModel` implements it. Even if that
line were guarded, validation runs the model autoregressively **through** the sampler: in
`validate()` the model is in `eval()`, `ScheduledSampling.get_current_ratio()` returns `1`, and
`MixerModel.forward` therefore routes to `self.scheduled_sampling(**kwargs)`. A `PathModel` whose
`forward` ignores the sampler would be teacher-forced at eval time (leaking ground truth) and would
never exercise the free-running decode the Mission requires. So the fix is not a guard — it is giving
`PathModel` the same four seams `MixerModel` already has.

**Reuse, don't re-implement, scheduled sampling.** `ScheduledSampling.__call__`
(`training_utils.py:345`) is model-agnostic: it sets `pass_sampler=True`, calls `model.encode(...)`
once, optionally `model.decode_denoise(...)` **only** when `phase=='Combined'`, optionally
`model.enable_memory_kv_cache()` (guarded by `hasattr`), then loops `model(**input)`. Every hook it
needs already exists on `PathModel` (`encode`, `decode_fixation`) or is trivial to add
(`decode_denoise` → `{}`; `phase`; `set_scheduled_sampling`; a sampler-aware `forward`). No change to
`training_utils.py`. This keeps the sampler's tested behaviour identical across both model families.

**Fixation-only, gaze-only — by decision.** Per the clarifying answers: PathModel gets **no** denoise
head and runs **only** the `Fixation` phase with `separated_reg` loss and the `multi_mlp` head (the
same fixation objective `MixerModel` uses, so the ablation isolates *image features*, not the loss).
It loads gaze only — the plain `DataLoader` branch, no `DeduplicatedMemoryDataset`, no DINOv3. This is
the Mission's literal definition of `PathModel` ("gaze-only … no image features").

**Split reuse over fresh splits.** Per the clarifying answer, the run reuses a saved `split.pth` from
a reference `MixerModel` run so the two models see the identical test set. `load_test_data` already
reads `<dir>/split.pth` and returns index tensors; `train()` already prefers `build_model`'s returned
`splits` over `make_splits()`. So the entire mechanism is one config key
(`training.reuse_split_from`) plus a three-line load in `build_model`. A side benefit: reusing a split
means `make_splits()` is never called on the gaze-only path, sidestepping its dependence on
`self.data` (only built when images load). FR9 adds a small guard for the fallback case so the
gaze-only path can't crash if someone runs it without a saved split.

**Additivity.** Everything lands in `path_model.py`, the PathModel branch of `build_model`,
`load_dataset` (a metadata-only guard), and new/edited configs. `MixerModel`, `ScheduledSampling`,
and their state-dicts are untouched, satisfying the project's dual-path / no-regression norm.

## Implementation Steps

### Step 1 — `PathModel` scheduled-sampling seams (`src/model/path_model.py`)
Add the four seams, mirroring `MixerModel`.

- In `__init__` (near the other attribute assignments, before the head construction), add:
  ```python
  self.phase = None
  self.scheduled_sampling = None
  ```
- Replace the no-op `set_phase`:
  ```python
  def set_phase(self, phase):
      self.phase = phase
  ```
  (No `requires_grad` toggling — PathModel has one parameter group.)
- Add `set_scheduled_sampling` and `decode_denoise`:
  ```python
  def set_scheduled_sampling(self, scheduled_sampling):
      self.scheduled_sampling = scheduled_sampling
      self.scheduled_sampling.set_model(self)

  def decode_denoise(self, **kwargs):
      return {}
  ```
- Replace `forward` with the sampler-aware version (FR5):
  ```python
  def forward(self, skip_denoise=False, **kwargs):
      if (self.scheduled_sampling is not None
              and not kwargs.get('pass_sampler', False)
              and self.scheduled_sampling.get_current_ratio() > 0):
          return self.scheduled_sampling(**kwargs)
      if not kwargs.get('pass_sampler', False):
          self.encode(**kwargs)
      if self.phase == 'Fixation':
          return self.decode_fixation(**kwargs)
      elif self.phase == 'Combined':
          denoise_output = {} if skip_denoise else self.decode_denoise(**kwargs)
          return {**denoise_output, **self.decode_fixation(**kwargs)}
      return self.decode_fixation(**kwargs)
  ```
  Note: `encode` already stores `self.memory`/`self.src_mask`, and `decode_fixation` reads them, so
  the sampler's "encode once, decode many with `pass_sampler=True`" pattern works without a KV cache
  (each decode step re-embeds the growing `tgt`; acceptable for a baseline). `encode`/`decode_fixation`
  signatures are unchanged; the extra kwargs the sampler and collate pass are absorbed by `**kwargs`.

### Step 2 — Split-reuse in `PipelineBuilder.build_model` (`src/training/pipeline_builder.py`)
`load_test_data` is already imported at the top of the module (line 24). At the **end** of
`build_model`, immediately before `return model, splits` (currently line 666), add:
```python
reuse_split_from = self.config.training.get('reuse_split_from', None)
if reuse_split_from is not None and splits is None:
    print(f"Reusing split from {reuse_split_from}")
    splits = load_test_data(self, reuse_split_from, return_dataloaders=False)
```
This runs for both model families but is a no-op unless the key is set and no earlier path already
populated `splits`. `train()` (`pipeline.py:16`) already consumes a non-`None` `splits` and skips
`make_splits()`.

### Step 3 — Gaze-only `make_splits` guard in `load_dataset` (`src/training/pipeline_builder.py`)
Only relevant when a saved split is **not** reused. In `load_dataset`, after `self.PathDataset` is
built and before the `use_img` block, ensure the split metadata exists for disjoint strategies even
without images:
```python
split_name = getattr(getattr(self.config.data, 'split_strategy', None), 'name', None)
needs_metadata = split_name in ('disjoint', 'stimuly_disjoint')
if needs_metadata and self.data is None:
    self.data = CocoFreeView(data_path=data_path)
    self.data.filter_by_idx(self.PathDataset.data_store['filtered_idx'])
```
The existing `use_img` block must not rebuild `self.data` when it is already set (it already guards
with `if self.data is None`). No image dataset is constructed in the gaze-only case. (This is a
defensive guard; the headline path reuses a split and never reaches `make_splits`.)

### Step 4 — Make `configs/model/path_model.yaml` self-consistent
Set the head to match the `separated_reg` loss (which reads `output['coord']`/`['dur']`/`['cls']`):
```yaml
head_type: "multi_mlp"
mlp_head_hidden_dim: [256, 128]
```
Leave `input_encoder: "shared_gaussian"`, `input_dim: 3`, `output_dim: 3`, `name: "PathModel"`,
`n_encoder`/`n_decoder`/`model_dim`/`n_heads`/`ff_dim`/dropouts as-is. (The `main.yaml`
`head_type=multi_mlp` group already overrides at compose time; this edit makes a standalone
`model=path_model` compose correctly and documents intent. `param_summary` iterates
`mlp_head_hidden_dim`, so it must be a list, not `None`.)

### Step 5 — New experiment config `configs/exp/path_model_training.yaml`
```yaml
# @package _global_
defaults:
  - override /model: path_model

training:
  decisive_metric: "reg_error_val"
  pretrained_model: null
  use_scheduled_sampling: true
  Phases: ["Fixation"]
  # Point this at a MixerModel run directory containing split.pth for an apples-to-apples
  # comparison; leave null to fall back to a freshly generated split (FR9 guard applies).
  reuse_split_from: null
  Fixation:
    name: "Fixation"
    denoise_weight: 0
    decisive_metric: "reg_error_val"
    epochs: 100

model:
  pretrained_encoder_path: null

# Drop the combined (denoise) loss: fall through to the separated_reg loss group.
loss:
  complex_type: null

data:
  load:
    use_img_dataset: False

# warmup + active <= Fixation.epochs
scheduled_sampling:
  warmup_epochs: 0
  active_epochs: 60
  n_updates: -1
  min_prob: 0
  max_prob: 0.85
```
Run with: `python train.py exp=path_model_training` (optionally
`+training.reuse_split_from=outputs/<date>/<time>`). Hyperparameters (`epochs`, `active_epochs`,
`max_prob`, model width/depth) are sensible defaults, freely tunable.

### Step 6 — Tests (`tests/test_path_model_training.py`)
New CPU-only suite (see validation.md for the assertions). Cover: the four model seams (Step 1),
sampler routing under training vs eval, `build_model` returning a PathModel and honoring
`reuse_split_from`, the gaze-only dataloader shape, and a 1–2 epoch end-to-end smoke run on a tiny
synthetic dataset. Use a stub `ScheduledSampling` where a full one is overkill, and a real one for
the end-to-end test.

## Implementation Order
1. Step 1 — PathModel seams (unblocks `pipeline.py:62` and eval autoregression).
2. Step 2 — `build_model` split reuse.
3. Step 3 — `load_dataset` gaze-only guard.
4. Step 4 — `path_model.yaml` head consistency.
5. Step 5 — `path_model_training.yaml` experiment.
6. Step 6 — tests + a short real smoke run.
