# Requirements — Revive PathModel Training (with Scheduled Sampling)

## Goal

The image-reliance diagnostic suite showed that the Mask2Former-backbone `MixerModel` barely
uses the stimulus image: shuffling images within a batch degrades regression error by only ~4%.
That result only *means* something relative to a gaze-only reference. The `PathModel` — a standard
encoder–decoder transformer that takes the noisy eye-tracking trajectory as context and
autoregressively produces the scanpath, with **no image features** — is exactly that reference, and
it is the ablation baseline the Mission names. But `PathModel` is currently **not trainable through
the pipeline**: `src/training/pipeline.py:62` unconditionally calls
`model.set_scheduled_sampling(...)`, a method `PathModel` does not have, and `PathModel.forward`
does not implement the `pass_sampler` / autoregressive-decode protocol that `ScheduledSampling`
drives. This feature makes `PathModel` trainable end-to-end **with scheduled sampling**, on the same
CocoFreeView data and the same fixed train/val/test split as a reference `MixerModel` run, so the two
models can be compared apples-to-apples.

## Scope

**In scope**
- Add the scheduled-sampling seams to `PathModel` so it satisfies the exact contract
  `ScheduledSampling.__call__` and `src/training/pipeline.py` already assume for `MixerModel`
  (`set_scheduled_sampling`, a `phase` attribute, `pass_sampler`/`skip_denoise`-aware `forward`,
  a no-op `decode_denoise`).
- A gaze-only data path: the plain `DataLoader` branch (no image dataset, no DINOv3), reusing the
  existing `seq2seq_padded_collate_fn`.
- A `PathModel` experiment config wired for **Fixation-only** training with scheduled sampling and
  the `separated_reg` loss + `multi_mlp` head (matching `MixerModel`'s fixation objective).
- **Split reuse**: an optional config key that loads a saved `split.pth` from an existing run
  directory so `PathModel` trains and is evaluated on the identical split as a `MixerModel` run.
- Self-consistency cleanup of `configs/model/path_model.yaml`.

**Out of scope (explicitly)**
- Any denoise capability for `PathModel` (no denoise head, no `Denoise`/`Combined` phases). PathModel
  stays a pure fixation predictor per the Mission's description.
- A formal PathModel-vs-MixerModel comparison **script**/table (the Roadmap "Formal baseline
  comparison infrastructure" item). This spec only makes both runnable on the same split.
- Image features of any kind for `PathModel` (DINOv3, Mask2Former, precomputed).
- New evaluation metrics (DTW / multi-match remain a separate publication blocker).
- Generating a *fresh* gaze-only stimuli-disjoint split. The provided path reuses a saved split; a
  minimal guard (FR9) keeps `make_splits` from crashing on the gaze-only path, but new-split
  generation for gaze-only is not a supported headline path.

## Functional Requirements

**FR1 — `PathModel` scheduled-sampling attributes.** `PathModel.__init__` initialises
`self.phase = None` and `self.scheduled_sampling = None` (mirroring
`MixerModel.__init__`). No other constructor signature change.

**FR2 — `set_scheduled_sampling`.** `PathModel.set_scheduled_sampling(self, scheduled_sampling)`
stores the sampler on `self.scheduled_sampling` and calls `scheduled_sampling.set_model(self)` —
byte-for-byte the `MixerModel` implementation. After this call, `src/training/pipeline.py:62` no
longer raises `AttributeError`.

**FR3 — `set_phase` records the phase.** `PathModel.set_phase(self, phase)` sets
`self.phase = phase` and returns `None`. It performs **no** `requires_grad` toggling (PathModel has a
single parameter group and no phase-specific submodule freezing). `phase` is only ever `"Fixation"`
in the supported config; the value is stored so `forward`'s dispatch and any `getattr(model,
'phase', …)` probe read the truth.

**FR4 — `decode_denoise` no-op.** `PathModel.decode_denoise(self, **kwargs)` returns `{}`. This makes
the `Combined`-phase branch and any defensive caller safe even though the supported config never
enters a denoise phase. It adds no parameters and no state-dict keys.

**FR5 — `forward` routes through scheduled sampling (parity with `MixerModel`).**
`PathModel.forward(self, skip_denoise=False, **kwargs)` behaves as:
1. If `self.scheduled_sampling is not None` **and** `kwargs.get('pass_sampler', False)` is falsy
   **and** `self.scheduled_sampling.get_current_ratio() > 0`: return
   `self.scheduled_sampling(**kwargs)`.
2. If `pass_sampler` is falsy: call `self.encode(**kwargs)`.
3. Dispatch on `self.phase`: `"Fixation"` → `self.decode_fixation(**kwargs)`; `"Combined"` →
   `{**({} if skip_denoise else self.decode_denoise(**kwargs)), **self.decode_fixation(**kwargs)}`;
   otherwise (`None`/anything else) → `self.decode_fixation(**kwargs)`.
   When `self.scheduled_sampling is None`, the old behaviour is preserved: `encode` then
   `decode_fixation`. `encode(src, src_mask, **kwargs)` and `decode_fixation(tgt, tgt_mask, src_mask,
   in_tgt=None, **kwargs)` keep their current signatures; extra keys (`pass_sampler`, `skip_denoise`,
   `image_src`, `fixation_len`, `sample_idx`, …) are absorbed by `**kwargs`.

**FR6 — `ScheduledSampling` contract is met unchanged.** No edit to
`src/training/training_utils.py`. The existing loop calls, in order: `self.model.encode(**input)`
(with `input['tgt']=None`, `input['tgt_mask']=None`, `input['pass_sampler']=True`); optionally
`self.model.decode_denoise(**input)` **only** when `getattr(self.model,'phase',None)=='Combined'`
(never for PathModel); optionally `self.model.enable_memory_kv_cache()` (guarded by `hasattr`, absent
on PathModel); then repeated `self.model(**input)` with `pass_sampler=True`. `get_latest_output`
slices `value[:, -1:, :]` for every non-`denoise` key, and `get_final_output` concatenates —
PathModel's `{'coord','dur','cls'}` (multi_mlp) satisfy this.

**FR7 — Gaze-only data loading.** With `data.load.use_img_dataset=False` (and
`use_precomputed_features` unset), `PipelineBuilder.build_dataloader` uses the plain-`DataLoader`
branch with `seq2seq_padded_collate_fn`; batches contain `src (B,T,3)`, `tgt (B,N,3)`, `src_mask`,
`tgt_mask`, `fixation_len` and **no** `image_src`. `PathModel(**input)` consumes `src`/`src_mask`/
`tgt`/`tgt_mask`; all other keys are ignored via `**kwargs`.

**FR8 — Split reuse.** A new optional config key `training.reuse_split_from` (default `null`) names a
run directory containing `split.pth`. In `PipelineBuilder.build_model`, immediately before
`return model, splits`, if `reuse_split_from` is set **and** `splits is None`, set
`splits = load_test_data(self, reuse_split_from, return_dataloaders=False)` (which reads
`<dir>/split.pth` and returns `(train_idx, val_idx, test_idx)`). `train()` then uses those indices
and never calls `make_splits()`. If `reuse_split_from` points to a directory with no `split.pth`,
`load_test_data` raises (its existing `Exception("Split not found !!!")`). The key is model-agnostic
but only takes effect when no other path (`pretrained_model`, `pretrained_encoder_path`) already
produced `splits`.

**FR9 — `make_splits` guard for the gaze-only fallback.** When `reuse_split_from` is unset and a
stimuli-disjoint split strategy is configured, `make_splits`/`load_dataset` must not dereference a
`None` `self.data`. `PipelineBuilder.load_dataset` builds the `CocoFreeView` metadata object
(`self.data`, filtered by `PathDataset.data_store['filtered_idx']`) whenever
`config.data.split_strategy.name in {'disjoint','stimuly_disjoint'}`, **even if** `use_img` is
`False`. No image tensors are loaded in this case (no `DeduplicatedMemoryDataset`). This keeps the
gaze-only path from raising `AttributeError: 'NoneType' object has no attribute 'get_all_stimuli'`.

**FR10 — Config: `configs/model/path_model.yaml`.** Updated to be self-consistent with the
`separated_reg` + `multi_mlp` objective: `head_type: "multi_mlp"`, add
`mlp_head_hidden_dim: [256, 128]`. `input_encoder` stays `"shared_gaussian"`, `input_dim: 3`,
`output_dim: 3`, `name: "PathModel"`. (The `head_type=multi_mlp` group in `main.yaml` already
overrides at compose time; this edit makes standalone `model=path_model` composition correct too.)

**FR11 — Config: `configs/exp/path_model_training.yaml`.** A new `@package _global_` experiment that
(a) selects the PathModel via a defaults override, (b) restricts to `Phases: ["Fixation"]`,
(c) enables scheduled sampling, (d) disables the denoise-oriented combined loss, (e) uses the
gaze-only loader, and (f) exposes `training.reuse_split_from`. Schedule invariant:
`scheduled_sampling.warmup_epochs + active_epochs <= training.Fixation.epochs`.

**FR12 — No regression to `MixerModel`.** All `PathModel` changes are additive to `path_model.py`
and to the PathModel branch of `build_model`; the `MixerModel` construction path, `forward`,
`ScheduledSampling`, and their state-dicts are byte-identical. `train.py` on the default
(`model=mixer_model`) config is unchanged.

## Public API Summary

```python
# src/model/path_model.py — PathModel additions
def __init__(self, ...):
    ...
    self.phase = None
    self.scheduled_sampling = None

def set_phase(self, phase):            # was a bare `return`
    self.phase = phase

def set_scheduled_sampling(self, scheduled_sampling):
    self.scheduled_sampling = scheduled_sampling
    self.scheduled_sampling.set_model(self)

def decode_denoise(self, **kwargs):
    return {}

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

# src/training/pipeline_builder.py — build_model, just before `return model, splits`
reuse_split_from = self.config.training.get('reuse_split_from', None)
if reuse_split_from is not None and splits is None:
    splits = load_test_data(self, reuse_split_from, return_dataloaders=False)
```

## Dependencies

| Reads from | Writes to |
|---|---|
| `configs/main.yaml` (Phases, scheduled_sampling, training.*) | `self.scheduled_sampling`, `self.phase` on `PathModel` |
| `src/training/training_utils.py::ScheduledSampling` (unchanged) | `splits` returned by `PipelineBuilder.build_model` |
| `src/model/model_io.py::load_test_data` (reads `<dir>/split.pth`) | run outputs: `metrics.json`, `model.pth`, `split.pth` |
| `src/data/datasets.py::seq2seq_padded_collate_fn` (gaze-only batches) | — |
| `configs/loss/separated_loss.yaml`, `configs/head_type/multi_mlp.yaml` | — |
