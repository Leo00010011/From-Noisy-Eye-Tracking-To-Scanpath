# Validation — Revive PathModel Training (with Scheduled Sampling)

## Code Correctness

### Group 1 — PathModel seams (unit, no sampler)
- [ ] Construct `PathModel(n_encoder=2, n_decoder=2, model_dim=32, total_dim=32, n_heads=4,
      ff_dim=64, head_type='multi_mlp', mlp_head_hidden_dim=[16,8],
      input_encoder='shared_gaussian', norm_first=True)`. Assert `model.phase is None` and
      `model.scheduled_sampling is None` immediately after construction (FR1).
- [ ] `model.set_phase('Fixation')` sets `model.phase == 'Fixation'` and returns `None` (FR3). Assert
      the parameter `requires_grad` flags are **unchanged** by `set_phase` (no freezing).
- [ ] `model.decode_denoise(src=torch.randn(2,5,32)) == {}` (FR4).
- [ ] With `scheduled_sampling is None` and `phase='Fixation'`, a teacher-forced
      `forward(src=(2,7,3), src_mask=None, tgt=(2,4,3), tgt_mask=causal(2,5))` returns a dict with
      keys `{'coord','dur','cls'}`; `coord` has shape `(2,5,2)`, `dur` `(2,5,1)`, `cls` `(2,5,1)`
      (start token prepended ⇒ output length `N+1=5`). This is the pre-existing behaviour preserved
      by FR5's `scheduled_sampling is None` branch.

### Group 2 — Sampler routing (unit, stub sampler)
- [ ] `set_scheduled_sampling(stub)` sets `model.scheduled_sampling is stub` and calls
      `stub.set_model(model)` exactly once with the model (FR2). After it, `hasattr(model,
      'set_scheduled_sampling')` is `True` — the exact attribute `pipeline.py:62` needs.
- [ ] Stub whose `get_current_ratio()` returns `0.0`: `forward(...)` with `pass_sampler` absent does
      **not** call `stub.__call__`; it runs the normal `encode`→`decode_fixation` path (assert the
      stub's `__call__` counter stays 0).
- [ ] Stub whose `get_current_ratio()` returns `0.5`: `forward(...)` with `pass_sampler` absent
      returns `stub(**kwargs)` (assert `__call__` invoked once, forward's return is the stub's
      sentinel object).
- [ ] `forward(pass_sampler=True, ...)` never calls the stub and never re-encodes: patch/spy
      `model.encode`; assert it is **not** called when `pass_sampler=True` (mirrors how
      `ScheduledSampling.__call__` re-enters the model). Then `decode_fixation` runs against the
      `self.memory` set by the sampler's own earlier `encode`.

### Group 3 — Real ScheduledSampling drive (unit)
- [ ] Build a real `ScheduledSampling(active_epochs=1, warmup_epochs=0, device='cpu',
      steps_per_epoch=1, max_prob=0.85)`, `set_scheduled_sampling` it, `set_phase('Fixation')`,
      `sampler.step()` so `get_current_ratio()>0`. In **train** mode, `model(**input)` (gaze-only
      `input`) returns concatenated `{'coord','dur','cls'}` whose sequence length equals
      `tgt_mask.size(1)` (the decode budget). No exception; `model.enable_memory_kv_cache` is absent
      so the `hasattr` guard skips it.
- [ ] In **eval** mode (`model.eval()`), `get_current_ratio()` returns `1`, so `model(**input)` runs
      fully autoregressive (assert the sampler `__call__` path is taken and output length ==
      `tgt_mask.size(1)`). This is the path `validate()` exercises.

### Group 4 — PipelineBuilder / build_model
- [ ] With a minimal config (`model.name='PathModel'`, gaze-only), `PipelineBuilder(cfg).build_model()`
      returns `(model, splits)` where `isinstance(model, PathModel)` and, with
      `training.reuse_split_from` unset, `splits is None` (FR8 no-op).
- [ ] `build_scheduled_sampling(steps_per_epoch=1)` returns a `ScheduledSampling` when
      `training.use_scheduled_sampling=True`; `train()`'s line
      `model.set_scheduled_sampling(scheduled_sampling)` executes without `AttributeError` (FR2/FR12
      regression check — the original bug).
- [ ] `build_loss_fn()` with `loss.complex_type=None` + `loss.type='separated_reg'` returns a
      `SeparatedRegLossFunction` (not `CombinedLossFunction`), i.e. no denoise term (FR11).

### Group 5 — Split reuse (FR8)
- [ ] Write a `split.pth` via `save_splits(train_idx, val_idx, test_idx, tmp/split.pth)` with known
      index tensors. Set `training.reuse_split_from=<tmp dir>`; `build_model()` returns `splits ==
      (train_idx, val_idx, test_idx)` (element-wise `torch.equal`), proving the exact indices are
      reused, not regenerated.
- [ ] `training.reuse_split_from` pointing at a directory **without** `split.pth` makes `build_model`
      raise (propagated `Exception("Split not found !!!")` from `load_test_data`).
- [ ] Integration: monkeypatch `PipelineBuilder.make_splits` to raise; run `train()` for 0-effective
      work with `reuse_split_from` set → `make_splits` is **never** called (splits came from
      `build_model`).

### Group 6 — Gaze-only dataloader (FR7) and make_splits guard (FR9)
- [ ] With `data.load.use_img_dataset=False`, `build_dataloader` returns plain `DataLoader`s (not
      `CoupledDataloader`); a batch dict has keys ⊇ `{'src','tgt','src_mask','tgt_mask',
      'fixation_len'}` and **no** `'image_src'`. `src` is `(B,T,3)`, `tgt` `(B,N,3)`.
- [ ] With `use_img_dataset=False` and `split_strategy.name='stimuly_disjoint'`, `load_dataset()`
      builds `self.data` (a `CocoFreeView`) and `make_splits()` completes without
      `AttributeError` — no image dataset is constructed (`self.img_dataset is None`).

## Data Validity

- [ ] **Autoregressive output is well-formed.** After a short real training run
      (`exp=path_model_training`, `Fixation.epochs=2`, a small `Subset`), run the model on a val batch
      in eval mode and invert normalization: predicted fixation coordinates lie in `[0,1]` before
      inversion; `pred_len` (first `sigmoid(cls)>0.5`) is ≥1 and ≤ the decode budget for every sample
      (no length collapse to 0 or run-away to the cap on every row).
- [ ] **Scheduled-sampling ratio schedule is monotonic non-decreasing** across `step()` calls from
      batch 0 through `warmup+active` epochs, saturating at `max_prob` (0.85). Print the ratio per
      epoch from a dry schedule and assert monotonicity.
- [ ] **`reg_error_val` is produced.** The 2-epoch run appends at least one `reg_error_val` to
      `metrics.json` (validation ran and `coord_error_acum>0`), and a `model.pth` + `split.pth` are
      written to the run directory.
- [ ] **Baseline sanity vs. input noise.** On the val split, PathModel's `reg_error_val` (normalized
      Euclidean) is finite and below the raw noisy-input error (noisy `src` vs clean `tgt`) — the
      model is learning *something* from gaze, not emitting the input. (Sanity threshold, not a
      publication metric.)

## Data Architecture Integrity

- [ ] **Split-file roundtrip is exact.** `load_test_data(builder, dir, return_dataloaders=False)`
      returns index tensors `torch.equal` to those `save_splits` wrote — no reshuffle, no dtype drift
      (indices stay integer). Reusing the same directory in two separate `build_model` calls yields
      identical splits.
- [ ] **Reuse bypasses regeneration.** When `reuse_split_from` is set, `make_splits` is not invoked
      (Group 5) — the invariant that a reused split is not silently replaced by a fresh random split
      is not bypassable.
- [ ] **Gaze-only path stays gaze-only.** No `image_src` key ever reaches `PathModel.forward` in the
      gaze-only config (assert over a full epoch of batches), and no `DeduplicatedMemoryDataset` /
      DINOv3 / Mask2Former object is constructed (`self.img_dataset is None`, no image encoder on the
      model). This guarantees the ablation truly removes image features.
- [ ] **MixerModel regression tripwire.** Constructing `MixerModel` from the default config and
      running one teacher-forced forward is byte-identical (`torch.equal` on `coord`) before and after
      this change on a fixed seed — the `path_model.py` edits and the `build_model` split-reuse block
      do not perturb the MixerModel path (FR12).
