# Requirements — Image-Feature Adaptation via Scanpath-Centroid Alignment Pretraining

## Goal

Add a Deformable-DETR-style **image-feature adaptation** stage to `MixerModel` on the
**precomputed frozen Mask2Former** path. For every image-feature token, a small head predicts a
2-D vector from the token's spatial anchor (its center in normalized `[0,1]` coordinates) to the
**nearest fixation centroid of that stimulus**. The head shares its trunk with the existing
`img_input_proj` adapter, so the *intermediate* (trunk) features the deformable cross-attentions
already consume as image memory are shaped by this alignment objective. The intent is to pre-adapt
the frozen image features for the fixation-regression task — resolving the image/scanpath spatial
misalignment — **before** the encoder/decoder are trained, mirroring how the (abandoned) Denoise
phase was meant to pre-adapt the gaze encoder. The predicted offset is an **auxiliary supervision
signal only**: it does **not** displace the deformable reference points.

**Well-posedness — the target is image-level, not per-scanpath.** Regressing each token toward the
nearest fixation of the *current sample's* scanpath is ill-posed: the same image token receives a
different target for every subject viewing that image, injecting subject-specific noise into a
supervision signal that is meant to be image-intrinsic. Instead the target is **scanpath-agnostic**:
for each stimulus we aggregate the fixations of **all** its scanpaths, spatially cluster them with
**Mean Shift** (bandwidth = **1 degree of visual angle**), and each token regresses toward the
nearest **cluster centroid**. Clustering de-noises the aggregated point cloud (many near-duplicate
fixations across subjects collapse into one centroid), yielding a stable, image-intrinsic
attentional-landmark target. These centroids are **precomputed once per unique image** and cached to
disk, keyed by image in the same first-seen order as the frozen-feature cache.

The full training then runs in **three ordered phases** in one run:
1. **`ImageAdaptation`** — train the adapter trunk + alignment head only (centroid-alignment loss).
2. **`Combined`** — train encoder + decoder + heads, **adapter frozen** (fixation loss).
3. **`FullFinetune`** — train the whole model **including** the adapter, except the (already-frozen,
   precomputed) Mask2Former pixel decoder (fixation loss).

## Scope

**In scope**
- An offline precompute of per-image fixation centroids (Mean Shift, 1-DVA bandwidth) and a keyed
  cache (`ScanpathCentroidCache`), aligned to the frozen-feature cache's first-seen unique order.
- A new alignment head on the Mask2Former **precomputed** path, sharing `img_input_proj`'s trunk.
- Delivery of per-image centroids into the training forward via the batch's existing `image_idx`,
  through non-persistent model buffers gathered per batch.
- A new `ImageAdaptation` phase and a new `FullFinetune` phase, wired through `set_phase` /
  `build_phases` / the forward router / the loss / the validation metric; a new `adapter_modules`
  parameter group.
- A new `AlignmentLoss` (nearest-centroid regression) integrated into `CombinedLossFunction`, a new
  `align_error_val` metric, a new `model.image_adaptation.*` config sub-tree (default **off**), and
  a new experiment config driving the 3-phase run.
- `scikit-learn` added to `requirements.txt` for `MeanShift` (with `DBSCAN` as a documented
  alternative).

**Explicitly out of scope**
- Per-scanpath targets (the rejected, ill-posed formulation).
- The **live** Mask2Former backbone and the **DINOv3** path. The feature only activates on the
  precomputed Mask2Former path; enabling it elsewhere raises at construction.
- Displacing / refining deformable **reference points** with the predicted offset (auxiliary-loss
  only, per the design decision).
- Any change to the **frozen-feature** HDF5 layout, its precompute driver, `ImageFeatureCache`, or
  `PrecomputedFeatureDataset`. The centroid cache is a **separate, additive** file.
- Fine-tuning / unfreezing the pixel decoder or the ResNet50 backbone.
- Keeping the alignment term as an auxiliary loss in `Combined` / `FullFinetune` (documented
  optional extension, not the default).
- `PathModel` and the online/live encoders.

## Functional Requirements

### Centroid precompute & cache

- **FR1** — New driver `scripts/build_scanpath_centroid_cache.py` (argparse CLI:
  `--out`, `--bandwidth-dva` default `1.0`, `--algorithm` default `meanshift` (`meanshift`|`dbscan`),
  `--min-samples`/`--dbscan-eps-dva` for DBSCAN, `--data-path`). It:
  1. Builds a `CocoFreeView` and enumerates **unique** stimuli in **first-seen order** — byte-identical
     to `DeduplicatedMemoryDataset.build_index` / `PrecomputedFeatureDataset` (the keying invariant).
  2. For each unique image, collects the **downscaled** fixation coordinates (`get_scanpath(...,
     downscale=True)`, i.e. the 512×320 `dest_res` space) of **all** scanpaths whose `img_path`
     equals that image, concatenated into a `(P, 2)` point cloud `(x_px, y_px)`.
  3. Clusters the cloud with `MeanShift(bandwidth = bandwidth_dva / CocoFreeView.ptoa)` — i.e.
     `1.0 / (1/16) = 16 px` per DVA in the `dest_res` space (isotropic there). Cluster **centers**
     are taken as centroids. (DBSCAN alternative: cluster, then use per-label means; noise points
     `label == -1` are dropped.)
  4. Normalizes centroids to `[0,1]` with the **same** per-axis divisor `Normalize(key='y',
     mode='coords')` uses (`max_value = [W, H] = dest_res reversed = [512, 320]`; FR19), so centroids
     live in the identical frame as `tgt`.
- **FR2** — `ScanpathCentroidCache` (new, in `src/data/image_feature_cache.py` or a sibling
  `src/data/scanpath_centroids.py`) writes/reads one HDF5 file, single group `/centroids`, mode
  `"w"`:

  | Dataset | Shape | dtype | Notes |
  |---|---|---|---|
  | `centroids` | `(U, C_max, 2)` | float32 | normalized `[0,1]` `(x,y)`; NaN-padded past `n_centroids[u]` |
  | `centroid_mask` | `(U, C_max)` | bool | True = real centroid |
  | `image_path` | `(U,)` | vlen utf8 | unique image path, first-seen order — keying invariant |

  Group attrs: `bandwidth_dva`, `ptoa`, `dest_res` (`[H,W]`), `max_value` (`[W,H]` normalizer),
  `algorithm`, `C_max`, `num_unique`, `created_at`. `C_max` = max centroids over all images.
- **FR3** — The read side rebuilds the first-seen unique index from the (filtered) `CocoFreeView`
  and **verifies** `image_path[u]` against the rebuilt path **unconditionally** — a mismatch raises
  `ValueError` (order invariant not bypassable, mirroring `PrecomputedFeatureDataset`). Exposes
  `centroids` `(U,C_max,2)` and `centroid_mask` `(U,C_max)` as tensors (NaN cleared to `0` on the
  masked-out slots; the mask is authoritative).

### Configuration & gating

- **FR4** — `MixerModel.__init__` gains `image_adaptation=False`, `align_head_hidden_dim=None`,
  `align_head_output_dropout=0`. All default to today's behavior.
- **FR5** — The alignment machinery is built **only** when `image_adaptation` is truthy **and**
  `image_encoder is not None` **and** `image_encoder_type == "mask2former"`; otherwise (truthy flag
  with wrong/absent encoder) `__init__` raises `ValueError` naming the constraint.
- **FR6** — When `image_adaptation` is falsy, `MixerModel` is **byte-identical** to today
  (`img_input_proj` in `denoise_modules`, no `align_head`, empty `adapter_modules`, no new state_dict
  keys, old checkpoints load clean, `set_phase` unchanged).

### Model architecture & centroid delivery

- **FR7** — When active (FR5), `img_input_proj` is placed in a new `self.adapter_modules` list
  **instead of** `denoise_modules`. It is unchanged structurally; its `model_dim` output is the
  "intermediate features" that feed cross-attention.
- **FR8** — New `self.align_head = MLP(model_dim, align_head_hidden_dim, 2,
  output_dropout_p=align_head_output_dropout, ...)` appended to `adapter_modules`; maps each token's
  trunk feature → a 2-D offset in normalized `[0,1]` units.
- **FR9** — Centroids reach the model as **non-persistent buffers** (absent from state_dict, so
  FR6 holds and checkpoints stay clean): `self.align_centroids (U, C_max, 2)` and
  `self.align_centroid_mask (U, C_max)`. They are installed post-construction by
  `MixerModel.set_alignment_centroids(centroids, mask)` (called by `PipelineBuilder`). Before they
  are set, `decode_align` raises `RuntimeError`.
- **FR10** — In `encode`, Mask2Former branch, immediately after
  `image_src = self.img_input_proj(bundle.value)`, store (pre-PE):
  `self.image_adapter_features = image_src` `(B,S,model_dim)` and
  `self.image_reference_grids = bundle.reference_grids` `(S,2)`. Non-image / DINOv3 paths set both
  `None`. `encode` is otherwise unchanged.
- **FR11** — `MixerModel.decode_align(**kwargs) -> dict` (reads `image_idx = kwargs["image_idx"]`,
  the batch's `(B,)` unique-image ids) returns:
  ```
  {"align":         self.align_head(self.image_adapter_features),      # (B, S, 2)
   "token_centers": self.image_reference_grids.unsqueeze(0),           # (1, S, 2)
   "image_centroids": self.align_centroids[image_idx],                 # (B, C_max, 2)
   "centroid_mask":   self.align_centroid_mask[image_idx]}             # (B, C_max) bool
  ```
  Raises `RuntimeError` if `image_adapter_features is None` (no image path) or the centroid buffers
  are unset (FR9).

### Forward routing & phase policy

- **FR12** — `forward` routing: `phase == "ImageAdaptation"` → `encode(**kwargs)` then
  `return decode_align(**kwargs)`; `phase == "FullFinetune"` → identical to the `"Combined"` branch.
  All existing branches unchanged. Scheduled sampling stays off in phase 1 (FR18).
- **FR13** — `set_phase` sets `requires_grad` explicitly for all three groups every call
  (`D`=denoise, `F`=fixation, `A`=adapter):

  | phase | D | F | A |
  |---|---|---|---|
  | `Denoise` | ✓ | ✗ | ✗ |
  | `Fixation` | ✗ | ✓ | ✗ |
  | `Combined` | ✓ | ✓ | ✗ |
  | `ImageAdaptation` | ✗ | ✗ | ✓ |
  | `FullFinetune` | ✓ | ✓ | ✓ |

  Empty `adapter_modules` (feature off) ⇒ the `A` column is a no-op and
  `Denoise`/`Fixation`/`Combined` reproduce today exactly (FR6).

### Loss & metric

- **FR14** — New `AlignmentLoss(coord_func=F.l1_loss)` in `src/model/loss_functions.py`.
  `forward(input, output) -> (loss, info)` computes, per token, the offset to its **nearest valid
  centroid** and regresses `output["align"]` toward it:
  - `pred = output["align"]` `(B,S,2)`; `centers = output["token_centers"]` `(1,S,2)`;
    `cents = output["image_centroids"]` `(B,C_max,2)`; `cmask = output["centroid_mask"]` `(B,C_max)`.
  - Under `torch.no_grad()`: pairwise dist `centers`↔`cents` `(B,S,C_max)`, `masked_fill(~cmask,
    inf)`, `argmin` over `C_max` → nearest centroid `(B,S,2)`; `target = nearest - centers`.
  - Rows whose image has **zero** centroids (`cmask.any(1)` False) are dropped; if no row has a
    centroid, return a differentiable zero (`pred.sum()*0`).
  - `loss = coord_func(pred[valid], target[valid])`; `info = {"align_loss": float(loss.item())}`.
  - **Does not read `tgt`** — the target is image-intrinsic (well-posedness fix).
- **FR15** — `CombinedLossFunction.__init__` gains `align_loss=None`; `forward` early-returns
  `self.align_loss(input, output)` **iff** `"align" in output`; byte-identical otherwise (FR6).
  `summary()` prints the align line when set.
- **FR16** — `eval_align(align_out, token_centers, image_centroids, centroid_mask) -> float` in
  `src/eval/eval_metrics.py`: mean `‖pred − target‖₂` over valid tokens/rows (same target as FR14),
  in normalized units.
- **FR17** — `MetricsStorage.metrics` gains `"align_error_val": []`;
  `compute_normalized_regression_metrics` accumulates `align_error` when `"align" in output`;
  `validate()` accumulates `align_coord_error_acum` when `"align" in output` and appends
  `align_error_val` only when it is `> 0` (mirrors the `denoise_error_val` guard). Logged when present.

### Builder & config wiring

- **FR18** — `PipelineBuilder`: `build_model` passes the FR4 kwargs from `model.image_adaptation.*`
  (read via `.get`); when the feature is enabled it loads the FR2 centroid cache (path from
  `model.image_adaptation.centroid_cache_path`), verifies the order invariant (FR3), and calls
  `model.set_alignment_centroids(...)`. A missing cache raises `FileNotFoundError` naming the
  precompute script. `build_loss_fn` (combined branch) passes
  `align_loss=AlignmentLoss(coord_func=STR_TO_LOSS_FUNC[config.loss.get("align_loss_type","l1")])`
  only when the feature is enabled, else `align_loss=None`. New experiment config
  `configs/exp/image_adaptation_training.yaml` (`# @package _global_`) sets
  `Phases: ["ImageAdaptation","Combined","FullFinetune"]`, the two new phase blocks
  (`ImageAdaptation` decisive metric `align_error_val`; `FullFinetune` decisive metric
  `reg_error_val`), `use_scheduled_sampling: false`, `model.image_adaptation.enabled: True` +
  `centroid_cache_path` + `align_head_hidden_dim`, `loss.align_loss_type: "l1"`, on top of
  `model/image_encoder: mask2former_precomputed` + `data.load.use_precomputed_features: True`.
  `build_phases` is unchanged (reads arbitrary named blocks; `denoise_weight` ignored by the align
  branch).

### Invariants

- **FR19** — Centroids, `bundle.reference_grids`, and `tgt` share one normalized `(x,y)` frame:
  centroids are normalized by the same `max_value` `Normalize` applies to `y`; reference grids are
  x-first `(x,y)` (F3); no axis flip is introduced. The precompute records `max_value`/`dest_res`
  in attrs and the loader is checked against the running `Normalize` config (warn on mismatch).
- **FR20** — With the feature off/absent, every `train.py` artifact is unchanged and the F6,
  precomputed-features, and hp-search default-config test suites stay green (no edit to
  `mixer_model.yaml` defaults; `model.image_adaptation` lives only in the new exp file, read via `.get`).
- **FR21 (leakage note)** — Centroids aggregate a stimulus's scanpaths across the **whole** dataset.
  Under `disjoint` / `stimuly_disjoint` splits (the main experimental splits) train and val/test
  stimuli are disjoint sets, so a per-image centroid is built from scanpaths of exactly one split —
  **no train↔eval leakage**. Under `random` split an image may straddle splits, giving mild target
  leakage; the precompute accepts an optional `--split-restrict train` (build each image's centroids
  only from its train-split scanpaths) to eliminate it. Default builds over all scanpaths; the spec
  documents the caveat.

## Public API Summary

```python
# scripts/build_scanpath_centroid_cache.py   (CLI: --out --bandwidth-dva --algorithm ...)

# src/data/scanpath_centroids.py  (or extend src/data/image_feature_cache.py)
class ScanpathCentroidCache:
    @staticmethod
    def build(data: CocoFreeView, bandwidth_dva=1.0, algorithm="meanshift",
              split_restrict=None, ...) -> "ScanpathCentroidCache": ...
    @staticmethod
    def write(path, centroids, centroid_mask, image_paths, attrs): ...
    def __init__(self, path, data: CocoFreeView): ...   # verifies image_path order (FR3)
    centroids: Tensor       # (U, C_max, 2) float32, [0,1]
    centroid_mask: Tensor   # (U, C_max) bool

# src/model/mixer_model.py
class MixerModel(nn.Module):
    def __init__(self, ..., image_adaptation=False,
                 align_head_hidden_dim=None, align_head_output_dropout=0.0, ...): ...
    def set_alignment_centroids(self, centroids: Tensor, mask: Tensor): ...   # non-persistent buffers
    def decode_align(self, **kwargs) -> dict:   # align, token_centers, image_centroids, centroid_mask
    def set_phase(self, phase): ...             # + "ImageAdaptation", "FullFinetune"
    def forward(self, skip_denoise=False, **kwargs): ...

# src/model/loss_functions.py
class AlignmentLoss(nn.Module):
    def __init__(self, coord_func=torch.nn.functional.l1_loss): ...
    def forward(self, input, output) -> tuple[Tensor, dict]: ...    # nearest-centroid target
class CombinedLossFunction(nn.Module):
    def __init__(self, denoise_loss, fixation_loss, denoise_weight=0, align_loss=None): ...

# src/eval/eval_metrics.py
def eval_align(align_out, token_centers, image_centroids, centroid_mask) -> float: ...
```

## Dependencies

| Reads from | Purpose |
|---|---|
| `CocoFreeView.get_scanpath / get_img_path / ptoa / dest_res` | offline: fixation clouds + DVA→px + normalizer |
| `sklearn.cluster.MeanShift` (or `DBSCAN`) | offline: spatial clustering of the aggregated cloud |
| `ScanpathCentroidCache` (`centroids`, `centroid_mask`) | per-image target centroids, keyed by unique order |
| `MultiScaleFeatures.value` / `.reference_grids` | trunk input + per-token anchors |
| `input["image_idx"]` (from `CoupledDataloader`) | gather per-image centroids for the batch |
| `config.model.image_adaptation.*` / `config.loss.align_loss_type` | enable + cache path + head + loss |

| Writes to | Purpose |
|---|---|
| `data/Coco FreeView/scanpath_centroids.h5` | the new centroid cache (additive, separate file) |
| `output["align"|"token_centers"|"image_centroids"|"centroid_mask"]` | loss/metric inputs |
| `MetricsStorage.metrics["align_error_val"]` | new validation metric; `align_loss_*` info |
| `model.pth` (active path only) | adds `align_head.*` params; centroid buffers are non-persistent |
| `configs/exp/image_adaptation_training.yaml` | the 3-phase driver experiment |

**Touches no frozen-feature HDF5 layout.** The centroid cache is a separate additive file; targets
are image-intrinsic centroids, independent of any single sample's `tgt`.
