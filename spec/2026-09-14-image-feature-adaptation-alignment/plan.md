# Plan — Image-Feature Adaptation via Scanpath-Centroid Alignment Pretraining

## Context and Design Decisions

**Why an alignment pretraining phase.** On the precomputed Mask2Former path, `img_input_proj`
(256→`model_dim`) is the *only* trainable image-side module, and its output is exactly the image
memory the deformable eye/fixation cross-attentions sample. If those features are spatially
misaligned with where fixations land, cross-attention must overcome that with limited signal. We
pre-shape `img_input_proj` with a Deformable-DETR-style per-token regression toward the nearest
attentional landmark of the stimulus, so the trunk encodes "how far, and in which direction, is a
likely fixation from here" *before* the encoder/decoder consume it — the image-side analogue of the
(abandoned) Denoise phase for the gaze encoder.

**Why an image-level, clustered target (the well-posedness fix — new).** Per-scanpath targets are
ill-posed: the same token gets a different nearest-fixation for every subject, so identical inputs
receive contradictory labels — subject noise masquerading as image signal. The target must be
**image-intrinsic**. We aggregate every scanpath's fixations for a stimulus into one point cloud and
**Mean-Shift cluster** it at **1-DVA bandwidth**; the cluster centroids are the stable attentional
landmarks each token regresses toward. Clustering both (a) makes the target independent of any single
scanpath and (b) de-noises the cloud — dozens of near-coincident fixations across subjects collapse
into one centroid rather than dragging the "nearest point" around. This is exactly the reviewer's
critique and its resolution.

**Why precompute + cache.** The centroid target depends only on the stimulus, so it is computed once
per unique image, offline, and cached — never recomputed per batch, never per scanpath. The cache is
keyed by image in the **same first-seen unique order** as the frozen-feature cache, so the batch's
existing `image_idx` (unique id) gathers the right centroids with a single index.

**DVA → pixels → normalized (grounded, not guessed).** `CocoFreeView.ptoa = 1/16` ("pixel to
angle") ⇒ **1 DVA = 16 px** in the `dest_res = (H,W) = (320,512)` downscaled space where
`get_scanpath(downscale=True)` returns fixations — the same 1-degree convention the dataset comment
cites (HAT σ=16). So Mean Shift runs in that px space at `bandwidth = bandwidth_dva / ptoa = 16 px`
(isotropic there), then centroids are normalized to `[0,1]` by the **same** `max_value = [W,H] =
[512,320]` that `Normalize(key='y', mode='coords')` applies — so centroids, `reference_grids`, and
`tgt` share one frame (FR19). (Normalizing per-axis makes a 1-DVA disk an ellipse in `[0,1]`; that is
correct and matches how `tgt` is normalized — we cluster in the isotropic px space precisely to avoid
clustering in the anisotropic normalized space.)

**Auxiliary-loss only / extend `img_input_proj` / precomputed-path-only (locked decisions).**
Reference points to `DeformableAttention` are unchanged (F1/F4 untouched); the align head sits on the
existing trunk output (already the cross-attn memory); the pixel decoder is cached+frozen so phase
3's "except the pixel decoder" is automatic.

**Additive / dual-path (constitution).** Everything is gated by `model.image_adaptation.enabled`
(default off) and `image_encoder_type`. Off ⇒ `img_input_proj` stays in `denoise_modules`, no
`align_head`, no centroid buffers, no new phases exercised, no new state_dict keys, `set_phase`
byte-identical; DINOv3 and old checkpoints untouched; `mixer_model.yaml` defaults unchanged (the
`test_default_configs_unmodified` tripwire stays green). Centroid buffers are **non-persistent** so
even the active model's state_dict gains only `align_head.*`.

**Single loss per run (constitution).** `pipeline.py` builds one `loss_fn`. Reuse the
`CombinedLossFunction` "dispatch by output keys" idiom: only the `ImageAdaptation` phase emits
`output["align"]`, so the loss early-returns the alignment term when present and is unchanged
otherwise.

**Scheduled-sampling caveat (unchanged from prior plan).** `use_scheduled_sampling` is a per-run
flag and `forward`'s sampler short-circuit is phase-agnostic; the `ImageAdaptation` phase must not be
routed through it. Default: `use_scheduled_sampling: false` for the whole run (matches
`denoise_pretraining.yaml`). Optional one-line follow-up: guard the short-circuit with `and
self.phase not in ("ImageAdaptation",)` to re-enable sampling in phases 2–3.

## Implementation Steps

### Step 1 — `scripts/build_scanpath_centroid_cache.py` + `ScanpathCentroidCache`

New loader module `src/data/scanpath_centroids.py` (keeps `image_feature_cache.py` focused; import
is small). `ScanpathCentroidCache`:

```python
class ScanpathCentroidCache:
    GROUP = "/centroids"

    @staticmethod
    def _first_seen_unique(data):           # identical to DeduplicatedMemoryDataset.build_index
        seen, order = {}, []
        for i in range(len(data)):
            p = data.get_img_path(i)
            if p not in seen: seen[p] = len(order); order.append(p)
        return order                          # list[str], row u == order[u]

    @staticmethod
    def build(data, bandwidth_dva=1.0, algorithm="meanshift",
              split_restrict=None, dbscan_eps_dva=None, min_samples=1):
        from sklearn.cluster import MeanShift, DBSCAN
        order = ScanpathCentroidCache._first_seen_unique(data)
        max_value = [data.dest_res[1], data.dest_res[0]]      # [W,H] = [512,320]
        bw_px = bandwidth_dva / data.ptoa                     # 16 px/DVA
        # group scanpath indices by img_path (respect split_restrict via data.df['split'])
        idx_by_path = {}                                       # path -> [row indices in data.df]
        for i in range(len(data)):
            if split_restrict and data.df.iloc[i]['split'] != split_restrict: continue
            idx_by_path.setdefault(data.get_img_path(i), []).append(i)
        per_image = []                                         # list of (Ci,2) np arrays, [0,1]
        for p in order:
            cloud = []                                         # (P,2) px in dest_res space
            for i in idx_by_path.get(p, []):
                x, y, _ = data.get_scanpath(i, downscale=True)
                cloud.append(np.stack([np.asarray(x), np.asarray(y)], axis=1))
            if not cloud:
                per_image.append(np.zeros((0,2), np.float32)); continue
            cloud = np.concatenate(cloud, 0)
            centers = _cluster(cloud, algorithm, bw_px, dbscan_eps_dva, data.ptoa, min_samples)
            centers = centers / np.asarray(max_value, np.float32)   # -> [0,1]
            per_image.append(centers.astype(np.float32))
        C_max = max((c.shape[0] for c in per_image), default=1)
        U = len(order)
        centroids = np.full((U, C_max, 2), np.nan, np.float32)
        mask = np.zeros((U, C_max), bool)
        for u, c in enumerate(per_image):
            centroids[u, :c.shape[0]] = c; mask[u, :c.shape[0]] = True
        attrs = dict(bandwidth_dva=bandwidth_dva, ptoa=data.ptoa,
                     dest_res=list(data.dest_res), max_value=max_value,
                     algorithm=algorithm, C_max=C_max, num_unique=U)
        return centroids, mask, order, attrs
```

- `_cluster`: `MeanShift(bandwidth=bw_px, bin_seeding=True).fit(cloud).cluster_centers_`; DBSCAN
  path clusters at `eps = (dbscan_eps_dva or bandwidth_dva)/ptoa`, drops `label==-1`, returns
  per-label means.
- `write(path, centroids, mask, image_paths, attrs)`: HDF5 group `/centroids`, mode `"w"`,
  datasets per FR2, attrs set.
- `__init__(path, data)`: reads arrays; rebuilds `_first_seen_unique(data)` and asserts equality with
  stored `image_path` (FR3, raises `ValueError` on any mismatch); exposes `self.centroids` (NaN→0 on
  masked slots) and `self.centroid_mask` as tensors.
- Driver `main()`: argparse per FR1 → `build` → `write`; prints `U`, `C_max`, mean centroids/image.

Add `scikit-learn` to `requirements.txt`.

### Step 2 — `src/eval/eval_metrics.py`: geometry helper + `eval_align`

Free function (single source of the target geometry; Step 3 imports it — `eval_metrics` has no model
dependency, so no cycle):

```python
def nearest_centroid_offsets(token_centers, image_centroids, centroid_mask):
    # token_centers (1,S,2) or (B,S,2); image_centroids (B,C,2); centroid_mask (B,C) bool
    with torch.no_grad():
        B = image_centroids.size(0)
        c = token_centers.expand(B, -1, -1) if token_centers.size(0) != B else token_centers
        d = torch.cdist(c, image_centroids)                        # (B,S,C)
        d = d.masked_fill(~centroid_mask.unsqueeze(1), float("inf"))
        nn_idx = d.argmin(dim=-1)                                  # (B,S)
        nearest = torch.gather(image_centroids, 1,
                               nn_idx.unsqueeze(-1).expand(-1, -1, 2))
        return nearest - c                                         # (B,S,2)

def eval_align(align_out, token_centers, image_centroids, centroid_mask):
    row = centroid_mask.any(dim=1)                                 # (B,)
    if not bool(row.any()): return 0.0
    target = nearest_centroid_offsets(token_centers, image_centroids, centroid_mask)
    r = row.view(-1,1,1).expand_as(align_out)
    return float((align_out[r] - target[r]).view(-1,2).norm(dim=-1).mean().item())
```

### Step 3 — `src/model/loss_functions.py`: `AlignmentLoss` + `CombinedLossFunction.align_loss`

```python
from src.eval.eval_metrics import nearest_centroid_offsets   # or lazy import inside forward

class AlignmentLoss(torch.nn.Module):
    def __init__(self, coord_func=torch.nn.functional.l1_loss):
        super().__init__(); self.coord_func = coord_func
    def summary(self):
        n = getattr(self.coord_func, "__name__", type(self.coord_func).__name__)
        print(f"AlignmentLoss: coord_func={n}")
    def forward(self, input, output):
        pred = output["align"]                       # (B,S,2)
        centers = output["token_centers"]            # (1,S,2)
        cents = output["image_centroids"]            # (B,C,2)
        cmask = output["centroid_mask"]              # (B,C) bool
        row = cmask.any(dim=1)                       # (B,)
        if not bool(row.any()):
            return pred.sum()*0.0, {"align_loss": 0.0}
        target = nearest_centroid_offsets(centers, cents, cmask)   # no_grad inside
        r = row.view(-1,1,1).expand_as(pred)
        loss = self.coord_func(pred[r], target[r])
        return loss, {"align_loss": float(loss.item())}
```

`CombinedLossFunction`: `__init__(..., align_loss=None)`; first line of `forward`:
`if "align" in output: return self.align_loss(input, output)`; `summary()` prints an align line when
set. Byte-identical when `output` has no `"align"` key (FR15).

*Import-cycle note:* `loss_functions.py` importing `eval_metrics.nearest_centroid_offsets` is safe
only if `eval_metrics` doesn't import `loss_functions` at module load. It currently imports
`create_cls_targets` etc. **from** `eval_metrics` **into** `training_utils`, not the reverse, and
`eval_metrics` imports nothing from `loss_functions`. If a cycle appears at wiring time, use a lazy
import of `nearest_centroid_offsets` inside `AlignmentLoss.forward`.

### Step 4 — `src/model/mixer_model.py`: constructor, buffers, encode

1. `__init__` args `image_adaptation=False, align_head_hidden_dim=None,
   align_head_output_dropout=0`; `self.image_adaptation = bool(image_adaptation)`.
2. Init `self.adapter_modules=[]`, `self.image_adapter_features=None`,
   `self.image_reference_grids=None` near the other module-list inits (~line 118).
3. In the FR8-guard block (~line 148): raise `ValueError` if `image_adaptation` and
   (`image_encoder is None` or `image_encoder_type != 'mask2former'`).
4. At `img_input_proj` creation (~line 242): append to `adapter_modules` (not `denoise_modules`) when
   `image_adaptation`; else current behavior.
5. After the head/`level_embed` blocks (~line 505), when `image_adaptation`: create `self.align_head`
   (MLP → 2), append to `adapter_modules`; and register **empty** non-persistent buffers so
   `.to(device)` and `decode_align`'s attribute access are well-defined before
   `set_alignment_centroids`:
   ```python
   self.register_buffer("align_centroids", torch.zeros(0, 0, 2, **factory_mode), persistent=False)
   self.register_buffer("align_centroid_mask", torch.zeros(0, 0, dtype=torch.bool, device=device),
                        persistent=False)
   self._centroids_ready = False
   ```
6. `encode`, Mask2Former branch, after `image_src = self.img_input_proj(bundle.value)`:
   `self.image_adapter_features = image_src`; `self.image_reference_grids = bundle.reference_grids`.
   DINOv3/non-image path: set both `None`.

### Step 5 — `src/model/mixer_model.py`: set_alignment_centroids, decode_align, set_phase, forward

```python
def set_alignment_centroids(self, centroids, mask):
    dev = self.factory_mode["device"]
    self.align_centroids = centroids.to(device=dev, dtype=self.factory_mode["dtype"])
    self.align_centroid_mask = mask.to(device=dev)
    self._centroids_ready = True

def decode_align(self, image_idx=None, **kwargs):
    if self.image_adapter_features is None:
        raise RuntimeError("decode_align requires the mask2former image path.")
    if not getattr(self, "_centroids_ready", False):
        raise RuntimeError("alignment centroids not set; call set_alignment_centroids().")
    image_idx = image_idx.to(self.align_centroids.device).long()
    offsets = self.align_head(self.image_adapter_features)          # (B,S,2)
    return {"align": offsets,
            "token_centers": self.image_reference_grids.unsqueeze(0),
            "image_centroids": self.align_centroids[image_idx],     # (B,C,2)
            "centroid_mask": self.align_centroid_mask[image_idx]}   # (B,C)
```
(`image_idx` arrives in `**kwargs` from the batch — `CoupledDataloader` sets
`et_data_batch['image_idx']`. Pull it as a named kwarg for clarity.)

`set_phase` — explicit three-group table (FR13), via a `_set_group(modules, flag)` helper; empty
`adapter_modules` ⇒ `A` no-op ⇒ off-path parity.

`forward` — add before the trailing `return self.decode_fixation`:
```python
elif self.phase == 'ImageAdaptation':
    return self.decode_align(**kwargs)
elif self.phase == 'FullFinetune':
    denoise_output = {} if skip_denoise else self.decode_denoise(**kwargs)
    return {**denoise_output, **self.decode_fixation(**kwargs)}
```
`encode(**kwargs)` already runs above (when `pass_sampler` falsy), populating
`image_adapter_features` before `decode_align`.

### Step 6 — `src/training/training_utils.py`: metric plumbing

1. `MetricsStorage.__init__`: add `'align_error_val': []`.
2. Import `eval_align` (and it internally reuses `nearest_centroid_offsets`).
3. `compute_normalized_regression_metrics`: after the `denoise` block,
   `if 'align' in output: results_dict['align_error'] = eval_align(output['align'],
   output['token_centers'], output['image_centroids'], output['centroid_mask'])`.
4. `validate`: `align_coord_error_acum = 0`; compute `eval_align(...)` right after `loss_fn(...)`
   (before `invert_transforms`, which never touches the align keys, so order is immaterial);
   after the loop append `metrics['align_error_val']` when `align_coord_error_acum > 0`; print under
   `if log:`.

### Step 7 — `src/training/pipeline_builder.py`: build_model + centroid load + build_loss_fn

- Import `AlignmentLoss` and `ScanpathCentroidCache`.
- `build_model`: read `ia = self.config.model.get('image_adaptation', {}) or {}`; pass
  `image_adaptation=bool(ia.get('enabled', False))`, `align_head_hidden_dim=ia.get(
  'align_head_hidden_dim')`, `align_head_output_dropout=ia.get('align_head_output_dropout', 0)` to
  `MixerModel(...)`. When enabled, after model construction (and after `self.load_dataset()` has run
  — it runs before `build_model` in `pipeline.py`, and the `CocoFreeView` metadata is available; the
  precomputed path already builds `CocoFreeView` for splits) load the centroid cache:
  ```python
  if ia.get('enabled', False):
      cache = ScanpathCentroidCache(ia['centroid_cache_path'], self._coco_freeview())
      model.set_alignment_centroids(cache.centroids, cache.centroid_mask)
  ```
  `_coco_freeview()` = the same `CocoFreeView` the dataset/splits use (reuse the existing handle;
  the precomputed-features path already constructs one — thread it through, do not re-parse if a
  handle exists). Missing file → `FileNotFoundError` naming `build_scanpath_centroid_cache.py`.
- `build_loss_fn`, `combined` branch: `align_loss = AlignmentLoss(coord_func=STR_TO_LOSS_FUNC[
  self.config.loss.get('align_loss_type','l1')]) if ia_enabled else None`, passed to
  `CombinedLossFunction(..., align_loss=align_loss)`.

### Step 8 — configs

`configs/exp/image_adaptation_training.yaml` (`# @package _global_`):
```yaml
# @package _global_
defaults:
  - override /model/image_encoder: mask2former_precomputed

training:
  decisive_metric: "reg_error_val"
  use_scheduled_sampling: false
  Phases: ["ImageAdaptation", "Combined", "FullFinetune"]
  ImageAdaptation: {name: "ImageAdaptation", denoise_weight: 0, decisive_metric: "align_error_val", epochs: 40}
  Combined:        {name: "Combined",        denoise_weight: 0, decisive_metric: ${training.decisive_metric}, epochs: 80}
  FullFinetune:    {name: "FullFinetune",    denoise_weight: 0, decisive_metric: ${training.decisive_metric}, epochs: 120}

model:
  pretrained_encoder_path: null
  add_denoise_head: False
  image_adaptation:
    enabled: True
    centroid_cache_path: "data/Coco FreeView/scanpath_centroids.h5"
    align_head_hidden_dim: [256, 128]
    align_head_output_dropout: 0

loss:
  complex_type: "combined"
  fixation_loss_type: "separated_reg"
  align_loss_type: "l1"

data:
  load:
    use_precomputed_features: True
```
Verify the `override /model/image_encoder` group-path syntax against how other exp files swap the
encoder in this repo's Hydra version; if none do, drop the `defaults:` block and pass
`model/image_encoder=mask2former_precomputed` on the CLI. **Do not** edit `mixer_model.yaml`.

### Step 9 — tests: `tests/test_image_adaptation.py`

CPU-only, no network, no real caches. Build `MixerModel` with a `PrecomputedFeatureAdapter`
(`embed_dim=256`, tiny `spatial_shapes`, `S` small) as `image_encoder`; feed synthetic centroid
buffers via `set_alignment_centroids`; a synthetic `CocoFreeView`-like stub for the cache/order test.
Cover FR1–FR21 per validation.md.

## Implementation Order

1. **Step 1** — precompute script + `ScanpathCentroidCache` (+ `scikit-learn` dep).
2. **Step 2** — `nearest_centroid_offsets` + `eval_align` (owns the geometry).
3. **Step 3** — `AlignmentLoss` + `CombinedLossFunction.align_loss`.
4. **Step 4** — `MixerModel` constructor / buffers / `encode`.
5. **Step 5** — `set_alignment_centroids` / `decode_align` / `set_phase` / `forward`.
6. **Step 6** — `training_utils` metric plumbing.
7. **Step 7** — `PipelineBuilder` (build_model + centroid load + build_loss_fn).
8. **Step 8** — configs.
9. **Step 9** — tests.

(Step 2 precedes 3 and 6; Step 1 precedes 7; Steps 4–5 precede 7; Step 8 precedes end-to-end tests.)
