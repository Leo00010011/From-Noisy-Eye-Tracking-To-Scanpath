# Validation — Image-Feature Adaptation via Scanpath-Centroid Alignment Pretraining

Tests are CPU-only, no network, no real caches. Build `MixerModel` with a `PrecomputedFeatureAdapter`
(`embed_dim=256`, `num_levels=3`, tiny `spatial_shapes=[[2,2],[3,3],[4,4]]` → `S=29`) as
`image_encoder` (so `image_encoder_type="mask2former"`), and install synthetic centroid buffers via
`set_alignment_centroids`. Use a small `CocoFreeView`-like stub (fixed `ptoa=1/16`,
`dest_res=(320,512)`, a handful of scanpaths over a few images) for the cache/clustering tests.

## Code Correctness

### Group 1 — Centroid precompute & clustering (`ScanpathCentroidCache.build`)

- [ ] **1.1 DVA→px bandwidth** — with `ptoa=1/16`, `bandwidth_dva=1.0` → Mean Shift bandwidth is
  `16.0` px (assert the value passed to `MeanShift`). `bandwidth_dva=2.0` → `32.0`.
- [ ] **1.2 clustering collapses duplicates** — a cloud of 50 points tightly around two locations
  ~10 DVA apart yields exactly 2 centroids, each within ~1 px of the true mean (`atol` a few px).
- [ ] **1.3 normalization to [0,1]** — centroids are divided by `max_value=[512,320]`; a centroid at
  px `(256,160)` stores as `(0.5,0.5)` (`atol=1e-6`). Confirms the x/W, y/H per-axis convention (FR19).
- [ ] **1.4 aggregation across scanpaths** — an image with 3 scanpaths contributes all three clouds
  to one clustering call (not per-scanpath); a point present in only one scanpath still influences a
  centroid. (Assert the input cloud size == sum of the 3 scanpath lengths.)
- [ ] **1.5 empty image** — a unique image with zero scanpaths (only possible via `split_restrict`)
  yields `n_centroids=0` for that row; `centroid_mask[u]` all-False; no crash.
- [ ] **1.6 padding & C_max** — `centroids` is `(U,C_max,2)` with `C_max = max n_centroids`; padded
  slots are NaN in the file and `centroid_mask` False; per-row `mask.sum() == n_centroids[u]`.
- [ ] **1.7 DBSCAN alternative** — `algorithm="dbscan"` returns per-label means and drops
  `label==-1` noise; on 1.2's cloud it also recovers 2 centroids.
- [ ] **1.8 split_restrict** — with `split_restrict="train"`, only train-split scanpaths of each
  image enter its cloud (a val-only image → 0 centroids). Confirms FR21's leak-free mode.

### Group 2 — Cache write / read / order invariant

- [ ] **2.1 roundtrip** — `write` then `__init__` returns `centroids`/`centroid_mask` equal to the
  built arrays (NaN slots read back as 0, mask False); attrs (`bandwidth_dva`, `ptoa`, `dest_res`,
  `max_value`, `algorithm`, `C_max`) preserved.
- [ ] **2.2 order verification passes** — `__init__(path, data)` with the same stub `data` used to
  build succeeds and `image_path[u]` == `_first_seen_unique(data)[u]` for all `u`.
- [ ] **2.3 order verification not bypassable** — permute two rows of `image_path` in the file →
  `__init__` raises `ValueError` (FR3). Reordering the stub's samples so first-seen order differs
  also raises.
- [ ] **2.4 masked NaN handling** — a file with NaN pad slots loads with those `centroids` entries
  finite (0) and mask False, so downstream `cdist` never sees NaN.

### Group 3 — target geometry (`nearest_centroid_offsets`) & `AlignmentLoss`

- [ ] **3.1 offset sign/value** — `token_centers=[[0.5,0.5]]` (S=1), one centroid `[0.8,0.2]`, mask
  True → target `[0.3,-0.3]` (`atol=1e-6`).
- [ ] **3.2 nearest selection** — 2 centroids `[[0.9,0.9],[0.55,0.45]]` both valid → target uses the
  nearer `[0.55,0.45]`.
- [ ] **3.3 mask excludes padded centroids** — 3.2 with `centroid_mask=[True,False]` → target uses
  `[0.9,0.9]` (the far one), proving `masked_fill(inf)` before `argmin`.
- [ ] **3.4 zero loss at perfect prediction** — `pred = target` → `AlignmentLoss` returns 0
  (`atol=1e-6`), `info["align_loss"]==0`.
- [ ] **3.5 row with no centroids dropped** — batch row 0 all-False mask, row 1 one centroid → loss
  == loss on row 1 alone; no NaN.
- [ ] **3.6 all-empty batch → safe zero** — every mask False → differentiable zero; `.backward()`
  raises nothing.
- [ ] **3.7 target no_grad** — `nearest_centroid_offsets` output `.requires_grad is False`; gradient
  flows only through `pred`.
- [ ] **3.8 does NOT read tgt** — `AlignmentLoss.forward` computes the same loss when `input["tgt"]`
  is replaced by garbage/removed (target comes from `output["image_centroids"]`). Proves the
  well-posedness fix — the target is image-intrinsic, not per-scanpath.
- [ ] **3.9 coord_func swap** — `mse_loss` vs `l1_loss` give different values on a non-trivial pred.

### Group 4 — CombinedLossFunction dispatch & eval_align

- [ ] **4.1 align branch early-return** — `output` with `"align"` → returns exactly
  `align_loss(input, output)`; `info` keys `{"align_loss"}` only.
- [ ] **4.2 no-align byte-identity** — `output` **without** `"align"` (normal Combined output) yields
  identical loss/info to a pre-change `CombinedLossFunction`, for both `align_loss=None` and set (FR15).
- [ ] **4.3 eval_align matches loss geometry** — hand-check equals `‖pred-target‖₂` on a
  1-token/1-centroid case; `pred==target` → 0.0; all-empty mask → 0.0 (so `validate`'s `>0` guard
  never appends spuriously).

### Group 5 — MixerModel construction & gating

- [ ] **5.1 off by default** — no `image_adaptation`: `img_input_proj` in `denoise_modules`,
  `adapter_modules==[]`, no `align_head`, no `align_centroids` buffer, `"align_head" not in`
  named_parameters.
- [ ] **5.2 on-path build** — `image_adaptation=True`: `img_input_proj` **not** in `denoise_modules`,
  `img_input_proj` and `align_head` in `adapter_modules`, `align_head` out dim 2; empty
  `align_centroids`/`align_centroid_mask` non-persistent buffers exist, `_centroids_ready is False`.
- [ ] **5.3 guard wrong encoder** — `image_adaptation=True` + DINOv3 stub → `ValueError` mentioning
  `mask2former`.
- [ ] **5.4 guard no encoder** — `image_adaptation=True, image_encoder=None` → `ValueError`.
- [ ] **5.5 state_dict** — off-model keys == pre-change keys exactly; on-model adds **only**
  `align_head.*` (centroid buffers are non-persistent ⇒ absent from state_dict). FR6/FR9/FR20.
- [ ] **5.6 set_alignment_centroids** — after the call, buffers hold the given tensors on the model
  device/dtype, `_centroids_ready is True`; index tensors (mask) stay bool.

### Group 6 — set_phase requires_grad matrix (FR13)

Probes: D→a `path_encoder` weight; F→a `coord_head` weight; A→an `align_head` weight (+ `img_input_proj`).

- [ ] **6.1 Denoise** (T,F,F) · **6.2 Fixation** (F,T,F) · **6.3 Combined** (T,T,F, adapter frozen —
  the "encoder+decoder, adapter frozen" stage) · **6.4 ImageAdaptation** (F,F,T) · **6.5 FullFinetune** (T,T,T).
- [ ] **6.6 off-path parity** — on an off-path model, `set_phase("Denoise"/"Fixation"/"Combined")`
  matches the pre-change layout exactly (adapter no-op). FR6/FR20.

### Group 7 — forward routing, shapes, gradient isolation

- [ ] **7.1 ImageAdaptation forward** — output keys exactly
  `{"align","token_centers","image_centroids","centroid_mask"}`; shapes `(B,S,2)`,`(1,S,2)`,
  `(B,C_max,2)`,`(B,C_max)`.
- [ ] **7.2 decode_align errors** — `image_adapter_features=None` → `RuntimeError`; centroids unset
  (`_centroids_ready False`) → `RuntimeError`.
- [ ] **7.3 FullFinetune forward** — Combined-phase keys (`coord`/`dur`/`cls`, `denoise` iff head &
  not `skip_denoise`); **no** `"align"`.
- [ ] **7.4 Combined has no align** — `"align" not in output` (so phases 2–3 never take the align
  loss branch).
- [ ] **7.5 encode stores trunk + grids** — `image_adapter_features.shape==(B,S,model_dim)`,
  `image_reference_grids.shape==(S,2)`, grids in `[0,1]`.
- [ ] **7.6 centroid gather by image_idx** — with distinct per-row centroid buffers, an
  `ImageAdaptation` forward on a batch with a known `image_idx` returns `image_centroids` equal to
  `align_centroids[image_idx]` (assert exact gather; catches an off-by-one vs row position).
- [ ] **7.7 gradient isolation** — one `ImageAdaptation` step + `backward()`: `align_head` and
  `img_input_proj` grads non-None; a `path_encoder` and a `coord_head` param `grad is None`.

### Group 8 — end-to-end train smoke (tiny synthetic)

- [ ] **8.1 three-phase run** — minimal `PipelineBuilder`-driven run (few samples, synthetic
  centroid cache, `epochs=1`/phase, `Phases=["ImageAdaptation","Combined","FullFinetune"]`) completes;
  `metrics["align_error_val"]` non-empty after phase 1 validates, `metrics["reg_error_val"]`
  non-empty after phases 2–3.
- [ ] **8.2 align loss decreases** — ~50 `ImageAdaptation` steps on a fixed tiny batch → final
  `align_loss < 0.9×` first-step (trunk+head can fit centroid offsets).
- [ ] **8.3 checkpoint round-trip** — save after phase 1, reload into a fresh on-path model → zero
  missing/unexpected keys, `align_head` weights match; reload into an **off-path** model →
  `align_head.*` land in `unexpected` (documents the dual-path boundary). Centroid buffers are not in
  the checkpoint (non-persistent).

## Data Validity

Run on real CocoFreeView + a real centroid cache once the code lands (notebook cells). Each states an
expected outcome.

- [ ] **DV1 centroid counts** — mean centroids/image is a small, plausible number of attentional
  landmarks (expected ~3–12 for natural COCO scenes at 1-DVA bandwidth), always ≥1 for images with
  scanpaths, and far below the raw fixation count (clustering actually collapses the cloud). Report
  the distribution of `n_centroids`.
- [ ] **DV2 well-posedness gain (headline for this revision)** — quantify target noise **reduced** by
  going image-level + clustered: for a sample of images, compare the spread of per-scanpath nearest
  fixations (old target) at a fixed token vs the single clustered-centroid target (new). Expect the
  new target's per-image variance ≈ 0 (deterministic) vs the old target's non-trivial cross-subject
  variance. This is the concrete evidence the reviewer's critique is resolved.
- [ ] **DV3 bandwidth sanity** — halving/doubling `bandwidth_dva` monotonically increases/decreases
  the centroid count; at 1 DVA the centroids visually land on distinct objects/regions when overlaid
  on a few stimuli (qualitative plot).
- [ ] **DV4 frame consistency** — for a stimulus, overlay stored centroids and that image's raw
  `tgt` fixations (both in `[0,1]`, inverted to px via `Normalize.inverse` for display): centroids sit
  among the fixation clusters, no systematic x/y swap or offset (guards FR19 end-to-end).
- [ ] **DV5 alignment beats center-anchor baseline** — end-of-`ImageAdaptation` `align_error_val` is
  clearly below the zero-offset baseline (predicting the token center), i.e. below the mean
  `‖nearest-centroid − center‖`. The phase's own success criterion.
- [ ] **DV6 downstream effect (research question)** — same split & seed, compare final
  `reg_error_val` of `["Combined","FullFinetune"]` (no alignment) vs
  `["ImageAdaptation","Combined","FullFinetune"]`. Report whether centroid-alignment pretraining
  lowers `reg_error_val`. A null result is a valid reported outcome, not a test failure.
- [ ] **DV7 image-reliance cross-check** — run `src/eval/image_reliance.py` on the alignment-pretrained
  vs baseline checkpoint; expect the image/gaze residual ratio and/or image-shuffle perturbation delta
  to rise if adapted features are more used. Diagnostic.

## Data Architecture Integrity

- [ ] **AI1 no frozen-feature-cache change** — `git diff` touches no `.h5` layout in
  `image_feature_cache.py` / `build_image_feature_cache.py`; the centroid cache is a **separate**
  additive file (`scanpath_centroids.h5`).
- [ ] **AI2 keying invariant shared with feature cache** — the centroid cache's `image_path` order is
  byte-identical to `PrecomputedFeatureDataset`'s first-seen unique order, so `image_idx` gathers the
  correct centroids **and** the correct features for the same row (assert both caches agree on
  `image_path[u]` for all `u` on the real data). Order verification is unconditional (AI/2.3).
- [ ] **AI3 reference-grid provenance** — `token_centers` in `decode_align` are exactly
  `bundle.reference_grids` from the `PrecomputedFeatureAdapter` (same grid the deformable memory
  uses), not a re-derived copy.
- [ ] **AI4 dual-path snapshot invariant** — `mixer_model.yaml` unmodified
  (`test_default_configs_unmodified` green); `model.image_adaptation` only in the new exp file, read
  via `.get`; a pre-existing run's config snapshot (no `image_adaptation` key) builds byte-identically.
- [ ] **AI5 off-path artifact parity** — a `train.py` run on `exp=whole_model_pretraining` (feature
  off) yields the identical state_dict keys, `set_phase` behavior, and per-phase output keys as before;
  the F6 (`tests/test_f6_integration.py`) and precomputed-features
  (`tests/test_image_feature_cache.py`) suites stay green.
- [ ] **AI6 leakage documentation holds** — under `stimuly_disjoint`/`disjoint`, each image's
  centroids are built from one split's scanpaths only (train and eval stimuli are disjoint); confirm
  no eval stimulus's centroids draw on train scanpaths (and vice-versa) for those strategies. Under
  `random`, `--split-restrict train` produces centroids with zero eval-scanpath contribution (FR21).
