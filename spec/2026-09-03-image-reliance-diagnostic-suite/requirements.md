# Requirements — Image-Reliance Diagnostic Suite

## Goal

The `MixerModel` trained with the new Mask2Former multiscale backbone (`train_ms.sh`,
`exp=only_combined model/image_encoder=mask2former`) reaches essentially the same validation
accuracy (≈50 reg / ≈86 dur) as prior DINOv3 runs. The leading hypothesis is that the model is
**ignoring the image backbone entirely** and predicting fixations from the gaze signal alone. This
feature delivers a small, additive diagnostic suite that answers *"does the model use the image?"*
with three converging measurements, run over two evaluation passes on the held-out **test** split
of CocoFreeView (same simulated-noise pipeline the model was trained on):

1. **Cross-attention residual magnitudes** — how much each decoder's *image* cross-attention adds to
   the residual stream, relative to the *gaze* cross-attention (fixation decoder) or the running
   stream (eye decoder). `image/gaze << 1` ⇒ image ignored.
2. **Input-perturbation test** — re-run the forward with the images **shuffled within each batch**
   (gaze untouched). If the regression error barely changes, the image provably carries no signal.
3. **Sampling-coordinates-in-range test** — the fraction of deformable sampling locations that land
   inside the `[0,1]` feature map. A low fraction means the deformable attention samples off-map,
   `grid_sample` returns zero-padded vectors, and attention weight is wasted on empty values — a
   mechanism that would *cause* the image to be effectively ignored.

The suite is **diagnosis only**: it records and reports, it does not modify the model, retrain, or
fix any defect it finds.

## Scope

**In scope**
- A reusable, unit-testable module `src/eval/image_reliance.py` holding the pure measurement
  functions (residual-norm extraction, in-range fraction, image shuffle, per-sample regression
  error) and the two pass drivers.
- A thin driver script `src/notebooks/save_image_reliance.py` (config block → runs both passes →
  writes HDF5 + JSON), modelled on the existing `src/notebooks/save_activations_eve_real.py`.
- **Pass A** (one recorder-on forward per batch): cross-attention residual norms **and** deformable
  sampling-in-range fractions for the **eye decoder** and the **fixation decoder**.
- **Pass B** (recorder-off): the input-perturbation test.
- Persistence: a per-sample HDF5 store (`outputs/image_reliance/{run_name}_reliance.h5`) **plus** an
  aggregated summary JSON + printed table (`outputs/image_reliance/{run_name}_summary.json`).
- Runs on the CocoFreeView **test** split via `model_io.load_pipeline` / `load_test_data` /
  `load_model` (rebuilding the run's saved `split.pth`).
- Target sequence for the Pass A recording forward = the **model's own autoregressive predictions**
  (predict with recorder off, then one clean causal forward feeding predictions back as `tgt` with
  recorder on) — matching inference-time behaviour and the `save_activations_eve_real.py` approach.

**Out of scope**
- Any change to `src/model/blocks.py`, `mixer_model.py`, or the recording hooks — all needed
  tensors (`first_cross_res`, `second_cross_res`, `cross_attention_res`, `sampling_locations`) are
  **already** recorded by the existing `norm_first`+deformable hooks.
- A DINOv3 A/B comparison run (single Mask2Former checkpoint only; the residual ratio, in-range %,
  and perturbation delta are each self-interpretable).
- Recording the pixel decoder's internal 6-layer MSDeformAttn self-attention (its reference points
  are grid centres, in-range by construction). Only the eye/fixation decoders — whose queries are
  gaze/fixation coords and can plausibly sample off-map — are covered.
- Fixing any defect the suite reveals (e.g. off-map sampling, dead image path).
- The EVE real-noise path (that already has its own `save_activations_eve_real.py`).

## Functional Requirements

### Data loading

- **FR1** The driver loads a checkpoint directory `ckpt_path` via
  `model_io.load_pipeline(ckpt_path)`, `pipe.load_dataset()`, and
  `load_test_data(pipe, ckpt_path, return_dataloaders=True)`, using **only the returned `test`
  dataloader**. Weights load via `model_io.load_model(pipe, ckpt_path)` (which strips the
  `_orig_mod.` prefix and loads with `strict=False`). The model is put in `set_phase("Fixation")`,
  moved to the device, and `.eval()`.
- **FR2** The test dataloader is a `CoupledDataloader` yielding matched `(gaze_batch, image_batch)`
  with `shuffle=False` so per-sample records align to a stable `sample_idx`. If the loaded config
  has `image_encoder.type != "mask2former"`, the driver prints a warning but still runs (the suite
  is backbone-agnostic; only the in-range test needs `n_levels`).

### Recording-support probe

- **FR3** A `probe_recording_support(model) -> dict` (ported from `save_activations_eve_real.py`)
  reports `norm_first`, `use_deformable_fixation_decoder`, `use_deformable_eye_decoder`, `n_decoder`,
  `n_eye_decoder`, and derived `fix_ok` / `eye_ok` booleans. Residuals **and** sampling locations are
  recorded only inside the `norm_first=True` branch of the *deformable* decoder blocks; when a stream
  is unavailable the corresponding datasets are skipped and an attr flag is set `False` (no crash).

### Pass A — residuals + sampling-in-range (one recorder-on forward per batch)

- **FR4** For each test batch: (a) run `eval_autoregressive(model, inp, only_last=True)` with the
  recorder inactive to obtain the predicted scanpath `pred_reg_norm (B, K, 3)` (normalised);
  (b) start the recorder (`recorder.start_batch(...)`), set `inp2 = {**inp, tgt: pred_reg_norm,
  tgt_mask: None, in_tgt: None}`, and run one clean `model.encode(**inp2)` + `model.decode_fixation(**inp2)`.
  `K1 = K + 1` (start token) is the fixed fixation-decoder query length.
- **FR5** Residual norms are extracted from the recorder payload
  (`recorder.current_payload["activations"]`) per layer:
  - Fixation decoder, module prefix `decoder`, keys
    `("self_attention_res", "first_cross_res", "second_cross_res", "ffn_res")` → each a
    `(n_decoder, K1)` float32 array of per-decode-step L2 norms over the feature dim. **Legend:
    `first_cross_res` = gaze contribution, `second_cross_res` = image contribution.**
  - Eye decoder, module prefix `eye_decoder`, keys
    `("self_attention_res", "cross_attention_res", "ffn_res")` → each `(n_eye_decoder, T)` where
    `T = valid gaze length` (from `src_mask`). **Legend: `cross_attention_res` = image contribution.**
- **FR6** Sampling-in-range fractions are extracted from the deformable sub-modules'
  `sampling_locations` tensor `(B, Nq, n_heads, n_levels, n_points, 2)` (recorded in `[0,1]` space):
  - Fixation decoder deformable op: module name `decoder.{l}.second_cross_attn`.
  - Eye decoder deformable op: module name `eye_decoder.{l}.cross_attn`.
  A location is **in-range** iff `0 <= x <= 1 AND 0 <= y <= 1` (both last-dim coords). For each layer
  `l` and level `L`, `in_range_fraction` = mean of the boolean over the query, head, and point axes
  (excluding padded query positions via `src_mask` for the eye decoder; all `K1` positions kept for
  the fixation decoder). Result: `dec_inrange (n_decoder, n_levels)` and
  `eye_inrange (n_eye_decoder, n_levels)` per sample, float32 in `[0,1]`.
- **FR7** Optional (`SAVE_FULL_RESIDUALS=True`, default on): also retain the full fixation-decoder
  cross-attention residual tensors `dec_first_cross_res` / `dec_second_cross_res` shaped
  `(N, n_decoder, K1, D)` as float16, for offline per-dimension inspection.
- **FR8** Pass A writes per-sample records keyed by `sample_idx` (the CocoFreeView dataset index from
  `inp["sample_idx"]`), plus `stimulus_name` when the image dataset exposes it (else empty string),
  `pred_len` (first decode step with `sigmoid(eos) > 0.5`, else `K`), and `src_len`.

### Pass B — input-perturbation (recorder off)

- **FR9** For each test batch, compute the per-sample **normalised** regression error twice with an
  identical model/state: once with the true image (`reg_error_clean`), once with the image tensor
  **shuffled within the batch** (`reg_error_shuffled`). Gaze input, masks, and `tgt` are identical
  between the two runs; only the image batch is permuted.
- **FR10** The per-batch image permutation is a **guaranteed derangement**: a cyclic roll by offset 1
  along the batch axis (no sample keeps its own image). For a trailing batch of size 1 the shuffle is
  impossible → that sample's `reg_error_shuffled` and `perm_index` are written as `NaN` / `-1` and a
  warning is printed. `perm_index[i]` records the source sample index whose image sample `i` received.
- **FR11** Per-sample regression error = mean Euclidean distance in normalised `[0,1]` space between
  predicted and ground-truth fixation coords over **valid** target positions (`tgt_mask`), computed
  consistently with `eval_metrics.eval_reg` (but reduced per sample, not per batch). `dur_error_clean`
  / `dur_error_shuffled` = mean absolute normalised duration error over valid positions. Predictions
  use `eval_autoregressive(model, inp, only_last=True)` so the perturbation propagates through the
  full autoregressive rollout.

### Persistence

- **FR12** Per-sample HDF5 at `outputs/image_reliance/{run_name}_reliance.h5`, mode `"w"`, single
  group `/reliance`. Layout in the *Public API Summary → HDF5 layout* block below. Both passes write
  into the same file keyed by `sample_idx` (Pass A first, Pass B updates/extends); the two passes
  iterate the same `shuffle=False` loader so row `i` is the same sample in both.
- **FR13** Aggregated summary JSON at `outputs/image_reliance/{run_name}_summary.json` and a printed
  table containing, at minimum:
  - **Residuals:** per fixation-decoder layer, mean over (samples × decode-steps) of
    `||second_cross_res|| / ||first_cross_res||` (image/gaze ratio) and the two absolute means;
    per eye-decoder layer, mean `||cross_attention_res||` and its ratio to mean `||self_attention_res||`.
  - **Perturbation:** mean `reg_error_clean`, mean `reg_error_shuffled`, absolute and relative delta,
    and the fraction of samples whose error changed by less than `eps_ignore` (default `1e-3` in
    normalised units).
  - **In-range:** per level, mean in-range fraction across layers and samples, separately for the eye
    and fixation decoders.
  Each block carries a one-line interpretation string (e.g. `"image/gaze ratio 0.02 << 1 ⇒ image
  contribution negligible"`).
- **FR14** Group attrs on `/reliance`: `run_name`, `checkpoint_path`, `img_size`,
  `image_encoder_type`, `n_image_levels`, `spatial_shapes` (flattened), `level_start_index`, `K1`,
  `model_dim`, `n_decoder`, `n_eye_decoder`, `target_mode` (`"pred"`), `split` (`"test"`),
  `eps_ignore`, `created_at`, `fix_residuals_saved`, `eye_residuals_saved`, `full_residuals_saved`,
  `inrange_saved`, and the decode legend (`fix_first_cross="gaze"`, `fix_second_cross="image"`,
  `eye_cross="image"`).

### Error conditions

- **FR15** `load_test_data` raises if `split.pth` is absent in `ckpt_path` (existing behaviour,
  surfaced unchanged).
- **FR16** `sampling_in_range_fraction` raises `ValueError` if the input tensor's last dim ≠ 2 or its
  `n_levels` axis ≠ the model's `n_image_levels`.
- **FR17** If both `fix_ok` and `eye_ok` are `False`, Pass A prints a warning and writes only the
  prediction/length metadata (no residual or in-range datasets); Pass B still runs.
- **FR18** A batch size of 1 across the *entire* loader makes Pass B degenerate; the driver asserts
  `batch_size >= 2` for Pass B (or warns and writes all-`NaN` perturbation columns).

## Public API Summary

```python
# src/eval/image_reliance.py

def probe_recording_support(model) -> dict: ...
    # {norm_first, fix_deform, eye_deform, n_decoder, n_eye_decoder, fix_ok, eye_ok}

def extract_residuals(recorder, module_prefix: str, n_layers: int,
                      value_names: tuple[str, ...]) -> dict[str, list[torch.Tensor]]: ...
    # {value_name: [layer0_tensor, layer1_tensor, ...]}, each (Nq, D) for one sample-batch pass

def residual_norms(res: dict[str, list[torch.Tensor]], sample_i: int,
                   value_names: tuple[str, ...]) -> dict[str, np.ndarray]: ...
    # {value_name: (n_layers, Nq)} L2 over feature dim for sample sample_i

def extract_sampling_locations(recorder, module_prefix: str, attn_attr: str,
                               n_layers: int) -> list[torch.Tensor]: ...
    # [layer0, ...], each (B, Nq, H, L, P, 2) in [0,1]; attn_attr in {"second_cross_attn","cross_attn"}

def sampling_in_range_fraction(sampling_locations: torch.Tensor,
                               query_mask: torch.Tensor | None = None) -> torch.Tensor: ...
    # (B, n_levels) mean over (valid Nq, H, P) of [0<=x<=1 & 0<=y<=1]; raises ValueError per FR16

def shuffle_images_in_batch(image_tensor: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor]: ...
    # cyclic roll by 1 along dim 0; returns (permuted, perm_index int64 (B,)); B>=2

def per_sample_reg_error(pred_reg: torch.Tensor, tgt: torch.Tensor,
                         tgt_mask: torch.Tensor | None) -> tuple[np.ndarray, np.ndarray]: ...
    # (coord_err (B,), dur_err (B,)) normalised, mean over valid target positions

def run_recording_pass(model, dataloader, device, recorder, *,
                       save_full_residuals: bool = True) -> tuple[list[dict], dict]: ...
    # Pass A -> (per-sample records, support dict)

def run_perturbation_pass(model, dataloader, device, *,
                          eps_ignore: float = 1e-3) -> list[dict]: ...
    # Pass B -> per-sample records {sample_idx, reg_error_clean, reg_error_shuffled,
    #                               dur_error_clean, dur_error_shuffled, perm_index}

def write_reliance_store(path, pass_a_records, pass_b_records, support,
                         attrs: dict) -> None: ...
def write_summary(path, pass_a_records, pass_b_records, support, attrs) -> dict: ...
```

```text
# HDF5 layout — outputs/image_reliance/{run_name}_reliance.h5, group /reliance, mode "w"
# N = number of test samples; K1 = max_fixations + 1; n_dec / n_eye = decoder depths;
# n_lvl = n_image_levels (3 for mask2former @256); T_max = max valid gaze length; D = model_dim

  sample_idx            (N,)                     int32     primary key (dataset index)
  stimulus_name         (N,)                     vlen utf8 "" when unavailable
  pred_len              (N,)                     int32
  src_len               (N,)                     int32
  # --- Pass A: fixation-decoder residual norms (if fix_ok) ---
  dec_self_attention_res_norm   (N, n_dec, K1)   float32
  dec_first_cross_res_norm      (N, n_dec, K1)   float32   gaze
  dec_second_cross_res_norm     (N, n_dec, K1)   float32   image
  dec_ffn_res_norm              (N, n_dec, K1)   float32
  # --- Pass A: eye-decoder residual norms (if eye_ok), NaN-padded to T_max ---
  eye_self_attention_res_norm   (N, n_eye, T_max) float32
  eye_cross_attention_res_norm  (N, n_eye, T_max) float32  image
  eye_ffn_res_norm              (N, n_eye, T_max) float32
  # --- Pass A: sampling-in-range fractions (if fix_ok / eye_ok) ---
  dec_inrange           (N, n_dec, n_lvl)        float32   fixation decoder, [0,1]
  eye_inrange           (N, n_eye, n_lvl)        float32   eye decoder, [0,1]
  # --- Pass A: optional full fixation cross residuals (if save_full & fix_ok) ---
  dec_first_cross_res   (N, n_dec, K1, D)        float16
  dec_second_cross_res  (N, n_dec, K1, D)        float16
  # --- Pass B: perturbation ---
  reg_error_clean       (N,)                     float32
  reg_error_shuffled    (N,)                     float32   NaN if unshuffleable
  dur_error_clean       (N,)                     float32
  dur_error_shuffled    (N,)                     float32   NaN if unshuffleable
  perm_index            (N,)                     int32     source image sample; -1 if unshuffleable
```

## Dependencies

| Reads from | For |
|---|---|
| `{ckpt_path}/.hydra/config.yaml`, `model.pth`, `split.pth` | model + weights + exact test split |
| `src/model/model_io.py` (`load_pipeline`, `load_test_data`, `load_model`) | canonical CocoFreeView load path |
| `src/training/pipeline_builder.py` (`PipelineBuilder`, `build_model`, `build_dataloader`) | model + dataloader construction |
| `src/training/inference_recorder.py` (`InferenceRecorder`, `record_module_value`) | activation capture |
| `src/model/blocks.py` recorded keys (`first_cross_res`, `second_cross_res`, `cross_attention_res`, `sampling_locations`) | residuals + sampling coords (unchanged) |
| `src/eval/eval_utils.py` (`eval_autoregressive`, `invert_transforms`) | autoregressive rollout, inversion |
| `src/eval/eval_metrics.py` (`eval_reg`) | regression-error definition to mirror per-sample |
| `src/training/training_utils.py` (`move_data_to_device`) | batch → device |

| Writes to | For |
|---|---|
| `outputs/image_reliance/{run_name}_reliance.h5` | per-sample residuals, in-range fractions, perturbation errors |
| `outputs/image_reliance/{run_name}_summary.json` | aggregated headline metrics + interpretation |
