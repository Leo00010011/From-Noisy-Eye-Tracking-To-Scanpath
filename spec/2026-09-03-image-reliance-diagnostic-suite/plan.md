# Plan — Image-Reliance Diagnostic Suite

## Context and Design Decisions

**Why this suite exists.** The Mask2Former backbone migration (F1–F6, all ✓) was built to be
*additive and dual-path* precisely so that the new backbone's contribution could be measured against
the untouched gaze path. The first end-to-end training (`train_ms.sh`) landed on the same accuracy as
DINOv3 — exactly the outcome the dual-path design lets us investigate. Per the constitution's Mission,
the whole premise is that image + gaze are complementary; if the model ignores the image, that premise
is unverified for this backbone. This suite is the measurement, not a fix.

**Reuse over rebuild.** Every tensor we need is *already recorded*. `src/model/blocks.py` records
`first_cross_res` / `second_cross_res` / `ffn_res` / `self_attention_res` (the post-norm sublayer
outputs — i.e. the actual residual deltas added to the stream) inside the `norm_first=True` branch of
the deformable decoder blocks, and the F1 `DeformableAttention` records `sampling_locations` (in `[0,1]`
space, before the `2x-1` remap to `grid_sample`'s `[-1,1]`) whenever an active recorder is attached.
The Mask2Former path forces `norm_first=True` + deformable decoders, so all streams are capturable with
**zero changes to model code**. `src/notebooks/save_activations_eve_real.py` already implements the
predict-then-clean-forward recording pattern; we port its structure to CocoFreeView and add the
in-range and perturbation logic.

**Why the model's own predictions as `tgt` (not teacher-forced GT).** Chosen in planning. Matching
inference-time behaviour means the residual/sampling measurements reflect what the model actually does
when generating, and it mirrors the established `save_activations_eve_real.py` methodology (predict with
recorder off → one clean causal forward feeding predictions back as `tgt` with recorder on → fixed-shape
`(B, K1, D)` activations). The autoregressive rollout is also what makes the perturbation test
meaningful: a shuffled image must propagate through the whole rollout to move the error.

**Why residual *ratio* for the fixation decoder, *stream-relative* for the eye decoder.** The fixation
decoder (`DeformableDoubleInputDecoder`) adds two clean cross-attention residuals per layer —
`first_cross_res` (gaze, `mem1`) and `second_cross_res` (image, `mem2`) — so their ratio is a direct
image-vs-gaze reliance measure (per the [[cross-attn-residual-recording]] memo). The eye decoder
(`DeformableDecoder`) has only one cross-attention (gaze tokens → image); there is no gaze-cross to
divide by, so we report its magnitude relative to the layer's `self_attention_res` (the running gaze
stream).

**Why the in-range test targets only the eye/fixation decoders.** Their deformable queries carry
gaze/fixation reference points (noisy, possibly off-image after normalisation), so their sampling
offsets can leave `[0,1]` and hit `grid_sample`'s zero padding — the exact failure the user observed
previously. The pixel decoder's internal MSDeformAttn uses grid-centre references and is in-range by
construction, so it is excluded (locked in the clarifying questions).

**Why per-sample HDF5 + aggregated JSON.** The headline conclusions are three scalars, but per-sample /
per-layer / per-level tensors let us see whether reliance is uniform or concentrated (e.g. only the last
layer uses the image, or only the coarsest level is in-range). This mirrors the existing
`save_activation_store` design and keeps the analysis reproducible in a notebook.

**Constitution constraints honoured.** Additive (no existing file modified except the new module +
driver); reproducibility via the run's own `.hydra/config.yaml` + `split.pth`; no HDF5 layout change to
the training dataset; the frozen ResNet50 stays `eval()` (we never call `model.train()`).

## Implementation Steps

### Step 1 — `src/eval/image_reliance.py`: measurement primitives (pure, testable)

Create the module with the low-level functions that carry no I/O:

- `probe_recording_support(model)` — port verbatim from `save_activations_eve_real.py` (FR3).
- `extract_residuals(recorder, module_prefix, n_layers, value_names)` — port verbatim; returns
  `{value_name: [tensor_layer0, ...]}` where each tensor is `(B, Nq, D)` for the whole batch (the EVE
  version indexes per sample later; keep the batch tensor and slice in `residual_norms`).
- `residual_norms(res, sample_i, value_names)` → `{name: (n_layers, Nq)}`:
  ```python
  return {v: np.stack([res[v][l][sample_i].float().norm(dim=-1).cpu().numpy()
                       for l in range(n_layers)]) for v in value_names}
  ```
- `extract_sampling_locations(recorder, module_prefix, attn_attr, n_layers)`:
  ```python
  acts = recorder.current_payload["activations"]
  # module name recorded by InferenceRecorder is the dotted named_modules() path
  return [ (lambda x: x[-1] if isinstance(x, list) else x)(
             acts[f"{module_prefix}.{l}.{attn_attr}"]["sampling_locations"])
           for l in range(n_layers) ]   # each (B, Nq, H, L, P, 2)
  ```
- `sampling_in_range_fraction(sampling_locations, query_mask=None)` (FR6, FR16):
  ```python
  if sampling_locations.shape[-1] != 2: raise ValueError(...)
  x = sampling_locations[..., 0]; y = sampling_locations[..., 1]   # (B,Nq,H,L,P)
  inside = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)               # bool
  # reduce over H,P always; over Nq with optional mask -> (B, L)
  if query_mask is not None:  # query_mask (B, Nq) True=valid
      m = query_mask[:, :, None, None, None]
      num = (inside & m).sum(dim=(1,2,4)); den = (m.expand_as(inside)).sum(dim=(1,2,4))
      return (num / den.clamp(min=1)).float()                      # (B, L)
  return inside.float().mean(dim=(1,2,4))                          # (B, L)
  ```
- `shuffle_images_in_batch(image_tensor)` (FR10): `perm = torch.roll(arange(B), 1)`;
  return `image_tensor[perm], perm`. Assert `B >= 2` (caller handles the B==1 trailing batch).
- `per_sample_reg_error(pred_reg, tgt, tgt_mask)` (FR11): Euclidean on `[..., :2]`, MAE on `[..., 2]`,
  masked mean per row. Cross-check the reduction against `eval_metrics.eval_reg` so aggregate ==
  mean of per-sample on a fixed batch (validation Group 5).

**Note on module names:** confirm the recorder key format by asserting the resolved
`_inference_recorder_module_name` equals the dotted `named_modules()` path (it does — see
`InferenceRecorder.attach`). The eye deformable op is `eye_decoder.{l}.cross_attn`; the fixation one is
`decoder.{l}.second_cross_attn` (from `blocks.py`: `DeformableDecoder.cross_attn`,
`DeformableDoubleInputDecoder.second_cross_attn`).

### Step 2 — `src/eval/image_reliance.py`: Pass A driver (`run_recording_pass`)

Port the `save_activations_eve_real.py` main loop, adapted to CocoFreeView + sampling capture (FR4–FR8):

```python
def run_recording_pass(model, dataloader, device, recorder, *, save_full_residuals=True):
    support = probe_recording_support(model)
    model.set_inference_recorder(recorder); recorder.clear()
    FIX = ("self_attention_res","first_cross_res","second_cross_res","ffn_res")
    EYE = ("self_attention_res","cross_attention_res","ffn_res")
    records = []; K1 = None
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inp = move_data_to_device(batch, device)
            out = eval_autoregressive(model, inp, only_last=True)
            pred = out["reg"].clone()                              # (B,K,3)
            capture = support["fix_ok"] or support["eye_ok"]
            if capture:
                inp2 = {**inp, "tgt": pred, "tgt_mask": None, "in_tgt": None}
                recorder.start_batch(0,"Fixation","cocofreeview_test",0)
                model.encode(**inp2); model.decode_fixation(**inp2)
                K1 = K1 or pred.size(1) + 1
                fix = extract_residuals(recorder, "decoder", support["n_decoder"], FIX) if support["fix_ok"] else None
                eye = extract_residuals(recorder, "eye_decoder", support["n_eye_decoder"], EYE) if support["eye_ok"] else None
                fix_loc = extract_sampling_locations(recorder,"decoder","second_cross_attn",support["n_decoder"]) if support["fix_ok"] else None
                eye_loc = extract_sampling_locations(recorder,"eye_decoder","cross_attn",support["n_eye_decoder"]) if support["eye_ok"] else None
                # in-range per layer -> stack to (n_layers, B, L); slice per sample below
                recorder.clear()
            for i in range(inp["src"].size(0)):
                idx = int(inp["sample_idx"][i]); T = int(inp["src_mask"][i].sum())
                rec = {"sample_idx": idx, "src_len": T,
                       "stimulus_name": _stim_name(dataloader, batch, i),
                       "pred_len": _pred_len(out, i)}
                if support["fix_ok"]:
                    rec["dec_norms"] = residual_norms(fix, i, FIX)                 # {v:(n_dec,K1)}
                    rec["dec_inrange"] = np.stack([sampling_in_range_fraction(fix_loc[l])[i].cpu().numpy()
                                                   for l in range(support["n_decoder"])])   # (n_dec,L)
                    if save_full_residuals:
                        rec["full"] = {v: np.stack([fix[v][l][i].half().cpu().numpy()
                                                    for l in range(support["n_decoder"])])
                                       for v in ("first_cross_res","second_cross_res")}
                if support["eye_ok"]:
                    qmask = inp["src_mask"][i:i+1]                                  # (1,Nq) valid
                    rec["eye_norms"] = {v: residual_norms(eye, i, EYE)[v][:, :T] for v in EYE}  # trim to T
                    rec["eye_inrange"] = np.stack([sampling_in_range_fraction(eye_loc[l][i:i+1], qmask)[0].cpu().numpy()
                                                   for l in range(support["n_eye_decoder"])])    # (n_eye,L)
                records.append(rec)
    return records, {**support, "K1": K1}
```

`_stim_name` reads the coupled image dataset's stimulus label if exposed (else `""`); `_pred_len`
computes the first `sigmoid(eos)>0.5` step from `out["cls"][i]`, else `K`.

### Step 3 — `src/eval/image_reliance.py`: Pass B driver (`run_perturbation_pass`)

Recorder off; two forwards per batch, image permuted on the second (FR9–FR11):

```python
def run_perturbation_pass(model, dataloader, device, *, eps_ignore=1e-3):
    records = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inp = move_data_to_device(batch, device)
            out_c = eval_autoregressive(model, inp, only_last=True)
            rc, dc = per_sample_reg_error(out_c["reg"], inp["tgt"], inp.get("tgt_mask"))
            B = inp["src"].size(0)
            if B >= 2:
                inp_s = dict(inp)
                img_key = _image_key(inp)                    # the image tensor key in the batch dict
                perm_img, perm = shuffle_images_in_batch(inp[img_key])
                inp_s[img_key] = perm_img
                out_s = eval_autoregressive(model, inp_s, only_last=True)
                rs, ds = per_sample_reg_error(out_s["reg"], inp["tgt"], inp.get("tgt_mask"))
            else:
                rs = ds = np.full(B, np.nan); perm = torch.full((B,), -1)
            for i in range(B):
                records.append({"sample_idx": int(inp["sample_idx"][i]),
                    "reg_error_clean": float(rc[i]), "reg_error_shuffled": float(rs[i]),
                    "dur_error_clean": float(dc[i]), "dur_error_shuffled": float(ds[i]),
                    "perm_index": int(perm[i])})
    return records
```

`_image_key(inp)` locates how the image enters `encode` (the coupled batch merges the image tensor
into the model-input dict; identify the exact key when wiring — likely `image_src`/`img`). **This is
the one integration detail to confirm against `CoupledDataloader` + `MixerModel.forward` at
implementation time**, because the perturbation must replace exactly the tensor `encode` consumes.

### Step 4 — `src/eval/image_reliance.py`: writers

- `write_reliance_store(path, a_records, b_records, support, attrs)` — port `save_activation_store`,
  key by `sample_idx`, add `dec_inrange` / `eye_inrange` (FR6) and the Pass-B columns (FR9), align the
  two record lists by `sample_idx` (build a dict from `b_records`), write group attrs (FR14). Eye
  arrays NaN-padded to `T_max`. Mode `"w"`.
- `write_summary(path, a_records, b_records, support, attrs)` — compute the FR13 headline scalars +
  interpretation strings, `json.dump`, and print a table. Return the dict for testing.

### Step 5 — `src/notebooks/save_image_reliance.py`: driver

Config block (`CKPT_PATH`, `RUN_NAME`, `OUT_DIR="outputs/image_reliance"`, `SAVE_FULL_RESIDUALS`,
`EPS_IGNORE`, `DEVICE`) → load per FR1 (`load_pipeline`/`load_dataset`/`load_test_data`/`load_model`)
→ `set_phase("Fixation")`, `.to(device)`, `.eval()` → warn if `image_encoder_type != "mask2former"`
(FR2) → `recorder = InferenceRecorder(OUT_DIR, enabled=True)` → **Pass A** `run_recording_pass` →
**Pass B** `run_perturbation_pass` (assert `batch_size>=2`, FR18) → `write_reliance_store` +
`write_summary` → print the summary table.

### Step 6 — `tests/test_image_reliance.py`

CPU-only, synthetic tensors + a tiny stub recorder payload (no checkpoint, no GPU). Cover the pure
functions (in-range math incl. mask + off-map cases, derangement no-fixed-point, per-sample error vs
`eval_reg`, residual-norm shapes, error conditions FR16). See validation.md.

## Implementation Order

1. **Step 1** — pure primitives in `src/eval/image_reliance.py`.
2. **Step 2** — Pass A driver (`run_recording_pass`).
3. **Step 3** — Pass B driver (`run_perturbation_pass`).
4. **Step 4** — HDF5 writer + summary writer.
5. **Step 5** — `src/notebooks/save_image_reliance.py` driver.
6. **Step 6** — `tests/test_image_reliance.py`.
