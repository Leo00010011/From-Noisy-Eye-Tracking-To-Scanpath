# Validation — Image-Reliance Diagnostic Suite

## Code Correctness

All Group 1–5 tests are CPU-only, use synthetic tensors and a stub recorder payload (no checkpoint,
no GPU, no network), and live in `tests/test_image_reliance.py`.

### Group 1 — In-range fraction math (`sampling_in_range_fraction`, FR6/FR16)
- [ ] All-inside: a `(2, 5, 4, 3, 4, 2)` tensor filled with `0.5` → returns all-ones `(2, 3)`
      (`atol=0`).
- [ ] All-outside: filled with `1.5` → returns all-zeros `(2, 3)`.
- [ ] Mixed per level: set level 0 fully inside, level 1 fully outside, level 2 half inside → row
      equals `[1.0, 0.0, 0.5]` (`atol=1e-6`).
- [ ] Boundary values `0.0` and `1.0` count as **inside** (inclusive); `-1e-6` and `1+1e-6` count as
      **outside**.
- [ ] `query_mask` excludes padded queries: masking out all but the first query, whose locations are
      inside, yields fraction `1.0` even when masked queries are off-map.
- [ ] Raises `ValueError` when last dim ≠ 2 (pass a `(...,3)` tensor).
- [ ] Raises `ValueError` when the level axis ≠ declared `n_levels` (shape-mismatch guard).

### Group 2 — Image shuffle is a guaranteed derangement (`shuffle_images_in_batch`, FR10)
- [ ] For `B` in `{2, 3, 8}`: `perm` has **no fixed point** (`(perm == arange(B)).any() == False`).
- [ ] `perm` is a permutation (sorted `perm == arange(B)`).
- [ ] The returned tensor equals `image_tensor[perm]` element-for-element.
- [ ] `B == 1` raises/asserts (caller handles the trailing-batch case, FR10/FR18).

### Group 3 — Per-sample regression error matches `eval_reg` (`per_sample_reg_error`, FR11)
- [ ] On a fixed batch with a known `tgt_mask`, `mean(per_sample_reg_error(...)[0])` equals
      `eval_metrics.eval_reg`'s coord error to `atol=1e-5` (reduction consistency).
- [ ] Masked (padded) target positions do **not** contribute: flipping a padded position's value
      leaves the per-sample error unchanged.
- [ ] Duration error uses only channel index 2 and is a masked MAE (hand-computed on a 2×2 example).

### Group 4 — Residual extraction shapes (`extract_residuals` / `residual_norms`, FR5)
- [ ] Against a stub `recorder.current_payload["activations"]` with keys `decoder.{0..n-1}` each
      holding `(B, K1, D)` tensors, `residual_norms(..., sample_i, FIX_KEYS)` returns
      `{key: (n_decoder, K1)}` with values equal to the hand-computed L2 over the feature dim.
- [ ] A list-valued activation bucket (defensive path) uses its **last** element.
- [ ] Missing eye keys → `eye_ok=False` path skipped without KeyError.

### Group 5 — Sampling-location extraction & module names (`extract_sampling_locations`)
- [ ] Given stub buckets `decoder.{l}.second_cross_attn` / `eye_decoder.{l}.cross_attn` each with a
      `sampling_locations (B, Nq, H, L, P, 2)` tensor, extraction returns a length-`n_layers` list of
      the correct tensors.
- [ ] Integration assertion (with a real tiny `MixerModel` at `n_image_levels=1`, CPU): after one
      recorder-on `encode`+`decode_fixation`, the recorder payload actually contains the
      `decoder.{l}.second_cross_attn` and `eye_decoder.{l}.cross_attn` `sampling_locations` keys — i.e.
      the assumed module-name format matches `InferenceRecorder.attach`'s resolved names.

### Group 6 — Recording-support probe (`probe_recording_support`, FR3/FR17)
- [ ] `norm_first=True` + deformable both → `fix_ok=True, eye_ok=True`.
- [ ] `norm_first=False` → both `False` and a warning is printed (no residuals recordable).
- [ ] Non-deformable fixation decoder → `fix_ok=False`, `eye_ok` independent.

### Group 7 — HDF5 writer roundtrip (`write_reliance_store`, FR12/FR14)
- [ ] Writing then reading back reproduces `sample_idx`, `dec_second_cross_res_norm`, `dec_inrange`,
      `eye_inrange`, and the Pass-B columns bit-for-bit (float32 exact, float16 for full residuals).
- [ ] Group attrs include every FR14 key; `fix_second_cross == "image"`, `eye_cross == "image"`,
      `fix_first_cross == "gaze"`.
- [ ] Eye arrays are NaN-padded beyond each row's `src_len`; `dec_*` arrays are dense `(N, n_dec, K1)`.
- [ ] Pass-A-only run (empty `b_records`) writes NaN/-1 perturbation columns without error.
- [ ] Skipped streams (`fix_ok=False`) omit the corresponding datasets and set the attr flag `False`.

### Group 8 — Summary aggregation (`write_summary`, FR13)
- [ ] On synthetic records where `second_cross_res` norms are exactly `0.02 ×` the `first_cross_res`
      norms, the reported image/gaze ratio is `0.02` (`atol=1e-6`) and the interpretation string flags
      `<< 1`.
- [ ] Perturbation delta and the `< eps_ignore` fraction are computed correctly on hand-set clean vs
      shuffled arrays (including NaN rows excluded from the mean).
- [ ] JSON is valid and contains the three test blocks with interpretation strings.

## Data Validity

Run on the actual `train_ms.sh` Mask2Former checkpoint (test split). Each check states its expected
outcome and the hypothesis it would confirm.

- [ ] **Sample coverage:** `N` in the store equals `len(test_dataloader.dataset)`; `sample_idx` is
      unique and matches the loader's iteration order (Pass A and Pass B agree row-for-row).
- [ ] **Residual reliance (Test 1):** report `||second_cross_res|| / ||first_cross_res||` per fixation
      layer. Ratio `<< 1` (e.g. `< 0.1`) across layers ⇒ **image ignored** (confirms the hypothesis);
      ratio `~1` or larger ⇒ image is used and the tie with DINOv3 is a different story.
- [ ] **Perturbation (Test 2):** report `mean(reg_error_shuffled) − mean(reg_error_clean)`. A delta
      within noise (and a high fraction of samples changing by `< eps_ignore`) ⇒ **image carries no
      signal**; a clear positive delta ⇒ image matters. Cross-check: `mean(reg_error_clean)` should
      be close to the training-time `reg_error_val` scale (sanity that the eval path is correct).
- [ ] **In-range (Test 3):** report per-level mean in-range fraction for the eye and fixation
      decoders. A low fraction (e.g. `< 0.5`) on any level ⇒ deformable attention samples off-map,
      `grid_sample` returns zero-padded values, attention weight is wasted — a **mechanistic cause**
      of image neglect worth fixing next. Near-`1.0` ⇒ sampling is healthy and neglect (if any) is not
      caused by off-map sampling.
- [ ] **Cross-test consistency:** if Test 1 shows negligible image residual AND Test 2 shows ~zero
      perturbation delta, the three tests agree the image path is inert; Test 3 then localises whether
      off-map sampling is the mechanism. Disagreement (e.g. large image residual but zero perturbation
      delta) flags a measurement bug to investigate before concluding.

## Data Architecture Integrity

- [ ] **Key roundtrip:** every `sample_idx` written to `/reliance` reads back identically and maps to
      the same dataset item on a re-run with the same `split.pth` (deterministic, `shuffle=False`).
- [ ] **No phantom keys:** the number of `sample_idx` entries equals the number of test items; no
      duplicates, no `-1`/placeholder ids in `sample_idx` (only `perm_index` may be `-1`).
- [ ] **Pass alignment is not bypassable:** Pass A and Pass B iterate the identical `shuffle=False`
      loader; the writer aligns Pass B to Pass A **by `sample_idx`** (dict lookup), not by row
      position, so a reordering cannot silently mismatch residuals to the wrong perturbation error.
- [ ] **Additivity:** running the suite modifies no training artifact — `dataset.hdf5`, the checkpoint,
      and `split.pth` are untouched; only files under `outputs/image_reliance/` are written.
- [ ] **Model-code untouched:** `git status` shows no change to `src/model/blocks.py`,
      `src/model/mixer_model.py`, or `src/training/inference_recorder.py` (the suite is purely additive).
