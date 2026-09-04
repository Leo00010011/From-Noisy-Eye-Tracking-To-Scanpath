# Validation — Use Pretrained (Frozen) Image Features

## Code Correctness

### Group 1 — ResNet50 remap (`remap_detectron2_resnet50`, FR1)
- [ ] Feeding the real `M2F_R50.pkl` dict, the output contains `conv1.weight`, `bn1.weight`,
      `bn1.running_mean`, `layer1.0.conv1.weight`, `layer1.0.bn1.running_var`,
      `layer1.0.downsample.0.weight`, `layer1.0.downsample.1.running_mean`, and
      `layer4.2.conv3.weight`. No key retains a `stem.`, `stages.`, `.norm.`, or `shortcut` substring.
- [ ] Block counts survive: exactly `layer1.{0,1,2}`, `layer2.{0..3}`, `layer3.{0..5}`, `layer4.{0,1,2}`
      block prefixes appear (3/4/6/3), and only block `.0` of each layer has a `downsample.*`.
- [ ] Every value tensor is passed through unchanged (`torch.equal(out[remapped], sd[original])`), no
      dtype/shape mutation.
- [ ] A synthetic dict with an unrecognized key (`"foo.bar"`) is dropped (not in output) and reported.

### Group 2 — Pixel-decoder remap + combined load (`remap_pixel_decoder`, `load_pretrained_mask2former`, FR2/FR3)
- [ ] `remap_pixel_decoder` prefixes every key with `pixel_decoder.` and changes nothing else;
      `set(out) == {"pixel_decoder."+k for k in sd}`.
- [ ] Build a tiny real `Mask2FormerBackbone(transformer_enc_layers=6, transformer_dim_feedforward=1024,
      return_stride4=True, imagenet_weights=None, device="cpu")`; `load_pretrained_mask2former(bb,
      M2F_R50, pixel_decoder)` returns a `LoadReport` with `n_resnet_loaded >= 265*... (all conv/bn)` and
      `missing_keys` containing **no** `feature_extractor.*` and **no** `pixel_decoder.transformer.*` /
      `pixel_decoder.input_proj.*` entry.
- [ ] After load, a chosen conv weight in the backbone equals the remapped pkl value:
      `torch.equal(bb.feature_extractor.state_dict()["layer4.2.conv3.weight"],
      remap(M2F_R50)["layer4.2.conv3.weight"])` and likewise
      `bb.pixel_decoder.transformer.level_embed == pkl["transformer.level_embed"]`.
- [ ] A deliberately broken remap (delete a `layer1.*` key before load) makes
      `load_pretrained_mask2former` **raise `RuntimeError`** naming the unloaded ResNet param.
- [ ] Loading a decoder pkl into a backbone built with `transformer_enc_layers=3` leaves layers 3–5
      of the pkl as `unexpected_keys` (reported, load still applies layers 0–2) — documents the
      layer-count requirement without a silent partial-init of core layers on the 6-layer build.

### Group 3 — `PrecomputedFeatureAdapter` bundle equality (FR8, FR9)
- [ ] `PrecomputedFeatureAdapter([[8,8],[16,16],[32,32]])` has `num_levels==3`, `embed_dim==256`,
      `level_start_index == [0,64,320]`, `reference_grids.shape == (1344,2)`, and **zero** parameters
      (`sum(p.numel() for p in a.parameters()) == 0`).
- [ ] Its buffers are not in `state_dict()` (`persistent=False`) — an old checkpoint gains no new key.
- [ ] `forward(value)` with `value:(2,1344,256)` returns a `MultiScaleFeatures` whose `value is value`,
      and `spatial_shapes/level_start_index/reference_grids` match the adapter buffers.
- [ ] **Online-vs-precomputed identity:** build one tiny real `Mask2FormerBackbone(imagenet_weights=
      None, transformer_enc_layers=2, return_stride4=True, device="cpu")` (random but fixed seed);
      `online = Mask2FormerFeatureAdapter(bb)(img)`; `stub = PrecomputedFeatureAdapter(online.spatial_shapes)
      (online.value)`; assert `torch.equal(online.value, stub.value)`,
      `torch.equal(online.spatial_shapes, stub.spatial_shapes)`,
      `torch.equal(online.level_start_index, stub.level_start_index)`, and
      `torch.allclose(online.reference_grids, stub.reference_grids, atol=0)`.
- [ ] Passing a wrong-`S` value (`(2,1000,256)`) into `forward` raises `ValueError` from
      `MultiScaleFeatures.__post_init__` (Σ Hₗ·Wₗ ≠ value length).

### Group 4 — HDF5 cache roundtrip + isolation (`ImageFeatureCache`, FR7)
- [ ] `ImageFeatureCache.write` then re-open: `ms_value` shape `(U,S,256)` float32,
      `mask_features` shape `(U,256,H4,W4)` float32, `image_path` a list of `U` `str`.
      Read-back `ms_value(u)`/`mask_features(u)` are `torch.equal` to what was written (float32, no
      lossy cast).
- [ ] Group is exactly `/features`; the file contains no other top-level group (isolation).
- [ ] Attrs present and correct: `img_size`, `S`, `spatial_shapes==[8,8,16,16,32,32]`,
      `level_start_index==[0,64,320]`, `embed_dim==256`, `num_levels==3`, `mask_dim==256`,
      `mask_feature_shape`, `normalization=="imagenet_rgb"`, `transformer_enc_layers==6`,
      `num_unique==U`, and the two checkpoint-path strings.
- [ ] Chunking: `ms_value.chunks == (1, S, 256)` so a single-image read touches one chunk.
- [ ] Opening a non-existent path raises `FileNotFoundError` mentioning the precompute script (FR12).

### Group 5 — `PrecomputedFeatureDataset` + keying invariant (FR10, FR4)
- [ ] With a stub `CocoFreeView` whose `get_img_path` yields duplicated paths
      (`[a,a,b,a,c,c]`), `_build_index` returns `unique_paths==[a,b,c]`, `indices==[0,0,1,0,2,2]` —
      **byte-identical** to `DeduplicatedMemoryDataset.build_index` on the same stub (assert both).
- [ ] `__getitem__(i)` returns `(feature, i, unique_idx)` with `feature.shape==(S,256)` float32 and
      `unique_idx==indices[i]`; `len(ds)==len(data)`.
- [ ] The tuple arity/shape matches `DeduplicatedMemoryDataset.__getitem__` so `CoupledDataloader`
      consumes it unchanged (a `CoupledDataloader` over a 4-sample `Subset` yields a batch with
      `image_src.shape==(4,S,256)` and integer `image_idx`).
- [ ] **Invariant enforcement:** constructing the dataset against a cache whose `image_path[1]` is
      altered raises `ValueError` naming the mismatched unique index — the order check cannot be skipped.
- [ ] `preload=True` yields tensors `torch.equal` to the lazy path.

### Group 6 — `PipelineBuilder` integration (FR11, FR12)
- [ ] Composing `model/image_encoder=mask2former_precomputed` and building the model installs a
      `PrecomputedFeatureAdapter` as `model.image_encoder`, `model.image_encoder_type=='mask2former'`,
      `model.n_image_levels==3`, and **no** `Mask2FormerBackbone` is constructed (patch
      `Mask2FormerBackbone.__init__` to raise if called → build must not hit it).
- [ ] The built model has `img_input_proj` (256→model_dim), a `level_embed` param of shape
      `(3, model_dim)`, and deformable eye/fixation decoders with `n_levels==3` — i.e. identical
      trainable surface to the online m2f model.
- [ ] End-to-end forward: feed a batch with `image_src` of shape `(B,S,256)` (a fake cached value)
      through `model(**input)` in `Fixation` phase → returns coord/dur/cls without error, and
      `model.image_spatial_shapes.tolist()==[[8,8],[16,16],[32,32]]`.
- [ ] FR12: `precomputed: True` with `use_precomputed_features` unset raises `ValueError` at build.

### Group 7 — Model untouched (regression)
- [ ] `git diff --stat` for the feature shows **no** change to `src/model/mixer_model.py` and
      `src/model/ms_deform_backbone.py`; `ms_features.py` changes are additive (only new
      `PrecomputedFeatureAdapter`, existing symbols byte-identical).
- [ ] The online mask2former model (`model/image_encoder=mask2former`) still builds and forwards
      identically (a saved reference forward on fixed seed is `torch.equal`).

## Data Validity

Checks on the actual cache produced by `scripts/build_image_feature_cache.py` on the real data
(notebook or a marked-slow test):

- [ ] `U` equals the unique-image count `DeduplicatedMemoryDataset` reports on the same filtered
      `CocoFreeView` (~4,315 at the current split; assert exact equality, not a range).
- [ ] `S == 1344` and `spatial_shapes == [[8,8],[16,16],[32,32]]`, `mask_feature_shape == [64,64]`
      at `img_size=256`.
- [ ] `ms_value` is finite everywhere (`isfinite().all()`), **non-constant** per image
      (`std over tokens > 1e-3` for a sample of images — rules out a dead/degenerate backbone from a
      broken remap or wrong normalization), and its per-channel magnitude is in a sane range
      (`|value|.mean()` within, say, `[1e-2, 1e2]`).
- [ ] **Pretrained ≠ random sanity:** features from the pretrained-loaded backbone differ
      substantially from a fresh `imagenet_weights="IMAGENET1K_V2"` + random-decoder backbone on the
      same image (`(a-b).abs().mean()` well above numerical noise) — confirms the checkpoints actually
      took effect.
- [ ] **File-vs-online agreement:** for 5 random unique images, `cache.ms_value(u)` equals the live
      `Mask2FormerFeatureAdapter(pretrained_backbone)(transform(img)).value[0]` to `atol=1e-4`
      (rules out a write/order/precision bug end-to-end).
- [ ] Cache file size ≈ `U·(1344·256 + 256·64·64)·4` bytes (~9 GB incl. `mask_features`; `ms_value`
      alone ≈ 5.9 GB) — logged and sanity-checked, not silently ballooned.

## Data Architecture Integrity

- [ ] **Unique-order roundtrip:** `cache.image_path` equals `PrecomputedFeatureDataset._build_index`'s
      `unique_paths` (list equality, normalized paths) — the precompute writer and the runtime reader
      agree on the id↔image mapping.
- [ ] **No phantom pairing:** for a sample `i`, `PrecomputedFeatureDataset[i]`'s `unique_idx` resolves
      (via `cache.image_path`) to the same file as `data.get_img_path(i)` — i.e. gaze sample `i` is
      paired with *its own* stimulus's features, spot-checked across ≥20 random `i`.
- [ ] **Order check not bypassable:** there is no flag/branch that skips the `image_path` verification
      in `PrecomputedFeatureDataset.__init__` (grep the class; the loop runs unconditionally).
- [ ] **Filter parity:** the `CocoFreeView` filtering in the precompute script uses the *same*
      `filtered_idx` source (`FreeViewInMemory.data_store['filtered_idx']`) as `PipelineBuilder.
      load_dataset`, so both enumerate the identical image set (assert equal `len(data)` and equal
      first/last `get_img_path`).
