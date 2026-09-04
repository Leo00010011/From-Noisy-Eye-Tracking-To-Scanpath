# Plan — Use Pretrained (Frozen) Image Features

## Context and Design Decisions

**Why cache the adapter `value`, not the raw images or a later tensor.** `MixerModel.encode` on the
mask2former path does exactly:

```python
bundle = self.image_encoder(image_src)          # MultiScaleFeatures
image_src = self.img_input_proj(bundle.value)   # 256 -> model_dim   (TRAINABLE)
pe  = pos_enc_mod(bundle.reference_grids.unsqueeze(0))                # (TRAINABLE)
lvl = repeat_interleave(self.level_embed, level_sizes)               # (TRAINABLE)
image_src = image_src + pe + lvl
self.image_spatial_shapes  = bundle.spatial_shapes
self.image_level_start_index = bundle.level_start_index
```

Everything the backbone produces enters through `bundle`. The frozen part ends at `bundle.value`;
`img_input_proj`, the positional projections, and `level_embed` are trainable and must keep training.
So the cache boundary is precisely `bundle.value` (`[S, 256]`) plus the fixed geometry — and the
cleanest way to re-present that to the model is to give it a stand-in `image_encoder` whose
`forward` returns the same `MultiScaleFeatures` contract. This is why **`MixerModel` needs zero
edits**: the polymorphism already lives at the `self.image_encoder(image_src)` seam.

**Why load both checkpoints (locked decision).** The user provided both `M2F_R50.pkl` (COCO-tuned
ResNet50) and `M2F_R50_MSDeformAttnPixelDecoder.pkl` (COCO-tuned pixel decoder). The pixel decoder
was trained on top of the COCO-tuned R50's features; pairing it with a plain ImageNet R50 would feed
it out-of-distribution features. The M2F R50 uses `WEIGHTS: detectron2://ImageNetPretrained/
torchvision/R-50.pkl`, `STRIDE_IN_1X1: False`, `FORMAT: RGB`, and ImageNet `PIXEL_MEAN/STD` — i.e. it
is a **torchvision-style** ResNet (stride on the 3×3 conv, RGB, ImageNet norm), only stored in
detectron2 *naming*. So the numerics transfer to torchvision `resnet50` directly; only a **key
remap** is needed, and the existing `make_transform` normalization is already correct.

**Why the pixel decoder pkl maps almost 1:1.** F2 ported `MSDeformAttnPixelDecoder` verbatim, so the
vendored submodule attribute names (`pixel_decoder.input_proj.*`, `pixel_decoder.transformer.
level_embed`, `pixel_decoder.transformer.encoder.layers.{i}.self_attn.{sampling_offsets,
attention_weights,value_proj,output_proj}`, `.norm1/.linear1/.linear2/.norm2`) match the pkl keys
under a single `pixel_decoder.` prefix. The only architectural constraint is
`transformer_enc_layers=6` (the pkl has 6 layers; the current `mask2former.yaml` default of 3 would
mismatch) and `transformer_dim_feedforward=1024` (pkl `linear1 = 1024×256`).

**Why `return_stride4=True` for precompute only.** The user wants `mask_features` cached for a future
heatmap head. That branch is built only when `return_stride4=True`. The precompute backbone therefore
enables it and stores `mask_features`; the *training* model (the stub) has no backbone and only ever
reads `ms_value`, so nothing downstream sees the 4th level — consistent with the Roadmap's deferral of
wiring stride-4 / heatmap on the m2f path.

**Why the ordering invariant is load-bearing.** `CoupledDataloader` couples gaze and image by the
sample index `idx`, and the image dataset returns a `unique_idx` produced by first-seen dedup. The
cache is indexed by that same `unique_idx`. If the precompute script and the runtime dataset build
the unique index differently, sample `i` would be paired with the wrong image silently. The invariant
is enforced two ways: (1) both sides reuse the *same* `build_index` logic over the *same* filtered
`CocoFreeView`; (2) the cache stores `image_path[u]` and the dataset asserts it matches the rebuilt
path at load — the check is not bypassable.

**Additive, dual-path.** No existing file's behavior changes for the online DINOv3 / online
Mask2Former paths. New code: one model helper file, one data file, one `nn.Module` in `ms_features.py`,
one config, and three additive `PipelineBuilder` branches. Old checkpoints and the training HDF5 are
untouched.

**Constitution constraints honored.** Pure PyTorch (no detectron2/fvcore/CUDA op) — the loaders are
plain key remaps over `torch.load`ed dicts. `InferenceRecorder` compatibility is unaffected (the stub
adds no hooks; recorder hooks live on the — now absent — backbone, which is fine because the frozen
features carry no gradient/record interest). Reproducibility: the precompute is deterministic
(`inference_mode`, frozen eval), and the cache records the checkpoint paths + config in its attrs.

---

## Implementation Steps

### Step 1 — Pretrained weight loaders (`src/model/m2f_pretrained.py`, new)

Add the remap + load helpers. No torch model construction here beyond receiving a built `backbone`.

```python
import re, torch
from dataclasses import dataclass, field

_STAGE = {"res2": "layer1", "res3": "layer2", "res4": "layer3", "res5": "layer4"}

def remap_detectron2_resnet50(sd: dict) -> dict:
    out, dropped = {}, []
    for k, v in sd.items():
        if k == "stem.conv1.weight":
            out["conv1.weight"] = v; continue
        m = re.match(r"stem\.conv1\.norm\.(.+)", k)
        if m: out[f"bn1.{m.group(1)}"] = v; continue
        m = re.match(r"stages\.(res\d)\.(\d+)\.(.+)", k)
        if m:
            stage, blk, rest = _STAGE[m.group(1)], m.group(2), m.group(3)
            pfx = f"{stage}.{blk}."
            r = re.match(r"conv(\d)\.norm\.(.+)", rest)
            if r: out[pfx + f"bn{r.group(1)}.{r.group(2)}"] = v; continue
            r = re.match(r"conv(\d)\.weight", rest)
            if r: out[pfx + f"conv{r.group(1)}.weight"] = v; continue
            if rest == "shortcut.weight": out[pfx + "downsample.0.weight"] = v; continue
            r = re.match(r"shortcut\.norm\.(.+)", rest)
            if r: out[pfx + f"downsample.1.{r.group(1)}"] = v; continue
        dropped.append(k)
    if dropped:
        print(f"[remap R50] dropped {len(dropped)} unrecognized keys: {dropped[:5]}...")
    return out

def remap_pixel_decoder(sd: dict, prefix: str = "pixel_decoder.") -> dict:
    return {prefix + k: v for k, v in sd.items()}

@dataclass
class LoadReport:
    missing_keys: list = field(default_factory=list)
    unexpected_keys: list = field(default_factory=list)
    n_resnet_loaded: int = 0
    n_pixdec_loaded: int = 0

def load_pretrained_mask2former(backbone, r50_path, pixel_decoder_path) -> LoadReport:
    r50 = torch.load(r50_path, map_location="cpu", weights_only=False)
    r50 = r50.get("model", r50)                       # tolerate {"model": ...} wrappers
    pdec = torch.load(pixel_decoder_path, map_location="cpu", weights_only=False)
    pdec = pdec.get("model", pdec)
    # feature_extractor keeps torchvision names (conv1/bn1/layerN...) under `feature_extractor.`
    fe_sd = {f"feature_extractor.{k}": v for k, v in remap_detectron2_resnet50(r50).items()}
    combined = {**fe_sd, **remap_pixel_decoder(pdec)}
    missing, unexpected = backbone.load_state_dict(combined, strict=False)
    # Guard: no ResNet or core pixel-decoder param may be left unloaded.
    bad_fe  = [k for k in missing if k.startswith("feature_extractor.")]
    bad_pd  = [k for k in missing if k.startswith("pixel_decoder.transformer.")
               or k.startswith("pixel_decoder.input_proj.")]
    if bad_fe:
        raise RuntimeError(f"ResNet50 remap incomplete; {len(bad_fe)} params unloaded: {bad_fe[:5]}")
    if bad_pd:
        raise RuntimeError(f"Pixel-decoder load incomplete; {len(bad_pd)} params unloaded: {bad_pd[:5]}")
    return LoadReport(list(missing), list(unexpected), len(fe_sd), len(remap_pixel_decoder(pdec)))
```

Notes: `create_feature_extractor` may retain a graph node for the dropped `fc` — its absence surfaces
as an *unexpected* key on the pkl side only if present, or nothing; either way it is not in `missing`
for `feature_extractor.*` conv/bn params. Verify against the real module in Step 6 tests.

### Step 2 — `PrecomputedFeatureAdapter` (`src/model/ms_features.py`, additive)

Append below `DinoV3FeatureAdapter`. Reuses the existing `build_level_start_index` /
`build_reference_grids` helpers already in this file.

```python
class PrecomputedFeatureAdapter(nn.Module):
    """Zero-parameter stand-in for a backbone: wraps a cached `ms_value` batch into a
    MultiScaleFeatures bundle using fixed geometry. Installed as MixerModel.image_encoder on the
    precomputed path so MixerModel.encode is unchanged."""
    def __init__(self, spatial_shapes, embed_dim: int = 256):
        super().__init__()
        ss = torch.as_tensor(spatial_shapes, dtype=torch.int64)
        self.embed_dim = embed_dim
        self.num_levels = ss.shape[0]
        self.register_buffer("spatial_shapes", ss, persistent=False)
        self.register_buffer("level_start_index", build_level_start_index(ss), persistent=False)
        self.register_buffer("reference_grids",
                             build_reference_grids(ss, dtype=torch.float32), persistent=False)

    def forward(self, value) -> MultiScaleFeatures:
        return MultiScaleFeatures(
            value=value,                                    # (B, S, embed_dim)
            spatial_shapes=self.spatial_shapes.to(value.device),
            level_start_index=self.level_start_index.to(value.device),
            reference_grids=self.reference_grids.to(device=value.device, dtype=value.dtype),
        )
```

`persistent=False` keeps these out of `state_dict` (they are pure geometry, reconstructed at build).
`MultiScaleFeatures.__post_init__` validates `S` vs `spatial_shapes` — a wrong-sized cache raises here.

### Step 3 — HDF5 cache IO + dataset (`src/data/image_feature_cache.py`, new)

```python
import os, datetime, h5py, numpy as np, torch
from torch.utils.data import Dataset
from src.data.datasets import DeduplicatedMemoryDataset  # reuse build_index via a light call, or inline

class ImageFeatureCache:
    GROUP = "/features"
    @staticmethod
    def write(path, ms_value, mask_features, image_paths, attrs):
        # ms_value: (U,S,256) f32 ; mask_features: (U,256,H4,W4) f32 ; image_paths: list[str]
        with h5py.File(path, "w") as f:
            g = f.create_group("features")
            g.create_dataset("ms_value", data=ms_value, dtype="float32",
                             chunks=(1,)+ms_value.shape[1:])
            g.create_dataset("mask_features", data=mask_features, dtype="float32",
                             chunks=(1,)+mask_features.shape[1:])
            dt = h5py.string_dtype("utf-8")
            g.create_dataset("image_path", data=np.array(image_paths, dtype=object), dtype=dt)
            for k, v in attrs.items(): g.attrs[k] = v

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; run scripts/build_image_feature_cache.py")
        self.path = path; self._f = None
        with h5py.File(path, "r") as f:
            self.attrs = dict(f["features"].attrs)
            self.image_path = [p.decode() if isinstance(p, bytes) else p
                               for p in f["features"]["image_path"][:]]
    def _grp(self):
        if self._f is None: self._f = h5py.File(self.path, "r")   # per-worker lazy handle
        return self._f["features"]
    def ms_value(self, u):     return torch.from_numpy(self._grp()["ms_value"][u]).float()
    def mask_features(self, u):return torch.from_numpy(self._grp()["mask_features"][u]).float()

class PrecomputedFeatureDataset(Dataset):
    def __init__(self, data, cache_path, preload=False, return_mask_features=False):
        self.data = data
        self.cache = ImageFeatureCache(cache_path)
        self.return_mask_features = return_mask_features
        unique_paths, indices = self._build_index(data)   # first-seen order (mirrors Dedup)
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        # KEYING INVARIANT — not bypassable
        for u, p in enumerate(unique_paths):
            if os.path.normpath(self.cache.image_path[u]) != os.path.normpath(p):
                raise ValueError(f"cache/order mismatch at unique {u}: "
                                 f"{self.cache.image_path[u]} != {p}")
        self._preloaded = None
        if preload:
            with h5py.File(cache_path, "r") as f:
                self._preloaded = torch.from_numpy(f["features"]["ms_value"][:]).float()
    @staticmethod
    def _build_index(data):
        path_to_id, unique_paths, indices = {}, [], []
        for i in range(len(data)):
            p = data.get_img_path(i)
            if p not in path_to_id:
                path_to_id[p] = len(unique_paths); unique_paths.append(p)
            indices.append(path_to_id[p])
        return unique_paths, indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        u = int(self.indices[idx])
        feat = self._preloaded[u] if self._preloaded is not None else self.cache.ms_value(u)
        if self.return_mask_features:
            return feat, idx, u, self.cache.mask_features(u)
        return feat, idx, u
```

`_build_index` intentionally duplicates `DeduplicatedMemoryDataset.build_index`'s algorithm (first-seen
`path_to_id`) rather than importing it, to avoid constructing an image bank. A validation test asserts
the two orderings are identical.

### Step 4 — Precompute driver (`scripts/build_image_feature_cache.py`, new)

argparse CLI (mirrors `scripts/build_eyenet_gaze_cache.py` conventions):

```
--data-root      default "data/Coco FreeView"
--r50            default "pretrained_models/M2F_R50.pkl"
--pixel-decoder  default "pretrained_models/M2F_R50_MSDeformAttnPixelDecoder.pkl"
--img-size       default 256
--out            default "data/Coco FreeView/image_features_{img_size}.h5"
--device         default "cuda" if available else "cpu"
--batch-size     default 16
```

`main()`:
1. Build `FreeViewInMemory(data_root)` to obtain `data_store['filtered_idx']`; build
   `CocoFreeView(data_root)`, `data.filter_by_idx(filtered_idx)`.
2. Rebuild the first-seen unique index (`PrecomputedFeatureDataset._build_index(data)`) → `unique_paths`.
3. Build `Mask2FormerBackbone(...return_stride4=True, transformer_enc_layers=6, freeze_backbone=True,
   freeze_pixel_decoder=True, imagenet_weights="IMAGENET1K_V2", device=device)`; call
   `load_pretrained_mask2former(backbone, r50, pixel_decoder)`, print the `LoadReport`; `backbone.eval()`.
4. Preprocess: `transform = PipelineBuilder.make_transform(img_size)`. Load the uint8 bank
   `all_images_{img_size}.pth` if present (index by `unique_paths` position — note the bank is built in
   the same first-seen order, so bank row == unique id) else decode each `unique_paths[u]` with PIL.
5. In `torch.inference_mode()`, over batches of unique images:
   `maps, mask_features = backbone(x)` → `res5,res4,res3 = maps[:3]`;
   `ms_value = cat([m.flatten(2).transpose(1,2) for m in (res5,res4,res3)], 1)` → `(b,S,256)`;
   accumulate to preallocated CPU float32 arrays `ms_value[U,S,256]`, `mask_features[U,256,H4,W4]`.
6. Assert `S == sum(H*W)` and `spatial_shapes == [[8,8],[16,16],[32,32]]` at img_size 256 (compute
   dynamically; store whatever the backbone produced).
7. `ImageFeatureCache.write(out, ms_value, mask_features, unique_paths, attrs=...)` with the FR7 attrs.
8. Print: U, S, mask_feature_shape, file size, elapsed.

### Step 5 — Config + `PipelineBuilder` wiring

- New `configs/model/image_encoder/mask2former_precomputed.yaml` (fields in FR11).
- `build_model` (around line 469–486): inside the `mask2former` branch, before constructing the
  backbone, check `if ie.get('precomputed', False):`
  ```python
  from src.model.ms_features import PrecomputedFeatureAdapter
  image_encoder = PrecomputedFeatureAdapter(
      spatial_shapes=ie.get('spatial_shapes', [[8,8],[16,16],[32,32]]),
      embed_dim=ie.get('embed_dim', 256))
  n_image_levels = image_encoder.num_levels
  # image_encoder_type stays 'mask2former'
  ```
  else the existing `Mask2FormerBackbone`+adapter path. `MixerModel(...)` args unchanged.
- `load_dataset` (around line 268–280): when
  `getattr(self.load_config, 'use_precomputed_features', False)`, build the `CocoFreeView`
  (as today) and set `self.img_dataset = PrecomputedFeatureDataset(self.data,
  cache_path=self.load_config.get('feature_cache_path'))` instead of `DeduplicatedMemoryDataset`.
  No transform is needed (features, not images).
- `build_dataloader`: no change — the `use_img_dataset` branch already wraps `self.img_dataset` in
  `Subset` + `CoupledDataloader`. Ensure `use_precomputed_features` implies `use_img_dataset` (set/
  validate in the config or add `use_img_dataset or use_precomputed_features` to the branch guard).
- FR12 cross-check: at `build_model`, if `image_encoder.precomputed` XOR
  `data.load.use_precomputed_features`, raise `ValueError`.

### Step 6 — Tests (`tests/test_image_feature_cache.py`, new)

CPU-only, synthetic where possible; one tiny real `Mask2FormerBackbone(imagenet_weights=None)` for the
adapter-equality test. See validation.md for the case list.

---

## Implementation Order

1. **Step 1** — `src/model/m2f_pretrained.py` (remap + load helpers).
2. **Step 2** — `PrecomputedFeatureAdapter` in `src/model/ms_features.py`.
3. **Step 3** — `ImageFeatureCache` + `PrecomputedFeatureDataset` in `src/data/image_feature_cache.py`.
4. **Step 4** — `scripts/build_image_feature_cache.py` (uses Steps 1 + 3 + the F2 backbone).
5. **Step 5** — config `mask2former_precomputed.yaml` + `PipelineBuilder` branches (uses Steps 2, 3).
6. **Step 6** — `tests/test_image_feature_cache.py`.
7. Run the precompute once on the real data; smoke-train a few steps on the precomputed path.
