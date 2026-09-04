"""Precomputed image-feature cache (frozen Mask2Former features).

Two pieces:

* :class:`ImageFeatureCache` — a keyed HDF5 writer/reader (single group ``/features``, mode
  ``"w"``) storing, per **unique** stimulus image, the 3-level deformable memory
  ``ms_value [S, 256]`` and the stride-4 ``mask_features [256, H4, W4]`` (reserved for a future
  heatmap head), plus the ``image_path`` list that encodes the first-seen ordering.
* :class:`PrecomputedFeatureDataset` — an image dataset interface-compatible with
  :class:`DeduplicatedMemoryDataset` for :class:`CoupledDataloader`. Instead of loading/resizing
  images it returns the cached ``ms_value`` for each sample's unique image, keyed by the
  identical first-seen unique-image ordering. The order invariant (cache id ``u`` ⇔ the image
  ``DeduplicatedMemoryDataset`` assigns ``unique_id == u``) is verified unconditionally at
  construction and is not bypassable.
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageFeatureCache:
    GROUP = "features"

    @staticmethod
    def write(path, ms_value, mask_features, image_paths, attrs):
        """Write the whole cache at once (FR7). ``ms_value``: ``(U, S, 256)`` f32;
        ``mask_features``: ``(U, 256, H4, W4)`` f32 **or None** (skip the dataset — the
        ``ms_value``-only cache the model actually trains on); ``image_paths``: ``list[str]``
        of length ``U``. Chunked per-image so a single-image read touches one chunk.

        Holds the full arrays in RAM — fine for tests / small caches. For a full-dataset build
        use :meth:`create_writer` and stream one batch at a time.
        """
        ms_value = np.asarray(ms_value, dtype=np.float32)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with h5py.File(path, "w") as f:
            g = f.create_group(ImageFeatureCache.GROUP)
            g.create_dataset("ms_value", data=ms_value, dtype="float32",
                             chunks=(1,) + ms_value.shape[1:])
            if mask_features is not None:
                mask_features = np.asarray(mask_features, dtype=np.float32)
                g.create_dataset("mask_features", data=mask_features, dtype="float32",
                                 chunks=(1,) + mask_features.shape[1:])
            dt = h5py.string_dtype("utf-8")
            g.create_dataset("image_path", data=np.array(image_paths, dtype=object), dtype=dt)
            for k, v in {**attrs, "has_mask_features": mask_features is not None}.items():
                g.attrs[k] = v

    @staticmethod
    def create_writer(path, U, S, embed_dim, image_paths, attrs,
                      mask_feature_shape=None, mask_dim=256):
        """Create an empty cache with full-shape, per-image-chunked datasets and return the open
        ``h5py.File`` for streaming. The caller writes ``f["features"]["ms_value"][a:b] = ...``
        one batch at a time (and ``mask_features`` likewise when ``mask_feature_shape`` is given),
        then closes ``f`` — so the full array never lives in RAM. ``image_path`` + attrs are
        written up front. ``mask_feature_shape=None`` builds an ``ms_value``-only cache.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        f = h5py.File(path, "w")
        g = f.create_group(ImageFeatureCache.GROUP)
        g.create_dataset("ms_value", shape=(U, S, embed_dim), dtype="float32",
                         chunks=(1, S, embed_dim))
        if mask_feature_shape is not None:
            H4, W4 = mask_feature_shape
            g.create_dataset("mask_features", shape=(U, mask_dim, H4, W4), dtype="float32",
                             chunks=(1, mask_dim, H4, W4))
        dt = h5py.string_dtype("utf-8")
        g.create_dataset("image_path", data=np.array(image_paths, dtype=object), dtype=dt)
        for k, v in {**attrs, "has_mask_features": mask_feature_shape is not None}.items():
            g.attrs[k] = v
        return f

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found; run scripts/build_image_feature_cache.py to build it.")
        self.path = path
        self._f = None
        with h5py.File(path, "r") as f:
            g = f[self.GROUP]
            self.attrs = dict(g.attrs)
            self.has_mask_features = "mask_features" in g
            self.image_path = [p.decode() if isinstance(p, bytes) else p
                               for p in g["image_path"][:]]

    def _grp(self):
        # Per-worker lazy handle: opened on first access so the object is fork-safe.
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f[self.GROUP]

    def ms_value(self, u):
        return torch.from_numpy(self._grp()["ms_value"][u]).float()

    def mask_features(self, u):
        if not self.has_mask_features:
            raise KeyError(
                f"{self.path} has no mask_features (built without them); rebuild with "
                f"mask_features enabled to use them.")
        return torch.from_numpy(self._grp()["mask_features"][u]).float()


class PrecomputedFeatureDataset(Dataset):
    """Drop-in replacement for :class:`DeduplicatedMemoryDataset` that serves cached features.

    ``__getitem__(idx) -> (feature (S, 256) float32, idx, unique_idx)`` — the same tuple arity
    as ``DeduplicatedMemoryDataset`` so ``CoupledDataloader`` consumes it unchanged.
    """

    def __init__(self, data, cache_path, preload=False, return_mask_features=False):
        self.data = data
        self.cache = ImageFeatureCache(cache_path)
        self.return_mask_features = return_mask_features

        unique_paths, indices = self._build_index(data)   # first-seen order (mirrors Dedup)
        self.unique_paths = unique_paths
        self.indices = torch.as_tensor(indices, dtype=torch.long)

        # KEYING INVARIANT — cache id u must be the same image the runtime assigns unique_id==u.
        # Runs unconditionally; there is no flag to skip it.
        if len(self.cache.image_path) != len(unique_paths):
            raise ValueError(
                f"cache has {len(self.cache.image_path)} unique images but the dataset "
                f"rebuilt {len(unique_paths)} — filter/coverage mismatch.")
        for u, p in enumerate(unique_paths):
            if os.path.normpath(self.cache.image_path[u]) != os.path.normpath(p):
                raise ValueError(
                    f"cache/order mismatch at unique {u}: "
                    f"{self.cache.image_path[u]} != {p}")

        self._preloaded = None
        if preload:
            with h5py.File(cache_path, "r") as f:
                self._preloaded = torch.from_numpy(
                    f[ImageFeatureCache.GROUP]["ms_value"][:]).float()

    @staticmethod
    def _build_index(data):
        """First-seen unique-image index — byte-identical to
        ``DeduplicatedMemoryDataset.build_index`` (``path_to_id`` assigns ids in iteration
        order). Duplicated to avoid constructing an image bank."""
        path_to_id, unique_paths, indices = {}, [], []
        for i in range(len(data)):
            p = data.get_img_path(i)
            if p not in path_to_id:
                path_to_id[p] = len(unique_paths)
                unique_paths.append(p)
            indices.append(path_to_id[p])
        return unique_paths, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        u = int(self.indices[idx])
        feat = self._preloaded[u] if self._preloaded is not None else self.cache.ms_value(u)
        if self.return_mask_features:
            return feat, idx, u, self.cache.mask_features(u)
        return feat, idx, u
