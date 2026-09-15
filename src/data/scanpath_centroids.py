"""Scanpath-centroid cache (image-intrinsic attentional-landmark targets).

For every unique stimulus we aggregate the fixations of **all** its scanpaths into one point
cloud (in the ``512x320`` ``dest_res`` pixel space) and spatially cluster it with **Mean Shift**
at a **1-degree-of-visual-angle** bandwidth. The resulting cluster centroids are stable,
scanpath-agnostic attentional landmarks that the image-feature adaptation head regresses toward
(see ``spec/2026-09-14-image-feature-adaptation-alignment/``).

Two pieces:

* :class:`ScanpathCentroidCache` — builds/writes/reads one HDF5 file (single group
  ``/centroids``, mode ``"w"``) storing, per **unique** stimulus image (in the same first-seen
  order as the frozen-feature cache), the normalized ``[0, 1]`` centroids ``(U, C_max, 2)``, a
  validity ``centroid_mask (U, C_max)``, and the ``image_path`` list encoding that ordering.
* the driver ``scripts/build_scanpath_centroid_cache.py`` calls
  :meth:`ScanpathCentroidCache.build` then :meth:`ScanpathCentroidCache.write`.

The cache is **additive** — a separate file from the frozen-feature cache, keyed by the identical
first-seen unique ordering so the batch's ``image_idx`` gathers the right centroids and the right
features for the same row.
"""

import os
from datetime import datetime, timezone

import h5py
import numpy as np
import torch

from sklearn.cluster import MeanShift, DBSCAN


def _cluster(cloud, algorithm, bw_px, dbscan_eps_dva, ptoa, min_samples):
    """Cluster a ``(P, 2)`` px cloud → ``(C, 2)`` centroids in the same px space.

    * ``meanshift``: ``MeanShift(bandwidth=bw_px).cluster_centers_``.
    * ``dbscan``: cluster at ``eps = (dbscan_eps_dva or bandwidth_dva) / ptoa`` px, drop the
      ``label == -1`` noise points, return per-label means.
    """
    if algorithm == "meanshift":
        ms = MeanShift(bandwidth=bw_px, bin_seeding=True).fit(cloud)
        return ms.cluster_centers_
    elif algorithm == "dbscan":
        eps = (dbscan_eps_dva / ptoa) if dbscan_eps_dva is not None else bw_px
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(cloud)
        centers = []
        for lab in sorted(set(labels.tolist())):
            if lab == -1:
                continue
            centers.append(cloud[labels == lab].mean(axis=0))
        return np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")


class ScanpathCentroidCache:
    GROUP = "centroids"

    @staticmethod
    def _first_seen_unique(data):
        """First-seen unique-image order — byte-identical to
        ``DeduplicatedMemoryDataset.build_index`` / ``PrecomputedFeatureDataset._build_index``
        (``path_to_id`` assigns ids in iteration order). Row ``u`` == ``order[u]``."""
        seen, order = {}, []
        for i in range(len(data)):
            p = data.get_img_path(i)
            if p not in seen:
                seen[p] = len(order)
                order.append(p)
        return order

    @staticmethod
    def build(data, bandwidth_dva=1.0, algorithm="meanshift",
              split_restrict=None, dbscan_eps_dva=None, min_samples=1):
        """Cluster each unique stimulus's aggregated fixations into centroids.

        Returns ``(centroids (U, C_max, 2) float32 NaN-padded, mask (U, C_max) bool, order
        list[str], attrs dict)``. Clustering runs in the isotropic ``dest_res`` px space at
        ``bandwidth = bandwidth_dva / data.ptoa`` px; centroids are then normalized to ``[0, 1]``
        by the same per-axis ``max_value = [W, H]`` that ``Normalize(key='y', mode='coords')``
        applies (FR19), so centroids live in the identical frame as ``tgt``.
        """
        order = ScanpathCentroidCache._first_seen_unique(data)
        max_value = [data.dest_res[1], data.dest_res[0]]      # [W, H] = [512, 320]
        bw_px = bandwidth_dva / data.ptoa                     # 16 px / DVA at ptoa=1/16

        # Group scanpath indices by img_path (respect split_restrict via data.df['split']).
        idx_by_path = {}
        for i in range(len(data)):
            if split_restrict is not None and data.df.iloc[i]["split"] != split_restrict:
                continue
            idx_by_path.setdefault(data.get_img_path(i), []).append(i)

        per_image = []                                        # list of (Ci, 2) [0,1] arrays
        for p in order:
            cloud = []
            for i in idx_by_path.get(p, []):
                x, y, _ = data.get_scanpath(i, downscale=True)
                cloud.append(np.stack([np.asarray(x, dtype=np.float32),
                                       np.asarray(y, dtype=np.float32)], axis=1))
            if not cloud:
                per_image.append(np.zeros((0, 2), np.float32))
                continue
            cloud = np.concatenate(cloud, 0)
            centers = _cluster(cloud, algorithm, bw_px, dbscan_eps_dva, data.ptoa, min_samples)
            centers = centers / np.asarray(max_value, np.float32)      # -> [0, 1]
            per_image.append(centers.astype(np.float32))

        C_max = max((c.shape[0] for c in per_image), default=1)
        C_max = max(C_max, 1)
        U = len(order)
        centroids = np.full((U, C_max, 2), np.nan, np.float32)
        mask = np.zeros((U, C_max), bool)
        for u, c in enumerate(per_image):
            centroids[u, :c.shape[0]] = c
            mask[u, :c.shape[0]] = True

        attrs = dict(bandwidth_dva=float(bandwidth_dva), ptoa=float(data.ptoa),
                     dest_res=list(data.dest_res), max_value=max_value,
                     algorithm=algorithm, C_max=int(C_max), num_unique=int(U))
        return centroids, mask, order, attrs

    @staticmethod
    def write(path, centroids, centroid_mask, image_paths, attrs):
        """Write the whole cache at once. Single group ``/centroids``, mode ``"w"``."""
        centroids = np.asarray(centroids, dtype=np.float32)
        centroid_mask = np.asarray(centroid_mask, dtype=bool)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with h5py.File(path, "w") as f:
            g = f.create_group(ScanpathCentroidCache.GROUP)
            g.create_dataset("centroids", data=centroids, dtype="float32",
                             chunks=(1,) + centroids.shape[1:])
            g.create_dataset("centroid_mask", data=centroid_mask, dtype=bool)
            dt = h5py.string_dtype("utf-8")
            g.create_dataset("image_path", data=np.array(image_paths, dtype=object), dtype=dt)
            all_attrs = {**attrs, "created_at": datetime.now(timezone.utc).isoformat()}
            for k, v in all_attrs.items():
                g.attrs[k] = v

    def __init__(self, path, data):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found; run scripts/build_scanpath_centroid_cache.py to build it.")
        self.path = path
        with h5py.File(path, "r") as f:
            g = f[self.GROUP]
            self.attrs = dict(g.attrs)
            centroids = np.asarray(g["centroids"][:], dtype=np.float32)
            mask = np.asarray(g["centroid_mask"][:], dtype=bool)
            stored_paths = [p.decode() if isinstance(p, bytes) else p
                            for p in g["image_path"][:]]
        self.image_path = stored_paths

        # KEYING INVARIANT (FR3) — cache row u must be the same image the runtime assigns
        # unique_id == u. Runs unconditionally; there is no flag to skip it.
        rebuilt = ScanpathCentroidCache._first_seen_unique(data)
        if len(rebuilt) != len(stored_paths):
            raise ValueError(
                f"centroid cache has {len(stored_paths)} unique images but the dataset rebuilt "
                f"{len(rebuilt)} — filter/coverage mismatch.")
        for u, p in enumerate(rebuilt):
            if os.path.normpath(stored_paths[u]) != os.path.normpath(p):
                raise ValueError(
                    f"centroid cache/order mismatch at unique {u}: {stored_paths[u]} != {p}")

        # The mask is authoritative; clear NaN pad slots to 0 so downstream cdist never sees NaN.
        centroids = np.where(mask[..., None], centroids, 0.0).astype(np.float32)
        self.centroids = torch.from_numpy(centroids)
        self.centroid_mask = torch.from_numpy(mask)
