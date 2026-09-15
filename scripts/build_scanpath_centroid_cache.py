"""Precompute per-image fixation centroids and write the scanpath-centroid cache.

Builds a :class:`~src.data.parsers.CocoFreeView`, enumerates unique stimuli in first-seen order
(the frozen-feature cache's keying invariant), aggregates every scanpath's fixations per image,
Mean-Shift clusters the cloud at a 1-DVA bandwidth, and writes a
:class:`~src.data.scanpath_centroids.ScanpathCentroidCache`.

Example:
    py scripts/build_scanpath_centroid_cache.py \
        --out "data/Coco FreeView/scanpath_centroids.h5" --bandwidth-dva 1.0
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.parsers import CocoFreeView
from src.data.scanpath_centroids import ScanpathCentroidCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/Coco FreeView/scanpath_centroids.h5",
                        help="output HDF5 path for the centroid cache")
    parser.add_argument("--bandwidth-dva", type=float, default=1.0,
                        help="Mean Shift / DBSCAN bandwidth in degrees of visual angle")
    parser.add_argument("--algorithm", default="meanshift", choices=["meanshift", "dbscan"])
    parser.add_argument("--min-samples", type=int, default=1, help="DBSCAN min_samples")
    parser.add_argument("--dbscan-eps-dva", type=float, default=None,
                        help="DBSCAN eps in DVA (defaults to --bandwidth-dva)")
    parser.add_argument("--split-restrict", default=None,
                        help="only aggregate scanpaths whose split == this value (FR21 leak-free)")
    parser.add_argument("--data-path", default=None, help="CocoFreeView data root")
    args = parser.parse_args()

    data = CocoFreeView(data_path=args.data_path)
    centroids, mask, order, attrs = ScanpathCentroidCache.build(
        data, bandwidth_dva=args.bandwidth_dva, algorithm=args.algorithm,
        split_restrict=args.split_restrict, dbscan_eps_dva=args.dbscan_eps_dva,
        min_samples=args.min_samples)
    ScanpathCentroidCache.write(args.out, centroids, mask, order, attrs)

    counts = mask.sum(axis=1)
    print(f"Wrote {args.out}")
    print(f"  U (unique images) = {len(order)}")
    print(f"  C_max             = {attrs['C_max']}")
    print(f"  mean centroids/img= {counts.mean():.2f} (min {int(counts.min())}, "
          f"max {int(counts.max())})")
    print(f"  images with 0     = {int((counts == 0).sum())}")


if __name__ == "__main__":
    main()
