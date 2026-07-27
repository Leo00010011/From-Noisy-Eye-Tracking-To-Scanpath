"""One-shot CLI: predictions.csv + bundle.h5 -> eyenet_gaze_cache.h5 (Step 9).

Usage:
    python scripts/build_eyenet_gaze_cache.py \
        --csv "../EyeNet Pipeline/predictions.csv" \
        --bundle-dir /path/to/bundle \
        --out data/eve_real_noise/eyenet_gaze_cache.h5

Prints a build report (cached count, skips grouped by reason, n_offscreen, frames per
experiment, the eyenet_split breakdown, and the headline median gaze-to-ground-truth
distance). Exits 1 if any experiment was skipped, so a silent coverage regression is
not mistaken for success.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.eve_real_noise import DEFAULT_CACHE_PATH, EyeNetGazeCache


def _median_combined_error(cache: EyeNetGazeCache) -> "tuple[float, float, float]":
    """Median / mean / p90 of ||gaze_px - gt_gaze_px|| over all valid frames."""
    dists = []
    for k in cache.exp_keys:
        v = cache.get_validity(k)
        g = cache.get_gaze(k)[v]
        gt = cache.get_gt_gaze(k)[v]
        both = ~np.isnan(g).any(axis=1) & ~np.isnan(gt).any(axis=1)
        if both.any():
            dists.append(np.linalg.norm(g[both] - gt[both], axis=1))
    if not dists:
        return float("nan"), float("nan"), float("nan")
    d = np.concatenate(dists)
    return float(np.median(d)), float(np.mean(d)), float(np.percentile(d, 90))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the EyeNet screen-space gaze cache.")
    parser.add_argument("--csv", required=True, help="Path to predictions.csv")
    parser.add_argument("--bundle-dir", required=True, help="Directory containing bundle.h5")
    parser.add_argument("--out", default=DEFAULT_CACHE_PATH, help="Output cache path")
    args = parser.parse_args()

    from evedataset import EveBundle

    print(f"Loading bundle from {args.bundle_dir} ...")
    bundle = EveBundle.load(args.bundle_dir)

    print(f"Building cache from {args.csv} ...")
    cache, skipped = EyeNetGazeCache.build(args.csv, bundle, cache_path=args.out)

    # Build report
    print("\n== Build report ==============================================")
    print(f"Cached experiments : {len(cache.exp_keys)}")
    print(f"Output             : {args.out}")
    print(f"n_offscreen        : {cache.attrs['n_offscreen']}")

    if skipped:
        reasons = Counter(reason.split(":")[0] for _, reason in skipped)
        print(f"Skipped            : {len(skipped)}")
        for reason, n in reasons.most_common():
            print(f"    {reason}: {n}")
    else:
        print("Skipped            : 0")

    frame_counts = np.array([int(cache.get_validity(k).sum()) for k in cache.exp_keys])
    if frame_counts.size:
        print(
            f"Valid frames/exp   : mean {frame_counts.mean():.2f}, median "
            f"{int(np.median(frame_counts))}, min {frame_counts.min()}, "
            f"max {frame_counts.max()}, total {int(frame_counts.sum())}"
        )

    sdf = cache.splits_df
    print(f"eyenet_split        : {dict(sdf['eyenet_split'].value_counts())}")

    med, mean, p90 = _median_combined_error(cache)
    print(f"Median ||gaze - gt|| : {med:.1f} px  (mean {mean:.1f}, p90 {p90:.1f})   [expect ~89.6 px]")
    print("==============================================================")

    if skipped:
        print("\nERROR: some experiments were skipped — investigate before proceeding.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
