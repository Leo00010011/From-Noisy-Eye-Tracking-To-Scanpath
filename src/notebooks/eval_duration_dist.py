# %% [markdown]
# # Duration Distribution — Scalar Metrics
# Computes mean L1 error and Pearson correlation between GT duration and two
# point estimates from the predicted log-normal:
#   - E[Y] = exp(mu + sigma²/2)   (expected value)
#   - mode = exp(mu - sigma²)
#
# All values are in normalised space (divide by 1200 ms).
#
# Usage:
#   python src/notebooks/eval_duration_dist.py

# %%
import os
import sys
import math

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)


# ── Configuration ─────────────────────────────────────────────────────────────

CHECKPOINTS = [
    {'path': os.path.join('predictions', 'duration_dist.pth'),  'type': 'lognormal'},
    {'path': os.path.join('predictions', 'first_recover.pth'),  'type': 'regression'},
]

DUR_MAX_MS = 1200.0
epsilon    = 1e-7


# ── Helpers ───────────────────────────────────────────────────────────────────

def softplus(x: float) -> float:
    return math.log1p(math.exp(x)) if x < 20 else x


def report(label: str, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
    l1      = np.mean(np.abs(pred - gt))
    r, pval = stats.pearsonr(pred, gt)
    
    print(f"  {label}")
    print(f"    Mean L1   : {l1:.6f}" + (f"({l1 * DUR_MAX_MS:.1f} ms)" if normalize else ""))
    print(f"    Pearson r : {r:.6f}  (p={pval:.2e})")


# ── Per-checkpoint evaluation ─────────────────────────────────────────────────

for ckpt in CHECKPOINTS:
    data    = torch.load(ckpt['path'], map_location='cpu', weights_only=False)
    name    = data['name']
    samples = data['samples']

    print(f"\n{'═' * 60}")
    print(f"Model : {name}  [{ckpt['type']}]")
    print(f"{'═' * 60}")

    gt_all = []

    if ckpt['type'] == 'lognormal':
        valid = [s for s in samples if 'dur_raw' in s]
        if not valid:
            print("  No 'dur_raw' found — re-run save_predictions_eve.py.")
            continue
        print(f"Samples with dur_raw: {len(valid)} / {len(samples)}")

        mean_all = []
        mode_all = []

        for s in valid:
            n       = s['fixation_len']
            dur_raw = s['dur_raw']        # [N, 2]
            gt_ms   = s['tgt_px'][:n, 2]

            for k in range(n):
                mu     = float(dur_raw[k, 0])
                sigma2 = softplus(float(dur_raw[k, 1])) + epsilon

                gt_all.append(float(gt_ms[k]) / DUR_MAX_MS)
                mean_all.append(math.exp(mu + sigma2 / 2.0))
                mode_all.append(math.exp(mu - sigma2))

        gt_all   = np.array(gt_all)
        mean_all = np.array(mean_all)
        mode_all = np.array(mode_all)

        print(f"Total fixations : {len(gt_all)}\n")
        report("E[Y] = exp(μ + σ²/2)", mean_all, gt_all)
        print()
        report("mode = exp(μ − σ²)  ", mode_all, gt_all)

    else:  # regression: reg_px[:, 2] is already the normalised duration
        print(f"Samples: {len(samples)}")

        pred_all = []

        for s in samples:
            n     = s['fixation_len']
            gt_ms = s['tgt_px'][:n, 2]
            pred  = s['reg_px'][:n, 2]   # normalised duration point estimate

            for k in range(n):
                gt_all.append(float(gt_ms[k]) )
                pred_all.append(float(pred[k]))

        gt_all   = np.array(gt_all)
        pred_all = np.array(pred_all)

        print(f"Total fixations : {len(gt_all)}\n")
        report("predicted duration", pred_all, gt_all, normalize = False)
