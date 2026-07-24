# %% [markdown]
# # Duration Distribution Visualiser
# Loads predictions saved by save_predictions_eve.py and renders the predicted
# log-normal duration distribution for each fixation in a scanpath.
#
# Duration normalisation: linear min-max with min=0, max=1200 ms.
# Everything is plotted in that normalised [0, 1] space so all subplots share
# the same x-axis and are directly comparable.
# Each PDF is peak-normalised to 1 so shape comparisons are unaffected by scale.
# The GT duration is converted to the same space and marked on the curve.
#
# Usage:
#   python src/notebooks/vis_duration_distributions.py
#
# Requires predictions saved with dur_raw key (re-run save_predictions_eve.py
# if the existing file predates the dur_raw addition).

# %%
import os
import sys
import random
import math

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)


# ── Configuration ─────────────────────────────────────────────────────────────

PRED_PATH   = os.path.join('predictions', 'duration_dist.pth')
OUT_DIR     = os.path.join('vis_output_offline', 'duration_distributions')
N_SAMPLES   = 6      # scanpaths to visualise
SEED        = 42
DUR_MAX_MS  = 1200.0 # min-max normalisation denominator
N_POINTS    = 500    # PDF resolution
NCOLS_MAX   = 6      # max columns per figure

epsilon = 1e-7


# ── Log-normal helpers ────────────────────────────────────────────────────────

def softplus(x: float) -> float:
    return math.log1p(math.exp(x)) if x < 20 else x


def lognormal_pdf(x: np.ndarray, mu: float, sigma2: float) -> np.ndarray:
    """Unnormalised log-normal PDF; x must be > 0."""
    x = np.maximum(x, epsilon)
    return (1.0 / (x * np.sqrt(2 * math.pi * sigma2))
            * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma2)))


# ── Main ──────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

data    = torch.load(PRED_PATH, map_location='cpu', weights_only=False)
name    = data['name']
samples = data['samples']

valid = [i for i, s in enumerate(samples) if 'dur_raw' in s]
if not valid:
    raise RuntimeError(
        "No samples contain 'dur_raw'. Re-run save_predictions_eve.py "
        "with the updated script to capture the raw duration outputs."
    )

chosen = random.sample(valid, min(N_SAMPLES, len(valid)))
print(f"Visualising {len(chosen)} samples from '{name}'")
print(f"Samples with dur_raw: {len(valid)} / {len(samples)}")

# Fixed x-axis in normalised [0, 1] space
x_norm = np.linspace(epsilon, 1.0, N_POINTS)

for fig_idx, idx in enumerate(chosen):
    s     = samples[idx]
    n_fix = s['fixation_len']

    dur_raw   = s['dur_raw']               # [N, 2]: col0=mu, col1=raw_sigma2
    gt_dur_ms = s['tgt_px'][:n_fix, 2]    # [N] ground-truth durations in ms

    ncols = min(n_fix, NCOLS_MAX)
    nrows = math.ceil(n_fix / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.8 * ncols, 3.2 * nrows),
                             squeeze=False)
    fig.suptitle(
        f'{name}  |  sample {fig_idx + 1}/{len(chosen)}  (idx {idx},  n_fix={n_fix})',
        fontsize=10,
    )

    for k in range(nrows * ncols):
        row, col = divmod(k, ncols)
        ax = axes[row][col]

        if k >= n_fix:
            ax.axis('off')
            continue

        mu     = float(dur_raw[k, 0])
        sigma2 = softplus(float(dur_raw[k, 1])) + epsilon

        # PDF evaluated on the fixed [0, 1] grid
        pdf = lognormal_pdf(x_norm, mu, sigma2)

        # GT in normalised space
        gt_norm = float(gt_dur_ms[k]) / DUR_MAX_MS
        gt_norm = np.clip(gt_norm, epsilon, 1.0)

        # Density at GT point
        pdf_at_gt = float(lognormal_pdf(np.array([gt_norm]), mu, sigma2)[0])

        # Mode = exp(mu - sigma²),  E[Y] = exp(mu + sigma²/2)
        mode_norm = np.exp(mu - sigma2)
        mean_norm = np.exp(mu + sigma2 / 2.0)

        ax.plot(x_norm, pdf, color='steelblue', linewidth=1.6)

        # GT vertical line + dot on the curve
        ax.axvline(gt_norm, color='tomato', linewidth=1.2, linestyle='--', alpha=0.85,
                   label=f'GT={gt_norm:.3f}  f={pdf_at_gt:.2f}')
        ax.scatter([gt_norm], [pdf_at_gt], color='tomato', s=40, zorder=5)

        # Mean (E[Y])
        if 0 < mean_norm < 1.0:
            ax.axvline(mean_norm, color='darkorange', linewidth=1.0,
                       linestyle='-', alpha=0.8, label=f'E[Y]={mean_norm:.3f}')

        # Mode
        if 0 < mode_norm < 1.0:
            ax.axvline(mode_norm, color='mediumpurple', linewidth=1.0,
                       linestyle='--', alpha=0.8, label=f'mode={mode_norm:.3f}')

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        ax.set_title(f'Fix {k + 1}  —  GT {gt_dur_ms[k]:.0f} ms', fontsize=8)
        ax.set_xlabel('Normalised duration  (0–1200 ms)', fontsize=6)
        ax.set_ylabel('Density', fontsize=6)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper right', handlelength=1.2)

        ax.text(0.03, 0.97,
                f'μ={mu:.3f}  σ={math.sqrt(sigma2):.3f}',
                transform=ax.transAxes, fontsize=6,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f'sample_{fig_idx + 1:02d}_idx{idx:04d}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  [{fig_idx + 1:02d}] → {out_path}')

print(f'\nDone. Figures saved under {OUT_DIR}/')
