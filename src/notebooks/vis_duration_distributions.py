# %% [markdown]
# # Duration Distribution Visualiser
# Loads predictions saved by save_predictions_eve.py and plots the predicted
# log-normal duration distributions for a few sample scanpaths.
#
# For each chosen sample, one subplot per fixation shows the log-normal PDF
# defined by the model's (mu, sigma²) output, with the ground-truth duration
# marked as a vertical line.
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

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)


# ── Configuration ─────────────────────────────────────────────────────────────

PRED_PATH  = os.path.join('predictions', 'duration_dist.pth')
OUT_DIR    = os.path.join('vis_output_offline', 'duration_distributions')
N_SAMPLES  = 6      # number of scanpaths to visualise
SEED       = 42
X_MAX_MS   = 1200   # upper x-axis limit for duration plots (ms)
N_POINTS   = 300    # resolution of the PDF curve

epsilon = 1e-7


# ── Log-normal PDF helpers ────────────────────────────────────────────────────

def lognormal_pdf(x, mu, sigma2):
    """Log-normal PDF evaluated at x (scalar or array), parameterised as in the loss."""
    sigma2 = max(float(sigma2), epsilon)
    x = np.maximum(x, epsilon)
    return (1.0 / (x * np.sqrt(2 * np.pi * sigma2))
            * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma2)))


def lognormal_mode(mu, sigma2):
    """Mode of the log-normal: exp(mu - sigma²)."""
    return np.exp(mu - max(float(sigma2), epsilon))


def lognormal_mean(mu, sigma2):
    """Mean of the log-normal: exp(mu + sigma²/2)."""
    return np.exp(mu + max(float(sigma2), epsilon) / 2)


# ── Main ──────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

data    = torch.load(PRED_PATH, map_location='cpu', weights_only=False)
name    = data['name']
samples = data['samples']

# Filter to samples that have dur_raw
valid = [i for i, s in enumerate(samples) if 'dur_raw' in s]
if not valid:
    raise RuntimeError(
        "No samples contain 'dur_raw'. Re-run save_predictions_eve.py "
        "with the updated script to capture the raw duration outputs."
    )

chosen = random.sample(valid, min(N_SAMPLES, len(valid)))
print(f"Visualising {len(chosen)} samples from '{name}'  ({PRED_PATH})")
print(f"Total samples with dur_raw: {len(valid)} / {len(samples)}")

x_ms = np.linspace(epsilon, X_MAX_MS, N_POINTS)

for fig_idx, idx in enumerate(chosen):
    s     = samples[idx]
    n_fix = s['fixation_len']

    # dur_raw: [N, 2]  — col 0 = mu (normalised), col 1 = raw sigma² (pre-softplus)
    dur_raw = s['dur_raw']           # numpy [N, 2]
    # gt duration in pixel space (ms), trimmed to n_fix
    gt_dur_ms = s['tgt_px'][:n_fix, 2]   # [N]

    # Number of columns in the subplot grid
    ncols = min(n_fix, 5)
    nrows = (n_fix + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3.5 * nrows),
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

        mu_norm    = float(dur_raw[k, 0])
        sigma2_raw = float(dur_raw[k, 1])
        # Apply softplus to get the actual sigma² used in the loss
        sigma2     = float(F.softplus(torch.tensor(sigma2_raw)).item()) + epsilon

        # The model operates in normalised duration space; tgt_px is already in ms.
        # We plot the distribution in normalised units and mark GT after the same
        # inversion (or simply plot both in normalised space).
        # Here we plot in normalised space and mark the GT normalised duration from
        # reg_px col 2 (which is the mu-only estimate in px space after inversion).
        # For simplicity: plot normalised distribution + mark gt_norm from tgt_norm.
        # tgt_px[:, 2] is in ms — we don't have the inverse transform here, so we
        # show the normalised distribution vs the normalised GT from reg_px.
        # reg_px[:, 2] = mu_norm after inversion; tgt_px[:, 2] is in ms.
        # → plot PDF in normalised space, mark GT normalised duration separately.

        # Normalised x axis (same space as mu)
        # The log-normal is defined over positive reals; mu is in log(normalised_dur) space.
        # x_norm range: 0 → a few sigmas above the mean
        mean_norm = lognormal_mean(mu_norm, sigma2)
        x_max_norm = max(mean_norm * 4, np.exp(mu_norm + 3 * np.sqrt(sigma2)))
        x_norm = np.linspace(epsilon, x_max_norm, N_POINTS)
        pdf    = lognormal_pdf(x_norm, mu_norm, sigma2)

        ax.plot(x_norm, pdf, color='steelblue', linewidth=1.8, label='predicted PDF')
        ax.axvline(np.exp(mu_norm), color='steelblue', linestyle='--',
                   linewidth=1, alpha=0.7, label=f'mode≈{lognormal_mode(mu_norm, sigma2):.3f}')
        ax.axvline(mean_norm, color='navy', linestyle=':',
                   linewidth=1, alpha=0.7, label=f'mean≈{mean_norm:.3f}')

        # Mark predicted point estimate (mu = exp(mu_norm) after exp)
        pred_dur_norm = float(s['reg_px'][k, 2]) if k < len(s['reg_px']) else None

        # GT normalised duration: we can get it from tgt_px if we knew the scale,
        # but we only have ms. Show it as a text annotation instead.
        ax.set_title(f'Fix {k + 1}  |  GT {gt_dur_ms[k]:.0f} ms', fontsize=8)
        ax.set_xlabel('Normalised duration', fontsize=7)
        ax.set_ylabel('Density', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.5, loc='upper right')

        # Annotate sigma
        ax.text(0.02, 0.97,
                f'μ={mu_norm:.3f}\nσ²={sigma2:.4f}\nσ={np.sqrt(sigma2):.3f}',
                transform=ax.transAxes, fontsize=6,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f'sample_{fig_idx + 1:02d}_idx{idx:04d}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  [{fig_idx + 1:02d}] → {out_path}')

print(f'\nDone. Figures saved under {OUT_DIR}/')
