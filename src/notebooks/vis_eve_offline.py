# %% [markdown]
# # EVE Sample Visualiser — Offline
# Loads predictions saved by save_predictions_eve.py and renders overlay figures
# without running the model.
#
# Images are NOT stored in the predictions file; supply them via the
# `images` list (one HWC uint8 numpy array per sample, in the same order as
# data['samples']), e.g. loaded with the EVE dataset helpers.
# If `images` is None or a given entry is None the panel uses a blank canvas.
#
# Each figure has up to three panels:
#   A — gaze overlay: WebGazer noisy (blue), Tobii clean (green),
#       model denoised (red).  Shown only when denoise data was saved.
#   B — scanpath overlay: GT fixations (green, numbered) vs predicted (orange).
#   C — combined: noisy gaze scatter + fixation markers side by side.

# %%
import os
import sys
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)

import torch


# ── Configuration ─────────────────────────────────────────────────────────────

pred_paths = [
    os.path.join('predictions', 'first_recover.pth'),
    os.path.join('predictions', 'duration_dist.pth'),
]

# One HWC uint8 numpy image per sample (same order as data['samples']).
# Set to None to plot on a blank canvas.
images: list | None = None

N_SAMPLES = 10     # random samples to render per prediction file
OUT_DIR   = 'vis_output_offline'
SEED      = 42


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_numbered_scanpath(ax, x, y, color: str, label: str,
                            marker_size: int = 130, fontsize: int = 7):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n == 0:
        return
    for k in range(n - 1):
        ax.plot([x[k], x[k + 1]], [y[k], y[k + 1]],
                color=color, linewidth=1.5,
                label=label if k == 0 else '')
    for k in range(n):
        ax.scatter(x[k], y[k], s=marker_size, c=color,
                   zorder=5, edgecolors='white', linewidths=0.7)
        ax.text(x[k], y[k], str(k + 1),
                ha='center', va='center',
                fontsize=fontsize, color='white', fontweight='bold', zorder=6)


def _setup_ax(ax, image: np.ndarray | None, W: int, H: int):
    """Show image or blank canvas; set axis to image convention (y down)."""
    if image is not None:
        ax.imshow(image)
    else:
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_facecolor('white')
    ax.axis('off')


def _infer_dims(s: dict) -> tuple[int, int]:
    """Estimate (W, H) from fixation pixel coordinates when no image is provided."""
    coords = np.concatenate([s['tgt_px'][:, :2], s['reg_px'][:, :2]], axis=0)
    W = max(1, int(coords[:, 0].max()) + 1)
    H = max(1, int(coords[:, 1].max()) + 1)
    return W, H


# ── Main ──────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

for pred_path in pred_paths:
    data    = torch.load(pred_path, map_location='cpu', weights_only=False)
    name    = data['name']
    samples = data['samples']

    safe_name = name.replace(' ', '_').replace('/', '_')
    sub_dir   = os.path.join(OUT_DIR, safe_name)
    os.makedirs(sub_dir, exist_ok=True)

    indices     = random.sample(range(len(samples)), min(N_SAMPLES, len(samples)))
    has_denoise = any('denoise_px' in samples[i] for i in indices)

    print(f'\n>>> {name}  —  {len(indices)} samples  (denoise={has_denoise})')

    for fig_idx, idx in enumerate(indices):
        s     = samples[idx]
        n_fix = s['fixation_len']

        # image and display dimensions
        img = images[idx] if (images is not None and idx < len(images)) else None
        if img is not None:
            H, W = img.shape[:2]
        else:
            W, H = 1920, 1080

        # noisy gaze: src_norm is [0,1] → scale to pixel space
        src_norm = s['src_norm']            # [T, 3]
        sx, sy   = src_norm[:, 0] * W, src_norm[:, 1] * H

        # fixations already in pixel space, trimmed to n_fix
        gx, gy = s['tgt_px'][:n_fix, 0], s['tgt_px'][:n_fix, 1]
        px, py = s['reg_px'][:n_fix, 0], s['reg_px'][:n_fix, 1]

        # layout: 2 or 3 panels
        n_panels = 3 if has_denoise else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))

        fig.suptitle(
            f'{name}  |  sample {fig_idx + 1}/{len(indices)}'
            f'  (idx {idx},  n_fix={n_fix})',
            fontsize=9,
        )

        col = 0

        # ── Panel A: gaze overlay ─────────────────────────────────────────────
        if has_denoise and 'denoise_px' in s:
            ax = axes[col]; col += 1
            _setup_ax(ax, img, W, H)
            ax.scatter(sx, sy, s=2, c='dodgerblue', alpha=0.30, label='WebGazer (noisy)')
            if 'clean_x_px' in s:
                cx, cy = s['clean_x_px'][:, 0], s['clean_x_px'][:, 1]
                ax.scatter(cx, cy, s=2, c='limegreen', alpha=0.45, label='Tobii (clean)')
            dx, dy = s['denoise_px'][:, 0], s['denoise_px'][:, 1]
            ax.scatter(dx, dy, s=2, c='tomato', alpha=0.55, label='Denoised')
            ax.set_title('Gaze overlay')
            ax.legend(loc='upper right', fontsize=7, markerscale=4,
                      framealpha=0.7, handletextpad=0.3)

        # ── Panel B: scanpath overlay ─────────────────────────────────────────
        ax = axes[col]; col += 1
        _setup_ax(ax, img, W, H)
        draw_numbered_scanpath(ax, gx, gy, color='limegreen', label='GT')
        draw_numbered_scanpath(ax, px, py, color='orange',    label='Predicted')
        ax.set_title(f'Scanpath  (n={n_fix} fixations)')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.7, handletextpad=0.3)

        # ── Panel C: noisy gaze + fixation markers ────────────────────────────
        ax = axes[col]
        _setup_ax(ax, img, W, H)
        ax.scatter(sx, sy, s=2, c='dodgerblue', alpha=0.35, label='WebGazer (noisy)')
        ax.scatter(gx, gy, s=60, c='limegreen', zorder=4,   label='GT fixations')
        ax.scatter(px, py, s=60, c='orange',    zorder=4,   label='Pred fixations')
        ax.set_title('Gaze + fixations')
        ax.legend(loc='upper right', fontsize=7, markerscale=2,
                  framealpha=0.7, handletextpad=0.3)

        fig.tight_layout()
        out_path = os.path.join(sub_dir, f'sample_{fig_idx + 1:02d}_idx{idx:04d}.png')
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'  [{fig_idx + 1:02d}] → {out_path}')

print(f'\nDone. Figures saved under {OUT_DIR}/')
