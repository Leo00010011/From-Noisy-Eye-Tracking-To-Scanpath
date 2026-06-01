# %% [markdown]
# # EVE Sample Visualiser
# For N_SAMPLES random validation samples, renders two panels per figure:
#   Left  — gaze overlay: WebGazer (blue), denoised (red), Tobii clean (green)
#   Right — scanpath overlay: GT fixations (green, numbered) vs predicted (orange, numbered)

# %%
import os
import sys
import random
import gc

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)

from omegaconf import OmegaConf
from src.eval.eval_utils import eval_autoregressive
from src.training.pipeline_builder import PipelineBuilder
from src.training.training_utils import move_data_to_device


# ── Configuration ─────────────────────────────────────────────────────────────

BUNDLE_DIR    = "/mnt/beegfs/home/leonardo.ulloa/projects/bundle"
DENOISE_CKPT  = os.path.join('outputs', '2026-05-27', '18-55-04')
FIXATION_CKPT = os.path.join('outputs', '2026-05-29', '15-33-21')

N_SAMPLES = 10
OUT_DIR   = "vis_output"
SEED      = 42

# ImageNet stats used during training (see PipelineBuilder.make_transform)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Helpers ───────────────────────────────────────────────────────────────────

def denorm_image(img_tensor: torch.Tensor) -> np.ndarray:
    """ImageNet-normalised [3,H,W] float tensor → HWC numpy array in [0,1]."""
    img = img_tensor.cpu().float().numpy().transpose(1, 2, 0)  # CHW → HWC
    img = img * _STD + _MEAN
    return np.clip(img, 0.0, 1.0)


def to_display(coords_norm: np.ndarray, W: int, H: int):
    """Scale [0,1]-normalised xy columns to display pixel coords.

    coords_norm : ndarray of shape [T, 2] where [:, 0] = x, [:, 1] = y
    Returns (x_px, y_px) as 1-D arrays.
    """
    return coords_norm[:, 0] * W, coords_norm[:, 1] * H


def draw_numbered_scanpath(ax, x, y, color: str, label: str,
                            marker_size: int = 130, fontsize: int = 7):
    """Draw connecting lines and numbered circle markers for a scanpath."""
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


# ── Model / dataloader loading ────────────────────────────────────────────────

def load_model_and_val_dl(ckpt_path: str, bundle_dir: str):
    """Return (model, val_dl) built from the checkpoint's training config merged with eve.yaml."""
    cfg = OmegaConf.load(os.path.join(ckpt_path, '.hydra', 'config.yaml'))
    eve_cfg = OmegaConf.load(os.path.join('configs', 'data', 'eve.yaml'))
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create({'data': OmegaConf.to_container(eve_cfg, resolve=True)})
    )
    cfg.data.bundle_dir = bundle_dir

    pipe = PipelineBuilder(cfg)
    pipe.load_dataset()
    train_idx, val_idx, test_idx = pipe.make_splits()
    _, val_dl, _ = pipe.build_dataloader(train_idx, val_idx, test_idx)

    model, _ = pipe.build_model()
    ckpt = torch.load(os.path.join(ckpt_path, 'model.pth'), map_location='cpu')
    state = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model_state_dict'].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  Missing keys ({len(missing)}): {missing[:5]}{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')
    return model, val_dl


# ── Setup ─────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

print("\nLoading denoise model...")
denoise_model, val_dl = load_model_and_val_dl(DENOISE_CKPT, BUNDLE_DIR)
denoise_model.set_phase('Denoise')
denoise_model.to(device).eval()

print("\nLoading fixation model...")
fixation_model, _ = load_model_and_val_dl(FIXATION_CKPT, BUNDLE_DIR)
fixation_model.set_phase('Fixation')
fixation_model.to(device).eval()

# ── Fetch one validation batch ────────────────────────────────────────────────

print("\nFetching first validation batch...")
batch = next(iter(val_dl))
inp = move_data_to_device(batch, device)
B = inp['src'].size(0)
print(f"Batch size: {B}")

sample_indices = random.sample(range(B), min(N_SAMPLES, B))
print(f"Selected batch indices: {sample_indices}")

# ── Inference ─────────────────────────────────────────────────────────────────

print("\nRunning denoise inference...")
with torch.no_grad():
    denoise_out = denoise_model(**inp)   # {'denoise': [B, T, 2]}

print("Running autoregressive fixation inference...")
with torch.no_grad():
    fix_out = eval_autoregressive(fixation_model, inp, only_last=True)  # {'reg': [B, N, 3]}

# Move to CPU
src_norm     = inp['src'].cpu()              # [B, T, 3]  normalised [0,1]
clean_x_norm = inp['clean_x'].cpu()          # [B, T, 3]  normalised [0,1]
tgt_norm     = inp['tgt'].cpu()              # [B, N, 3]  normalised [0,1]
fix_len      = inp['fixation_len'].cpu()     # [B]
denoise_norm = denoise_out['denoise'].cpu()  # [B, T, 2]  normalised [0,1]
reg_norm     = fix_out['reg'].cpu()          # [B, N, 3]  normalised [0,1]
image_src    = inp['image_src'].cpu()        # [B, 3, H, W]  ImageNet-normalised

# ── Plot ──────────────────────────────────────────────────────────────────────

print(f"\nRendering {len(sample_indices)} figures...")
for fig_idx, i in enumerate(sample_indices):
    img_np = denorm_image(image_src[i])          # HWC [0,1]
    H, W   = img_np.shape[:2]                    # display size (256 × 256)
    n_fix  = int(fix_len[i].item())

    # Gaze arrays (full trajectory)
    sx, sy = to_display(src_norm[i, :, :2].numpy(),     W, H)
    cx, cy = to_display(clean_x_norm[i, :, :2].numpy(), W, H)
    dx, dy = to_display(denoise_norm[i, :, :2].numpy(), W, H)

    # Fixation arrays (valid positions only)
    gx, gy = to_display(tgt_norm[i, :n_fix, :2].numpy(),  W, H)
    px, py = to_display(reg_norm[i, :n_fix, :2].numpy(),  W, H)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'Sample {fig_idx + 1} / {len(sample_indices)}  '
        f'(batch idx {i},  n_fixations={n_fix})',
        fontsize=10
    )

    # — Panel 1: gaze overlay ——————————————————————————————————————————————
    ax = axes[0]
    ax.imshow(img_np)
    ax.scatter(sx, sy, s=2,  c='dodgerblue', alpha=0.30, label='WebGazer (noisy)')
    ax.scatter(cx, cy, s=2,  c='limegreen',  alpha=0.45, label='Tobii (clean)')
    ax.scatter(dx, dy, s=2,  c='tomato',     alpha=0.55, label='Denoised')
    ax.set_title('Gaze overlay')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=7, markerscale=4,
              framealpha=0.7, handletextpad=0.3)

    # — Panel 2: scanpath overlay ——————————————————————————————————————————
    ax = axes[1]
    ax.imshow(img_np)
    draw_numbered_scanpath(ax, gx, gy, color='limegreen', label='GT')
    draw_numbered_scanpath(ax, px, py, color='orange',    label='Predicted')
    ax.set_title(f'Scanpath  (n={n_fix} fixations)')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=7,
              framealpha=0.7, handletextpad=0.3)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f'sample_{fig_idx + 1:02d}_bidx{i:03d}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  [{fig_idx + 1:02d}] Saved {out_path}')

print(f'\nDone. {len(sample_indices)} figures saved to {OUT_DIR}/')

del denoise_model, fixation_model
torch.cuda.empty_cache()
gc.collect()
