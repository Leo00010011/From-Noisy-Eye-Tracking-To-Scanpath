# %% [markdown]
# # Save Predictions (EVE)
# Runs autoregressive fixation prediction and (optionally) denoising on EVE data,
# then writes all inputs and outputs to disk as numpy arrays.
#
# The resulting .pth file can be loaded offline — no model required — for metric
# computation, scanpath visualisation, and further analysis.
#
# Usage:
#   python src/notebooks/save_predictions_eve.py
#
# Output: one .pth per checkpoint in OUT_DIR (default: 'predictions/').
#
#   data    = torch.load('predictions/<name>.pth')
#   samples = data['samples']   # list[dict], one dict per sample
#
# Per-sample keys
# ───────────────
#   tgt_px       [N, 3] float32   ground-truth fixations  (x_px, y_px, dur_ms)
#   reg_px       [N, 3] float32   predicted fixations     (x_px, y_px, dur_ms)
#   cls          [*, 1] float32   raw EOS logits; apply sigmoid for probability
#   fixation_len int              true fixation count N (used to trim padded arrays)
#   tgt_mask     [*]   bool/float padding mask for fixation positions
#   src_norm     [T, 3] float32   noisy gaze in normalised [0,1] space (x, y, t)
# Present only when the checkpoint has a denoise head:
#   clean_x_px   [T, 2] float32   ground-truth clean gaze (x_px, y_px)
#   denoise_px   [T, 2] float32   model denoised gaze     (x_px, y_px)

# %%
import os
import sys
import gc
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)

from src.eval.eval_utils import invert_transforms, eval_autoregressive
from src.training.pipeline_builder import PipelineBuilder
from src.training.training_utils import move_data_to_device


# ── Configuration ─────────────────────────────────────────────────────────────

BUNDLE_DIR = "/mnt/beegfs/home/leonardo.ulloa/projects/bundle"
OUT_DIR = "predictions"

ckpt_paths = [
    # os.path.join('outputs', '2026-05-29', '15-33-21'),
    os.path.join('outputs','2026-06-15','19-18-05'),
]

names = [
    # 'first recover',
    'duration dist',
]


# ── Helper ────────────────────────────────────────────────────────────────────

def load_eve_model_and_data(ckpt_path: str, bundle_dir: str):
    """Load a MixerModel checkpoint and wire it to the EVE dataloaders."""
    cfg = OmegaConf.load(os.path.join(ckpt_path, '.hydra', 'config.yaml'))
    eve_data = OmegaConf.load(os.path.join('configs', 'data', 'eve.yaml'))
    cfg = OmegaConf.merge(cfg, OmegaConf.create({'data': OmegaConf.to_container(eve_data, resolve=True)}))
    cfg.data.bundle_dir = bundle_dir

    pipe = PipelineBuilder(cfg)
    pipe.load_dataset()
    train_idx, val_idx, test_idx = pipe.make_splits()
    train_dl, val_dl, _ = pipe.build_dataloader(train_idx, val_idx, test_idx)

    model, _ = pipe.build_model()
    ckpt = torch.load(os.path.join(ckpt_path, 'model.pth'), map_location='cpu')
    state = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model_state_dict'].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  Missing keys ({len(missing)}): {missing[:5]}{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')

    return model, train_dl, val_dl


def _to_numpy(t) -> np.ndarray | None:
    if t is None:
        return None
    return t.detach().cpu().numpy()


def _get_transforms(dl):
    if hasattr(dl, 'path_dataset'):
        return dl.path_dataset.transforms
    return dl.dataset.dataset.transforms


# ── Main ──────────────────────────────────────────────────────────────────────

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(OUT_DIR, exist_ok=True)

for ckpt_path, name in zip(ckpt_paths, names):
    print(f'\n>>> Model: {name}  ({ckpt_path})')

    model, train_dl, val_dl = load_eve_model_and_data(ckpt_path, BUNDLE_DIR)
    model.set_phase('Fixation')
    model.to(device)
    model.eval()

    has_denoise = callable(getattr(model, 'decode_denoise', None))
    samples = []

    with torch.no_grad():
        for dl in (train_dl, val_dl):
            for batch in tqdm(dl, desc='Saving'):
                inp = move_data_to_device(batch, device)

                # snapshot normalised gaze before any inversion
                src_norm_cpu = inp['src'].cpu()

                # autoregressive fixation prediction
                output = eval_autoregressive(model, inp, only_last=True)

                # denoise head — reuses the encoder output already cached by encode()
                if has_denoise:
                    output.update(model.decode_denoise(**inp))

                # invert transforms: normalised → pixel space for fixations and denoise
                inp_px, out_px = invert_transforms(inp, output, dl, remove_outliers=True)

                # capture raw dur output (mu, sigma²) before it's collapsed to [N,1]
                dur_raw_cpu = output['dur'].cpu() if 'dur' in output else None

                B = inp['fixation_len'].size(0)
                for i in range(B):
                    n = int(inp['fixation_len'][i].item())
                    sample = {
                        'tgt_px':      _to_numpy(inp_px['tgt'][i, :n, :]),  # [N, 3]
                        'reg_px':      _to_numpy(out_px['reg'][i, :n, :]),  # [N, 3]
                        'cls':         _to_numpy(out_px['cls'][i]),          # [*, 1]
                        'fixation_len': n,
                        'tgt_mask':    _to_numpy(inp['tgt_mask'][i]),        # [*]
                        'src_norm':    src_norm_cpu[i].numpy(),              # [T, 3]
                    }
                    # dur_raw: [N, 2] — col 0 = mu, col 1 = raw sigma² (apply softplus to get sigma²)
                    if dur_raw_cpu is not None and dur_raw_cpu.shape[-1] >= 2:
                        sample['dur_raw'] = dur_raw_cpu[i, :n, :2].numpy()
                    if has_denoise and 'clean_x' in inp_px and 'denoise' in out_px:
                        T = inp['src'].size(1)
                        sample['clean_x_px'] = _to_numpy(inp_px['clean_x'][i, :T, :2])  # [T, 2]
                        sample['denoise_px']  = _to_numpy(out_px['denoise'][i, :T, :])   # [T, 2]

                    samples.append(sample)

    safe_name = name.replace(' ', '_').replace('/', '_')
    out_path = os.path.join(OUT_DIR, f'{safe_name}.pth')
    torch.save({'name': name, 'ckpt_path': ckpt_path, 'samples': samples}, out_path)
    print(f'Saved {len(samples)} samples → {out_path}')

    del model
    torch.cuda.empty_cache()
    gc.collect()

print('\nDone.')
print("Load offline with:")
print("  data = torch.load('predictions/<name>.pth')")
print("  samples = data['samples']  # list of dicts with numpy arrays")
