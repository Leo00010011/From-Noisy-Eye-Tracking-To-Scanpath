# %% [markdown]
# # EVE Fixation Prediction Evaluation
# Evaluates the fixation prediction head of a trained MixerModel on real EVE WebGazer data.
# Outputs: Euclidean coord error, duration MAE, DTW distance (px), and EOS accuracy.

# %%
import os
import torch
from tqdm import tqdm
import numpy as np
import json
import sys
import gc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)

from omegaconf import OmegaConf
from src.eval.eval_metrics import eval_reg, accuracy, precision, recall, create_cls_targets
from src.eval.eval_utils import invert_transforms, eval_autoregressive
from src.training.pipeline_builder import PipelineBuilder
from src.training.training_utils import move_data_to_device


# ── Configuration ─────────────────────────────────────────────────────────────

BUNDLE_DIR = "/mnt/beegfs/home/leonardo.ulloa/projects/bundle"

ckpt_paths = [
    os.path.join('outputs', '2026-05-27', '18-55-04'),
]

names = [
    'first denoiser',
]


# ── DTW ───────────────────────────────────────────────────────────────────────

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """DTW distance between two 2D fixation sequences of shape [T, 2].

    Uses Euclidean distance at each step. Returns the raw (un-normalised) DTW
    cost, which is in the same units as the input coordinates (pixels).
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0.0
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


# ── Helper ────────────────────────────────────────────────────────────────────

def load_eve_model_and_data(ckpt_path: str, bundle_dir: str):
    """Load a MixerModel checkpoint and wire it to the EVE test dataloader."""
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


# ── Eval loop ─────────────────────────────────────────────────────────────────

device = 'cuda' if torch.cuda.is_available() else 'cpu'
results = []

for ckpt_path, name in zip(ckpt_paths, names):
    print(f'\n>>> Model: {name}  ({ckpt_path})')

    model, train_dl, val_dl = load_eve_model_and_data(ckpt_path, BUNDLE_DIR)
    model.set_phase('Fixation')
    model.to(device)
    model.eval()

    # normalised-space accumulators (per batch)
    coord_norm_acum = 0.0
    dur_norm_acum   = 0.0
    # pixel-space accumulators (per batch)
    coord_px_acum   = 0.0
    dur_px_acum     = 0.0
    # DTW accumulator (per sample)
    dtw_px_acum     = 0.0
    dtw_sample_count = 0
    # EOS classification accumulators (per batch)
    acc_acum  = 0.0
    prec_acum = 0.0
    rec_acum  = 0.0
    count = 0

    with torch.no_grad():
        for dl in (train_dl, val_dl):
            for batch in tqdm(dl, desc='Eval'):
                inp = move_data_to_device(batch, device)

                # autoregressive fixation prediction
                output = eval_autoregressive(model, inp, only_last=True)

                reg_out  = output['reg']
                cls_out  = output['cls']
                y        = inp['tgt']
                y_mask   = inp['tgt_mask']
                fix_len  = inp['fixation_len']

                # normalised-space coord / duration error
                coord_err, dur_err = eval_reg(reg_out, y, y_mask)
                coord_norm_acum += coord_err
                dur_norm_acum   += dur_err

                # EOS classification metrics
                cls_targets = create_cls_targets(cls_out, fix_len)
                acc_acum  += accuracy(cls_out, y_mask, cls_targets)
                prec_acum += precision(cls_out, y_mask, cls_targets)
                rec_acum  += recall(cls_out, y_mask, cls_targets)

                # pixel-space coord / duration error
                inp_px, out_px = invert_transforms(
                    {k: v for k, v in inp.items()},
                    {k: v for k, v in output.items()},
                    dl,
                    remove_outliers=True,
                )
                coord_px, dur_px = eval_reg(out_px['reg'], inp_px['tgt'], y_mask)
                coord_px_acum += coord_px
                dur_px_acum   += dur_px

                # DTW in pixel space — one value per sample, then averaged
                B = fix_len.size(0)
                for i in range(B):
                    gt_len  = fix_len[i].item()
                    # reg has shape [B, max_N+1, 3]; compare first gt_len positions
                    gt_xy   = inp_px['tgt'][i, :gt_len, :2].cpu().numpy()
                    pred_xy = out_px['reg'][i, :gt_len, :2].cpu().numpy()
                    dtw_px_acum += dtw_distance(pred_xy, gt_xy)
                dtw_sample_count += B

                count += 1

    n = count
    print(f'  Coord error (normalised):      {coord_norm_acum / n:.4f}')
    print(f'  Duration error (normalised):   {dur_norm_acum   / n:.4f}')
    print(f'  Coord error (pixels):          {coord_px_acum   / n:.1f} px')
    print(f'  Duration error (pixels):       {dur_px_acum     / n:.1f} ms')
    print(f'  DTW distance (pixels):         {dtw_px_acum / dtw_sample_count:.1f} px')
    print(f'  EOS accuracy:                  {acc_acum  / n:.4f}')
    print(f'  EOS precision:                 {prec_acum / n:.4f}')
    print(f'  EOS recall:                    {rec_acum  / n:.4f}')
    print('  --------------------------------')

    results.append({
        'name':               name,
        'ckpt_path':          ckpt_path,
        'coord_error_norm':   coord_norm_acum / n,
        'dur_error_norm':     dur_norm_acum   / n,
        'coord_error_px':     coord_px_acum   / n,
        'dur_error_px':       dur_px_acum     / n,
        'dtw_px':             dtw_px_acum / dtw_sample_count,
        'eos_accuracy':       acc_acum  / n,
        'eos_precision':      prec_acum / n,
        'eos_recall':         rec_acum  / n,
    })

    del model
    torch.cuda.empty_cache()
    gc.collect()


# ── Summary table ─────────────────────────────────────────────────────────────

print('\n\n=== Summary ===')
print(f'{"Model":<40}  {"coord(n)":>8}  {"dur(n)":>7}  {"coord(px)":>9}  {"dur(px)":>7}  {"DTW(px)":>8}  {"acc":>6}')
print('-' * 100)
for r in results:
    print(
        f'{r["name"]:<40}  '
        f'{r["coord_error_norm"]:>8.4f}  '
        f'{r["dur_error_norm"]:>7.4f}  '
        f'{r["coord_error_px"]:>9.1f}  '
        f'{r["dur_error_px"]:>7.1f}  '
        f'{r["dtw_px"]:>8.1f}  '
        f'{r["eos_accuracy"]:>6.4f}'
    )

with open("output.json", "w") as f:
    json.dump(results, f)
