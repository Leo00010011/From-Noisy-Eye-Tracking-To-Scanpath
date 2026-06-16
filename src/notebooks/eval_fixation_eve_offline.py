# %% [markdown]
# # EVE Fixation Evaluation — Offline
# Loads predictions saved by save_predictions_eve.py and computes fixation metrics.
# No model or dataset loading required.
#
# Matches the metrics reported by eval_fixation_eve.py:
#   - Coord error (pixels):   mean Euclidean distance between predicted and
#                             ground-truth fixation (x, y) positions.
#   - Duration error (ms):    mean absolute error on fixation duration.
#   - DTW distance (pixels):  normalised DTW between predicted and GT scanpaths.
#   - EOS accuracy / precision / recall: end-of-sequence classification metrics.

# %%
import os
import sys
import json
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)


# ── Configuration ─────────────────────────────────────────────────────────────

pred_paths = [
    os.path.join('predictions', 'first_recover.pth'),
    os.path.join('predictions', 'duration_dist.pth'),
    
]

names = [
    'first recover',
    'duration dist'
]


# ── DTW ───────────────────────────────────────────────────────────────────────

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Normalised DTW distance between two 2-D fixation sequences of shape [T, 2].

    Returns mean cost per warping step (total DTW cost / path length) so the
    result is in the same units as per-fixation Euclidean distance.
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0.0
    dtw_cost = np.full((n + 1, m + 1), np.inf)
    path_len  = np.zeros((n + 1, m + 1), dtype=np.int32)
    dtw_cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            candidates = (
                (dtw_cost[i - 1, j],     path_len[i - 1, j]),
                (dtw_cost[i, j - 1],     path_len[i, j - 1]),
                (dtw_cost[i - 1, j - 1], path_len[i - 1, j - 1]),
            )
            best_cost, best_len = min(candidates, key=lambda x: x[0])
            dtw_cost[i, j] = cost + best_cost
            path_len[i, j] = best_len + 1
    return float(dtw_cost[n, m] / path_len[n, m])


# ── EOS helpers ───────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _eos_metrics(cls: np.ndarray, tgt_mask: np.ndarray, fixation_len: int):
    """Compute EOS accuracy, precision, and recall for a single sample.

    cls        : [L, 1] raw logits
    tgt_mask   : [L]   bool/float mask (True = valid position)
    fixation_len: int  index of the EOS token in cls
    """
    L = len(cls)
    mask = tgt_mask.astype(bool)[:, np.newaxis]         # [L, 1]

    targets = np.zeros((L, 1), dtype=np.float32)
    if fixation_len < L:
        targets[fixation_len] = 1.0

    preds = (_sigmoid(cls) >= 0.5)                      # [L, 1] bool
    tgt_bool = targets >= 0.5                           # [L, 1] bool

    valid = mask.sum()
    if valid == 0:
        return 0.0, 0.0, 0.0

    correct = ((preds == tgt_bool) & mask).sum()
    acc = correct / valid

    tp  = (preds & tgt_bool & mask).sum()
    pp  = (preds & mask).sum()
    ap  = (tgt_bool & mask).sum()

    prec = float(tp / pp) if pp > 0 else 0.0
    rec  = float(tp / ap) if ap > 0 else 0.0
    return float(acc), prec, rec


# ── Eval ──────────────────────────────────────────────────────────────────────

results = []

for pred_path, name in zip(pred_paths, names):
    print(f'\n>>> Model: {name}  ({pred_path})')

    data    = torch.load(pred_path, map_location='cpu', weights_only=False)
    samples = data['samples']

    # accumulators — sum over valid fixation positions across all samples
    total_coord_err = 0.0
    total_dur_err   = 0.0
    total_positions = 0
    dtw_total       = 0.0
    acc_total = prec_total = rec_total = 0.0
    n_samples = len(samples)

    for s in samples:
        n = s['fixation_len']
        if n == 0:
            continue

        tgt = s['tgt_px']   # [N, 3]  (x_px, y_px, dur_ms)
        reg = s['reg_px']   # [N, 3]

        # coord error (pixel Euclidean per fixation)
        diff_xy = tgt[:, :2] - reg[:, :2]
        coord_err = np.sqrt((diff_xy ** 2).sum(axis=-1))  # [N]
        total_coord_err += coord_err.sum()

        # duration error (absolute, ms)
        total_dur_err += np.abs(tgt[:, 2] - reg[:, 2]).sum()
        total_positions += n

        # DTW in pixel space
        dtw_total += dtw_distance(reg[:, :2], tgt[:, :2])

        # EOS metrics
        if 'cls' in s and 'tgt_mask' in s:
            acc, prec, rec = _eos_metrics(s['cls'], s['tgt_mask'], n)
            acc_total  += acc
            prec_total += prec
            rec_total  += rec

    coord_px = total_coord_err / total_positions if total_positions > 0 else float('nan')
    dur_px   = total_dur_err   / total_positions if total_positions > 0 else float('nan')
    dtw_px   = dtw_total       / n_samples
    acc  = acc_total  / n_samples
    prec = prec_total / n_samples
    rec  = rec_total  / n_samples

    print(f'  Samples:                       {n_samples}')
    print(f'  Coord error (pixels):          {coord_px:.1f} px')
    print(f'  Duration error (ms):           {dur_px:.1f} ms')
    print(f'  DTW distance (pixels):         {dtw_px:.1f} px')
    print(f'  EOS accuracy:                  {acc:.4f}')
    print(f'  EOS precision:                 {prec:.4f}')
    print(f'  EOS recall:                    {rec:.4f}')
    print('  --------------------------------')

    results.append({
        'name':            name,
        'pred_path':       pred_path,
        'n_samples':       n_samples,
        'coord_error_px':  float(coord_px),
        'dur_error_ms':    float(dur_px),
        'dtw_px':          float(dtw_px),
        'eos_accuracy':    float(acc),
        'eos_precision':   float(prec),
        'eos_recall':      float(rec),
    })


# ── Summary table ─────────────────────────────────────────────────────────────

print('\n\n=== Summary ===')
print(f'{"Model":<40}  {"coord(px)":>9}  {"dur(ms)":>8}  {"DTW(px)":>8}  {"acc":>6}')
print('-' * 80)
for r in results:
    print(
        f'{r["name"]:<40}  '
        f'{r["coord_error_px"]:>9.1f}  '
        f'{r["dur_error_ms"]:>8.1f}  '
        f'{r["dtw_px"]:>8.1f}  '
        f'{r["eos_accuracy"]:>6.4f}'
    )

with open('fixation_offline_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nResults saved to fixation_offline_results.json')
