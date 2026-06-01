# %% [markdown]
# # EVE Denoiser Evaluation — Offline
# Loads predictions saved by save_predictions_eve.py and computes denoise metrics
# in pixel space. No model or dataset loading required.
#
# Matches the metrics reported by eval_denoise_eve.py:
#   - Denoise error (pixels): mean per-step Euclidean distance between
#     the model's denoised gaze and the Tobii ground-truth clean gaze.

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
]

names = [
    'first recover',
]


# ── Eval ──────────────────────────────────────────────────────────────────────

results = []

for pred_path, name in zip(pred_paths, names):
    print(f'\n>>> Model: {name}  ({pred_path})')

    data = torch.load(pred_path, map_location='cpu', weights_only=False)
    samples = data['samples']

    denoise_samples = [s for s in samples if 'denoise_px' in s and 'clean_x_px' in s]
    if not denoise_samples:
        print('  No denoise data found — skipping.')
        continue

    total_error = 0.0
    total_count = 0

    for s in denoise_samples:
        # both arrays are [T, 2] pixel-space coordinates
        diff = s['denoise_px'] - s['clean_x_px']
        err_per_step = np.sqrt((diff ** 2).sum(axis=-1))  # [T]
        total_error += err_per_step.sum()
        total_count += len(err_per_step)

    denoise_px = total_error / total_count

    print(f'  Samples with denoise data:    {len(denoise_samples)}')
    print(f'  Denoise error (pixels):       {denoise_px:.1f} px')
    print('  --------------------------------')

    results.append({
        'name':              name,
        'pred_path':         pred_path,
        'n_samples':         len(denoise_samples),
        'denoise_error_px':  float(denoise_px),
    })


# ── Summary table ─────────────────────────────────────────────────────────────

print('\n\n=== Summary ===')
print(f'{"Model":<40}  {"n":>6}  {"px":>8}')
print('-' * 60)
for r in results:
    print(f'{r["name"]:<40}  {r["n_samples"]:>6}  {r["denoise_error_px"]:>8.1f}')

with open('denoise_offline_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nResults saved to denoise_offline_results.json')
