# %% [markdown]
# # EVE Attention Recording
# Runs fixation prediction on EVE data with the InferenceRecorder enabled, capturing
# the attention weights and deformable-attention sampling locations for offline analysis.
#
# Recorded activations per batch (accumulated as lists across autoregressive steps):
#
# Encoder — eye_decoder layers (DeformableDecoder, gaze tokens → image patches):
#   decoder.N.cross_attn:
#     - attention_weights   [B, n_heads, n_points]  (deformable softmax weights)
#     - sampling_offsets    [B, T_gaze, n_heads, n_points, 2]
#     - sampling_locations  [B, T_gaze, n_heads, n_points, 2]  (normalised [0,1])
#     - reference_points    [B, T_gaze, 2]
#
# Fixation decoder — DeformableDoubleInputDecoder layers:
#   decoder.N.first_cross_attn  (scanpath tokens → gaze tokens):
#     - attention_weights   [B, n_heads, N_fix_step, T_gaze]
#   decoder.N.second_cross_attn (scanpath tokens → image patches, deformable):
#     - attention_weights   [B, N_fix_step, n_heads, n_points]
#     - sampling_offsets    [B, N_fix_step, n_heads, n_points, 2]
#     - sampling_locations  [B, N_fix_step, n_heads, n_points, 2]  (normalised [0,1])
#     - reference_points    [B, N_fix_step, 2]

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
from src.eval.eval_utils import eval_autoregressive
from src.training.pipeline_builder import PipelineBuilder
from src.training.training_utils import move_data_to_device
from src.training.inference_recorder import InferenceRecorder


# ── Configuration ─────────────────────────────────────────────────────────────

BUNDLE_DIR = "/mnt/beegfs/home/leonardo.ulloa/projects/bundle"

# Directory where per-batch .pt files will be written.
RECORDINGS_DIR = "recordings/attention_eve"

# Number of batches to record per dataloader (set to None to record all).
MAX_BATCHES = 20

ckpt_paths = [
    os.path.join('outputs', '2026-05-29', '15-33-21'),
]

names = [
    'first recover',
]


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


# ── Recording loop ─────────────────────────────────────────────────────────────

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for ckpt_path, name in zip(ckpt_paths, names):
    print(f'\n>>> Model: {name}  ({ckpt_path})')

    model, train_dl, val_dl = load_eve_model_and_data(ckpt_path, BUNDLE_DIR)
    model.set_phase('Fixation')
    model.to(device)
    model.eval()

    # Create one recorder per checkpoint; output goes under RECORDINGS_DIR/<name>/
    run_dir = os.path.join(RECORDINGS_DIR, name.replace(' ', '_'))
    recorder = InferenceRecorder(output_dir=run_dir, enabled=True)
    model.set_inference_recorder(recorder)
    print(f'  Saving recordings to: {run_dir}')

    with torch.no_grad():
        for split_name, dl in (('train', train_dl), ('val', val_dl)):
            for batch_idx, batch in enumerate(tqdm(dl, desc=f'  {split_name}')):
                if MAX_BATCHES is not None and batch_idx >= MAX_BATCHES:
                    break

                inp = move_data_to_device(batch, device)

                # Start recording for this batch.
                recorder.start_batch(
                    epoch=0,
                    phase='Fixation',
                    split=split_name,
                    batch_index=batch_idx,
                )

                # encode() triggers eye_decoder attention (gaze → image).
                # decode_fixation() inside eval_autoregressive triggers
                # first_cross_attn (scanpath → gaze) and second_cross_attn
                # (scanpath → image) at every autoregressive step.
                output = eval_autoregressive(model, inp, only_last=True)

                recorder.record_batch(inp, output)
                saved_path = recorder.save_batch()
                if saved_path:
                    print(f'    Saved: {saved_path.name}')

    model.set_inference_recorder(None)
    del model
    torch.cuda.empty_cache()
    gc.collect()

print('\nDone. Load individual batches with torch.load(path).')
print('Keys in each file: "metadata", "data", "outputs", "activations".')
print('Activations are nested as activations[module_name][value_name].')
