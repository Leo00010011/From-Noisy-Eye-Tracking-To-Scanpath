# %% [markdown]
# # EVE Denoiser Evaluation
# Evaluates the denoise head of a trained MixerModel on real EVE WebGazer data.
# Outputs normalised coordinate error and pixel-space coordinate error.

# %%
import os
import torch
from tqdm import tqdm
import numpy as np
import sys
import gc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)

from omegaconf import OmegaConf
from src.eval.eval_metrics import eval_denoise
from src.eval.eval_utils import invert_transforms
from src.training.pipeline_builder import PipelineBuilder
from src.training.training_utils import move_data_to_device


# ── Configuration ─────────────────────────────────────────────────────────────

BUNDLE_DIR = "/mnt/beegfs/home/leonardo.ulloa/projects/bundle"   # path to the EVE bundle directory

ckpt_paths = [
    os.path.join('outputs','2026-05-27','18-55-04'),
]

names = [
    'first denoiser',
]

# ── Helper ────────────────────────────────────────────────────────────────────

def load_eve_model_and_data(ckpt_path: str, bundle_dir: str):
    """Load a MixerModel checkpoint and wire it to the EVE test dataloader.

    The model's original training config is preserved for all settings except
    the data section, which is replaced by configs/data/eve.yaml so that the
    EVE WebGazer signal and Tobii clean_x are used instead of COCO.
    """
    # Load the checkpoint's training config and override data with EVE.
    cfg = OmegaConf.load(os.path.join(ckpt_path, '.hydra', 'config.yaml'))
    eve_data = OmegaConf.load(os.path.join('configs', 'data', 'eve.yaml'))
    cfg = OmegaConf.merge(cfg, OmegaConf.create({'data': OmegaConf.to_container(eve_data, resolve=True)}))
    cfg.data.bundle_dir = bundle_dir

    pipe = PipelineBuilder(cfg)
    pipe.load_dataset()
    train_idx, val_idx, test_idx = pipe.make_splits()
    _, val_dl, test_dl = pipe.build_dataloader(train_idx, val_idx, test_idx)

    # Build model architecture from config, then load weights.
    model, _ = pipe.build_model()
    ckpt = torch.load(os.path.join(ckpt_path, 'model.pth'), map_location='cpu')
    state = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model_state_dict'].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  Missing keys ({len(missing)}): {missing[:5]}{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')

    return model, val_dl, test_dl


# ── Eval loop ─────────────────────────────────────────────────────────────────

device = 'cuda' if torch.cuda.is_available() else 'cpu'
results = []

for ckpt_path, name in zip(ckpt_paths, names):
    print(f'\n>>> Model: {name}  ({ckpt_path})')

    model, val_dl, test_dl = load_eve_model_and_data(ckpt_path, BUNDLE_DIR)
    model.set_phase('Denoise')
    model.to(device)
    model.eval()

    denoise_norm_acum = 0.0   # normalised [0,1] Euclidean error
    denoise_px_acum   = 0.0   # pixel-space Euclidean error
    count = 0

    with torch.no_grad():
        for batch in tqdm(test_dl, desc='Eval'):
            inp = move_data_to_device(batch, device)

            # encode → denoise head
            output = model(**inp)   # phase=Denoise → encode + decode_denoise

            # normalised error (matches the training denoise_error_val metric)
            denoise_norm_acum += eval_denoise(output['denoise'], inp['clean_x'])

            # pixel-space error — invert NormalizeCoords on both sides
            inp_px, out_px = invert_transforms(
                {k: v for k, v in inp.items()},
                {k: v for k, v in output.items()},
                test_dl,
            )
            diff_px = out_px['denoise'] - inp_px['clean_x'][:, :, :2]
            denoise_px_acum += torch.sqrt((diff_px ** 2).sum(-1)).mean().item()

            count += 1

    print(f'  Denoise error (normalised):  {denoise_norm_acum / count:.4f}')
    print(f'  Denoise error (pixels):      {denoise_px_acum   / count:.1f} px')
    print('  --------------------------------')

    results.append({
        'name':                name,
        'ckpt_path':           ckpt_path,
        'denoise_error_norm':  denoise_norm_acum / count,
        'denoise_error_px':    denoise_px_acum   / count,
    })

    del model
    torch.cuda.empty_cache()
    gc.collect()

# ── Summary table ─────────────────────────────────────────────────────────────
print('\n\n=== Summary ===')
print(f'{"Model":<40}  {"norm":>8}  {"px":>8}')
print('-' * 62)
for r in results:
    print(f'{r["name"]:<40}  {r["denoise_error_norm"]:>8.4f}  {r["denoise_error_px"]:>8.1f}')
