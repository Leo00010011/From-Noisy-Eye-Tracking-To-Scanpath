# From Noisy Eye Tracking to Scanpath

Deep learning pipeline for scanpath prediction from noisy eye-tracking signals. This repository combines dataset preprocessing, modular training and evaluation code, Hydra-based experiment configuration, custom augmentations and normalization, and notebooks for analysis and visualization.

## Key Features

- End-to-end workflow from raw fixation data to trained models and visual analysis
- Hydra-based experiment management with composable config groups and CLI overrides
- Custom preprocessing for the COCO FreeView dataset and related image assets
- Modular training pipeline with phased training, scheduled sampling, and checkpointing
- Custom data transforms for temporal cropping, noise injection, discretization, and normalization
- Multiple model variants, including path-only and image-conditioned architectures
- Evaluation and notebook utilities for qualitative and quantitative inspection

## Overview

This project is organized as an experiment-driven research codebase for learning scanpath dynamics from noisy gaze trajectories. The core workflow is:

1. Prepare the dataset and image assets.
2. Preprocess raw scanpaths into an HDF5 cache used during training.
3. Configure an experiment with Hydra.
4. Train a model, optionally using staged denoising and fixation phases.
5. Evaluate checkpoints and inspect metrics, trajectories, and qualitative predictions.
6. Use notebooks to compare runs, visualize scanpaths, and analyze failure modes.

The repository is designed to be modular and extensible: dataset loading, transforms, models, losses, schedulers, and experiment settings are separated so that new variants can be added with limited code changes.

## Quick Start

Minimal end-to-end example:

```powershell
# 1) Install Python dependencies
py -m pip install -r requirements.txt

# 2) Install PyTorch separately if needed
# Replace with the command appropriate for your CUDA / CPU setup
py -m pip install torch torchvision

# 3) Preprocess the dataset
py preprocessing.py

# 4) Train a baseline experiment
py train.py exp=denoise_pretraining

# 5) Train the full model using the best denoising encoder found in outputs/
py train.py exp=whole_model_pretraining

# 6) Open the analysis notebooks
py -m jupyter lab src/notebooks
```

Replace the following placeholders before running:

- `<DATASET_ROOT>`: local path containing the COCO FreeView annotations and image folders
- `<DINO_REPO_PATH>`: local path to a DINOv3 checkout if you use the image-conditioned `MixerModel`
- `<DINO_WEIGHTS>`: local path to pretrained image encoder weights
- `<RUN_DIR>`: Hydra output folder for a specific experiment, for example `outputs/2026-01-20/17-17-57`

## Repository Structure

The exact layout may evolve, but the repository is organized roughly as follows:

```text
.
├── configs/                   # Hydra configuration tree
│   ├── data/                 # Dataset loading, transforms, split strategies
│   ├── exp/                  # Experiment presets
│   ├── head_type/            # Prediction head variants
│   ├── loss/                 # Loss definitions
│   ├── model/                # Model architecture presets
│   └── scheduler/            # Learning-rate schedulers
├── src/
│   ├── data/                 # Dataset classes, parsers, collate functions
│   ├── eval/                 # Evaluation metrics and visualization utilities
│   ├── model/                # Custom modules and architectures
│   ├── notebooks/            # Jupyter notebooks and analysis scripts
│   ├── preprocess/           # Noise simulation and preprocessing helpers
│   └── training/             # Pipeline builder, training loop, utilities
├── Output/                   # Example qualitative outputs, plots, videos
├── training metrics/         # Example exported metrics
├── preprocessing.py          # Dataset preprocessing entrypoint
├── train.py                  # Hydra training entrypoint
├── test.py                   # Local loading / inspection script
├── requirements.txt          # Python dependencies
└── README.md
```

## Installation

### 1. Clone the repository

```powershell
git clone <REPO_URL>
cd From-Noisy-Eye-Tracking-To-Scanpath
```

### 2. Install dependencies

```powershell
py -m pip install -r requirements.txt
```

`requirements.txt` includes Hydra, HDF5, plotting, data-processing, and notebook-related packages. PyTorch is not pinned in `requirements.txt`, so install it separately for your environment:

```powershell
py -m pip install torch torchvision
```

If you plan to work with notebooks:

```powershell
py -m pip install jupyterlab
```

### 3. Check hardware settings

Model configs default to `device: "cuda"` in the provided presets. If you want CPU execution or a different device string, override it from the command line:

```powershell
py train.py model.device=cpu
```

## Dataset

This repository is built around the **COCO FreeView** eye-tracking data and associated image assets. The parser expects a dataset directory similar to:

```text
data/
└── Coco FreeView/
    ├── COCOFreeView_fixations_trainval.json
    ├── COCOSearch18-images-TA/
    └── COCOSearch18-images-TP/
```

Expected inputs include:

- fixation annotations in JSON format
- stimulus images referenced by filename
- optional cached files generated by preprocessing or dataset indexing

Notes:

- The code internally rescales images and coordinates to a working resolution.
- A cached mapping file such as `ntop.json` may be generated automatically.
- Preprocessed HDF5 files such as `dataset.hdf5` or `dataset_<downsample>.hdf5` are used for training.

If your dataset lives somewhere else, update the default paths in the parser and preprocessing scripts or adapt the config/code to point to your local storage.

## Data Processing

Raw gaze and fixation data are converted into an HDF5 dataset used during training. The preprocessing pipeline:

- loads the raw scanpaths
- simulates gaze trajectories and fixation masks
- downsamples the gaze stream
- filters invalid or short sequences
- stores processed arrays and filtered sample indices in HDF5

Run preprocessing with:

```powershell
py preprocessing.py
```

By default, this script writes output like:

```text
data/Coco FreeView/dataset_<sampling_rate>.hdf5
```

Before running, review the constants near the bottom of `preprocessing.py` and replace as needed:

- sampling rate
- downsampling interval
- minimum scanpath duration
- fixation duration limits
- dataset save path

## Augmentations and Normalization

Training-time transforms are configured through `configs/data/default.yaml` and instantiated in `src/training/pipeline_builder.py`. The current pipeline supports operations such as:

- random temporal window extraction
- time standardization
- saving a clean copy of the source trajectory for denoising supervision
- correlated radial noise centered around a moving reference point
- discretization noise
- coordinate normalization
- time normalization
- fixation-duration normalization
- optional Gaussian noise injected into target fixations
- optional heatmap target generation
- optional curriculum noise scheduling

These transforms are applied compositionally using the `data.transforms.transform_list` config entry. Typical uses include:

- changing coordinate normalization mode
- switching between duration normalization schemes
- enabling denoising or curriculum training
- adding or removing data augmentations for ablation studies

## Configuration with Hydra

Experiments are managed with Hydra via `configs/main.yaml`. The default config composes several groups:

- `scheduler`
- `data`
- `model`
- `loss`
- `head_type`
- `exp`

Example config groups available in this repository:

- `configs/model/`: `mixer_model`, `path_model`
- `configs/exp/`: `denoise_pretraining`, `whole_model_pretraining`
- `configs/scheduler/`: `warmup`, `one_cycle`, `multistep_lr`
- `configs/loss/`: `separated_loss`, `entire_loss`, `focal_loss`
- `configs/head_type/`: `linear`, `mlp`, `multi_mlp`, `heatmap`, `argmax_regressor`, `start_head`
- `configs/data/split_strategy/`: `random`, `stimuly_disjoint`, `disjoint`

Hydra overrides are the main way to launch controlled experiments. Examples:

```powershell
py train.py model=path_model
py train.py exp=denoise_pretraining scheduler=one_cycle
py train.py training.learning_rate=1e-4 training.val_interval=5
py train.py data.load.batch_size=64 data.split_strategy=random
py train.py model.use_rope=True model.input_encoder=shared_gaussian
```

Hydra automatically creates per-run output folders under `outputs/` and stores:

- the resolved config in `.hydra/`
- `metrics.json`
- `model.pth`
- `split.pth`
- optional inference recorder outputs

## Training

The main training entrypoint is:

```powershell
py train.py
```

Training is built around `PipelineBuilder`, which:

- loads the processed eye-tracking dataset
- optionally couples gaze sequences with image data
- creates train/validation/test splits
- builds the selected model, optimizer, scheduler, and loss
- runs multi-phase training and validation
- saves checkpoints and metrics to the Hydra run directory

The codebase supports phased training through `training.Phases`, for example:

- `Denoise`
- `Fixation`
- `Combined`

This is useful when pretraining an encoder for denoising before optimizing the full fixation prediction objective.

### Common training patterns

Baseline path-only training:

```powershell
py train.py model=path_model head_type=multi_mlp
```

Denoising pretraining:

```powershell
py train.py exp=denoise_pretraining
```

Full model training using the best denoising checkpoint found under `outputs/`:

```powershell
py train.py exp=whole_model_pretraining
```

Disable scheduled sampling:

```powershell
py train.py training.use_scheduled_sampling=False
```

Train on CPU for debugging:

```powershell
py train.py model.device=cpu data.load.num_workers=0
```

## Evaluation

Evaluation utilities are provided in `src/eval/` and in the analysis notebooks. The repository currently leans toward script- and notebook-based evaluation rather than a single dedicated CLI.

Typical evaluation workflow:

1. Load a run from its Hydra output directory.
2. Restore the saved model and split indices.
3. Run inference on the validation or test set.
4. Invert preprocessing transforms where needed.
5. Compute regression and denoising metrics.
6. Generate plots and qualitative visualizations.

Key helpers include:

- `src/model/model_io.py` for loading checkpoints and saved splits
- `src/eval/eval_metrics.py` for regression, denoising, and classification metrics
- `src/eval/eval_utils.py` for transform inversion, metric aggregation, and plotting
- `src/eval/vis_scanpath.py` for scanpath rendering

Example evaluation entry points:

```powershell
# Open notebook-driven evaluation
py -m jupyter lab src/notebooks/eval.ipynb

# Run the script-style batch evaluation notebook if adapted for your checkpoints
py src/notebooks/eval_batch.py
```

Replace hard-coded checkpoint paths in notebook scripts with your own `<RUN_DIR>` values before running.

## Visualization Notebooks

The `src/notebooks/` directory contains notebooks for:

- exploratory data analysis
- training-metric review
- validation and inference inspection
- scanpath visualization
- augmentation and normalization experiments
- ablation analysis

Examples in the repository include notebooks for:

- COCO / free-view analysis
- duration normalization
- Gaussian map experiments
- noise curriculum experiments
- evaluation and validation review
- inference recorder inspection

Launch notebooks with:

```powershell
py -m jupyter lab src/notebooks
```

Most notebooks assume access to:

- preprocessed dataset files
- one or more Hydra output folders under `outputs/`
- matching image assets

## Custom Models / Modules

The repository includes custom deep learning modules in `src/model/`, with two main model families:

### `PathModel`

A path-only sequence model that predicts scanpath outputs from trajectory inputs without image features. This is useful for debugging, ablations, or lightweight baselines.

### `MixerModel`

A richer architecture that combines noisy eye-tracking sequences with image-conditioned features. It supports:

- separate denoising and fixation components
- configurable input encoders
- optional image encoder integration
- feature enhancement and adapter layers
- multiple prediction heads
- deformable decoders
- optional RoPE-based position handling
- scheduled sampling integration

Additional custom modules cover:

- transformer blocks
- positional encoders
- RoPE embeddings
- loss functions
- image-encoder wrapping
- inference recording
- learning-rate and weighting schedulers

If you use `MixerModel` with the image encoder enabled, review and replace the local paths in `configs/model/mixer_model.yaml`, especially:

- `model.image_encoder.repo_path`
- `model.image_encoder.weights`

## Example Commands

Install dependencies:

```powershell
py -m pip install -r requirements.txt
py -m pip install torch torchvision jupyterlab
```

Preprocess data:

```powershell
py preprocessing.py
```

Train with default config:

```powershell
py train.py
```

Train with Hydra overrides:

```powershell
py train.py exp=denoise_pretraining scheduler=warmup training.val_interval=5
py train.py model=path_model head_type=multi_mlp training.learning_rate=5e-5
py train.py data.load.batch_size=64 data.transforms.AddGaussianNoiseToFixations.std=8.0
py train.py data.split_strategy=stimuly_disjoint model.use_rope=False
```

Resume from or initialize with a checkpoint:

```powershell
py train.py training.pretrained_model=<RUN_DIR>
py train.py model.pretrained_encoder_path=<RUN_DIR>\\model.pth
```

Run evaluation utilities:

```powershell
py src/notebooks/eval_batch.py
```

Open notebooks:

```powershell
py -m jupyter lab src/notebooks
```

## Results / Outputs

Each Hydra run creates an experiment directory under `outputs/` containing artifacts such as:

- resolved Hydra config
- `metrics.json`
- `model.pth`
- `split.pth`
- optional recorded inference batches

The repository also contains an `Output/` directory with example plots and qualitative exports, such as:

- scanpath visualizations
- training-loss curves
- distribution plots
- qualitative comparison images
- videos

Treat these as examples of downstream analysis outputs rather than fixed benchmark results.

## Reproducibility

This repository is structured for reproducible experiments, but reproducibility still depends on controlling paths, seeds, hardware, and data versions.

Recommended practices:

- keep each experiment in its own Hydra run directory
- commit config changes alongside code changes
- prefer Hydra overrides over ad hoc code edits
- save and reference exact checkpoint paths
- keep dataset layout stable across runs
- record the image-encoder weights and external dependency versions used

To improve reproducibility further, consider explicitly setting and logging:

- random seeds for NumPy and PyTorch
- CUDA determinism settings when needed
- dataset version hashes
- exact image encoder revision and weights

## Troubleshooting

### Dataset file not found

Check that the expected COCO FreeView annotation JSON and image folders exist under your local dataset root.

### Preprocessed HDF5 file missing

Run:

```powershell
py preprocessing.py
```

Then verify that `data/Coco FreeView/dataset*.hdf5` was created.

### CUDA not available

Override the device:

```powershell
py train.py model.device=cpu
```

### Image encoder paths are invalid

If `MixerModel` is enabled with image features, update `configs/model/mixer_model.yaml` to point to valid local paths for:

- the DINOv3 repository
- pretrained weights

Alternatively, disable the image encoder or switch to `model=path_model`.

### Notebook paths are hard-coded

Some analysis scripts and notebooks contain run-specific checkpoint paths. Replace them with your own `<RUN_DIR>` before execution.

### Windows path issues

Use quoted paths when needed and prefer PowerShell-friendly commands. This repository is already using Windows-style path conventions in some configs and scripts.

## Future Improvements

Potential next steps for the project include:

- adding a dedicated evaluation CLI
- centralizing dataset-path configuration instead of relying on script defaults
- pinning PyTorch and optional notebook dependencies explicitly
- improving seed handling and deterministic execution
- documenting expected dataset layout with sample metadata files
- adding automated experiment summaries and report generation
- expanding unit tests for data transforms and model I/O
- simplifying external image-encoder integration

---

If you use this repository for new experiments, a good workflow is to start from an existing config preset, override only the parameters you need, and keep each run self-contained under Hydra’s generated output directory.
