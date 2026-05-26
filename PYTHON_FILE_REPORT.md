# Python File Report

This report focuses on the Python files that carry most of the project logic. I treated "interesting" as files that define the data pipeline, model architectures, training loop, evaluation flow, preprocessing, or analysis tooling; tiny glue files, temporary scripts, and very small helpers are included only when they say something useful about how the repository is used.

## Entry Scripts

### `train.py`

`train.py` is the Hydra entrypoint for training. Its job is intentionally small: it resolves the Hydra output folder for the current run, injects paths for `metrics.json`, `model.pth`, `split.pth`, and optionally the inference-recording directory into the config, and then hands control to `PipelineBuilder` plus the main `train()` routine. In other words, this file does not define training behavior itself; it defines how a configured experiment is materialized into a run-specific workspace.

### `preprocessing.py`

`preprocessing.py` is the offline dataset builder that converts event-level scanpaths from `CocoFreeView` into the HDF5 cache used later during training. For each sample it synthesizes a continuous gaze trajectory, downsamples it, filters out unusable sequences based on duration and fixation constraints, and stores both the clean event representation and the generated temporal signals. The file is important because it defines the exact training substrate: if this preprocessing changes, the whole model sees a different problem.

### `best_models.py`

`best_models.py` is a small reporting script that scans the `outputs/` tree, loads metric JSON files, and ranks runs by the best coordinate error or duration error they achieved. It is not part of the training runtime, but it is clearly part of the research workflow: it gives a quick way to sift through many Hydra runs and find promising checkpoints, including a filtered view that keeps only models with a minimally acceptable positive recall.

## Data Layer

### `src/data/parsers.py`

`parsers.py` defines the raw dataset interfaces. The main class, `CocoFreeView`, loads the COCO FreeView fixation JSON, resolves each scanpath to its underlying image path, builds a pandas-backed table, exposes helpers to retrieve scanpaths and images, and supports subject/stimulus-aware splitting. The file also hardcodes the project's image geometry assumptions, including the downscaled working resolution and the pixel-to-angle conversion used throughout the simulation and noise code. The secondary `OurDataset` class looks like an auxiliary loader for a custom CSV-based dataset, but the real backbone here is `CocoFreeView`, because nearly every later stage assumes its indexing, path lookup, and split logic.

### `src/data/datasets.py`

`datasets.py` is one of the central files in the repository because it defines the objects that training actually consumes. The older `FreeViewBatch` class streams from HDF5 in a batch-oriented way and appears to reflect an earlier optimization path, but the more important class is `FreeViewInMemory`, which loads the entire HDF5 cache into RAM and then applies a configurable transform chain sample by sample. That is the dataset object the current pipeline builder instantiates for path data.

The file also contains the temporal cropping logic in `extract_random_period`, which is more subtle than a normal window crop. It does not just slice a gaze signal; it tries to align the crop with full fixation events so that the paired target scanpath contains only complete fixations inside the selected noisy interval. That makes this file conceptually important because it is where the model's input-output pairing is enforced.

The rest of the file handles batching and image coupling. `seq2seq_padded_collate_fn` turns variable-length trajectories into padded tensors with source masks, decoder masks, fixation lengths, and optional auxiliary channels like `clean_x`, `in_tgt`, `heatmaps`, and sample indices. `FreeViewImgDataset`, `DeduplicatedMemoryDataset`, and `CoupledDataloader` then extend the path-only pipeline into an image-conditioned one by caching unique resized images and zipping them with path samples. Together these pieces define the exact contract between storage, augmentation, and model forward passes.

### `src/data/transforms.py`

`transforms.py` is the configurable augmentation and normalization layer for the in-memory dataset. Each class takes the project's dictionary-shaped sample format and mutates or enriches it, which lets `PipelineBuilder` assemble a transform pipeline directly from Hydra config names. The file covers random temporal window extraction, coordinate and time normalization, duration normalization by either log-statistics or a learned quantile transform, and preservation of clean inputs for denoising supervision.

A second major responsibility is noise and supervision shaping. `AddRandomCenterCorrelatedRadialNoise` injects structured synthetic corruption into gaze coordinates, `DiscretizationNoise` simulates raster or device-level rounding, `AddGaussianNoiseToFixations` perturbs decoder inputs, and `AddCurriculumNoise` gradually interpolates between clean and noisy trajectories as training progresses. This means the file does not just preprocess data; it actively defines the difficulty schedule and the denoising task that the models are asked to solve.

The last important branch is target reformulation. `AddHeatmaps` converts fixation coordinates into Gaussian spatial targets and also knows how to invert heatmap predictions back into coordinates for evaluation. That makes this file a bridge between multiple output formulations: direct regression, coordinate-plus-duration heads, and heatmap-based objectives all meet here through a shared transform contract.

## Preprocessing and Noise

### `src/preprocess/simulation.py`

`simulation.py` converts sparse scanpath annotations into dense synthetic gaze trajectories. It estimates saccade amplitudes and durations, generates saccadic interpolation curves, adds fixation-level microsaccadic noise, constructs a fixation mask over a sampled timeline, and finally synthesizes an `(x, y, t)` gaze stream that can later be downsampled. The code is important because the repository is not training directly on raw fixations alone; it first constructs a time-series view of gaze dynamics, and this file defines that generative assumption.

### `src/preprocess/noise.py`

`noise.py` contains the corruption models used to make the gaze stream look like noisy eye tracking. It includes simpler Gaussian and elliptical perturbations, learned Gaussian-mixture-based noise, and the more distinctive correlated radial-noise pipeline that pushes samples away from a drifting center with controllable temporal correlation. Several functions are Numba-compiled, which suggests this file became performance-sensitive once preprocessing or augmentation started scaling to large datasets. Conceptually, this module is what turns the project from plain scanpath prediction into "scanpath prediction from noisy gaze."

## Training System

### `src/training/pipeline_builder.py`

`pipeline_builder.py` is arguably the main orchestration file in the whole repository. It translates Hydra config into concrete Python objects: datasets, transforms, data splits, image preprocessing, dataloaders, models, losses, optimizers, schedulers, phased training plans, scheduled sampling, curriculum noise, denoise-dropout scheduling, and inference recording. Most of the project's flexibility lives here, because nearly every experiment knob eventually gets resolved by this builder.

The dataset side of the builder is especially important because it fuses together three layers that are defined elsewhere: raw HDF5 path data, optional image data, and the transform stack. It decides whether the experiment is path-only or image-conditioned, whether to use preprocessed data from scratch or a locally mirrored scratch directory, and whether the train/val/test split should be random, disjoint by subject and stimulus, or disjoint by stimulus only. That means reproducibility and experimental validity depend heavily on this file.

The model side is equally central. `PipelineBuilder` instantiates either `PathModel` or `MixerModel`, optionally wraps a DINO image encoder, resolves automatic loading of the best denoising encoder from past runs, and chooses the appropriate loss and scheduling mechanisms based on the training phases. In practical terms, if someone wants to understand how a Hydra config becomes a full experiment, this is the single best file to read first.

### `src/training/pipeline.py`

`pipeline.py` implements the main training loop over the objects prepared by `PipelineBuilder`. It supports multi-phase training such as denoise-only, fixation-only, and combined modes; runs validation on a configured interval; logs metrics; steps batch-wise or epoch-wise schedulers; saves split files and best checkpoints; and optionally captures inference traces at selected batches. The file is less configurable than `pipeline_builder.py`, but it is the place where all the moving parts are actually composed into the lifecycle of an experiment.

### `src/training/training_utils.py`

`training_utils.py` holds the runtime utilities that make the training loop measurable and more advanced than a plain epoch iterator. `MetricsStorage` aggregates per-batch losses, computes denormalized evaluation metrics after inverting transforms, tracks the decisive validation metric, and writes metrics to disk. The `validate()` function is also here, and it is more than a loss check: it runs inverse transforms, computes coordinate and duration errors, evaluates end-of-sequence classification, and optionally integrates with the inference recorder.

The second half of the file contains training-control machinery. Besides data-to-device helpers and an older standalone loss helper, it defines a custom warmup-stable-decay LR schedule, an inverted-sigmoid-based scheduled-sampling mechanism, cosine schedules for denoising curricula, and a denoise-dropout scheduler that can progressively alter model behavior during training. This file therefore sits between the clean abstraction of the builder and the concrete loop in `pipeline.py`, supplying most of the dynamic policies that make experiments nontrivial.

### `src/training/inference_recorder.py`

`inference_recorder.py` is a compact but high-leverage debugging and analysis utility. It can attach itself to every module in a model, open a recording context for a specific batch, store selected batch tensors and outputs under stable names, and collect intermediate activations from modules that explicitly call `record_module_value()`. Each capture is serialized to a per-batch `.pt` payload with metadata such as epoch, split, phase, and global step.

This file matters because it quietly upgrades the codebase from "train and inspect final metrics" to "instrument the internal behavior of attention and decoding blocks." The bigger model modules in `blocks.py` and the training loop are clearly written to cooperate with this recorder, so this is not an isolated utility; it is a built-in observability layer for model analysis notebooks.

### `src/training/weights_scheduler.py`

`weights_scheduler.py` is a small helper that adjusts loss-related weights over epochs. It is not a large file, but it exists because some experiments change the emphasis between components over time, and this object gives the training loop a clean interface for doing that without hardcoding the schedule into the main epoch logic.

## Model Layer

### `src/model/path_model.py`

`path_model.py` defines the simpler baseline architecture: a transformer encoder-decoder that operates only on temporal eye-tracking inputs and predicts scanpath outputs autoregressively. It supports either a direct linear input projection or a Gaussian Fourier-style encoding path, adds sinusoidal order embeddings, prepends a learned start token at decode time, and produces either full `(x, y, duration)` regression outputs or separated coordinate and duration heads plus an end-of-sequence classifier. This file is useful because it shows the minimal modeling story in the repo without the image-conditioning complexity.

### `src/model/mixer_model.py`

`mixer_model.py` defines the flagship architecture of the project. It extends the path-only model into a multimodal system that can encode noisy gaze trajectories, attend over image features, optionally denoise the trajectory in one branch while predicting fixations in another, and switch trainable submodules depending on whether the current phase is `Denoise`, `Fixation`, or `Combined`. The constructor is large because the model is genuinely modular: different input encoders, different decoder variants, several head types, optional DINO features, optional RoPE, feature enhancement stacks, adapters, deformable attention, scheduled sampling, and key-value caching all get wired here.

A key idea in this file is the split between `denoise_modules` and `fixation_modules`. The model keeps explicit lists of which submodules belong to each training objective, and `set_phase()` toggles `requires_grad` on those groups. That gives the codebase a clean way to do staged pretraining or partial weight loading, including the ability to load only encoder-side parameters from a denoise-pretrained checkpoint. This is one of the clearest places where the repository's research agenda shows up in the design.

The forward path is also unusually rich. `encode()` processes the noisy eye trajectory, optionally fuses in image encoder tokens, optionally enhances image features with bidirectional attention, and caches the encoded states. `decode_fixation()` then performs autoregressive decoding over both path memory and image memory, with several alternative output heads ranging from direct regression to heatmap generation. `decode_denoise()` gives a parallel denoising target over the encoded trajectory itself. If someone wants to understand the repository's main modeling claim, this is the file to study in depth.

### `src/model/blocks.py`

`blocks.py` is the repository's neural-network toolbox. It starts with reusable pieces such as an `MLP`, RoPE helper functions, and a custom `MultiHeadedAttention` implementation that supports self-attention, cross-attention, causal masking, optional key-value caching, and activation recording for later inspection. On top of that it defines standard transformer encoder and decoder blocks as well as the custom double-input decoder used by the multimodal model.

The middle of the file adds higher-level fusion and prediction components. `FeatureEnhancer` performs mutual interaction between path and image streams; `ArgMaxRegressor` turns image-token similarity into a soft spatial coordinate; `LearnableCoordinateDropout` masks token embeddings with a learned replacement vector; `GatedFusion` mixes alternative feature sources; and `TrajectoryHeatmapGenerator` converts trajectory tokens into spatial heatmaps over visual features. These are not generic utilities in the abstract; they are the concrete mechanisms that let the project try multiple ways of relating gaze and image evidence.

The last section introduces deformable attention for image-conditioned decoding. `DeformableAttention`, `DeformableDecoder`, and `DeformableDoubleInputDecoder` replace full dense spatial attention with learned sampling offsets around reference points, and they also emit internal tensors to the inference recorder when instrumentation is active. This file is therefore both the architectural foundation of the project and the place where its more experimental decoding ideas live.

### `src/model/loss_functions.py`

`loss_functions.py` collects the objective functions for the different output formats. It supports a single combined regression head (`EntireRegLossFunction`), a separated coordinate/duration formulation (`SeparatedRegLossFunction`), a denoising loss, a weighted combined objective for multi-phase training, and a heatmap-style focal loss (`PenaltyReducedFocalLoss`). It also includes custom end-of-sequence classification variants using either BCE-with-logits or a softmax-over-time interpretation. The file matters because the repository does not optimize just one target; it switches loss semantics depending on the architecture and experiment phase.

### `src/model/model_io.py`

`model_io.py` is the persistence and reload layer for experiments. It saves checkpoints with either model-only or full optimizer/scheduler state, reloads them permissively with missing/unexpected-key reporting, reconstructs a pipeline from the stored Hydra config, restores saved data splits, and provides helpers to load whole runs for evaluation or notebook analysis. It also contains logic to strip `_orig_mod.` prefixes from compiled-model checkpoints, which is a practical compatibility fix for `torch.compile` workflows.

### `src/model/dino_wrapper.py`

`dino_wrapper.py` is a thin adapter around an external DINOv3 image encoder. Its role is to initialize the visual backbone from a local repository and weights path, optionally freeze it, and expose its embedding dimension and token outputs in a form the `MixerModel` can consume. The file is small, but it is the key abstraction boundary between this project's code and the external vision backbone.

### `src/model/pos_encoders.py`

`pos_encoders.py` contains the coordinate and order encoders used across the models. It provides classic sinusoidal positional encoding for sequence order, a NeRF-style Fourier encoder for scalar or vector inputs, and a Gaussian random-feature encoder that can either work independently or share a positional basis across eye, fixation, and image coordinates. This file is important because the repository treats coordinates and durations as signals that need expressive embedding, not just raw numeric inputs.

### `src/model/rope_positional_embeddings.py`

`rope_positional_embeddings.py` implements a standalone 2D rotary positional embedding module adapted from the DINOv3 style. It creates embeddings either for trajectory coordinates or image patch grids, supports coordinate normalization and geometric augmentations, and returns sin/cos tensors that the custom attention layers in `blocks.py` apply to queries and keys. In practice it is the project's bridge between learned multimodal attention and geometry-aware spatial encoding.

## Evaluation and Analysis

### `src/eval/eval_metrics.py`

`eval_metrics.py` holds the scalar metrics used during validation: end-token target construction, accuracy, precision, recall, regression error over coordinates and durations, and denoising error. It is intentionally small and low-level, acting as the arithmetic backend for the richer validation flow in `training_utils.py`.

### `src/eval/eval_utils.py`

`eval_utils.py` is a broader evaluation toolkit for both formal metrics and exploratory analysis. It plots training curves, gathers best-run summaries from saved metric files, inverts dataset transforms so predictions can be compared in original units, computes amplitude and angle statistics over gaze trajectories, and exposes autoregressive evaluation helpers for decoding behavior. The file is less central than the builder or models, but it is clearly where the project's research-style postprocessing and interpretation workflows live.

### `src/eval/vis_scanpath.py`

`vis_scanpath.py` is the qualitative-visualization companion to the numeric evaluation code. It draws scanpaths on images with Matplotlib, renders videos with moving gaze points and trails using OpenCV, and supports overlays for comparing two gaze streams over the same image. This is the file that turns predictions and recorded trajectories into artifacts a human can inspect frame by frame.

## Notebook Utilities

### `src/notebooks/review_utils.py`

`review_utils.py` is a large notebook support module for inspecting saved inference-recorder payloads and training metrics. It resolves payload files, loads image banks, summarizes captures, extracts and reshapes attention-related activations, reconstructs decoder reference points, and provides a broad set of plotting helpers for scanpath overlays, attention heatmaps, deformable-attention sampling locations, decoder query profiles, and run-to-run metric comparisons. It is effectively the analysis frontend for the recorder system.

### `src/notebooks/generate_review_notebooks.py`

`generate_review_notebooks.py` programmatically builds notebook files with prefilled cells for recorder inspection and metric review. Instead of keeping static notebooks in sync by hand, this script creates them from code, which is a useful sign that the project is trying to standardize model-review workflows.

### `src/notebooks/eval_batch.py`

`eval_batch.py` is a notebook-oriented batch evaluation script that pulls together model loading, loss computation, metric utilities, transform inversion, and scanpath plotting. It looks like a hands-on experimentation file for inspecting individual batches, decoder outputs, and qualitative behavior rather than a reusable production module.

## Smaller Utilities and Tests

### `tests/test_inference_recorder.py`

This test file validates the recorder infrastructure with a tiny dummy model and also exercises attention modules that expose internal captures. Its presence is reassuring because the recorder touches many modules indirectly, and this test anchors at least part of that instrumentation contract.

### `test.py` and `test_pipeline.py`

These two files look like ad hoc local inspection or experimentation scripts rather than formal tests. `test.py` checks data/image pairing and model-loading flows, while `test_pipeline.py` appears to be an older or alternate Hydra training entrypoint that predates the cleaner `train.py` plus `src/training/pipeline.py` arrangement.

### `rename_logs.py` and `temp.py`

These are maintenance scripts. `rename_logs.py` renames output folders based on date-like patterns, while `temp.py` moves logs around on disk. They are not part of the learning pipeline itself, but they show the repository has accumulated some practical housekeeping around Hydra-style experiment outputs.

## Overall Reading Order

If you want the fastest route to understanding the repository, I would read the files in this order: `src/training/pipeline_builder.py`, `src/data/datasets.py`, `src/data/transforms.py`, `src/model/mixer_model.py`, `src/model/blocks.py`, `src/training/pipeline.py`, `src/training/training_utils.py`, and then `src/preprocess/simulation.py` plus `src/preprocess/noise.py`. That sequence follows the real lifecycle of an experiment: cached data, runtime transforms, model construction, training behavior, and the synthetic-noise assumptions that define the task.
