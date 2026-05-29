# Mission

## Problem Statement

Eye-tracking is a powerful tool for studying human visual attention, but high-quality lab-grade
eye-trackers are expensive and inaccessible to the broader research community. Webcam-based
alternatives such as WebGazer make gaze collection cheap and scalable, but they introduce
substantial noise: errors averaging ~4.17° of visual angle with a standard deviation of ~3.54°,
temporal correlations, elliptical bias, and discretisation artefacts. This noise degrades the
signal enough that the resulting gaze data cannot be directly used to study fine-grained
attentional behaviour (scanpaths).

The central question this project addresses is: **can we recover a clean scanpath — a precise
sequence of fixations with accurate spatial coordinates and durations — from the combination of
the stimulus image and a noisy WebGazer eye-tracking signal?**

The theoretical premise is that the two inputs are complementary:
- The **stimulus image** encodes the bottom-up attentional landscape: salient regions, object
  boundaries, semantic content — the scene-level cues that drive where any observer is likely
  to look.
- The **noisy gaze signal** encodes top-down information: the individual's personal biases,
  task-related attention, and viewing strategy. Even buried in WebGazer noise, this signal
  retains enough structure to disambiguate between competing attentional hypotheses that the
  image alone cannot resolve.

Combining both sources should allow a model to reconstruct the full attentional signal with
accuracy that neither input achieves alone.

## Goal

Train a deep learning model that takes a raw noisy WebGazer gaze trajectory and a stimulus
image as input and autoregressively outputs a clean scanpath — a variable-length sequence of
fixations, each with normalised (x, y) coordinates and duration.

## Scope

**In scope**
- Free-viewing paradigm on natural images.
- Source dataset: COCO FreeView (CocoFreeView), a large-scale scanpath dataset used as the
  ground-truth signal.
- Simulated WebGazer noise: a physics-informed noise pipeline (isotropic Gaussian, elliptical
  Gaussian, correlated radial noise with drifting center, mixture-of-Gaussians, discretisation)
  applied to CocoFreeView trajectories to produce paired noisy / clean training examples.
- Two model families: `PathModel` (gaze-only ablation baseline) and `MixerModel` (full
  image-conditioned architecture).
- Offline inference only; no real-time or streaming requirements.
- Reproducibility: all experiments configured via Hydra; results must be re-runnable from a
  config file and a random seed.

**Out of scope**
- Real-time gaze correction or browser-side deployment.
- Saliency map generation as a primary output.
- Datasets other than CocoFreeView (in the main experimental track).
- Other webcam eye-trackers beyond WebGazer.
- Supervised learning from real paired WebGazer / lab-grade data (no such large dataset
  currently exists).

## Success Criteria

**Quantitative**
- The model outperforms a gaze-only baseline (PathModel without image features) on held-out
  scanpaths from CocoFreeView, measured by Euclidean fixation coordinate error and duration
  error.
- Denoising phase: per-sample coordinate MSE on the noisy→clean task lower than the input
  noise level.
- Scanpath prediction: DTW distance and/or multi-match score on the held-out split
  competitive with or better than state-of-the-art scanpath prediction models that receive
  clean input.
- End-of-sequence classification accuracy ≥ a reasonable threshold so that autoregressively
  generated scanpaths have realistic lengths.

**Qualitative**
- Predicted scanpaths on sample stimuli are visually plausible: fixations cluster on
  semantically meaningful regions, scanpath length is realistic, and the trajectory does not
  collapse to a single point or diverge off the image.

**Practical deliverable**
- A trained `MixerModel` checkpoint that can be loaded and run inference on a new
  (image, noisy-gaze) pair.
- A documented Hydra experiment config that fully specifies the winning training run so that
  results can be reproduced on a fresh machine.

## Motivation / Context

Scanpath modelling sits at the intersection of computer vision and cognitive neuroscience.
Understanding where people look and in what order reveals how visual attention is deployed, with
applications in interface design, clinical diagnosis, advertising research, and more. Prior work
on scanpath prediction (e.g., IORE, ScanDy, Gazeformer) typically assumes clean lab-grade
eye-tracking input or generates scanpaths purely from the image. This project explores a
different niche: leveraging degraded but real (or realistically simulated) gaze data as a
personalisation signal that constrains the image-driven prediction.

The choice of WebGazer as the noise model is deliberate. WebGazer is the most widely used
webcam eye-tracker in the psychology community, partly because it is embedded in jsPsych — a
dominant open-source framework for online behavioural experiments. A successful denoising and
scanpath recovery method for WebGazer output would immediately benefit a large body of existing
and future psychological research that currently cannot exploit its gaze data for fine-grained
attentional analysis.

The results are intended for publication at a top-tier computer vision venue (CVPR, ICCV, ECCV,
or equivalent journal). This sets the bar for experimental rigour: ablation studies, comparison
against baselines, and full reproducibility are not optional.
