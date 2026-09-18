"""Overfit-a-batch diagnostic: the ``data.overfit`` split collapse + ``exp=overfit_batch``.

CPU-only, no data, no network. ``PipelineBuilder.apply_overfit_subset`` only reads
``config.data.overfit`` and ``self.load_config``, so it is exercised against a minimal stub
rather than a constructed builder (which would want the HDF5 dataset and a GPU).

The composition group asserts the two things that make the run a valid capacity test:
the transform pipeline is deterministic across epochs, and nothing stochastic or
regularising (dropout, weight decay) stands between the model and a zero loss. It also
pins the invariant that every OTHER experiment is left untouched by this feature.
"""

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import os

from src.training.pipeline_builder import PipelineBuilder

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "configs"))

# Transforms that redraw a fresh random value on every ``__getitem__``. Any of these in the
# overfit pipeline means the same sample is a different (src, tgt) pair each epoch.
STOCHASTIC_TRANSFORMS = {
    "AddIsotropicGaussianNoise",
    "AddRandomCenterCorrelatedRadialNoise",
    "DiscretizationNoise",
    "AddGaussianNoiseToFixations",
    "AddCurriculumNoise",
}


# ---------------------------------------------------------------------------
# Stub
# ---------------------------------------------------------------------------

class _FakeBuilder:
    """Carries only the two attributes ``apply_overfit_subset`` dereferences."""

    def __init__(self, config):
        self.config = config
        self.load_config = config.data.load


def _cfg(overfit=None, batch_size=8):
    d = {"data": {"load": {"batch_size": batch_size}}}
    if overfit is not None:
        d["data"]["overfit"] = overfit
    return OmegaConf.create(d)


def _apply(builder, train_idx, val_idx, test_idx):
    return PipelineBuilder.apply_overfit_subset(builder, train_idx, val_idx, test_idx)


@pytest.fixture
def splits():
    return torch.arange(100, 200), torch.arange(50), torch.arange(50, 80)


# ---------------------------------------------------------------------------
# Group 1 — the feature is inert unless explicitly enabled
# ---------------------------------------------------------------------------

def test_absent_overfit_block_is_a_pure_noop(splits):
    tr, va, te = splits
    a, b, c = _apply(_FakeBuilder(_cfg()), tr, va, te)
    assert a is tr and b is va and c is te


def test_disabled_overfit_block_is_a_pure_noop(splits):
    tr, va, te = splits
    builder = _FakeBuilder(_cfg({"enabled": False, "n_samples": 4}))
    a, b, c = _apply(builder, tr, va, te)
    assert a is tr and b is va and c is te


# ---------------------------------------------------------------------------
# Group 2 — the collapse itself
# ---------------------------------------------------------------------------

def test_collapses_to_first_n_training_rows(splits):
    tr, va, te = splits
    a, b, c = _apply(_FakeBuilder(_cfg({"enabled": True, "n_samples": 8})), tr, va, te)
    assert a.tolist() == list(range(100, 108))
    assert torch.equal(a, b) and torch.equal(a, c), "val/test must be the training rows"


def test_returned_splits_are_distinct_tensors(splits):
    """Aliasing would let a later in-place edit of one split mutate the others."""
    tr, va, te = splits
    a, b, c = _apply(_FakeBuilder(_cfg({"enabled": True, "n_samples": 8})), tr, va, te)
    assert b is not a and c is not a and b is not c


def test_n_samples_defaults_to_batch_size(splits):
    tr, va, te = splits
    builder = _FakeBuilder(_cfg({"enabled": True}, batch_size=5))
    a, _, _ = _apply(builder, tr, va, te)
    assert a.tolist() == list(range(100, 105))


def test_n_samples_larger_than_split_is_clamped(splits):
    tr, va, te = splits
    a, _, _ = _apply(_FakeBuilder(_cfg({"enabled": True, "n_samples": 999})), tr, va, te)
    assert a.numel() == tr.numel()


def test_empty_training_split_raises(splits):
    _, va, te = splits
    builder = _FakeBuilder(_cfg({"enabled": True, "n_samples": 8}))
    with pytest.raises(ValueError, match="training split is empty"):
        _apply(builder, torch.tensor([], dtype=torch.long), va, te)


def test_accepts_a_non_tensor_split(splits):
    """``make_splits`` returns tensors, but a reused ``split.pth`` need not."""
    _, va, te = splits
    builder = _FakeBuilder(_cfg({"enabled": True, "n_samples": 3}))
    a, _, _ = _apply(builder, [7, 8, 9, 10, 11], va, te)
    assert a.tolist() == [7, 8, 9]


# ---------------------------------------------------------------------------
# Group 3 — the exp=overfit_batch config composes into a valid capacity test
# ---------------------------------------------------------------------------

def _compose(exp):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="main", overrides=[f"exp={exp}"])


@pytest.fixture(scope="module")
def overfit_cfg():
    return _compose("overfit_batch")


def test_overfit_inherits_the_parallel_decoding_recipe(overfit_cfg):
    """The point of the test is to probe THIS recipe, so the inherited pieces must survive."""
    assert overfit_cfg.model.parallel_decoding is True
    assert overfit_cfg.model.image_encoder.type == "mask2former"
    assert overfit_cfg.model.image_encoder.precomputed is True
    assert overfit_cfg.data.load.use_precomputed_features is True
    assert overfit_cfg.model.image_adaptation.enabled is True
    assert overfit_cfg.model.image_adaptation.pretrained_adapter_path is not None
    assert overfit_cfg.training.use_scheduled_sampling is False


def test_overfit_enables_the_split_collapse(overfit_cfg):
    assert overfit_cfg.data.overfit.enabled is True
    assert overfit_cfg.data.overfit.n_samples == overfit_cfg.data.load.batch_size, \
        "n_samples must equal batch_size so the run is exactly one step per epoch"


def test_overfit_transform_pipeline_is_deterministic(overfit_cfg):
    """No stochastic transform, and the crop window pinned => (src, tgt) fixed per epoch."""
    used = set(overfit_cfg.data.transforms.transform_list)
    assert not (used & STOCHASTIC_TRANSFORMS), \
        f"stochastic transforms leak into the overfit run: {used & STOCHASTIC_TRANSFORMS}"
    assert overfit_cfg.data.transforms.ExtractRandomPeriod.random_offset is False


def test_overfit_has_no_regularisation(overfit_cfg):
    """Any live dropout or weight decay puts a floor under the achievable training loss."""
    assert overfit_cfg.training.weight_decay == 0
    for key, value in overfit_cfg.model.items():
        if "dropout" in key and isinstance(value, (int, float)):
            assert value == 0, f"model.{key} = {value} must be 0 in the overfit run"


def test_overfit_runs_combined_only_and_validates_every_epoch(overfit_cfg):
    assert list(overfit_cfg.training.Phases) == ["Combined"]
    assert overfit_cfg.training.validate is True
    assert overfit_cfg.training.val_interval == 1


def test_overfit_logs_to_wandb(overfit_cfg):
    """The W&B curves are the deliverable of this run; a disabled logger makes it unwatchable."""
    assert overfit_cfg.training.wandb.enabled is True
    assert overfit_cfg.training.wandb.group == "overfit-batch", \
        "keep the overfit runs in their own group so they don't pollute real training curves"
    assert overfit_cfg.training.log is True


def test_overfit_lr_schedule_spans_the_whole_run(overfit_cfg):
    """warmup/stable/decay are in EPOCHS (build_scheduler scales by steps_per_epoch); if they
    undershoot num_epochs the LR bottoms out early and the run stalls short of convergence."""
    s = overfit_cfg.scheduler
    assert s.warmup_steps + s.stable_steps + s.decay_steps == overfit_cfg.training.num_epochs
    assert overfit_cfg.training.Combined.epochs == overfit_cfg.training.num_epochs


# ---------------------------------------------------------------------------
# Group 4 — every other experiment is untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp", [
    "parallel_decoding_training",
    "image_adaptation_training",
    "path_model_training",
    "whole_model_pretraining",
])
def test_other_experiments_do_not_gain_an_overfit_block(exp):
    cfg = _compose(exp)
    assert cfg.data.get("overfit", None) is None, \
        f"exp={exp} must compose without a data.overfit block (apply_overfit_subset stays inert)"


def test_parallel_decoding_recipe_is_unchanged():
    """The overfit config must not have perturbed the run it is derived from."""
    cfg = _compose("parallel_decoding_training")
    assert list(cfg.training.Phases) == ["Combined", "FullFinetune"]
    assert cfg.data.load.batch_size == 128
    assert cfg.training.val_interval == 10
    assert cfg.data.transforms.ExtractRandomPeriod.random_offset is True
    assert set(cfg.data.transforms.transform_list) & STOCHASTIC_TRANSFORMS
