"""Tests for the EVE pipeline integration (Groups 5 and 6 from validation.md).

Group 5: has_precomputed_clean_x flag in PipelineBuilder._build_transforms()
Group 6: configs/data/eve.yaml correctness

Groups 7 and 8 (full stack and COCO regression) require a real bundle/dataset
and are tested manually or via separate integration fixtures.
"""
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

# Ensure project root is on the path when running tests directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.transforms import Normalize, SaveCleanX
from src.training.pipeline_builder import PipelineBuilder


# ---------------------------------------------------------------------------
# Minimal config helpers
# ---------------------------------------------------------------------------

def _minimal_base_config(**data_overrides):
    """Build a minimal OmegaConf config sufficient for _build_transforms()."""
    base = OmegaConf.create({
        "model": {"device": "cpu"},
        "data": {
            "load": {"img_size": 64},
            "transforms": {
                "transform_list": [],
            },
        },
    })
    if data_overrides:
        data_patch = OmegaConf.create({"data": data_overrides})
        base = OmegaConf.merge(base, data_patch)
    return base


def _make_builder(data_cfg: dict) -> PipelineBuilder:
    cfg = _minimal_base_config(**data_cfg)
    return PipelineBuilder(cfg)


def _normalize_keys(transform_list) -> list[str]:
    """Return the key attribute of every Normalize transform in the list."""
    return [t.key for t in transform_list if isinstance(t, Normalize)]


# ---------------------------------------------------------------------------
# Group 5 — has_precomputed_clean_x flag
# ---------------------------------------------------------------------------

class TestHasPrecomputedCleanX:
    _COORD_TRANSFORM = {
        "transform_list": ["NormalizeCoords", "NormalizeTime", "NormalizeFixationCoords", "NormalizeDuration"],
        "NormalizeCoords": {"key": "x", "mode": "coords", "image_H": 1080, "image_W": 1920},
        "NormalizeTime": {"key": "x", "mode": "time", "period_duration": 3000},
        "NormalizeFixationCoords": {"key": "y", "mode": "coords", "image_H": 1080, "image_W": 1920},
        "NormalizeDuration": {"key": "y", "mode": "time", "period_duration": 1200},
    }

    def test_flag_true_produces_clean_x_normalizers(self):
        builder = _make_builder({
            "has_precomputed_clean_x": True,
            "transforms": self._COORD_TRANSFORM,
        })
        transforms = builder._build_transforms()
        clean_x_normalizers = [t for t in transforms if isinstance(t, Normalize) and t.key == "clean_x"]
        assert len(clean_x_normalizers) == 2, (
            f"Expected 2 clean_x Normalize transforms, got {len(clean_x_normalizers)}"
        )

    def test_flag_false_no_clean_x_normalizers(self):
        builder = _make_builder({
            "has_precomputed_clean_x": False,
            "transforms": self._COORD_TRANSFORM,
        })
        transforms = builder._build_transforms()
        clean_x_normalizers = [t for t in transforms if isinstance(t, Normalize) and t.key == "clean_x"]
        assert len(clean_x_normalizers) == 0

    def test_flag_missing_no_clean_x_normalizers(self):
        """Omitting has_precomputed_clean_x defaults to False (COCO path unaffected)."""
        builder = _make_builder({"transforms": self._COORD_TRANSFORM})
        transforms = builder._build_transforms()
        clean_x_normalizers = [t for t in transforms if isinstance(t, Normalize) and t.key == "clean_x"]
        assert len(clean_x_normalizers) == 0

    def test_save_clean_x_in_list_still_produces_clean_x_normalizers(self):
        """COCO regression: SaveCleanX in transform_list must still trigger clean_x normalisation."""
        transforms_cfg = {
            "transform_list": [
                "SaveCleanX", "NormalizeCoords", "NormalizeTime",
                "NormalizeFixationCoords", "NormalizeDuration",
            ],
            "NormalizeCoords": {"key": "x", "mode": "coords", "image_H": 320, "image_W": 512},
            "NormalizeTime": {"key": "x", "mode": "time", "period_duration": 2600},
            "NormalizeFixationCoords": {"key": "y", "mode": "coords", "image_H": 320, "image_W": 512},
            "NormalizeDuration": {"key": "y", "mode": "time", "period_duration": 1200},
        }
        builder = _make_builder({"transforms": transforms_cfg})
        transforms = builder._build_transforms()
        clean_x_normalizers = [t for t in transforms if isinstance(t, Normalize) and t.key == "clean_x"]
        assert len(clean_x_normalizers) == 2

    def test_save_clean_x_and_flag_true_no_duplicates(self):
        """Both SaveCleanX and has_precomputed_clean_x=True must not duplicate normalizers."""
        transforms_cfg = {
            "transform_list": [
                "SaveCleanX", "NormalizeCoords", "NormalizeTime",
                "NormalizeFixationCoords", "NormalizeDuration",
            ],
            "NormalizeCoords": {"key": "x", "mode": "coords", "image_H": 1080, "image_W": 1920},
            "NormalizeTime": {"key": "x", "mode": "time", "period_duration": 3000},
            "NormalizeFixationCoords": {"key": "y", "mode": "coords", "image_H": 1080, "image_W": 1920},
            "NormalizeDuration": {"key": "y", "mode": "time", "period_duration": 1200},
        }
        builder = _make_builder({
            "has_precomputed_clean_x": True,
            "transforms": transforms_cfg,
        })
        transforms = builder._build_transforms()
        clean_x_normalizers = [t for t in transforms if isinstance(t, Normalize) and t.key == "clean_x"]
        assert len(clean_x_normalizers) == 2, (
            f"Expected exactly 2 clean_x normalizers (no duplicates), got {len(clean_x_normalizers)}"
        )


# ---------------------------------------------------------------------------
# Group 6 — configs/data/eve.yaml correctness
# ---------------------------------------------------------------------------

EVE_YAML = ROOT / "configs" / "data" / "eve.yaml"


@pytest.fixture(scope="module")
def eve_cfg():
    return OmegaConf.load(EVE_YAML)


class TestEveYaml:
    def test_dataset_type(self, eve_cfg):
        assert eve_cfg.dataset_type == "eve"

    def test_has_precomputed_clean_x(self, eve_cfg):
        assert eve_cfg.has_precomputed_clean_x is True

    def test_no_extract_random_period(self, eve_cfg):
        assert "ExtractRandomPeriod" not in eve_cfg.transforms.transform_list

    def test_no_save_clean_x(self, eve_cfg):
        assert "SaveCleanX" not in eve_cfg.transforms.transform_list

    def test_no_noise_transforms(self, eve_cfg):
        noise_transforms = {
            "AddIsotropicGaussianNoise",
            "AddRandomCenterCorrelatedRadialNoise",
            "DiscretizationNoise",
        }
        for t in noise_transforms:
            assert t not in eve_cfg.transforms.transform_list, f"{t} must not appear in EVE transform list"

    def test_normalize_coords_image_size(self, eve_cfg):
        assert eve_cfg.transforms.NormalizeCoords.image_W == 1920
        assert eve_cfg.transforms.NormalizeCoords.image_H == 1080

    def test_normalize_time_period(self, eve_cfg):
        assert eve_cfg.transforms.NormalizeTime.period_duration == 3000

    def test_normalize_duration_period(self, eve_cfg):
        assert eve_cfg.transforms.NormalizeDuration.period_duration == 1200
