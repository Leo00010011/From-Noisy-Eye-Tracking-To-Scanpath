"""Tests for the EVE head-to-head prediction contract (src/eval/prediction_contract.py).

Synthetic tests need no data. The real-data group is skipped unless the copied
ground truth (data/eve_bridge/) and the local EVE bundle are present.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import seq2seq_padded_collate_fn
from src.data.eve_real_noise import EyeNetGazeCache, EveRealNoiseDataset, EveRealNoiseImgDataset
from src.eval import prediction_contract as pc
from src.eval.eval_utils import invert_transforms_fixations
from src.training.pipeline_builder import PipelineBuilder
from test_eve_real_noise import FakeBundle, _write_csv

GT_PATH = ROOT / "data" / "eve_bridge" / "fixations.json"
MAP_PATH = ROOT / "data" / "eve_bridge" / "subject_id_map.json"
BUNDLE_DIR = ROOT.parent / "eve_shared" / "EveDataset" / "bundle"
_HAS_REAL = GT_PATH.exists() and MAP_PATH.exists() and (BUNDLE_DIR / "bundle.h5").exists()


def _cell(name="MIT-a.jpg", subject=3, eve="train05"):
    return pc.Cell(name, subject, eve)


# ── Group 1 — input resolution ─────────────────────────────────────────────────

class TestInputs:
    def test_gt_hash_guard(self, tmp_path):
        p = tmp_path / "fixations.json"
        p.write_text("[]")
        with pytest.raises(ValueError, match="wrong fixations.json"):
            pc.load_ground_truth(p)
        assert pc.load_ground_truth(p, expected_sha256=None) == []

    def test_subject_map_int_keys(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"to_dense": {"train01": 0}, "to_eve": {"0": "train01", "25": "train30"}}))
        assert pc.load_subject_map(p) == {0: "train01", 25: "train30"}

    def test_cells_only_test_split(self):
        gt = [{"name": "a.jpg", "subject": 0, "split": "test"},
              {"name": "a.jpg", "subject": 1, "split": "train"},
              {"name": "b.jpg", "subject": 1, "split": "test"}]
        cells = pc.contract_cells(gt, {0: "train01", 1: "train02"})
        assert [c.key for c in cells] == [("a.jpg", 0), ("b.jpg", 1)]
        assert cells[1].subject_eve == "train02" and cells[1].stimulus_name == "b"

    def test_cells_duplicate_raises(self):
        gt = [{"name": "a.jpg", "subject": 0, "split": "test"}] * 2
        with pytest.raises(ValueError, match="duplicate"):
            pc.contract_cells(gt, {0: "train01"})

    def test_resolve_exp_keys(self):
        df = pd.DataFrame({"exp_key": ["train05_s1", "train06_s1", "train05_s2"],
                           "subject": ["train05", "train06", "train05"],
                           "stimulus_name": ["MIT-a", "MIT-a", "MIT-b"]})
        cells = [_cell("MIT-a.jpg", 3, "train05"), _cell("MIT-a.jpg", 4, "train06")]
        assert pc.resolve_exp_keys(cells, df) == {("MIT-a.jpg", 3): "train05_s1",
                                                  ("MIT-a.jpg", 4): "train06_s1"}

    def test_resolve_missing_or_ambiguous_raises(self):
        df = pd.DataFrame({"exp_key": ["x1", "x2"], "subject": ["train05"] * 2,
                           "stimulus_name": ["MIT-a"] * 2})
        with pytest.raises(ValueError, match="exactly one"):
            pc.resolve_exp_keys([_cell("MIT-a.jpg")], df)          # ambiguous
        with pytest.raises(ValueError, match="exactly one"):
            pc.resolve_exp_keys([_cell("MIT-z.jpg")], df)          # missing


# ── Group 2 — output conversion ────────────────────────────────────────────────

class TestConversion:
    def test_predicted_length(self):
        assert pc.predicted_length(np.array([-5.0, -5.0, 3.0, 3.0])) == 2
        assert pc.predicted_length(np.array([4.0, -5.0])) == 0
        assert pc.predicted_length(np.full(21, -5.0)) == 21
        assert pc.predicted_length(np.array([[-1.0], [0.0], [0.1]])) == 2   # 0.0 is not > 0.5

    def test_screen_to_metric_non_uniform(self):
        x, y = pc.screen_to_metric(np.array([1920.0, 960.0, -10.0]), np.array([1080.0, 540.0, 2000.0]))
        np.testing.assert_allclose(x, [512.0, 256.0, 0.0])
        np.testing.assert_allclose(y, [384.0, 192.0, 384.0])

    def test_make_record_types_and_trim(self):
        sp = np.array([[1920.0, 1080.0, 212.4], [375.0, 281.25, 305.6], [0, 0, -3.0], [9, 9, 9]],
                      dtype=np.float32)
        rec = pc.make_record(_cell(), "train05_s1", sp, length=3)
        assert rec["X"] == [512.0, 100.0, 0.0] and rec["Y"] == [384.0, 100.0, 0.0]
        assert rec["T"] == [212, 306, 0]
        assert all(type(v) is float for v in rec["X"] + rec["Y"])
        assert all(type(t) is int for t in rec["T"])
        assert type(rec["subject"]) is int
        json.dumps(rec)   # no numpy scalars

    def test_make_record_caps_length(self):
        sp = np.full((21, 3), 100.0)
        assert len(pc.make_record(_cell(), "k", sp, length=21)["X"]) == pc.MAX_LENGTH

    def test_make_record_empty(self):
        rec = pc.make_record(_cell(), "k", None, 0, status="no_eyenet_gaze")
        assert rec["X"] == rec["Y"] == rec["T"] == [] and rec["status"] == "no_eyenet_gaze"
        assert pc.make_record(_cell(), "k", np.ones((3, 3)), 0)["X"] == []

    def test_make_record_nonfinite_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            pc.make_record(_cell(), "k", np.array([[np.nan, 1, 1]]), 1)

    def test_eve_real_transform_roundtrip(self):
        """Known screen-px fixations → eve_real.yaml forward → inverse → metric space.

        Guards the in-range-but-wrong-space failure the scorer cannot catch: the
        model's normalized output must invert to 1920×1080 trial pixels.
        """
        real = OmegaConf.load(ROOT / "configs" / "data" / "eve_real.yaml")
        cfg = OmegaConf.create({"data": OmegaConf.to_container(real, resolve=True),
                                "model": {"device": "cpu"}})
        transforms = PipelineBuilder(cfg)._build_transforms()
        y_px = np.array([[1440.0, 480.0], [270.0, 810.0], [300.0, 600.0]])   # rows x, y, dur
        x = np.array([[100.0, 200.0], [100.0, 200.0], [0.0, 33.3]])
        inp = {"x": x.copy(), "y": y_px.copy(), "fixation_mask": np.zeros(2, np.uint8)}
        for t in transforms:
            inp = t(inp)
        y_norm = torch.as_tensor(np.asarray(inp["y"]), dtype=torch.float32).T.unsqueeze(0)  # (1,2,3)
        assert 0.0 <= float(y_norm[..., :2].min()) and float(y_norm[..., :2].max()) <= 1.0

        pred = torch.cat([y_norm, torch.zeros(1, 1, 3)], dim=1)    # K = N + 1 decode steps
        tgt_mask = torch.ones(1, 3, dtype=torch.bool)
        _, out = invert_transforms_fixations(
            {"tgt": y_norm.clone(), "tgt_mask": tgt_mask}, {"reg": pred}, transforms)
        np.testing.assert_allclose(out["reg"][0, :2].numpy(), y_px.T, rtol=1e-4, atol=1e-2)

        rec = pc.make_record(_cell(), "k", out["reg"][0].numpy(), length=2)
        np.testing.assert_allclose(rec["X"], [1440 / 3.75, 480 / 3.75], rtol=1e-5)
        np.testing.assert_allclose(rec["Y"], [270 / 2.8125, 810 / 2.8125], rtol=1e-5)
        assert rec["T"] == [300, 600]


# ── Group 3 — validation ───────────────────────────────────────────────────────

def _records(cells, n=2):
    return [{"name": c.name, "subject": c.subject, "X": [10.0] * n, "Y": [20.0] * n, "T": [100] * n}
            for c in cells]


class TestValidation:
    cells = [_cell("a.jpg", 0), _cell("b.jpg", 1), _cell("c.jpg", 2)]

    def test_ok_and_stats(self):
        recs = _records(self.cells)
        recs[0].update(X=[], Y=[], T=[])
        recs[1].update(X=[511.0], Y=[383.0], T=[900])
        s = pc.validate_records(recs, self.cells)
        assert s["n_records"] == 3 and s["n_empty"] == 1 and s["n_short"] == 2
        assert s["x_range"] == (10.0, 511.0) and s["t_range"] == (100, 900)
        assert pc.format_stats("p.json", s).startswith("OK p.json: 3 records")

    @pytest.mark.parametrize("mutate, msg", [
        (lambda r: r.update(subject=np.int64(0)), "JSON int"),
        (lambda r: r.update(subject=0.0), "JSON int"),
        (lambda r: r.update(T=[100.0, 100.0]), "int ms"),
        (lambda r: r.update(T=[-1, 100]), "int ms"),
        (lambda r: r.update(X=[10.0]), "ragged"),
        (lambda r: r.update(X=[600.0, 1.0]), "outside"),
        (lambda r: r.update(Y=[385.0, 1.0]), "outside"),
        (lambda r: r.update(Y=[float("nan"), 1.0]), "bad coordinate"),
        (lambda r: r.update(X=[-0.1, 1.0]), "bad coordinate"),
        (lambda r: r.pop("T"), "missing key"),
        (lambda r: r.update(X=[1.0] * 17, Y=[1.0] * 17, T=[1] * 17), "length"),
    ])
    def test_rule_violations(self, mutate, msg):
        recs = _records(self.cells)
        mutate(recs[0])
        with pytest.raises(ValueError, match=msg):
            pc.validate_records(recs, self.cells)

    def test_key_set(self):
        with pytest.raises(ValueError, match="missing 1"):
            pc.validate_records(_records(self.cells[:2]), self.cells)
        with pytest.raises(ValueError, match="duplicate"):
            pc.validate_records(_records(self.cells + self.cells[:1]), self.cells)
        with pytest.raises(ValueError, match="extra 1"):
            pc.validate_records(_records(self.cells + [_cell("z.jpg", 9)]), self.cells)

    def test_write_refuses_metrics_sibling(self, tmp_path):
        pc.write_prediction_json(tmp_path / "pred_seed0.json", [])
        (tmp_path / "metrics.json").write_text("{}")
        with pytest.raises(ValueError, match="metrics.json"):
            pc.write_prediction_json(tmp_path / "pred_seed0.json", [])


# ── Group 4 — dataset additions (exp_keys restriction, per-trial images) ───────

@pytest.fixture
def cache(tmp_path):
    c, _ = EyeNetGazeCache.build(_write_csv(tmp_path / "p.csv"), FakeBundle(), tmp_path / "c.h5")
    return c


class TestDatasets:
    def test_exp_keys_restricts_both(self, cache):
        b = FakeBundle()
        g = EveRealNoiseDataset(cache, b, min_valid_frames=1, exp_keys=["exp02", "exp03", "nope"])
        i = EveRealNoiseImgDataset(cache, b, min_valid_frames=1, exp_keys=["exp02", "exp03", "nope"])
        assert [g.exp_key_at(k) for k in range(len(g))] == ["exp02", "exp03"]
        assert [i.exp_key_at(k) for k in range(len(i))] == ["exp02", "exp03"]

    def test_default_unchanged(self, cache):
        b = FakeBundle()
        assert len(EveRealNoiseDataset(cache, b, min_valid_frames=1)) == 3
        i = EveRealNoiseImgDataset(cache, b, min_valid_frames=1)
        assert i.image_bank.shape[0] == 2 and i.unique_idx == [0, 0, 1]   # exp01/exp02 share stimA

    def test_dedup_by_exp_key(self, cache):
        i = EveRealNoiseImgDataset(cache, FakeBundle(), min_valid_frames=1, dedup_by="exp_key")
        assert i.image_bank.shape[0] == 3 and i.unique_idx == [0, 1, 2]

    def test_dedup_by_invalid(self, cache):
        with pytest.raises(ValueError, match="dedup_by"):
            EveRealNoiseImgDataset(cache, FakeBundle(), dedup_by="subject")

    def test_collate_after_restriction(self, cache):
        g = EveRealNoiseDataset(cache, FakeBundle(), min_valid_frames=1, exp_keys=["exp01", "exp02"])
        batch = seq2seq_padded_collate_fn([g[0], g[1]])
        assert batch["tgt_mask"].shape == (2, 21)


# ── Group 5 — real ground truth + bundle ───────────────────────────────────────

@pytest.mark.skipif(not _HAS_REAL, reason="data/eve_bridge ground truth or EVE bundle absent")
class TestRealData:
    @pytest.fixture(scope="class")
    def real(self):
        from evedataset import EveBundle
        gt = pc.load_ground_truth(GT_PATH)
        cells = pc.contract_cells(gt, pc.load_subject_map(MAP_PATH))
        return cells, EveBundle.load(str(BUNDLE_DIR))

    def test_cells_resolve_one_to_one(self, real):
        cells, bundle = real
        assert len(cells) == pc.N_TEST_CELLS
        m = pc.resolve_exp_keys(cells, bundle.samples_df)
        assert len(m) == pc.N_TEST_CELLS and len(set(m.values())) == pc.N_TEST_CELLS
        subj = bundle.samples_df.set_index("exp_key")["subject"]
        assert all(subj[m[c.key]] == c.subject_eve for c in cells)

    def test_stdlib_validator_agrees(self, real, tmp_path, capsys):
        cells, _ = real
        recs = [pc.make_record(c, None, np.array([[960.0, 540.0, 250.4]]), 1) for c in cells]
        recs[0] = pc.make_record(cells[0], None, None, 0)
        pc.validate_records(recs, cells)
        path = tmp_path / "pred_seed0.json"
        pc.write_prediction_json(path, recs)
        spec = importlib.util.spec_from_file_location("vp", ROOT / "scripts" / "validate_prediction.py")
        vp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vp)
        vp.main(str(GT_PATH), str(path))
        out = capsys.readouterr().out
        assert "OK" in out and "1062 records" in out and "empty 1" in out
