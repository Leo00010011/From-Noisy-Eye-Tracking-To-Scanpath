"""EVE head-to-head prediction contract (``prediction.json``).

Pure helpers that turn this repo's EVE real-noise predictions into the file the
``few-shot-scanpath`` evaluation repo scores (``tools/rescore/contract.py`` there;
handoff spec ``spec/handoff/prediction_contract.md``). No model code lives here —
the driver is ``src/notebooks/save_prediction_contract_eve.py``.

Contract summary:
  * one record per ``split == "test"`` ground-truth cell, keyed ``(name, subject)``
    with ``subject`` the dense id (0–37); key set must equal the 1062 test keys;
  * ``X``/``Y`` in the 512×384 metric screen (native 1920×1080 divided by
    3.75 / 2.8125 — non-uniform), finite and clipped to the screen;
  * ``T`` Python ``int`` milliseconds, ``>= 0``;
  * any length (0 allowed), capped at ``MAX_LENGTH``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GT_SHA256 = "46c6926f6075f4c7038138ea5116feaeec52efb201104133d6c96a7032c5ac9b"
N_TEST_CELLS = 1062
SCREEN_W, SCREEN_H = 1920, 1080
METRIC_W, METRIC_H = 512, 384
SCALE_X = SCREEN_W / METRIC_W     # 3.75
SCALE_Y = SCREEN_H / METRIC_H     # 2.8125
MAX_LENGTH = 16
EOS_THRESHOLD = 0.5               # same rule as RealNoiseInferenceStore.pred_len


@dataclass(frozen=True)
class Cell:
    """One ``split == "test"`` ground-truth trial to predict."""
    name: str            # e.g. "MIT-i1000274881.jpg" — exactly as in fixations.json
    subject: int         # dense id 0–37
    subject_eve: str     # EVE participant id, e.g. "train30"

    @property
    def key(self) -> "tuple[str, int]":
        return (self.name, self.subject)

    @property
    def stimulus_name(self) -> str:
        """Bundle ``stimulus_name`` (file name without extension)."""
        return os.path.splitext(self.name)[0]


# ── Inputs ─────────────────────────────────────────────────────────────────────

def sha256_file(path: "str | Path") -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ground_truth(path: "str | Path", expected_sha256: "str | None" = GT_SHA256) -> list:
    """Load ``fixations.json``, refusing a file whose hash is not the contract's."""
    if expected_sha256 is not None:
        got = sha256_file(path)
        if got != expected_sha256:
            raise ValueError(f"wrong fixations.json at {path}: sha256 {got} != {expected_sha256}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_subject_map(path: "str | Path") -> "dict[int, str]":
    """``subject_id_map.json["to_eve"]`` with integer keys (dense id → EVE id)."""
    with open(path, encoding="utf-8") as f:
        to_eve = json.load(f)["to_eve"]
    return {int(k): str(v) for k, v in to_eve.items()}


def contract_cells(gt: list, to_eve: "dict[int, str]") -> "list[Cell]":
    """The cells to predict, taken straight from the ground truth (never rebuilt)."""
    cells = [Cell(r["name"], int(r["subject"]), to_eve[int(r["subject"])])
             for r in gt if r["split"] == "test"]
    if len({c.key for c in cells}) != len(cells):
        raise ValueError("ground truth has duplicate (name, subject) test keys")
    return cells


def resolve_exp_keys(cells: "list[Cell]", samples_df) -> "dict[tuple[str, int], str]":
    """Map each cell to the EVE trial ``(stimulus_name, subject_eve)`` it came from.

    Raises unless every cell matches exactly one bundle trial — the prediction must
    use *that* participant's trial (its gaze and its per-trial screen render).
    """
    groups = samples_df.groupby(["stimulus_name", "subject"])["exp_key"].apply(list).to_dict()
    out: dict = {}
    bad = []
    for c in cells:
        keys = groups.get((c.stimulus_name, c.subject_eve), [])
        if len(keys) != 1:
            bad.append((c.key, c.subject_eve, keys))
            continue
        out[c.key] = str(keys[0])
    if bad:
        raise ValueError(f"{len(bad)} cells do not map to exactly one bundle trial, e.g. {bad[:3]}")
    return out


# ── Output conversion ──────────────────────────────────────────────────────────

def predicted_length(eos_logit: np.ndarray, threshold: float = EOS_THRESHOLD) -> int:
    """First step whose ``sigmoid(eos) > threshold``; all steps if it never fires."""
    el = np.asarray(eos_logit, dtype=np.float64).reshape(-1)
    fired = np.where(1.0 / (1.0 + np.exp(-el)) > threshold)[0]
    return int(fired[0]) if fired.size else int(el.size)


def screen_to_metric(x_px: np.ndarray, y_px: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """1920×1080 trial screen pixels → the 512×384 metric screen, clipped."""
    x = np.clip(np.asarray(x_px, np.float64) / SCALE_X, 0.0, METRIC_W)
    y = np.clip(np.asarray(y_px, np.float64) / SCALE_Y, 0.0, METRIC_H)
    return x, y


def make_record(cell: Cell, exp_key: "str | None", scanpath_px: "np.ndarray | None",
                length: int, status: str = "predicted") -> dict:
    """Build one contract record.

    ``scanpath_px`` is ``(K, 3)`` ``[x_px, y_px, dur_ms]`` in the trial's 1920×1080
    screen; the first ``min(length, MAX_LENGTH)`` rows are kept. ``None`` or
    ``length == 0`` yields an empty scanpath. Values are plain Python ``float``/``int``
    so ``json.dump`` never meets a numpy scalar.
    """
    rec = {"name": cell.name, "subject": int(cell.subject), "X": [], "Y": [], "T": [],
           "subject_eve": cell.subject_eve, "exp_key": exp_key, "status": status}
    if scanpath_px is None or length <= 0:
        return rec
    sp = np.asarray(scanpath_px, np.float64)[: min(int(length), MAX_LENGTH)]
    if not np.isfinite(sp).all():
        raise ValueError(f"non-finite prediction for {cell.key} ({exp_key})")
    x, y = screen_to_metric(sp[:, 0], sp[:, 1])
    rec["X"] = [float(v) for v in x]
    rec["Y"] = [float(v) for v in y]
    rec["T"] = [int(round(max(float(t), 0.0))) for t in sp[:, 2]]
    return rec


# ── Validation (mirrors the scorer's FR4 / handoff §5) ─────────────────────────

def validate_records(records: list, cells: "list[Cell]") -> dict:
    """Check every contract rule; raise ``ValueError`` on the first violation.

    Returns summary stats (ranges, length distribution) for the run notes.
    """
    if not isinstance(records, list):
        raise ValueError("top level must be a list")
    for i, r in enumerate(records):
        for k in ("name", "subject", "X", "Y", "T"):
            if k not in r:
                raise ValueError(f"record {i}: missing key {k!r}")
        if not isinstance(r["name"], str):
            raise ValueError(f"record {i}: name must be str")
        if type(r["subject"]) is not int:
            raise ValueError(f"record {i}: subject must be a JSON int, got {type(r['subject'])}")
        if not len(r["X"]) == len(r["Y"]) == len(r["T"]):
            raise ValueError(f"record {i}: ragged X/Y/T")
        for v in r["X"] + r["Y"]:
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError(f"record {i}: bad coordinate {v!r}")
        if any(v > METRIC_W for v in r["X"]) or any(v > METRIC_H for v in r["Y"]):
            raise ValueError(f"record {i}: coordinate outside {METRIC_W}x{METRIC_H}")
        for t in r["T"]:
            if type(t) is not int or t < 0:
                raise ValueError(f"record {i}: T must be non-negative int ms, got {t!r}")
        if len(r["X"]) > MAX_LENGTH:
            raise ValueError(f"record {i}: length {len(r['X'])} > {MAX_LENGTH}")

    counts = Counter((r["name"], r["subject"]) for r in records)
    dups = [k for k, n in counts.items() if n > 1]
    if dups:
        raise ValueError(f"duplicate keys, e.g. {dups[:3]}")
    keys = {c.key for c in cells}
    if set(counts) != keys:
        raise ValueError(f"key set differs: missing {len(keys - set(counts))}, "
                         f"extra {len(set(counts) - keys)}")

    xs = [v for r in records for v in r["X"]]
    ys = [v for r in records for v in r["Y"]]
    ts = [t for r in records for t in r["T"]]
    lengths = [len(r["X"]) for r in records]
    return {
        "n_records": len(records),
        "x_range": (min(xs), max(xs)) if xs else None,
        "y_range": (min(ys), max(ys)) if ys else None,
        "t_range": (min(ts), max(ts)) if ts else None,
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "n_empty": lengths.count(0),
        "n_short": sum(1 <= n < 3 for n in lengths),
    }


def format_stats(path: str, s: dict) -> str:
    """One line in the handoff validator's output format."""
    fmt = lambda r, p: "n/a" if r is None else f"{r[0]:.{p}f}-{r[1]:.{p}f}"
    return (f"OK {path}: {s['n_records']} records | X {fmt(s['x_range'], 1)} | "
            f"Y {fmt(s['y_range'], 1)} | T {fmt(s['t_range'], 0)} ms | "
            f"length mean {s['length_mean']:.2f} min {s['length_min']} max {s['length_max']} | "
            f"empty {s['n_empty']} | short(1-2) {s['n_short']}")


def write_prediction_json(path: "str | Path", records: list) -> None:
    path = Path(path)
    if path.parent.joinpath("metrics.json").exists():
        raise ValueError(f"{path.parent} contains metrics.json — the scorer would treat it as "
                         "this run's fingerprint. Write predictions to a clean directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)
