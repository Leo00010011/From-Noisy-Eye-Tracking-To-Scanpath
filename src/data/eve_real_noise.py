"""EVE real-noise scanpath inference — dataset substrate.

Turns EyeNet's per-frame normalized gaze predictions (``predictions.csv``) into a
screen-space noisy gaze cache (:class:`EyeNetGazeCache`), and pairs each projected
trajectory with its stimulus image for autoregressive inference
(:class:`EveRealNoiseDataset`, :class:`EveRealNoiseImgDataset`).

This is an inference-only path. No ground-truth EVE scanpath is consumed and no
``clean_x`` (Tobii) supervision is produced — the datasets are self-contained and
never touch ``PipelineBuilder`` or the existing ``dataset_type: "eve"`` path.

Two split labels are in play and they mean different things: ``eyenet_split`` is the
ResNet18's own train/val/test partition (the *operative* filter — the recovery model
must be tested on data EyeNet did not train on), while ``eve_split`` is EVE's
partition, carried only as descriptive metadata. Filtering is named ``eyenet_split``
everywhere so an EVE split label cannot be passed by accident.

Projection is delegated to ``EveBundle.project_normalized_gaze`` — the canonical
normalized-frame → screen-pixel path — and is never reimplemented here.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.data.datasets import PAD_TOKEN_ID

# ── Constants ──────────────────────────────────────────────────────────────────

CENTER_FPS: float = 30.0
CENTER_FRAME_COUNT: int = 90
STIMULUS_W: int = 1920
STIMULUS_H: int = 1080
DEFAULT_CACHE_PATH: str = "data/eve_real_noise/eyenet_gaze_cache.h5"

# Sentinel direction for frames absent from the CSV. Never read back — a per-frame
# mask gates every downstream use — so its exact value is irrelevant.
_FILL_DIRECTION: np.ndarray = np.array([0.0, 0.0, -1.0])

_EYES = ("left", "right")
_GROUP = "/eyenet_gaze"
_EYENET_SPLITS = ("val", "test")

_REQUIRED_COLUMNS = {
    "split", "exp_key", "frame", "patch",
    "pred_x", "pred_y", "pred_z",
    "target_x", "target_y", "target_z",
    "angular_error_deg",
}

# The 7 numeric columns cast to float32 by load_eyenet_predictions.
_NUMERIC_COLUMNS = [
    "pred_x", "pred_y", "pred_z",
    "target_x", "target_y", "target_z",
    "angular_error_deg",
]

# Datasets written to / read from the /eyenet_gaze group (order-independent).
_STRING_DATASETS = ("exp_keys", "eyenet_split", "eve_split", "stimulus_name")
_ARRAY_DATASETS = (
    "gaze_px", "validity", "gt_gaze_px",
    "left_px", "right_px", "left_validity", "right_validity",
    "angular_error_deg",
)


# ── Step 1 — CSV loader ────────────────────────────────────────────────────────

def load_eyenet_predictions(csv_path: "str | Path") -> pd.DataFrame:
    """Read ``predictions.csv`` and validate it (FR1).

    ``exp_key``/``patch``/``split`` are read as ``str``, ``frame`` as ``int32`` and
    the 7 numeric columns as ``float32``. Raises ``ValueError`` on a missing column,
    a duplicate ``(exp_key, frame, patch)`` row, a frame outside ``[0, 90)``, or a
    ``patch`` other than ``left``/``right``. Emits a ``warnings.warn`` (never raises)
    if any prediction vector deviates from unit norm by more than ``1e-3``; vectors
    are **not** re-normalized here — ``project_normalized_gaze`` normalizes after the
    rotation.
    """
    df = pd.read_csv(csv_path, dtype={"exp_key": str, "patch": str, "split": str})

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"predictions CSV missing columns: {sorted(missing)}")

    bad_patch = set(df["patch"].unique()) - set(_EYES)
    if bad_patch:
        raise ValueError(f"unexpected patch values (expected left/right): {sorted(bad_patch)}")

    if df.duplicated(["exp_key", "frame", "patch"]).any():
        n = int(df.duplicated(["exp_key", "frame", "patch"]).sum())
        raise ValueError(f"{n} duplicate (exp_key, frame, patch) rows in predictions CSV")

    if df["frame"].min() < 0 or df["frame"].max() >= CENTER_FRAME_COUNT:
        raise ValueError(
            f"frame outside [0, {CENTER_FRAME_COUNT}): "
            f"min={int(df['frame'].min())}, max={int(df['frame'].max())}"
        )

    norms = np.linalg.norm(df[["pred_x", "pred_y", "pred_z"]].to_numpy(np.float64), axis=1)
    n_bad = int((np.abs(norms - 1.0) > 1e-3).sum())
    if n_bad:
        warnings.warn(f"{n_bad} prediction vectors deviate from unit norm by > 1e-3")

    return df.astype({"frame": "int32", **{c: "float32" for c in _NUMERIC_COLUMNS}})


# ── Step 2 — scatter + projection ──────────────────────────────────────────────

def _combine_eyes(lpx: np.ndarray, lval: np.ndarray,
                  rpx: np.ndarray, rval: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """FR3.3 — per-frame mean where both eyes valid, single eye where only one,
    NaN where neither.

    Averaging is done in screen-pixel space (after each ray's own eye origin has been
    applied), not in vector space. Returns ``(gaze_px (90, 2) float32,
    validity (90,) bool)``.
    """
    both = lval & rval
    out = np.full_like(lpx, np.nan)
    out[both] = (lpx[both] + rpx[both]) / 2.0
    out[lval & ~rval] = lpx[lval & ~rval]
    out[rval & ~lval] = rpx[rval & ~lval]
    return out.astype(np.float32), (lval | rval)


def _project_experiment(sub_df: pd.DataFrame, exp_key: str, bundle) -> dict:
    """Project all CSV rows for one ``exp_key`` (both eyes) to screen pixels (FR2, FR3).

    Returns the per-experiment arrays that :class:`EyeNetGazeCache` stacks into the
    HDF5 layout. Raises ``KeyError``/``ValueError`` on a projection failure (caught and
    recorded as ``skipped`` by ``build``).
    """
    per_eye: dict[str, dict] = {}
    for eye in _EYES:
        s = sub_df[sub_df["patch"] == eye].sort_values("frame")
        fr = s["frame"].to_numpy(np.int64)
        mask = np.zeros(CENTER_FRAME_COUNT, dtype=bool)
        mask[fr] = True

        pred = np.tile(_FILL_DIRECTION, (CENTER_FRAME_COUNT, 1))
        tgt = np.tile(_FILL_DIRECTION, (CENTER_FRAME_COUNT, 1))
        pred[fr] = s[["pred_x", "pred_y", "pred_z"]].to_numpy(np.float64)
        tgt[fr] = s[["target_x", "target_y", "target_z"]].to_numpy(np.float64)

        p = bundle.project_normalized_gaze(exp_key, pred, eye=eye, spherical=False)
        t = bundle.project_normalized_gaze(exp_key, tgt, eye=eye, spherical=False)
        valid = mask & p["validity"]                                    # FR3.2

        ae = np.full(CENTER_FRAME_COUNT, np.nan, np.float32)
        ae[fr] = s["angular_error_deg"].to_numpy(np.float32)

        per_eye[eye] = {
            "px": np.where(valid[:, None], p["hit_px"], np.nan).astype(np.float32),
            "gt_px": np.where(valid[:, None], t["hit_px"], np.nan).astype(np.float32),
            "validity": valid,
            "ae": ae,
        }

    gaze_px, validity = _combine_eyes(
        per_eye["left"]["px"], per_eye["left"]["validity"],
        per_eye["right"]["px"], per_eye["right"]["validity"])
    gt_gaze_px, _ = _combine_eyes(
        per_eye["left"]["gt_px"], per_eye["left"]["validity"],
        per_eye["right"]["gt_px"], per_eye["right"]["validity"])

    return {
        "gaze_px": gaze_px,
        "validity": validity,
        "gt_gaze_px": gt_gaze_px,
        "left_px": per_eye["left"]["px"],
        "right_px": per_eye["right"]["px"],
        "left_validity": per_eye["left"]["validity"],
        "right_validity": per_eye["right"]["validity"],
        # [..., 0] = left, [..., 1] = right
        "angular_error_deg": np.stack([per_eye["left"]["ae"], per_eye["right"]["ae"]], axis=-1),
    }


# ── Step 3 — EyeNetGazeCache ───────────────────────────────────────────────────

class EyeNetGazeCache:
    """Screen-space projected gaze cache, addressed by ``exp_key`` (FR5, FR6)."""

    def __init__(self, arrays: dict, attrs: dict) -> None:
        self._arrays = arrays
        self._attrs = dict(attrs)
        self._exp_keys = [str(k) for k in arrays["exp_keys"]]
        self._idx = {k: i for i, k in enumerate(self._exp_keys)}

    # -- construction ----------------------------------------------------------

    @classmethod
    def build(cls, csv_path, bundle,
              cache_path: str = DEFAULT_CACHE_PATH) -> "tuple[EyeNetGazeCache, list[tuple[str, str]]]":
        """Project every experiment in the CSV and persist the cache (FR6.1).

        Returns ``(cache, skipped)`` where ``skipped`` is a list of
        ``(exp_key, reason)``. Experiments are skipped — never crashed on — when
        absent from the bundle, lacking gaze-norm / gaze-ray coverage, or failing
        projection. A key spanning both EyeNet splits raises (corrupt CSV).
        """
        df = load_eyenet_predictions(csv_path)
        samples = bundle.samples_df.set_index("exp_key")

        records: list[dict] = []
        skipped: list[tuple[str, str]] = []

        for k in sorted(df["exp_key"].unique()):
            sub = df[df["exp_key"] == k]
            if k not in samples.index:
                skipped.append((k, "not_in_bundle"))
                continue
            if not bundle.has_gaze_norm(k):
                skipped.append((k, "no_gaze_norm"))
                continue
            if not bundle.has_gaze_ray(k):
                skipped.append((k, "no_gaze_ray"))
                continue
            if sub["split"].nunique() != 1:
                raise ValueError(
                    f"exp_key {k!r} spans multiple EyeNet splits "
                    f"{sorted(sub['split'].unique())} — corrupt predictions CSV "
                    "(a key must carry exactly one EyeNet split)."
                )
            try:
                rec = _project_experiment(sub, k, bundle)
            except (KeyError, ValueError) as e:
                skipped.append((k, f"projection_failed: {e}"))
                continue

            rec["exp_key"] = k
            rec["eyenet_split"] = str(sub["split"].iloc[0])
            rec["eve_split"] = str(samples.loc[k, "split"])
            rec["stimulus_name"] = str(samples.loc[k, "stimulus_name"])
            records.append(rec)

        arrays = cls._stack_records(records)
        n_offscreen = cls._count_offscreen(arrays)

        attrs = {
            "timestamp_source": "synthesized_30hz",
            "center_fps": CENTER_FPS,
            "source_csv": str(csv_path),
            "bundle_dir": str(bundle.bundle_dir),
            "built_at": datetime.now(timezone.utc).isoformat(),
            "n_offscreen": int(n_offscreen),
        }

        cache = cls(arrays, attrs)
        cache.save(cache_path)
        return cache, skipped

    @staticmethod
    def _stack_records(records: "list[dict]") -> dict:
        arrays: dict = {}
        for key in _STRING_DATASETS:
            src = "exp_key" if key == "exp_keys" else key
            arrays[key] = [r[src] for r in records]
        for key in _ARRAY_DATASETS:
            if records:
                arrays[key] = np.stack([r[key] for r in records])
            else:
                dtype = bool if key.endswith("validity") else np.float32
                trailing = () if key.endswith("validity") else (2,)
                arrays[key] = np.zeros((0, CENTER_FRAME_COUNT, *trailing), dtype=dtype)
        return arrays

    @staticmethod
    def _count_offscreen(arrays: dict) -> int:
        g, v = arrays["gaze_px"], arrays["validity"]
        if g.shape[0] == 0:
            return 0
        off = (
            (g[..., 0] < 0) | (g[..., 0] >= STIMULUS_W)
            | (g[..., 1] < 0) | (g[..., 1] >= STIMULUS_H)
        )
        return int((off & v).sum())

    # -- persistence -----------------------------------------------------------

    def save(self, cache_path: str = DEFAULT_CACHE_PATH) -> None:
        """Write only the ``/eyenet_gaze`` group, in append mode, deleting and
        recreating it if present (FR6.2). Never touches another group."""
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        str_dt = h5py.string_dtype()
        with h5py.File(path, "a") as f:
            if _GROUP in f:
                del f[_GROUP]
            g = f.require_group(_GROUP)
            for name in _STRING_DATASETS:
                g.create_dataset(name, data=np.array(self._arrays[name], dtype=object), dtype=str_dt)
            for name in _ARRAY_DATASETS:
                g.create_dataset(name, data=self._arrays[name])
            g.attrs.update(self._attrs)

    @classmethod
    def load(cls, cache_path: str = DEFAULT_CACHE_PATH) -> "EyeNetGazeCache":
        """Load the cache and verify its primary key (FR6.3)."""
        path = Path(cache_path)
        if not path.exists():
            raise FileNotFoundError(f"gaze cache not found at {path}")
        with h5py.File(path, "r") as f:
            if _GROUP not in f:
                raise ValueError(f"'{_GROUP.strip('/')}' group missing in {path}")
            g = f[_GROUP]
            arrays: dict = {}
            for name in (*_STRING_DATASETS, *_ARRAY_DATASETS):
                if name not in g:
                    raise ValueError(f"expected dataset {name!r} missing from '{_GROUP.strip('/')}'")
                data = g[name][:]
                if name in _STRING_DATASETS:
                    data = [v.decode() if isinstance(v, bytes) else str(v) for v in data]
                arrays[name] = data
            attrs = dict(g.attrs)

        if len(set(arrays["exp_keys"])) != len(arrays["exp_keys"]):
            raise ValueError("exp_keys contains a duplicate — corrupt cache")
        return cls(arrays, attrs)

    def verify(self, bundle) -> "list[tuple[str, str]]":
        """QA hook (FR6.4). Returns ``(exp_key, reason)`` for every inconsistency —
        never raises."""
        issues: list[tuple[str, str]] = []
        samples = bundle.samples_df.set_index("exp_key")
        for i, k in enumerate(self._exp_keys):
            if k not in samples.index:
                issues.append((k, "not_in_bundle"))
                continue
            if self._arrays["eve_split"][i] != str(samples.loc[k, "split"]):
                issues.append((k, "eve_split_mismatch"))
            if self._arrays["eyenet_split"][i] not in _EYENET_SPLITS:
                issues.append((k, "eyenet_split_not_val_or_test"))
        return issues

    # -- accessors -------------------------------------------------------------

    def _require(self, exp_key: str) -> int:
        if exp_key not in self._idx:
            raise KeyError(f"exp_key {exp_key!r} not found in gaze cache.")
        return self._idx[exp_key]

    @property
    def exp_keys(self) -> "list[str]":
        return list(self._exp_keys)

    @property
    def splits_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "exp_key": list(self._exp_keys),
            "eyenet_split": list(self._arrays["eyenet_split"]),
            "eve_split": list(self._arrays["eve_split"]),
            "stimulus_name": list(self._arrays["stimulus_name"]),
        })

    @property
    def attrs(self) -> dict:
        return dict(self._attrs)

    def get_gaze(self, exp_key: str) -> np.ndarray:
        return self._arrays["gaze_px"][self._require(exp_key)]

    def get_validity(self, exp_key: str) -> np.ndarray:
        return self._arrays["validity"][self._require(exp_key)]

    def get_gt_gaze(self, exp_key: str) -> np.ndarray:
        return self._arrays["gt_gaze_px"][self._require(exp_key)]

    def get_eye_gaze(self, exp_key: str, eye: str) -> dict:
        if eye not in _EYES:
            raise ValueError(f"eye must be one of {_EYES}, got {eye!r}")
        i = self._require(exp_key)
        return {
            "px": self._arrays[f"{eye}_px"][i],
            "validity": self._arrays[f"{eye}_validity"][i],
        }

    def get_angular_error(self, exp_key: str) -> np.ndarray:
        return self._arrays["angular_error_deg"][self._require(exp_key)]


# ── Step 4 — shared row filter ─────────────────────────────────────────────────

def _accepted_rows(cache: EyeNetGazeCache, eyenet_split: "str | None",
                   min_valid_frames: int) -> "list[tuple[int, str]]":
    """Return ``[(row_index_in_cache, exp_key), ...]`` in cache order (FR8.2).

    Shared by both datasets so positional index ``i`` refers to the same ``exp_key``
    in each. ``eyenet_split`` is validated against EyeNet's split vocabulary, not
    EVE's — passing an EVE-only label such as ``"train"`` raises rather than silently
    returning an empty dataset.
    """
    if eyenet_split is not None and eyenet_split not in _EYENET_SPLITS:
        raise ValueError(
            f"eyenet_split must be one of {_EYENET_SPLITS} or None, got {eyenet_split!r}. "
            "EVE split labels ('train'/'val'/'test' from the bundle) are not accepted here — "
            "the recovery model never saw EVE, so only EyeNet's split is meaningful."
        )
    sdf = cache.splits_df
    out: list[tuple[int, str]] = []
    for i, k in enumerate(cache.exp_keys):
        if eyenet_split is not None and sdf.at[i, "eyenet_split"] != eyenet_split:
            continue
        if int(cache.get_validity(k).sum()) < min_valid_frames:
            continue
        out.append((i, k))
    return out


# ── Step 5 — EveRealNoiseDataset ───────────────────────────────────────────────

class EveRealNoiseDataset(Dataset):
    """Gaze dataset over the projected cache (FR7).

    Materializes every accepted row into RAM. ``x`` is ``(3, T_valid)`` with rows
    ``[x_px, y_px, t_ms]`` over the valid frames in ascending order; ``y`` is a
    ``(3, max_fixations)`` placeholder of ``PAD_TOKEN_ID`` (no ground truth is
    available — the placeholder only fixes the decode-step budget and is never read
    as supervision). ``clean_x`` is never emitted.
    """

    def __init__(self, cache: EyeNetGazeCache, bundle, eyenet_split: "str | None" = None,
                 max_fixations: int = 20, min_valid_frames: int = 5,
                 transforms: list = (), log: bool = False) -> None:
        self.transforms = list(transforms)
        self.max_fixations = max_fixations
        self.eyenet_split = eyenet_split

        accepted = _accepted_rows(cache, eyenet_split, min_valid_frames)
        sdf = cache.splits_df

        xs, ys, masks = [], [], []
        exp_keys, eyenet_splits, eve_splits, frame_indices = [], [], [], []
        for i, k in accepted:
            val = cache.get_validity(k)                 # (90,) bool
            frames = np.where(val)[0]                    # ascending
            gaze = cache.get_gaze(k)[frames]            # (T, 2) float32

            x = np.empty((3, len(frames)), dtype=np.float64)
            x[0], x[1] = gaze[:, 0], gaze[:, 1]
            x[2] = frames * (1000.0 / CENTER_FPS)        # FR4.1
            if np.isnan(x[:2]).any():
                raise ValueError(
                    f"exp_key {k!r} has NaN gaze at a valid frame — cache 'validity' "
                    "and 'gaze_px' disagree (corrupt cache)."
                )

            y = np.full((3, max_fixations), PAD_TOKEN_ID, dtype=np.float64)   # FR7.4
            fixation_mask = np.zeros(len(frames), dtype=np.uint8)

            xs.append(x)
            ys.append(y)
            masks.append(fixation_mask)
            exp_keys.append(k)
            eyenet_splits.append(sdf.at[i, "eyenet_split"])
            eve_splits.append(sdf.at[i, "eve_split"])
            frame_indices.append(frames.astype(np.int64))

        self.data_store = {
            "x": xs,
            "y": ys,
            "fixation_mask": masks,
            "exp_keys": exp_keys,
            "eyenet_splits": eyenet_splits,
            "eve_splits": eve_splits,
            "frame_indices": frame_indices,
        }
        self.length = len(xs)

        if log:
            skipped = len(cache.exp_keys) - self.length
            print(
                f"EveRealNoiseDataset eyenet_split={eyenet_split}: {self.length} samples "
                f"({skipped} skipped of {len(cache.exp_keys)})"
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> dict:
        x = self.data_store["x"][i].copy()
        y = self.data_store["y"][i].copy()
        fixation_mask = self.data_store["fixation_mask"][i].copy()

        inp = {"x": x, "y": y, "fixation_mask": fixation_mask}
        for t in self.transforms:
            inp = t(inp)

        out = {"x": inp["x"], "y": inp["y"], "sample_idx": i}
        for k in ("in_tgt", "down_offset", "heatmaps"):
            if k in inp:
                out[k] = inp[k]
        return out

    def exp_key_at(self, i: int) -> str:
        return self.data_store["exp_keys"][i]

    def eyenet_split_at(self, i: int) -> str:
        return self.data_store["eyenet_splits"][i]

    def eve_split_at(self, i: int) -> str:
        return self.data_store["eve_splits"][i]

    def frame_indices_at(self, i: int) -> np.ndarray:
        return self.data_store["frame_indices"][i]


# ── Step 6 — EveRealNoiseImgDataset ────────────────────────────────────────────

class EveRealNoiseImgDataset(Dataset):
    """Stimulus dataset paired with :class:`EveRealNoiseDataset` (FR8).

    Applies the identical accept/skip filter, so positional index ``i`` refers to
    the same ``exp_key`` in both. Images are deduplicated by ``stimulus_name`` and
    squashed non-uniformly from 1920×1080 to ``resize_size``×``resize_size`` — the
    same ingest as ``DeduplicatedMemoryDataset`` / ``EveImgDataset``.
    """

    def __init__(self, cache: EyeNetGazeCache, bundle, eyenet_split: "str | None" = None,
                 max_fixations: int = 20, min_valid_frames: int = 5,
                 resize_size: int = 256, transform=None) -> None:
        accepted = _accepted_rows(cache, eyenet_split, min_valid_frames)
        sdf = cache.splits_df

        ingest = v2.Compose([
            v2.ToImage(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.ToDtype(torch.uint8, scale=False),
        ])

        stim_to_uid: dict[str, int] = {}
        first_key_per_stimulus: list[str] = []
        self.unique_idx: list[int] = []
        self._exp_keys: list[str] = []
        for i, k in accepted:
            name = sdf.at[i, "stimulus_name"]
            if name not in stim_to_uid:
                stim_to_uid[name] = len(first_key_per_stimulus)
                first_key_per_stimulus.append(k)
            self.unique_idx.append(stim_to_uid[name])
            self._exp_keys.append(k)

        N_unique = len(first_key_per_stimulus)
        self.image_bank = torch.empty((N_unique, 3, resize_size, resize_size), dtype=torch.uint8)
        for uid, exp_key in enumerate(first_key_per_stimulus):
            self.image_bank[uid] = ingest(bundle.get_stimulus(exp_key))   # HWC uint8 -> CHW

        self.runtime_transform = transform
        self.length = len(self.unique_idx)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        uid = self.unique_idx[i]
        img = self.image_bank[uid]
        if self.runtime_transform is not None:
            img = self.runtime_transform(img)
        return img, i, uid

    def exp_key_at(self, i: int) -> str:
        return self._exp_keys[i]
