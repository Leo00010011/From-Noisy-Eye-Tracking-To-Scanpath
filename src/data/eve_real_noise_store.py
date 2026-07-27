"""EVE real-noise inference store — keyed HDF5 writer/reader for model outputs (FR9).

Persists autoregressive scanpath predictions (and, when the checkpoint has a denoise
head, the denoised gaze) keyed by ``exp_key``. Depends on nothing from
``eve_real_noise``: it takes plain record dicts, so it is written and tested
independently.

Unlike :class:`EyeNetGazeCache` (append mode, single-group delete), this store writes
with mode ``"w"`` and therefore replaces the whole output file — a run's outputs are a
fresh artifact, not an addition to an existing one.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

_GROUP = "/inference"
_EOS_THRESHOLD = 0.5

_STRING_DATASETS = ("exp_keys", "eyenet_split", "eve_split")


class RealNoiseInferenceStore:
    """Reader for a ``/inference`` HDF5 artifact; ``save`` is a classmethod writer."""

    def __init__(self, arrays: dict, attrs: dict) -> None:
        self._arrays = arrays
        self._attrs = dict(attrs)
        self._exp_keys = [str(k) for k in arrays["exp_keys"]]
        self._idx = {k: i for i, k in enumerate(self._exp_keys)}

    # -- writer ----------------------------------------------------------------

    @classmethod
    def save(cls, path, run_name: str, records: "list[dict]", attrs: dict) -> None:
        """Write the inference store (FR9.1).

        ``records`` is a ``list[dict]``; every dict must carry ``exp_key``,
        ``pred_scanpath`` ``(K, 3)``, ``eos_logit`` ``(K,)``, ``src_px`` ``(src_len, 3)``,
        ``src_len``, ``frame_indices`` ``(src_len,)``, and optionally ``denoise_px``
        ``(src_len, 2)``. Raises ``ValueError`` on a missing/duplicate ``exp_key``, a
        differing decode-step count ``K`` (mixed ``max_fixations``), or a denoise head
        present on only some records (mixed checkpoints).
        """
        if not records:
            raise ValueError("records is empty — nothing to save.")

        keys = [r.get("exp_key") for r in records]
        if any(k is None for k in keys):
            raise ValueError("every record must carry 'exp_key'.")
        if len(set(keys)) != len(keys):
            dup = sorted({str(k) for k in keys if keys.count(k) > 1})
            raise ValueError(f"duplicate exp_key in records: {dup}")

        K = int(np.asarray(records[0]["pred_scanpath"]).shape[0])
        for r in records:
            if int(np.asarray(r["pred_scanpath"]).shape[0]) != K:
                raise ValueError(
                    f"pred_scanpath step count differs ({int(np.asarray(r['pred_scanpath']).shape[0])} "
                    f"vs {K}) — a mixed max_fixations run cannot be stored."
                )

        has_flags = ["denoise_px" in r for r in records]
        if any(has_flags) and not all(has_flags):
            raise ValueError(
                "some records carry 'denoise_px' and some do not — two different "
                "checkpoints appear to have been merged into one run."
            )
        has_denoise = all(has_flags)

        N = len(records)
        T = max(int(r["src_len"]) for r in records)

        exp_keys, eyenet, eve = [], [], []
        pred_scanpath = np.full((N, K, 3), np.nan, np.float32)
        eos_logit = np.full((N, K), np.nan, np.float32)
        pred_len = np.zeros(N, np.int32)
        src_px = np.full((N, T, 3), np.nan, np.float32)
        src_len = np.zeros(N, np.int32)
        frame_indices = np.full((N, T), -1, np.int32)
        denoise_px = np.full((N, T, 2), np.nan, np.float32) if has_denoise else None

        for i, r in enumerate(records):
            exp_keys.append(str(r["exp_key"]))
            eyenet.append(str(r.get("eyenet_split", "")))
            eve.append(str(r.get("eve_split", "")))

            pred_scanpath[i] = np.asarray(r["pred_scanpath"], np.float32)
            el = np.asarray(r["eos_logit"], np.float32).reshape(-1)
            eos_logit[i] = el
            prob = 1.0 / (1.0 + np.exp(-el))
            fired = np.where(prob > _EOS_THRESHOLD)[0]
            pred_len[i] = int(fired[0]) if fired.size else K            # FR9.3

            L = int(r["src_len"])
            src_len[i] = L
            src_px[i, :L] = np.asarray(r["src_px"], np.float32)
            frame_indices[i, :L] = np.asarray(r["frame_indices"], np.int32)
            if has_denoise:
                denoise_px[i, :L] = np.asarray(r["denoise_px"], np.float32)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        str_dt = h5py.string_dtype()
        with h5py.File(path, "w") as f:
            g = f.require_group(_GROUP)
            g.create_dataset("exp_keys", data=np.array(exp_keys, dtype=object), dtype=str_dt)
            g.create_dataset("eyenet_split", data=np.array(eyenet, dtype=object), dtype=str_dt)
            g.create_dataset("eve_split", data=np.array(eve, dtype=object), dtype=str_dt)
            g.create_dataset("pred_scanpath", data=pred_scanpath)
            g.create_dataset("eos_logit", data=eos_logit)
            g.create_dataset("pred_len", data=pred_len)
            g.create_dataset("src_px", data=src_px)
            g.create_dataset("src_len", data=src_len)
            g.create_dataset("frame_indices", data=frame_indices)
            if has_denoise:
                g.create_dataset("denoise_px", data=denoise_px)

            merged = dict(attrs)
            merged["run_name"] = run_name
            merged["has_denoise"] = bool(has_denoise)
            merged["eos_threshold"] = _EOS_THRESHOLD
            g.attrs.update(merged)

    # -- reader ----------------------------------------------------------------

    @classmethod
    def load(cls, path) -> "RealNoiseInferenceStore":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"inference store not found at {path}")
        with h5py.File(path, "r") as f:
            if _GROUP not in f:
                raise ValueError(f"'{_GROUP.strip('/')}' group missing in {path}")
            g = f[_GROUP]
            if "exp_keys" not in g:
                raise ValueError("'exp_keys' dataset missing — corrupt inference store")
            arrays: dict = {}
            for name in g.keys():
                data = g[name][:]
                if name in _STRING_DATASETS:
                    data = [v.decode() if isinstance(v, bytes) else str(v) for v in data]
                arrays[name] = data
            attrs = dict(g.attrs)

        if len(set(arrays["exp_keys"])) != len(arrays["exp_keys"]):
            raise ValueError("exp_keys contains a duplicate — corrupt inference store")
        return cls(arrays, attrs)

    def _require(self, exp_key: str) -> int:
        if exp_key not in self._idx:
            raise KeyError(f"exp_key {exp_key!r} not found in inference store.")
        return self._idx[exp_key]

    @property
    def has_denoise(self) -> bool:
        return bool(self._attrs.get("has_denoise", False))

    @property
    def attrs(self) -> dict:
        return dict(self._attrs)

    @property
    def exp_keys(self) -> "list[str]":
        return list(self._exp_keys)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "exp_key": list(self._exp_keys),
            "eyenet_split": list(self._arrays["eyenet_split"]),
            "eve_split": list(self._arrays["eve_split"]),
            "pred_len": np.asarray(self._arrays["pred_len"]),
            "src_len": np.asarray(self._arrays["src_len"]),
        })

    def get(self, exp_key: str) -> dict:
        """Return every stored field for ``exp_key``, with ``src_px`` /
        ``frame_indices`` / ``denoise_px`` trimmed to this row's ``src_len``."""
        i = self._require(exp_key)
        L = int(self._arrays["src_len"][i])
        out = {
            "exp_key": exp_key,
            "eyenet_split": self._arrays["eyenet_split"][i],
            "eve_split": self._arrays["eve_split"][i],
            "pred_scanpath": self._arrays["pred_scanpath"][i],
            "eos_logit": self._arrays["eos_logit"][i],
            "pred_len": int(self._arrays["pred_len"][i]),
            "src_px": self._arrays["src_px"][i, :L],
            "src_len": L,
            "frame_indices": self._arrays["frame_indices"][i, :L],
        }
        if self.has_denoise and "denoise_px" in self._arrays:
            out["denoise_px"] = self._arrays["denoise_px"][i, :L]
        return out

    def get_scanpath(self, exp_key: str) -> np.ndarray:
        """Return ``pred_scanpath`` trimmed to ``pred_len`` — ``(pred_len, 3)`` (FR9.2).

        The stored ``pred_scanpath`` retains all ``K`` decode steps (FR9.3); this is a
        convenience slice, not a destructive truncation.
        """
        i = self._require(exp_key)
        pl = int(self._arrays["pred_len"][i])
        return self._arrays["pred_scanpath"][i, :pl]
