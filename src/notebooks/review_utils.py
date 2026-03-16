from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _tensor_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return None


def _shape_of(value: Any) -> tuple[int, ...] | None:
    array = _tensor_to_numpy(value)
    if array is not None:
        return tuple(array.shape)
    if isinstance(value, dict):
        return None
    return None


def list_metric_files(base_dir: str | Path = OUTPUTS_DIR) -> pd.DataFrame:
    base_path = Path(base_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(base_path.rglob("metrics.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        rows.append(
            {
                "run_dir": str(path.parent.relative_to(PROJECT_ROOT)),
                "metrics_file": str(path.relative_to(PROJECT_ROOT)),
                "modified": pd.Timestamp(path.stat().st_mtime, unit="s"),
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
        )
    return pd.DataFrame(rows)


def list_recorder_files(base_dir: str | Path = PROJECT_ROOT) -> pd.DataFrame:
    base_path = Path(base_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(base_path.rglob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
        rows.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT)),
                "modified": pd.Timestamp(path.stat().st_mtime, unit="s"),
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
        )
    return pd.DataFrame(rows)


def looks_like_recorder_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return {"metadata", "data", "outputs", "activations"}.issubset(payload.keys())


def discover_recorder_payloads(base_dir: str | Path = PROJECT_ROOT, limit: int = 25) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(base_dir).rglob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not looks_like_recorder_payload(payload):
            continue
        metadata = payload.get("metadata", {})
        rows.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT)),
                "split": metadata.get("split"),
                "phase": metadata.get("phase"),
                "epoch": metadata.get("epoch"),
                "batch_index": metadata.get("batch_index"),
                "global_step": metadata.get("global_step"),
                "modified": pd.Timestamp(path.stat().st_mtime, unit="s"),
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
        )
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def load_recorder_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not looks_like_recorder_payload(payload):
        raise ValueError(f"{path} is not an inference recorder payload.")
    return payload


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "metadata": payload.get("metadata", {}),
        "data": {},
        "outputs": {},
        "activations": {},
    }

    for section in ("data", "outputs"):
        for key, value in payload.get(section, {}).items():
            summary[section][key] = {
                "shape": _shape_of(value),
                "dtype": str(value.dtype) if isinstance(value, torch.Tensor) else type(value).__name__,
            }

    for module_name, activations in payload.get("activations", {}).items():
        summary["activations"][module_name] = {}
        for key, value in activations.items():
            if isinstance(value, list):
                summary["activations"][module_name][key] = {
                    "list_len": len(value),
                    "first_shape": _shape_of(value[0]) if value else None,
                }
            else:
                summary["activations"][module_name][key] = {
                    "shape": _shape_of(value),
                }
    return summary


def payload_summary_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for section in ("data", "outputs"):
        for key, value in payload.get(section, {}).items():
            rows.append(
                {
                    "section": section,
                    "name": key,
                    "shape": _shape_of(value),
                    "dtype": str(value.dtype) if isinstance(value, torch.Tensor) else type(value).__name__,
                }
            )
    for module_name, activations in payload.get("activations", {}).items():
        for key, value in activations.items():
            rows.append(
                {
                    "section": "activations",
                    "name": f"{module_name}.{key}",
                    "shape": _shape_of(value[0]) if isinstance(value, list) and value else _shape_of(value),
                    "dtype": "list" if isinstance(value, list) else type(value).__name__,
                }
            )
    return pd.DataFrame(rows)


def _trim_to_valid_points(array: np.ndarray) -> np.ndarray:
    if array.ndim != 2 or array.shape[1] < 2:
        return array
    finite_mask = np.isfinite(array[:, 0]) & np.isfinite(array[:, 1])
    non_zero_mask = np.any(np.abs(array[:, :2]) > 1e-8, axis=1)
    mask = finite_mask & non_zero_mask
    if mask.any():
        return array[mask]
    return array


def plot_scanpath_overview(payload: dict[str, Any], sample_index: int = 0) -> None:
    data = payload.get("data", {})
    outputs = payload.get("outputs", {})

    eye_input = _tensor_to_numpy(data.get("eye_tracking_input"))
    gt = _tensor_to_numpy(data.get("fixation_ground_truth"))
    decoder = _tensor_to_numpy(data.get("decoder_input_fixations"))
    reg = _tensor_to_numpy(outputs.get("scanpath_output"))
    coord = _tensor_to_numpy(outputs.get("scanpath_coordinates"))
    dur = _tensor_to_numpy(outputs.get("scanpath_duration"))
    cls_logits = _tensor_to_numpy(outputs.get("scanpath_end_logits"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if eye_input is not None:
        noisy = _trim_to_valid_points(np.asarray(eye_input[sample_index])[..., :2])
        axes[0].plot(noisy[:, 0], noisy[:, 1], marker="o", linewidth=1.5, alpha=0.8, label="eye_tracking_input")
    if decoder is not None:
        dec = _trim_to_valid_points(np.asarray(decoder[sample_index])[..., :2])
        axes[0].plot(dec[:, 0], dec[:, 1], marker="x", linewidth=1.2, alpha=0.8, label="decoder_input_fixations")
    if gt is not None:
        target = _trim_to_valid_points(np.asarray(gt[sample_index])[..., :2])
        axes[0].plot(target[:, 0], target[:, 1], marker="s", linewidth=2, alpha=0.9, label="fixation_ground_truth")
    if reg is not None:
        pred = _trim_to_valid_points(np.asarray(reg[sample_index])[..., :2])
        axes[0].plot(pred[:, 0], pred[:, 1], marker="^", linewidth=2, alpha=0.9, label="scanpath_output")
    elif coord is not None:
        pred = _trim_to_valid_points(np.asarray(coord[sample_index])[..., :2])
        axes[0].plot(pred[:, 0], pred[:, 1], marker="^", linewidth=2, alpha=0.9, label="scanpath_coordinates")
    axes[0].set_title("XY Trajectories")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].invert_yaxis()
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.2)

    duration_series: list[tuple[str, np.ndarray]] = []
    if gt is not None and gt.shape[-1] >= 3:
        duration_series.append(("gt_duration", np.asarray(gt[sample_index])[:, 2]))
    if decoder is not None and decoder.shape[-1] >= 3:
        duration_series.append(("decoder_duration", np.asarray(decoder[sample_index])[:, 2]))
    if reg is not None and reg.shape[-1] >= 3:
        duration_series.append(("pred_duration", np.asarray(reg[sample_index])[:, 2]))
    if dur is not None:
        squeezed = np.asarray(dur[sample_index]).squeeze(-1)
        duration_series.append(("pred_duration_head", squeezed))
    for label, series in duration_series:
        axes[1].plot(series, marker="o", linewidth=1.8, label=label)
    axes[1].set_title("Duration Channels")
    axes[1].set_xlabel("fixation step")
    axes[1].set_ylabel("duration")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.2)

    if cls_logits is not None:
        logits = np.asarray(cls_logits[sample_index]).squeeze(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        axes[2].bar(np.arange(len(probs)), probs, color="#d97706", alpha=0.85)
        axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1)
        axes[2].set_ylim(0, 1)
        axes[2].set_title("End-Token Probability")
        axes[2].set_xlabel("fixation step")
        axes[2].set_ylabel("sigmoid(logit)")
        axes[2].grid(alpha=0.2, axis="y")
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "No classification head saved", ha="center", va="center")

    metadata = payload.get("metadata", {})
    fig.suptitle(
        f"split={metadata.get('split')} | phase={metadata.get('phase')} | "
        f"epoch={metadata.get('epoch')} | batch={metadata.get('batch_index')}",
        fontsize=13,
    )
    plt.tight_layout()


def activation_inventory_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for module_name, activations in payload.get("activations", {}).items():
        for name, value in activations.items():
            if isinstance(value, list):
                shape = _shape_of(value[0]) if value else None
                rows.append({"module": module_name, "activation": name, "shape": shape, "captures": len(value)})
            else:
                rows.append({"module": module_name, "activation": name, "shape": _shape_of(value), "captures": 1})
    return pd.DataFrame(rows)


def _pick_activation(payload: dict[str, Any], activation_name: str) -> tuple[str, np.ndarray] | None:
    for module_name, activations in payload.get("activations", {}).items():
        if activation_name not in activations:
            continue
        value = activations[activation_name]
        if isinstance(value, list):
            value = value[0]
        array = _tensor_to_numpy(value)
        if array is not None:
            return module_name, array
    return None


def plot_attention_heatmap(payload: dict[str, Any], sample_index: int = 0, head_index: int = 0) -> None:
    picked = _pick_activation(payload, "attention_weights")
    if picked is None:
        raise ValueError("No attention_weights activation found in payload.")

    module_name, weights = picked
    if weights.ndim == 4:
        matrix = weights[sample_index, head_index]
    elif weights.ndim == 5:
        matrix = weights[sample_index, :, head_index].mean(axis=0)
    else:
        raise ValueError(f"Unsupported attention weight shape: {weights.shape}")

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, aspect="auto", cmap="magma")
    plt.colorbar(label="weight")
    plt.title(f"Attention Heatmap: {module_name} | sample={sample_index} | head={head_index}")
    plt.xlabel("key positions")
    plt.ylabel("query positions")
    plt.tight_layout()


def plot_sampling_locations(payload: dict[str, Any], sample_index: int = 0, query_index: int = 0) -> None:
    locations = _pick_activation(payload, "sampling_locations")
    if locations is None:
        raise ValueError("No sampling_locations activation found in payload.")

    module_name, points = locations
    sample_points = points[sample_index, query_index]
    flat_points = sample_points.reshape(-1, 2)

    plt.figure(figsize=(6, 6))
    plt.scatter(flat_points[:, 0], flat_points[:, 1], s=25, alpha=0.75, c=np.linspace(0, 1, len(flat_points)), cmap="viridis")
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.title(f"Sampling Locations: {module_name} | sample={sample_index} | query={query_index}")
    plt.xlabel("normalized x")
    plt.ylabel("normalized y")
    plt.grid(alpha=0.2)
    plt.tight_layout()


def load_metrics(path: str | Path) -> dict[str, list[float]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def metrics_to_frame(metrics: dict[str, list[float]]) -> pd.DataFrame:
    epochs = metrics.get("epoch", [])
    rows: list[dict[str, Any]] = []
    for metric_name, values in metrics.items():
        if metric_name == "epoch":
            continue
        for index, value in enumerate(values):
            epoch = epochs[index] if index < len(epochs) else index + 1
            rows.append({"metric": metric_name, "epoch": epoch, "value": value})
    return pd.DataFrame(rows)


def plot_metric_groups(metrics: dict[str, list[float]]) -> None:
    frame = metrics_to_frame(metrics)
    groups = {
        "Losses": [name for name in frame["metric"].unique() if "loss" in name],
        "Regression": [name for name in frame["metric"].unique() if "error" in name or "outliers" in name],
        "Classification": [name for name in frame["metric"].unique() if name in {"accuracy", "precision_pos", "recall_pos", "precision_neg", "recall_neg"}],
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, (title, metric_names) in zip(axes, groups.items()):
        subset = frame[frame["metric"].isin(metric_names)]
        if subset.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"No {title.lower()} metrics", ha="center", va="center")
            continue
        for metric_name, metric_frame in subset.groupby("metric"):
            ax.plot(metric_frame["epoch"], metric_frame["value"], marker="o", linewidth=1.8, label=metric_name)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    plt.tight_layout()


def compare_runs(metric_files: list[str | Path], metric_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_file in metric_files:
        metrics = load_metrics(metric_file)
        values = metrics.get(metric_name, [])
        epochs = metrics.get("epoch", [])
        label = str(Path(metric_file).parent.relative_to(PROJECT_ROOT))
        for index, value in enumerate(values):
            epoch = epochs[index] if index < len(epochs) else index + 1
            rows.append({"run": label, "epoch": epoch, "value": value})
    return pd.DataFrame(rows)


def plot_run_comparison(metric_files: list[str | Path], metric_name: str) -> None:
    frame = compare_runs(metric_files, metric_name)
    if frame.empty:
        raise ValueError(f"Metric '{metric_name}' was not found in the selected runs.")
    plt.figure(figsize=(9, 5))
    for run_name, run_frame in frame.groupby("run"):
        plt.plot(run_frame["epoch"], run_frame["value"], marker="o", linewidth=1.8, label=run_name)
    plt.title(f"Run Comparison: {metric_name}")
    plt.xlabel("epoch")
    plt.ylabel(metric_name)
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
