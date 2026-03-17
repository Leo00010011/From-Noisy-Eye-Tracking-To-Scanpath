from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import json

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
COCO_IMAGE_BANK_RELATIVE_PATH = Path("data") / "Coco FreeView" / "all_images_256.pth"
PAD_TOKEN_VALUE = 0.5


def _tensor_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (tuple)):
        return np.asarray(value)
    if isinstance(value, (list)):
        return np.asarray(value[-1])
    return None


def _shape_of(value: Any) -> tuple[int, ...] | None:
    array = _tensor_to_numpy(value)
    if array is not None:
        return tuple(array.shape)
    if isinstance(value, dict):
        return None
    return None


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (PROJECT_ROOT / resolved).resolve()


def _iter_payload_root_candidates(payload_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for parent in [payload_path.parent, *payload_path.parents]:
        if parent not in candidates:
            candidates.append(parent)
    return candidates


def resolve_image_bank_path(
    payload_path: str | Path | None = None,
    image_bank_path: str | Path | None = None,
) -> Path:
    explicit_path = _resolve_path(image_bank_path)
    if explicit_path is not None and explicit_path.exists():
        return explicit_path

    project_candidate = PROJECT_ROOT / COCO_IMAGE_BANK_RELATIVE_PATH
    if project_candidate.exists():
        return project_candidate

    resolved_payload_path = _resolve_path(payload_path)
    if resolved_payload_path is not None:
        for candidate_root in _iter_payload_root_candidates(resolved_payload_path):
            candidate = candidate_root / COCO_IMAGE_BANK_RELATIVE_PATH
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Unable to locate the image bank. Provide `image_bank_path` explicitly or place "
        f"`{COCO_IMAGE_BANK_RELATIVE_PATH}` under the active project root."
    )


@lru_cache(maxsize=2)
def _load_image_bank_cached(image_bank_path: str) -> torch.Tensor:
    return torch.load(image_bank_path, map_location="cpu", weights_only=False)


def load_image_bank(
    payload_path: str | Path | None = None,
    image_bank_path: str | Path | None = None,
) -> torch.Tensor:
    resolved_path = resolve_image_bank_path(payload_path=payload_path, image_bank_path=image_bank_path)
    return _load_image_bank_cached(str(resolved_path))


def get_payload_image(
    payload: dict[str, Any],
    sample_index: int = 0,
    payload_path: str | Path | None = None,
    image_bank_path: str | Path | None = None,
) -> np.ndarray:
    image_indices = payload.get("data", {}).get("image_idx")
    if image_indices is None:
        raise KeyError("Payload does not contain `image_idx` in the data section.")

    image_index_array = _tensor_to_numpy(image_indices)
    if image_index_array is None:
        raise ValueError("`image_idx` could not be converted to numpy.")

    image_index = int(image_index_array[sample_index])
    image_bank = load_image_bank(payload_path=payload_path, image_bank_path=image_bank_path)
    image_tensor = image_bank[image_index]
    return image_tensor.permute(1, 2, 0).detach().cpu().numpy()


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


def _infer_valid_fixation_count(fixations: np.ndarray, pad_value: float = PAD_TOKEN_VALUE) -> int:
    if fixations.ndim != 2 or fixations.shape[1] < 3:
        return fixations.shape[0]
    pad_rows = np.isclose(fixations, pad_value, atol=1e-6).all(axis=1)
    pad_indices = np.flatnonzero(pad_rows)
    if pad_indices.size == 0:
        return fixations.shape[0]
    return int(pad_indices[0])


def get_decoder_reference_points(payload: dict[str, Any], sample_index: int = 0) -> np.ndarray:
    fixations = _tensor_to_numpy(payload.get("data", {}).get("fixation_ground_truth"))
    if fixations is None:
        raise KeyError("Payload does not contain `fixation_ground_truth` in the data section.")
    fixation_points = np.asarray(fixations[sample_index, :, :2], dtype=np.float32)
    start_point = np.array([[0.5, 0.5]], dtype=np.float32)
    return np.concatenate([start_point, fixation_points], axis=0)


def get_decoder_layer_activations(
    payload: dict[str, Any],
    decoder_layer: int,
) -> tuple[str, np.ndarray, np.ndarray]:
    module_name = f"decoder.{decoder_layer}.second_cross_attn"
    activations = payload.get("activations", {}).get(module_name)
    if activations is None:
        available = sorted(name for name in payload.get("activations", {}) if name.startswith("decoder."))
        raise KeyError(f"Module `{module_name}` not found. Available decoder modules: {available}")
    print("SAMPLING LOCATIONS")
    print(type(activations.get("sampling_locations")))
    print(len(activations.get("sampling_locations")))
    for i in range(len(activations.get("sampling_locations"))):
        print(activations.get("sampling_locations")[i].shape)
    sampling_locations = _tensor_to_numpy(activations.get("sampling_locations"))
    attention_weights = _tensor_to_numpy(activations.get("attention_weights"))
    if sampling_locations is None or attention_weights is None:
        raise KeyError(f"Module `{module_name}` is missing `sampling_locations` or `attention_weights`.")
    return module_name, np.asarray(sampling_locations), np.asarray(attention_weights)


def get_decoder_ground_truth_vector(
    payload: dict[str, Any],
    sample_index: int,
    query_index: int,
) -> dict[str, Any] | None:
    fixations = _tensor_to_numpy(payload.get("data", {}).get("fixation_ground_truth"))
    if fixations is None:
        raise KeyError("Payload does not contain `fixation_ground_truth` in the data section.")

    sample_fixations = np.asarray(fixations[sample_index], dtype=np.float32)
    valid_fixation_count = _infer_valid_fixation_count(sample_fixations)
    if valid_fixation_count <= 0 or query_index >= valid_fixation_count:
        return None

    if query_index == 0:
        origin = np.array([0.5, 0.5], dtype=np.float32)
        target = sample_fixations[0, :2]
    else:
        origin = sample_fixations[query_index - 1, :2]
        target = sample_fixations[query_index, :2]

    return {
        "origin": origin,
        "target": target,
        "delta": target - origin,
        "valid_fixation_count": valid_fixation_count,
    }


def extract_decoder_deformable_attention(
    payload: dict[str, Any],
    sample_index: int = 0,
    decoder_layer: int = 0,
    query_index: int = 0,
    aggregate_heads: bool = True,
    head_index: int | None = None,
) -> dict[str, Any]:
    module_name, sampling_locations, attention_weights = get_decoder_layer_activations(payload, decoder_layer=decoder_layer)
    if sample_index < 0 or sample_index >= sampling_locations.shape[0]:
        raise IndexError(f"sample_index={sample_index} is out of bounds for batch size {sampling_locations.shape[0]}.")
    if query_index < 0 or query_index >= sampling_locations.shape[1]:
        raise IndexError(f"query_index={query_index} is out of bounds for decoder length {sampling_locations.shape[1]}.")

    sample_locations = np.asarray(sampling_locations[sample_index, query_index], dtype=np.float32)
    sample_weights = np.asarray(attention_weights[sample_index, query_index], dtype=np.float32)
    reference_points = get_decoder_reference_points(payload, sample_index=sample_index)
    if query_index >= reference_points.shape[0]:
        raise IndexError(
            f"query_index={query_index} is out of bounds for reconstructed reference points "
            f"with shape {reference_points.shape}."
        )

    origin = np.asarray(reference_points[query_index], dtype=np.float32)

    vectors: np.ndarray
    vector_weights: np.ndarray
    vector_labels: list[str]
    selected_head_index = head_index

    if aggregate_heads:
        summed_weights = sample_weights.sum(axis=0, keepdims=True)
        normalized_weights = np.divide(
            sample_weights,
            np.where(summed_weights == 0, 1.0, summed_weights),
        )
        vectors = (sample_locations * normalized_weights[..., None]).sum(axis=0)
        vector_weights = sample_weights.mean(axis=0)
        vector_labels = [f"point_{point_idx}" for point_idx in range(vectors.shape[0])]
        selected_head_index = None
    else:
        if head_index is None:
            vectors = sample_locations.reshape(-1, 2)
            vector_weights = sample_weights.reshape(-1)
            vector_labels = [
                f"head_{head_idx}_point_{point_idx}"
                for head_idx in range(sample_locations.shape[0])
                for point_idx in range(sample_locations.shape[1])
            ]
        else:
            if head_index < 0 or head_index >= sample_locations.shape[0]:
                raise IndexError(
                    f"head_index={head_index} is out of bounds for number of heads {sample_locations.shape[0]}."
                )
            vectors = sample_locations[head_index]
            vector_weights = sample_weights[head_index]
            vector_labels = [f"head_{head_index}_point_{point_idx}" for point_idx in range(vectors.shape[0])]

    valid_fixation_count = _infer_valid_fixation_count(
        np.asarray(_tensor_to_numpy(payload.get("data", {}).get("fixation_ground_truth"))[sample_index], dtype=np.float32)
    )
    true_vector = get_decoder_ground_truth_vector(payload, sample_index=sample_index, query_index=query_index)

    return {
        "module_name": module_name,
        "origin": origin,
        "sampling_locations": sample_locations,
        "attention_weights": sample_weights,
        "vector_targets": np.asarray(vectors, dtype=np.float32),
        "vector_deltas": np.asarray(vectors, dtype=np.float32) - origin[None, :],
        "vector_weights": np.asarray(vector_weights, dtype=np.float32),
        "vector_labels": vector_labels,
        "true_vector": true_vector,
        "valid_fixation_count": valid_fixation_count,
        "is_terminal_or_padded": query_index >= valid_fixation_count,
        "selected_head_index": selected_head_index,
        "query_count": int(sampling_locations.shape[1]),
        "num_heads": int(sampling_locations.shape[2]),
        "num_points": int(sampling_locations.shape[3]),
    }


def decoder_attention_summary_frame(
    payload: dict[str, Any],
    sample_index: int = 0,
    decoder_layer: int = 0,
    query_index: int = 0,
    aggregate_heads: bool = True,
    head_index: int | None = None,
) -> pd.DataFrame:
    attention_info = extract_decoder_deformable_attention(
        payload,
        sample_index=sample_index,
        decoder_layer=decoder_layer,
        query_index=query_index,
        aggregate_heads=aggregate_heads,
        head_index=head_index,
    )
    rows: list[dict[str, Any]] = []
    for label, target, delta, weight in zip(
        attention_info["vector_labels"],
        attention_info["vector_targets"],
        attention_info["vector_deltas"],
        attention_info["vector_weights"],
    ):
        rows.append(
            {
                "label": label,
                "target_x": float(target[0]),
                "target_y": float(target[1]),
                "delta_x": float(delta[0]),
                "delta_y": float(delta[1]),
                "attention_weight": float(weight),
            }
        )
    return pd.DataFrame(rows)


def _compute_decoder_plot_limits(
    origin: np.ndarray,
    vector_targets: np.ndarray,
    true_vector: dict[str, Any] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [0.0, 1.0, float(origin[0]), *vector_targets[:, 0].tolist()]
    ys = [0.0, 1.0, float(origin[1]), *vector_targets[:, 1].tolist()]
    if true_vector is not None:
        xs.append(float(true_vector["target"][0]))
        ys.append(float(true_vector["target"][1]))

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    margin = 0.05
    if x_min < 0.0 or x_max > 1.0:
        x_limits = (x_min - margin, x_max + margin)
    else:
        x_limits = (0.0, 1.0)
    if y_min < 0.0 or y_max > 1.0:
        y_limits = (y_max + margin, y_min - margin)
    else:
        y_limits = (1.0, 0.0)
    return x_limits, y_limits


def plot_decoder_deformable_attention_overlay(
    payload: dict[str, Any],
    sample_index: int = 0,
    decoder_layer: int = 0,
    query_index: int = 0,
    aggregate_heads: bool = True,
    head_index: int | None = None,
    payload_path: str | Path | None = None,
    image_bank_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> dict[str, Any]:
    attention_info = extract_decoder_deformable_attention(
        payload,
        sample_index=sample_index,
        decoder_layer=decoder_layer,
        query_index=query_index,
        aggregate_heads=aggregate_heads,
        head_index=head_index,
    )
    image = get_payload_image(
        payload,
        sample_index=sample_index,
        payload_path=payload_path,
        image_bank_path=image_bank_path,
    )
    image_indices = _tensor_to_numpy(payload.get("data", {}).get("image_idx"))
    sample_indices = _tensor_to_numpy(payload.get("data", {}).get("sample_idx"))
    image_index = int(image_indices[sample_index]) if image_indices is not None else None
    sample_id = int(sample_indices[sample_index]) if sample_indices is not None else None

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(image, extent=(0, 1, 1, 0), interpolation="nearest")

    vector_weights = attention_info["vector_weights"]
    if vector_weights.size == 0:
        raise ValueError("No decoder vectors are available to plot.")

    min_weight = float(vector_weights.min())
    max_weight = float(vector_weights.max())
    if np.isclose(min_weight, max_weight):
        max_weight = min_weight + 1e-6
    color_norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
    cmap = plt.cm.viridis
    vector_targets = attention_info["vector_targets"]
    origin = attention_info["origin"]

    for target, weight in zip(vector_targets, vector_weights):
        ax.annotate(
            "",
            xy=(float(target[0]), float(target[1])),
            xytext=(float(origin[0]), float(origin[1])),
            arrowprops={
                "arrowstyle": "->",
                "color": cmap(color_norm(float(weight))),
                "lw": 2,
                "alpha": 0.95,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=3,
        )

    out_of_bounds_mask = (
        (vector_targets[:, 0] < 0.0)
        | (vector_targets[:, 0] > 1.0)
        | (vector_targets[:, 1] < 0.0)
        | (vector_targets[:, 1] > 1.0)
    )
    if np.any(~out_of_bounds_mask):
        in_bounds = vector_targets[~out_of_bounds_mask]
        ax.scatter(
            in_bounds[:, 0],
            in_bounds[:, 1],
            c=vector_weights[~out_of_bounds_mask],
            cmap=cmap,
            norm=color_norm,
            s=55,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
    if np.any(out_of_bounds_mask):
        out_bounds = vector_targets[out_of_bounds_mask]
        ax.scatter(
            out_bounds[:, 0],
            out_bounds[:, 1],
            marker="x",
            color="#dc2626",
            s=65,
            linewidths=1.6,
            zorder=5,
            label="out-of-bounds sample",
        )

    ax.scatter(
        [float(origin[0])],
        [float(origin[1])],
        marker="o",
        s=100,
        color="#f97316",
        edgecolors="black",
        linewidths=0.9,
        zorder=6,
        label="decoder origin",
    )

    true_vector = attention_info["true_vector"]
    if true_vector is not None:
        target = true_vector["target"]
        ax.annotate(
            "",
            xy=(float(target[0]), float(target[1])),
            xytext=(float(origin[0]), float(origin[1])),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#ef4444",
                "lw": 2.5,
                "linestyle": "--",
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=7,
        )
        ax.scatter(
            [float(target[0])],
            [float(target[1])],
            marker="D",
            s=60,
            color="#ef4444",
            edgecolors="white",
            linewidths=0.8,
            zorder=8,
            label="ground-truth next fixation",
        )

    x_limits, y_limits = _compute_decoder_plot_limits(origin, vector_targets, true_vector)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("normalized x")
    ax.set_ylabel("normalized y")
    ax.grid(alpha=0.18)

    head_label = "aggregated heads" if aggregate_heads else (
        f"head {head_index}" if head_index is not None else "all heads"
    )
    status_label = "terminal/padded query" if attention_info["is_terminal_or_padded"] else "valid next-step query"
    ax.set_title(
        f"{attention_info['module_name']} | sample={sample_index} | query={query_index} | {head_label}\n"
        f"image_idx={image_index} | sample_idx={sample_id} | {status_label}"
    )
    scalar_mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=cmap)
    scalar_mappable.set_array([])
    plt.colorbar(scalar_mappable, ax=ax, fraction=0.046, pad=0.04, label="attention weight")
    ax.legend(loc="upper right")
    return attention_info


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
