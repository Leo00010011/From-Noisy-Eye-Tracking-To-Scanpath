from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

import torch


def _sanitize_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _to_serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_to_serializable(item) for item in value]
        return tuple(converted) if isinstance(value, tuple) else converted
    return value


@dataclass
class RecorderContext:
    epoch: int
    phase: str
    split: str
    batch_index: int
    global_step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceRecorder:
    def __init__(self, output_dir: str | Path, enabled: bool = True):
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_context: RecorderContext | None = None
        self.current_payload: dict[str, Any] | None = None

    def attach(self, model: torch.nn.Module) -> None:
        for module_name, module in model.named_modules():
            resolved_name = module_name or model.__class__.__name__
            setattr(module, "_inference_recorder", self)
            setattr(module, "_inference_recorder_module_name", resolved_name)

    def start_batch(
        self,
        epoch: int,
        phase: str,
        split: str,
        batch_index: int,
        global_step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.current_context = RecorderContext(
            epoch=epoch,
            phase=phase,
            split=split,
            batch_index=batch_index,
            global_step=global_step,
            metadata={} if metadata is None else dict(metadata),
        )
        self.current_payload = {
            "metadata": {
                "epoch": epoch,
                "phase": phase,
                "split": split,
                "batch_index": batch_index,
                "global_step": global_step,
                **({} if metadata is None else dict(metadata)),
            },
            "data": {},
            "outputs": {},
            "activations": {},
        }

    def is_active(self) -> bool:
        return self.enabled and self.current_payload is not None

    def record_activation(self, module: torch.nn.Module, name: str, value: Any) -> None:
        if not self.is_active():
            return
        module_name = getattr(module, "_inference_recorder_module_name", module.__class__.__name__)
        module_bucket = self.current_payload["activations"].setdefault(module_name, {})
        serialized_value = _to_serializable(value)
        if name not in module_bucket:
            module_bucket[name] = serialized_value
            return
        if not isinstance(module_bucket[name], list):
            module_bucket[name] = [module_bucket[name]]
        module_bucket[name].append(serialized_value)

    def record_batch(
        self,
        batch: dict[str, Any],
        output: dict[str, Any],
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.is_active():
            return

        data_payload: dict[str, Any] = {}
        if "sample_idx" in batch:
            data_payload["sample_idx"] = _to_serializable(batch["sample_idx"])
        if "image_idx" in batch:
            data_payload["image_idx"] = _to_serializable(batch["image_idx"])
        if "src" in batch:
            data_payload["eye_tracking_input"] = _to_serializable(batch["src"])
        if "tgt" in batch:
            data_payload["fixation_ground_truth"] = _to_serializable(batch["tgt"])
        if "in_tgt" in batch:
            data_payload["decoder_input_fixations"] = _to_serializable(batch["in_tgt"])

        output_payload: dict[str, Any] = {}
        if "denoise" in output:
            output_payload["denoise_output"] = _to_serializable(output["denoise"])
        if "coord" in output:
            output_payload["scanpath_coordinates"] = _to_serializable(output["coord"])
        if "dur" in output:
            output_payload["scanpath_duration"] = _to_serializable(output["dur"])
        if "reg" in output:
            output_payload["scanpath_output"] = _to_serializable(output["reg"])
        if "cls" in output:
            output_payload["scanpath_end_logits"] = _to_serializable(output["cls"])

        self.current_payload["data"].update(data_payload)
        self.current_payload["outputs"].update(output_payload)
        if extra_metadata:
            self.current_payload["metadata"].update(_to_serializable(extra_metadata))

    def save_batch(self) -> Path | None:
        if not self.is_active() or self.current_context is None:
            return None

        metadata = self.current_payload["metadata"]
        step = metadata.get("global_step")
        step_fragment = f"_step_{int(step):06d}" if step is not None else ""
        filename = (
            f"{_sanitize_fragment(metadata['split'])}"
            f"_epoch_{int(metadata['epoch']):04d}"
            f"_batch_{int(metadata['batch_index']):05d}"
            f"{step_fragment}"
            f"_{_sanitize_fragment(metadata['phase'])}.pt"
        )
        output_path = self.output_dir / filename
        torch.save(self.current_payload, output_path)
        self.current_context = None
        self.current_payload = None
        return output_path

    def clear(self) -> None:
        self.current_context = None
        self.current_payload = None


def record_module_value(module: torch.nn.Module, name: str, value: Any) -> None:
    recorder = getattr(module, "_inference_recorder", None)
    if recorder is None:
        return
    recorder.record_activation(module, name, value)
