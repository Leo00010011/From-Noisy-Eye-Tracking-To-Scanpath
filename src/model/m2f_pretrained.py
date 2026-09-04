"""Pretrained-weight loaders for the vendored :class:`Mask2FormerBackbone`.

Populates the F2 backbone from the two detectron2-format checkpoints the user provided:

* ``M2F_R50.pkl`` — a COCO-panoptic-tuned ResNet50 stored in **detectron2 naming**
  (``stem.*``, ``stages.res{2,3,4,5}.*``, ``.norm.``, ``shortcut``). It is numerically a
  torchvision-style ResNet (stride on the 3×3 conv, RGB, ImageNet norm), so only a **key
  remap** into torchvision naming (``conv1``, ``bn1``, ``layer{1..4}``, ``downsample``) is
  needed; the values transfer unchanged.
* ``M2F_R50_MSDeformAttnPixelDecoder.pkl`` — the pixel decoder, whose keys are already 1:1
  with the vendored ``MSDeformAttnPixelDecoder`` (``input_proj.*``,
  ``transformer.level_embed``, ``transformer.encoder.layers.{i}.self_attn.*``,
  ``transformer.encoder.layers.{i}.{norm1,linear1,linear2,norm2}.*``, plus ``mask_features.*``
  and the detectron2 FPN convs ``adapter_1.*`` / ``layer_1.*`` when present). They only need a
  ``pixel_decoder.`` prefix.

Pure PyTorch: plain key remaps over ``torch.load``-ed dicts, no detectron2/fvcore/CUDA op.
"""

import re
from dataclasses import dataclass, field

import torch

# detectron2 stage name -> torchvision layer name.
_STAGE = {"res2": "layer1", "res3": "layer2", "res4": "layer3", "res5": "layer4"}


def remap_detectron2_resnet50(sd: dict) -> dict:
    """Convert a detectron2 R50 state dict into torchvision ``resnet50`` naming (FR1).

    Values are passed through unchanged (no dtype/shape mutation). Any key not matching a
    known detectron2 pattern is dropped and collected into a printed warning.
    """
    out, dropped = {}, []
    for k, v in sd.items():
        if k == "stem.conv1.weight":
            out["conv1.weight"] = v
            continue
        m = re.match(r"stem\.conv1\.norm\.(.+)", k)
        if m:
            out[f"bn1.{m.group(1)}"] = v
            continue
        m = re.match(r"stages\.(res\d)\.(\d+)\.(.+)", k)
        if m:
            stage, blk, rest = _STAGE[m.group(1)], m.group(2), m.group(3)
            pfx = f"{stage}.{blk}."
            r = re.match(r"conv(\d)\.norm\.(.+)", rest)
            if r:
                out[pfx + f"bn{r.group(1)}.{r.group(2)}"] = v
                continue
            r = re.match(r"conv(\d)\.weight", rest)
            if r:
                out[pfx + f"conv{r.group(1)}.weight"] = v
                continue
            if rest == "shortcut.weight":
                out[pfx + "downsample.0.weight"] = v
                continue
            r = re.match(r"shortcut\.norm\.(.+)", rest)
            if r:
                out[pfx + f"downsample.1.{r.group(1)}"] = v
                continue
        dropped.append(k)
    if dropped:
        print(f"[remap R50] dropped {len(dropped)} unrecognized keys: {dropped[:5]}...")
    return out


def remap_pixel_decoder(sd: dict, prefix: str = "pixel_decoder.") -> dict:
    """Prepend ``prefix`` to every pixel-decoder key; change nothing else (FR2)."""
    return {prefix + k: v for k, v in sd.items()}


@dataclass
class LoadReport:
    missing_keys: list = field(default_factory=list)
    unexpected_keys: list = field(default_factory=list)
    n_resnet_loaded: int = 0
    n_pixdec_loaded: int = 0


def _unwrap(obj):
    """Tolerate ``{"model": state_dict}`` wrappers around a raw state dict."""
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj


def load_pretrained_mask2former(backbone, r50_path: str,
                                pixel_decoder_path: str) -> LoadReport:
    """Load both remapped checkpoints into ``backbone`` (FR3).

    Uses ``load_state_dict(strict=False)`` and returns a :class:`LoadReport`. Raises
    ``RuntimeError`` if a systematically broken remap leaves any ResNet50 *parameter*
    (``feature_extractor.*``) unloaded, or any core pixel-decoder *parameter*
    (``pixel_decoder.transformer.*`` / ``pixel_decoder.input_proj.*``) missing.

    Tolerated (reported, not raised): torchvision's ``fc.*`` (dropped by
    ``create_feature_extractor``); BatchNorm ``num_batches_tracked`` buffers (absent from
    detectron2 ``FrozenBatchNorm2d`` checkpoints — they are non-parameter buffers, harmless in
    frozen ``eval()``); and the stride-4 FPN / ``mask_features`` keys when the backbone was
    built without that branch. The detectron2 ``FrozenBatchNorm2d`` and torchvision
    ``BatchNorm2d`` (frozen, ``eval()``) apply the same affine of the running stats, so
    ``weight/bias/running_mean/running_var`` transfer directly (≤1e-5 eps discrepancy accepted).
    """
    r50 = _unwrap(torch.load(r50_path, map_location="cpu", weights_only=False))
    pdec = _unwrap(torch.load(pixel_decoder_path, map_location="cpu", weights_only=False))

    fe_sd = {f"feature_extractor.{k}": v for k, v in remap_detectron2_resnet50(r50).items()}
    pd_sd = remap_pixel_decoder(pdec)
    combined = {**fe_sd, **pd_sd}

    missing, unexpected = backbone.load_state_dict(combined, strict=False)

    # Guard on *parameters* only — buffers (num_batches_tracked) are legitimately absent from
    # detectron2 checkpoints and must not trigger a false "broken remap" error.
    param_names = {n for n, _ in backbone.named_parameters()}
    bad_fe = [k for k in missing if k in param_names and k.startswith("feature_extractor.")]
    bad_pd = [k for k in missing if k in param_names and (
        k.startswith("pixel_decoder.transformer.") or k.startswith("pixel_decoder.input_proj."))]
    if bad_fe:
        raise RuntimeError(
            f"ResNet50 remap incomplete; {len(bad_fe)} params unloaded: {bad_fe[:5]}")
    if bad_pd:
        raise RuntimeError(
            f"Pixel-decoder load incomplete; {len(bad_pd)} params unloaded: {bad_pd[:5]}")
    return LoadReport(list(missing), list(unexpected), len(fe_sd), len(pd_sd))
