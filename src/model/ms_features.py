"""F3 — Multi-scale feature contract / backbone adapter.

Defines one backbone-agnostic image-feature contract — the :class:`MultiScaleFeatures`
bundle — that both image backbones produce and that every downstream consumer (the F1
``DeformableAttention`` op, the F4 decoders, and the F6 ``MixerModel``) reads. Today
``MixerModel`` is welded to DINOv3 specifics (a ``mem[:, 1:, :]`` CLS strip, a single square
``patch_resolution`` tuple, direct ``image_encoder.model.patch_size`` / ``.rope_embed`` access).
F3 replaces that implicit, single-scale, CLS-prefixed convention with an explicit, N-level,
CLS-free bundle carrying the flattened multi-level memory plus the geometry needed to sample
and position-encode it.

F3 ships the bundle and its two producing adapters only — it modifies no existing file. The
migration of ``MixerModel`` / the decoders to *consume* the bundle is F6 / F4.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Step 1 — Geometry helpers (shared by both adapters, reusable by F6).
# ---------------------------------------------------------------------------
def build_level_start_index(spatial_shapes: torch.Tensor) -> torch.Tensor:
    """``(L, 2)`` int ``spatial_shapes`` -> ``(L,)`` int64 ``[0, H0W0, H0W0+H1W1, ...]``.

    The last level's start is excluded from the tail, matching the F1 ``DeformableAttention``
    derivation (Deformable-DETR convention). Returned on ``spatial_shapes.device``.
    """
    sizes = spatial_shapes[:, 0] * spatial_shapes[:, 1]                       # (L,)
    return torch.cat([spatial_shapes.new_zeros((1,)), sizes.cumsum(0)[:-1]]).to(torch.int64)


def build_reference_grids(spatial_shapes: torch.Tensor,
                          device="cpu", dtype=torch.float32) -> torch.Tensor:
    """``(L, 2)`` ``spatial_shapes`` -> ``(S, 2)`` per-token normalized ``(x, y)`` centers.

    For a level ``(H, W)`` token ``h*W + w`` maps to ``(x=(w+0.5)/W, y=(h+0.5)/H)``. Levels
    are concatenated in ``spatial_shapes`` row order; within a level tokens are row-major
    (``W`` fastest), matching ``map.flatten(2).transpose(1, 2)``. Byte-identical (up to the
    level-broadcast repeat and leading batch axis) to
    ``MSDeformAttnTransformerEncoder.get_reference_points``.
    """
    grids = []
    for (H, W) in spatial_shapes.tolist():
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=device, dtype=dtype) / H,
            torch.linspace(0.5, W - 0.5, W, device=device, dtype=dtype) / W,
            indexing="ij")                                                   # each (H, W)
        grids.append(torch.stack((ref_x.reshape(-1), ref_y.reshape(-1)), dim=-1))  # (H*W, 2)
    return torch.cat(grids, dim=0)                                           # (S, 2)


# ---------------------------------------------------------------------------
# Step 2 — MultiScaleFeatures bundle.
# ---------------------------------------------------------------------------
@dataclass
class MultiScaleFeatures:
    """Backbone-agnostic multi-scale image-feature bundle.

    Fields (``S := Σₗ Hₗ·Wₗ``, ``L := n_levels``):
      * ``value`` — ``(B, S, D)`` float. Flattened multi-level memory, levels concatenated in
        the producer's order (coarse→fine for Mask2Former), tokens row-major within a level.
      * ``spatial_shapes`` — ``(L, 2)`` int64, row ``l`` is ``(Hₗ, Wₗ)``.
      * ``level_start_index`` — ``(L,)`` int64, ``[0, H0W0, H0W0+H1W1, ...]``.
      * ``reference_grids`` — ``(S, 2)`` float, normalized ``(x, y)`` center of every token
        in the same order as ``value``.
    """

    value: torch.Tensor
    spatial_shapes: torch.Tensor
    level_start_index: torch.Tensor
    reference_grids: torch.Tensor

    def __post_init__(self):
        if self.value.dim() != 3:
            raise ValueError(f"value must be (B, S, D); got {tuple(self.value.shape)}")
        if not torch.is_floating_point(self.value):
            raise ValueError(f"value must be a floating tensor; got {self.value.dtype}")
        if self.spatial_shapes.dim() != 2 or self.spatial_shapes.shape[1] != 2:
            raise ValueError(
                f"spatial_shapes must be (L, 2); got {tuple(self.spatial_shapes.shape)}")
        L = self.spatial_shapes.shape[0]
        for name, t in (("spatial_shapes", self.spatial_shapes),
                        ("level_start_index", self.level_start_index)):
            if t.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must be an integer tensor; got {t.dtype}")
        if self.level_start_index.shape != (L,):
            raise ValueError(
                f"level_start_index must be (L,)=({L},); got {tuple(self.level_start_index.shape)}")
        S = int((self.spatial_shapes[:, 0] * self.spatial_shapes[:, 1]).sum())
        if self.value.shape[1] != S:
            raise ValueError(f"Σ Hₗ·Wₗ ({S}) != value length ({self.value.shape[1]})")
        if tuple(self.reference_grids.shape) != (S, 2):
            raise ValueError(
                f"reference_grids must be (S, 2)=({S}, 2); got {tuple(self.reference_grids.shape)}")
        expected = build_level_start_index(self.spatial_shapes)
        if self.level_start_index.tolist() != expected.tolist():
            raise ValueError(
                f"level_start_index {self.level_start_index.tolist()} inconsistent with "
                f"spatial_shapes (expected {expected.tolist()})")

    @property
    def num_levels(self) -> int:
        return self.spatial_shapes.shape[0]

    @property
    def embed_dim(self) -> int:
        return self.value.shape[-1]

    @property
    def batch_size(self) -> int:
        return self.value.shape[0]

    @property
    def seq_len(self) -> int:
        return self.value.shape[1]

    def level_sizes(self) -> list:
        return (self.spatial_shapes[:, 0] * self.spatial_shapes[:, 1]).tolist()

    def to(self, device=None, dtype=None) -> "MultiScaleFeatures":
        """Return a new bundle. ``device`` moves all four tensors; ``dtype`` casts **only**
        the float tensors (``value``, ``reference_grids``) — index tensors stay int64.

        ``Tensor.to(device=d, dtype=None)`` moves device only, so passing neither returns a
        bundle whose tensors are unchanged references.
        """
        return MultiScaleFeatures(
            value=self.value.to(device=device, dtype=dtype),
            spatial_shapes=self.spatial_shapes.to(device=device),
            level_start_index=self.level_start_index.to(device=device),
            reference_grids=self.reference_grids.to(device=device, dtype=dtype),
        )


# ---------------------------------------------------------------------------
# Step 3 — Mask2FormerFeatureAdapter (wraps the F2 backbone).
# ---------------------------------------------------------------------------
class Mask2FormerFeatureAdapter(nn.Module):
    """Wraps a built :class:`Mask2FormerBackbone` and packages its coarse→fine
    ``[B, 256, Hₗ, Wₗ]`` maps into a :class:`MultiScaleFeatures` bundle.

    Adds no parameters of its own; the backbone is held as a submodule (keys prefixed
    ``backbone.``) so ``.to()`` / ``.train()`` / ``.state_dict()`` / recorder hooks propagate.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim          # 256
        self.num_levels = backbone.num_levels        # 3 (or 4 if return_stride4)

    def forward(self, x) -> MultiScaleFeatures:
        out = self.backbone(x)
        # return_stride4 backbones return (maps, mask_features); take maps, drop mask_features.
        maps = out[0] if self.backbone.return_stride4 else out
        flats, shapes = [], []
        for m in maps:                               # coarse→fine [B, 256, Hₗ, Wₗ]
            B, C, H, W = m.shape
            flats.append(m.flatten(2).transpose(1, 2))           # (B, H*W, C)
            shapes.append((H, W))
        value = torch.cat(flats, dim=1)              # (B, S, 256)
        spatial_shapes = torch.as_tensor(shapes, dtype=torch.int64, device=value.device)
        return MultiScaleFeatures(
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=build_level_start_index(spatial_shapes),
            reference_grids=build_reference_grids(spatial_shapes, value.device, value.dtype),
        )


# ---------------------------------------------------------------------------
# Step 4 — DinoV3FeatureAdapter (wraps DinoV3Wrapper).
# ---------------------------------------------------------------------------
class DinoV3FeatureAdapter(nn.Module):
    """Wraps a built :class:`DinoV3Wrapper`, strips the CLS prefix once, and emits a 1-level
    :class:`MultiScaleFeatures` bundle.

    ``backbone.model.patch_size`` is read **once at init** (the sole DINOv3-internal attribute
    access after F4/F6). Adds no parameters of its own.
    """

    def __init__(self, backbone, num_prefix_tokens: int = 1):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim          # 384
        self.num_levels = 1
        self.num_prefix_tokens = num_prefix_tokens
        self.patch_size = backbone.model.patch_size  # sole DINOv3-internal access, init-time

    def forward(self, x) -> MultiScaleFeatures:
        tokens = self.backbone(x)                    # (B, prefix + H'W', D), CLS-prefixed
        value = tokens[:, self.num_prefix_tokens:, :]           # strip prefix ONCE
        Hs = x.shape[-2] // self.patch_size
        Ws = x.shape[-1] // self.patch_size
        if Hs * Ws != value.shape[1]:
            raise ValueError(
                f"patch grid {Hs}x{Ws}={Hs * Ws} != token count {value.shape[1]} "
                f"(check patch_size / num_prefix_tokens)")
        spatial_shapes = torch.as_tensor([[Hs, Ws]], dtype=torch.int64, device=value.device)
        return MultiScaleFeatures(
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=build_level_start_index(spatial_shapes),   # [0]
            reference_grids=build_reference_grids(spatial_shapes, value.device, value.dtype),
        )
