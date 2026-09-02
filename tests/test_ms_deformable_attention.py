"""F1 — Multi-scale Deformable Attention Primitive.

Tests for the generalized ``DeformableAttention`` (``src/model/blocks.py``): the
``n_levels=1`` byte-identity / retro-compat contract, multi-level forward correctness,
input polymorphism / error conditions, and the guarantee that F1 leaves the existing
decoders untouched. CPU-only, fixed seeds, small dummy tensors.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.blocks import (
    DeformableAttention,
    DeformableDecoder,
    DeformableDoubleInputDecoder,
)
from src.training.inference_recorder import InferenceRecorder


# Reference dims (validation.md)
EMBED_DIM = 32
NUM_HEADS = 4
NUM_POINTS = 4
B = 2
NQ = 5


# ---------------------------------------------------------------------------
# Pre-F1 reference implementation (the "old" single-scale DeformableAttention).
# Reproduced verbatim so the byte-identity tests have something to compare to.
# ---------------------------------------------------------------------------
class LegacyDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_points=4, attn_dropout=0.0,
                 geometric_sigma=0, normalize_grid_init=True, device='cpu', dtype=torch.float32):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.device = device
        self.dtype = dtype
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        self.geometric_sigma = geometric_sigma
        self.normalize_grid_init = normalize_grid_init
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2, **factory_kwargs)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points, **factory_kwargs)
        self.value_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.output_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.dropout = nn.Dropout(attn_dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=self.dtype, device=self.device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        if self.normalize_grid_init:
            grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.unsqueeze(1).repeat(1, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.xavier_uniform_(self.output_proj.weight.data)

    def forward(self, query, reference_points, value, spatial_shape, level_start_index=None):
        # level_start_index accepted for signature parity with the F4 decoders (which now
        # forward it) but ignored — this single-scale reference has no level structure.
        bs, num_queries, _ = query.shape
        H, W = spatial_shape
        value = self.value_proj(value)
        value = value.view(bs, H * W, self.num_heads, self.head_dim)
        value = value.permute(0, 2, 3, 1).view(bs, self.num_heads, self.head_dim, H, W)
        sampling_offsets = self.sampling_offsets(query).view(bs, num_queries, self.num_heads, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(bs, num_queries, self.num_heads, self.num_points)
        attention_weights = F.softmax(attention_weights, -1)
        attention_weights = self.dropout(attention_weights)
        if self.training and self.geometric_sigma > 0:
            sampling_offsets = sampling_offsets + torch.randn_like(sampling_offsets) * self.geometric_sigma
        offset_normalizer = torch.tensor([W, H], device=query.device, dtype=query.dtype)
        sampling_locations = reference_points[:, :, None, None, :] \
            + sampling_offsets / offset_normalizer[None, None, None, None, :]
        value_flat = value.reshape(bs * self.num_heads, self.head_dim, H, W)
        sampling_grid = 2 * sampling_locations - 1
        sampling_grid = sampling_grid.permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampled_values = F.grid_sample(value_flat, sampling_grid, mode='bilinear',
                                       padding_mode='zeros', align_corners=False)
        attention_weights = attention_weights.permute(0, 2, 1, 3).flatten(0, 1).unsqueeze(1)
        output = (sampled_values * attention_weights).sum(-1)
        output = output.view(bs, self.num_heads * self.head_dim, num_queries).transpose(1, 2)
        return self.output_proj(output)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_with_recorder(module, *args, **kwargs):
    """Run ``module`` forward with an active InferenceRecorder; return (output, activations)."""
    import tempfile
    recorder = InferenceRecorder(tempfile.mkdtemp(), enabled=True)
    recorder.attach(module)
    recorder.start_batch(epoch=0, phase="test", split="val", batch_index=0)
    out = module(*args, **kwargs)
    name = getattr(module, "_inference_recorder_module_name", module.__class__.__name__)
    acts = recorder.current_payload["activations"].get(name, {})
    return out, acts


def make_multiscale_value(bs, shapes, embed_dim, generator):
    total = sum(h * w for h, w in shapes)
    return torch.randn(bs, total, embed_dim, generator=generator)


# ===========================================================================
# Group 1 — n_levels=1 byte-identity (retro-compat contract)
# ===========================================================================
def test_param_shapes_match_today():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=1)
    assert tuple(m.sampling_offsets.weight.shape) == (NUM_HEADS * 1 * NUM_POINTS * 2, EMBED_DIM) == (32, 32)
    assert tuple(m.sampling_offsets.bias.shape) == (32,)
    assert tuple(m.attention_weights.weight.shape) == (16, 32)
    assert tuple(m.attention_weights.bias.shape) == (16,)


def test_state_dict_byte_identity():
    torch.manual_seed(0)
    new = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=1)
    torch.manual_seed(0)
    legacy = LegacyDeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS)
    sd_new, sd_old = new.state_dict(), legacy.state_dict()
    assert set(sd_new.keys()) == set(sd_old.keys())
    for k in sd_new:
        assert torch.equal(sd_new[k], sd_old[k]), f"mismatch in {k}"


def test_forward_output_identity_at_one_level():
    torch.manual_seed(0)
    new = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=1)
    legacy = LegacyDeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS)
    legacy.load_state_dict(new.state_dict())
    new.eval()
    legacy.eval()
    g = torch.Generator().manual_seed(1)
    H = W = 8
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 2, generator=g)
    value = torch.randn(B, H * W, EMBED_DIM, generator=g)
    out_new = new(query, ref, value, (H, W))
    out_old = legacy(query, ref, value, (H, W))
    assert torch.equal(out_new, out_old)


def test_old_checkpoint_load_strict():
    legacy = LegacyDeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS)
    new = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=1)
    result = new.load_state_dict(legacy.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_softmax_reduces_to_points_only_at_one_level():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=1)
    m.eval()
    g = torch.Generator().manual_seed(2)
    H = W = 8
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 2, generator=g)
    value = torch.randn(B, H * W, EMBED_DIM, generator=g)
    _, acts = run_with_recorder(m, query, ref, value, (H, W))
    aw = acts["attention_weights"]
    assert tuple(aw.shape) == (B, NQ, NUM_HEADS, 1, NUM_POINTS)
    sums = aw.sum(dim=-1)  # over points
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    # squeezing the level axis reproduces today's (B,Nq,H,P) normalization
    assert tuple(aw.squeeze(3).shape) == (B, NQ, NUM_HEADS, NUM_POINTS)


# ===========================================================================
# Group 2 — Multi-level forward correctness
# ===========================================================================
def test_shapes_at_n_levels_3():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=3)
    m.eval()
    g = torch.Generator().manual_seed(3)
    shapes = torch.tensor([[8, 8], [4, 4], [2, 2]])
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 3, 2, generator=g)
    value = make_multiscale_value(B, [(8, 8), (4, 4), (2, 2)], EMBED_DIM, g)
    out = m(query, ref, value, shapes)
    assert tuple(out.shape) == (B, NQ, EMBED_DIM)


def test_value_split_alignment():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=3)
    m.eval()
    m.enable_memory_kv_cache()
    g = torch.Generator().manual_seed(4)
    shapes = torch.tensor([[8, 8], [4, 4], [2, 2]])
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 3, 2, generator=g)
    value = make_multiscale_value(B, [(8, 8), (4, 4), (2, 2)], EMBED_DIM, g)
    m(query, ref, value, shapes)
    levels = m.value_cache
    head_dim = EMBED_DIM // NUM_HEADS
    assert [v.shape[-2] * v.shape[-1] for v in levels] == [64, 16, 4]
    expected = [(8, 8), (4, 4), (2, 2)]
    for v, (H, W) in zip(levels, expected):
        assert tuple(v.shape) == (B * NUM_HEADS, head_dim, H, W)


def test_joint_softmax_over_levels_and_points():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=3)
    m.eval()
    g = torch.Generator().manual_seed(5)
    shapes = torch.tensor([[8, 8], [4, 4], [2, 2]])
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 3, 2, generator=g)
    value = make_multiscale_value(B, [(8, 8), (4, 4), (2, 2)], EMBED_DIM, g)
    _, acts = run_with_recorder(m, query, ref, value, shapes)
    aw = acts["attention_weights"]
    assert tuple(aw.shape) == (B, NQ, NUM_HEADS, 3, NUM_POINTS)
    sums = aw.sum(dim=(-2, -1))  # jointly over (level, point)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_single_level_equivalence_embedded_in_multilevel():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=3)
    m.eval()
    # Force it to behave like 1 level: zero the offsets (weight and bias) so every sampled
    # point/level collapses onto the reference location; uniform attention weights (bias/weight
    # already zero) then average identical reads back to that single read.
    with torch.no_grad():
        m.sampling_offsets.weight.zero_()
        m.sampling_offsets.bias.zero_()
    g = torch.Generator().manual_seed(6)
    H = W = 8
    shapes = torch.tensor([[H, W], [H, W], [H, W]])
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref2d = torch.rand(B, NQ, 2, generator=g)
    single_map = torch.randn(B, H * W, EMBED_DIM, generator=g)
    value = single_map.repeat(1, 3, 1)  # same map at all three levels
    out = m(query, ref2d, value, shapes)

    # Manual: bilinear read of value_proj(map) at ref, then output_proj.
    head_dim = EMBED_DIM // NUM_HEADS
    vproj = m.value_proj(single_map).view(B, H * W, NUM_HEADS, head_dim)
    vproj = vproj.permute(0, 2, 3, 1).reshape(B * NUM_HEADS, head_dim, H, W)
    grid = (2 * ref2d - 1)[:, None].expand(B, NUM_HEADS, NQ, 2).reshape(B * NUM_HEADS, NQ, 1, 2)
    sampled = F.grid_sample(vproj, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    sampled = sampled.squeeze(-1)  # (B*H, head_dim, Nq)
    manual = sampled.view(B, EMBED_DIM, NQ).transpose(1, 2)
    manual = m.output_proj(manual)
    assert torch.allclose(out, manual, atol=1e-5)


def test_level_start_index_optional():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=3)
    m.eval()
    g = torch.Generator().manual_seed(7)
    shapes = torch.tensor([[8, 8], [4, 4], [2, 2]])
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 3, 2, generator=g)
    value = make_multiscale_value(B, [(8, 8), (4, 4), (2, 2)], EMBED_DIM, g)
    out_none = m(query, ref, value, shapes, level_start_index=None)
    out_explicit = m(query, ref, value, shapes, level_start_index=torch.tensor([0, 64, 80]))
    assert torch.equal(out_none, out_explicit)


def test_gradients_flow_to_all_levels():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=3)
    m.train()
    g = torch.Generator().manual_seed(8)
    shapes = torch.tensor([[8, 8], [4, 4], [2, 2]])
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 3, 2, generator=g)
    value = make_multiscale_value(B, [(8, 8), (4, 4), (2, 2)], EMBED_DIM, g)
    out = m(query, ref, value, shapes)
    out.sum().backward()
    for name in ["sampling_offsets", "attention_weights", "value_proj", "output_proj"]:
        grad = getattr(m, name).weight.grad
        assert grad is not None, f"{name} grad is None"
        assert torch.isfinite(grad).all(), f"{name} grad not finite"


# ===========================================================================
# Group 3 — Input polymorphism and error conditions
# ===========================================================================
def test_tuple_and_tensor_spatial_shape_interchangeable():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=1)
    m.eval()
    g = torch.Generator().manual_seed(9)
    H = W = 8
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 2, generator=g)
    value = torch.randn(B, H * W, EMBED_DIM, generator=g)
    out_tuple = m(query, ref, value, (8, 8))
    out_tensor = m(query, ref, value, torch.tensor([[8, 8]]))
    assert torch.equal(out_tuple, out_tensor)


def test_2d_reference_points_broadcast():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=3)
    m.eval()
    g = torch.Generator().manual_seed(10)
    shapes = torch.tensor([[8, 8], [4, 4], [2, 2]])
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref2d = torch.rand(B, NQ, 2, generator=g)
    value = make_multiscale_value(B, [(8, 8), (4, 4), (2, 2)], EMBED_DIM, g)
    out_2d = m(query, ref2d, value, shapes)
    ref3d = ref2d[:, :, None, :].expand(B, NQ, 3, 2).contiguous()
    out_3d = m(query, ref3d, value, shapes)
    assert torch.equal(out_2d, out_3d)


def test_level_count_mismatch_raises():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=3)
    g = torch.Generator().manual_seed(11)
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 2, generator=g)
    value = make_multiscale_value(B, [(2, 2), (2, 2)], EMBED_DIM, g)
    with pytest.raises(ValueError):
        m(query, ref, value, torch.tensor([[2, 2], [2, 2]]))


def test_value_length_mismatch_raises():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=3)
    g = torch.Generator().manual_seed(12)
    shapes = torch.tensor([[8, 8], [4, 4], [2, 2]])
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 3, 2, generator=g)
    value = torch.randn(B, 80, EMBED_DIM, generator=g)  # should be 84
    with pytest.raises(ValueError):
        m(query, ref, value, shapes)


def test_bad_reference_point_last_dim_raises():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=1)
    g = torch.Generator().manual_seed(13)
    H = W = 8
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 4, generator=g)  # box refs, out of scope
    value = torch.randn(B, H * W, EMBED_DIM, generator=g)
    with pytest.raises(ValueError):
        m(query, ref, value, (8, 8))


def test_embed_dim_not_divisible_raises():
    with pytest.raises(ValueError):
        DeformableAttention(embed_dim=30, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=1)


# ===========================================================================
# Group 4 — Decoder integration (F1 leaves decoders untouched)
# ===========================================================================
def test_deformable_decoder_forward_unchanged():
    torch.manual_seed(0)
    dec = DeformableDecoder(model_dim=EMBED_DIM, total_dim=EMBED_DIM, n_heads=NUM_HEADS,
                            ff_dim=64, spatial_shape=(8, 8), num_points=NUM_POINTS)
    dec.eval()
    g = torch.Generator().manual_seed(20)
    src = torch.randn(B, NQ, EMBED_DIM, generator=g)
    mem = torch.randn(B, 1 + 64, EMBED_DIM, generator=g)  # CLS + 8*8
    ref = torch.rand(B, NQ, 2, generator=g)
    out_new = dec(src, mem, reference_points=ref)
    assert tuple(out_new.shape) == (B, NQ, EMBED_DIM)
    # Swap in the legacy op with identical weights; the whole decoder must produce the same output.
    legacy = LegacyDeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS)
    legacy.load_state_dict(dec.cross_attn.state_dict())
    dec.cross_attn = legacy
    out_legacy = dec(src, mem, reference_points=ref)
    assert torch.equal(out_new, out_legacy)


def test_deformable_double_input_decoder_forward_unchanged():
    torch.manual_seed(0)
    dec = DeformableDoubleInputDecoder(model_dim=EMBED_DIM, total_dim=EMBED_DIM, n_heads=NUM_HEADS,
                                       ff_dim=64, norm_first=True, spatial_shape=(8, 8),
                                       num_points=NUM_POINTS)
    dec.eval()
    g = torch.Generator().manual_seed(21)
    src = torch.randn(B, NQ, EMBED_DIM, generator=g)
    mem1 = torch.randn(B, 7, EMBED_DIM, generator=g)
    mem2 = torch.randn(B, 1 + 64, EMBED_DIM, generator=g)  # CLS + 8*8
    ref = torch.rand(B, NQ, 2, generator=g)
    out_new = dec(src, mem1, mem2, reference_points=ref)
    assert tuple(out_new.shape) == (B, NQ, EMBED_DIM)
    legacy = LegacyDeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS)
    legacy.load_state_dict(dec.second_cross_attn.state_dict())
    dec.second_cross_attn = legacy
    out_legacy = dec(src, mem1, mem2, reference_points=ref)
    assert torch.equal(out_new, out_legacy)


def test_kv_cache_parity():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=1)
    m.eval()
    g = torch.Generator().manual_seed(22)
    H = W = 8
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 2, generator=g)
    value = torch.randn(B, H * W, EMBED_DIM, generator=g)

    out_nocache = m(query, ref, value, (H, W))

    m.enable_memory_kv_cache()
    out_c1 = m(query, ref, value, (H, W))
    out_c2 = m(query, ref, value, (H, W))
    assert torch.equal(out_c1, out_nocache)
    assert torch.equal(out_c2, out_nocache)

    # After clearing, a different value must be reflected (not the stale cache).
    m.clear_kv_cache()
    value2 = torch.randn(B, H * W, EMBED_DIM, generator=g)
    out_new_mem = m(query, ref, value2, (H, W))
    m.disable_memory_kv_cache()
    out_ref = m(query, ref, value2, (H, W))
    assert torch.equal(out_new_mem, out_ref)
    assert not torch.equal(out_new_mem, out_nocache)


@pytest.mark.parametrize("n_levels", [1, 3])
def test_recorder_keys_present_and_shaped(n_levels):
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=n_levels)
    m.eval()
    g = torch.Generator().manual_seed(23)
    if n_levels == 1:
        shapes = (8, 8)
        value = torch.randn(B, 64, EMBED_DIM, generator=g)
    else:
        shapes = torch.tensor([[8, 8], [4, 4], [2, 2]])
        value = make_multiscale_value(B, [(8, 8), (4, 4), (2, 2)], EMBED_DIM, g)
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 2, generator=g)
    _, acts = run_with_recorder(m, query, ref, value, shapes)
    L = n_levels
    assert tuple(acts["sampling_offsets"].shape) == (B, NQ, NUM_HEADS, L, NUM_POINTS, 2)
    assert tuple(acts["attention_weights"].shape) == (B, NQ, NUM_HEADS, L, NUM_POINTS)
    assert tuple(acts["sampling_locations"].shape) == (B, NQ, NUM_HEADS, L, NUM_POINTS, 2)
    assert tuple(acts["reference_points"].shape) == (B, NQ, L, 2)


# ===========================================================================
# Data Validity — lightweight numerical sanity checks
# ===========================================================================
def test_star_pattern_geometry():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS,
                            n_levels=3, normalize_grid_init=True)
    bias = m.sampling_offsets.bias.data.view(NUM_HEADS, 3, NUM_POINTS, 2)
    # Same star pattern replicated across levels.
    for lvl in range(1, 3):
        assert torch.equal(bias[:, 0], bias[:, lvl])
    # Per head, point i sits at radius ~ (i+1) times the point-0 radius, along a fixed axis.
    for h in range(NUM_HEADS):
        r0 = bias[h, 0, 0].norm()
        for i in range(NUM_POINTS):
            ri = bias[h, 0, i].norm()
            assert torch.allclose(ri, (i + 1) * r0, atol=1e-5)
    # Head directions evenly spaced on the circle (equal angular gaps).
    dirs = bias[:, 0, 0]  # (heads, 2)
    angles = torch.atan2(dirs[:, 1], dirs[:, 0])
    gaps = (angles[1:] - angles[:-1]) % (2 * math.pi)
    assert torch.allclose(gaps, gaps[0].expand_as(gaps), atol=1e-5)


def test_sampling_locations_near_reference_at_init():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=1)
    m.eval()
    H = W = 8
    query = torch.zeros(B, NQ, EMBED_DIM)  # weight==0 => offsets are the star bias only
    ref = torch.full((B, NQ, 2), 0.5)
    value = torch.randn(B, H * W, EMBED_DIM)
    _, acts = run_with_recorder(m, query, ref, value, (H, W))
    loc = acts["sampling_locations"]
    assert (loc >= -1e-6).all() and (loc <= 1 + 1e-6).all()


def test_geometric_sigma_is_train_only():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS,
                            n_levels=1, geometric_sigma=0.5)
    g = torch.Generator().manual_seed(24)
    H = W = 8
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 2, generator=g)
    value = torch.randn(B, H * W, EMBED_DIM, generator=g)

    m.eval()
    assert torch.equal(m(query, ref, value, (H, W)), m(query, ref, value, (H, W)))

    m.train()
    torch.manual_seed(100)
    a = m(query, ref, value, (H, W))
    torch.manual_seed(101)
    b = m(query, ref, value, (H, W))
    assert not torch.equal(a, b)


def test_output_magnitude_sanity():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=3)
    m.eval()
    g = torch.Generator().manual_seed(25)
    shapes = torch.tensor([[8, 8], [4, 4], [2, 2]])
    query = torch.randn(B, NQ, EMBED_DIM, generator=g)
    ref = torch.rand(B, NQ, 3, 2, generator=g)
    value = make_multiscale_value(B, [(8, 8), (4, 4), (2, 2)], EMBED_DIM, g)
    out = m(query, ref, value, shapes)
    assert torch.isfinite(out).all()
    assert out.std().item() < 10 * value.std().item()


# ===========================================================================
# Data Architecture Integrity
# ===========================================================================
def test_param_name_stability():
    m = DeformableAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_points=NUM_POINTS, n_levels=1)
    expected = {
        "sampling_offsets.weight", "sampling_offsets.bias",
        "attention_weights.weight", "attention_weights.bias",
        "value_proj.weight", "value_proj.bias",
        "output_proj.weight", "output_proj.bias",
    }
    assert set(m.state_dict().keys()) == expected


def test_mask2former_layout_match():
    # Contract F2's weight loader relies on: shapes must equal Mask2Former's
    # MSDeformAttn(d_model=256, n_levels=3, n_heads=8, n_points=4).
    #   sampling_offsets  = n_heads*n_levels*n_points*2 = 8*3*4*2 = 192
    #   attention_weights = n_heads*n_levels*n_points   = 8*3*4   = 96
    # (validation.md quotes 768/384, which would imply n_points=16 — a spec typo;
    #  verified directly against mask2former/.../ops/modules/ms_deform_attn.py.)
    m = DeformableAttention(embed_dim=256, num_heads=8, num_points=4, n_levels=3)
    assert tuple(m.sampling_offsets.weight.shape) == (192, 256)
    assert tuple(m.attention_weights.weight.shape) == (96, 256)
    assert tuple(m.value_proj.weight.shape) == (256, 256)
    assert tuple(m.output_proj.weight.shape) == (256, 256)
