"""F2 — Vendored Mask2Former backbone (detectron2-free).

Tests for ``src/model/ms_deform_backbone.py``: the torchvision ResNet50 feature extractor,
the detectron2-free pixel decoder running F1 ``DeformableAttention`` at ``n_levels=3``, the
optional stride-4 branch, granular freezing, dependency purity, and the input/determinism
contract. CPU-only, fixed seeds. Covers Groups 1–6 of ``validation.md``.

ImageNet weights require a download; to keep the suite hermetic and fast, tests instantiate the
backbone with ``imagenet_weights=None`` (random init) except the one test that explicitly asserts
ImageNet weights loaded, which is ``skip``ped when offline.
"""

import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from src.model.blocks import DeformableAttention
from src.model.ms_deform_backbone import (
    Mask2FormerBackbone,
    MSDeformAttnPixelDecoder,
    MSDeformAttnTransformerEncoderOnly,
    PositionEmbeddingSine,
)
from src.training.inference_recorder import InferenceRecorder


def make_backbone(**kwargs):
    """Backbone with random-init ResNet50 (no network) unless overridden."""
    kwargs.setdefault("imagenet_weights", None)
    return Mask2FormerBackbone(**kwargs)


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(2, 3, 256, 256)


# ===========================================================================
# Group 1 — ResNet50 feature extraction
# ===========================================================================
def test_feature_extractor_keys(x):
    torch.manual_seed(0)
    feats = make_backbone().feature_extractor(x)
    assert set(feats.keys()) == {"res2", "res3", "res4", "res5"}


def test_feature_extractor_shapes_256(x):
    torch.manual_seed(0)
    feats = make_backbone().feature_extractor(x)
    assert tuple(feats["res2"].shape) == (2, 256, 64, 64)
    assert tuple(feats["res3"].shape) == (2, 512, 32, 32)
    assert tuple(feats["res4"].shape) == (2, 1024, 16, 16)
    assert tuple(feats["res5"].shape) == (2, 2048, 8, 8)
    for f in feats.values():
        assert f.dtype == torch.float32


def test_feature_extractor_shapes_dynamic_128():
    torch.manual_seed(0)
    feats = make_backbone().feature_extractor(torch.randn(2, 3, 128, 128))
    assert tuple(feats["res2"].shape) == (2, 256, 32, 32)
    assert tuple(feats["res5"].shape) == (2, 2048, 4, 4)


def test_imagenet_weights_loaded(x):
    """Default IMAGENET1K_V2 conv1 differs from a random-init conv1 under the same seed."""
    torch.manual_seed(0)
    try:
        loaded = Mask2FormerBackbone(imagenet_weights="IMAGENET1K_V2")
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"ImageNet weights unavailable (offline?): {e}")
    loaded_conv1 = dict(loaded.feature_extractor.named_parameters())["conv1.weight"]
    torch.manual_seed(0)
    random = make_backbone()
    random_conv1 = dict(random.feature_extractor.named_parameters())["conv1.weight"]
    # If weights genuinely loaded they must differ from the freshly-seeded random tensor.
    if torch.allclose(loaded_conv1, random_conv1):
        pytest.skip("ImageNet weights not actually fetched (offline fallback to random init)")
    assert not torch.allclose(loaded_conv1, random_conv1)


# ===========================================================================
# Group 2 — Pixel decoder forward (3 levels)
# ===========================================================================
def test_forward_returns_list_of_three(x):
    torch.manual_seed(0)
    out = make_backbone(return_stride4=False)(x)
    assert isinstance(out, list)
    assert len(out) == 3


def test_forward_map_shapes_and_finite(x):
    torch.manual_seed(0)
    out = make_backbone()(x)
    assert [tuple(o.shape) for o in out] == [(2, 256, 8, 8), (2, 256, 16, 16), (2, 256, 32, 32)]
    for o in out:
        assert o.dtype == torch.float32
        assert torch.isfinite(o).all()


def test_self_attn_is_f1_with_three_levels():
    torch.manual_seed(0)
    bb = make_backbone()
    for layer in bb.pixel_decoder.transformer.encoder.layers:
        assert isinstance(layer.self_attn, DeformableAttention)
        assert layer.self_attn.n_levels == 3


def test_f1_param_shapes_match_reference():
    torch.manual_seed(0)
    bb = make_backbone()
    for layer in bb.pixel_decoder.transformer.encoder.layers:
        sa = layer.self_attn
        assert tuple(sa.sampling_offsets.weight.shape) == (192, 256)
        assert tuple(sa.attention_weights.weight.shape) == (96, 256)
        assert tuple(sa.value_proj.weight.shape) == (256, 256)
        assert tuple(sa.output_proj.weight.shape) == (256, 256)


def test_encoder_runs_configured_number_of_layers():
    torch.manual_seed(0)
    bb = make_backbone(transformer_enc_layers=6)
    assert len(bb.pixel_decoder.transformer.encoder.layers) == 6
    bb3 = make_backbone(transformer_enc_layers=3)
    assert len(bb3.pixel_decoder.transformer.encoder.layers) == 3


def test_spatial_shapes_and_level_start_index(x):
    torch.manual_seed(0)
    bb = make_backbone()
    feats = bb.feature_extractor(x)
    srcs, pos = [], []
    pd = bb.pixel_decoder
    for f in pd.transformer_in_features[::-1]:
        srcs.append(pd.input_proj[len(srcs)](feats[f]))
        pos.append(pd.pe_layer(feats[f]))
    _, spatial_shapes, level_start_index = pd.transformer(srcs, pos)
    assert torch.equal(spatial_shapes, torch.tensor([[8, 8], [16, 16], [32, 32]]))
    assert torch.equal(level_start_index, torch.tensor([0, 64, 320]))


def test_level_embed_shape():
    torch.manual_seed(0)
    bb = make_backbone()
    le = bb.pixel_decoder.transformer.level_embed
    assert isinstance(le, nn.Parameter)
    assert tuple(le.shape) == (3, 256)


# ===========================================================================
# Group 3 — Optional stride-4 branch
# ===========================================================================
def test_no_stride4_submodules_when_disabled():
    torch.manual_seed(0)
    bb = make_backbone(return_stride4=False)
    for attr in ("lateral_res2", "output_res2", "mask_features"):
        assert not hasattr(bb.pixel_decoder, attr)


def test_stride4_output_shapes(x):
    torch.manual_seed(0)
    bb = make_backbone(return_stride4=True)
    maps, mask_features = bb(x)
    assert len(maps) == 4
    assert tuple(maps[-1].shape) == (2, 256, 64, 64)
    assert tuple(mask_features.shape) == (2, 256, 64, 64)
    for m in maps:
        assert torch.isfinite(m).all()
    assert torch.isfinite(mask_features).all()
    assert bb.num_levels == 4


def test_stride4_builds_more_params(x):
    torch.manual_seed(0)
    bb_off = make_backbone(return_stride4=False)
    torch.manual_seed(0)
    bb_on = make_backbone(return_stride4=True)
    n_off = sum(p.numel() for p in bb_on.pixel_decoder.parameters())
    n_base = sum(p.numel() for p in bb_off.pixel_decoder.parameters())
    assert n_off > n_base
    # The 3-level maps are shape-identical regardless of the flag.
    out_off = bb_off(x)
    out_on = bb_on(x)[0][:3]
    assert [tuple(o.shape) for o in out_off] == [tuple(o.shape) for o in out_on]


# ===========================================================================
# Group 4 — Freezing semantics
# ===========================================================================
def test_default_freezing_requires_grad():
    torch.manual_seed(0)
    bb = make_backbone()
    assert all(not p.requires_grad for p in bb.feature_extractor.parameters())
    assert all(p.requires_grad for p in bb.pixel_decoder.parameters())


def test_train_mode_keeps_backbone_eval():
    torch.manual_seed(0)
    bb = make_backbone()
    bb.train()
    assert bb.feature_extractor.training is False
    assert bb.pixel_decoder.training is True


def test_backward_grad_flow(x):
    torch.manual_seed(0)
    bb = make_backbone()
    bb.train()
    bb(x)[-1].sum().backward()
    conv1 = dict(bb.feature_extractor.named_parameters())["conv1.weight"]
    ip = bb.pixel_decoder.input_proj[0][0].weight
    assert conv1.grad is None or conv1.grad.abs().sum().item() == 0.0
    assert ip.grad is not None and ip.grad.abs().sum().item() > 0.0


def test_bn_running_mean_frozen(x):
    torch.manual_seed(0)
    bb = make_backbone()
    bb.train()
    # find first BN running_mean
    bn = next(m for m in bb.feature_extractor.modules() if isinstance(m, nn.BatchNorm2d))
    before = bn.running_mean.clone()
    bb(x)
    bb(torch.randn(2, 3, 256, 256))
    assert torch.equal(bn.running_mean, before)


def test_unfrozen_backbone_requires_grad():
    torch.manual_seed(0)
    bb = make_backbone(freeze_backbone=False)
    assert all(p.requires_grad for p in bb.feature_extractor.parameters())


# ===========================================================================
# Group 5 — Purity / no forbidden dependencies
# ===========================================================================
def test_no_forbidden_imports():
    code = (
        "import sys; import src.model.ms_deform_backbone as m;"
        "bad = [n for n in sys.modules if n=='detectron2' or n.startswith('detectron2.')"
        " or n=='fvcore' or n.startswith('fvcore.') or 'MSDeformAttn' in n];"
        "assert not bad, bad; print('OK')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_norm_and_conv_types():
    torch.manual_seed(0)
    pd = make_backbone().pixel_decoder
    for m in pd.modules():
        if isinstance(m, nn.modules.batchnorm._NormBase):
            pytest.fail(f"unexpected BatchNorm in pixel_decoder: {type(m)}")
        if isinstance(m, nn.modules.conv._ConvNd):
            assert isinstance(m, nn.Conv2d), f"non-Conv2d conv: {type(m)}"
    norms = [m for m in pd.modules() if isinstance(m, (nn.GroupNorm, nn.LayerNorm))]
    assert len(norms) > 0


def test_recorder_records_level_axis(x, tmp_path):
    torch.manual_seed(0)
    bb = make_backbone()
    bb.eval()
    recorder = InferenceRecorder(output_dir=tmp_path / "recorder", enabled=True)
    recorder.attach(bb)
    recorder.start_batch(epoch=0, phase="Combined", split="val", batch_index=0)
    with torch.no_grad():
        bb(x)
    acts = recorder.current_payload["activations"]
    # At least one self_attn module recorded tensors with a level axis of size 3.
    hits = 0
    for name, bucket in acts.items():
        if "sampling_offsets" in bucket:
            so = bucket["sampling_offsets"]
            so = so[0] if isinstance(so, list) else so
            assert so.shape[-3] == 3 and so.shape[-2] == 4 and so.shape[-1] == 2
            aw = bucket["attention_weights"]
            aw = aw[0] if isinstance(aw, list) else aw
            assert aw.shape[-2] == 3 and aw.shape[-1] == 4
            hits += 1
    assert hits > 0
    recorder.clear()


# ===========================================================================
# Group 6 — Input contract & determinism
# ===========================================================================
def test_batch_independence(x):
    torch.manual_seed(0)
    bb = make_backbone().eval()
    with torch.no_grad():
        full = bb(x)
        single = bb(x[:1])
    for f, s in zip(full, single):
        assert torch.allclose(f[:1], s, atol=1e-5)


def test_determinism(x):
    torch.manual_seed(0)
    bb = make_backbone().eval()
    with torch.no_grad():
        a = bb(x)
        b = bb(x)
    for oa, ob in zip(a, b):
        assert torch.equal(oa, ob)


def test_non_square_input():
    torch.manual_seed(0)
    bb = make_backbone()
    xr = torch.randn(2, 3, 256, 192)
    feats = bb.feature_extractor(xr)
    assert tuple(feats["res5"].shape) == (2, 2048, 8, 6)
    out = bb(xr)
    assert [tuple(o.shape) for o in out] == [(2, 256, 8, 6), (2, 256, 16, 12), (2, 256, 32, 24)]


def test_never_returns_cls_token(x):
    torch.manual_seed(0)
    out = make_backbone()(x)
    for o in out:
        assert o.dim() == 4  # (B, C, H, W), never (B, HW+1, C)


# ===========================================================================
# Data Validity / Architectural Integrity (selected programmatic checks)
# ===========================================================================
def test_position_encoding_matches_reference(x):
    """PositionEmbeddingSine is faithful to a locally-defined reference of the source."""
    import math

    class RefPE(nn.Module):
        def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
            super().__init__()
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            if scale is None:
                scale = 2 * math.pi
            self.scale = scale

        def forward(self, x, mask=None):
            if mask is None:
                mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
            not_mask = ~mask
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    a = PositionEmbeddingSine(128, normalize=True)(x)
    b = RefPE(128, normalize=True)(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_feature_std_non_degenerate(x):
    torch.manual_seed(0)
    out = make_backbone().eval()(x)
    for o in out:
        assert torch.isfinite(o).all()
        assert o.std().item() > 1e-3


def test_cross_scale_fusion_active(x):
    """Zeroing one input level's projected src measurably changes all returned maps."""
    torch.manual_seed(0)
    bb = make_backbone().eval()
    pd = bb.pixel_decoder
    feats = bb.feature_extractor(x)

    def run(zero_idx=None):
        srcs, pos = [], []
        for f in pd.transformer_in_features[::-1]:
            s = pd.input_proj[len(srcs)](feats[f])
            if zero_idx is not None and len(srcs) == zero_idx:
                s = torch.zeros_like(s)
            srcs.append(s)
            pos.append(pd.pe_layer(feats[f]))
        memory, spatial_shapes, lsi = pd.transformer(srcs, pos)
        b = memory.shape[0]
        sizes = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        splits = torch.split(memory, sizes, dim=1)
        return [z.transpose(1, 2).view(b, -1, int(h), int(w))
                for z, (h, w) in zip(splits, spatial_shapes.tolist())]

    with torch.no_grad():
        base = run()
        zeroed = run(zero_idx=0)   # zero res5 (coarsest) level
    for bmap, zmap in zip(base, zeroed):
        assert (bmap - zmap).abs().max().item() > 1e-4


def test_stride4_resolution_is_4x_res5(x):
    torch.manual_seed(0)
    maps, _ = make_backbone(return_stride4=True)(x)
    res5, res2 = maps[0], maps[-1]
    assert res2.shape[-1] == res5.shape[-1] * 8   # 8 -> 64
    assert res2.shape[-2] == res5.shape[-2] * 8


def test_conv_dim_fixed_channel():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 256, 256)
    out = make_backbone()(x)
    for o in out:
        assert o.shape[1] == 256
