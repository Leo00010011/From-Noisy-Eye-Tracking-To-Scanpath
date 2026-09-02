"""F3 — Multi-scale feature contract / backbone adapter.

Tests for ``src/model/ms_features.py``: the ``MultiScaleFeatures`` bundle + validation, the
two geometry helpers, both producing adapters (Mask2Former → 3 levels, DINOv3 → 1 level with
CLS stripped once), F1 consumability, and module integrity. CPU-only, fixed seeds, no network.

Mask2Former tests build ``Mask2FormerBackbone(imagenet_weights=None)`` (random init, no
download); DINOv3 tests use a ``FakeDino`` stub returning CLS-prefixed random tokens so the
DINOv3 path is testable without ``torch.hub``.
"""

import pytest
import torch
import torch.nn as nn

from src.model.blocks import DeformableAttention
from src.model.ms_deform_backbone import (
    Mask2FormerBackbone,
    MSDeformAttnTransformerEncoder,
)
from src.model.ms_features import (
    DinoV3FeatureAdapter,
    Mask2FormerFeatureAdapter,
    MultiScaleFeatures,
    build_level_start_index,
    build_reference_grids,
)
from src.training.inference_recorder import InferenceRecorder


# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------
def make_backbone(**kwargs):
    """Backbone with random-init ResNet50 (no network) unless overridden."""
    kwargs.setdefault("imagenet_weights", None)
    return Mask2FormerBackbone(**kwargs)


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(2, 3, 256, 256)


class _FakeDinoInner(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size


class FakeDino(nn.Module):
    """Stub DINOv3 wrapper: exposes ``.embed_dim`` and ``.model.patch_size`` and returns
    ``(B, n_prefix + (H//p)*(W//p), D)`` random tokens. ``force_tokens`` overrides the token
    count to exercise the phantom-prefix guard.
    """

    def __init__(self, embed_dim=384, patch_size=16, n_prefix=1, force_tokens=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.model = _FakeDinoInner(patch_size)
        self.n_prefix = n_prefix
        self.force_tokens = force_tokens
        # A real parameter so state_dict / parameter counting has something to see.
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        p = self.model.patch_size
        Hs, Ws = x.shape[-2] // p, x.shape[-1] // p
        n = self.force_tokens if self.force_tokens is not None else self.n_prefix + Hs * Ws
        return torch.randn(x.shape[0], n, self.embed_dim)


def well_formed_bundle():
    spatial_shapes = torch.tensor([[8, 8], [16, 16], [32, 32]], dtype=torch.int64)
    return MultiScaleFeatures(
        value=torch.randn(2, 1344, 256),
        spatial_shapes=spatial_shapes,
        level_start_index=torch.tensor([0, 64, 320], dtype=torch.int64),
        reference_grids=build_reference_grids(spatial_shapes),
    )


# ===========================================================================
# Group 1 — Bundle construction and validation
# ===========================================================================
def test_well_formed_constructs():
    b = well_formed_bundle()
    assert isinstance(b, MultiScaleFeatures)


def test_bundle_properties():
    b = well_formed_bundle()
    assert b.num_levels == 3
    assert b.embed_dim == 256
    assert b.batch_size == 2
    assert b.seq_len == 1344
    assert b.level_sizes() == [64, 256, 1024]


def test_value_wrong_ndim_raises():
    ss = torch.tensor([[8, 8], [16, 16], [32, 32]], dtype=torch.int64)
    with pytest.raises(ValueError):
        MultiScaleFeatures(torch.randn(2, 1344), ss,
                           build_level_start_index(ss), build_reference_grids(ss))


def test_value_length_mismatch_raises():
    ss = torch.tensor([[8, 8], [16, 16], [32, 32]], dtype=torch.int64)
    with pytest.raises(ValueError):
        MultiScaleFeatures(torch.randn(2, 1000, 256), ss,
                           build_level_start_index(ss), build_reference_grids(ss))


def test_reference_grids_wrong_shape_raises():
    ss = torch.tensor([[8, 8], [16, 16], [32, 32]], dtype=torch.int64)
    lsi = build_level_start_index(ss)
    with pytest.raises(ValueError):
        MultiScaleFeatures(torch.randn(2, 1344, 256), ss, lsi, torch.randn(1344, 3))
    with pytest.raises(ValueError):
        MultiScaleFeatures(torch.randn(2, 1344, 256), ss, lsi, torch.randn(1000, 2))


def test_spatial_shapes_wrong_shape_raises():
    # (L, 3) is not (L, 2).
    ss = torch.tensor([[8, 8, 1]], dtype=torch.int64)
    with pytest.raises(ValueError):
        MultiScaleFeatures(torch.randn(2, 64, 256), ss,
                           torch.tensor([0], dtype=torch.int64), torch.randn(64, 2))


def test_inconsistent_level_start_index_raises():
    ss = torch.tensor([[8, 8], [16, 16], [32, 32]], dtype=torch.int64)
    with pytest.raises(ValueError):
        MultiScaleFeatures(torch.randn(2, 1344, 256), ss,
                           torch.tensor([0, 64, 300], dtype=torch.int64),
                           build_reference_grids(ss))


def test_float_index_tensors_raise():
    ss = torch.tensor([[8, 8], [16, 16], [32, 32]], dtype=torch.int64)
    rg = build_reference_grids(ss)
    val = torch.randn(2, 1344, 256)
    with pytest.raises(ValueError):
        MultiScaleFeatures(val, ss.float(), build_level_start_index(ss), rg)
    with pytest.raises(ValueError):
        MultiScaleFeatures(val, ss, build_level_start_index(ss).float(), rg)


def test_non_floating_value_raises():
    ss = torch.tensor([[8, 8], [16, 16], [32, 32]], dtype=torch.int64)
    with pytest.raises(ValueError):
        MultiScaleFeatures(torch.zeros(2, 1344, 256, dtype=torch.int64), ss,
                           build_level_start_index(ss), build_reference_grids(ss))


# ===========================================================================
# Group 2 — Geometry helpers
# ===========================================================================
def test_build_level_start_index():
    lsi = build_level_start_index(torch.tensor([[8, 8], [16, 16], [32, 32]]))
    assert lsi.tolist() == [0, 64, 320]
    assert lsi.dtype == torch.int64


def test_build_reference_grids_2x2():
    rg = build_reference_grids(torch.tensor([[2, 2]]))
    expected = torch.tensor([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    assert torch.allclose(rg, expected, atol=1e-6)


def test_build_reference_grids_shape_dtype_range():
    ss = torch.tensor([[8, 8], [16, 16], [32, 32]], dtype=torch.int64)
    rg = build_reference_grids(ss)
    assert rg.shape == (1344, 2)
    assert rg.dtype == torch.float32
    assert (rg > 0).all() and (rg < 1).all()


def test_reference_grids_ordering_cross_check():
    ss = torch.tensor([[8, 8]], dtype=torch.int64)
    rg = build_reference_grids(ss)
    for i in range(64):
        assert torch.allclose(
            rg[i], torch.tensor([(i % 8 + 0.5) / 8, (i // 8 + 0.5) / 8]), atol=1e-6)


def test_reference_grids_matches_backbone_get_reference_points():
    ss = torch.tensor([[8, 8], [16, 16], [32, 32]], dtype=torch.int64)
    ref = MSDeformAttnTransformerEncoder.get_reference_points(ss, device="cpu")
    # (1, S, L, 2); level slice is level-independent (no valid_ratios).
    assert torch.allclose(ref[0, :, 0, :], build_reference_grids(ss), atol=1e-6)


# ===========================================================================
# Group 3 — Mask2Former adapter
# ===========================================================================
def test_m2f_adapter_attrs():
    torch.manual_seed(0)
    ad = Mask2FormerFeatureAdapter(make_backbone())
    assert ad.embed_dim == 256
    assert ad.num_levels == 3


def test_m2f_adapter_forward(x):
    torch.manual_seed(0)
    ad = Mask2FormerFeatureAdapter(make_backbone())
    b = ad(x)
    assert b.value.shape == (2, 1344, 256)
    assert b.spatial_shapes.tolist() == [[8, 8], [16, 16], [32, 32]]
    assert b.level_start_index.tolist() == [0, 64, 320]
    assert b.reference_grids.shape == (1344, 2)


def test_m2f_adapter_round_trip_identity(x):
    torch.manual_seed(0)
    bb = make_backbone()
    ad = Mask2FormerFeatureAdapter(bb)
    with torch.no_grad():
        maps = bb(x)
        b = ad(x)
    expected = torch.cat([m.flatten(2).transpose(1, 2) for m in maps], dim=1)
    assert torch.equal(b.value, expected)


def test_m2f_adapter_geometry_matches_pixel_decoder(x):
    torch.manual_seed(0)
    ad = Mask2FormerFeatureAdapter(make_backbone())
    b = ad(x)
    # Reconstruct the tensors the pixel decoder produced internally.
    ss = torch.tensor([[8, 8], [16, 16], [32, 32]], dtype=torch.int64)
    lsi = torch.cat((ss.new_zeros((1,)), ss.prod(1).cumsum(0)[:-1]))
    assert torch.equal(b.spatial_shapes, ss)
    assert torch.equal(b.level_start_index, lsi)


def test_m2f_adapter_dynamic_size():
    torch.manual_seed(0)
    ad = Mask2FormerFeatureAdapter(make_backbone())
    b = ad(torch.randn(2, 3, 128, 128))
    assert b.spatial_shapes.tolist() == [[4, 4], [8, 8], [16, 16]]
    assert b.seq_len == 16 + 64 + 256   # 336


def test_m2f_adapter_return_stride4():
    torch.manual_seed(0)
    ad = Mask2FormerFeatureAdapter(make_backbone(return_stride4=True))
    assert ad.num_levels == 4
    b = ad(torch.randn(2, 3, 256, 256))
    assert b.value.shape == (2, 1344 + 4096, 256)
    assert b.spatial_shapes.tolist() == [[8, 8], [16, 16], [32, 32], [64, 64]]
    # The bundle has exactly four tensor fields; mask_features is not among them.
    assert set(vars(b).keys()) == {
        "value", "spatial_shapes", "level_start_index", "reference_grids"}


# ===========================================================================
# Group 4 — DINOv3 adapter
# ===========================================================================
def test_dino_adapter_forward():
    torch.manual_seed(0)
    ad = DinoV3FeatureAdapter(FakeDino(embed_dim=384, patch_size=16))
    b = ad(torch.randn(2, 3, 256, 256))
    assert b.value.shape == (2, 256, 384)
    assert b.spatial_shapes.tolist() == [[16, 16]]
    assert b.level_start_index.tolist() == [0]
    assert b.num_levels == 1
    assert b.embed_dim == 384


def test_dino_adapter_strips_prefix():
    torch.manual_seed(0)
    stub = FakeDino(embed_dim=384, patch_size=16)
    ad = DinoV3FeatureAdapter(stub)
    x = torch.randn(2, 3, 256, 256)
    torch.manual_seed(1)
    out = stub(x)
    torch.manual_seed(1)
    b = ad(x)
    assert torch.equal(b.value, out[:, 1:, :])


def test_dino_adapter_phantom_prefix_guard():
    # Stub returns 256 tokens (no CLS) while patch_size=16 → 256 patches; after stripping 1,
    # 256 patches != 255 tokens → ValueError.
    torch.manual_seed(0)
    ad = DinoV3FeatureAdapter(FakeDino(embed_dim=384, patch_size=16, force_tokens=256))
    with pytest.raises(ValueError):
        ad(torch.randn(2, 3, 256, 256))


def test_dino_adapter_num_prefix_tokens_zero():
    torch.manual_seed(0)
    stub = FakeDino(embed_dim=384, patch_size=16, n_prefix=0)
    ad = DinoV3FeatureAdapter(stub, num_prefix_tokens=0)
    b = ad(torch.randn(2, 3, 256, 256))
    assert b.value.shape == (2, 256, 384)


def test_dino_adapter_non_square():
    torch.manual_seed(0)
    ad = DinoV3FeatureAdapter(FakeDino(embed_dim=384, patch_size=16))
    b = ad(torch.randn(2, 3, 256, 128))
    assert b.spatial_shapes.tolist() == [[16, 8]]
    assert b.seq_len == 128


# ===========================================================================
# Group 5 — F1 consumability
# ===========================================================================
def test_m2f_bundle_feeds_f1(x):
    torch.manual_seed(0)
    ad = Mask2FormerFeatureAdapter(make_backbone())
    b = ad(x)
    attn = DeformableAttention(embed_dim=256, num_heads=8, num_points=4, n_levels=3)
    query = torch.randn(2, 5, 256)
    ref = torch.rand(2, 5, 2)
    out = attn(query, ref, b.value, b.spatial_shapes, b.level_start_index)
    assert out.shape == (2, 5, 256)


def test_m2f_bundle_feeds_f1_grid_anchored_refs(x):
    torch.manual_seed(0)
    ad = Mask2FormerFeatureAdapter(make_backbone())
    b = ad(x)
    attn = DeformableAttention(embed_dim=256, num_heads=8, num_points=4, n_levels=3)
    query = torch.randn(2, 5, 256)
    ref = b.reference_grids[None, :5]
    out = attn(query, ref, b.value, b.spatial_shapes, b.level_start_index)
    assert out.shape == (2, 5, 256)


def test_dino_bundle_feeds_f1():
    torch.manual_seed(0)
    ad = DinoV3FeatureAdapter(FakeDino(embed_dim=384, patch_size=16))
    b = ad(torch.randn(2, 3, 256, 256))
    attn = DeformableAttention(embed_dim=384, num_heads=8, num_points=4, n_levels=1)
    query = torch.randn(2, 5, 384)
    ref = torch.rand(2, 5, 2)
    out = attn(query, ref, b.value, b.spatial_shapes, b.level_start_index)
    assert out.shape == (2, 5, 384)


def test_wrong_n_levels_attention_raises(x):
    torch.manual_seed(0)
    ad = Mask2FormerFeatureAdapter(make_backbone())
    b = ad(x)
    attn = DeformableAttention(embed_dim=256, num_heads=8, num_points=4, n_levels=1)
    query = torch.randn(2, 5, 256)
    ref = torch.rand(2, 5, 2)
    with pytest.raises(ValueError):
        attn(query, ref, b.value, b.spatial_shapes, b.level_start_index)


# ===========================================================================
# Group 6 — Module integrity
# ===========================================================================
def test_bundle_to_dtype_preserves_index_dtypes():
    b = well_formed_bundle().to(dtype=torch.float64)
    assert b.value.dtype == torch.float64
    assert b.reference_grids.dtype == torch.float64
    assert b.spatial_shapes.dtype == torch.int64
    assert b.level_start_index.dtype == torch.int64


def test_bundle_to_returns_new_object():
    b = well_formed_bundle()
    b2 = b.to(dtype=torch.float64)
    assert b2 is not b
    assert b.value.dtype == torch.float32   # original unchanged


def test_adapter_registers_backbone_submodule():
    torch.manual_seed(0)
    bb = make_backbone()
    ad = Mask2FormerFeatureAdapter(bb)
    assert all(k.startswith("backbone.") for k in ad.state_dict().keys())
    assert sum(p.numel() for p in ad.parameters()) == sum(p.numel() for p in bb.parameters())


def test_adapter_eval_train_propagates():
    torch.manual_seed(0)
    ad = Mask2FormerFeatureAdapter(make_backbone())
    ad.eval()
    assert ad.backbone.feature_extractor.training is False
    ad.train()
    # Frozen ResNet stays in eval via F2's train() override through the submodule.
    assert ad.backbone.feature_extractor.training is False


def test_recorder_captures_level_axis(x, tmp_path):
    torch.manual_seed(0)
    ad = Mask2FormerFeatureAdapter(make_backbone())
    recorder = InferenceRecorder(output_dir=tmp_path / "recorder", enabled=True)
    recorder.attach(ad)
    recorder.start_batch(epoch=0, phase="Combined", split="val", batch_index=0)
    with torch.no_grad():
        ad(x)
    acts = recorder.current_payload["activations"]
    hits = 0
    for name, bucket in acts.items():
        if "sampling_offsets" in bucket:
            so = bucket["sampling_offsets"]
            so = so[0] if isinstance(so, list) else so
            assert so.shape[-3] == 3 and so.shape[-2] == 4 and so.shape[-1] == 2
            hits += 1
    assert hits > 0
    recorder.clear()


# ===========================================================================
# Data Validity
# ===========================================================================
def test_reference_grid_coverage():
    ss = torch.tensor([[8, 8], [16, 16], [32, 32]], dtype=torch.int64)
    rg = build_reference_grids(ss)
    lsi = build_level_start_index(ss).tolist() + [rg.shape[0]]
    for l, (H, W) in enumerate(ss.tolist()):
        seg = rg[lsi[l]:lsi[l + 1]]
        assert torch.allclose(seg[:, 0].min(), torch.tensor(0.5 / W), atol=1e-6)
        assert torch.allclose(seg[:, 0].max(), torch.tensor((W - 0.5) / W), atol=1e-6)
        assert torch.allclose(seg[:, 1].min(), torch.tensor(0.5 / H), atol=1e-6)
        assert torch.allclose(seg[:, 1].max(), torch.tensor((H - 0.5) / H), atol=1e-6)
    assert (rg > 0).all() and (rg < 1).all()


def test_reference_grid_spacing():
    ss = torch.tensor([[8, 8], [16, 16], [32, 32]], dtype=torch.int64)
    rg = build_reference_grids(ss)
    seg = rg[320:1344]   # the (32, 32) level
    # Consecutive x-centers differ by 1/32 within a row.
    dx = seg[1, 0] - seg[0, 0]
    assert torch.allclose(dx, torch.tensor(1.0 / 32), atol=1e-6)
    # y-center increments by 1/32 every 32 tokens.
    dy = seg[32, 1] - seg[0, 1]
    assert torch.allclose(dy, torch.tensor(1.0 / 32), atol=1e-6)


def test_value_finiteness(x):
    torch.manual_seed(0)
    ad = Mask2FormerFeatureAdapter(make_backbone())
    b = ad(x)
    assert torch.isfinite(b.value).all()


def test_level_energy_sanity(x):
    torch.manual_seed(0)
    ad = Mask2FormerFeatureAdapter(make_backbone())
    b = ad(x)
    lsi = b.level_start_index.tolist() + [b.seq_len]
    means = []
    for l in range(b.num_levels):
        means.append(b.value[:, lsi[l]:lsi[l + 1], :].abs().mean().item())
    assert min(means) > 0
    assert max(means) / min(means) < 100   # same order of magnitude


# ===========================================================================
# Data Architecture Integrity
# ===========================================================================
def test_no_phantom_prefix_token():
    torch.manual_seed(0)
    ad = DinoV3FeatureAdapter(FakeDino(embed_dim=384, patch_size=16))
    b = ad(torch.randn(2, 3, 256, 256))
    assert b.value.shape[1] == 16 * 16   # exactly H'·W', never + prefix


def test_native_channel_dim_preserved(x):
    torch.manual_seed(0)
    bb = make_backbone()
    ad = Mask2FormerFeatureAdapter(bb)
    assert ad(x).embed_dim == bb.embed_dim == 256
    dino = FakeDino(embed_dim=384, patch_size=16)
    dad = DinoV3FeatureAdapter(dino)
    assert dad(torch.randn(2, 3, 256, 256)).embed_dim == dino.embed_dim == 384
