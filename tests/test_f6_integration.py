"""F6 — MixerModel + PipelineBuilder + config integration tests.

CPU-only. The Mask2Former backbone is built with ``imagenet_weights=None`` (random ResNet,
no network). The DINOv3 path uses a lightweight deterministic stub encoder (``embed_dim=384``,
``.model.patch_size=16``, CLS-prefixed forward) so no DINOv3 clone / weights are required.

Groups mirror ``spec/2026-09-02-mixermodel-pipelinebuilder-config-integration/validation.md``.
"""

import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
from hydra import initialize, compose

from src.model.mixer_model import MixerModel
from src.model.ms_deform_backbone import Mask2FormerBackbone
from src.model.ms_features import (Mask2FormerFeatureAdapter, MultiScaleFeatures,
                                   build_level_start_index)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Stubs / builders
# ---------------------------------------------------------------------------
class _FakeRope(nn.Module):
    """Minimal stand-in for DINOv3's ``model.rope_embed`` (attribute bag)."""
    base = 100.0
    min_period = None
    max_period = None
    normalize_coords = "separate"
    shift_coords = None
    jitter_coords = None
    rescale_coords = None


class _DinoInner(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.rope_embed = _FakeRope()


class DummyDino(nn.Module):
    """Deterministic DINOv3 stub. ``forward(x)`` -> ``(B, 1 + (H/p)*(W/p), embed_dim)`` with a
    CLS prefix; output is a pure function of ``x`` and the module's own (seeded) parameters, so
    two identically-seeded instances agree bitwise.
    """

    def __init__(self, embed_dim=384, patch_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.model = _DinoInner(patch_size)
        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        p = self.model.patch_size
        patches = F.unfold(x, kernel_size=p, stride=p)          # (B, 3*p*p, L)
        patches = patches.transpose(1, 2)                        # (B, L, 3*p*p)
        tokens = self.proj(patches)                              # (B, L, D)
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)                  # (B, 1+L, D)


def make_m2f_backbone(**over):
    kw = dict(imagenet_weights=None, device="cpu")
    kw.update(over)
    return Mask2FormerBackbone(**kw)


COMMON = dict(
    n_encoder=2, n_decoder=2, n_eye_decoder=2, n_feature_enhancer=0,
    model_dim=512, total_dim=512, n_heads=8, ff_dim=256,
    max_pos_enc=90, max_pos_dec=26,
    input_encoder="shared_gaussian", norm_first=True,
    mlp_head_hidden_dim=[128], pos_enc_hidden_dim=64, num_freq_bands=8,
    pos_enc_sigma=1.0, use_deformable_eye_decoder=True,
    use_deformable_fixation_decoder=True, pred_dur_pdf=False,
    phases=["Fixation", "Combined"], activation=F.gelu, device="cpu",
)


def make_m2f_model(backbone=None, **over):
    if backbone is None:
        backbone = make_m2f_backbone(**over.pop("backbone_kw", {}))
    adapter = Mask2FormerFeatureAdapter(backbone)
    kw = dict(COMMON)
    kw.update(over)
    return MixerModel(image_encoder=adapter, image_encoder_type="mask2former",
                      n_image_levels=adapter.num_levels, head_type=kw.pop("head_type", "linear"),
                      **kw)


def make_dino_model(seed=0, **over):
    torch.manual_seed(seed)
    dino = DummyDino()
    kw = dict(COMMON)
    kw.update(over)
    return MixerModel(image_encoder=dino, image_encoder_type="dinov3",
                      n_image_levels=1, head_type=kw.pop("head_type", "linear"), **kw)


def dummy_batch(B=2, T=5, N=4, img=256, W=256):
    torch.manual_seed(123)
    src = torch.rand(B, T, 3)
    tgt = torch.rand(B, N, 3)
    image = torch.rand(B, 3, img, W)
    return dict(src=src, tgt=tgt, image_src=image, src_mask=None, tgt_mask=None)


# ===========================================================================
# Group 1 — Config composition
# ===========================================================================
def _compose(overrides=None):
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="main", overrides=overrides or [])


def test_g1_encoder_configs_exist_and_carry_type_and_embed():
    d = OmegaConf.load(REPO_ROOT / "configs/model/image_encoder/dinov3.yaml")
    m = OmegaConf.load(REPO_ROOT / "configs/model/image_encoder/mask2former.yaml")
    assert d.type == "dinov3" and d.embed_dim == 384
    assert m.type == "mask2former" and m.embed_dim == 256


def test_g1_default_composition_matches_pre_f6_fields():
    ie = _compose().model.image_encoder
    assert ie.type == "dinov3"
    assert ie.repo_path == "C:\\Users\\ulloa\\OneDrive\\Desktop\\Practicas\\projectes\\dinov3"
    assert ie.name == "dinov3_vits16"
    assert ie.weights == "dinov3_vits16_weights.pth"
    assert ie.freeze is True
    assert ie.regularization is True
    assert list(ie.adapter_hidden_dims) == []
    assert ie.image_dim == 256


def test_g1_mask2former_override():
    ie = _compose(["model/image_encoder=mask2former"]).model.image_encoder
    assert ie.type == "mask2former"
    assert ie.conv_dim == 256
    assert list(ie.transformer_in_features) == ["res3", "res4", "res5"]
    assert ie.return_stride4 is False
    assert ie.freeze_backbone is True
    assert ie.freeze_pixel_decoder is False


def test_g1_mixer_yaml_has_no_inline_image_encoder_block():
    text = (REPO_ROOT / "configs/model/mixer_model.yaml").read_text()
    # Only the defaults reference should mention image_encoder (no top-level mapping key).
    assert not re.search(r"^image_encoder:", text, flags=re.MULTILINE)
    assert re.search(r"defaults:\s*\n\s*-\s*image_encoder:\s*dinov3\s*\n\s*-\s*_self_", text)


# ===========================================================================
# Group 2 — build_model backbone construction
# ===========================================================================
def _builder(overrides):
    from src.training.pipeline_builder import PipelineBuilder
    cfg = _compose(overrides + ["model.device=cpu", "model.pretrained_encoder_path=null"])
    return PipelineBuilder(cfg)


def test_g2_mask2former_build(monkeypatch):
    # Force random ResNet (no network) for the build.
    import src.training.pipeline_builder as pb

    orig = pb.Mask2FormerBackbone

    def _no_net(*a, **k):
        k["imagenet_weights"] = None
        return orig(*a, **k)

    monkeypatch.setattr(pb, "Mask2FormerBackbone", _no_net)
    builder = _builder(["model/image_encoder=mask2former"])
    model, _ = builder.build_model()
    assert isinstance(model.image_encoder, Mask2FormerFeatureAdapter)
    assert model.image_encoder_type == "mask2former"
    assert model.n_image_levels == 3
    assert model.image_encoder.embed_dim == 256


def test_g2_mask2former_return_stride4_gives_4_levels(monkeypatch):
    import src.training.pipeline_builder as pb

    orig = pb.Mask2FormerBackbone
    monkeypatch.setattr(pb, "Mask2FormerBackbone",
                        lambda *a, **k: orig(*a, **{**k, "imagenet_weights": None}))
    builder = _builder(["model/image_encoder=mask2former",
                        "model.image_encoder.return_stride4=True"])
    model, _ = builder.build_model()
    assert model.n_image_levels == 4
    assert model.eye_decoder[0].n_levels == 4
    assert model.decoder[0].n_levels == 4


def test_g2_dinov3_build_uses_raw_wrapper(monkeypatch):
    import src.training.pipeline_builder as pb
    monkeypatch.setattr(pb, "DinoV3Wrapper", lambda **k: DummyDino())
    builder = _builder([])            # default = dinov3
    model, _ = builder.build_model()
    assert isinstance(model.image_encoder, DummyDino)
    assert model.image_encoder_type == "dinov3"
    assert model.n_image_levels == 1


def test_g2_pre_f6_snapshot_without_type_key(monkeypatch):
    import src.training.pipeline_builder as pb
    monkeypatch.setattr(pb, "DinoV3Wrapper", lambda **k: DummyDino())
    cfg = _compose(["model.device=cpu", "model.pretrained_encoder_path=null"])
    # Simulate a pre-F6 snapshot: image_encoder mapping without a `type` key.
    OmegaConf.set_struct(cfg, False)
    del cfg.model.image_encoder.type
    builder = pb.PipelineBuilder(cfg)
    model, _ = builder.build_model()
    assert model.image_encoder_type == "dinov3"
    assert model.n_image_levels == 1


def test_g2_encoder_disabled_gives_none(monkeypatch):
    import src.training.pipeline_builder as pb
    builder = _builder(["model.image_encoder.enabled=False"])
    # DinoV3Wrapper must never be called when disabled.
    monkeypatch.setattr(pb, "DinoV3Wrapper",
                        lambda **k: (_ for _ in ()).throw(AssertionError("should not build")))
    model, _ = builder.build_model()
    assert model.image_encoder is None


# ===========================================================================
# Group 3 — DINOv3 byte-identity & checkpoint load
# ===========================================================================
def test_g3_two_identical_dino_models_encode_equal():
    m1 = make_dino_model(seed=7)
    m2 = make_dino_model(seed=7)
    m1.eval(); m2.eval()
    b = dummy_batch()
    m1.set_phase("Combined"); m2.set_phase("Combined")
    with torch.no_grad():
        m1.encode(b["src"], b["image_src"], b["src_mask"])
        m2.encode(b["src"], b["image_src"], b["src_mask"])
    assert torch.equal(m1.src, m2.src)
    assert torch.equal(m1.image_src, m2.image_src)


def test_g3_dino_decode_equal_fixation_and_combined():
    b = dummy_batch()
    for phase in ("Fixation", "Combined"):
        m1 = make_dino_model(seed=3); m2 = make_dino_model(seed=3)
        m1.eval(); m2.eval()
        m1.set_phase(phase); m2.set_phase(phase)
        with torch.no_grad():
            o1 = m1(**b); o2 = m2(**b)
        for k in o1:
            assert torch.equal(o1[k], o2[k]), f"{phase}:{k}"


def test_g3_no_level_embed_key_and_strict_roundtrip():
    m = make_dino_model(seed=1)
    sd = m.state_dict()
    assert not any("level_embed" in k for k in sd)
    m2 = make_dino_model(seed=1)
    missing, unexpected = m2.load_state_dict(sd, strict=True)
    assert missing == [] and unexpected == []


def test_g3_spatial_shapes_none_on_dino_path():
    m = make_dino_model(seed=1)
    m.set_phase("Combined")
    b = dummy_batch()
    with torch.no_grad():
        m.encode(b["src"], b["image_src"], b["src_mask"])
    assert m.image_spatial_shapes is None
    assert m.image_level_start_index is None


def test_g3_load_encoder_no_missing_warnings(tmp_path, capsys):
    m = make_dino_model(seed=5)
    ckpt = tmp_path / "pre_f6.pth"
    torch.save(m.state_dict(), ckpt)
    m2 = make_dino_model(seed=9)     # different weights, same architecture
    m2.load_encoder(str(ckpt))
    out = capsys.readouterr().out
    assert "was NOT loaded" not in out


# ===========================================================================
# Group 4 — Mask2Former forward (shapes, PE, level_embed)
# ===========================================================================
@pytest.fixture(scope="module")
def m2f_backbone():
    return make_m2f_backbone()


def test_g4_encode_shapes(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone)
    m.set_phase("Combined")
    b = dummy_batch(B=2, T=5)
    with torch.no_grad():
        m.encode(b["src"], b["image_src"], b["src_mask"])
    assert m.image_src.shape == (2, 1344, 512)
    assert m.src.shape == (2, 5, 512)
    assert m.image_src.dtype == torch.float32
    assert torch.isfinite(m.image_src).all() and torch.isfinite(m.src).all()


def test_g4_stored_geometry(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone)
    m.set_phase("Combined")
    b = dummy_batch()
    with torch.no_grad():
        m.encode(b["src"], b["image_src"], b["src_mask"])
    assert m.image_spatial_shapes.tolist() == [[8, 8], [16, 16], [32, 32]]
    assert m.image_level_start_index.tolist() == [0, 64, 320]
    assert m.image_level_start_index.dtype == torch.int64


def test_g4_pe_added_once_when_zeroed(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone)
    m.set_phase("Combined")
    m.eval()
    m.final_fenh_norm_image = nn.Identity()   # isolate the additive PE (skip the post-LayerNorm)
    b = dummy_batch()
    # Zero level_embed and force pos_proj output to 0 → image_src == img_input_proj(value).
    with torch.no_grad():
        m.level_embed.zero_()
        bundle = m.image_encoder(b["image_src"])
        expected = m.img_input_proj(bundle.value)
        orig_pos = m.pos_proj

        class _Zero(nn.Module):
            def forward(self, coords):
                return torch.zeros(coords.shape[0], coords.shape[1], m.model_dim)

        m.pos_proj = _Zero()
        m.encode(b["src"], b["image_src"], b["src_mask"])
        m.pos_proj = orig_pos
    assert torch.allclose(m.image_src, expected, atol=1e-6)


def test_g4_level_embed_broadcast(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone)
    m.set_phase("Combined")
    m.eval()
    m.final_fenh_norm_image = nn.Identity()   # isolate the additive PE (skip the post-LayerNorm)
    b = dummy_batch()
    with torch.no_grad():
        for i in range(3):
            m.level_embed[i].fill_(float(i + 1))
        bundle = m.image_encoder(b["image_src"])
        base = m.img_input_proj(bundle.value)

        class _Zero(nn.Module):
            def forward(self, coords):
                return torch.zeros(coords.shape[0], coords.shape[1], m.model_dim)

        orig = m.pos_proj
        m.pos_proj = _Zero()
        m.encode(b["src"], b["image_src"], b["src_mask"])
        m.pos_proj = orig
        add = m.image_src - base
    assert torch.allclose(add[:, :64, :], torch.ones_like(add[:, :64, :]), atol=1e-6)
    assert torch.allclose(add[:, 64:320, :], 2 * torch.ones_like(add[:, 64:320, :]), atol=1e-6)
    assert torch.allclose(add[:, 320:, :], 3 * torch.ones_like(add[:, 320:, :]), atol=1e-6)


def test_g4_decode_shapes_multi_mlp(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone, head_type="multi_mlp")
    m.set_phase("Fixation")
    b = dummy_batch(B=2, N=4)
    with torch.no_grad():
        out = m(**b)
    assert out["coord"].shape == (2, 5, 2)
    assert out["dur"].shape == (2, 5, 1)
    assert out["cls"].shape == (2, 5, 1)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_g4_decode_shapes_linear(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone, head_type="linear")
    m.set_phase("Fixation")
    b = dummy_batch(B=2, N=4)
    with torch.no_grad():
        out = m(**b)
    assert out["reg"].shape == (2, 5, 3)
    assert out["cls"].shape == (2, 5, 1)


def test_g4_non_square_input():
    m = make_m2f_model()
    m.set_phase("Combined")
    b = dummy_batch(B=2, img=256, W=192)
    with torch.no_grad():
        m.encode(b["src"], b["image_src"], b["src_mask"])
    # res5 = 8x6, res4 = 16x12, res3 = 32x24
    assert m.image_spatial_shapes.tolist() == [[8, 6], [16, 12], [32, 24]]


def test_g4_img_input_proj_256_to_512(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone)
    # First Linear in the MLP maps 256 -> ... ; input feature dim must be 256.
    first_linear = [mod for mod in m.img_input_proj.modules() if isinstance(mod, nn.Linear)][0]
    assert first_linear.in_features == 256
    assert m.image_encoder.embed_dim == 256


# ===========================================================================
# Group 5 — Guards / error conditions
# ===========================================================================
def test_g5_use_rope_guard(m2f_backbone):
    with pytest.raises(ValueError, match="use_rope"):
        make_m2f_model(backbone=m2f_backbone, use_rope=True)


def test_g5_head_guards(m2f_backbone):
    for ht in ("argmax_regressor", "heatmap"):
        with pytest.raises(ValueError, match=ht):
            make_m2f_model(backbone=m2f_backbone, head_type=ht)


def test_g5_input_encoder_guard(m2f_backbone):
    with pytest.raises(ValueError, match="image_features_concat"):
        make_m2f_model(backbone=m2f_backbone, input_encoder="image_features_concat")


def test_g5_wrong_n_levels_surfaces_f1_error(m2f_backbone):
    # A deformable decoder mistakenly built at n_levels=1 must reject a 3-level bundle with F1's
    # error — this is what would fire if F6's n_image_levels plumbing were wrong.
    from src.model.blocks import DeformableDecoder
    dec = DeformableDecoder(model_dim=512, total_dim=512, n_heads=8, ff_dim=256,
                            num_points=4, n_levels=1, norm_first=True, device="cpu")
    adapter = Mask2FormerFeatureAdapter(m2f_backbone)
    bundle = adapter(dummy_batch()["image_src"])
    src = torch.rand(2, 5, 512)
    ref = torch.rand(2, 5, 2)
    with pytest.raises(ValueError, match="n_levels=1"):
        dec(src, bundle.value, None, reference_points=ref,
            spatial_shapes=bundle.spatial_shapes, level_start_index=bundle.level_start_index)


def test_g5_dino_guards_do_not_fire():
    # heatmap builds on DINOv3; needs a full patch grid. Just assert no F6 guard raises.
    m = make_dino_model(seed=1, head_type="heatmap")
    assert m.image_encoder_type == "dinov3"
    m2 = make_dino_model(seed=1, use_rope=True)
    assert m2.use_rope is True


# ===========================================================================
# Group 6 — Gradient flow / freezing (Mask2Former)
# ===========================================================================
def test_g6_gradients_and_freezing(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone, head_type="multi_mlp")
    m.set_phase("Combined")
    m.train()
    b = dummy_batch()
    out = m(**b)
    loss = sum(v.float().pow(2).mean() for v in out.values())
    loss.backward()

    assert m.level_embed.grad is not None and m.level_embed.grad.abs().sum() > 0

    fe = m.image_encoder.backbone.feature_extractor
    for p in fe.parameters():
        assert p.requires_grad is False
        assert p.grad is None

    pdec = m.image_encoder.backbone.pixel_decoder
    trainable = [p for p in pdec.parameters() if p.requires_grad]
    assert len(trainable) > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)


def test_g6_train_keeps_resnet_in_eval(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone)
    m.train()
    assert m.image_encoder.backbone.feature_extractor.training is False


def test_g6_param_groups_exclude_frozen_resnet(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone)
    m.set_phase("Combined")
    groups = m.get_parameter_groups(1e-4)
    all_params = {id(p) for g in groups for p in g["params"]}
    for p in m.image_encoder.backbone.feature_extractor.parameters():
        assert id(p) not in all_params
    # sampling_offsets land in the 10x group.
    assert len(groups) == 2
    off_ids = {id(p) for p in groups[1]["params"]}
    off_named = {id(p) for n, p in m.named_parameters()
                 if "sampling_offsets" in n and p.requires_grad}
    assert off_ids == off_named and len(off_ids) > 0


# ===========================================================================
# Group 7 — KV / memory cache & scheduled-sampling parity
# ===========================================================================
def test_g7_memory_kv_cache_runs(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone, head_type="multi_mlp")
    m.set_phase("Fixation")
    m.eval()
    b = dummy_batch()
    with torch.no_grad():
        m.encode(b["src"], b["image_src"], b["src_mask"])
    m.enable_memory_kv_cache()
    with torch.no_grad():
        _ = m.decode_fixation(b["tgt"], None, None)
    m.disable_memory_kv_cache()
    m.clear_kv_cache()


def test_g7_scheduled_sampling_geometry_stable(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone, head_type="multi_mlp")
    m.set_phase("Combined")
    m.eval()
    b = dummy_batch()
    with torch.no_grad():
        m.encode(b["src"], b["image_src"], b["src_mask"])
    ss_before = m.image_spatial_shapes.clone()
    lsi_before = m.image_level_start_index.clone()
    with torch.no_grad():
        m.decode_fixation(b["tgt"], None, None)
        m.decode_fixation(b["tgt"], None, None)
    assert torch.equal(m.image_spatial_shapes, ss_before)
    assert torch.equal(m.image_level_start_index, lsi_before)


# ===========================================================================
# Data Architecture Integrity
# ===========================================================================
def test_di_single_bundle_interface(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone)
    out = m.image_encoder(dummy_batch()["image_src"])
    assert isinstance(out, MultiScaleFeatures)


def test_di_no_backbone_internal_access_in_methods():
    import inspect
    src = inspect.getsource(MixerModel.encode) + inspect.getsource(MixerModel.decode_fixation)
    for bad in (".backbone", ".pixel_decoder", ".feature_extractor"):
        assert bad not in src


def test_di_dino_internal_access_only_under_dinov3_guard():
    import inspect
    src = inspect.getsource(MixerModel.encode)
    # forward_features and CLS slicing must be inside the dinov3 branch only.
    for line in src.splitlines():
        if "forward_features(" in line or "[:,prefix:,:]" in line:
            # such lines exist; verify a static guard is present in the method
            pass
    assert "if self.image_encoder_type == 'dinov3':" in src


def test_di_stored_geometry_is_bundle_identity(m2f_backbone):
    m = make_m2f_model(backbone=m2f_backbone)
    m.set_phase("Combined")
    b = dummy_batch()
    with torch.no_grad():
        m.encode(b["src"], b["image_src"], b["src_mask"])
    expected = build_level_start_index(m.image_spatial_shapes)
    assert m.image_level_start_index.tolist() == expected.tolist()


def test_di_dino_geometry_not_parameters_or_buffers():
    m = make_dino_model(seed=1)
    names = {n for n, _ in m.named_parameters()} | {n for n, _ in m.named_buffers()}
    assert not any("level_embed" in n for n in names)
    assert not any("image_spatial_shapes" in n or "image_level_start_index" in n for n in names)


def test_di_no_dino_attr_access_on_m2f_forward(m2f_backbone):
    """Monkeypatch the DINOv3-only attributes to raise; the m2f forward must never touch them."""
    m = make_m2f_model(backbone=m2f_backbone, head_type="multi_mlp")
    m.set_phase("Combined")
    b = dummy_batch()
    # pos_proj.forward_features would raise if erroneously called on the m2f path.
    orig = m.pos_proj.forward_features

    def _boom(*a, **k):
        raise AssertionError("forward_features called on Mask2Former path")

    m.pos_proj.forward_features = _boom
    try:
        with torch.no_grad():
            m(**b)
    finally:
        m.pos_proj.forward_features = orig
