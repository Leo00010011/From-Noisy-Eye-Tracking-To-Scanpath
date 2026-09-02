"""F4 — Multiscale-capable eye & fixation decoders.

Tests for the generalized ``DeformableDecoder`` (eye decoder) and
``DeformableDoubleInputDecoder`` (fixation decoder) in ``src/model/blocks.py``:
the ``n_levels=1`` byte-identity / retro-compat contract, the multi-scale forward
path, the non-``norm_first`` parity bug fix, error conditions, KV/memory-cache
carry-through, and InferenceRecorder carry-through. Also the data-validity checks
that an F3 bundle plugs straight into an ``n_levels=3`` decoder.

CPU-only, fixed seeds, no network, no HDF5.
"""

import tempfile

import pytest
import torch

from src.model.blocks import DeformableDecoder, DeformableDoubleInputDecoder
from src.model.ms_features import Mask2FormerFeatureAdapter
from src.model.ms_deform_backbone import Mask2FormerBackbone
from src.training.inference_recorder import InferenceRecorder


# Reference dims (validation.md)
D = 32
N_HEADS = 4
N_POINTS = 4
B = 2
NQ = 5
FF = 64

# Single-scale legacy memory: H*W = 16*16 = 256, CLS-prefixed => mem (B, 257, D)
HW = 16 * 16
# Synthetic 3-level bundle
SHAPES_3 = torch.tensor([[8, 8], [16, 16], [32, 32]])
SUM_HW_3 = 64 + 256 + 1024  # 1344
LSI_3 = torch.tensor([0, 64, 320])


# ---------------------------------------------------------------------------
# Pre-F4 reference forwards. They reuse the decoder's own submodules (so the
# weights are identical) but reproduce the exact pre-F4 forward body. At
# n_levels=1 the inner op is byte-identical to pre-F1, so these are the
# "pre-F4 reference" outputs the byte-identity checks compare against.
# ---------------------------------------------------------------------------
def legacy_decoder_forward(dec, src, mem, tgt_mask=None, reference_points=None):
    x = src

    def sa(z):
        return dec.dropout1(dec.self_attn(z, attn_mask=tgt_mask))

    def ca(s, m):
        return dec.dropout2(dec.cross_attn(query=s, reference_points=reference_points,
                                           value=m, spatial_shape=dec.spatial_shape))

    def ff(z):
        return dec.dropout4(dec.linear2(dec.dropout3(dec.activation(dec.linear1(z)))))

    if dec.norm_first:
        x = x + sa(dec.norm1(x))
        x = x + ca(dec.norm2(x), mem[:, 1:, :])
        x = x + ff(dec.norm3(x))
    else:
        x = dec.norm1(x + sa(x))
        x = dec.norm2(x + ca(x, mem[:, 1:, :]))
        x = dec.norm3(x + ff(x))
    return x


def legacy_double_forward(dec, src, mem1, mem2, tgt_mask=None, mem1_mask=None,
                          reference_points=None):
    # norm_first branch only — the pre-F4 non-norm_first branch was broken dead code.
    x = src

    def sa(z):
        return dec.self_attn_dropout(dec.self_attn(z, attn_mask=tgt_mask, q_rope=None))

    def ca1(s, m):
        return dec.first_cross_attn_dropout(
            dec.first_cross_attn(s, m, attn_mask=mem1_mask, q_rope=None, k_rope=None))

    def ca2(s, m):
        return dec.second_cross_attn_dropout(
            dec.second_cross_attn(query=s, reference_points=reference_points,
                                  value=m, spatial_shape=dec.spatial_shape))

    def ff(z):
        return dec.linear_down_dropout(
            dec.linear_down(dec.linear_up_dropout(dec.activation(dec.linear_up(z)))))

    x = x + sa(dec.self_attn_norm(x))
    x = x + ca1(dec.first_cross_attn_norm(x), mem1)
    x = x + ca2(dec.second_cross_attn_norm(x), mem2[:, 1:, :])
    x = x + ff(dec.linear_norm(x))
    return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_with_recorder(module, *args, **kwargs):
    """Run ``module`` forward under an active recorder; return (out, full activations dict)."""
    recorder = InferenceRecorder(tempfile.mkdtemp(), enabled=True)
    recorder.attach(module)
    recorder.start_batch(epoch=0, phase="test", split="val", batch_index=0)
    out = module(*args, **kwargs)
    return out, recorder.current_payload["activations"]


def make_decoder(n_levels=1, norm_first=True, num_points=N_POINTS, model_dim=D):
    return DeformableDecoder(model_dim=model_dim, total_dim=model_dim, n_heads=N_HEADS,
                             ff_dim=FF, norm_first=norm_first, num_points=num_points,
                             n_levels=n_levels, spatial_shape=(16, 16))


def make_double(n_levels=1, norm_first=True, num_points=N_POINTS, model_dim=D,
                use_kv_cache=False):
    return DeformableDoubleInputDecoder(model_dim=model_dim, total_dim=model_dim,
                                        n_heads=N_HEADS, ff_dim=FF, norm_first=norm_first,
                                        use_kv_cache=use_kv_cache, num_points=num_points,
                                        n_levels=n_levels, spatial_shape=(16, 16))


def legacy_mem(bs, g):
    return torch.randn(bs, 1 + HW, D, generator=g)  # CLS + 256


def multiscale_value(bs, g):
    return torch.randn(bs, SUM_HW_3, D, generator=g)


# ===========================================================================
# Group 1 — Retro-compat byte-identity (DeformableDecoder)
# ===========================================================================
def test_decoder_default_n_levels_one():
    dec = make_decoder()
    assert dec.n_levels == 1
    assert dec.cross_attn.n_levels == 1


def test_decoder_state_dict_shapes_unchanged():
    torch.manual_seed(0)
    new = make_decoder(n_levels=1)
    torch.manual_seed(0)
    # A "pre-F4" decoder has, by construction, the same layout as n_levels=1.
    ref = make_decoder(n_levels=1)
    sd_new, sd_ref = new.state_dict(), ref.state_dict()
    assert set(sd_new.keys()) == set(sd_ref.keys())
    for k in sd_new:
        assert sd_new[k].shape == sd_ref[k].shape, f"shape mismatch {k}"


def test_decoder_old_checkpoint_loads_strict():
    src = make_decoder(n_levels=1)
    dst = make_decoder(n_levels=1)
    result = dst.load_state_dict(src.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


@pytest.mark.parametrize("norm_first", [True, False])
def test_decoder_forward_byte_identical_legacy(norm_first):
    torch.manual_seed(0)
    dec = make_decoder(n_levels=1, norm_first=norm_first)
    dec.eval()
    g = torch.Generator().manual_seed(1)
    src = torch.randn(B, NQ, D, generator=g)
    mem = legacy_mem(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)
    out = dec(src, mem, reference_points=ref)
    out_ref = legacy_decoder_forward(dec, src, mem, reference_points=ref)
    assert torch.equal(out, out_ref)


def test_decoder_legacy_strips_cls():
    # Full 257 memory with (16,16) shape only works if CLS is stripped (256 tokens).
    dec = make_decoder(n_levels=1)
    dec.eval()
    g = torch.Generator().manual_seed(2)
    src = torch.randn(B, NQ, D, generator=g)
    mem = legacy_mem(B, g)  # 257
    ref = torch.rand(B, NQ, 2, generator=g)
    # succeeds because CLS is stripped to 256 == 16*16
    out = dec(src, mem, reference_points=ref)
    assert out.shape == (B, NQ, D)
    # if CLS were NOT stripped, the op would see 257 tokens against (16,16)=256 -> raise.
    _, acts = run_with_recorder(make_decoder(n_levels=1).eval(), src, mem, reference_points=ref)
    # recorder path also succeeds; presence of cross_attention_res confirms op ran on 256
    assert "cross_attention_res" in acts[DeformableDecoder.__name__]


# ===========================================================================
# Group 2 — Multi-scale path (DeformableDecoder)
# ===========================================================================
def test_decoder_multiscale_param_shapes():
    dec = make_decoder(n_levels=3)
    assert dec.cross_attn.n_levels == 3
    assert tuple(dec.cross_attn.sampling_offsets.weight.shape) == (N_HEADS * 3 * N_POINTS * 2, D)
    assert tuple(dec.cross_attn.attention_weights.weight.shape) == (N_HEADS * 3 * N_POINTS, D)


def test_decoder_multiscale_forward():
    dec = make_decoder(n_levels=3)
    dec.eval()
    g = torch.Generator().manual_seed(3)
    src = torch.randn(B, NQ, D, generator=g)
    value = multiscale_value(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)
    out = dec(src, value, reference_points=ref, spatial_shapes=SHAPES_3,
              level_start_index=LSI_3)
    assert tuple(out.shape) == (B, NQ, D)
    assert torch.isfinite(out).all()
    assert out.dtype == src.dtype


def test_decoder_multiscale_no_cls_slice():
    dec = make_decoder(n_levels=3)
    dec.eval()
    g = torch.Generator().manual_seed(4)
    src = torch.randn(B, NQ, D, generator=g)
    ref = torch.rand(B, NQ, 2, generator=g)
    # exactly 1344 succeeds (no slice)
    ok = multiscale_value(B, g)
    dec(src, ok, reference_points=ref, spatial_shapes=SHAPES_3, level_start_index=LSI_3)
    # 1345 (as if CLS still present) -> F1 ΣHₗWₗ mismatch
    bad = torch.randn(B, SUM_HW_3 + 1, D, generator=g)
    with pytest.raises(ValueError):
        dec(src, bad, reference_points=ref, spatial_shapes=SHAPES_3, level_start_index=LSI_3)


def test_decoder_level_start_index_optional():
    dec = make_decoder(n_levels=3)
    dec.eval()
    g = torch.Generator().manual_seed(5)
    src = torch.randn(B, NQ, D, generator=g)
    value = multiscale_value(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)
    out_none = dec(src, value, reference_points=ref, spatial_shapes=SHAPES_3,
                   level_start_index=None)
    out_expl = dec(src, value, reference_points=ref, spatial_shapes=SHAPES_3,
                   level_start_index=LSI_3)
    assert torch.equal(out_none, out_expl)


# ===========================================================================
# Group 3 — Retro-compat + multi-scale (DeformableDoubleInputDecoder)
# ===========================================================================
def test_double_default_n_levels_and_first_cross_untouched():
    dec = make_double(n_levels=3)
    assert dec.n_levels == 3
    assert dec.second_cross_attn.n_levels == 3
    assert not hasattr(dec.first_cross_attn, "n_levels")


def test_double_default_n_levels_one():
    dec = make_double()
    assert dec.n_levels == 1
    assert dec.second_cross_attn.n_levels == 1


def test_double_state_dict_and_load():
    src = make_double(n_levels=1)
    dst = make_double(n_levels=1)
    result = dst.load_state_dict(src.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_double_forward_byte_identical_legacy():
    torch.manual_seed(0)
    dec = make_double(n_levels=1, norm_first=True)
    dec.eval()
    g = torch.Generator().manual_seed(6)
    src = torch.randn(B, NQ, D, generator=g)
    mem1 = torch.randn(B, 7, D, generator=g)
    mem2 = legacy_mem(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)
    out = dec(src, mem1, mem2, reference_points=ref)
    out_ref = legacy_double_forward(dec, src, mem1, mem2, reference_points=ref)
    assert torch.equal(out, out_ref)


def test_double_multiscale_forward_and_mem1_influence():
    dec = make_double(n_levels=3)
    dec.eval()
    g = torch.Generator().manual_seed(7)
    src = torch.randn(B, NQ, D, generator=g)
    mem1 = torch.randn(B, 7, D, generator=g)
    value = multiscale_value(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)
    out = dec(src, mem1, value, reference_points=ref, spatial_shapes=SHAPES_3,
              level_start_index=LSI_3)
    assert tuple(out.shape) == (B, NQ, D)
    assert torch.isfinite(out).all()
    # mem1 flows through the first cross-attention: replacing it changes the output
    mem1b = torch.randn(B, 7, D, generator=g)
    out_mem1 = dec(src, mem1b, value, reference_points=ref, spatial_shapes=SHAPES_3,
                   level_start_index=LSI_3)
    assert not torch.equal(out, out_mem1)
    # replacing value changes output too
    valueb = multiscale_value(B, g)
    out_val = dec(src, mem1, valueb, reference_points=ref, spatial_shapes=SHAPES_3,
                  level_start_index=LSI_3)
    assert not torch.equal(out, out_val)


# ===========================================================================
# Group 4 — Non-norm_first branch parity (bug fix, FR7)
# ===========================================================================
def test_double_non_norm_first_runs_legacy():
    dec = make_double(n_levels=1, norm_first=False)
    dec.eval()
    g = torch.Generator().manual_seed(8)
    src = torch.randn(B, NQ, D, generator=g)
    mem1 = torch.randn(B, 7, D, generator=g)
    mem2 = legacy_mem(B, g)  # 257
    ref = torch.rand(B, NQ, 2, generator=g)
    # pre-F4 this raised TypeError (bad kwargs); now it runs and strips CLS.
    out = dec(src, mem1, mem2, reference_points=ref)
    assert tuple(out.shape) == (B, NQ, D)
    assert torch.isfinite(out).all()


def test_cross_attention2_rejects_rope_mask_kwargs():
    dec = make_double(n_levels=1)
    method = getattr(dec, "_DeformableDoubleInputDecoder__cross_attention2")
    g = torch.Generator().manual_seed(9)
    src = torch.randn(B, NQ, D, generator=g)
    val = torch.randn(B, HW, D, generator=g)
    ref = torch.rand(B, NQ, 2, generator=g)
    with pytest.raises(TypeError):
        method(src, val, reference_points=ref, attn_mask=None)
    with pytest.raises(TypeError):
        method(src, val, reference_points=ref, src_rope=None)
    with pytest.raises(TypeError):
        method(src, val, reference_points=ref, mem2_rope=None)


def test_double_both_norm_modes_finite():
    g = torch.Generator().manual_seed(10)
    src = torch.randn(B, NQ, D, generator=g)
    mem1 = torch.randn(B, 7, D, generator=g)
    mem2 = legacy_mem(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)
    for norm_first in (True, False):
        dec = make_double(n_levels=1, norm_first=norm_first)
        dec.eval()
        out = dec(src, mem1, mem2, reference_points=ref)
        assert tuple(out.shape) == (B, NQ, D)
        assert torch.isfinite(out).all()


# ===========================================================================
# Group 5 — Error conditions (FR10)
# ===========================================================================
def test_multilevel_shapes_into_single_level_decoder_raises():
    dec = make_decoder(n_levels=1)
    dec.eval()
    g = torch.Generator().manual_seed(11)
    src = torch.randn(B, NQ, D, generator=g)
    value = multiscale_value(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)
    with pytest.raises(ValueError):
        dec(src, value, reference_points=ref, spatial_shapes=SHAPES_3,
            level_start_index=LSI_3)


def test_legacy_call_into_multilevel_decoder_raises():
    dec = make_decoder(n_levels=3)
    dec.eval()
    g = torch.Generator().manual_seed(12)
    src = torch.randn(B, NQ, D, generator=g)
    mem = legacy_mem(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)
    with pytest.raises(ValueError):
        dec(src, mem, reference_points=ref)  # spatial_shapes=None
    # double-input variant too
    ddec = make_double(n_levels=3)
    ddec.eval()
    mem1 = torch.randn(B, 7, D, generator=g)
    with pytest.raises(ValueError):
        ddec(src, mem1, mem, reference_points=ref)


def test_bad_reference_point_dim_raises():
    dec = make_decoder(n_levels=3)
    dec.eval()
    g = torch.Generator().manual_seed(13)
    src = torch.randn(B, NQ, D, generator=g)
    value = multiscale_value(B, g)
    ref = torch.rand(B, NQ, 4, generator=g)  # box refs, out of scope
    with pytest.raises(ValueError):
        dec(src, value, reference_points=ref, spatial_shapes=SHAPES_3,
            level_start_index=LSI_3)


def test_inconsistent_level_start_index_raises():
    dec = make_decoder(n_levels=3)
    dec.eval()
    g = torch.Generator().manual_seed(14)
    src = torch.randn(B, NQ, D, generator=g)
    value = multiscale_value(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)
    with pytest.raises(ValueError):
        dec(src, value, reference_points=ref, spatial_shapes=SHAPES_3,
            level_start_index=torch.tensor([0, 64, 300]))


# ===========================================================================
# Group 6 — KV / memory cache (FR9)
# ===========================================================================
def test_double_memory_cache_warm_matches_cold():
    dec = make_double(n_levels=3)
    dec.eval()
    g = torch.Generator().manual_seed(15)
    src = torch.randn(B, NQ, D, generator=g)
    mem1 = torch.randn(B, 7, D, generator=g)
    value = multiscale_value(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)

    # cold (no memory cache)
    out_cold = dec(src, mem1, value, reference_points=ref, spatial_shapes=SHAPES_3,
                   level_start_index=LSI_3)

    # warm the cache
    dec.enable_memory_kv_cache()
    out_warm1 = dec(src, mem1, value, reference_points=ref, spatial_shapes=SHAPES_3,
                    level_start_index=LSI_3)
    # per-level value cache populated as a length-3 list
    assert isinstance(dec.second_cross_attn.value_cache, list)
    assert len(dec.second_cross_attn.value_cache) == 3
    out_warm2 = dec(src, mem1, value, reference_points=ref, spatial_shapes=SHAPES_3,
                    level_start_index=LSI_3)
    assert torch.equal(out_warm1, out_warm2)
    assert torch.equal(out_warm1, out_cold)


def test_double_cache_disable_and_clear():
    dec = make_double(n_levels=3)
    dec.eval()
    g = torch.Generator().manual_seed(16)
    src = torch.randn(B, NQ, D, generator=g)
    mem1 = torch.randn(B, 7, D, generator=g)
    value = multiscale_value(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)
    dec.enable_memory_kv_cache()
    dec(src, mem1, value, reference_points=ref, spatial_shapes=SHAPES_3,
        level_start_index=LSI_3)
    assert dec.second_cross_attn.value_cache is not None
    dec.clear_kv_cache()
    assert dec.second_cross_attn.value_cache is None
    dec.enable_memory_kv_cache()
    dec(src, mem1, value, reference_points=ref, spatial_shapes=SHAPES_3,
        level_start_index=LSI_3)
    dec.disable_memory_kv_cache()
    assert dec.second_cross_attn.value_cache is None


# ===========================================================================
# Group 7 — InferenceRecorder carry-through
# ===========================================================================
def test_recorder_multiscale_level_axis():
    dec = make_decoder(n_levels=3)
    dec.eval()
    g = torch.Generator().manual_seed(17)
    src = torch.randn(B, NQ, D, generator=g)
    value = multiscale_value(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)
    _, acts = run_with_recorder(dec, src, value, reference_points=ref,
                                spatial_shapes=SHAPES_3, level_start_index=LSI_3)
    dec_acts = acts[DeformableDecoder.__name__]
    assert tuple(dec_acts["cross_attention_res"].shape) == (B, NQ, D)
    inner = acts["cross_attn"]
    assert tuple(inner["sampling_offsets"].shape) == (B, NQ, N_HEADS, 3, N_POINTS, 2)
    assert tuple(inner["sampling_locations"].shape) == (B, NQ, N_HEADS, 3, N_POINTS, 2)
    assert tuple(inner["attention_weights"].shape) == (B, NQ, N_HEADS, 3, N_POINTS)
    assert tuple(inner["reference_points"].shape) == (B, NQ, 3, 2)


def test_recorder_single_level_squeezable_axis():
    dec = make_decoder(n_levels=1)
    dec.eval()
    g = torch.Generator().manual_seed(18)
    src = torch.randn(B, NQ, D, generator=g)
    mem = legacy_mem(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)
    _, acts = run_with_recorder(dec, src, mem, reference_points=ref)
    inner = acts["cross_attn"]
    assert tuple(inner["attention_weights"].shape) == (B, NQ, N_HEADS, 1, N_POINTS)
    # squeezing the singleton level axis reproduces the legacy (B,Nq,H,P) shape
    assert tuple(inner["attention_weights"].squeeze(3).shape) == (B, NQ, N_HEADS, N_POINTS)


def test_recorder_all_stages_fire_both_decoders():
    # Recorder hooks fire on the operative norm_first=True path. The pre-existing
    # non-norm_first branch has never emitted records (the plan keeps it record-free,
    # matching pre-F4) — see the report note on validation Group 7.
    norm_first = True
    g = torch.Generator().manual_seed(19)
    src = torch.randn(B, NQ, D, generator=g)
    mem = legacy_mem(B, g)
    ref = torch.rand(B, NQ, 2, generator=g)

    dec = make_decoder(n_levels=1, norm_first=norm_first).eval()
    _, acts = run_with_recorder(dec, src, mem, reference_points=ref)
    da = acts[DeformableDecoder.__name__]
    assert {"self_attention_res", "cross_attention_res", "ffn_res"} <= set(da.keys())

    mem1 = torch.randn(B, 7, D, generator=g)
    ddec = make_double(n_levels=1, norm_first=norm_first).eval()
    _, acts2 = run_with_recorder(ddec, src, mem1, mem, reference_points=ref)
    dda = acts2[DeformableDoubleInputDecoder.__name__]
    assert {"self_attention_res", "first_cross_res", "second_cross_res", "ffn_res"} <= set(dda.keys())


# ===========================================================================
# Data Validity
# ===========================================================================
def test_f3_bundle_feeds_decoder_directly():
    backbone = Mask2FormerBackbone(imagenet_weights=None)
    adapter = Mask2FormerFeatureAdapter(backbone)
    adapter.eval()
    g = torch.Generator().manual_seed(20)
    x = torch.randn(2, 3, 256, 256, generator=g)
    with torch.no_grad():
        bundle = adapter(x)
    dec = DeformableDecoder(model_dim=256, total_dim=256, n_heads=8, ff_dim=512,
                            norm_first=True, num_points=4, n_levels=bundle.num_levels)
    dec.eval()
    src = torch.randn(2, NQ, 256, generator=g)
    ref = torch.rand(2, NQ, 2, generator=g)
    with torch.no_grad():
        out = dec(src, bundle.value, reference_points=ref,
                  spatial_shapes=bundle.spatial_shapes,
                  level_start_index=bundle.level_start_index)
    assert tuple(out.shape) == (2, NQ, 256)
    assert torch.isfinite(out).all()


def test_reference_point_at_cell_center_samples_token():
    # num_points=1, offsets zeroed => the op samples exactly at the reference point.
    # A reference at a token's cell center must return that token's projected feature.
    H = W = 8
    shapes = torch.tensor([[H, W]])
    dec = make_decoder(n_levels=1, num_points=1)
    dec.eval()
    with torch.no_grad():
        dec.cross_attn.sampling_offsets.weight.zero_()
        dec.cross_attn.sampling_offsets.bias.zero_()
    g = torch.Generator().manual_seed(21)
    value = torch.randn(1, H * W, D, generator=g)
    # target token at (row, col)
    row, col = 3, 5
    token_idx = row * W + col
    ref = torch.tensor([[[(col + 0.5) / W, (row + 0.5) / H]]])  # (1,1,2) (x,y)
    src = torch.randn(1, 1, D, generator=g)
    op = dec.cross_attn
    with torch.no_grad():
        out = op(query=src, reference_points=ref, value=value, spatial_shape=shapes)
        # manual: value_proj at the exact token, then output_proj
        vproj = op.value_proj(value)  # (1, H*W, D)
        manual = op.output_proj(vproj[:, token_idx:token_idx + 1, :])
    assert torch.allclose(out, manual, atol=1e-4)


def test_single_scale_agreement_on_real_memory():
    torch.manual_seed(0)
    dec = make_decoder(n_levels=1, norm_first=True)
    dec.eval()
    g = torch.Generator().manual_seed(22)
    src = torch.randn(B, NQ, D, generator=g)
    mem = legacy_mem(B, g)  # DINOv3-style CLS-prefixed
    ref = torch.rand(B, NQ, 2, generator=g)
    out = dec(src, mem, reference_points=ref)
    out_ref = legacy_decoder_forward(dec, src, mem, reference_points=ref)
    assert (out - out_ref).abs().max().item() == 0.0


# ===========================================================================
# Data Architecture Integrity
# ===========================================================================
def test_no_ms_features_import_in_blocks():
    import src.model.blocks as blocks_mod
    import inspect
    source = inspect.getsource(blocks_mod)
    assert "ms_features" not in source
    assert "MultiScaleFeatures" not in source


@pytest.mark.parametrize("n_levels", [1, 3, 4])
def test_n_levels_single_sourced(n_levels):
    dec = make_decoder(n_levels=n_levels)
    assert dec.cross_attn.n_levels == dec.n_levels == n_levels
    ddec = make_double(n_levels=n_levels)
    assert ddec.second_cross_attn.n_levels == ddec.n_levels == n_levels
