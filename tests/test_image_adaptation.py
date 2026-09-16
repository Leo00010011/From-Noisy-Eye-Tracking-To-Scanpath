"""Image-Feature Adaptation via Scanpath-Centroid Alignment Pretraining.

CPU-only, no network, no real caches. The Mask2Former path is exercised with a
``PrecomputedFeatureAdapter`` (embed_dim=256, tiny spatial_shapes -> S=29) as the image encoder,
so ``image_encoder_type == "mask2former"`` without building ResNet50. Centroid buffers are
synthetic (installed via ``set_alignment_centroids``). Clustering/cache tests use a small
``CocoFreeView``-like stub.

Groups mirror ``spec/2026-09-14-image-feature-adaptation-alignment/validation.md``.
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.data import scanpath_centroids as sc
from src.data.scanpath_centroids import ScanpathCentroidCache
from src.eval.eval_metrics import nearest_centroid_offsets, eval_align
from src.model.loss_functions import (AlignmentLoss, CombinedLossFunction, DenoiseRegLoss,
                                       SeparatedRegLossFunction, EndBinaryCrossEntropy)
from src.model.mixer_model import MixerModel
from src.model.ms_features import PrecomputedFeatureAdapter


# ---------------------------------------------------------------------------
# Stubs / builders
# ---------------------------------------------------------------------------
SPATIAL = [[2, 2], [3, 3], [4, 4]]          # S = 4 + 9 + 16 = 29
S_TOKENS = 29


class StubData:
    """Minimal CocoFreeView-like stub: fixed ptoa/dest_res, a handful of scanpaths over images."""

    def __init__(self, samples, dest_res=(320, 512)):
        # samples: list of dict(img_path, X, Y, split)
        self.samples = samples
        self.dest_res = dest_res
        self.ori_res = dest_res                # -> downscale factor 1.0
        self.ptoa = 1 / 16
        self.df = pd.DataFrame(samples)

    def __len__(self):
        return len(self.samples)

    def get_img_path(self, i):
        return self.samples[i]["img_path"]

    def get_scanpath(self, i, downscale=True):
        s = self.samples[i]
        fx = self.dest_res[1] / self.ori_res[1] if downscale else 1.0
        fy = self.dest_res[0] / self.ori_res[0] if downscale else 1.0
        x = np.asarray(s["X"], dtype=np.float32) * fx
        y = np.asarray(s["Y"], dtype=np.float32) * fy
        return x, y, np.arange(len(x))


def two_cluster_samples(img_path="a.jpg", n_per=25, split="train", seed=0):
    """50 points tightly around two locations ~10 DVA (160 px) apart -> 2 centroids."""
    rng = np.random.default_rng(seed)
    c1 = np.array([100.0, 100.0])
    c2 = np.array([260.0, 100.0])            # 160 px == 10 DVA apart
    p1 = c1 + rng.normal(0, 1.0, size=(n_per, 2))
    p2 = c2 + rng.normal(0, 1.0, size=(n_per, 2))
    return dict(img_path=img_path, X=np.concatenate([p1[:, 0], p2[:, 0]]),
                Y=np.concatenate([p1[:, 1], p2[:, 1]]), split=split), c1, c2


IA_KW = dict(
    n_encoder=1, n_decoder=1, n_eye_decoder=1, n_feature_enhancer=0,
    model_dim=64, total_dim=64, n_heads=8, ff_dim=64,
    max_pos_enc=90, max_pos_dec=26,
    input_encoder="shared_gaussian", norm_first=True,
    mlp_head_hidden_dim=[32], pos_enc_hidden_dim=32, num_freq_bands=8,
    pos_enc_sigma=1.0, use_deformable_eye_decoder=True,
    use_deformable_fixation_decoder=True, head_type="multi_mlp",
    phases=["ImageAdaptation", "Combined", "FullFinetune"],
    add_denoise_head=False, activation=F.gelu, device="cpu",
)


def make_model(image_adaptation=True, image_encoder="mask2former", **over):
    kw = dict(IA_KW)
    kw.update(over)
    if image_encoder == "mask2former":
        enc = PrecomputedFeatureAdapter(spatial_shapes=SPATIAL, embed_dim=256)
        etype = "mask2former"
        nlev = enc.num_levels
    elif image_encoder == "dinov3":
        enc = _DummyDino()
        etype = "dinov3"
        nlev = 1
    else:
        enc = None
        etype = "dinov3"
        nlev = 1
    return MixerModel(image_encoder=enc, image_encoder_type=etype, n_image_levels=nlev,
                      image_adaptation=image_adaptation,
                      align_head_hidden_dim=[32], **kw)


class _DinoInner(torch.nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size


class _DummyDino(torch.nn.Module):
    """DINOv3-shaped stub (has .embed_dim and .model.patch_size) for the wrong-encoder guard."""

    def __init__(self, embed_dim=384, patch_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.model = _DinoInner(patch_size)

    def forward(self, x):
        return x


def set_synth_centroids(model, U=4, C_max=3, seed=0):
    torch.manual_seed(seed)
    centroids = torch.rand(U, C_max, 2)
    mask = torch.ones(U, C_max, dtype=torch.bool)
    model.set_alignment_centroids(centroids, mask)
    return centroids, mask


def ia_batch(B=2, T=5, N=3, image_idx=None):
    torch.manual_seed(7)
    src = torch.rand(B, T, 3)
    tgt = torch.rand(B, N, 3)
    image_src = torch.rand(B, S_TOKENS, 256)
    if image_idx is None:
        image_idx = torch.arange(B, dtype=torch.long)
    return dict(src=src, tgt=tgt, image_src=image_src, src_mask=None, tgt_mask=None,
                image_idx=image_idx)


# ===========================================================================
# Group 1 — Centroid precompute & clustering
# ===========================================================================
def test_1_1_dva_to_px_bandwidth(monkeypatch):
    captured = {}

    class FakeMeanShift:
        def __init__(self, bandwidth=None, bin_seeding=False):
            captured["bandwidth"] = bandwidth

        def fit(self, cloud):
            self._c = np.asarray(cloud, np.float32)
            return self

        @property
        def cluster_centers_(self):
            return self._c.mean(axis=0, keepdims=True)

    monkeypatch.setattr(sc, "MeanShift", FakeMeanShift)
    s, _, _ = two_cluster_samples()
    data = StubData([s])
    ScanpathCentroidCache.build(data, bandwidth_dva=1.0)
    assert captured["bandwidth"] == pytest.approx(16.0)
    ScanpathCentroidCache.build(data, bandwidth_dva=2.0)
    assert captured["bandwidth"] == pytest.approx(32.0)


def test_1_2_clustering_collapses_duplicates():
    s, c1, c2 = two_cluster_samples()
    data = StubData([s])
    centroids, mask, order, attrs = ScanpathCentroidCache.build(data, bandwidth_dva=1.0)
    n = int(mask[0].sum())
    assert n == 2
    max_value = np.array([512.0, 320.0])
    px = centroids[0, :2] * max_value                       # back to px
    truth = np.stack([c1, c2])
    # match each recovered centroid to nearest true mean, within a few px
    for p in px:
        assert min(np.linalg.norm(p - t) for t in truth) < 3.0


def test_1_3_normalization_to_unit():
    s = dict(img_path="a.jpg", X=[256.0] * 10, Y=[160.0] * 10, split="train")
    data = StubData([s])
    centroids, mask, order, attrs = ScanpathCentroidCache.build(data, bandwidth_dva=1.0)
    assert int(mask[0].sum()) == 1
    assert centroids[0, 0] == pytest.approx(np.array([0.5, 0.5]), abs=1e-6)


def test_1_4_aggregation_across_scanpaths(monkeypatch):
    sizes = {}

    class FakeMeanShift:
        def __init__(self, bandwidth=None, bin_seeding=False):
            pass

        def fit(self, cloud):
            sizes["n"] = len(cloud)
            self._c = np.asarray(cloud, np.float32)
            return self

        @property
        def cluster_centers_(self):
            return self._c.mean(axis=0, keepdims=True)

    monkeypatch.setattr(sc, "MeanShift", FakeMeanShift)
    samples = [dict(img_path="a.jpg", X=[10.0, 20.0], Y=[10.0, 20.0], split="train"),
               dict(img_path="a.jpg", X=[30.0], Y=[30.0], split="train"),
               dict(img_path="a.jpg", X=[40.0, 50.0, 60.0], Y=[40.0, 50.0, 60.0], split="train")]
    data = StubData(samples)
    ScanpathCentroidCache.build(data, bandwidth_dva=1.0)
    assert sizes["n"] == 2 + 1 + 3


def test_1_5_empty_image():
    samples = [dict(img_path="a.jpg", X=[10.0], Y=[10.0], split="valid"),
               dict(img_path="b.jpg", X=[20.0], Y=[20.0], split="train")]
    data = StubData(samples)
    centroids, mask, order, attrs = ScanpathCentroidCache.build(data, split_restrict="train")
    # 'a.jpg' is first-seen row 0 but has no train scanpath -> 0 centroids, no crash.
    a = order.index("a.jpg")
    assert int(mask[a].sum()) == 0
    assert not mask[a].any()


def test_1_6_padding_and_cmax():
    s_two, _, _ = two_cluster_samples(img_path="a.jpg")
    s_one = dict(img_path="b.jpg", X=[100.0] * 5, Y=[100.0] * 5, split="train")
    data = StubData([s_two, s_one])
    centroids, mask, order, attrs = ScanpathCentroidCache.build(data, bandwidth_dva=1.0)
    C_max = centroids.shape[1]
    assert C_max == attrs["C_max"] == max(int(mask[u].sum()) for u in range(len(order)))
    a = order.index("a.jpg")
    # padded slots are NaN in the built array, mask False
    assert np.isnan(centroids[a, int(mask[a].sum()):]).all()
    for u in range(len(order)):
        assert int(mask[u].sum()) == int((~np.isnan(centroids[u, :, 0])).sum())


def test_1_7_dbscan_alternative():
    s, c1, c2 = two_cluster_samples()
    data = StubData([s])
    centroids, mask, order, attrs = ScanpathCentroidCache.build(
        data, bandwidth_dva=1.0, algorithm="dbscan")
    assert int(mask[0].sum()) == 2
    assert attrs["algorithm"] == "dbscan"


def test_1_8_split_restrict():
    samples = [dict(img_path="a.jpg", X=[10.0], Y=[10.0], split="valid"),
               dict(img_path="a.jpg", X=[12.0], Y=[12.0], split="valid"),
               dict(img_path="b.jpg", X=[20.0], Y=[20.0], split="train")]
    data = StubData(samples)
    centroids, mask, order, attrs = ScanpathCentroidCache.build(data, split_restrict="train")
    a, b = order.index("a.jpg"), order.index("b.jpg")
    assert int(mask[a].sum()) == 0            # val-only image
    assert int(mask[b].sum()) >= 1


# ===========================================================================
# Group 2 — Cache write / read / order invariant
# ===========================================================================
def _build_and_write(tmp_path, data, **kw):
    centroids, mask, order, attrs = ScanpathCentroidCache.build(data, **kw)
    path = os.path.join(str(tmp_path), "centroids.h5")
    ScanpathCentroidCache.write(path, centroids, mask, order, attrs)
    return path, centroids, mask, order, attrs


def test_2_1_roundtrip(tmp_path):
    s_two, _, _ = two_cluster_samples(img_path="a.jpg")
    s_one = dict(img_path="b.jpg", X=[100.0] * 5, Y=[100.0] * 5, split="train")
    data = StubData([s_two, s_one])
    path, centroids, mask, order, attrs = _build_and_write(tmp_path, data, bandwidth_dva=1.0)
    cache = ScanpathCentroidCache(path, data)
    exp = np.where(mask[..., None], centroids, 0.0)
    assert torch.allclose(cache.centroids, torch.from_numpy(exp.astype(np.float32)))
    assert torch.equal(cache.centroid_mask, torch.from_numpy(mask))
    for k in ("bandwidth_dva", "ptoa", "dest_res", "max_value", "algorithm", "C_max"):
        assert k in cache.attrs


def test_2_2_order_verification_passes(tmp_path):
    samples = [dict(img_path="a.jpg", X=[10.0], Y=[10.0], split="train"),
               dict(img_path="b.jpg", X=[20.0], Y=[20.0], split="train")]
    data = StubData(samples)
    path, *_ = _build_and_write(tmp_path, data)
    cache = ScanpathCentroidCache(path, data)
    order = ScanpathCentroidCache._first_seen_unique(data)
    for u, p in enumerate(order):
        assert cache.image_path[u] == p


def test_2_3_order_verification_not_bypassable(tmp_path):
    import h5py
    samples = [dict(img_path="a.jpg", X=[10.0], Y=[10.0], split="train"),
               dict(img_path="b.jpg", X=[20.0], Y=[20.0], split="train")]
    data = StubData(samples)
    path, *_ = _build_and_write(tmp_path, data)
    # permute two rows of image_path in the file
    with h5py.File(path, "r+") as f:
        g = f[ScanpathCentroidCache.GROUP]
        paths = [p.decode() if isinstance(p, bytes) else p for p in g["image_path"][:]]
        paths[0], paths[1] = paths[1], paths[0]
        del g["image_path"]
        dt = h5py.string_dtype("utf-8")
        g.create_dataset("image_path", data=np.array(paths, dtype=object), dtype=dt)
    with pytest.raises(ValueError):
        ScanpathCentroidCache(path, data)
    # reordering the stub so first-seen order differs also raises
    reordered = StubData([samples[1], samples[0]])
    path2, *_ = _build_and_write(tmp_path, StubData(samples))
    with pytest.raises(ValueError):
        ScanpathCentroidCache(path2, reordered)


def test_2_4_masked_nan_handling(tmp_path):
    s_two, _, _ = two_cluster_samples(img_path="a.jpg")
    s_one = dict(img_path="b.jpg", X=[100.0] * 5, Y=[100.0] * 5, split="train")
    data = StubData([s_two, s_one])
    path, *_ = _build_and_write(tmp_path, data, bandwidth_dva=1.0)
    cache = ScanpathCentroidCache(path, data)
    assert torch.isfinite(cache.centroids).all()
    # a masked-out slot reads as 0
    b = ScanpathCentroidCache._first_seen_unique(data).index("b.jpg")
    if int(cache.centroid_mask[b].sum()) < cache.centroids.shape[1]:
        pad = cache.centroids[b, int(cache.centroid_mask[b].sum()):]
        assert torch.all(pad == 0)


# ===========================================================================
# Group 3 — target geometry & AlignmentLoss
# ===========================================================================
def test_3_1_offset_sign_value():
    tc = torch.tensor([[[0.5, 0.5]]])
    cents = torch.tensor([[[0.8, 0.2]]])
    mask = torch.tensor([[True]])
    off = nearest_centroid_offsets(tc, cents, mask)
    assert torch.allclose(off, torch.tensor([[[0.3, -0.3]]]), atol=1e-6)


def test_3_2_nearest_selection():
    tc = torch.tensor([[[0.5, 0.5]]])
    cents = torch.tensor([[[0.9, 0.9], [0.55, 0.45]]])
    mask = torch.tensor([[True, True]])
    off = nearest_centroid_offsets(tc, cents, mask)
    assert torch.allclose(off, torch.tensor([[[0.05, -0.05]]]), atol=1e-6)


def test_3_3_mask_excludes_padded():
    tc = torch.tensor([[[0.5, 0.5]]])
    cents = torch.tensor([[[0.9, 0.9], [0.55, 0.45]]])
    mask = torch.tensor([[True, False]])
    off = nearest_centroid_offsets(tc, cents, mask)
    assert torch.allclose(off, torch.tensor([[[0.4, 0.4]]]), atol=1e-6)


def test_3_4_zero_loss_perfect():
    tc = torch.tensor([[[0.5, 0.5]]])
    cents = torch.tensor([[[0.8, 0.2]]])
    mask = torch.tensor([[True]])
    target = nearest_centroid_offsets(tc, cents, mask)
    loss_fn = AlignmentLoss()
    loss, info = loss_fn({}, {"align": target.clone(), "token_centers": tc,
                             "image_centroids": cents, "centroid_mask": mask})
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert info["align_loss"] == pytest.approx(0.0, abs=1e-6)


def test_3_5_row_with_no_centroids_dropped():
    tc = torch.tensor([[0.5, 0.5]]).view(1, 1, 2)
    pred = torch.tensor([[[0.1, 0.1]], [[0.2, -0.1]]])          # (2,1,2)
    cents = torch.tensor([[[0.0, 0.0]], [[0.8, 0.2]]])          # (2,1,2)
    mask = torch.tensor([[False], [True]])
    loss_fn = AlignmentLoss()
    loss_both, _ = loss_fn({}, {"align": pred, "token_centers": tc,
                                "image_centroids": cents, "centroid_mask": mask})
    # loss on row 1 alone
    loss_one, _ = loss_fn({}, {"align": pred[1:2], "token_centers": tc,
                               "image_centroids": cents[1:2], "centroid_mask": mask[1:2]})
    assert loss_both.item() == pytest.approx(loss_one.item(), abs=1e-6)
    assert not torch.isnan(loss_both)


def test_3_6_all_empty_safe_zero():
    tc = torch.tensor([[[0.5, 0.5]]])
    pred = torch.zeros(2, 1, 2, requires_grad=True)
    cents = torch.zeros(2, 1, 2)
    mask = torch.zeros(2, 1, dtype=torch.bool)
    loss_fn = AlignmentLoss()
    loss, info = loss_fn({}, {"align": pred, "token_centers": tc,
                             "image_centroids": cents, "centroid_mask": mask})
    assert loss.item() == 0.0
    loss.backward()                                            # no exception


def test_3_7_target_no_grad():
    tc = torch.tensor([[[0.5, 0.5]]])
    cents = torch.tensor([[[0.8, 0.2]]], requires_grad=True)
    mask = torch.tensor([[True]])
    off = nearest_centroid_offsets(tc, cents, mask)
    assert off.requires_grad is False


def test_3_8_does_not_read_tgt():
    tc = torch.tensor([[[0.5, 0.5]]])
    pred = torch.tensor([[[0.1, 0.1]]])
    cents = torch.tensor([[[0.8, 0.2]]])
    mask = torch.tensor([[True]])
    loss_fn = AlignmentLoss()
    base = {"align": pred, "token_centers": tc, "image_centroids": cents, "centroid_mask": mask}
    l1, _ = loss_fn({"tgt": torch.rand(1, 3, 3)}, base)
    l2, _ = loss_fn({"tgt": torch.full((1, 3, 3), 999.0)}, base)
    assert l1.item() == pytest.approx(l2.item(), abs=1e-8)


def test_3_9_coord_func_swap():
    tc = torch.tensor([[[0.5, 0.5]]])
    pred = torch.tensor([[[0.1, 0.9]]])
    cents = torch.tensor([[[0.8, 0.2]]])
    mask = torch.tensor([[True]])
    out = {"align": pred, "token_centers": tc, "image_centroids": cents, "centroid_mask": mask}
    l1, _ = AlignmentLoss(coord_func=F.l1_loss)({}, out)
    l2, _ = AlignmentLoss(coord_func=F.mse_loss)({}, out)
    assert abs(l1.item() - l2.item()) > 1e-4


# ===========================================================================
# Group 4 — CombinedLossFunction dispatch & eval_align
# ===========================================================================
def _combined(align=None):
    fix = SeparatedRegLossFunction(cls_func=EndBinaryCrossEntropy(),
                                   coord_func=F.mse_loss, dur_func=F.mse_loss)
    den = DenoiseRegLoss(F.l1_loss)
    return CombinedLossFunction(denoise_loss=den, fixation_loss=fix, align_loss=align)


def test_4_1_align_branch_early_return():
    align = AlignmentLoss()
    comb = _combined(align)
    tc = torch.tensor([[[0.5, 0.5]]])
    out = {"align": torch.tensor([[[0.1, 0.1]]]), "token_centers": tc,
           "image_centroids": torch.tensor([[[0.8, 0.2]]]), "centroid_mask": torch.tensor([[True]])}
    loss, info = comb({}, out)
    exp_loss, exp_info = align({}, out)
    assert loss.item() == pytest.approx(exp_loss.item())
    assert set(info.keys()) == {"align_loss"}


def test_4_2_no_align_byte_identity():
    # a normal Combined output (no "align"); result independent of align_loss being set.
    B, L = 2, 4
    out = {"coord": torch.rand(B, L, 2), "dur": torch.rand(B, L, 1), "cls": torch.rand(B, L, 1)}
    inp = {"tgt": torch.rand(B, L - 1, 3), "tgt_mask": torch.ones(B, L, dtype=torch.bool),
           "fixation_len": torch.tensor([L - 1, L - 1])}
    comb_none = _combined(None)
    comb_set = _combined(AlignmentLoss())
    comb_none.set_denoise_weight(0)
    comb_set.set_denoise_weight(0)
    l0, i0 = comb_none(inp, out)
    l1, i1 = comb_set(inp, out)
    assert l0.item() == pytest.approx(l1.item())
    assert set(i0.keys()) == set(i1.keys())


def test_4_3_eval_align_matches_geometry():
    tc = torch.tensor([[[0.5, 0.5]]])
    cents = torch.tensor([[[0.8, 0.2]]])
    mask = torch.tensor([[True]])
    pred = torch.tensor([[[0.0, 0.0]]])
    # target = (0.3, -0.3); ||pred - target|| = sqrt(0.18)
    val = eval_align(pred, tc, cents, mask)
    assert val == pytest.approx(float(np.sqrt(0.18)), abs=1e-6)
    assert eval_align(nearest_centroid_offsets(tc, cents, mask), tc, cents, mask) == pytest.approx(0.0, abs=1e-6)
    assert eval_align(pred, tc, cents, torch.tensor([[False]])) == 0.0


def test_4_4_eval_align_pixel_scale():
    tc = torch.tensor([[[0.5, 0.6]]])
    cents = torch.tensor([[[0.8, 0.2]]])
    mask = torch.tensor([[True]])
    pred = torch.tensor([[[0.0, 0.0]]])
    # target = (0.3, -0.4); px offset = (0.3*512, -0.4*320) = (153.6, -128.0)
    val = eval_align(pred, tc, cents, mask, pixel_scale=[512.0, 320.0])
    assert val == pytest.approx(float(np.hypot(153.6, 128.0)), abs=1e-3)
    # scale is anisotropic: swapping W/H changes the value (asymmetric offset)
    val_swapped = eval_align(pred, tc, cents, mask, pixel_scale=[320.0, 512.0])
    assert abs(val - val_swapped) > 1.0


def test_4_5_set_alignment_pixel_scale():
    m = make_model(image_adaptation=True)
    cents = torch.rand(3, 2, 2)
    mask = torch.ones(3, 2, dtype=torch.bool)
    m.set_alignment_centroids(cents, mask, pixel_scale=[512.0, 320.0])
    assert torch.equal(m.align_pixel_scale, torch.tensor([512.0, 320.0]))
    # default (no scale) leaves it None
    m2 = make_model(image_adaptation=True)
    m2.set_alignment_centroids(cents, mask)
    assert m2.align_pixel_scale is None


# ===========================================================================
# Group 5 — MixerModel construction & gating
# ===========================================================================
def test_5_1_off_by_default():
    m = make_model(image_adaptation=False)
    assert any(mod is m.img_input_proj for mod in m.denoise_modules)
    assert m.adapter_modules == []
    assert not hasattr(m, "align_head")
    assert not hasattr(m, "align_centroids")
    assert not any("align_head" in n for n, _ in m.named_parameters())


def test_5_2_on_path_build():
    m = make_model(image_adaptation=True)
    assert not any(mod is m.img_input_proj for mod in m.denoise_modules)
    assert any(mod is m.img_input_proj for mod in m.adapter_modules)
    assert any(mod is m.align_head for mod in m.adapter_modules)
    out = m.align_head(torch.rand(1, 3, m.model_dim))
    assert out.shape[-1] == 2
    assert m.align_centroids.numel() == 0
    assert m.align_centroid_mask.numel() == 0
    assert m._centroids_ready is False


def test_5_3_guard_wrong_encoder():
    with pytest.raises(ValueError, match="mask2former"):
        make_model(image_adaptation=True, image_encoder="dinov3")


def test_5_4_guard_no_encoder():
    with pytest.raises(ValueError):
        make_model(image_adaptation=True, image_encoder=None)


def test_5_5_state_dict():
    off = make_model(image_adaptation=False)
    on = make_model(image_adaptation=True)
    off_keys = set(off.state_dict().keys())
    on_keys = set(on.state_dict().keys())
    added = on_keys - off_keys
    assert all(k.startswith("align_head.") for k in added)
    assert added                                              # something was added
    assert "align_centroids" not in on_keys
    assert "align_centroid_mask" not in on_keys


def test_5_6_set_alignment_centroids():
    m = make_model(image_adaptation=True)
    cents, mask = set_synth_centroids(m)
    assert torch.equal(m.align_centroids, cents.to(m.align_centroids.dtype))
    assert m.align_centroid_mask.dtype == torch.bool
    assert m._centroids_ready is True


# ===========================================================================
# Group 6 — set_phase requires_grad matrix (FR13)
# ===========================================================================
def _probe(m):
    d = next(m.path_encoder[0].parameters()).requires_grad
    f = next(m.coord_head.parameters()).requires_grad
    a = next(m.align_head.parameters()).requires_grad
    return d, f, a


@pytest.mark.parametrize("phase,exp", [
    ("Denoise", (True, False, False)),
    ("Fixation", (False, True, False)),
    ("Combined", (True, True, False)),
    ("ImageAdaptation", (False, False, True)),
    ("FullFinetune", (True, True, True)),
])
def test_6_phase_matrix(phase, exp):
    m = make_model(image_adaptation=True)
    m.set_phase(phase)
    assert _probe(m) == exp
    if phase in ("ImageAdaptation", "FullFinetune"):
        assert next(m.img_input_proj.parameters()).requires_grad is True


def test_6_6_off_path_parity():
    on = make_model(image_adaptation=True)
    off = make_model(image_adaptation=False)
    for phase in ("Denoise", "Fixation", "Combined"):
        off.set_phase(phase)
        d = next(off.path_encoder[0].parameters()).requires_grad
        f = next(off.coord_head.parameters()).requires_grad
        # img_input_proj (in denoise_modules off-path) follows the D column
        ip = next(off.img_input_proj.parameters()).requires_grad
        expected = {"Denoise": (True, False, True), "Fixation": (False, True, False),
                    "Combined": (True, True, True)}[phase]
        assert (d, f, ip) == expected


# ===========================================================================
# Group 7 — forward routing, shapes, gradient isolation
# ===========================================================================
def test_7_1_image_adaptation_forward():
    m = make_model(image_adaptation=True)
    set_synth_centroids(m, U=4, C_max=3)
    m.set_phase("ImageAdaptation")
    m.eval()
    batch = ia_batch(B=2)
    out = m(**batch)
    assert set(out.keys()) == {"align", "token_centers", "image_centroids", "centroid_mask"}
    assert out["align"].shape == (2, S_TOKENS, 2)
    assert out["token_centers"].shape == (1, S_TOKENS, 2)
    assert out["image_centroids"].shape == (2, 3, 2)
    assert out["centroid_mask"].shape == (2, 3)


def test_7_2_decode_align_errors():
    m = make_model(image_adaptation=True)
    # centroids not set yet
    m.image_adapter_features = torch.rand(1, S_TOKENS, m.model_dim)
    m.image_reference_grids = torch.rand(S_TOKENS, 2)
    with pytest.raises(RuntimeError):
        m.decode_align(image_idx=torch.zeros(1, dtype=torch.long))
    # no image path
    set_synth_centroids(m)
    m.image_adapter_features = None
    with pytest.raises(RuntimeError):
        m.decode_align(image_idx=torch.zeros(1, dtype=torch.long))


def test_7_3_full_finetune_forward():
    m = make_model(image_adaptation=True)
    set_synth_centroids(m)
    m.set_phase("FullFinetune")
    m.eval()
    out = m(**ia_batch(B=2))
    assert "align" not in out
    assert {"coord", "dur", "cls"} <= set(out.keys())


def test_7_4_combined_has_no_align():
    m = make_model(image_adaptation=True)
    set_synth_centroids(m)
    m.set_phase("Combined")
    m.eval()
    out = m(**ia_batch(B=2))
    assert "align" not in out


def test_7_5_encode_stores_trunk_and_grids():
    m = make_model(image_adaptation=True)
    m.eval()
    batch = ia_batch(B=2)
    m.encode(batch["src"], batch["image_src"], batch["src_mask"])
    assert m.image_adapter_features.shape == (2, S_TOKENS, m.model_dim)
    assert m.image_reference_grids.shape == (S_TOKENS, 2)
    assert m.image_reference_grids.min() >= 0.0 and m.image_reference_grids.max() <= 1.0


def test_7_6_centroid_gather_by_image_idx():
    m = make_model(image_adaptation=True)
    U, C = 5, 3
    torch.manual_seed(1)
    cents = torch.rand(U, C, 2)
    mask = torch.ones(U, C, dtype=torch.bool)
    m.set_alignment_centroids(cents, mask)
    m.set_phase("ImageAdaptation")
    m.eval()
    image_idx = torch.tensor([3, 0], dtype=torch.long)
    out = m(**ia_batch(B=2, image_idx=image_idx))
    assert torch.equal(out["image_centroids"], cents[image_idx])


def test_7_7_gradient_isolation():
    m = make_model(image_adaptation=True)
    set_synth_centroids(m)
    m.set_phase("ImageAdaptation")
    m.train()
    align = AlignmentLoss()
    out = m(**ia_batch(B=2))
    loss, _ = align({}, out)
    loss.backward()
    assert next(m.align_head.parameters()).grad is not None
    assert next(m.img_input_proj.parameters()).grad is not None
    assert next(m.path_encoder[0].parameters()).grad is None
    assert next(m.coord_head.parameters()).grad is None


# ===========================================================================
# Group 8 — end-to-end smoke (model-level, tiny synthetic)
# ===========================================================================
class _FakePathDataset:
    transforms = []            # invert_transforms fetches this; unused on the align path


class _FakeLoader:
    """Minimal val_dataloader for validate(): iterable + .path_dataset.transforms."""

    def __init__(self, batches):
        self._batches = batches
        self.path_dataset = _FakePathDataset()

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def test_8_1_three_phase_metrics():
    from src.training.training_utils import MetricsStorage, validate
    m = make_model(image_adaptation=True)
    set_synth_centroids(m, U=4, C_max=3)
    align = AlignmentLoss()
    metrics = MetricsStorage(decisive_metric="align_error_val").metrics

    # Phase 1 — ImageAdaptation: align_error_val populated.
    m.set_phase("ImageAdaptation")
    loader = _FakeLoader([ia_batch(B=2), ia_batch(B=2)])
    validate(m, align, loader, epoch=0, device="cpu", metrics=metrics, log=False)
    assert len(metrics["align_error_val"]) == 1
    assert len(metrics["reg_error_val"]) == 0

    # Phases 2-3 — Combined/FullFinetune produce reg_error_val.
    fix = SeparatedRegLossFunction(cls_func=EndBinaryCrossEntropy(),
                                   coord_func=F.mse_loss, dur_func=F.mse_loss)
    comb = CombinedLossFunction(denoise_loss=DenoiseRegLoss(F.l1_loss), fixation_loss=fix,
                                align_loss=align)
    comb.set_denoise_weight(0)

    def reg_batch(B=2, N=3):
        b = ia_batch(B=B, N=N)
        b["tgt_mask"] = torch.ones(B, N + 1, dtype=torch.bool)
        b["fixation_len"] = torch.tensor([N] * B)
        return b

    m.set_phase("Combined")
    loader2 = _FakeLoader([reg_batch(), reg_batch()])
    validate(m, comb, loader2, epoch=1, device="cpu", metrics=metrics, log=False)
    assert len(metrics["reg_error_val"]) == 1


def test_8_4_validate_reports_pixel_align_error():
    from src.training.training_utils import MetricsStorage, validate
    m = make_model(image_adaptation=True)
    cents = torch.rand(4, 3, 2)
    mask = torch.ones(4, 3, dtype=torch.bool)
    m.set_alignment_centroids(cents, mask, pixel_scale=[512.0, 320.0])
    m.set_phase("ImageAdaptation")
    metrics = MetricsStorage(decisive_metric="align_error_val").metrics
    loader = _FakeLoader([ia_batch(B=2), ia_batch(B=2)])
    validate(m, AlignmentLoss(), loader, epoch=0, device="cpu", metrics=metrics, log=False)
    assert len(metrics["align_error_px_val"]) == 1
    # pixel error is larger than the normalized error (axes scaled by 512 / 320)
    assert metrics["align_error_px_val"][-1] > metrics["align_error_val"][-1]


def test_8_2_align_loss_decreases():
    m = make_model(image_adaptation=True)
    set_synth_centroids(m, U=2, C_max=2)
    m.set_phase("ImageAdaptation")
    m.train()
    align = AlignmentLoss()
    batch = ia_batch(B=2)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    first = None
    last = None
    for step in range(50):
        opt.zero_grad()
        out = m(**batch)
        loss, _ = align({}, out)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < 0.9 * first


def test_8_3_checkpoint_roundtrip():
    m = make_model(image_adaptation=True)
    set_synth_centroids(m)
    sd = m.state_dict()
    assert "align_centroids" not in sd                        # non-persistent

    fresh_on = make_model(image_adaptation=True)
    missing, unexpected = fresh_on.load_state_dict(sd, strict=False)
    assert missing == [] and unexpected == []
    for (n, p), (_, q) in zip(m.align_head.named_parameters(),
                              fresh_on.align_head.named_parameters()):
        assert torch.equal(p, q)

    off = make_model(image_adaptation=False)
    missing, unexpected = off.load_state_dict(sd, strict=False)
    assert any(k.startswith("align_head.") for k in unexpected)


# ===========================================================================
# Group 9 — scheduled sampling on the three-phase run
# ===========================================================================
def _attach_sampler(m):
    from src.training.training_utils import ScheduledSampling
    s = ScheduledSampling(active_epochs=1, warmup_epochs=0, device="cpu", steps_per_epoch=1)
    m.set_scheduled_sampling(s)
    return s


def test_9_1_image_adaptation_bypasses_sampler_in_eval():
    # eval ⇒ get_current_ratio() == 1; ImageAdaptation must still return align outputs.
    m = make_model(image_adaptation=True)
    set_synth_centroids(m, U=4, C_max=3)
    _attach_sampler(m)
    m.set_phase("ImageAdaptation")
    m.eval()
    out = m(**ia_batch(B=2))
    assert set(out.keys()) == {"align", "token_centers", "image_centroids", "centroid_mask"}
    assert out["align"].shape == (2, S_TOKENS, 2)


def test_9_2_full_finetune_uses_sampler_in_eval():
    m = make_model(image_adaptation=True)
    set_synth_centroids(m)
    _attach_sampler(m)
    m.set_phase("FullFinetune")
    m.eval()
    batch = ia_batch(B=2, N=3)
    batch["tgt_mask"] = torch.ones(2, 4, dtype=torch.bool)   # K1 = N + 1 decode steps
    out = m(**batch)
    assert "align" not in out
    assert out["coord"].shape[:2] == (2, 4)


def test_9_3_exp_config_schedule_invariants():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join("configs", "exp", "image_adaptation_training.yaml"))
    t, ss = cfg.training, cfg.scheduled_sampling
    assert t.use_scheduled_sampling is True
    total = sum(t[p].epochs for p in t.Phases)
    assert ss.warmup_epochs >= t.ImageAdaptation.epochs      # sampler counts global epochs
    assert cfg.model.image_adaptation.pretrained_adapter_path is None
    assert ss.warmup_epochs + ss.active_epochs <= total


# ===========================================================================
# Group 10 — reusing a finished ImageAdaptation phase
# ===========================================================================
def _adapter_items(m):
    return {k: v for k, v in m.state_dict().items() if k.startswith(MixerModel.ADAPTER_PREFIXES)}


def test_10_1_load_image_adapter_copies_only_adapter(tmp_path):
    src = make_model(image_adaptation=True)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1.0)                                      # make src differ from any fresh init
    ckpt = tmp_path / "model.pth"
    torch.save({"model_state_dict": {"_orig_mod." + k: v for k, v in src.state_dict().items()}},
               ckpt)

    dst = make_model(image_adaptation=True)
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    dst.load_image_adapter(str(ckpt))

    src_ad = _adapter_items(src)
    assert src_ad and any(k.startswith("align_head.") for k in src_ad)
    for k, v in _adapter_items(dst).items():
        assert torch.equal(v, src_ad[k]), k
    for k, v in dst.state_dict().items():                    # everything else untouched
        if not k.startswith(MixerModel.ADAPTER_PREFIXES):
            assert torch.equal(v, before[k]), k


def test_10_2_load_image_adapter_rejects_checkpoint_without_adapter(tmp_path):
    off = make_model(image_adaptation=True)
    sd = {k: v for k, v in off.state_dict().items()
          if not k.startswith(MixerModel.ADAPTER_PREFIXES)}
    ckpt = tmp_path / "model.pth"
    torch.save({"model_state_dict": sd}, ckpt)
    with pytest.raises(ValueError):
        make_model(image_adaptation=True).load_image_adapter(str(ckpt))


def test_10_3_load_image_adapter_requires_adaptation_on(tmp_path):
    with pytest.raises(RuntimeError):
        make_model(image_adaptation=False).load_image_adapter(str(tmp_path / "x.pth"))
