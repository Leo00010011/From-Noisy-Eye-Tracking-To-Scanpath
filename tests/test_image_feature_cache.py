"""Use Pretrained (Frozen) Image Features — validation suite.

Groups mirror ``spec/2026-09-04-use-pretrained-features/validation.md``:
  1. detectron2 → torchvision ResNet50 remap (``remap_detectron2_resnet50``)
  2. pixel-decoder remap + combined load (``remap_pixel_decoder``, ``load_pretrained_mask2former``)
  3. ``PrecomputedFeatureAdapter`` bundle equality (online-vs-precomputed identity)
  4. ``ImageFeatureCache`` HDF5 roundtrip + isolation
  5. ``PrecomputedFeatureDataset`` + the (non-bypassable) keying invariant
  6. ``PipelineBuilder`` integration + FR12 failure modes
  7. Model untouched (regression)

CPU-only. Real-backbone tests use ``imagenet_weights=None`` (no network). The two
``pretrained_models/*.pkl`` checkpoints exist in this repo, so Groups 1–2 run against them.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf
from hydra import initialize, compose

from src.data.datasets import DeduplicatedMemoryDataset, seq2seq_padded_collate_fn
from src.data.image_feature_cache import ImageFeatureCache, PrecomputedFeatureDataset
from src.model.m2f_pretrained import (LoadReport, load_pretrained_mask2former,
                                      remap_detectron2_resnet50, remap_pixel_decoder)
from src.model.mixer_model import MixerModel
from src.model.ms_deform_backbone import Mask2FormerBackbone
from src.model.ms_features import (Mask2FormerFeatureAdapter, MultiScaleFeatures,
                                   PrecomputedFeatureAdapter)

REPO_ROOT = Path(__file__).resolve().parents[1]
R50_PKL = REPO_ROOT / "pretrained_models" / "M2F_R50.pkl"
PDEC_PKL = REPO_ROOT / "pretrained_models" / "M2F_R50_MSDeformAttnPixelDecoder.pkl"
HAS_PKLS = R50_PKL.exists() and PDEC_PKL.exists()
needs_pkls = pytest.mark.skipif(not HAS_PKLS, reason="pretrained_models/*.pkl not present")


def _load_sd(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


@pytest.fixture(scope="module")
def r50_sd():
    return _load_sd(R50_PKL)


@pytest.fixture(scope="module")
def pdec_sd():
    return _load_sd(PDEC_PKL)


# ===========================================================================
# Group 1 — ResNet50 remap
# ===========================================================================
@needs_pkls
def test_g1_expected_torchvision_keys_present(r50_sd):
    out = remap_detectron2_resnet50(r50_sd)
    for k in ("conv1.weight", "bn1.weight", "bn1.running_mean",
              "layer1.0.conv1.weight", "layer1.0.bn1.running_var",
              "layer1.0.downsample.0.weight", "layer1.0.downsample.1.running_mean",
              "layer4.2.conv3.weight"):
        assert k in out, k
    for k in out:
        assert "stem." not in k and "stages." not in k
        assert ".norm." not in k and "shortcut" not in k


@needs_pkls
def test_g1_block_counts_and_downsample(r50_sd):
    out = remap_detectron2_resnet50(r50_sd)
    for layer, n in (("layer1", 3), ("layer2", 4), ("layer3", 6), ("layer4", 3)):
        blocks = {k.split(".")[1] for k in out if k.startswith(layer + ".")}
        assert blocks == {str(i) for i in range(n)}, (layer, blocks)
        # only block .0 carries downsample.*
        ds_blocks = {k.split(".")[1] for k in out
                     if k.startswith(layer + ".") and ".downsample." in k}
        assert ds_blocks == {"0"}, (layer, ds_blocks)


@needs_pkls
def test_g1_values_unchanged(r50_sd):
    out = remap_detectron2_resnet50(r50_sd)
    assert torch.equal(out["conv1.weight"], r50_sd["stem.conv1.weight"])
    assert torch.equal(out["layer4.2.conv3.weight"], r50_sd["stages.res5.2.conv3.weight"])
    assert out["conv1.weight"].dtype == r50_sd["stem.conv1.weight"].dtype


def test_g1_unrecognized_key_dropped(capsys):
    out = remap_detectron2_resnet50({"foo.bar": torch.zeros(2), "stem.conv1.weight": torch.zeros(1)})
    assert "foo.bar" not in out
    assert "conv1.weight" in out
    assert "dropped" in capsys.readouterr().out


# ===========================================================================
# Group 2 — pixel-decoder remap + combined load
# ===========================================================================
def test_g2_pixel_decoder_prefix_only():
    sd = {"input_proj.0.0.weight": torch.zeros(1), "transformer.level_embed": torch.zeros(3, 4)}
    out = remap_pixel_decoder(sd)
    assert set(out) == {"pixel_decoder." + k for k in sd}
    assert torch.equal(out["pixel_decoder.transformer.level_embed"], sd["transformer.level_embed"])


def _tiny_backbone(**over):
    kw = dict(transformer_enc_layers=6, transformer_dim_feedforward=1024,
              return_stride4=True, imagenet_weights=None, device="cpu")
    kw.update(over)
    return Mask2FormerBackbone(**kw)


@needs_pkls
def test_g2_combined_load_no_core_params_missing():
    bb = _tiny_backbone()
    report = load_pretrained_mask2former(bb, str(R50_PKL), str(PDEC_PKL))
    assert isinstance(report, LoadReport)
    assert report.n_resnet_loaded >= 265
    param_names = {n for n, _ in bb.named_parameters()}
    bad_fe = [k for k in report.missing_keys
              if k in param_names and k.startswith("feature_extractor.")]
    bad_pd = [k for k in report.missing_keys if k in param_names and (
        k.startswith("pixel_decoder.transformer.") or k.startswith("pixel_decoder.input_proj."))]
    assert bad_fe == [], bad_fe
    assert bad_pd == [], bad_pd


@needs_pkls
def test_g2_loaded_values_match_pkl(r50_sd, pdec_sd):
    bb = _tiny_backbone()
    load_pretrained_mask2former(bb, str(R50_PKL), str(PDEC_PKL))
    fe_sd = bb.feature_extractor.state_dict()
    assert torch.equal(fe_sd["layer4.2.conv3.weight"],
                       remap_detectron2_resnet50(r50_sd)["layer4.2.conv3.weight"])
    assert torch.equal(bb.pixel_decoder.transformer.level_embed,
                       pdec_sd["transformer.level_embed"])


@needs_pkls
def test_g2_broken_remap_raises(tmp_path, r50_sd):
    broken = dict(r50_sd)
    del broken["stages.res2.0.conv1.weight"]     # -> layer1.0.conv1.weight will be missing
    p = tmp_path / "broken_r50.pkl"
    torch.save(broken, p)
    bb = _tiny_backbone()
    with pytest.raises(RuntimeError, match=r"layer1\.0\.conv1\.weight|ResNet50 remap incomplete"):
        load_pretrained_mask2former(bb, str(p), str(PDEC_PKL))


@needs_pkls
def test_g2_layer_count_mismatch_reports_unexpected():
    bb = _tiny_backbone(transformer_enc_layers=3)
    report = load_pretrained_mask2former(bb, str(R50_PKL), str(PDEC_PKL))
    # layers 3-5 of the 6-layer pkl have no home on a 3-layer build -> unexpected (not raised).
    assert any(("layers.3." in k or "layers.4." in k or "layers.5." in k)
               for k in report.unexpected_keys)


# ===========================================================================
# Group 3 — PrecomputedFeatureAdapter
# ===========================================================================
def test_g3_adapter_geometry_and_zero_params():
    a = PrecomputedFeatureAdapter([[8, 8], [16, 16], [32, 32]])
    assert a.num_levels == 3 and a.embed_dim == 256
    assert a.level_start_index.tolist() == [0, 64, 320]
    assert tuple(a.reference_grids.shape) == (1344, 2)
    assert sum(p.numel() for p in a.parameters()) == 0


def test_g3_buffers_not_in_state_dict():
    a = PrecomputedFeatureAdapter([[8, 8], [16, 16], [32, 32]])
    assert a.state_dict() == {}


def test_g3_forward_returns_bundle_identity():
    a = PrecomputedFeatureAdapter([[8, 8], [16, 16], [32, 32]])
    value = torch.randn(2, 1344, 256)
    b = a.forward(value)
    assert isinstance(b, MultiScaleFeatures)
    assert b.value is value
    assert torch.equal(b.spatial_shapes, a.spatial_shapes)
    assert torch.equal(b.level_start_index, a.level_start_index)
    assert torch.equal(b.reference_grids, a.reference_grids)


def test_g3_online_vs_precomputed_identity():
    torch.manual_seed(0)
    bb = Mask2FormerBackbone(imagenet_weights=None, transformer_enc_layers=2,
                             return_stride4=True, device="cpu")
    bb.eval()
    img = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        online = Mask2FormerFeatureAdapter(bb)(img)
        stub = PrecomputedFeatureAdapter(online.spatial_shapes)(online.value)
    assert torch.equal(online.value, stub.value)
    assert torch.equal(online.spatial_shapes, stub.spatial_shapes)
    assert torch.equal(online.level_start_index, stub.level_start_index)
    assert torch.allclose(online.reference_grids, stub.reference_grids, atol=0)


def test_g3_wrong_S_raises():
    a = PrecomputedFeatureAdapter([[8, 8], [16, 16], [32, 32]])
    with pytest.raises(ValueError):
        a.forward(torch.randn(2, 1000, 256))


# ===========================================================================
# Group 4 — ImageFeatureCache roundtrip + isolation
# ===========================================================================
def _write_tiny_cache(path, U=3, S=5, D=256, H4=4, W4=4, paths=None):
    ms = np.random.RandomState(0).randn(U, S, D).astype(np.float32)
    mf = np.random.RandomState(1).randn(U, D, H4, W4).astype(np.float32)
    paths = paths or [f"img_{i}.jpg" for i in range(U)]
    attrs = {
        "img_size": 256, "S": S,
        "spatial_shapes": np.array([8, 8, 16, 16, 32, 32], dtype=np.int64),
        "level_start_index": np.array([0, 64, 320], dtype=np.int64),
        "embed_dim": D, "num_levels": 3, "mask_dim": D,
        "mask_feature_shape": np.array([H4, W4], dtype=np.int64),
        "normalization": "imagenet_rgb", "transformer_enc_layers": 6, "num_unique": U,
        "r50_checkpoint": "r50.pkl", "pixel_decoder_checkpoint": "pdec.pkl",
    }
    ImageFeatureCache.write(path, ms, mf, paths, attrs)
    return ms, mf, paths


def test_g4_roundtrip(tmp_path):
    p = str(tmp_path / "c.h5")
    ms, mf, paths = _write_tiny_cache(p)
    cache = ImageFeatureCache(p)
    assert cache.image_path == paths
    for u in range(len(paths)):
        assert torch.equal(cache.ms_value(u), torch.from_numpy(ms[u]))
        assert torch.equal(cache.mask_features(u), torch.from_numpy(mf[u]))
    assert cache.ms_value(0).dtype == torch.float32


def test_g4_group_isolation(tmp_path):
    import h5py
    p = str(tmp_path / "c.h5")
    _write_tiny_cache(p)
    with h5py.File(p, "r") as f:
        assert list(f.keys()) == ["features"]


def test_g4_attrs(tmp_path):
    p = str(tmp_path / "c.h5")
    _write_tiny_cache(p)
    a = ImageFeatureCache(p).attrs
    assert int(a["img_size"]) == 256 and int(a["S"]) == 5
    assert list(np.asarray(a["spatial_shapes"]).flatten()) == [8, 8, 16, 16, 32, 32]
    assert list(np.asarray(a["level_start_index"])) == [0, 64, 320]
    assert int(a["embed_dim"]) == 256 and int(a["num_levels"]) == 3
    assert int(a["mask_dim"]) == 256 and int(a["transformer_enc_layers"]) == 6
    assert str(a["normalization"]) == "imagenet_rgb"
    assert "r50_checkpoint" in a and "pixel_decoder_checkpoint" in a


def test_g4_chunking(tmp_path):
    import h5py
    p = str(tmp_path / "c.h5")
    _write_tiny_cache(p, S=5, D=256)
    with h5py.File(p, "r") as f:
        assert f["features"]["ms_value"].chunks == (1, 5, 256)


def test_g4_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_image_feature_cache"):
        ImageFeatureCache(str(tmp_path / "nope.h5"))


def test_g4_write_without_mask_features(tmp_path):
    import h5py
    p = str(tmp_path / "c.h5")
    ms = np.random.RandomState(0).randn(3, 5, 256).astype(np.float32)
    ImageFeatureCache.write(p, ms, None, ["a", "b", "c"], {"img_size": 256})
    with h5py.File(p, "r") as f:
        assert "mask_features" not in f["features"]
    cache = ImageFeatureCache(p)
    assert cache.has_mask_features is False
    assert torch.equal(cache.ms_value(0), torch.from_numpy(ms[0]))
    with pytest.raises(KeyError, match="no mask_features"):
        cache.mask_features(0)


def test_g4_create_writer_streaming_roundtrip(tmp_path):
    p = str(tmp_path / "c.h5")
    U, S, D, H4, W4 = 4, 5, 256, 4, 4
    ms = np.random.RandomState(0).randn(U, S, D).astype(np.float32)
    mf = np.random.RandomState(1).randn(U, D, H4, W4).astype(np.float32)
    f = ImageFeatureCache.create_writer(
        p, U, S, D, [f"i{u}.jpg" for u in range(U)], {"img_size": 256},
        mask_feature_shape=(H4, W4), mask_dim=D)
    g = f["features"]
    for start in range(0, U, 2):                      # stream in batches of 2
        end = min(start + 2, U)
        g["ms_value"][start:end] = ms[start:end]
        g["mask_features"][start:end] = mf[start:end]
    f.close()
    cache = ImageFeatureCache(p)
    assert cache.has_mask_features is True
    for u in range(U):
        assert torch.equal(cache.ms_value(u), torch.from_numpy(ms[u]))
        assert torch.equal(cache.mask_features(u), torch.from_numpy(mf[u]))


# ===========================================================================
# Group 5 — PrecomputedFeatureDataset + keying invariant
# ===========================================================================
class StubData:
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def get_img_path(self, i):
        return self.paths[i]


class StubPath:
    """Minimal FreeViewInMemory stand-in for CoupledDataloader's gaze side."""
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"x": np.zeros((3, 4), dtype=np.float32),
                "y": np.zeros((3, 2), dtype=np.float32),
                "sample_idx": int(i)}


def test_g5_build_index_matches_dedup():
    paths = ["a", "a", "b", "a", "c", "c"]
    stub = StubData(paths)
    unique_paths, indices = PrecomputedFeatureDataset._build_index(stub)
    assert unique_paths == ["a", "b", "c"]
    assert indices == [0, 0, 1, 0, 2, 2]
    # byte-identical to DeduplicatedMemoryDataset.build_index on the same data.

    class _Shim:
        pass
    shim = _Shim()
    shim.data = stub
    dedup_unique, dedup_indices = DeduplicatedMemoryDataset.build_index(shim)
    assert dedup_unique == unique_paths
    assert list(dedup_indices) == indices


def test_g5_getitem_and_len(tmp_path):
    paths = ["a", "a", "b", "a", "c", "c"]
    p = str(tmp_path / "c.h5")
    _write_tiny_cache(p, U=3, S=5, paths=["a", "b", "c"])
    ds = PrecomputedFeatureDataset(StubData(paths), p)
    assert len(ds) == len(paths)
    feat, idx, u = ds[3]
    assert feat.shape == (5, 256) and feat.dtype == torch.float32
    assert idx == 3 and u == 0                       # sample 3 -> path "a" -> unique 0


def test_g5_coupled_dataloader_consumes(tmp_path):
    from torch.utils.data import Subset
    from src.data.datasets import CoupledDataloader
    paths = ["a", "a", "b", "a", "c", "c"]
    p = str(tmp_path / "c.h5")
    _write_tiny_cache(p, U=3, S=5, paths=["a", "b", "c"])
    ds = PrecomputedFeatureDataset(StubData(paths), p)
    subset = Subset(ds, [0, 1, 2, 3])
    dl = CoupledDataloader(StubPath(len(paths)), subset, batch_size=4, shuffle=False,
                           num_workers=0, prefetch_factor=None, persistent_workers=False,
                           pin_memory=False, drop_last_batch=False)
    batch = next(iter(dl))
    assert batch["image_src"].shape == (4, 5, 256)
    assert batch["image_idx"].dtype in (torch.int64, torch.int32, torch.long)


def test_g5_invariant_enforced(tmp_path):
    paths = ["a", "a", "b", "a", "c", "c"]
    p = str(tmp_path / "c.h5")
    _write_tiny_cache(p, U=3, S=5, paths=["a", "X", "c"])   # altered image_path[1]
    with pytest.raises(ValueError, match=r"mismatch at unique 1"):
        PrecomputedFeatureDataset(StubData(paths), p)


def test_g5_preload_matches_lazy(tmp_path):
    paths = ["a", "b", "c"]
    p = str(tmp_path / "c.h5")
    _write_tiny_cache(p, U=3, S=5, paths=paths)
    lazy = PrecomputedFeatureDataset(StubData(paths), p, preload=False)
    pre = PrecomputedFeatureDataset(StubData(paths), p, preload=True)
    for i in range(len(paths)):
        assert torch.equal(lazy[i][0], pre[i][0])


def test_g5_order_check_not_bypassable():
    src = (REPO_ROOT / "src" / "data" / "image_feature_cache.py").read_text()
    # The verification loop runs unconditionally in __init__ (no guarding flag).
    assert "for u, p in enumerate(unique_paths):" in src
    assert "cache/order mismatch" in src


# ===========================================================================
# Group 6 — PipelineBuilder integration + FR12
# ===========================================================================
def _compose(overrides):
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="main",
                       overrides=overrides + ["model.device=cpu",
                                              "model.pretrained_encoder_path=null"])


def test_g6_build_installs_stub_no_backbone(monkeypatch):
    import src.training.pipeline_builder as pb
    monkeypatch.setattr(pb, "Mask2FormerBackbone",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("Mask2FormerBackbone must not be built")))
    cfg = _compose(["model/image_encoder=mask2former_precomputed",
                    "+data.load.use_precomputed_features=True"])
    model, _ = pb.PipelineBuilder(cfg).build_model()
    assert isinstance(model.image_encoder, PrecomputedFeatureAdapter)
    assert model.image_encoder_type == "mask2former"
    assert model.n_image_levels == 3


def test_g6_trainable_surface_matches_online():
    cfg = _compose(["model/image_encoder=mask2former_precomputed",
                    "+data.load.use_precomputed_features=True"])
    from src.training.pipeline_builder import PipelineBuilder
    model, _ = PipelineBuilder(cfg).build_model()
    md = model.model_dim
    assert hasattr(model, "img_input_proj")
    assert model.level_embed.shape == (3, md)
    assert model.eye_decoder[0].n_levels == 3
    assert model.decoder[0].n_levels == 3


def test_g6_end_to_end_forward():
    cfg = _compose(["model/image_encoder=mask2former_precomputed",
                    "+data.load.use_precomputed_features=True"])
    from src.training.pipeline_builder import PipelineBuilder
    model, _ = PipelineBuilder(cfg).build_model()
    model.set_phase("Fixation")
    model.eval()
    B, S, T, N = 2, 1344, 5, 4
    batch = dict(src=torch.rand(B, T, 3), tgt=torch.rand(B, N, 3),
                 image_src=torch.rand(B, S, 256), src_mask=None, tgt_mask=None)
    with torch.no_grad():
        out = model(**batch)
    assert out is not None
    assert model.image_spatial_shapes.tolist() == [[8, 8], [16, 16], [32, 32]]


def test_g6_fr12_precomputed_without_data_flag_raises():
    cfg = _compose(["model/image_encoder=mask2former_precomputed"])   # data flag unset
    from src.training.pipeline_builder import PipelineBuilder
    with pytest.raises(ValueError, match="use_precomputed_features"):
        PipelineBuilder(cfg).build_model()


# ===========================================================================
# Group 7 — Model untouched (regression)
# ===========================================================================
def test_g7_model_files_unchanged_by_feature():
    out = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--",
         "src/model/mixer_model.py", "src/model/ms_deform_backbone.py"],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert out.stdout.strip() == "", f"unexpected diff:\n{out.stdout}"


def test_g7_online_mask2former_forward_identical():
    def build(seed):
        torch.manual_seed(seed)
        bb = Mask2FormerBackbone(imagenet_weights=None, transformer_enc_layers=2, device="cpu")
        adapter = Mask2FormerFeatureAdapter(bb)
        torch.manual_seed(seed + 1000)
        return MixerModel(
            image_encoder=adapter, image_encoder_type="mask2former",
            n_image_levels=3, n_encoder=2, n_decoder=2, n_eye_decoder=2,
            n_feature_enhancer=0, model_dim=256, total_dim=256, n_heads=8, ff_dim=128,
            max_pos_enc=90, max_pos_dec=26, input_encoder="shared_gaussian", norm_first=True,
            mlp_head_hidden_dim=[64], pos_enc_hidden_dim=32, num_freq_bands=8, pos_enc_sigma=1.0,
            use_deformable_eye_decoder=True, use_deformable_fixation_decoder=True,
            pred_dur_pdf=False, phases=["Fixation", "Combined"], activation=F.gelu,
            head_type="linear", device="cpu")
    m1, m2 = build(0), build(0)
    m1.eval(); m2.eval()
    m1.set_phase("Combined"); m2.set_phase("Combined")
    torch.manual_seed(9)
    b = dict(src=torch.rand(2, 5, 3), tgt=torch.rand(2, 4, 3),
             image_src=torch.rand(2, 3, 256, 256), src_mask=None, tgt_mask=None)
    with torch.no_grad():
        o1, o2 = m1(**b), m2(**b)
    for k in o1:
        assert torch.equal(o1[k], o2[k]), k
