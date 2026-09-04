"""One-shot CLI: precompute frozen Mask2Former image features -> image_features_{img_size}.h5.

Builds the vendored ``Mask2FormerBackbone`` (ResNet50 + 6-layer MSDeformAttn pixel decoder,
everything frozen, ``return_stride4=True``), loads the two COCO-tuned detectron2 checkpoints
(``M2F_R50.pkl`` + ``M2F_R50_MSDeformAttnPixelDecoder.pkl``), runs it once over every **unique**
stimulus image on the CocoFreeView data root (same filter as training, first-seen order), and
writes a keyed HDF5 cache with the 3-level deformable memory ``ms_value`` and the stride-4
``mask_features``.

Usage:
    python scripts/build_image_feature_cache.py --img-size 256 --batch-size 16
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.datasets import FreeViewInMemory
from src.data.image_feature_cache import ImageFeatureCache, PrecomputedFeatureDataset
from src.data.parsers import CocoFreeView
from src.model.m2f_pretrained import load_pretrained_mask2former
from src.model.ms_deform_backbone import Mask2FormerBackbone
from src.training.pipeline_builder import PipelineBuilder


def _parse_args():
    p = argparse.ArgumentParser(description="Precompute frozen Mask2Former image features.")
    p.add_argument("--data-root", default=os.path.join("data", "Coco FreeView"))
    p.add_argument("--r50", default=os.path.join("pretrained_models", "M2F_R50.pkl"))
    p.add_argument("--pixel-decoder",
                   default=os.path.join("pretrained_models",
                                        "M2F_R50_MSDeformAttnPixelDecoder.pkl"))
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--out", default=None,
                   help="Output path (default data-root/image_features_{img_size}.h5)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--downsample-int", type=int, default=None)
    p.add_argument("--no-mask-features", action="store_true",
                   help="Skip the stride-4 mask_features (the model never reads them); "
                        "produces an ms_value-only cache ~4x smaller.")
    return p.parse_args()


def main():
    args = _parse_args()
    out = args.out or os.path.join(args.data_root, f"image_features_{args.img_size}.h5")
    device = torch.device(args.device)

    # 1. Same filter as training: FreeViewInMemory.data_store['filtered_idx'] over CocoFreeView.
    print("Loading FreeViewInMemory to obtain filtered_idx ...")
    path_ds = FreeViewInMemory(data_path=args.data_root, transforms=[],
                               downsample_int=args.downsample_int, log=False)
    data = CocoFreeView(data_path=args.data_root)
    data.filter_by_idx(path_ds.data_store['filtered_idx'])
    print(f"Filtered CocoFreeView: {len(data)} scanpaths")

    # 2. First-seen unique-image index (identical to DeduplicatedMemoryDataset ordering).
    unique_paths, _ = PrecomputedFeatureDataset._build_index(data)
    U = len(unique_paths)
    print(f"Unique stimulus images: {U}")

    # 3. Backbone (all frozen) + pretrained load. mask_features needs the stride-4 branch.
    store_mask = not args.no_mask_features
    print(f"Building Mask2FormerBackbone (return_stride4={store_mask}, 6 enc layers, frozen) ...")
    backbone = Mask2FormerBackbone(
        conv_dim=256, n_heads=8, n_points=4, transformer_enc_layers=6,
        transformer_dim_feedforward=1024, transformer_dropout=0.0,
        transformer_in_features=("res3", "res4", "res5"), return_stride4=store_mask, mask_dim=256,
        freeze_backbone=True, freeze_pixel_decoder=True,
        imagenet_weights="IMAGENET1K_V2", device=device)
    report = load_pretrained_mask2former(backbone, args.r50, args.pixel_decoder)
    print(f"Load report: n_resnet_loaded={report.n_resnet_loaded}, "
          f"n_pixdec_loaded={report.n_pixdec_loaded}, "
          f"missing={len(report.missing_keys)}, unexpected={len(report.unexpected_keys)}")
    backbone.eval()

    # 4. Preprocessing (RGB ImageNet norm) + optional uint8 unique-image bank.
    transform = PipelineBuilder.make_transform(resize_size=args.img_size)
    bank = None
    bank_path = os.path.join(args.data_root, f"all_images_{args.img_size}.pth")
    if os.path.exists(bank_path):
        loaded = torch.load(bank_path)
        if len(loaded) == U:
            bank = loaded
            print(f"Using uint8 image bank {bank_path} (row u == unique id u).")
        else:
            print(f"Image bank size {len(loaded)} != U={U}; decoding from disk instead.")

    def load_image(u):
        if bank is not None:
            return transform(bank[u])
        return transform(Image.open(unique_paths[u]).convert("RGB"))

    # 5. Peek shapes from one forward, then stream batches straight into HDF5 (never holds the
    #    full array in RAM — the full cache can be tens of GB).
    with torch.inference_mode():
        probe = backbone(torch.stack([load_image(0)]).to(device))
        maps0 = probe[0] if store_mask else probe
        res5, res4, res3 = maps0[:3]
        S = sum(m.shape[2] * m.shape[3] for m in (res5, res4, res3))
        spatial_shapes = [[m.shape[2], m.shape[3]] for m in (res5, res4, res3)]
        embed_dim = res5.shape[1]
        mask_feature_shape = list(probe[1].shape[2:]) if store_mask else None
    level_start_index = [0]
    for h, w in spatial_shapes[:-1]:
        level_start_index.append(level_start_index[-1] + h * w)
    print(f"S={S}, spatial_shapes={spatial_shapes}, "
          f"mask_feature_shape={mask_feature_shape}, store_mask={store_mask}")

    attrs = {
        "img_size": args.img_size,
        "S": S,
        "spatial_shapes": np.array(spatial_shapes, dtype=np.int64).flatten(),
        "level_start_index": np.array(level_start_index, dtype=np.int64),
        "embed_dim": embed_dim,
        "num_levels": len(spatial_shapes),
        "mask_dim": 256,
        "mask_feature_shape": np.array(mask_feature_shape if store_mask else [0, 0], dtype=np.int64),
        "normalization": "imagenet_rgb",
        "r50_checkpoint": os.path.abspath(args.r50),
        "pixel_decoder_checkpoint": os.path.abspath(args.pixel_decoder),
        "imagenet_weights": "IMAGENET1K_V2",
        "transformer_enc_layers": 6,
        "num_unique": U,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    f = ImageFeatureCache.create_writer(
        out, U, S, embed_dim, unique_paths, attrs,
        mask_feature_shape=mask_feature_shape, mask_dim=256)
    g = f[ImageFeatureCache.GROUP]

    # 6. Stream every unique image into the file, one batch at a time.
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, U, args.batch_size):
            end = min(start + args.batch_size, U)
            x = torch.stack([load_image(u) for u in range(start, end)]).to(device)
            out_maps = backbone(x)
            maps = out_maps[0] if store_mask else out_maps
            res5, res4, res3 = maps[:3]
            val = torch.cat([m.flatten(2).transpose(1, 2) for m in (res5, res4, res3)], dim=1)
            g["ms_value"][start:end] = val.float().cpu().numpy()
            if store_mask:
                g["mask_features"][start:end] = out_maps[1].float().cpu().numpy()
            print(f"  {end}/{U}", end="\r")
    print()
    f.close()

    # 7. Report.
    size_gb = os.path.getsize(out) / (1024 ** 3)
    print("\n== Build report ==============================================")
    print(f"Unique images (U) : {U}")
    print(f"S                 : {S}   spatial_shapes {spatial_shapes}")
    print(f"mask_features     : {'yes ' + str(mask_feature_shape) if store_mask else 'no (--no-mask-features)'}")
    print(f"Output            : {out}   ({size_gb:.2f} GiB)")
    print(f"Elapsed           : {time.time() - t0:.1f} s")
    print("==============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
