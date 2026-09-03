"""One-shot CLI: pre-build the deduplicated resized-image cache (``all_images_{img_size}.pth``)
that ``DeduplicatedMemoryDataset`` otherwise builds lazily on first dataset load.

Reuses ``PipelineBuilder.load_dataset()`` (same code path ``train.py`` takes) so the cache is
guaranteed consistent with whatever a real training run would build/use for the composed config
— same ``data_path``/``LOCAL_SCRATCH`` resolution, same ``filtered_idx`` exclusion, same
``img_size``. Running this ahead of a training/eval job avoids paying the resize cost (and the
RAM spike) inside that job, and lets the cache be built once and shared.

Usage:
    python scripts/build_image_cache.py
    python scripts/build_image_cache.py data.load.img_size=512
    python scripts/build_image_cache.py data=eve_real data.load.img_size=512

Any Hydra override understood by ``configs/main.yaml`` works, since this loads the same config
tree as ``train.py``. Only ``data.load.img_size`` (and ``data_path``/``LOCAL_SCRATCH``) matter for
which cache file gets built.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hydra
from omegaconf import DictConfig

from src.training.pipeline_builder import PipelineBuilder


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(config: DictConfig) -> None:
    load_config = config.data.load if hasattr(config.data, "load") else config.data
    img_size = load_config.img_size
    use_img_dataset = getattr(load_config, "use_img_dataset", False)

    if not use_img_dataset:
        print("data.load.use_img_dataset is False — no image dataset would be built at train "
              "time, so there is nothing to cache. Pass data.load.use_img_dataset=True to force it.")
        return

    print(f"Building image cache for img_size={img_size} ...")
    t0 = time.time()

    builder = PipelineBuilder(config)
    builder.load_dataset()

    elapsed = time.time() - t0
    img_dataset = builder.img_dataset
    n_unique = len(img_dataset.unique_paths)
    cache_path = img_dataset.all_image_path
    cache_bytes = os.path.getsize(cache_path) if os.path.exists(cache_path) else 0

    print("\n== Build report ==============================================")
    print(f"img_size           : {img_size}")
    print(f"Unique images       : {n_unique}")
    print(f"Cache path          : {cache_path}")
    print(f"Cache size          : {cache_bytes / (1024 ** 3):.2f} GB")
    print(f"Elapsed             : {elapsed:.1f}s")
    print("==============================================================")


if __name__ == "__main__":
    main()
