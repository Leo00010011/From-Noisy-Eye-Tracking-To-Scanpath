#!/bin/bash
#SBATCH --job-name=build_features
#SBATCH --output=logs/features_out_%j.log
#SBATCH --error=logs/features_err_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leonardo.ulloa@rai.usc.gal


echo "Starting features cache build at: $(date)"

echo "Running on node: $SLURM_NODELIST"

echo "Moving to home"
cd /mnt/beegfs/home/leonardo.ulloa
HOME_DIR="$(pwd)"

echo "Mounting image "
sudo mount_image.py my_env.ext4 --rw

# Use single quotes for the definition to be safe
SOURCE_DATA='projects/From-Noisy-Eye-Tracking-To-Scanpath/data/Coco FreeView'
DEST_DATA="$LOCAL_SCRATCH/data/Coco FreeView"

# The --out path below is made absolute with $HOME_DIR on purpose: this script cd's into the
# project dir before running python, so a path relative to $SOURCE_DATA would resolve to a
# DOUBLED projects/From-.../projects/From-.../data/... directory (the HDF5 writer makedirs
# it silently, so the build still "succeeds" in the wrong place). train_ms.sh reads
# "$HOME_DIR/$SOURCE_DATA/image_features_${IMG_SIZE}.h5" - keep the two in sync.

# Create the directory
mkdir -p "$DEST_DATA"

echo "Transferring data to local scratch..."

# Ensure we quote the variables in the command
rsync -aq "$SOURCE_DATA/" "$DEST_DATA/"

echo "Conda INIT"
source /mnt/beegfs/home/leonardo.ulloa/miniconda3/etc/profile.d/conda.sh

echo "Activating Conda env"
conda activate scanpath

echo "Moving to project"
cd projects/From-Noisy-Eye-Tracking-To-Scanpath/

# ---------------------------------------------------------------------------
# Cache config. IMG_SIZE=512 => S = 16²+32²+64² = 5376 tokens per image.
#
# Size at 512 (U = 4317 unique images, float32):
#   ms_value      (U, 5376, 256)      ~= 22.1 GiB  (23.8 GB)
#   mask_features (U, 256, 128, 128)  ~= 67.5 GiB  (72.4 GB)   <- model never reads these
#   full cache                        ~= 89.6 GiB  (96.2 GB)
#
# The training model only ever reads ms_value, so mask_features (reserved for a future heatmap
# head) is left OUT by default here. Set INCLUDE_MASK_FEATURES=1 to store the full ~90 GiB cache.
# ---------------------------------------------------------------------------
IMG_SIZE=512
BATCH_SIZE=32
INCLUDE_MASK_FEATURES=0

MASK_FLAG="--no-mask-features"
if [ "$INCLUDE_MASK_FEATURES" = "1" ]; then
    MASK_FLAG=""
fi

echo "STARTING FEATURE CACHE BUILD (img_size=$IMG_SIZE, mask_flag='$MASK_FLAG')"
python scripts/build_image_feature_cache.py \
    --img-size "$IMG_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --device cuda \
    --data-root "$DEST_DATA" \
    --out "$HOME_DIR/$SOURCE_DATA/image_features_${IMG_SIZE}.h5" \
    $MASK_FLAG

echo "Finished features cache build at: $(date)"
