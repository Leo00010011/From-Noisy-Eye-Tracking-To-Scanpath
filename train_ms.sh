#!/bin/bash
#SBATCH --job-name=train_ms
#SBATCH --output=logs/denoise_out_%j.log
#SBATCH --error=logs/denoise_err_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leonardo.ulloa@rai.usc.gal


echo "Starting debug at: $(date)"

echo "Running on node: $SLURM_NODELIST"

echo "Moving to home"
cd /mnt/beegfs/home/leonardo.ulloa
HOME_DIR="$(pwd)"

echo "Mounting image "
sudo mount_image.py my_env.ext4 --rw

# Use single quotes for the definition to be safe
SOURCE_DATA='projects/From-Noisy-Eye-Tracking-To-Scanpath/data/Coco FreeView'
DEST_DATA="$LOCAL_SCRATCH/data/Coco FreeView"

# Absolute BeeGFS path to the feature cache (Hydra may chdir, so relative paths are unsafe).
# The ~22 GB cache is read straight from BeeGFS, NOT copied into the size-limited scratch.
FEATURE_CACHE="$HOME_DIR/$SOURCE_DATA/image_features_512.h5"

# Create the directory
mkdir -p "$DEST_DATA"

echo "Transferring data to local scratch..."

# Copy dataset.hdf5 etc. to scratch, but EXCLUDE the large feature caches: they won't fit in
# scratch alongside the dataset and are streamed from BeeGFS instead (previous run silently
# failed to rsync the 22 GB .h5 and training then couldn't find it).
rsync -aq --exclude 'image_features_*.h5' "$SOURCE_DATA/" "$DEST_DATA/"

echo "Conda INIT"
source /mnt/beegfs/home/leonardo.ulloa/miniconda3/etc/profile.d/conda.sh

echo "Activating Conda env"
conda activate scanpath

echo "Moving to project"
cd projects/From-Noisy-Eye-Tracking-To-Scanpath/

echo "Exporting WANDB_API_KEY"

export WANDB_API_KEY="$(cat ~/.wandb_api_key)"

echo "STARTING TRAINING (precomputed frozen Mask2Former features, img_size=512)"
# Train off the cached frozen features (image_features_512.h5) instead of the live backbone.
#   - model/image_encoder=mask2former_precomputed : precomputed=True, spatial_shapes for 512
#   - data.load.use_precomputed_features=True      : FR12 requires features on both sides
#   - feature_cache_path -> absolute BeeGFS path (streamed from BeeGFS; too big for scratch)
# img_size is already 512 in configs/data/default.yaml, matching the cache attrs.
python train.py exp=only_combined \
    model/image_encoder=mask2former_precomputed \
    +data.load.use_precomputed_features=True \
    +data.load.feature_cache_path="$FEATURE_CACHE"

echo "Finished debug at: $(date)"
