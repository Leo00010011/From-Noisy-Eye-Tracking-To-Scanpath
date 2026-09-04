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

echo "Mounting image "
sudo mount_image.py my_env.ext4 --rw

# Use single quotes for the definition to be safe
SOURCE_DATA='projects/From-Noisy-Eye-Tracking-To-Scanpath/data/Coco FreeView'
DEST_DATA="$LOCAL_SCRATCH/data/Coco FreeView"

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

echo "Exporting WANDB_API_KEY"

export WANDB_API_KEY="$(cat ~/.wandb_api_key)"

echo "STARTING TRAINING (precomputed frozen Mask2Former features, img_size=512)"
# Train off the cached frozen features (image_features_512.h5) instead of the live backbone.
#   - model/image_encoder=mask2former_precomputed : precomputed=True, spatial_shapes for 512
#   - data.load.use_precomputed_features=True      : FR12 requires features on both sides
#   - feature_cache_path -> the scratch copy rsync'd above (fast NVMe reads; ~24 GB streamed/epoch)
# img_size is already 512 in configs/data/default.yaml, matching the cache attrs.
python train.py exp=only_combined \
    model/image_encoder=mask2former_precomputed \
    +data.load.use_precomputed_features=True \
    +data.load.feature_cache_path="$DEST_DATA/image_features_512.h5"

echo "Finished debug at: $(date)"
