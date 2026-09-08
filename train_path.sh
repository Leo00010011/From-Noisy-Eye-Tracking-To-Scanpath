#!/bin/bash
#SBATCH --job-name=train_path
#SBATCH --output=logs/path_out_%j.log
#SBATCH --error=logs/path_err_%j.log
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

# Create the directory
mkdir -p "$DEST_DATA"

echo "Transferring data to local scratch..."

# PathModel is gaze-only: it needs dataset.hdf5, not the image feature caches. Exclude the
# large image_features_*.h5 so they don't blow the size-limited scratch (the gaze-only path
# never reads them).
rsync -aq --exclude 'image_features_*.h5' "$SOURCE_DATA/" "$DEST_DATA/"

echo "Conda INIT"
source /mnt/beegfs/home/leonardo.ulloa/miniconda3/etc/profile.d/conda.sh

echo "Activating Conda env"
conda activate scanpath

echo "Moving to project"
cd projects/From-Noisy-Eye-Tracking-To-Scanpath/

echo "Exporting WANDB_API_KEY"

export WANDB_API_KEY="$(cat ~/.wandb_api_key)"

echo "STARTING TRAINING (PathModel, gaze-only ablation baseline)"
# Gaze-only encoder-decoder transformer (no image features, no DINOv3/Mask2Former).
#   - exp=path_model_training : Fixation-only + scheduled sampling + separated_reg loss
#   - reuse_split_from        : (optional) point at a MixerModel run dir with split.pth so
#                               PathModel is evaluated on the IDENTICAL test split for an
#                               apples-to-apples ablation. Leave unset to generate a fresh
#                               stimuli-disjoint split (FR9 guard covers the gaze-only path).
python train.py exp=path_model_training
#   +training.reuse_split_from="$HOME_DIR/projects/From-Noisy-Eye-Tracking-To-Scanpath/outputs/<date>/<time>"

echo "Finished debug at: $(date)"
