#!/bin/bash
#SBATCH --job-name=overfit
#SBATCH --output=logs/overfit_out_%j.log
#SBATCH --error=logs/overfit_err_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leonardo.ulloa@rai.usc.gal

# OVERFIT-A-BATCH SANITY TEST for the parallel-decoding recipe.
#
# Same environment and same caches as train_parallel_decoding.sh — precomputed frozen
# Mask2Former features, the image-adapter checkpoint, DETR-style parallel decoding — but
# the run is collapsed onto ONE batch of 8 scanpaths with every source of randomness and
# regularisation removed (see configs/exp/overfit_batch.yaml for the full list).
#
# Read the result as: reg_error_val should fall toward ~0 within a few hundred epochs.
# If it plateaus at a non-trivial error, the fault is in the architecture, the loss wiring
# or the optimisation — NOT in the data, the noise model, or the training budget.
#
# 2h and one GPU is plenty: 1 step/epoch x 400 epochs.

echo "Starting overfit test at: $(date)"

echo "Running on node: $SLURM_NODELIST"

echo "Moving to home"
cd /mnt/beegfs/home/leonardo.ulloa
HOME_DIR="$(pwd)"

echo "Mounting image "
sudo mount_image.py my_env.ext4 --rw

SOURCE_DATA='projects/From-Noisy-Eye-Tracking-To-Scanpath/data/Coco FreeView'
DEST_DATA="$LOCAL_SCRATCH/data/Coco FreeView"

# Absolute BeeGFS paths (Hydra chdirs into the run directory, so relative paths are unsafe).
# The ~22 GB frozen-feature cache is read straight from BeeGFS, NOT copied into scratch.
FEATURE_CACHE="$HOME_DIR/$SOURCE_DATA/image_features_512.h5"
CENTROID_CACHE="$HOME_DIR/$SOURCE_DATA/scanpath_centroids.h5"

mkdir -p "$DEST_DATA"

echo "Transferring data to local scratch..."
rsync -aq --exclude 'image_features_*.h5' "$SOURCE_DATA/" "$DEST_DATA/"

echo "Conda INIT"
source /mnt/beegfs/home/leonardo.ulloa/miniconda3/etc/profile.d/conda.sh

echo "Activating Conda env"
conda activate scanpath

echo "Moving to project"
cd projects/From-Noisy-Eye-Tracking-To-Scanpath/

if [ ! -f "$CENTROID_CACHE" ]; then
    echo "Centroid cache not found — building it: $CENTROID_CACHE"
    python scripts/build_scanpath_centroid_cache.py \
        --out "$CENTROID_CACHE" \
        --bandwidth-dva 1.0 \
        --data-path "$HOME_DIR/$SOURCE_DATA"
else
    echo "Reusing existing centroid cache: $CENTROID_CACHE"
fi

echo "Exporting WANDB_API_KEY"
export WANDB_API_KEY="$(cat ~/.wandb_api_key)"

echo "STARTING OVERFIT-A-BATCH TEST (8 samples, Combined phase only, no noise, no dropout)"
# The exp config already sets model/image_encoder=mask2former_precomputed,
# data.load.use_precomputed_features=True, model.image_adaptation.enabled=True,
# model.parallel_decoding=True, data.overfit.enabled=True and the single Combined phase.
# Only the two cache paths need the absolute BeeGFS form.
python train.py exp=overfit_batch \
    +data.load.feature_cache_path="$FEATURE_CACHE" \
    model.image_adaptation.centroid_cache_path="$CENTROID_CACHE"

echo "Finished overfit test at: $(date)"
