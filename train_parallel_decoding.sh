#!/bin/bash
#SBATCH --job-name=train_par
#SBATCH --output=logs/par_out_%j.log
#SBATCH --error=logs/par_err_%j.log
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

# Absolute BeeGFS paths (Hydra may chdir, so relative paths are unsafe).
# The ~22 GB frozen-feature cache is read straight from BeeGFS, NOT copied into scratch.
FEATURE_CACHE="$HOME_DIR/$SOURCE_DATA/image_features_512.h5"
# The centroid cache is tiny (per-image centroids); it lives beside the data on BeeGFS.
CENTROID_CACHE="$HOME_DIR/$SOURCE_DATA/scanpath_centroids.h5"

# Create the directory
mkdir -p "$DEST_DATA"

echo "Transferring data to local scratch..."

# Copy dataset.hdf5, ntop.json, etc. to scratch, but EXCLUDE the large feature caches (streamed
# from BeeGFS instead). scanpath_centroids.h5 is small and IS copied — but training reads it from
# the absolute BeeGFS path below either way. ntop.json travels with the data so get_img_path keys
# identically to how both caches were built (the first-seen order invariant, FR3/AI2).
rsync -aq --exclude 'image_features_*.h5' "$SOURCE_DATA/" "$DEST_DATA/"

echo "Conda INIT"
source /mnt/beegfs/home/leonardo.ulloa/miniconda3/etc/profile.d/conda.sh

echo "Activating Conda env"
conda activate scanpath
pip3 install -U scikit-learn
echo "Moving to project"
cd projects/From-Noisy-Eye-Tracking-To-Scanpath/

# Build the scanpath-centroid cache once if it is missing (CPU-only, quick — unlike the frozen
# feature cache). Built against the BeeGFS data root so its ntop.json / image_path order matches
# the frozen-feature cache. 1-DVA Mean Shift is the default; the stimuly_disjoint split is
# leak-free so no --split-restrict is needed.
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

echo "STARTING PARALLEL-DECODING TRAINING (DETR-style parallel decoder, precomputed frozen Mask2Former features, img_size=512)"
# Same 3-phase run as image adaptation: ImageAdaptation -> Combined (adapter frozen) ->
# FullFinetune (pixel decoder frozen), but the fixation decoder is non-autoregressive
# (model.parallel_decoding=True: learned index query embeddings + non-causal self-attention,
# every fixation slot predicted in one pass, no GT leak, no scheduled sampling).
# The exp config already sets model/image_encoder=mask2former_precomputed,
# data.load.use_precomputed_features=True, model.image_adaptation.enabled=True,
# model.parallel_decoding=True, and the phases.
# Override the two caches to their absolute BeeGFS paths (relative paths are unsafe under chdir):
#   - feature_cache_path      : streamed frozen features (too big for scratch)
#   - centroid_cache_path     : the alignment target centroids
python train.py exp=parallel_decoding_training \
    +data.load.feature_cache_path="$FEATURE_CACHE" \
    model.image_adaptation.centroid_cache_path="$CENTROID_CACHE"

echo "Finished debug at: $(date)"
