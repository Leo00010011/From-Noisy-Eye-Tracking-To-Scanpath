#!/bin/bash
#SBATCH --job-name=denoise_pretrain
#SBATCH --output=logs/denoise_out_%j.log
#SBATCH --error=logs/denoise_err_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
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

echo "STARTING TRAINING"
python train.py +head_type=multi_mlp

echo "Finished debug at: $(date)"