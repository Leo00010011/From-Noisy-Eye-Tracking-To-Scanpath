#!/bin/bash
#SBATCH --job-name=whole_train
#SBATCH --output=logs/denoise_out_%j.log
#SBATCH --error=logs/denoise_err_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=16:00:00
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
# Never hardcode the key here -- it belongs in a file outside version control.
# Create it once with: echo <key> > ~/.wandb_api_key && chmod 600 ~/.wandb_api_key
# Exporting the key is what stops wandb from dropping into its interactive login prompt.
export WANDB_API_KEY="$(cat ~/.wandb_api_key)"
# If the compute node has no outbound internet, log offline instead and `wandb sync` later:
#   export WANDB_MODE=offline

echo "STARTING TRAINING"
python train.py exp=final_c41 scheduled_sampling.warmup_epochs=20 training.wandb.name=c41_w20_mp085

echo "Finished debug at: $(date)"
