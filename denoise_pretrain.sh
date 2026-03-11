#!/bin/bash
#SBATCH --job-name=denoise_pretrain
#SBATCH --output=denoise_out_%j.log
#SBATCH --error=denoise_err_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=leonardo.ulloa@rai.usc.gal


echo "Starting debug at: $(date)"
echo "Running on node: $SLURM_NODELIST"

sudo mount_image.py my_env.ext4 --rw

conda init

conda activate scanpath

cd projects/From-Noisy-Eye-Tracking-To-Scanpath/

python train.py +head_type=multi_mlp

echo "Finished debug at: $(date)"