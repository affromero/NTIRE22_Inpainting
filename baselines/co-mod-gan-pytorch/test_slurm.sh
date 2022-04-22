#!/bin/bash
#SBATCH  --output=log_slurm/test_%j.txt
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
#SBATCH  --constraint='titan_xp'

export TORCH_EXTENSIONS_DIR=__pytorch__
source /scratch_net/schusch/Andres/anaconda3/etc/profile.d/conda.sh
# From here, it's just what you executed in srun
conda activate /scratch_net/schusch/Andres/Code/Inpainting/baselines/AOT-GAN-for-Inpainting/envs

export TORCH_HOME=$(pwd) && export PYTHONPATH=.

sh inpainting_challenge.sh

cd ../../inpainting_evaluation
sh run_comod.sh

# python bin/train.py -cn big-lama-pretrained location=imagenet