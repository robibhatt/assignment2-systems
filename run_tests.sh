#!/bin/bash
#SBATCH --job-name=ibor_train_bpe
#SBATCH --output=logs/tests.out
#SBATCH --error=logs/tests.err
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=00:10:00

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

uv run pytest -k test_flash_forward_pass_pytorch