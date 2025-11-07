#!/bin/bash
#SBATCH --job-name=end_to_end
#SBATCH --output=scripts/end_to_end/logs/end_to_end.out
#SBATCH --error=scripts/end_to_end/logs/end_to_end.err
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=20:00:00

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

uv run cs336_systems/end_to_end_benchmark/end_to_end_benchmark.py