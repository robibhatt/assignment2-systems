#!/bin/bash
#SBATCH --job-name=make_sweep_bori
#SBATCH --output=cs336_systems/end_to_end_benchmark/end_to_end_benchmark.out
#SBATCH --error=cs336_systems/end_to_end_benchmark/end_to_end_benchmark.err
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=30:00:00

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

uv run cs336_systems/end_to_end_benchmark/end_to_end_benchmark.py