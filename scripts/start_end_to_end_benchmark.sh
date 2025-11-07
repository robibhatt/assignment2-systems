#!/bin/bash
#SBATCH --job-name=end_to_end
#SBATCH --output=scripts/end_to_end/logs/end_to_end.out
#SBATCH --error=scripts/end_to_end/logs/end_to_end.err
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time=10:00:00

# Diagnostics
scontrol show job "$SLURM_JOB_ID"
pwd
nvidia-smi || true

# --- make nsys visible on the compute node ---
NSYS_DIR="/mnt/lustre/work/luxburg/luj210/tools/nsight-systems/opt/nvidia/nsight-systems-cli/2025.5.1/target-linux-x64"
export PATH="$NSYS_DIR:$PATH"

# --- run the assignment command (records everything; NVTX ranges still appear) ---
uv run nsys profile --trace=cuda,nvtx -o result \
  python cs336_systems/end_to_end_benchmark/end_to_end_benchmark.py

# --- move the .nsys-rep into the newest results/<timestamp>/ directory ---
DEST_FILE=$(ls -t scripts/end_to_end/results/*/results.out 2>/dev/null | head -n1 || true)
if [[ -n "$DEST_FILE" && -f result.nsys-rep ]]; then
  DEST_DIR=$(dirname "$DEST_FILE")
  mv result.nsys-rep "$DEST_DIR"/
  echo "Moved Nsight report to: $DEST_DIR/result.nsys-rep"
else
  echo "Leaving Nsight report in: $(pwd)/result.nsys-rep"
fi
