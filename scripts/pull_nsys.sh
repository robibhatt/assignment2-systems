#!/bin/bash
# Usage: scripts/pull_nsys.sh <DATETIME_STRING>
# Example: scripts/pull_nsys.sh 2025-11-07_19-47-45
# Run this from your local project root.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <DATETIME_STRING>"
  exit 1
fi

DT="$1"

# --- remote info ---
SSH_USER="luj210"
SSH_HOST="galvani-login.mlcloud.uni-tuebingen.de"
REMOTE_BASE="/mnt/lustre/work/luxburg/luj210/projects/stanford_class/assignment2-systems"
REMOTE_FILE="${REMOTE_BASE}/scripts/end_to_end/results/${DT}/result.nsys-rep"
# --------------------

# --- local info ---
LOCAL_DEST_DIR="scripts/end_to_end/tmp"
LOCAL_DEST_FILE="${LOCAL_DEST_DIR}/result.nsys-rep"
# -------------------

mkdir -p "${LOCAL_DEST_DIR}"
rm -f "${LOCAL_DEST_FILE}" 2>/dev/null || true

echo "Pulling Nsight report from server..."
echo "  ${SSH_USER}@${SSH_HOST}:${REMOTE_FILE}"
echo "to local path:"
echo "  ${LOCAL_DEST_FILE}"

scp "${SSH_USER}@${SSH_HOST}:${REMOTE_FILE}" "${LOCAL_DEST_FILE}"

echo "✅ Done. File copied to ${LOCAL_DEST_FILE}"
