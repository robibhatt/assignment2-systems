#!/bin/bash
# Usage: scripts/pull_nsys.sh <DATETIME_STRING>
# Example: scripts/pull_nsys.sh 2025-11-07_19-07-36
# Assumes you run this from the project root on your LOCAL machine.

set -euo pipefail

DT="${1:-}"

# ---- CONFIG: edit if your server/user/path differ ----
REMOTE_USER="luj210"
REMOTE_HOST="galvani-login"
REMOTE_BASE="/mnt/lustre/work/luxburg/luj210/projects/stanford_class/assignment2-systems"
REMOTE_FILE="${REMOTE_BASE}/scripts/results/${DT}/result.nsys-rep"

LOCAL_DEST_DIR="scripts/end_to_end/tmp"
LOCAL_DEST_FILE="${LOCAL_DEST_DIR}/result.nsys-rep"
# ------------------------------------------------------

# Ensure destination dir exists
mkdir -p "${LOCAL_DEST_DIR}"

# Replace existing file if present (scp will overwrite; rm makes intent explicit)
rm -f "${LOCAL_DEST_FILE}" 2>/dev/null || true

echo "Pulling ${REMOTE_FILE}"
echo "   from ${REMOTE_USER}@${REMOTE_HOST}"
echo "   to   ${LOCAL_DEST_FILE}"

scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_FILE}" "${LOCAL_DEST_FILE}"

echo "Done."
