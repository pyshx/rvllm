#!/bin/bash
# rsync_and_run.sh -- Push code to H100 instance and run deploy_and_bench.sh
#
# Usage:
#   bash deploy/rsync_and_run.sh                          # defaults
#   bash deploy/rsync_and_run.sh ssh8.vast.ai 20236       # explicit host/port
#   bash deploy/rsync_and_run.sh myhost 22 --skip-compile # pass flags through
#
# All arguments after host and port are forwarded to deploy_and_bench.sh.

set -euo pipefail

HOST=${1:-ssh8.vast.ai}
PORT=${2:-20236}
shift 2 2>/dev/null || true
EXTRA_ARGS="$*"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${PORT}"
REMOTE_DIR="/root/rvllm"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Target: root@${HOST}:${PORT} -> ${REMOTE_DIR}"
echo ""

# --- Rsync source code ---
echo "Syncing code..."
rsync -avz --delete \
    --exclude target \
    --exclude .git \
    --exclude '.claude' \
    --exclude '*.ptx' \
    --exclude 'kernels/sm_*' \
    -e "ssh ${SSH_OPTS}" \
    "${REPO_DIR}/" "root@${HOST}:${REMOTE_DIR}/"

echo ""
echo "Running deploy_and_bench.sh on remote..."
echo "========================================"

# --- Run on remote ---
ssh ${SSH_OPTS} "root@${HOST}" "
    export PATH=/root/.cargo/bin:\$PATH
    cd ${REMOTE_DIR}
    bash deploy/deploy_and_bench.sh ${EXTRA_ARGS}
"
