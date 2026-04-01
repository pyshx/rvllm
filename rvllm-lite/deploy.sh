#!/bin/bash
set -euo pipefail

# Deploy rvllm-lite to a vast.ai instance.
# Usage:
#   ./rvllm-lite/deploy.sh                    # auto-detect instance
#   ./rvllm-lite/deploy.sh <instance_id>      # explicit instance

INSTANCE_ID=${1:-$(cat deploy/.instance_id_rvllm 2>/dev/null || echo "")}
if [[ -z "$INSTANCE_ID" ]]; then
    echo "No instance ID. Run deploy/vastai-provision.sh first or pass ID."
    exit 1
fi

MODEL="Qwen/Qwen2.5-7B"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH_URL=$(vastai ssh-url "$INSTANCE_ID")
SSH_HOST=$(echo "$SSH_URL" | sed 's|ssh://||' | cut -d: -f1)
SSH_PORT=$(echo "$SSH_URL" | sed 's|ssh://||' | cut -d: -f2)
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT $SSH_HOST"
SCP="scp -o StrictHostKeyChecking=no -P $SSH_PORT"

echo "Instance: $INSTANCE_ID ($SSH_HOST:$SSH_PORT)"
echo "Model: $MODEL"
echo ""

# Push rvllm-lite to instance
echo "=== Syncing rvllm-lite ==="
rsync -az --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
    -e "ssh -o StrictHostKeyChecking=no -p $SSH_PORT" \
    "$REPO_DIR/rvllm-lite/" "$SSH_HOST:/root/rvllm-lite/"

$SSH 'bash -l' << 'REMOTE'
set -euo pipefail

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# Python env
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null
    export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

if [[ ! -d /root/.venv/rvllm-lite ]]; then
    uv venv /root/.venv/rvllm-lite --python 3.11 2>/dev/null || uv venv /root/.venv/rvllm-lite 2>/dev/null
fi
source /root/.venv/rvllm-lite/bin/activate

echo "=== Installing deps ==="
cd /root/rvllm-lite
uv pip install -e "." 2>&1 | tail -3

# Download model
echo "=== Model weights ==="
MODEL_LOCAL="/root/models/Qwen2.5-7B"
if [[ -f "$MODEL_LOCAL/config.json" ]]; then
    echo "Model already at $MODEL_LOCAL"
else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B', local_dir='$MODEL_LOCAL')
print('done')
"
fi

# Kill anything on 8000
pkill -9 -f rvllm_lite 2>/dev/null || true
sleep 1

echo "=== Starting rvllm-lite ==="
python3 -m rvllm_lite --model "$MODEL_LOCAL" --port 8000 --dtype float16 \
    > /tmp/rvllm_lite.log 2>&1 &
PID=$!
echo "Server PID: $PID"

# Wait for healthy
for i in $(seq 1 180); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        echo "Server healthy after ${i}s"
        break
    fi
    if ! kill -0 $PID 2>/dev/null; then
        echo "SERVER DIED"
        tail -40 /tmp/rvllm_lite.log
        exit 1
    fi
    sleep 1
done

if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "Timed out waiting for server"
    tail -40 /tmp/rvllm_lite.log
    exit 1
fi

echo ""
echo "=== Smoke test ==="
curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt":"The theory of general relativity states that","max_tokens":64,"temperature":0.0}' \
    --max-time 120 | python3 -m json.tool

echo ""
echo "=== Quick bench (3 runs, 256 tokens) ==="
for run in 1 2 3; do
    START=$(date +%s%N)
    RESP=$(curl -s -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"prompt":"The theory of general relativity states that","max_tokens":256,"temperature":0.0}' \
        --max-time 120)
    END=$(date +%s%N)
    TOKS=$(echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); c=r.get('choices',[{}])[0].get('text',''); print(len(c.split()))" 2>/dev/null || echo 0)
    MS=$(( (END - START) / 1000000 ))
    echo "  run$run: ${MS}ms"
done

echo ""
echo "=== Server running on port 8000 ==="
echo "Logs: tail -f /tmp/rvllm_lite.log"
REMOTE
