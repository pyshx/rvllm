#!/usr/bin/env bash
# rvLLM load test wrapper.
# Builds server, starts it, runs loadtest.py, kills server, prints results.
#
# Usage:
#   bash bench/loadtest.sh
#   CONCURRENCY=64 DURATION=120 MAX_TOKENS=256 bash bench/loadtest.sh
#   MODEL=meta-llama/Llama-3-8B bash bench/loadtest.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
PORT="${PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-16}"
DURATION="${DURATION:-30}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.8}"
JSON_OUT="${JSON_OUT:-/tmp/rvllm_loadtest.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[loadtest]${NC} $*"; }
ok()  { echo -e "${GREEN}[ok]${NC} $*"; }
err() { echo -e "${RED}[err]${NC} $*" >&2; }

# --- Prerequisites ---
command -v cargo >/dev/null 2>&1 || { err "cargo not found"; exit 1; }
command -v python3 >/dev/null 2>&1 || { err "python3 not found"; exit 1; }
python3 -c "import aiohttp" 2>/dev/null || {
    log "Installing aiohttp..."
    pip3 install aiohttp --quiet
}

# --- Build ---
log "Building rvllm-server..."
cd "$REPO_DIR"
cargo build --release --features cuda -p rvllm-server 2>&1 | tail -3
BINARY="./target/release/rvllm"
[ -f "$BINARY" ] || { err "Binary not found at $BINARY"; exit 1; }
ok "Built $(ls -lh "$BINARY" | awk '{print $5}')"

# --- Start server ---
pkill -9 rvllm 2>/dev/null || true
sleep 1

log "Starting rvllm (model=${MODEL}, port=${PORT})..."
nohup "$BINARY" serve --model "$MODEL" --port "$PORT" > /tmp/rvllm_loadtest_server.log 2>&1 &
SERVER_PID=$!

cleanup() {
    log "Stopping server (pid=$SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- Wait for ready ---
BASE_URL="http://localhost:${PORT}"
for i in $(seq 1 120); do
    if curl -s "${BASE_URL}/health" | grep -q "ok"; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        err "Server died during startup"
        cat /tmp/rvllm_loadtest_server.log
        exit 1
    fi
    if [ "$i" -eq 120 ]; then
        err "Server failed to start in 120s"
        cat /tmp/rvllm_loadtest_server.log
        exit 1
    fi
    sleep 1
done
ok "Server ready"

# --- GPU info ---
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)
    log "GPU: ${GPU_NAME} (${GPU_MEM})"
fi

# --- Run load test ---
log "Running load test: concurrency=${CONCURRENCY} duration=${DURATION}s max_tokens=${MAX_TOKENS}"
python3 "${SCRIPT_DIR}/loadtest.py" \
    --url "${BASE_URL}" \
    --concurrency "$CONCURRENCY" \
    --duration "$DURATION" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --model "$MODEL" \
    --json "$JSON_OUT"

ok "Load test complete. Results: ${JSON_OUT}"
