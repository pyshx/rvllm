#!/usr/bin/env bash
# Side-by-side load test: rvLLM vs vLLM.
# Runs identical load tests against both servers and prints comparison.
#
# Usage:
#   bash bench/compare_vllm.sh
#   MODEL=Qwen/Qwen2.5-1.5B CONCURRENCY=32 DURATION=60 bash bench/compare_vllm.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
RVLLM_PORT="${RVLLM_PORT:-8000}"
VLLM_PORT="${VLLM_PORT:-8001}"
CONCURRENCY="${CONCURRENCY:-16}"
DURATION="${DURATION:-30}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
RVLLM_JSON="/tmp/compare_rvllm.json"
VLLM_JSON="/tmp/compare_vllm.json"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

log() { echo -e "${BLUE}[compare]${NC} $*"; }
ok()  { echo -e "${GREEN}[ok]${NC} $*"; }
err() { echo -e "${RED}[err]${NC} $*" >&2; }

# --- Prerequisites ---
command -v python3 >/dev/null 2>&1 || { err "python3 not found"; exit 1; }
python3 -c "import aiohttp" 2>/dev/null || {
    log "Installing aiohttp..."
    pip3 install aiohttp --quiet
}

wait_for_server() {
    local url="$1"
    local timeout="$2"
    for i in $(seq 1 "$timeout"); do
        if curl -s "${url}/health" | grep -q "ok"; then
            return 0
        fi
        # vLLM uses different health endpoints
        if curl -s "${url}/health" 2>/dev/null | grep -q -i "healthy\|ok\|true"; then
            return 0
        fi
        if curl -s -o /dev/null -w "%{http_code}" "${url}/v1/models" 2>/dev/null | grep -q "200"; then
            return 0
        fi
        sleep 1
    done
    return 1
}

kill_port() {
    local port="$1"
    lsof -ti:"$port" 2>/dev/null | xargs kill -9 2>/dev/null || true
}

# =====================================================================
#  Phase 1: rvLLM
# =====================================================================
log "Phase 1: rvLLM"

# Build rvllm
log "Building rvllm-server..."
cd "$REPO_DIR"
cargo build --release --features cuda -p rvllm-server 2>&1 | tail -3
BINARY="./target/release/rvllm"
[ -f "$BINARY" ] || { err "rvllm binary not found"; exit 1; }

kill_port "$RVLLM_PORT"
sleep 1

log "Starting rvllm on port ${RVLLM_PORT}..."
nohup "$BINARY" serve --model "$MODEL" --port "$RVLLM_PORT" > /tmp/compare_rvllm_server.log 2>&1 &
RVLLM_PID=$!

if ! wait_for_server "http://localhost:${RVLLM_PORT}" 120; then
    err "rvllm failed to start"
    cat /tmp/compare_rvllm_server.log
    kill "$RVLLM_PID" 2>/dev/null || true
    exit 1
fi
ok "rvllm ready (pid=$RVLLM_PID)"

log "Running load test against rvllm..."
python3 "${SCRIPT_DIR}/loadtest.py" \
    --url "http://localhost:${RVLLM_PORT}" \
    --concurrency "$CONCURRENCY" \
    --duration "$DURATION" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --model "$MODEL" \
    --seed 42 \
    --json "$RVLLM_JSON"

log "Stopping rvllm..."
kill "$RVLLM_PID" 2>/dev/null || true
wait "$RVLLM_PID" 2>/dev/null || true
sleep 2

# =====================================================================
#  Phase 2: vLLM
# =====================================================================
log "Phase 2: vLLM"

command -v vllm >/dev/null 2>&1 || python3 -c "import vllm" 2>/dev/null || {
    err "vLLM not installed. Install with: pip install vllm"
    err "Skipping vLLM benchmark, showing rvllm results only."
    echo ""
    cat "$RVLLM_JSON"
    exit 0
}

kill_port "$VLLM_PORT"
sleep 1

log "Starting vLLM on port ${VLLM_PORT}..."
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$VLLM_PORT" \
    --dtype auto \
    --max-model-len 2048 \
    > /tmp/compare_vllm_server.log 2>&1 &
VLLM_PID=$!

if ! wait_for_server "http://localhost:${VLLM_PORT}" 180; then
    err "vLLM failed to start in 180s"
    cat /tmp/compare_vllm_server.log
    kill "$VLLM_PID" 2>/dev/null || true
    exit 1
fi
ok "vLLM ready (pid=$VLLM_PID)"

log "Running load test against vLLM..."
python3 "${SCRIPT_DIR}/loadtest.py" \
    --url "http://localhost:${VLLM_PORT}" \
    --concurrency "$CONCURRENCY" \
    --duration "$DURATION" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --model "$MODEL" \
    --seed 42 \
    --json "$VLLM_JSON"

log "Stopping vLLM..."
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true

# =====================================================================
#  Phase 3: Comparison
# =====================================================================
echo ""
echo ""

python3 - "$RVLLM_JSON" "$VLLM_JSON" <<'PYEOF'
import json
import sys

def load(path):
    with open(path) as f:
        return json.load(f)

rvllm = load(sys.argv[1])
vllm = load(sys.argv[2])

rs = rvllm["summary"]
vs = vllm["summary"]
rl = rvllm["latency_ms"]
vl = vllm["latency_ms"]

def ratio(a, b):
    if b == 0:
        return "N/A"
    r = a / b
    return f"{r:.2f}x"

def winner(a, b, higher_better=True):
    if higher_better:
        return "rvllm" if a > b else "vllm" if b > a else "tie"
    else:
        return "rvllm" if a < b else "vllm" if b < a else "tie"

def fmt_winner(w):
    if w == "rvllm":
        return "<-- rvllm"
    elif w == "vllm":
        return "vllm -->"
    return ""

W = 62
print("=" * W)
print("  rvLLM vs vLLM -- Load Test Comparison")
print("=" * W)
print(f"  Model:       {rvllm['config']['model']}")
print(f"  Concurrency: {rvllm['config']['concurrency']}")
print(f"  Duration:    {rvllm['config']['duration_target']}s")
print(f"  Max tokens:  {rvllm['config']['max_tokens']}")
print("=" * W)

header = f"  {'Metric':<25} {'rvllm':>10} {'vllm':>10} {'ratio':>8}  {'winner'}"
print(header)
print("-" * W)

rows = [
    ("Throughput (tok/s)", rs["throughput_tok_s"], vs["throughput_tok_s"], True),
    ("Requests/s", rs["requests_per_s"], vs["requests_per_s"], True),
    ("Total requests", rs["total_requests"], vs["total_requests"], True),
    ("Errors", rs["errors"], vs["errors"], False),
    ("Avg latency (ms)", rl["avg"], vl["avg"], False),
    ("P50 latency (ms)", rl["p50"], vl["p50"], False),
    ("P75 latency (ms)", rl["p75"], vl["p75"], False),
    ("P90 latency (ms)", rl["p90"], vl["p90"], False),
    ("P95 latency (ms)", rl["p95"], vl["p95"], False),
    ("P99 latency (ms)", rl["p99"], vl["p99"], False),
    ("Max latency (ms)", rl["max"], vl["max"], False),
]

for name, rv, vv, higher_better in rows:
    if isinstance(rv, int) and isinstance(vv, int):
        rv_s = f"{rv:>10d}"
        vv_s = f"{vv:>10d}"
    else:
        rv_s = f"{rv:>10.1f}"
        vv_s = f"{vv:>10.1f}"

    if higher_better:
        r = ratio(rv, vv)
    else:
        r = ratio(vv, rv)

    w = fmt_winner(winner(rv, vv, higher_better))
    print(f"  {name:<25} {rv_s} {vv_s} {r:>8}  {w}")

print("=" * W)

# Overall verdict
tput_ratio = rs["throughput_tok_s"] / vs["throughput_tok_s"] if vs["throughput_tok_s"] > 0 else float("inf")
p50_ratio = vl["p50"] / rl["p50"] if rl["p50"] > 0 else float("inf")

print(f"\n  Throughput: rvllm is {tput_ratio:.2f}x of vLLM")
print(f"  P50 latency: rvllm is {p50_ratio:.2f}x {'faster' if p50_ratio > 1 else 'slower'} than vLLM")
print()
PYEOF

ok "Comparison complete. Raw data: ${RVLLM_JSON} ${VLLM_JSON}"
