#!/bin/bash
# bench.sh -- minimal H100 benchmark. Pull, build, run.
# Usage: ssh -p PORT root@HOST 'bash -s' < deploy/bench.sh
set -euo pipefail

source /root/.cargo/env 2>/dev/null || true

cd /root/rvllm
git fetch origin && git reset --hard origin/main
echo "HEAD: $(git log --oneline -1)"

# Compile PTX kernels
echo "=== PTX kernels ==="
cd kernels && bash build.sh sm_90 2>&1 | grep -E "Done|WARNING|FAIL" && cd ..

# Verify CUTLASS .so
ls -lh kernels/sm_90/libcutlass_kernels.so

# Build
echo "=== Building ==="
apt-get install -y -qq pkg-config libssl-dev 2>/dev/null | tail -1
cargo build --release --features cuda,cublaslt 2>&1 | tail -3
echo "Binary: $(ls -lh target/release/rvllm)"

# Kill any rogue GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 1

# GPU info
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Benchmark -- each N is a separate process (model reload, unavoidable for CUDA graphs)
echo ""
echo "=== Benchmark: Qwen2.5-7B fp16, 512 output tokens ==="
export RVLLM_PTX_DIR=kernels/sm_90
for N in 16 32 48 64 96 128; do
    echo -n "N=$N ... "
    OUTPUT=$(target/release/rvllm benchmark \
        --model Qwen/Qwen2.5-7B \
        --dtype half \
        --gpu-memory-utilization 0.98 \
        --output-len 512 \
        --n "$N" 2>&1)
    # Extract tok/s from output
    TOKS=$(echo "$OUTPUT" | grep -oE '[0-9]+ tok/s' | head -1)
    echo "${TOKS:-FAILED}"
    echo "$OUTPUT" | grep -v "^$" | tail -3
    echo ""
done

echo "=== DONE ==="
