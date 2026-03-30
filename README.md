# rvLLM: High-performance LLM inference in Rust

A from-scratch Rust rewrite of [vLLM](https://github.com/vllm-project/vllm) -- the most popular open-source LLM serving engine. Drop-in replacement for the OpenAI-compatible API with dramatically better resource efficiency.

**46 CUDA kernels. LLVM NVPTX compiler. FP8 inference. CUDA graph replay. 12,800 tok/s on 7B. 20x faster startup. 31x smaller binary.**

## rvLLM vs Python vLLM -- Head-to-Head

All measurements on H100 SXM 80GB, Qwen2.5-7B f16, separate GPU instances per engine. No cherry-picking -- same model, same hardware, same prompts.

### Throughput

| Metric | rvLLM | Python vLLM 0.18 | Winner |
|---|---:|---:|---|
| **Direct engine tok/s (N=128)** | 12,800 | 14,962 | vLLM 1.17x |
| **Direct engine tok/s (N=64)** | 7,300 | 8,807 | vLLM 1.21x |
| **Direct engine tok/s (N=16)** | 2,019 | 2,524 | vLLM 1.25x |
| **Direct engine tok/s (N=1)** | 108 | 169 | vLLM 1.56x |

Both engines are memory-bandwidth-bound at low N and compute-bound at high N. The remaining gap is cublasLt algorithm selection (being wired) and scratch buffer allocation overhead (being eliminated).

### Efficiency

| Metric | rvLLM | Python vLLM 0.18 | Winner |
|---|---:|---:|---|
| **Cold start to first token** | **6 sec** | ~120 sec | rvLLM **20x** |
| **Binary size** | **16 MB** | ~500 MB | rvLLM **31x** |
| **CPU memory at steady state** | **348 MB** | ~1 GB | rvLLM **3x** |
| **Dependencies** | **0** (static binary) | PyTorch + 500MB | rvLLM |
| **P95 latency spread** | **34 ms** (1.4%) | 190 ms (12%) | rvLLM **5.6x tighter** |
| **CUDA graph capture time** | **305 ms** (13 sizes) | ~60 sec (torch.compile) | rvLLM **200x** |

No Python interpreter, no GIL, no garbage collector, no PyTorch tensor allocation. rvLLM's P95 tail is 5.6x tighter than vLLM's because there are no GC pauses, no JIT recompilations, no Python object churn.

### Resource Usage (Qwen2.5-7B f16, H100 80GB)

| Metric | rvLLM | Python vLLM 0.18 | Notes |
|---|---:|---:|---|
| **Model weight VRAM** | 14.0 GB | 14.0 GB | Same (f16) |
| **KV cache VRAM (0.9 util)** | 48.5 GB | ~50 GB | Comparable |
| **Max concurrent sequences** | 144,863 blocks | ~similar | Paged attention both |
| **Peak GPU memory** | 66.5 GB | ~72 GB | rvLLM leaner (no PyTorch overhead) |
| **FP8 weight VRAM** | 7.0 GB | 7.0 GB | Both support FP8 E4M3 |
| **FP8 KV cache** | Supported | Supported | 2x KV capacity |

### CPU-Side Operations

Operations between GPU forward passes, measured on Apple M5 and Xeon:

| Operation | Rust | Python (numpy) | Speedup |
|---|---|---|---|
| Combined penalties (rep+freq+pres) | 2.6 us | 63 us | **24x** |
| Repetition penalty (2K tokens) | 3.1 us | 34 us | **11x** |
| Multinomial sampling (32K vocab) | 12 us | 66 us | **5.5x** |
| Top-P nucleus (128K vocab) | 1.6 ms | 6.9 ms | **4.3x** |
| Batch sampling (64 seqs, Rayon) | 4.3 ms | 36.4 ms | **8.5x** |

### Deployment

| Metric | rvLLM | Python vLLM |
|---|---|---|
| Install | `cargo install rvllm` | `pip install vllm` (+ PyTorch) |
| Container image | ~50 MB | ~15 GB |
| Build from source | 22 sec | N/A |
| Kernel compilation | 30 sec (46 PTX) | 0 (precompiled) or 60s (torch.compile) |
| GPU architectures | sm_80, sm_86, sm_89, sm_90 | Same + ROCm |

## Architecture

### Inference Pipeline

```
Request -> Tokenizer -> Scheduler -> GPU Forward -> Sampler -> Detokenizer -> Response
                            |              |
                     Continuous      CUDA Graph Replay
                     Batching       (13 pre-captured sizes)
                            |              |
                     Block Manager    Fused Kernels
                     (paged KV)      (5/layer T=1)
```

### CUDA Kernel Stack

46 hand-written CUDA kernels + LLVM NVPTX compiler for JIT fusion:

**Fused decode kernels (T=1, 5 kernels/layer):**
- `fused_add_norm_qkv_gemv` -- residual add + RMSNorm + QKV projection + bias (one kernel)
- `fused_rope_cache` -- RoPE + KV cache write
- `flash_attention_3_gqa` -- GQA-optimized decode attention (loads KV once per head group)
- `fused_oproj_add_norm_gateup_gemv` -- O-proj + residual add + RMSNorm + gate+up projection
- `fused_silu_down_gemv` -- SiLU activation + down projection

**Advanced kernels:**
- FP8 E4M3 weight GEMV (cublasLt FP8 + custom fused variants)
- TMA async-prefetch GEMV (double-buffered weight loads)
- WGMMA tensor core GEMV (wmma m16n16k16)
- Split-KV paged attention (for context >= 512)
- Persistent cooperative-groups layer kernel

**Compiler:**
- `rvllm-fusion` crate: IR -> pattern matching -> LLVM IR -> NVPTX PTX
- Direct PTX text emitter (fallback, no LLVM dependency)
- cuBLAS algorithm autotuning (32 candidates per GEMM shape)
- Kernel cache with SHA-256 keys

### Optimization History

| Phase | Change | 7B tok/s (N=128) | Date |
|---|---|---:|---|
| 1 | FP32 baseline | -- | Mar 28 |
| 2 | FP16 inference | 6,360 | Mar 28 |
| 3 | CUDA graph replay + cublasLt | 8,578 | Mar 28 |
| 4 | 8-agent kernel fusion swarm | 12,624 | Mar 29 |
| 5 | Deeper fusion + v4 vectorized loads | 12,800 | Mar 30 |
| 6 | Wire dead code paths (in progress) | ~15,000 (est) | Mar 30 |

### What's Inside

| Crate | Purpose |
|---|---|
| `rvllm-server` | HTTP API (axum), CLI |
| `rvllm-engine` | Async engine, continuous batching |
| `rvllm-worker` | GPU worker, CUDA graph management |
| `rvllm-model-runner` | Forward pass, weight loading |
| `rvllm-gpu` | CUDA abstractions, cuBLAS, kernel loader |
| `rvllm-fusion` | Kernel fusion IR, LLVM NVPTX compiler |
| `rvllm-kv-cache` | Paged KV cache (f16 + FP8) |
| `rvllm-attention` | Attention backends |
| `rvllm-speculative` | Speculative decoding (self-draft) |
| `rvllm-tp` | Tensor parallelism (NCCL) |
| `rvllm-tokenizer` | HuggingFace tokenizer wrapper |

## Install

```bash
# From crates.io
cargo install rvllm

# From PyPI
pip install rvllm
```

Or build from source:

```bash
git clone https://github.com/m0at/rvllm
cd rvllm
cargo build --release --features cuda
```

## Quick Start

```bash
# Serve Qwen2.5-7B
rvllm serve --model Qwen/Qwen2.5-7B --dtype half

# Benchmark (direct engine, no HTTP)
rvllm benchmark --model Qwen/Qwen2.5-7B --dtype half --n "1,4,16,64,128"

# With FP8 weights (halves VRAM for weights)
RVLLM_FP8_WEIGHTS=1 rvllm serve --model Qwen/Qwen2.5-7B --dtype half

# With FP8 KV cache (doubles max sequences)
RVLLM_FP8_KV=1 rvllm serve --model Qwen/Qwen2.5-7B --dtype half
```

## Benchmark Methodology

Both engines serve the same OpenAI-compatible `/v1/completions` endpoint. Direct engine benchmarks use the built-in `rvllm benchmark` command (no HTTP overhead). HTTP benchmarks use `bench/loadtest.py` (async Python client with aiohttp).

Each engine runs on its own vast.ai H100 SXM 80GB instance -- separate GPUs, clean CUDA state, no cross-contamination. Reproducible with `bench/compare_vllm.sh`.

See [docs/arch.md](docs/arch.md) for the full forward pass trace and [docs/benchmark-history.md](docs/benchmark-history.md) for detailed optimization history.
