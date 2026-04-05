# rvllm-model-runner: Decode Paths and GEMM Policy

The model runner owns the actual forward-path selection for decode and batched execution. Its core rule is still **no silent fallback**: if a path is selected, it must run that path or fail loud.

## Decode Path Architecture

The runner selects a `ForwardPath` from batch size, weight format, available kernels, and env vars.

```
Request arrives
    |
    v
T == 1? ----yes----> Check explicit experimental env vars first
    |                     |
    no                    v
    |              FP8 weights? --------yes--> Fp8Decode
    |                     |
    v                     no
Batched path              |
Hybrid/CUBLAS/CUTLASS     v
                     Batched (default)
                     CublasGemvDecode if RVLLM_BATCHED_DECODE_1=0
                     Legacy FusedDecode only if forced
```

## What Was Wrong Before

Two things were out of date:

1. Batch-1 normal decode still defaulted to the older single-token family.
2. Batched execution had a conceptual "hybrid" policy, but the actual `GemmStrategy` could not encode it cleanly, so QKV routing was inconsistent.

That meant the docs said one thing, the runner thought another thing, and the real per-op policy could still drift based on whether CUTLASS happened to be loaded.

## Current Batch-1 Decode

### Batched (default, `T=1`)

This is now the normal batch-1 decode path.

Per-layer shape:
```
RMSNorm
cuBLAS / cublasLt QKV projection
RoPE + KV cache write
Flash attention decode
cuBLAS / cublasLt O-proj
RMSNorm
GateUp + SiLU via CUTLASS
cuBLAS / cublasLt down
```

Latest verified number on H100 with Qwen2.5-7B f16:
- **133.1 tok/s**

This replaced the older single-token default because it wins end-to-end on the real benchmark while reusing scratch buffers across the layer loop.

### CublasGemvDecode (legacy single-token path)

This is still available when `RVLLM_BATCHED_DECODE_1=0`, and it remains better than the older fused path, but it is no longer the current default.

### FusedDecode (legacy, `RVLLM_BATCHED_DECODE_1=0 RVLLM_CUBLAS_DECODE=0`)

The older fused f16 GEMV path is still present, but it is no longer the default batch-1 choice.

Per-layer shape:
```
residual + RMSNorm + QKV GEMV
RoPE + KV cache write
attention decode
O-proj GEMV
residual + RMSNorm + GateUp GEMV
SiLU * Mul + Down GEMV
```

This remains useful as an explicit comparison path and as a base for kernel work, but not as the current default.

### Megakernel / Persistent / FP8

- **MegakernelDecode**: experimental instruction-tape path, still around `~50 tok/s` on the measured H100 run.
- **PersistentDecode / PersistentV3Decode**: experimental runner-level paths for custom decode kernels.
- **Fp8Decode**: selected when FP8 weights are present.

## Current Batched Policy (`T >= 2`)

The important fix was making batched GEMM routing explicit.

### `GemmStrategy::Hybrid` (default when CUTLASS is available)

Hybrid now means exactly this:

- **QKV**: cuBLAS / cublasLt
- **O-proj**: cuBLAS / cublasLt
- **GateUp + SiLU**: CUTLASS
- **Down-proj**: cuBLAS / cublasLt

This is the best current policy on the measured H100 Qwen2.5-7B run.

Latest verified `output-len=128` numbers:
- `N=32`: **4407.5 tok/s**
- `N=64`: **8038.0 tok/s**
- `N=128`: **13110.1 tok/s**

### Explicit batched strategy override

Use:

```bash
RVLLM_BATCHED_GEMM_STRATEGY=cublas
RVLLM_BATCHED_GEMM_STRATEGY=hybrid
RVLLM_BATCHED_GEMM_STRATEGY=cutlass
```

The latest explicit H100 sweep showed:
- `cublas`: slower than `hybrid`
- `hybrid`: best current policy
- `cutlass`: slower than `hybrid`

### cublasLt routing

Within the cuBLAS-backed ops, the runner still uses:
- `cublasLt` for smaller `M`
- cuBLAS for larger `M`

Autotuned cublasLt algorithm selection still benchmarks 32 candidates per shape and caches results on disk.

## Kernel Loader

## Kernel Loader

The `KernelLoader` (`rvllm-gpu/src/kernel_loader.rs`) manages ~60 CUDA kernel modules with convention-based function name resolution:

```
activation/         -- silu_mul, gelu, swiglu
attention/          -- flash_attention_v3, gqa_attention, split_kv_attention
cache/              -- reshape_and_cache, copy_blocks
decode/             -- fused decode kernels (add_rmsnorm_qkv_gemv, etc.)
embedding/          -- token_embedding_lookup
fp8/                -- fp8_quantize, fp8_dequantize
gemv/               -- hgemv_f16, gemv_int4
norm/               -- rmsnorm_f16, layernorm_f16
rope/               -- rotary_embedding, rope_cache_write
sampling/           -- top_k_sampling, argmax
```

`validate_required_kernels()` checks at startup that all kernels needed by the selected decode path are present. Missing kernels cause a hard error, not a silent fallback.

## CUDA Graph Capture

The `GraphRunner` (`rvllm-worker/src/graph_runner.rs`) pre-captures CUDA graphs for 35 common batch sizes to eliminate kernel launch overhead in the decode loop:

```
Batch sizes: 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, ...
```

For batch sizes without an exact graph, the runner pads up to the next captured size and unpads the logits afterward. Graph capture includes the full layer stack: GEMMs, attention, and sampling prep.

## Architecture Registration

New model architectures are registered in `architectures/mod.rs` via a match arm mapping the HuggingFace `architectures` field from `config.json` to a Rust struct implementing the `Architecture` trait:

```rust
pub trait Architecture: Send + Sync {
    fn forward(
        &self,
        input: &ModelInput,
        cache: &mut [KVCache],
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>>;
}
```

Each architecture struct owns its weights and implements the full forward pass: embedding lookup, per-layer QKV projections, rotary embeddings, attention, MLP (dense or MoE), residual connections, final norm, and LM head projection.
