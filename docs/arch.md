# rvLLM Architecture (April 4, 2026)

This is the current architecture summary after the April 4 dispatch and GEMM-policy fixes.

## End-to-End Flow

```text
HTTP / benchmark request
-> scheduler builds mixed prefill + decode batch
-> gpu_worker uploads metadata
-> gpu_runner selects forward path
-> model layers execute on GPU
-> lm head + sampling / argmax
-> token ids copied back
-> engine updates sequences
```

The dedicated GPU thread still owns CUDA state, graph capture, replay, and the runner.

## The Important Split: `T=1` vs `T>=2`

### `T=1`: batch-1 normal decode

The normal path is now:

```text
Batched
```

That was not true before. The path had already been moved off the old fused default, but it was still staying on the older single-token family instead of the reusable batched scratch path.

### `T>=2`: batched decode and prefill

The normal path is now an explicit per-op hybrid GEMM policy:

```text
QKV / O / down  -> cuBLAS or cublasLt
GateUp + SiLU   -> CUTLASS
```

This is the current best policy on H100 for Qwen2.5-7B.

## What Was Wrong Before

Two architectural problems were masking the real performance picture:

1. **Wrong batch-1 default**
   - The standard `T=1` path still used the legacy single-token stack.
   - Fix: normal `T=1` now defaults to `ForwardPath::Batched`, which keeps reusable scratch buffers alive and avoids per-layer output churn.

2. **Half-wired batched hybrid policy**
   - The runner wanted a hybrid strategy, but the actual enum and dispatch did not enforce one.
   - CUTLASS presence could change QKV routing even when that was not the intended policy.
   - Fix: `GemmStrategy::Hybrid` is now real and stable.

## Current Layer Stack

### Batch-1 normal decode

```text
RMSNorm
QKV via cuBLAS / cublasLt
RoPE + KV cache write
attention decode
O-proj via cuBLAS / cublasLt
RMSNorm
GateUp + SiLU via CUTLASS
down via cuBLAS / cublasLt
```

This uses the same reusable scratch buffers as the batched path instead of the older single-token allocator-heavy route.

### Batched decode / prefill

```text
RMSNorm
QKV via cuBLAS / cublasLt
RoPE + cache update
attention backend
O-proj via cuBLAS / cublasLt
residual + RMSNorm
GateUp + SiLU via CUTLASS
down via cuBLAS / cublasLt
```

## Current Comparison vs vLLM 0.19.0

Same H100, same Qwen2.5-7B snapshot, direct engine, `output-len=128`:

| N | vLLM 0.19.0 | rvLLM | rvLLM / vLLM |
|---:|---:|---:|---:|
| 1 | 165.5 | 133.1 | 0.80x |
| 32 | 4467.7 | 4407.5 | 0.99x |
| 64 | 7972.1 | 8038.0 | 1.01x |
| 128 | 13903.5 | 13110.1 | 0.94x |

So the architecture story is now:

- single-stream decode still needs work
- batched decode is nearly there
- the current public docs should be talking about the batched hybrid stack, not the old fused-default story

## Relevant Files

- `crates/rvllm-model-runner/src/gpu_runner.rs`
- `crates/rvllm-model-runner/src/gpu_layer/mod.rs`
- `crates/rvllm-model-runner/src/gpu_layer/batched.rs`
- `crates/rvllm-worker/src/gpu_worker.rs`

## Remaining Work

- make the `cublasLt` autotune cache degrade safely on bad cached algos
- keep closing the `N=1` gap
- improve the `N=128` gap without regressing `N=32/64`
