# rvLLM-lite

rvLLM's serving architecture + vLLM's model execution internals.

## Architecture

```
rvLLM-lite serving stack (Python)
  api/         -- OpenAI-compatible axum-style server (uvicorn + fastapi)
  engine/      -- AsyncGpuLLMEngine pattern: dedicated GPU thread, async request drain
  scheduler/   -- Continuous batching, chunked prefill, preemption (recompute/swap)
  executor/    -- Single/multi-GPU dispatch, tensor parallel coordination
  worker/      -- Per-GPU worker owning model runner + KV cache

vLLM internals (imported as library)
  torch.compile     -- compiled model forward pass
  flash_attn        -- FA3 attention backend
  model zoo         -- HF model implementations (Qwen, Llama, Mistral, ...)
  weight loading    -- safetensors, sharded loading, dtype casting
  sampling          -- top-k, top-p, temperature, repetition penalty
```

## What rvLLM-lite owns (our code)

- Request lifecycle: accept, queue, schedule, dispatch, stream back
- Continuous batching scheduler with chunked prefill
- KV cache block management and preemption
- Dedicated GPU thread (not asyncio task) for forward passes
- OpenAI-compatible API with streaming SSE
- Telemetry, health checks, batch API

## What comes from vLLM (imported)

- Model forward pass (torch.compile'd transformer layers)
- Flash Attention 3 kernels
- Model weight loading and HF config parsing
- Tensor parallel weight sharding
- Sampling kernels

## Quick start

```bash
pip install -e ".[dev]"
python -m rvllm_lite --model Qwen/Qwen3-8B --port 8000
```
