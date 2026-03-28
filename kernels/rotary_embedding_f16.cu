// Half-precision Rotary Positional Embedding (RoPE) kernel.
// Q/K are f16, cos/sin caches remain f32. Computes rotation in f32 for precision,
// reads/writes f16 to halve memory bandwidth.
//
// Launch config:
//   Grid:  (num_tokens, num_heads, 1)
//   Block: (head_dim / 2, 1, 1)   -- one thread per rotation pair
//   Shared memory: none

#include <cuda_fp16.h>

extern "C"
__global__ void rotary_embedding_f16_kernel(
    __half* __restrict__ query,           // [num_tokens, num_heads, head_dim]
    __half* __restrict__ key,             // [num_tokens, num_kv_heads, head_dim]
    const float* __restrict__ cos_cache,  // [max_position, head_dim / 2]
    const float* __restrict__ sin_cache,  // [max_position, head_dim / 2]
    const int* __restrict__ positions,    // [num_tokens]
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int pair_idx  = threadIdx.x;

    if (token_idx >= num_tokens) return;
    if (pair_idx >= head_dim / 2) return;

    const int pos = positions[token_idx];
    const int half_dim = head_dim / 2;

    const float cos_val = cos_cache[pos * half_dim + pair_idx];
    const float sin_val = sin_cache[pos * half_dim + pair_idx];

    // Apply to query
    {
        const int base = (token_idx * num_heads + head_idx) * head_dim;
        const int i0 = base + 2 * pair_idx;
        const int i1 = base + 2 * pair_idx + 1;

        float x0 = __half2float(query[i0]);
        float x1 = __half2float(query[i1]);
        query[i0] = __float2half(x0 * cos_val - x1 * sin_val);
        query[i1] = __float2half(x0 * sin_val + x1 * cos_val);
    }

    // Apply to key (only if this head maps to a KV head, for GQA support)
    if (head_idx < num_kv_heads) {
        const int base = (token_idx * num_kv_heads + head_idx) * head_dim;
        const int i0 = base + 2 * pair_idx;
        const int i1 = base + 2 * pair_idx + 1;

        float x0 = __half2float(key[i0]);
        float x1 = __half2float(key[i1]);
        key[i0] = __float2half(x0 * cos_val - x1 * sin_val);
        key[i1] = __float2half(x0 * sin_val + x1 * cos_val);
    }
}
