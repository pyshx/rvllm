// Half-precision Rotary Positional Embedding (RoPE) kernel.
// Q/K are f16, cos/sin caches remain f32. Computes rotation in f32 for precision,
// reads/writes f16 to halve memory bandwidth.
//
// Uses split-half convention: pair (x[i], x[i + half_dim]) for frequency index i.
// This matches Llama, Qwen2, Mistral, and most modern transformer architectures.
//
// Launch config:
//   Grid:  (num_tokens, max(num_heads, num_kv_heads), 1)
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
    const int half_dim = head_dim / 2;
    if (pair_idx >= half_dim) return;

    const int pos = positions[token_idx];
    const float cos_val = cos_cache[pos * half_dim + pair_idx];
    const float sin_val = sin_cache[pos * half_dim + pair_idx];

    // Apply to query -- split-half: pair (q[i], q[i + half_dim])
    if (head_idx < num_heads) {
        const int base = (token_idx * num_heads + head_idx) * head_dim;
        float x0 = __half2float(query[base + pair_idx]);
        float x1 = __half2float(query[base + half_dim + pair_idx]);
        query[base + pair_idx]            = __float2half(x0 * cos_val - x1 * sin_val);
        query[base + half_dim + pair_idx] = __float2half(x0 * sin_val + x1 * cos_val);
    }

    // Apply to key -- split-half: pair (k[i], k[i + half_dim])
    if (head_idx < num_kv_heads) {
        const int base = (token_idx * num_kv_heads + head_idx) * head_dim;
        float x0 = __half2float(key[base + pair_idx]);
        float x1 = __half2float(key[base + half_dim + pair_idx]);
        key[base + pair_idx]            = __float2half(x0 * cos_val - x1 * sin_val);
        key[base + half_dim + pair_idx] = __float2half(x0 * sin_val + x1 * cos_val);
    }
}
