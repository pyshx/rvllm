// Rotary Positional Embedding (RoPE) kernel (f32).
// Applies rotary embedding to query and key tensors in-place.
//
// Uses split-half convention: pair (x[i], x[i + half_dim]) for frequency index i.
//
// Launch config:
//   Grid:  (num_tokens, max(num_heads, num_kv_heads), 1)
//   Block: (head_dim / 2, 1, 1)   -- one thread per rotation pair
//   Shared memory: none

extern "C"
__global__ void rotary_embedding_kernel(
    float* __restrict__ query,           // [num_tokens, num_heads, head_dim]
    float* __restrict__ key,             // [num_tokens, num_kv_heads, head_dim]
    const float* __restrict__ cos_cache, // [max_position, head_dim / 2]
    const float* __restrict__ sin_cache, // [max_position, head_dim / 2]
    const int* __restrict__ positions,   // [num_tokens]
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

    // Apply to query -- split-half
    if (head_idx < num_heads) {
        const int base = (token_idx * num_heads + head_idx) * head_dim;
        float x0 = query[base + pair_idx];
        float x1 = query[base + half_dim + pair_idx];
        query[base + pair_idx]            = x0 * cos_val - x1 * sin_val;
        query[base + half_dim + pair_idx] = x0 * sin_val + x1 * cos_val;
    }

    // Apply to key -- split-half
    if (head_idx < num_kv_heads) {
        const int base = (token_idx * num_kv_heads + head_idx) * head_dim;
        float x0 = key[base + pair_idx];
        float x1 = key[base + half_dim + pair_idx];
        key[base + pair_idx]            = x0 * cos_val - x1 * sin_val;
        key[base + half_dim + pair_idx] = x0 * sin_val + x1 * cos_val;
    }
}
