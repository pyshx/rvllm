// Half-precision block copy kernel for KV cache management.
// Copies f16 key and value cache blocks according to a mapping table.
//
// Launch config:
//   Grid:  (num_pairs, 1, 1)
//   Block: (min(block_size * num_heads * head_dim, 1024), 1, 1)
//   Shared memory: none

#include <cuda_fp16.h>

extern "C"
__global__ void copy_blocks_f16_kernel(
    __half* __restrict__ key_cache,         // [num_blocks, block_size, num_heads, head_dim]
    __half* __restrict__ value_cache,       // [num_blocks, block_size, num_heads, head_dim]
    const long* __restrict__ block_mapping, // [num_pairs, 2]
    int num_pairs,
    int block_size,
    int num_heads,
    int head_dim
) {
    const int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;

    const long src_block = block_mapping[pair_idx * 2];
    const long dst_block = block_mapping[pair_idx * 2 + 1];

    const int elems_per_block = block_size * num_heads * head_dim;
    const int src_offset = src_block * elems_per_block;
    const int dst_offset = dst_block * elems_per_block;

    for (int i = threadIdx.x; i < elems_per_block; i += blockDim.x) {
        key_cache[dst_offset + i] = key_cache[src_offset + i];
    }

    for (int i = threadIdx.x; i < elems_per_block; i += blockDim.x) {
        value_cache[dst_offset + i] = value_cache[src_offset + i];
    }
}
