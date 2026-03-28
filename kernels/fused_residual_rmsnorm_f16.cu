// Half-precision fused residual add + RMS normalization kernel.
// Reads/writes f16, accumulates in f32 for numerical stability.
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size, 1024), 1, 1)
//   Shared memory: blockDim.x * sizeof(float)

#include <cuda_fp16.h>

extern "C"
__global__ void fused_residual_rmsnorm_f16_kernel(
    __half* __restrict__ output,        // [num_tokens, hidden_size]
    __half* __restrict__ residual,      // [num_tokens, hidden_size]
    const __half* __restrict__ input,   // [num_tokens, hidden_size]
    const __half* __restrict__ add,     // [num_tokens, hidden_size]
    const __half* __restrict__ weight,  // [hidden_size]
    float eps,
    int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = token_idx * hidden_size;

    extern __shared__ float sdata[];

    // Step 1: fused residual add + sum-of-squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = __half2float(input[row_offset + i]) + __half2float(add[row_offset + i]);
        residual[row_offset + i] = __float2half(val);
        local_ss += val * val;
    }
    sdata[tid] = local_ss;
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms_scale = rsqrtf(sdata[0] / (float)hidden_size + eps);

    // Step 2: normalize and scale
    for (int i = tid; i < hidden_size; i += stride) {
        float val = __half2float(residual[row_offset + i]) * __half2float(weight[i]) * rms_scale;
        output[row_offset + i] = __float2half(val);
    }
}
