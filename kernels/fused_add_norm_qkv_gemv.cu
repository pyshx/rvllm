// Fused residual-add + RMSNorm + QKV GEMV kernel for M=1 decode.
// f16 I/O, f32 accumulation. Each block handles RPB=8 output rows.
// Phase 2 uses warp-per-row: 8 warps compute 8 rows in parallel.
//
// 4 variants: {add+norm, norm-only} x {no-bias, with-bias}
//
// Launch config:
//   Grid:  ((qkv_dim + RPB - 1) / RPB, 1, 1)
//   Block: (256, 1, 1)
//   Shared mem: hidden_size * sizeof(float) + RPB * sizeof(float)

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define THREADS 256
#define RPB 8

__device__ __forceinline__ float warp_reduce_sum_anqg(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// --------------------------------------------------------------------------
// Fused add + RMSNorm + QKV GEMV (layers 1..N where residual add is needed)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_add_norm_qkv_gemv(
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    const __half* __restrict__ proj_weight,
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)input;
    const half2* add2 = (const half2*)add_vec;

    for (int i = tid; i < h2; i += THREADS) {
        half2 a = in2[i];
        half2 b = add2[i];
        float v0 = __half2float(a.x) + __half2float(b.x);
        float v1 = __half2float(a.y) + __half2float(b.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float v = __half2float(input[hidden_size - 1]) + __half2float(add_vec[hidden_size - 1]);
        s_normed[hidden_size - 1] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    // Phase 2: GEMV -- warp-per-row, 8 warps = 8 rows in parallel
    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const half2* w2 = (const half2*)(proj_weight + (long long)row * hidden_size);
            float acc = 0.0f;
            for (int i = lane_id; i < h2; i += 32) {
                half2 w = w2[i];
                acc += __half2float(w.x) * s_normed[i * 2] + __half2float(w.y) * s_normed[i * 2 + 1];
            }
            if ((hidden_size & 1) && lane_id == 0)
                acc += __half2float(proj_weight[row * hidden_size + hidden_size - 1]) * s_normed[hidden_size - 1];
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc);
        }
    }
}

// --------------------------------------------------------------------------
// Fused add + RMSNorm + QKV GEMV + bias (models with QKV bias like Qwen2.5)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_add_norm_qkv_bias_gemv(
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    const __half* __restrict__ proj_weight,
    const __half* __restrict__ bias,        // [qkv_dim]
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)input;
    const half2* add2 = (const half2*)add_vec;

    for (int i = tid; i < h2; i += THREADS) {
        half2 a = in2[i];
        half2 b = add2[i];
        float v0 = __half2float(a.x) + __half2float(b.x);
        float v1 = __half2float(a.y) + __half2float(b.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float v = __half2float(input[hidden_size - 1]) + __half2float(add_vec[hidden_size - 1]);
        s_normed[hidden_size - 1] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    // Phase 2: GEMV + bias -- warp-per-row
    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const half2* w2 = (const half2*)(proj_weight + (long long)row * hidden_size);
            float acc = 0.0f;
            for (int i = lane_id; i < h2; i += 32) {
                half2 w = w2[i];
                acc += __half2float(w.x) * s_normed[i * 2] + __half2float(w.y) * s_normed[i * 2 + 1];
            }
            if ((hidden_size & 1) && lane_id == 0)
                acc += __half2float(proj_weight[row * hidden_size + hidden_size - 1]) * s_normed[hidden_size - 1];
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc + __half2float(bias[row]));
        }
    }
}

// --------------------------------------------------------------------------
// First-layer variant: RMSNorm + QKV GEMV (no residual add)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_norm_qkv_gemv(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ norm_weight,
    const __half* __restrict__ proj_weight,
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)input;

    for (int i = tid; i < h2; i += THREADS) {
        half2 a = in2[i];
        float v0 = __half2float(a.x);
        float v1 = __half2float(a.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float v = __half2float(input[hidden_size - 1]);
        s_normed[hidden_size - 1] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const half2* w2 = (const half2*)(proj_weight + (long long)row * hidden_size);
            float acc = 0.0f;
            for (int i = lane_id; i < h2; i += 32) {
                half2 w = w2[i];
                acc += __half2float(w.x) * s_normed[i * 2] + __half2float(w.y) * s_normed[i * 2 + 1];
            }
            if ((hidden_size & 1) && lane_id == 0)
                acc += __half2float(proj_weight[row * hidden_size + hidden_size - 1]) * s_normed[hidden_size - 1];
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc);
        }
    }
}

// --------------------------------------------------------------------------
// First-layer variant: RMSNorm + QKV GEMV + bias (no residual add)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_norm_qkv_bias_gemv(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ norm_weight,
    const __half* __restrict__ proj_weight,
    const __half* __restrict__ bias,        // [qkv_dim]
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)input;

    for (int i = tid; i < h2; i += THREADS) {
        half2 a = in2[i];
        float v0 = __half2float(a.x);
        float v1 = __half2float(a.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float v = __half2float(input[hidden_size - 1]);
        s_normed[hidden_size - 1] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const half2* w2 = (const half2*)(proj_weight + (long long)row * hidden_size);
            float acc = 0.0f;
            for (int i = lane_id; i < h2; i += 32) {
                half2 w = w2[i];
                acc += __half2float(w.x) * s_normed[i * 2] + __half2float(w.y) * s_normed[i * 2 + 1];
            }
            if ((hidden_size & 1) && lane_id == 0)
                acc += __half2float(proj_weight[row * hidden_size + hidden_size - 1]) * s_normed[hidden_size - 1];
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc + __half2float(bias[row]));
        }
    }
}

// ==========================================================================
// FP8 variants: weights stored as E4M3 bytes with per-row scale
// ==========================================================================

// --------------------------------------------------------------------------
// FP8: Fused add + RMSNorm + QKV GEMV (layers 1..N)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_add_norm_qkv_fp8_gemv(
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    const unsigned char* __restrict__ proj_weight,
    const __half* __restrict__ weight_scale,
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)input;
    const half2* add2 = (const half2*)add_vec;

    for (int i = tid; i < h2; i += THREADS) {
        half2 a = in2[i];
        half2 b = add2[i];
        float v0 = __half2float(a.x) + __half2float(b.x);
        float v1 = __half2float(a.y) + __half2float(b.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float v = __half2float(input[hidden_size - 1]) + __half2float(add_vec[hidden_size - 1]);
        s_normed[hidden_size - 1] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const unsigned char* w_row = proj_weight + (long long)row * hidden_size;
            float row_sc = __half2float(weight_scale[row]);
            float acc = 0.0f;
            const int h4 = hidden_size / 4;
            for (int i = lane_id; i < h4; i += 32) {
                int base = i * 4;
                unsigned int packed = *reinterpret_cast<const unsigned int*>(w_row + base);
                unsigned char b0 = packed & 0xFF;
                unsigned char b1 = (packed >> 8) & 0xFF;
                unsigned char b2 = (packed >> 16) & 0xFF;
                unsigned char b3 = (packed >> 24) & 0xFF;
                float w0 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b0)) * row_sc;
                float w1 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b1)) * row_sc;
                float w2 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b2)) * row_sc;
                float w3 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b3)) * row_sc;
                acc += w0 * s_normed[base]
                     + w1 * s_normed[base + 1]
                     + w2 * s_normed[base + 2]
                     + w3 * s_normed[base + 3];
            }
            for (int i = h4 * 4 + lane_id; i < hidden_size; i += 32) {
                unsigned char b = w_row[i];
                float w = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b)) * row_sc;
                acc += w * s_normed[i];
            }
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc);
        }
    }
}

// --------------------------------------------------------------------------
// FP8: First-layer variant -- RMSNorm + QKV GEMV (no residual add)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_norm_qkv_fp8_gemv(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ norm_weight,
    const unsigned char* __restrict__ proj_weight,
    const __half* __restrict__ weight_scale,
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)input;

    for (int i = tid; i < h2; i += THREADS) {
        half2 a = in2[i];
        float v0 = __half2float(a.x);
        float v1 = __half2float(a.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float v = __half2float(input[hidden_size - 1]);
        s_normed[hidden_size - 1] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const unsigned char* w_row = proj_weight + (long long)row * hidden_size;
            float row_sc = __half2float(weight_scale[row]);
            float acc = 0.0f;
            const int h4 = hidden_size / 4;
            for (int i = lane_id; i < h4; i += 32) {
                int base = i * 4;
                unsigned int packed = *reinterpret_cast<const unsigned int*>(w_row + base);
                unsigned char b0 = packed & 0xFF;
                unsigned char b1 = (packed >> 8) & 0xFF;
                unsigned char b2 = (packed >> 16) & 0xFF;
                unsigned char b3 = (packed >> 24) & 0xFF;
                float w0 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b0)) * row_sc;
                float w1 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b1)) * row_sc;
                float w2 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b2)) * row_sc;
                float w3 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b3)) * row_sc;
                acc += w0 * s_normed[base]
                     + w1 * s_normed[base + 1]
                     + w2 * s_normed[base + 2]
                     + w3 * s_normed[base + 3];
            }
            for (int i = h4 * 4 + lane_id; i < hidden_size; i += 32) {
                unsigned char b = w_row[i];
                float w = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b)) * row_sc;
                acc += w * s_normed[i];
            }
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc);
        }
    }
}

// --------------------------------------------------------------------------
// FP8: Fused add + RMSNorm + QKV GEMV + bias
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_add_norm_qkv_fp8_bias_gemv(
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    const unsigned char* __restrict__ proj_weight,
    const __half* __restrict__ weight_scale,
    const __half* __restrict__ bias,
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)input;
    const half2* add2 = (const half2*)add_vec;

    for (int i = tid; i < h2; i += THREADS) {
        half2 a = in2[i];
        half2 b = add2[i];
        float v0 = __half2float(a.x) + __half2float(b.x);
        float v1 = __half2float(a.y) + __half2float(b.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float v = __half2float(input[hidden_size - 1]) + __half2float(add_vec[hidden_size - 1]);
        s_normed[hidden_size - 1] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const unsigned char* w_row = proj_weight + (long long)row * hidden_size;
            float row_sc = __half2float(weight_scale[row]);
            float acc = 0.0f;
            const int h4 = hidden_size / 4;
            for (int i = lane_id; i < h4; i += 32) {
                int base = i * 4;
                unsigned int packed = *reinterpret_cast<const unsigned int*>(w_row + base);
                unsigned char b0 = packed & 0xFF;
                unsigned char b1 = (packed >> 8) & 0xFF;
                unsigned char b2 = (packed >> 16) & 0xFF;
                unsigned char b3 = (packed >> 24) & 0xFF;
                float w0 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b0)) * row_sc;
                float w1 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b1)) * row_sc;
                float w2 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b2)) * row_sc;
                float w3 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b3)) * row_sc;
                acc += w0 * s_normed[base]
                     + w1 * s_normed[base + 1]
                     + w2 * s_normed[base + 2]
                     + w3 * s_normed[base + 3];
            }
            for (int i = h4 * 4 + lane_id; i < hidden_size; i += 32) {
                unsigned char b = w_row[i];
                float w = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b)) * row_sc;
                acc += w * s_normed[i];
            }
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc + __half2float(bias[row]));
        }
    }
}

// --------------------------------------------------------------------------
// FP8: First-layer variant -- RMSNorm + QKV GEMV + bias (no residual add)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_norm_qkv_fp8_bias_gemv(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ norm_weight,
    const unsigned char* __restrict__ proj_weight,
    const __half* __restrict__ weight_scale,
    const __half* __restrict__ bias,
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)input;

    for (int i = tid; i < h2; i += THREADS) {
        half2 a = in2[i];
        float v0 = __half2float(a.x);
        float v1 = __half2float(a.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float v = __half2float(input[hidden_size - 1]);
        s_normed[hidden_size - 1] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    for (int i = tid; i < hidden_size; i += THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    __syncthreads();

    {
        const int row = block_base + warp_id;
        if (row < qkv_dim) {
            const unsigned char* w_row = proj_weight + (long long)row * hidden_size;
            float row_sc = __half2float(weight_scale[row]);
            float acc = 0.0f;
            const int h4 = hidden_size / 4;
            for (int i = lane_id; i < h4; i += 32) {
                int base = i * 4;
                unsigned int packed = *reinterpret_cast<const unsigned int*>(w_row + base);
                unsigned char b0 = packed & 0xFF;
                unsigned char b1 = (packed >> 8) & 0xFF;
                unsigned char b2 = (packed >> 16) & 0xFF;
                unsigned char b3 = (packed >> 24) & 0xFF;
                float w0 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b0)) * row_sc;
                float w1 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b1)) * row_sc;
                float w2 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b2)) * row_sc;
                float w3 = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b3)) * row_sc;
                acc += w0 * s_normed[base]
                     + w1 * s_normed[base + 1]
                     + w2 * s_normed[base + 2]
                     + w3 * s_normed[base + 3];
            }
            for (int i = h4 * 4 + lane_id; i < hidden_size; i += 32) {
                unsigned char b = w_row[i];
                float w = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b)) * row_sc;
                acc += w * s_normed[i];
            }
            acc = warp_reduce_sum_anqg(acc);
            if (lane_id == 0) output[row] = __float2half(acc + __half2float(bias[row]));
        }
    }
}
