//! CUDA linear (dense / GEMM) layer via cuBLAS.
//!
//! Implements `output[m,n] = input[m,k] @ weight^T[k,n] [+ bias]` where weight
//! is stored as `[n, k]` row-major (out_features x in_features), matching the
//! convention in `linear.rs`.
//!
//! This module is intended to be gated behind `#[cfg(feature = "cuda")]` in the
//! parent `mod.rs`. It delegates the unsafe cuBLAS call to `rvllm_gpu::cublas_ops`
//! so this crate's `#![forbid(unsafe_code)]` is respected.

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice as _, LaunchAsync, LaunchConfig};
use half::f16;
use rvllm_core::prelude::{LLMError, Result};
use rvllm_gpu::cublas::CublasHandle;
use rvllm_gpu::cublas_ops::CublasOps;
use std::sync::Arc;

/// GPU-accelerated dense linear projection using cuBLAS SGEMM.
///
/// Owns a `CublasOps` handle so cuBLAS init cost is amortized across calls.
pub struct CudaLinearLayer {
    ops: CublasOps,
}

impl CudaLinearLayer {
    /// Create a new layer bound to the given CUDA device.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self {
            ops: CublasOps::new(device)?,
        })
    }

    /// Convenience constructor sharing the device from an existing `CublasHandle`.
    pub fn from_handle(blas: &CublasHandle) -> Result<Self> {
        Self::new(blas.device().clone())
    }

    /// Compute `output[m,n] = input[m,k] @ weight^T[k,n] [+ bias]`.
    ///
    /// # Arguments
    /// * `input`  - `[m, k]` row-major activation tensor on GPU
    /// * `weight` - `[n, k]` row-major weight matrix on GPU
    /// * `bias`   - optional `[n]` bias vector on GPU
    /// * `m`      - number of tokens / rows in input
    /// * `n`      - output features (rows in weight)
    /// * `k`      - input features (cols in weight, cols in input)
    pub fn forward(
        &self,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaSlice<f32>> {
        if input.len() < m * k {
            return Err(LLMError::GpuError(format!(
                "CudaLinearLayer: input len {} < m*k = {}",
                input.len(),
                m * k
            )));
        }
        if weight.len() < n * k {
            return Err(LLMError::GpuError(format!(
                "CudaLinearLayer: weight len {} < n*k = {}",
                weight.len(),
                n * k
            )));
        }
        if let Some(b) = bias {
            if b.len() < n {
                return Err(LLMError::GpuError(format!(
                    "CudaLinearLayer: bias len {} < n = {}",
                    b.len(),
                    n
                )));
            }
        }

        let device = self.ops.device();

        // Allocate output [m, n]. If bias is present, tile it into every row so
        // sgemm accumulates on top with beta=1.
        let mut output: CudaSlice<f32> = if let Some(b) = bias {
            let bias_host = device
                .dtoh_sync_copy(b)
                .map_err(|e| LLMError::GpuError(format!("bias dtoh failed: {e}")))?;
            let mut tiled = Vec::with_capacity(m * n);
            for _ in 0..m {
                tiled.extend_from_slice(&bias_host[..n]);
            }
            device
                .htod_sync_copy(&tiled)
                .map_err(|e| LLMError::GpuError(format!("tiled bias htod failed: {e}")))?
        } else {
            device
                .alloc_zeros::<f32>(m * n)
                .map_err(|e| LLMError::GpuError(format!("output alloc failed: {e}")))?
        };

        let beta = if bias.is_some() { 1.0f32 } else { 0.0f32 };

        // C[m,n] = 1.0 * input[m,k] @ weight^T[k,n] + beta * C[m,n]
        self.ops
            .sgemm_a_bt(m, n, k, 1.0, input, weight, beta, &mut output)?;

        Ok(output)
    }

    /// Static forward matching the spec signature. Creates a temporary CublasOps;
    /// prefer the instance method [`Self::forward`] for repeated calls.
    pub fn forward_once(
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
    ) -> Result<CudaSlice<f32>> {
        let layer = Self::from_handle(blas)?;
        layer.forward(input, weight, bias, m, n, k)
    }

    /// Static forward with f16 weights: cast f32 input -> f16, hgemm, cast f16 output -> f32.
    ///
    /// Used for the LM head projection when `use_fp16` is enabled.
    pub fn forward_once_f16(
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f16>,
        m: usize,
        n: usize,
        k: usize,
        blas: &CublasHandle,
    ) -> Result<CudaSlice<f32>> {
        let device = blas.device();

        // Cast input f32 -> f16
        let input_f16 = Self::gpu_cast_f32_to_f16(device, input, m * k)?;

        // Allocate f16 output
        let mut output_f16 = device
            .alloc_zeros::<f16>(m * n)
            .map_err(|e| LLMError::GpuError(format!("forward_once_f16 alloc: {e}")))?;

        // hgemm: output = input @ weight^T
        blas.hgemm(
            m, n, k,
            f16::ONE,
            &input_f16,
            weight,
            f16::ZERO,
            &mut output_f16,
        )?;

        // Cast output f16 -> f32
        Self::gpu_cast_f16_to_f32(device, &output_f16, m * n)
    }

    fn gpu_cast_f32_to_f16(
        device: &Arc<CudaDevice>,
        input: &CudaSlice<f32>,
        n: usize,
    ) -> Result<CudaSlice<f16>> {
        let mut output = device
            .alloc_zeros::<f16>(n)
            .map_err(|e| LLMError::GpuError(format!("cast_f32_to_f16 alloc: {e}")))?;

        let kernel = device
            .get_func("cast_fp", "cast_f32_to_f16_kernel")
            .ok_or_else(|| LLMError::GpuError("cast_f32_to_f16_kernel not loaded".into()))?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(cfg, (&mut output, input, n as i32))
                .map_err(|e| LLMError::GpuError(format!("cast_f32_to_f16 launch: {e}")))?;
        }
        Ok(output)
    }

    fn gpu_cast_f16_to_f32(
        device: &Arc<CudaDevice>,
        input: &CudaSlice<f16>,
        n: usize,
    ) -> Result<CudaSlice<f32>> {
        let mut output = device
            .alloc_zeros::<f32>(n)
            .map_err(|e| LLMError::GpuError(format!("cast_f16_to_f32 alloc: {e}")))?;

        let kernel = device
            .get_func("cast_fp", "cast_f16_to_f32_kernel")
            .ok_or_else(|| LLMError::GpuError("cast_f16_to_f32_kernel not loaded".into()))?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel
                .launch(cfg, (&mut output, input, n as i32))
                .map_err(|e| LLMError::GpuError(format!("cast_f16_to_f32 launch: {e}")))?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    // Tests require the `cuda` feature and a real GPU.
    // Run with: cargo test -p rvllm-model-runner --features cuda
}
