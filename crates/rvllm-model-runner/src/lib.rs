#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
//! Transformer forward pass for vllm-rs.
//!
//! Provides `ModelRunner` which orchestrates the forward pass through
//! embedding, transformer layers, and the final LM head. Layer primitives
//! (norms, linear, rotary, activations, MLP) are naive CPU mock
//! implementations; real GPU kernels dispatch to CUDA.

pub mod architectures;
pub mod bridge;
pub mod input;
pub mod layers;
pub mod runner;

// GPU forward-pass modules (CUDA-only)
#[cfg(feature = "cuda")]
pub mod gpu_layer;
#[cfg(feature = "cuda")]
pub mod gpu_runner;
#[cfg(feature = "cuda")]
pub mod megakernel;

/// Type alias for cublasLt handle. Compiles to a usable type with the
/// `cublaslt` feature, or a zero-size dummy without it. This lets
/// function signatures reference it unconditionally.
#[cfg(feature = "cublaslt")]
pub type CublasLtRef = rvllm_gpu::cublaslt_ops::CublasLtOps;
#[cfg(not(feature = "cublaslt"))]
pub type CublasLtRef = ();

pub use architectures::{create_model, Architecture};
pub use input::ModelInput;
pub use runner::{ModelRunner, ModelRunnerConfig};
