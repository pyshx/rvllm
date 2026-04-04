//! Profile-guided kernel autotuning via CUPTI.
//!
//! Profiles the actual inference step, identifies the slowest kernels,
//! sweeps configurations, and persists optimal configs to disk.
//!
//! Architecture:
//! 1. `CuptiProfiler` captures per-kernel timing via CUPTI activity callbacks
//! 2. `KernelRanker` ranks kernels by total GPU time
//! 3. `ConfigSweeper` tries alternative launch configs for the top-N kernels
//! 4. `AutotuneCache` persists winning configs keyed by (GPU, kernel, shape)
//!
//! The profiler uses CUPTI's activity API (not callback API) for minimal
//! overhead: kernels complete normally, timing records are collected
//! asynchronously from a ring buffer after synchronization.

pub mod profiler;
pub mod ranker;
pub mod sweeper;
pub mod cache;

pub use profiler::{CuptiProfiler, KernelRecord};
pub use ranker::{KernelRanker, RankedKernel};
pub use sweeper::{ConfigSweeper, SweepResult, LaunchCandidate};
pub use cache::{TuneCache, TuneCacheKey, TunedConfig};
