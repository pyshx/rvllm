//! AttentionBackend trait definition and backend selection.

use crate::buffer::GpuBuffer;
use half::f16;
use rvllm_core::prelude::Result;

use crate::flash_attention_impl::FlashAttention2;
use crate::paged_attention::PagedAttentionV2;
use crate::split_kv::SplitKvAttention;

/// Trait for pluggable attention backends.
///
/// Implementations must be thread-safe (`Send + Sync`) so they can be shared
/// across worker threads.
pub trait AttentionBackend: Send + Sync {
    /// Run a single paged-attention forward pass.
    ///
    /// # Arguments
    /// * `query`           - `[num_tokens, num_heads, head_dim]`
    /// * `key_cache`       - `[num_blocks, block_size, num_heads, head_dim]`
    /// * `value_cache`     - same shape as key_cache
    /// * `block_tables`    - `[num_seqs, max_blocks_per_seq]`
    /// * `context_lens`    - `[num_seqs]`
    /// * `max_context_len` - maximum context length across all sequences
    /// * `scale`           - softmax scale factor (typically 1/sqrt(head_dim))
    fn forward(
        &self,
        query: &GpuBuffer<f16>,
        key_cache: &GpuBuffer<f16>,
        value_cache: &GpuBuffer<f16>,
        block_tables: &GpuBuffer<i32>,
        context_lens: &GpuBuffer<i32>,
        max_context_len: usize,
        scale: f32,
    ) -> Result<GpuBuffer<f16>>;

    /// Human-readable name for logging / debug.
    fn name(&self) -> &str;
}

/// Select the best attention backend for the given compute capability.
///
/// Returns `FlashAttention2` (CPU reference) for SM >= 8.0, `PagedAttentionV2` otherwise.
/// When the `cuda` feature is enabled, use `FlashAttention2::with_cuda()` directly
/// to get the GPU-accelerated kernel path.
/// Select the best attention backend for the given compute capability.
///
/// For SM >= 8.0 (Ampere+), uses SplitKvAttention which distributes KV blocks
/// across multiple thread blocks for better high-concurrency scaling.
/// Falls back to PagedAttentionV2 for older GPUs.
///
/// When the `cuda` feature is enabled, the CUDA-backed variants are used.
/// The `use_split_kv` parameter allows callers to force or disable split-KV.
pub fn select_backend(compute_capability: (u32, u32)) -> Box<dyn AttentionBackend> {
    select_backend_with_options(compute_capability, true)
}

/// Blackwell-specific attention configuration.
///
/// SM100/SM120 support 228 KB shared memory per SM, UMMA (Unified Matrix
/// Multiply-Accumulate) instructions, and 2-CTA cluster launch for
/// cooperative attention tiles.
#[cfg(feature = "cuda")]
pub struct BlackwellAttentionConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub cluster_ctas: usize,
    pub shared_mem_bytes: usize,
    pub use_umma: bool,
}

#[cfg(feature = "cuda")]
impl Default for BlackwellAttentionConfig {
    fn default() -> Self {
        Self {
            tile_m: 256,
            tile_n: 128,
            cluster_ctas: 2,
            shared_mem_bytes: 228 * 1024,
            use_umma: true,
        }
    }
}

fn is_blackwell(cc: (u32, u32)) -> bool {
    cc.0 >= 10
}

/// Select backend with explicit control over split-KV.
pub fn select_backend_with_options(
    compute_capability: (u32, u32),
    use_split_kv: bool,
) -> Box<dyn AttentionBackend> {
    let (major, minor) = compute_capability;

    if is_blackwell(compute_capability) {
        tracing::info!(
            backend = "SplitKvAttention",
            sm = %format!("{major}.{minor}"),
            blackwell = true,
            "selected Blackwell split-KV attention backend"
        );
        return Box::new(SplitKvAttention::new());
    }

    if major >= 8 && use_split_kv {
        tracing::info!(
            backend = "SplitKvAttention",
            sm = %format!("{major}.{minor}"),
            "selected split-KV attention backend"
        );
        Box::new(SplitKvAttention::new())
    } else if major >= 8 {
        tracing::info!(
            backend = "FlashAttention2",
            sm = %format!("{major}.{minor}"),
            "selected FlashAttention-2 backend"
        );
        Box::new(FlashAttention2::new())
    } else {
        tracing::info!(
            backend = "PagedAttentionV2",
            sm = %format!("{major}.{minor}"),
            "selected paged attention v2 backend"
        );
        Box::new(PagedAttentionV2::new())
    }
}

/// Context-length-aware backend selection for decode steps.
///
/// For short contexts (<= 1024 tokens), FlashAttention-2 is preferred since
/// the work fits in a single thread block. For longer contexts, SplitKV
/// distributes across blocks for better occupancy.
///
/// Blackwell (SM100+) always uses SplitKV -- the larger shared memory and
/// 2-CTA cluster launch make it faster even at short context lengths.
pub fn select_decode_backend(
    compute_capability: (u32, u32),
    max_context_len: usize,
) -> Box<dyn AttentionBackend> {
    let (major, _) = compute_capability;

    if is_blackwell(compute_capability) {
        return Box::new(SplitKvAttention::new());
    }

    if major < 8 {
        return Box::new(PagedAttentionV2::new());
    }
    if max_context_len <= 1024 {
        Box::new(FlashAttention2::new())
    } else {
        Box::new(SplitKvAttention::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_split_kv_for_sm80() {
        let backend = select_backend((8, 0));
        assert_eq!(backend.name(), "SplitKvAttention-CPU");
    }

    #[test]
    fn select_split_kv_for_sm90() {
        let backend = select_backend((9, 0));
        assert_eq!(backend.name(), "SplitKvAttention-CPU");
    }

    #[test]
    fn select_split_kv_for_sm120_blackwell() {
        let backend = select_backend((12, 0));
        assert_eq!(backend.name(), "SplitKvAttention-CPU");
    }

    #[test]
    fn select_split_kv_for_sm100_blackwell() {
        let backend = select_backend((10, 0));
        assert_eq!(backend.name(), "SplitKvAttention-CPU");
    }

    #[test]
    fn blackwell_decode_always_split_kv() {
        // Even at short context, Blackwell should use SplitKV
        let backend = select_decode_backend((10, 0), 128);
        assert_eq!(backend.name(), "SplitKvAttention-CPU");
        let backend = select_decode_backend((12, 0), 512);
        assert_eq!(backend.name(), "SplitKvAttention-CPU");
    }

    #[test]
    fn blackwell_ignores_split_kv_disable() {
        // Blackwell path takes priority over use_split_kv=false
        let backend = select_backend_with_options((10, 0), false);
        assert_eq!(backend.name(), "SplitKvAttention-CPU");
    }

    #[test]
    fn select_flash_when_split_kv_disabled() {
        let backend = select_backend_with_options((8, 0), false);
        assert_eq!(backend.name(), "FlashAttention2-CPU");
    }

    #[test]
    fn select_paged_for_sm75() {
        let backend = select_backend((7, 5));
        assert_eq!(backend.name(), "PagedAttentionV2");
    }

    #[test]
    fn backend_trait_is_object_safe() {
        fn _assert_object_safe(_: &dyn AttentionBackend) {}
    }
}
