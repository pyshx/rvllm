//! Ranks kernels by optimization potential.
//!
//! Given profiling aggregates, identifies which kernels have the most
//! room for improvement based on total GPU time and distance from
//! theoretical peak.

use crate::profiler::KernelAggregate;

/// A kernel ranked by optimization potential.
#[derive(Debug, Clone)]
pub struct RankedKernel {
    /// Kernel name.
    pub name: String,
    /// Total GPU time in microseconds.
    pub total_us: f64,
    /// Average per-invocation time in microseconds.
    pub avg_us: f64,
    /// Number of invocations.
    pub count: u64,
    /// Percentage of total GPU time.
    pub pct_gpu: f64,
    /// Grid and block config.
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
    /// Is this kernel tunable (not cuBLAS/cuDNN internal)?
    pub tunable: bool,
    /// Category (gemm, attention, norm, activation, memory, other).
    pub category: KernelCategory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelCategory {
    Gemm,
    Attention,
    Norm,
    Activation,
    Memory,
    Sampling,
    Other,
}

impl KernelCategory {
    fn classify(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("gemm") || lower.contains("gemv") || lower.contains("hgemm")
            || lower.contains("cublas") || lower.contains("splitk")
        {
            Self::Gemm
        } else if lower.contains("attention") || lower.contains("attn") || lower.contains("fa3")
            || lower.contains("flash")
        {
            Self::Attention
        } else if lower.contains("norm") || lower.contains("rmsnorm") || lower.contains("layernorm") {
            Self::Norm
        } else if lower.contains("silu") || lower.contains("gelu") || lower.contains("activation") {
            Self::Activation
        } else if lower.contains("memcpy") || lower.contains("memset") || lower.contains("copy")
            || lower.contains("reshape") || lower.contains("cache")
        {
            Self::Memory
        } else if lower.contains("argmax") || lower.contains("softmax") || lower.contains("sample") {
            Self::Sampling
        } else {
            Self::Other
        }
    }
}

/// Ranks profiled kernels by optimization potential.
pub struct KernelRanker;

impl KernelRanker {
    /// Rank kernels from aggregated profiling data.
    ///
    /// Returns kernels sorted by total GPU time descending, with metadata
    /// about tunability and category.
    pub fn rank(aggregates: &[KernelAggregate]) -> Vec<RankedKernel> {
        let total_gpu_ns: u64 = aggregates.iter().map(|a| a.total_ns).sum();

        let mut result: Vec<RankedKernel> = aggregates
            .iter()
            .map(|a| {
                let category = KernelCategory::classify(&a.name);
                let tunable = Self::is_tunable(&a.name, category);
                RankedKernel {
                    name: a.name.clone(),
                    total_us: a.total_us(),
                    avg_us: a.avg_us(),
                    count: a.count,
                    pct_gpu: if total_gpu_ns > 0 {
                        a.total_ns as f64 / total_gpu_ns as f64 * 100.0
                    } else {
                        0.0
                    },
                    grid: a.grid,
                    block: a.block,
                    shared_mem: a.shared_mem,
                    tunable,
                    category,
                }
            })
            .collect();
        result.sort_by(|a, b| b.total_us.partial_cmp(&a.total_us).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Filter to only tunable kernels (our kernels, not cuBLAS internals).
    pub fn tunable_only(ranked: &[RankedKernel]) -> Vec<&RankedKernel> {
        ranked.iter().filter(|r| r.tunable).collect()
    }

    /// Get the top-N kernels by total GPU time.
    pub fn top_n(ranked: &[RankedKernel], n: usize) -> Vec<&RankedKernel> {
        ranked.iter().take(n).collect()
    }

    /// Kernels that are ours (tunable) vs library internals.
    fn is_tunable(name: &str, category: KernelCategory) -> bool {
        // cuBLAS internal kernels start with specific prefixes
        if name.starts_with("void cublas") || name.starts_with("void cublasLt") {
            return false;
        }
        // cuDNN kernels
        if name.contains("cudnn") {
            return false;
        }
        // NCCL kernels
        if name.contains("nccl") || name.contains("ncclKernel") {
            return false;
        }
        // Our kernels and any custom CUDA kernel are tunable
        matches!(category, KernelCategory::Attention | KernelCategory::Norm
            | KernelCategory::Activation | KernelCategory::Memory
            | KernelCategory::Sampling | KernelCategory::Other)
    }

    /// Print a ranked summary with tunability markers.
    pub fn print_ranked(ranked: &[RankedKernel], top_n: usize) {
        println!("{:<55} {:>6} {:>9} {:>9} {:>7} {:>5} {:>10}",
            "Kernel", "Count", "Total(ms)", "Avg(us)", "% GPU", "Tune", "Category");
        println!("{}", "-".repeat(110));

        for (i, r) in ranked.iter().enumerate() {
            if i >= top_n { break; }
            let tune_marker = if r.tunable { " YES" } else { "  --" };
            let cat = format!("{:?}", r.category);
            println!("{:<55} {:>6} {:>9.2} {:>9.1} {:>6.1}% {:>5} {:>10}",
                &r.name[..r.name.len().min(55)],
                r.count,
                r.total_us / 1000.0,
                r.avg_us,
                r.pct_gpu,
                tune_marker,
                cat,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiler::KernelAggregate;

    #[test]
    fn classify_kernels() {
        assert_eq!(KernelCategory::classify("fa3_v3_decode_gqa_kernel"), KernelCategory::Attention);
        assert_eq!(KernelCategory::classify("fused_residual_rmsnorm_f16_kernel"), KernelCategory::Norm);
        assert_eq!(KernelCategory::classify("silu_mul_interleaved_f16_kernel"), KernelCategory::Activation);
        assert_eq!(KernelCategory::classify("void cublasLt_xxx"), KernelCategory::Gemm);
        assert_eq!(KernelCategory::classify("reshape_and_cache_f16io_kernel"), KernelCategory::Memory);
        assert_eq!(KernelCategory::classify("argmax_kernel"), KernelCategory::Sampling);
    }

    #[test]
    fn rank_sorts_by_total() {
        let aggs = vec![
            KernelAggregate {
                name: "small".into(), count: 10, total_ns: 1000,
                min_ns: 80, max_ns: 120, grid: (1,1,1), block: (128,1,1), shared_mem: 0,
            },
            KernelAggregate {
                name: "big".into(), count: 5, total_ns: 50000,
                min_ns: 9000, max_ns: 11000, grid: (32,1,1), block: (256,1,1), shared_mem: 4096,
            },
        ];
        let ranked = KernelRanker::rank(&aggs);
        assert_eq!(ranked[0].name, "big");
        assert_eq!(ranked[1].name, "small");
    }

    #[test]
    fn cublas_not_tunable() {
        let aggs = vec![
            KernelAggregate {
                name: "void cublasLt_gemm_xxx".into(), count: 100, total_ns: 100000,
                min_ns: 900, max_ns: 1100, grid: (1,1,1), block: (128,1,1), shared_mem: 0,
            },
        ];
        let ranked = KernelRanker::rank(&aggs);
        assert!(!ranked[0].tunable);
    }

    #[test]
    fn our_kernels_tunable() {
        let aggs = vec![
            KernelAggregate {
                name: "fa3_v3_decode_gqa_kernel".into(), count: 28, total_ns: 800000,
                min_ns: 25000, max_ns: 30000, grid: (32,4,1), block: (256,1,1), shared_mem: 32768,
            },
        ];
        let ranked = KernelRanker::rank(&aggs);
        assert!(ranked[0].tunable);
        assert_eq!(ranked[0].category, KernelCategory::Attention);
    }
}
