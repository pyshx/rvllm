//! Configuration sweeper for tunable kernels.
//!
//! Given a kernel identified by the ranker, generates alternative launch
//! configurations and benchmarks each one to find the fastest.

use serde::{Deserialize, Serialize};
use tracing::debug;

/// A candidate launch configuration to benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchCandidate {
    /// Block dimensions (x, y, z).
    pub block: (u32, u32, u32),
    /// Shared memory in bytes.
    pub shared_mem: u32,
    /// Any kernel-specific tuning parameter (e.g., tile size, unroll factor).
    pub extra: Vec<(String, u32)>,
}

/// Result of benchmarking a single candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    /// The candidate configuration.
    pub candidate: LaunchCandidate,
    /// Average kernel time in nanoseconds.
    pub avg_ns: u64,
    /// Minimum kernel time in nanoseconds.
    pub min_ns: u64,
    /// Number of benchmark iterations run.
    pub iterations: u32,
    /// Whether this candidate produced correct results.
    pub correct: bool,
}

/// Generates and evaluates launch configurations for a given kernel.
pub struct ConfigSweeper {
    /// Standard block sizes to try.
    block_sizes: Vec<u32>,
    /// Number of warmup iterations before timing.
    warmup_iters: u32,
    /// Number of timed iterations.
    bench_iters: u32,
}

impl Default for ConfigSweeper {
    fn default() -> Self {
        Self {
            block_sizes: vec![64, 128, 256, 512],
            warmup_iters: 5,
            bench_iters: 50,
        }
    }
}

impl ConfigSweeper {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_block_sizes(mut self, sizes: Vec<u32>) -> Self {
        self.block_sizes = sizes;
        self
    }

    pub fn with_iterations(mut self, warmup: u32, bench: u32) -> Self {
        self.warmup_iters = warmup;
        self.bench_iters = bench;
        self
    }

    /// Generate candidate configurations for a kernel based on its category.
    pub fn candidates_for(
        &self,
        kernel_name: &str,
        current_block: (u32, u32, u32),
        current_shared: u32,
    ) -> Vec<LaunchCandidate> {
        let mut candidates = Vec::new();

        // Always include the current config as baseline
        candidates.push(LaunchCandidate {
            block: current_block,
            shared_mem: current_shared,
            extra: vec![],
        });

        // Sweep block sizes (1D blocks for most kernels)
        for &bs in &self.block_sizes {
            if bs == current_block.0 {
                continue;
            }
            candidates.push(LaunchCandidate {
                block: (bs, 1, 1),
                shared_mem: current_shared,
                extra: vec![],
            });
        }

        // For attention kernels: sweep shared memory sizes
        let lower = kernel_name.to_lowercase();
        if lower.contains("attention") || lower.contains("attn")
            || lower.contains("fa3") || lower.contains("flash") || lower.contains("gqa")
        {
            for &smem_kb in &[16, 32, 48, 64, 96, 128, 164, 228] {
                let smem = smem_kb * 1024;
                if smem == current_shared {
                    continue;
                }
                candidates.push(LaunchCandidate {
                    block: current_block,
                    shared_mem: smem,
                    extra: vec![("smem_kb".into(), smem_kb)],
                });
            }
        }

        // For GEMV kernels: sweep rows-per-block
        if lower.contains("gemv") {
            for &rpb in &[1, 2, 4, 8, 16, 32] {
                candidates.push(LaunchCandidate {
                    block: current_block,
                    shared_mem: current_shared,
                    extra: vec![("rows_per_block".into(), rpb)],
                });
            }
        }

        debug!(
            kernel = kernel_name,
            num_candidates = candidates.len(),
            "generated sweep candidates"
        );
        candidates
    }

    /// Find the best configuration from a set of sweep results.
    pub fn best(results: &[SweepResult]) -> Option<&SweepResult> {
        results
            .iter()
            .filter(|r| r.correct)
            .min_by_key(|r| r.min_ns)
    }

    /// Calculate speedup of best vs baseline (first result).
    pub fn speedup(results: &[SweepResult]) -> Option<f64> {
        if results.is_empty() {
            return None;
        }
        let baseline = results[0].avg_ns as f64;
        let best = Self::best(results)?;
        Some(baseline / best.avg_ns as f64)
    }

    /// Print sweep results.
    pub fn print_results(kernel_name: &str, results: &[SweepResult]) {
        println!("\nSweep results for: {kernel_name}");
        println!("{:<30} {:>10} {:>10} {:>8}",
            "Config", "Avg(us)", "Min(us)", "Status");
        println!("{}", "-".repeat(65));

        for r in results {
            let config = format!("block=({},{},{}), smem={}",
                r.candidate.block.0, r.candidate.block.1, r.candidate.block.2,
                r.candidate.shared_mem);
            let status = if r.correct { "OK" } else { "WRONG" };
            println!("{:<30} {:>10.1} {:>10.1} {:>8}",
                config,
                r.avg_ns as f64 / 1000.0,
                r.min_ns as f64 / 1000.0,
                status,
            );
        }

        if let Some(speedup) = Self::speedup(results) {
            println!("Best speedup vs baseline: {speedup:.2}x");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_sweeper() {
        let s = ConfigSweeper::new();
        assert_eq!(s.block_sizes, vec![64, 128, 256, 512]);
        assert_eq!(s.warmup_iters, 5);
        assert_eq!(s.bench_iters, 50);
    }

    #[test]
    fn candidates_include_baseline() {
        let s = ConfigSweeper::new();
        let candidates = s.candidates_for("my_kernel", (128, 1, 1), 4096);
        assert!(candidates.iter().any(|c| c.block == (128, 1, 1) && c.shared_mem == 4096));
    }

    #[test]
    fn attention_kernel_gets_smem_sweep() {
        let s = ConfigSweeper::new();
        let candidates = s.candidates_for("fa3_v3_decode_gqa_kernel", (256, 1, 1), 32768);
        // Sweep generates smem_kb values [16,32,48,64,96,128,164,228] * 1024
        // current_shared=32768 (32KB) is skipped since it matches
        let smem_vals: Vec<u32> = candidates.iter().map(|c| c.shared_mem).collect();
        assert!(smem_vals.contains(&(48 * 1024)), "missing 48KB: {smem_vals:?}");
        assert!(smem_vals.contains(&(96 * 1024)), "missing 96KB: {smem_vals:?}");
    }

    #[test]
    fn gemv_kernel_gets_rpb_sweep() {
        let s = ConfigSweeper::new();
        let candidates = s.candidates_for("fused_gemv_bias_3584x4608", (128, 1, 1), 0);
        let rpb_candidates: Vec<_> = candidates.iter()
            .filter(|c| c.extra.iter().any(|(k, _)| k == "rows_per_block"))
            .collect();
        assert!(!rpb_candidates.is_empty());
    }

    #[test]
    fn best_picks_fastest_correct() {
        let results = vec![
            SweepResult {
                candidate: LaunchCandidate { block: (128,1,1), shared_mem: 0, extra: vec![] },
                avg_ns: 1000, min_ns: 900, iterations: 50, correct: true,
            },
            SweepResult {
                candidate: LaunchCandidate { block: (256,1,1), shared_mem: 0, extra: vec![] },
                avg_ns: 800, min_ns: 750, iterations: 50, correct: true,
            },
            SweepResult {
                candidate: LaunchCandidate { block: (512,1,1), shared_mem: 0, extra: vec![] },
                avg_ns: 600, min_ns: 500, iterations: 50, correct: false, // wrong results
            },
        ];
        let best = ConfigSweeper::best(&results).unwrap();
        assert_eq!(best.candidate.block, (256, 1, 1)); // 512 excluded (incorrect)
    }
}
