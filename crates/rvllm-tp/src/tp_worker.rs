use std::sync::Arc;
use std::thread;

use tracing::{debug, info};

use crate::comm::TpCommunicator;
use crate::config::TpConfig;
use crate::tp_runner::TpModelRunner;
use crate::{LLMError, Result};

/// Multi-GPU tensor-parallel worker.
///
/// Spawns one thread per rank, each holding a `TpModelRunner` with its sharded
/// weights. On each forward step every rank computes its local GEMMs and the
/// runner inserts all-reduce calls at the correct points (after O-proj and
/// down-proj).
pub struct TpWorker {
    pub config: TpConfig,
    pub runners: Vec<Arc<TpModelRunner>>,
}

/// Per-rank handle returned by the worker.
pub struct RankHandle {
    pub rank: usize,
    pub runner: Arc<TpModelRunner>,
}

impl TpWorker {
    /// Build a TP worker group.
    ///
    /// `build_runner` is called once per rank to construct the `TpModelRunner`
    /// for that rank (loading + sharding weights). This allows the caller to
    /// control weight loading (from disk, from GPU, etc.).
    pub fn new<F>(config: TpConfig, build_runner: F) -> Result<Self>
    where
        F: Fn(usize) -> Result<TpModelRunner> + Send + Sync,
    {
        info!(
            world_size = config.world_size,
            "initializing TpWorker group"
        );

        let mut runners = Vec::with_capacity(config.world_size);
        for rank in 0..config.world_size {
            let runner = build_runner(rank)?;
            debug!(rank, "rank runner built");
            runners.push(Arc::new(runner));
        }

        Ok(Self { config, runners })
    }

    /// Get the runner for a specific rank.
    pub fn rank_runner(&self, rank: usize) -> &Arc<TpModelRunner> {
        &self.runners[rank]
    }

    /// Get a handle for each rank (useful for spawning per-rank threads).
    pub fn rank_handles(&self) -> Vec<RankHandle> {
        self.runners
            .iter()
            .enumerate()
            .map(|(rank, runner)| RankHandle {
                rank,
                runner: Arc::clone(runner),
            })
            .collect()
    }

    /// Execute a function on all ranks in parallel (one thread per rank).
    /// Returns results in rank order.
    pub fn parallel_execute<F, T>(&self, f: F) -> Result<Vec<T>>
    where
        F: Fn(&TpModelRunner) -> Result<T> + Send + Sync + 'static,
        T: Send + 'static,
    {
        let f = Arc::new(f);
        let handles: Vec<_> = self
            .runners
            .iter()
            .map(|runner| {
                let runner = Arc::clone(runner);
                let f = Arc::clone(&f);
                thread::spawn(move || f(&runner))
            })
            .collect();

        let mut results = Vec::with_capacity(handles.len());
        for (rank, h) in handles.into_iter().enumerate() {
            let res = h
                .join()
                .map_err(|_| LLMError::GpuError(format!("rank {rank} thread panicked")))?;
            results.push(res?);
        }
        Ok(results)
    }

    pub fn world_size(&self) -> usize {
        self.config.world_size
    }
}

/// Mock communicator for single-GPU testing.
///
/// All-reduce is identity, all-gather copies input to output, barrier is a no-op.
/// This allows the full TP code path to be exercised without multiple GPUs.
pub struct MockTpCommunicator {
    world_size: usize,
    rank: usize,
}

impl MockTpCommunicator {
    pub fn new(world_size: usize, rank: usize) -> Self {
        Self { world_size, rank }
    }

    pub fn single() -> Self {
        Self {
            world_size: 1,
            rank: 0,
        }
    }
}

impl TpCommunicator for MockTpCommunicator {
    fn all_reduce_sum_f16(&self, _buf: &mut [u8], _count: usize) -> Result<()> {
        // Identity: with one rank (or mock), the buffer already has the full result.
        Ok(())
    }

    fn all_gather_f16(
        &self,
        input: &[u8],
        output: &mut [u8],
        _send_count: usize,
    ) -> Result<()> {
        // Single rank: output == input
        let len = input.len().min(output.len());
        output[..len].copy_from_slice(&input[..len]);
        Ok(())
    }

    fn reduce_scatter_f16(
        &self,
        input: &[u8],
        output: &mut [u8],
        _recv_count: usize,
    ) -> Result<()> {
        let len = input.len().min(output.len());
        output[..len].copy_from_slice(&input[..len]);
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        Ok(())
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn rank(&self) -> usize {
        self.rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tp_runner::WeightStore;

    fn make_test_store(num_layers: usize) -> WeightStore {
        let mut store = WeightStore::new();
        let hidden = 8;
        let heads = 2;
        let head_dim = 4;
        let kv_heads = 2;
        let intermediate = 16;

        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        store.insert("model.embed_tokens.weight".into(), vec![0u8; 32 * hidden * 2], [32, hidden]);
        store.insert("model.norm.weight".into(), vec![0u8; hidden * 2], [hidden, 1]);
        store.insert("lm_head.weight".into(), vec![0u8; 32 * hidden * 2], [32, hidden]);

        for i in 0..num_layers {
            let p = format!("model.layers.{i}");
            store.insert(format!("{p}.self_attn.q_proj.weight"), vec![1u8; q_dim * hidden * 2], [q_dim, hidden]);
            store.insert(format!("{p}.self_attn.k_proj.weight"), vec![2u8; kv_dim * hidden * 2], [kv_dim, hidden]);
            store.insert(format!("{p}.self_attn.v_proj.weight"), vec![3u8; kv_dim * hidden * 2], [kv_dim, hidden]);
            store.insert(format!("{p}.self_attn.o_proj.weight"), vec![4u8; hidden * q_dim * 2], [hidden, q_dim]);
            store.insert(format!("{p}.mlp.gate_proj.weight"), vec![5u8; intermediate * hidden * 2], [intermediate, hidden]);
            store.insert(format!("{p}.mlp.up_proj.weight"), vec![6u8; intermediate * hidden * 2], [intermediate, hidden]);
            store.insert(format!("{p}.mlp.down_proj.weight"), vec![7u8; hidden * intermediate * 2], [hidden, intermediate]);
        }
        store
    }

    #[test]
    fn mock_communicator_identity() {
        let comm = MockTpCommunicator::single();
        let mut buf = vec![10u8, 20, 30, 40];
        comm.all_reduce_sum_f16(&mut buf, 2).unwrap();
        assert_eq!(buf, vec![10, 20, 30, 40]);
    }

    #[test]
    fn mock_all_gather() {
        let comm = MockTpCommunicator::single();
        let input = vec![1u8, 2, 3, 4];
        let mut output = vec![0u8; 4];
        comm.all_gather_f16(&input, &mut output, 2).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn mock_barrier() {
        let comm = MockTpCommunicator::single();
        comm.barrier().unwrap();
    }

    #[test]
    fn tp_worker_single_rank() {
        let store = Arc::new(make_test_store(1));
        let config = TpConfig::single_gpu();

        let worker = TpWorker::new(config.clone(), |rank| {
            let comm: Arc<dyn TpCommunicator> = Arc::new(MockTpCommunicator::new(1, rank));
            let tp_cfg = TpConfig::new(1, rank, crate::config::TpBackend::Nccl).unwrap();
            TpModelRunner::new(&store, 1, tp_cfg, comm)
        })
        .unwrap();

        assert_eq!(worker.world_size(), 1);
        assert_eq!(worker.rank_runner(0).rank(), 0);
    }

    #[test]
    fn tp_worker_parallel_execute() {
        let store = Arc::new(make_test_store(1));
        let config = TpConfig::single_gpu();

        let worker = TpWorker::new(config, |rank| {
            let comm: Arc<dyn TpCommunicator> = Arc::new(MockTpCommunicator::new(1, rank));
            let tp_cfg = TpConfig::new(1, rank, crate::config::TpBackend::Nccl).unwrap();
            TpModelRunner::new(&store, 1, tp_cfg, comm)
        })
        .unwrap();

        let results = worker
            .parallel_execute(|runner| Ok(runner.rank()))
            .unwrap();
        assert_eq!(results, vec![0]);
    }

    #[test]
    fn tp2_worker_builds() {
        let store = Arc::new(make_test_store(1));
        let config = TpConfig::new(2, 0, crate::config::TpBackend::Nccl).unwrap();

        let worker = TpWorker::new(config, |rank| {
            let comm: Arc<dyn TpCommunicator> = Arc::new(MockTpCommunicator::new(2, rank));
            let tp_cfg = TpConfig::new(2, rank, crate::config::TpBackend::Nccl).unwrap();
            TpModelRunner::new(&store, 1, tp_cfg, comm)
        })
        .unwrap();

        assert_eq!(worker.world_size(), 2);

        let results = worker
            .parallel_execute(|runner| Ok(runner.rank()))
            .unwrap();
        assert_eq!(results, vec![0, 1]);
    }

    #[test]
    fn tp2_allreduce_after_row_parallel() {
        let store = Arc::new(make_test_store(1));
        let config = TpConfig::new(2, 0, crate::config::TpBackend::Nccl).unwrap();

        let worker = TpWorker::new(config, |rank| {
            let comm: Arc<dyn TpCommunicator> = Arc::new(MockTpCommunicator::new(2, rank));
            let tp_cfg = TpConfig::new(2, rank, crate::config::TpBackend::Nccl).unwrap();
            TpModelRunner::new(&store, 1, tp_cfg, comm)
        })
        .unwrap();

        let results = worker
            .parallel_execute(|runner| {
                let mut buf = vec![42u8, 0, 99, 0];
                runner.allreduce_attn_output(0, &mut buf, 2)?;
                Ok(buf)
            })
            .unwrap();

        assert_eq!(results[0], vec![42, 0, 99, 0]);
        assert_eq!(results[1], vec![42, 0, 99, 0]);
    }
}
