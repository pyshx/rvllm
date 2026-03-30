use std::collections::HashMap;
use std::sync::Arc;

use tracing::{debug, info};

use crate::comm::TpCommunicator;
use crate::config::TpConfig;
use crate::parallel_layer::{TpAttentionLayout, TpLinearLayer, TpMlpLayout};
use crate::shard::{shard_shape, shard_weight, ShardStrategy, TransformerTpPlan};
use crate::{LLMError, Result};

/// Weight name patterns for a single transformer layer.
struct LayerWeightNames {
    qkv_proj: &'static [&'static str],
    o_proj: &'static str,
    gate_up_proj: &'static [&'static str],
    down_proj: &'static str,
}

const LLAMA_WEIGHT_NAMES: LayerWeightNames = LayerWeightNames {
    qkv_proj: &["q_proj.weight", "k_proj.weight", "v_proj.weight"],
    o_proj: "o_proj.weight",
    gate_up_proj: &["gate_proj.weight", "up_proj.weight"],
    down_proj: "down_proj.weight",
};

/// A sharded weight tensor: raw bytes + shape after sharding.
#[derive(Clone)]
pub struct ShardedWeight {
    pub data: Vec<u8>,
    pub shape: [usize; 2],
    pub strategy: ShardStrategy,
}

/// Raw weight store: maps weight names to (f16 bytes, [out_dim, in_dim]).
/// This is the input to TpModelRunner -- caller loads weights into this map
/// before constructing the runner.
pub struct WeightStore {
    pub weights: HashMap<String, (Vec<u8>, [usize; 2])>,
}

impl WeightStore {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, data: Vec<u8>, shape: [usize; 2]) {
        self.weights.insert(name, (data, shape));
    }

    pub fn get(&self, name: &str) -> Option<(&[u8], [usize; 2])> {
        self.weights.get(name).map(|(d, s)| (d.as_slice(), *s))
    }
}

/// Per-layer TP-sharded weights for a transformer block.
pub struct TpLayerWeights {
    pub attn: TpAttentionLayout,
    pub mlp: TpMlpLayout,
}

/// TP-aware model runner wrapping the per-rank forward pass.
///
/// Holds sharded weights and a communicator. On each forward step the caller
/// performs the local GEMM on its weight shard and calls `post_gemm_communicate`
/// on the appropriate `TpLinearLayer` to execute the all-reduce (row-parallel)
/// or no-op (column-parallel).
pub struct TpModelRunner {
    pub config: TpConfig,
    pub communicator: Arc<dyn TpCommunicator>,
    pub plan: TransformerTpPlan,
    pub layers: Vec<TpLayerWeights>,
    /// Replicated weights: embed_tokens, final norm, lm_head.
    pub embed_weight: Vec<u8>,
    pub norm_weight: Vec<u8>,
    pub lm_head_weight: Vec<u8>,
}

impl TpModelRunner {
    /// Build a TP runner by sharding raw host-side weight bytes.
    ///
    /// `store` holds all model weights keyed by name (e.g.
    /// "model.layers.0.self_attn.q_proj.weight") with their raw f16 bytes
    /// and [out_dim, in_dim] shapes.
    ///
    /// `num_layers` is the number of transformer layers.
    pub fn new(
        store: &WeightStore,
        num_layers: usize,
        tp_config: TpConfig,
        communicator: Arc<dyn TpCommunicator>,
    ) -> Result<Self> {
        let plan = TransformerTpPlan::standard();
        let rank = tp_config.rank;
        let ws = tp_config.world_size;

        info!(rank, world_size = ws, num_layers, "building TpModelRunner");

        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let prefix = format!("model.layers.{i}");

            // QKV: column-parallel (shard along output dim for each head matrix)
            let qkv_weight = shard_fused_weight(
                store,
                &LLAMA_WEIGHT_NAMES
                    .qkv_proj
                    .iter()
                    .map(|s| format!("{prefix}.self_attn.{s}"))
                    .collect::<Vec<_>>(),
                plan.qkv_proj,
                rank,
                ws,
            )?;

            // O-proj: row-parallel
            let o_name = format!("{prefix}.self_attn.{}", LLAMA_WEIGHT_NAMES.o_proj);
            let o_weight = shard_single_weight(store, &o_name, plan.o_proj, rank, ws)?;

            // Gate+Up: column-parallel
            let gate_up_weight = shard_fused_weight(
                store,
                &LLAMA_WEIGHT_NAMES
                    .gate_up_proj
                    .iter()
                    .map(|s| format!("{prefix}.mlp.{s}"))
                    .collect::<Vec<_>>(),
                plan.gate_up_proj,
                rank,
                ws,
            )?;

            // Down: row-parallel
            let down_name = format!("{prefix}.mlp.{}", LLAMA_WEIGHT_NAMES.down_proj);
            let down_weight = shard_single_weight(store, &down_name, plan.down_proj, rank, ws)?;

            let comm = Arc::clone(&communicator);

            let attn = TpAttentionLayout {
                qkv: TpLinearLayer::new(
                    qkv_weight.strategy,
                    qkv_weight.data,
                    qkv_weight.shape,
                    Arc::clone(&comm),
                ),
                o_proj: TpLinearLayer::new(
                    o_weight.strategy,
                    o_weight.data,
                    o_weight.shape,
                    Arc::clone(&comm),
                ),
            };

            let mlp = TpMlpLayout {
                gate_up: TpLinearLayer::new(
                    gate_up_weight.strategy,
                    gate_up_weight.data,
                    gate_up_weight.shape,
                    Arc::clone(&comm),
                ),
                down: TpLinearLayer::new(
                    down_weight.strategy,
                    down_weight.data,
                    down_weight.shape,
                    Arc::clone(&comm),
                ),
            };

            layers.push(TpLayerWeights { attn, mlp });
            debug!(layer = i, "sharded layer weights");
        }

        // Replicated weights
        let embed_weight = get_replicated(store, "model.embed_tokens.weight")?;
        let norm_weight = get_replicated(store, "model.norm.weight")?;
        let lm_head_weight = get_replicated(store, "lm_head.weight")
            .or_else(|_| get_replicated(store, "model.embed_tokens.weight"))?;

        info!(rank, "TpModelRunner ready");

        Ok(Self {
            config: tp_config,
            communicator,
            plan,
            layers,
            embed_weight,
            norm_weight,
            lm_head_weight,
        })
    }

    /// Execute post-GEMM communication for attention O-projection (row-parallel).
    /// Call this after the local GEMM for O-proj, before the residual add.
    pub fn allreduce_attn_output(
        &self,
        layer_idx: usize,
        output: &mut [u8],
        element_count: usize,
    ) -> Result<()> {
        self.layers[layer_idx]
            .attn
            .o_proj
            .post_gemm_communicate(output, element_count)?;
        Ok(())
    }

    /// Execute post-GEMM communication for MLP down-projection (row-parallel).
    /// Call this after the local GEMM for down-proj, before the residual add.
    pub fn allreduce_mlp_output(
        &self,
        layer_idx: usize,
        output: &mut [u8],
        element_count: usize,
    ) -> Result<()> {
        self.layers[layer_idx]
            .mlp
            .down
            .post_gemm_communicate(output, element_count)?;
        Ok(())
    }

    /// Barrier across all TP ranks.
    pub fn barrier(&self) -> Result<()> {
        self.communicator.barrier()
    }

    pub fn rank(&self) -> usize {
        self.config.rank
    }

    pub fn world_size(&self) -> usize {
        self.config.world_size
    }
}

/// Shard a single weight matrix.
fn shard_single_weight(
    store: &WeightStore,
    name: &str,
    strategy: ShardStrategy,
    rank: usize,
    world_size: usize,
) -> Result<ShardedWeight> {
    let (data, shape) = store.get(name).ok_or_else(|| {
        LLMError::ConfigError(format!("missing weight: {name}"))
    })?;
    let shard_data = shard_weight(data, &shape, 2, strategy, rank, world_size)?;
    let shard_sh = shard_shape(&shape, strategy, world_size);
    Ok(ShardedWeight {
        data: shard_data,
        shape: shard_sh,
        strategy,
    })
}

/// Shard and fuse multiple weight matrices (e.g. Q+K+V -> fused QKV).
/// Each sub-weight is sharded independently then concatenated along the output dim.
fn shard_fused_weight(
    store: &WeightStore,
    names: &[String],
    strategy: ShardStrategy,
    rank: usize,
    world_size: usize,
) -> Result<ShardedWeight> {
    let mut fused_data = Vec::new();
    let mut total_out = 0usize;
    let mut in_dim = 0usize;

    for name in names {
        let sw = shard_single_weight(store, name, strategy, rank, world_size)?;
        total_out += sw.shape[0];
        if in_dim == 0 {
            in_dim = sw.shape[1];
        } else if in_dim != sw.shape[1] {
            return Err(LLMError::ConfigError(format!(
                "fused weights have mismatched in_dim: {} vs {}",
                in_dim, sw.shape[1]
            )));
        }
        fused_data.extend_from_slice(&sw.data);
    }

    Ok(ShardedWeight {
        data: fused_data,
        shape: [total_out, in_dim],
        strategy,
    })
}

/// Get a replicated weight (no sharding, full copy).
fn get_replicated(store: &WeightStore, name: &str) -> Result<Vec<u8>> {
    let (data, _shape) = store.get(name).ok_or_else(|| {
        LLMError::ConfigError(format!("missing weight: {name}"))
    })?;
    Ok(data.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tp_worker::MockTpCommunicator;

    fn make_test_store(num_layers: usize) -> WeightStore {
        let mut store = WeightStore::new();
        let hidden = 8;
        let heads = 2;
        let head_dim = 4;
        let kv_heads = 2;
        let intermediate = 16;

        let q_dim = heads * head_dim;      // 8
        let kv_dim = kv_heads * head_dim;  // 8

        store.insert(
            "model.embed_tokens.weight".into(),
            vec![0u8; 32 * hidden * 2],
            [32, hidden],
        );
        store.insert(
            "model.norm.weight".into(),
            vec![0u8; hidden * 2],
            [hidden, 1],
        );
        store.insert(
            "lm_head.weight".into(),
            vec![0u8; 32 * hidden * 2],
            [32, hidden],
        );

        for i in 0..num_layers {
            let p = format!("model.layers.{i}");
            store.insert(
                format!("{p}.self_attn.q_proj.weight"),
                vec![1u8; q_dim * hidden * 2],
                [q_dim, hidden],
            );
            store.insert(
                format!("{p}.self_attn.k_proj.weight"),
                vec![2u8; kv_dim * hidden * 2],
                [kv_dim, hidden],
            );
            store.insert(
                format!("{p}.self_attn.v_proj.weight"),
                vec![3u8; kv_dim * hidden * 2],
                [kv_dim, hidden],
            );
            store.insert(
                format!("{p}.self_attn.o_proj.weight"),
                vec![4u8; hidden * q_dim * 2],
                [hidden, q_dim],
            );
            store.insert(
                format!("{p}.mlp.gate_proj.weight"),
                vec![5u8; intermediate * hidden * 2],
                [intermediate, hidden],
            );
            store.insert(
                format!("{p}.mlp.up_proj.weight"),
                vec![6u8; intermediate * hidden * 2],
                [intermediate, hidden],
            );
            store.insert(
                format!("{p}.mlp.down_proj.weight"),
                vec![7u8; hidden * intermediate * 2],
                [hidden, intermediate],
            );
        }
        store
    }

    #[test]
    fn tp1_runner_builds() {
        let store = make_test_store(2);
        let config = TpConfig::single_gpu();
        let comm: Arc<dyn TpCommunicator> = Arc::new(MockTpCommunicator::new(1, 0));
        let runner = TpModelRunner::new(&store, 2, config, comm).unwrap();
        assert_eq!(runner.layers.len(), 2);
    }

    #[test]
    fn tp2_runner_shards_correctly() {
        let store = make_test_store(1);
        let hidden = 8;

        for rank in 0..2 {
            let config = TpConfig::new(2, rank, crate::config::TpBackend::Nccl).unwrap();
            let comm: Arc<dyn TpCommunicator> = Arc::new(MockTpCommunicator::new(2, rank));
            let runner = TpModelRunner::new(&store, 1, config, comm).unwrap();
            let layer = &runner.layers[0];

            // QKV is column-parallel: in_dim halved
            assert_eq!(layer.attn.qkv.shard_shape[1], hidden / 2);
            // O-proj is row-parallel: out_dim halved (which is shape[0])
            assert_eq!(layer.attn.o_proj.shard_shape[0], hidden / 2);
            // Gate+Up is column-parallel: in_dim halved
            assert_eq!(layer.mlp.gate_up.shard_shape[1], hidden / 2);
            // Down is row-parallel: out_dim halved
            assert_eq!(layer.mlp.down.shard_shape[0], hidden / 2);
        }
    }

    #[test]
    fn allreduce_identity_tp1() {
        let store = make_test_store(1);
        let config = TpConfig::single_gpu();
        let comm: Arc<dyn TpCommunicator> = Arc::new(MockTpCommunicator::new(1, 0));
        let runner = TpModelRunner::new(&store, 1, config, comm).unwrap();

        let mut buf = vec![1u8, 0, 2, 0, 3, 0, 4, 0];
        let original = buf.clone();
        runner.allreduce_attn_output(0, &mut buf, 4).unwrap();
        assert_eq!(buf, original);

        runner.allreduce_mlp_output(0, &mut buf, 4).unwrap();
        assert_eq!(buf, original);
    }
}
