//! Gemma4ForCausalLM architecture.
//!
//! Gemma 4 is a sparse MoE model with interleaved dense and MoE layers,
//! per-layer sliding window attention, shared experts, Gemma-style RMSNorm
//! (+1 offset), RoPE, and SiLU activation. Dense layers use a standard
//! gated MLP while MoE layers route through top-k experts plus a shared expert.

use half::f16;
use tracing::trace;

use crate::bridge::{AttentionBackend, CacheEngine, GpuBuffer, ModelWeights, Result};
use crate::input::ModelInput;
use crate::layers::linear::LinearLayer;
use crate::layers::moe::{ExpertFFN, MoELayer};
use crate::layers::rotary::RotaryEmbedding;
use crate::runner::ModelRunnerConfig;

use super::llama::{add_inplace, embed_tokens, get_or_zeros, lm_head};
use super::Architecture;

const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;

// ---------------------------------------------------------------------------
// Gemma-style RMSNorm: weight applied as (1 + w) instead of w
// ---------------------------------------------------------------------------

struct GemmaRMSNorm;

impl GemmaRMSNorm {
    #[inline]
    fn forward(
        input: &GpuBuffer<f16>,
        weight: &GpuBuffer<f16>,
        eps: f32,
    ) -> crate::bridge::Result<GpuBuffer<f16>> {
        let hidden = weight.len();
        let num_tokens = input.len() / hidden;
        let total = num_tokens * hidden;
        let mut out = vec![f16::ZERO; total];
        let w_f32: Vec<f32> = weight.data.iter().map(|v| 1.0 + v.to_f32()).collect();

        for t in 0..num_tokens {
            let start = t * hidden;
            let row = &input.data[start..start + hidden];
            let dst = &mut out[start..start + hidden];

            let mut sum_sq = 0.0f32;
            let chunks = row.chunks_exact(8);
            let remainder = chunks.remainder();
            for chunk in chunks {
                let a0 = chunk[0].to_f32();
                let a1 = chunk[1].to_f32();
                let a2 = chunk[2].to_f32();
                let a3 = chunk[3].to_f32();
                let a4 = chunk[4].to_f32();
                let a5 = chunk[5].to_f32();
                let a6 = chunk[6].to_f32();
                let a7 = chunk[7].to_f32();
                sum_sq +=
                    a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7;
            }
            for v in remainder {
                let f = v.to_f32();
                sum_sq += f * f;
            }

            let inv_rms = (sum_sq / hidden as f32 + eps).sqrt().recip();

            let row_chunks = row.chunks_exact(8);
            let row_rem = row_chunks.remainder();
            let w_chunks = w_f32.chunks_exact(8);
            let w_rem = w_chunks.remainder();
            let d_chunks = dst.chunks_exact_mut(8);

            for ((r, w), d) in row_chunks.zip(w_chunks).zip(d_chunks) {
                d[0] = f16::from_f32(r[0].to_f32() * inv_rms * w[0]);
                d[1] = f16::from_f32(r[1].to_f32() * inv_rms * w[1]);
                d[2] = f16::from_f32(r[2].to_f32() * inv_rms * w[2]);
                d[3] = f16::from_f32(r[3].to_f32() * inv_rms * w[3]);
                d[4] = f16::from_f32(r[4].to_f32() * inv_rms * w[4]);
                d[5] = f16::from_f32(r[5].to_f32() * inv_rms * w[5]);
                d[6] = f16::from_f32(r[6].to_f32() * inv_rms * w[6]);
                d[7] = f16::from_f32(r[7].to_f32() * inv_rms * w[7]);
            }
            let rem_start = hidden - row_rem.len();
            for (i, (rv, wv)) in row_rem.iter().zip(w_rem.iter()).enumerate() {
                dst[rem_start + i] = f16::from_f32(rv.to_f32() * inv_rms * wv);
            }
        }

        Ok(GpuBuffer::from_vec(out, vec![num_tokens, hidden]))
    }
}

// ---------------------------------------------------------------------------
// Gemma embedding scaling: embed * sqrt(hidden_size)
// ---------------------------------------------------------------------------

fn gemma_embed_tokens(embed: &GpuBuffer<f16>, token_ids: &[u32], hidden: usize) -> GpuBuffer<f16> {
    let mut buf = embed_tokens(embed, token_ids, hidden);
    let scale = (hidden as f32).sqrt();
    for v in buf.data.iter_mut() {
        *v = f16::from_f32(v.to_f32() * scale);
    }
    buf
}

// ---------------------------------------------------------------------------
// SiLU-gated MLP for dense layers
// ---------------------------------------------------------------------------

fn silu_mlp_forward(
    input: &GpuBuffer<f16>,
    gate_weight: &GpuBuffer<f16>,
    up_weight: &GpuBuffer<f16>,
    down_weight: &GpuBuffer<f16>,
) -> crate::bridge::Result<GpuBuffer<f16>> {
    let gate = LinearLayer::forward(input, gate_weight, None)?;
    let up = LinearLayer::forward(input, up_weight, None)?;
    let fused = crate::layers::activation::fused_silu_mul(&gate.data, &up.data);
    let fused_buf = GpuBuffer::from_vec(fused, gate.shape);
    LinearLayer::forward(&fused_buf, down_weight, None)
}

// ===========================================================================
// Layer types
// ===========================================================================

struct Gemma4DenseLayer {
    input_layernorm: GpuBuffer<f16>,
    post_attention_layernorm: GpuBuffer<f16>,
    pre_feedforward_layernorm: GpuBuffer<f16>,
    post_feedforward_layernorm: GpuBuffer<f16>,
    q_proj: GpuBuffer<f16>,
    k_proj: GpuBuffer<f16>,
    v_proj: GpuBuffer<f16>,
    o_proj: GpuBuffer<f16>,
    gate_proj: GpuBuffer<f16>,
    up_proj: GpuBuffer<f16>,
    down_proj: GpuBuffer<f16>,
}

struct Gemma4MoELayerData {
    input_layernorm: GpuBuffer<f16>,
    post_attention_layernorm: GpuBuffer<f16>,
    pre_feedforward_layernorm: GpuBuffer<f16>,
    post_feedforward_layernorm: GpuBuffer<f16>,
    q_proj: GpuBuffer<f16>,
    k_proj: GpuBuffer<f16>,
    v_proj: GpuBuffer<f16>,
    o_proj: GpuBuffer<f16>,
    moe: MoELayer,
}

enum Gemma4Layer {
    Dense(Gemma4DenseLayer),
    MoE(Gemma4MoELayerData),
}

// ===========================================================================
// Gemma4ForCausalLM
// ===========================================================================

pub struct Gemma4ForCausalLM {
    hidden_size: usize,
    head_dim: usize,
    global_head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    sliding_window: usize,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<Gemma4Layer>,
    norm_weight: GpuBuffer<f16>,
    lm_head_weight: GpuBuffer<f16>,
    moe_layer_freq: usize,
}

impl Gemma4ForCausalLM {
    pub fn new(weights: ModelWeights, config: &ModelRunnerConfig) -> Result<Self> {
        let embed_tokens = weights
            .get_as_buffer("model.embed_tokens.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.vocab_size, config.hidden_size]));

        let global_head_dim = config.head_dim * 2;
        let moe_layer_freq = 2;

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("model.layers.{}", i);
            let is_moe = (i + 1) % moe_layer_freq == 0;

            let is_global = !is_moe;
            let effective_head_dim = if is_global { global_head_dim } else { config.head_dim };

            let input_layernorm = get_or_zeros(
                &weights,
                &format!("{p}.input_layernorm.weight"),
                &[config.hidden_size],
            );
            let post_attention_layernorm = get_or_zeros(
                &weights,
                &format!("{p}.post_attention_layernorm.weight"),
                &[config.hidden_size],
            );
            let pre_feedforward_layernorm = get_or_zeros(
                &weights,
                &format!("{p}.pre_feedforward_layernorm.weight"),
                &[config.hidden_size],
            );
            let post_feedforward_layernorm = get_or_zeros(
                &weights,
                &format!("{p}.post_feedforward_layernorm.weight"),
                &[config.hidden_size],
            );

            let q_proj = get_or_zeros(
                &weights,
                &format!("{p}.self_attn.q_proj.weight"),
                &[config.num_heads * effective_head_dim, config.hidden_size],
            );
            let k_proj = get_or_zeros(
                &weights,
                &format!("{p}.self_attn.k_proj.weight"),
                &[config.num_kv_heads * effective_head_dim, config.hidden_size],
            );
            let v_proj = get_or_zeros(
                &weights,
                &format!("{p}.self_attn.v_proj.weight"),
                &[config.num_kv_heads * effective_head_dim, config.hidden_size],
            );
            let o_proj = get_or_zeros(
                &weights,
                &format!("{p}.self_attn.o_proj.weight"),
                &[config.hidden_size, config.num_heads * effective_head_dim],
            );

            if is_moe {
                let gate = get_or_zeros(
                    &weights,
                    &format!("{p}.block_sparse_moe.gate.weight"),
                    &[NUM_EXPERTS, config.hidden_size],
                );

                let mut experts = Vec::with_capacity(NUM_EXPERTS);
                for e in 0..NUM_EXPERTS {
                    let ep = format!("{p}.block_sparse_moe.experts.{e}");
                    experts.push(ExpertFFN {
                        gate_proj: get_or_zeros(
                            &weights,
                            &format!("{ep}.w1.weight"),
                            &[config.intermediate_size, config.hidden_size],
                        ),
                        up_proj: get_or_zeros(
                            &weights,
                            &format!("{ep}.w3.weight"),
                            &[config.intermediate_size, config.hidden_size],
                        ),
                        down_proj: get_or_zeros(
                            &weights,
                            &format!("{ep}.w2.weight"),
                            &[config.hidden_size, config.intermediate_size],
                        ),
                    });
                }

                let shared_expert = ExpertFFN {
                    gate_proj: get_or_zeros(
                        &weights,
                        &format!("{p}.block_sparse_moe.shared_expert.w1.weight"),
                        &[config.intermediate_size, config.hidden_size],
                    ),
                    up_proj: get_or_zeros(
                        &weights,
                        &format!("{p}.block_sparse_moe.shared_expert.w3.weight"),
                        &[config.intermediate_size, config.hidden_size],
                    ),
                    down_proj: get_or_zeros(
                        &weights,
                        &format!("{p}.block_sparse_moe.shared_expert.w2.weight"),
                        &[config.hidden_size, config.intermediate_size],
                    ),
                };

                layers.push(Gemma4Layer::MoE(Gemma4MoELayerData {
                    input_layernorm,
                    post_attention_layernorm,
                    pre_feedforward_layernorm,
                    post_feedforward_layernorm,
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    moe: MoELayer {
                        gate,
                        experts,
                        top_k: TOP_K,
                        renormalize: true,
                        shared_expert: Some(shared_expert),
                    },
                }));
            } else {
                layers.push(Gemma4Layer::Dense(Gemma4DenseLayer {
                    input_layernorm,
                    post_attention_layernorm,
                    pre_feedforward_layernorm,
                    post_feedforward_layernorm,
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    gate_proj: get_or_zeros(
                        &weights,
                        &format!("{p}.mlp.gate_proj.weight"),
                        &[config.intermediate_size, config.hidden_size],
                    ),
                    up_proj: get_or_zeros(
                        &weights,
                        &format!("{p}.mlp.up_proj.weight"),
                        &[config.intermediate_size, config.hidden_size],
                    ),
                    down_proj: get_or_zeros(
                        &weights,
                        &format!("{p}.mlp.down_proj.weight"),
                        &[config.hidden_size, config.intermediate_size],
                    ),
                }));
            }
        }

        let norm_weight = weights
            .get_as_buffer("model.norm.weight")
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.hidden_size]));

        let lm_head_weight = weights
            .get_as_buffer("lm_head.weight")
            .or_else(|_| weights.get_as_buffer("model.embed_tokens.weight"))
            .unwrap_or_else(|_| GpuBuffer::zeros(&[config.vocab_size, config.hidden_size]));

        Ok(Self {
            hidden_size: config.hidden_size,
            head_dim: config.head_dim,
            global_head_dim,
            vocab_size: config.vocab_size,
            rms_norm_eps: 1e-6,
            sliding_window: 4096,
            embed_tokens,
            layers,
            norm_weight,
            lm_head_weight,
            moe_layer_freq,
        })
    }

    fn is_sliding_window_layer(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.moe_layer_freq == 0
    }

    fn effective_head_dim(&self, layer_idx: usize) -> usize {
        if self.is_sliding_window_layer(layer_idx) {
            self.head_dim
        } else {
            self.global_head_dim
        }
    }
}

impl Architecture for Gemma4ForCausalLM {
    fn forward(
        &self,
        input: &ModelInput,
        _cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>> {
        let num_tokens = input.num_tokens();
        let mut hidden = gemma_embed_tokens(&self.embed_tokens, &input.token_ids, self.hidden_size);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            trace!(layer = layer_idx, "gemma4 layer forward");

            let head_dim = self.effective_head_dim(layer_idx);

            match layer {
                Gemma4Layer::Dense(dense) => {
                    let normed = GemmaRMSNorm::forward(
                        &hidden,
                        &dense.input_layernorm,
                        self.rms_norm_eps,
                    )?;

                    let q = LinearLayer::forward(&normed, &dense.q_proj, None)?;
                    let k = LinearLayer::forward(&normed, &dense.k_proj, None)?;
                    let v = LinearLayer::forward(&normed, &dense.v_proj, None)?;

                    let (q_rot, k_rot) =
                        RotaryEmbedding::forward(&input.position_ids, &q, &k, head_dim)?;

                    let attn_out = attention.forward(
                        &q_rot,
                        &k_rot,
                        &v,
                        &input.attention_metadata,
                        layer_idx,
                    )?;
                    let attn_proj = LinearLayer::forward(&attn_out, &dense.o_proj, None)?;

                    let attn_normed = GemmaRMSNorm::forward(
                        &attn_proj,
                        &dense.post_attention_layernorm,
                        self.rms_norm_eps,
                    )?;
                    add_inplace(&mut hidden, &attn_normed);

                    let ff_normed = GemmaRMSNorm::forward(
                        &hidden,
                        &dense.pre_feedforward_layernorm,
                        self.rms_norm_eps,
                    )?;
                    let mlp_out = silu_mlp_forward(
                        &ff_normed,
                        &dense.gate_proj,
                        &dense.up_proj,
                        &dense.down_proj,
                    )?;
                    let mlp_normed = GemmaRMSNorm::forward(
                        &mlp_out,
                        &dense.post_feedforward_layernorm,
                        self.rms_norm_eps,
                    )?;
                    add_inplace(&mut hidden, &mlp_normed);
                }
                Gemma4Layer::MoE(moe_layer) => {
                    let normed = GemmaRMSNorm::forward(
                        &hidden,
                        &moe_layer.input_layernorm,
                        self.rms_norm_eps,
                    )?;

                    let q = LinearLayer::forward(&normed, &moe_layer.q_proj, None)?;
                    let k = LinearLayer::forward(&normed, &moe_layer.k_proj, None)?;
                    let v = LinearLayer::forward(&normed, &moe_layer.v_proj, None)?;

                    let (q_rot, k_rot) =
                        RotaryEmbedding::forward(&input.position_ids, &q, &k, head_dim)?;

                    let _use_sliding = self.sliding_window > 0;
                    let attn_out = attention.forward(
                        &q_rot,
                        &k_rot,
                        &v,
                        &input.attention_metadata,
                        layer_idx,
                    )?;
                    let attn_proj = LinearLayer::forward(&attn_out, &moe_layer.o_proj, None)?;

                    let attn_normed = GemmaRMSNorm::forward(
                        &attn_proj,
                        &moe_layer.post_attention_layernorm,
                        self.rms_norm_eps,
                    )?;
                    add_inplace(&mut hidden, &attn_normed);

                    let ff_normed = GemmaRMSNorm::forward(
                        &hidden,
                        &moe_layer.pre_feedforward_layernorm,
                        self.rms_norm_eps,
                    )?;
                    let moe_out = moe_layer.moe.forward(&ff_normed)?;
                    let moe_normed = GemmaRMSNorm::forward(
                        &moe_out,
                        &moe_layer.post_feedforward_layernorm,
                        self.rms_norm_eps,
                    )?;
                    add_inplace(&mut hidden, &moe_normed);
                }
            }
        }

        let normed_final = GemmaRMSNorm::forward(&hidden, &self.norm_weight, self.rms_norm_eps)?;
        lm_head(
            &normed_final,
            &self.lm_head_weight,
            num_tokens,
            self.vocab_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{AttentionMetadata, CacheEngine, MockAttentionBackend, ModelWeights};
    use crate::input::ModelInput;
    use crate::runner::ModelRunnerConfig;

    fn test_config() -> ModelRunnerConfig {
        ModelRunnerConfig {
            num_layers: 4,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            intermediate_size: 4,
            vocab_size: 8,
            max_position: 32,
            dtype: rvllm_core::types::Dtype::Float16,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            architecture: "Gemma4ForCausalLM".into(),
        }
    }

    fn test_input() -> ModelInput {
        ModelInput {
            token_ids: vec![1, 2],
            position_ids: vec![0, 1],
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0, 1],
                context_lens: vec![2],
                block_tables: vec![vec![0]],
                query_lens: vec![1],
                max_context_len: 2,
            },
            is_prefill: true,
        }
    }

    #[test]
    fn gemma4_forward_smoke() {
        let config = test_config();
        let weights = ModelWeights::default();
        let model = Gemma4ForCausalLM::new(weights, &config).unwrap();
        let input = test_input();
        let cache = CacheEngine::new(4, 64);
        let attention = MockAttentionBackend;
        let logits = model.forward(&input, &cache, &attention).unwrap();
        assert_eq!(logits.shape, vec![2, 8]);
    }

    #[test]
    fn gemma4_via_factory() {
        let config = test_config();
        let weights = ModelWeights::default();
        let model = super::super::create_model("Gemma4ForCausalLM", weights, &config).unwrap();
        let input = test_input();
        let cache = CacheEngine::new(4, 64);
        let attention = MockAttentionBackend;
        let logits = model.forward(&input, &cache, &attention).unwrap();
        assert_eq!(logits.shape, vec![2, 8]);
    }

    #[test]
    fn gemma4_single_token() {
        let config = test_config();
        let weights = ModelWeights::default();
        let model = Gemma4ForCausalLM::new(weights, &config).unwrap();
        let input = ModelInput {
            token_ids: vec![0],
            position_ids: vec![0],
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0],
                context_lens: vec![1],
                block_tables: vec![vec![0]],
                query_lens: vec![1],
                max_context_len: 1,
            },
            is_prefill: false,
        };
        let cache = CacheEngine::new(4, 64);
        let attention = MockAttentionBackend;
        let logits = model.forward(&input, &cache, &attention).unwrap();
        assert_eq!(logits.shape, vec![1, 8]);
    }

    #[test]
    fn gemma4_interleaved_layer_types() {
        let config = test_config();
        let weights = ModelWeights::default();
        let model = Gemma4ForCausalLM::new(weights, &config).unwrap();
        assert_eq!(model.layers.len(), 4);
        assert!(matches!(model.layers[0], Gemma4Layer::Dense(_)));
        assert!(matches!(model.layers[1], Gemma4Layer::MoE(_)));
        assert!(matches!(model.layers[2], Gemma4Layer::Dense(_)));
        assert!(matches!(model.layers[3], Gemma4Layer::MoE(_)));
    }
}
