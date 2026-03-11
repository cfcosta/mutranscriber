//! Custom Qwen3 decoder with embedding injection support for multimodal inputs.
//!
//! This module implements a Qwen3 decoder that can accept pre-computed embeddings
//! instead of just token IDs, enabling audio feature injection for ASR.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    embedding,
    linear_no_bias,
    rms_norm,
    rotary_emb,
    Embedding,
    Linear,
    VarBuilder,
};
use candle_transformers::models::qwen3::Config;

/// RMS normalization.
///
/// Candle 0.9 provides fused CUDA/Metal kernels for contiguous inputs, which
/// is substantially faster than composing rms-norm from basic tensor ops.
type RmsNorm = candle_nn::RmsNorm;

/// Rotary position embedding.
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?
            .to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply rotary embeddings to Q and K in (batch, seq, heads, dim) layout.
    fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_, seq_len, _, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = rotary_emb::rope_thd(&q.contiguous()?, &cos, &sin)?;
        let k_embed = rotary_emb::rope_thd(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

/// MLP block with SiLU activation.
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Maximum sequence length for pre-allocated KV cache.
const MAX_SEQ_LEN: usize = 2048;

/// Self-attention with pre-allocated KV cache.
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
    cache_len: usize,
}

impl Attention {
    fn new(
        cfg: &Config,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        Ok(Self {
            q_proj: linear_no_bias(
                cfg.hidden_size,
                num_heads * head_dim,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_no_bias(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_no_bias(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_no_bias(
                num_heads * head_dim,
                cfg.hidden_size,
                vb.pp("o_proj"),
            )?,
            q_norm: rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary,
            k_cache: None,
            v_cache: None,
            cache_len: 0,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, seq, num_heads, head_dim) — already the layout
        // flash_attn and our KV cache expect, so no transposes needed on CUDA.
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k =
            k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v =
            v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;

        // Apply QK normalization (operates on last dim = head_dim)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Apply rotary embeddings directly on (batch, seq, heads, dim) — no transpose needed
        let (q, k) = self.rotary.apply(&q, &k, offset)?;

        // V is already in cache layout from reshape — contiguous, no transpose needed.
        // K comes from RoPE's Tensor::cat which produces a contiguous result.

        // Pre-allocated KV cache in (batch, MAX_SEQ_LEN, num_kv_heads, head_dim)
        if self.k_cache.is_none() {
            let k_buf = Tensor::zeros(
                (batch, MAX_SEQ_LEN, self.num_kv_heads, self.head_dim),
                k.dtype(),
                k.device(),
            )?;
            let v_buf = Tensor::zeros(
                (batch, MAX_SEQ_LEN, self.num_kv_heads, self.head_dim),
                v.dtype(),
                v.device(),
            )?;
            self.k_cache = Some(k_buf);
            self.v_cache = Some(v_buf);
        }

        let k_cache = self.k_cache.as_ref().unwrap();
        let v_cache = self.v_cache.as_ref().unwrap();
        k_cache.slice_set(&k, 1, self.cache_len)?;
        v_cache.slice_set(&v, 1, self.cache_len)?;
        self.cache_len += seq_len;

        // Flash attention path (CUDA only) — everything already in (batch, seq, heads, dim)
        #[cfg(feature = "cuda")]
        {
            if q.device().is_cuda() {
                let k_full = k_cache.narrow(1, 0, self.cache_len)?;
                let v_full = v_cache.narrow(1, 0, self.cache_len)?;
                let scale = (self.head_dim as f32).powf(-0.5);
                let out = candle_flash_attn::flash_attn(
                    &q, &k_full, &v_full, scale, true,
                )?;
                let out = out.reshape((
                    batch,
                    seq_len,
                    self.num_heads * self.head_dim,
                ))?;
                return self.o_proj.forward(&out);
            }
        }

        // CPU/Metal fallback: transpose to (batch, heads, seq, dim) for naive matmul attention
        let q = q.transpose(1, 2)?.contiguous()?;
        let k_full = k_cache
            .narrow(1, 0, self.cache_len)?
            .transpose(1, 2)?
            .contiguous()?;
        let v_full = v_cache
            .narrow(1, 0, self.cache_len)?
            .transpose(1, 2)?
            .contiguous()?;

        // GQA expansion
        let num_groups = self.num_heads / self.num_kv_heads;
        let k = if num_groups > 1 {
            let (b, h, s, d) = k_full.dims4()?;
            k_full
                .unsqueeze(2)?
                .expand((b, h, num_groups, s, d))?
                .reshape((b, h * num_groups, s, d))?
        } else {
            k_full
        };
        let v = if num_groups > 1 {
            let (b, h, s, d) = v_full.dims4()?;
            v_full
                .unsqueeze(2)?
                .expand((b, h, num_groups, s, d))?
                .reshape((b, h * num_groups, s, d))?
        } else {
            v_full
        };

        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = if let Some(m) = mask {
            attn.broadcast_add(m)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax(&attn, candle_core::D::Minus1)?;

        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        self.o_proj.forward(&out)
    }

    fn clear_kv_cache(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
        self.cache_len = 0;
    }
}

/// Transformer decoder layer.
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Config,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, rotary, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, offset)?;
        let x = (residual + x)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Qwen3 decoder with embedding injection support.
pub struct Qwen3Decoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl Qwen3Decoder {
    /// Load decoder from pretrained weights.
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;
        let rotary =
            Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg,
                rotary.clone(),
                vb_layers.pp(i),
            )?);
        }

        let norm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        // lm_head is tied with embed_tokens for Qwen3
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Forward pass with token IDs.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        let embeddings = self.embed_tokens.forward(input_ids)?;
        self.forward_embeds(&embeddings, offset)
    }

    /// Forward pass with pre-computed embeddings (for multimodal input).
    pub fn forward_embeds(
        &mut self,
        embeddings: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = embeddings.dims3()?;

        let mask = if seq_len > 1 {
            #[cfg(feature = "cuda")]
            {
                if self.device.is_cuda() {
                    None // flash attention handles causal masking
                } else {
                    Some(self.causal_mask(batch, seq_len, offset)?)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                Some(self.causal_mask(batch, seq_len, offset)?)
            }
        } else {
            None
        };

        let mut h = embeddings.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, mask.as_ref(), offset)?;
        }

        let h = self.norm.forward(&h)?;
        // Only take the last token's logits
        let h = h.narrow(1, seq_len - 1, 1)?;
        self.lm_head.forward(&h)
    }

    /// Get token embeddings for specific token IDs.
    pub fn get_token_embeddings(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(token_ids)
    }

    /// Get the hidden size.
    /// Clear KV cache.
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }

    /// Create causal attention mask.
    fn causal_mask(
        &self,
        batch: usize,
        seq_len: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let total_len = seq_len + offset;
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..total_len).map(move |j| {
                    if j <= i + offset {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (batch, 1, seq_len, total_len), &self.device)?
            .to_dtype(self.dtype)
    }
}
