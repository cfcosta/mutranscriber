//! Custom Qwen3 decoder with embedding injection support for multimodal inputs.
//!
//! This module implements a Qwen3 decoder that can accept pre-computed embeddings
//! instead of just token IDs, enabling audio feature injection for ASR.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, embedding, linear_no_bias};
use candle_transformers::models::qwen3::Config;

/// RMS normalization layer.
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = if dtype == DType::F32 {
            x.clone()
        } else {
            x.to_dtype(DType::F32)?
        };
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        if dtype != DType::F32 {
            x.to_dtype(dtype)?.broadcast_mul(&self.weight)
        } else {
            x.broadcast_mul(&self.weight)
        }
    }
}

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

    fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = Self::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = Self::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    /// CUDA-compatible rotary position embedding using basic tensor ops.
    fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (_, _, _, d) = x.dims4()?;
        let half_d = d / 2;

        // Split into first half and second half
        let x1 = x.narrow(3, 0, half_d)?;
        let x2 = x.narrow(3, half_d, half_d)?;

        // Broadcast cos/sin to match tensor shape: (seq, half_d) -> (b, h, seq, half_d)
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
        let y1 = x1
            .broadcast_mul(&cos)?
            .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
        let y2 = x2
            .broadcast_mul(&cos)?
            .broadcast_add(&x1.broadcast_mul(&sin)?)?;

        Tensor::cat(&[&y1, &y2], 3)
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

/// Self-attention with KV cache.
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
    kv_cache: Option<(Tensor, Tensor)>,
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
            q_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary,
            kv_cache: None,
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

        // Reshape to (batch, seq, num_heads, head_dim)
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k =
            k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v =
            v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;

        // Apply QK normalization
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose to (batch, num_heads, seq, head_dim)
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = self.rotary.apply(&q, &k, offset)?;

        // KV cache - store first, then borrow for GQA to avoid unnecessary clones
        // Note: Tensor::cat copies all previous data + new data, which is O(n²) total
        // for n tokens. This is inherent to Candle's immutable tensor design.
        let (k, v) = if let Some((prev_k, prev_v)) = &self.kv_cache {
            // Concatenate along sequence dimension (dim 2)
            let k = Tensor::cat(&[prev_k, &k], 2)?;
            let v = Tensor::cat(&[prev_v, &v], 2)?;
            (k, v)
        } else {
            (k, v)
        };
        self.kv_cache = Some((k, v));

        // Borrow from cache for GQA expansion (avoids clone when num_groups > 1)
        let (k_cache, v_cache) = self.kv_cache.as_ref().unwrap();

        // Repeat KV for GQA
        let num_groups = self.num_heads / self.num_kv_heads;
        let k = if num_groups > 1 {
            let (b, h, s, d) = k_cache.dims4()?;
            k_cache
                .unsqueeze(2)?
                .expand((b, h, num_groups, s, d))?
                .reshape((b, h * num_groups, s, d))?
        } else {
            k_cache.clone()
        };
        let v = if num_groups > 1 {
            let (b, h, s, d) = v_cache.dims4()?;
            v_cache
                .unsqueeze(2)?
                .expand((b, h, num_groups, s, d))?
                .reshape((b, h * num_groups, s, d))?
        } else {
            v_cache.clone()
        };

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = if let Some(m) = mask {
            attn.broadcast_add(m)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax(&attn, candle_core::D::Minus1)?;

        // Output projection
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        self.o_proj.forward(&out)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
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
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
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

        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("model.norm"),
        )?;

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
            Some(self.causal_mask(batch, seq_len, offset)?)
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
