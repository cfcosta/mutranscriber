//! Qwen3 Audio Encoder (AuT - Audio Transformer).
//!
//! This module implements the audio encoder consisting of:
//! - Conv2D downsampling stack (8x total downsampling)
//! - 24-layer Transformer encoder
//! - Output projection to LLM hidden dimension

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{
    Activation,
    Conv2d,
    Conv2dConfig,
    Embedding,
    LayerNorm,
    Linear,
    VarBuilder,
    conv2d,
    embedding,
    layer_norm,
    linear,
};

use crate::config::AudioEncoderConfig;

/// Convolutional downsampling block.
struct ConvBlock {
    conv: Conv2d,
    activation: Activation,
}

impl ConvBlock {
    fn new(in_channels: usize, out_channels: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let config = Conv2dConfig {
            stride,
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(in_channels, out_channels, 3, config, vb)?;
        Ok(Self {
            conv,
            activation: Activation::Gelu,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        self.activation.forward(&x)
    }
}

/// Multi-head self-attention layer.
struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    n_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MultiHeadAttention {
    fn new(d_model: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = d_model / n_heads;
        let scale = (head_dim as f64).powf(-0.5);

        let q_proj = linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = linear(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = linear(d_model, d_model, vb.pp("v_proj"))?;
        let out_proj = linear(d_model, d_model, vb.pp("out_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            n_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, n_heads, seq_len, head_dim)
        let q = q
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * self.scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;

        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to (batch, seq_len, d_model)
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.n_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&attn_output)
    }
}

/// Feed-forward network with GELU activation.
struct FeedForward {
    fc1: Linear,
    fc2: Linear,
    activation: Activation,
}

impl FeedForward {
    fn new(d_model: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(d_model, ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear(ffn_dim, d_model, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            activation: Activation::Gelu,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = self.activation.forward(&x)?;
        self.fc2.forward(&x)
    }
}

/// Transformer encoder layer with pre-normalization.
/// Matches Qwen3-ASR tensor naming: self_attn_layer_norm, final_layer_norm
struct EncoderLayer {
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: LayerNorm,
    ffn: FeedForward,
    final_layer_norm: LayerNorm,
}

impl EncoderLayer {
    fn new(config: &AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = MultiHeadAttention::new(
            config.d_model,
            config.encoder_attention_heads,
            vb.pp("self_attn"),
        )?;
        let self_attn_layer_norm = layer_norm(config.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        // FFN uses fc1/fc2 directly at layer level, not nested
        let ffn = FeedForward::new(config.d_model, config.encoder_ffn_dim, vb.clone())?;
        let final_layer_norm = layer_norm(config.d_model, 1e-5, vb.pp("final_layer_norm"))?;

        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            ffn,
            final_layer_norm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-norm self-attention with residual
        let residual = x.clone();
        let x = self.self_attn_layer_norm.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = (residual + x)?;

        // Pre-norm FFN with residual
        let residual = x.clone();
        let x = self.final_layer_norm.forward(&x)?;
        let x = self.ffn.forward(&x)?;
        residual + x
    }
}

/// Output projection from encoder to LLM hidden dimension.
struct OutputProjection {
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    activation: Activation,
}

impl OutputProjection {
    fn new(d_model: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let ln_post = layer_norm(d_model, 1e-5, vb.pp("ln_post"))?;
        let proj1 = linear(d_model, d_model, vb.pp("proj1"))?;
        let proj2 = linear(d_model, output_dim, vb.pp("proj2"))?;

        Ok(Self {
            ln_post,
            proj1,
            proj2,
            activation: Activation::Gelu,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ln_post.forward(x)?;
        let x = self.proj1.forward(&x)?;
        let x = self.activation.forward(&x)?;
        self.proj2.forward(&x)
    }
}

/// Positional embeddings for the encoder.
struct PositionalEmbedding {
    embedding: Embedding,
    #[allow(dead_code)]
    max_positions: usize,
}

impl PositionalEmbedding {
    fn new(max_positions: usize, d_model: usize, vb: VarBuilder) -> Result<Self> {
        let embedding = embedding(max_positions, d_model, vb)?;
        Ok(Self {
            embedding,
            max_positions,
        })
    }

    fn forward(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let positions = Tensor::arange(0u32, seq_len as u32, device)?;
        self.embedding.forward(&positions)
    }
}

/// Qwen3 Audio Encoder (AuT).
///
/// Architecture:
/// 1. Conv2D downsampling (8x total): 128 -> 480 -> 480 -> 480 -> 1024
/// 2. Positional embedding
/// 3. 24 Transformer encoder layers
/// 4. Output projection to LLM dimension
pub struct Qwen3AudioEncoder {
    conv1: ConvBlock,
    conv2: ConvBlock,
    conv3: ConvBlock,
    conv_out: ConvBlock,
    pos_embed: PositionalEmbedding,
    layers: Vec<EncoderLayer>,
    output_proj: OutputProjection,
    config: AudioEncoderConfig,
}

impl Qwen3AudioEncoder {
    /// Load audio encoder from pretrained weights.
    pub fn new(config: &AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        // Conv2D downsampling stack (8x total)
        // Input: (batch, 128, time, 1) -> Output: (batch, 1024, time/8, 1)
        let conv1 = ConvBlock::new(config.num_mel_bins, 480, 2, vb.pp("conv1"))?;
        let conv2 = ConvBlock::new(480, 480, 2, vb.pp("conv2"))?;
        let conv3 = ConvBlock::new(480, 480, 2, vb.pp("conv3"))?;
        let conv_out = ConvBlock::new(480, config.d_model, 1, vb.pp("conv_out"))?;

        // Positional embedding (max 3000 positions for 30s audio with 8x downsample)
        let max_positions = (30 * 16000 / 160 / 8) + 100; // ~3100
        let pos_embed = PositionalEmbedding::new(max_positions, config.d_model, vb.pp("pos_embed"))?;

        // Transformer encoder layers
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            let layer = EncoderLayer::new(config, vb.pp(format!("layers.{}", i)))?;
            layers.push(layer);
        }

        // Output projection
        let output_proj =
            OutputProjection::new(config.d_model, config.output_dim, vb.pp("output_proj"))?;

        Ok(Self {
            conv1,
            conv2,
            conv3,
            conv_out,
            pos_embed,
            layers,
            output_proj,
            config: config.clone(),
        })
    }

    /// Forward pass through the audio encoder.
    ///
    /// Input: Mel spectrogram tensor of shape (batch, n_mels, n_frames)
    /// Output: Audio features of shape (batch, seq_len, output_dim)
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let device = mel.device();

        // Add channel dimension: (batch, n_mels, n_frames) -> (batch, n_mels, n_frames, 1)
        let x = mel.unsqueeze(3)?;

        // Conv2D downsampling (8x in time dimension)
        let x = self.conv1.forward(&x)?;
        let x = self.conv2.forward(&x)?;
        let x = self.conv3.forward(&x)?;
        let x = self.conv_out.forward(&x)?;

        // Reshape for transformer: (batch, d_model, time', 1) -> (batch, time', d_model)
        let (_batch_size, _d_model, time_len, _) = x.dims4()?;
        let x = x.squeeze(3)?; // (batch, d_model, time')
        let x = x.transpose(1, 2)?; // (batch, time', d_model)

        // Add positional embedding
        let pos = self.pos_embed.forward(time_len, device)?;
        let x = x.broadcast_add(&pos)?;

        // Transformer encoder layers
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // Output projection
        self.output_proj.forward(&x)
    }

    /// Get the output dimension.
    pub fn output_dim(&self) -> usize {
        self.config.output_dim
    }

    /// Get the time downsampling factor.
    pub fn downsample_factor(&self) -> usize {
        8 // 2 * 2 * 2 from conv layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = AudioEncoderConfig::default();
        assert_eq!(config.d_model, 1024);
        assert_eq!(config.encoder_layers, 24);
        assert_eq!(config.num_mel_bins, 128);
    }
}
