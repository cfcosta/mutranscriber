//! Qwen3 Audio Encoder (AuT - Audio Transformer).
//!
//! This module implements the audio encoder consisting of:
//! - Conv2D downsampling stack (8x total downsampling)
//! - 24-layer Transformer encoder
//! - Output projection to LLM hidden dimension

use candle_core::{Module, Result, Tensor};
use candle_nn::{
    conv2d,
    layer_norm,
    linear,
    Activation,
    Conv2d,
    Conv2dConfig,
    LayerNorm,
    Linear,
    VarBuilder,
};

use crate::config::AudioEncoderConfig;

/// Convolutional downsampling block.
struct ConvBlock {
    conv: Conv2d,
    activation: Activation,
}

impl ConvBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
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
        let attn_weights =
            candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;

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
        let self_attn_layer_norm =
            layer_norm(config.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        // FFN uses fc1/fc2 directly at layer level, not nested
        let ffn = FeedForward::new(
            config.d_model,
            config.encoder_ffn_dim,
            vb.clone(),
        )?;
        let final_layer_norm =
            layer_norm(config.d_model, 1e-5, vb.pp("final_layer_norm"))?;

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

/// Qwen3 Audio Encoder (AuT).
///
/// Architecture:
/// 1. Conv2D downsampling: (batch, 1, mel_bins, time) -> (batch, 480, mel_bins/8, time/8)
/// 2. Linear projection: flatten and project to d_model
/// 3. 18 Transformer encoder layers
/// 4. Output projection to LLM dimension
pub struct Qwen3AudioEncoder {
    conv1: ConvBlock,
    conv2: ConvBlock,
    conv3: ConvBlock,
    conv_out: Linear, // Linear projection after conv, not Conv2d
    layers: Vec<EncoderLayer>,
    output_proj: OutputProjection,
}

impl Qwen3AudioEncoder {
    /// Load audio encoder from pretrained weights.
    pub fn new(config: &AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        // Conv2D downsampling stack
        // Input: (batch, 1, mel_bins=128, time)
        // After 3 stride-2 convs: (batch, 480, 16, time/8)
        let conv1 = ConvBlock::new(1, 480, 2, vb.pp("conv2d1"))?;
        let conv2 = ConvBlock::new(480, 480, 2, vb.pp("conv2d2"))?;
        let conv3 = ConvBlock::new(480, 480, 2, vb.pp("conv2d3"))?;

        // Linear projection (no bias): 480 * 16 = 7680 -> d_model (896)
        let conv_out = candle_nn::linear_no_bias(
            480 * 16,
            config.d_model,
            vb.pp("conv_out"),
        )?;

        // Transformer encoder layers
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            let layer =
                EncoderLayer::new(config, vb.pp(format!("layers.{}", i)))?;
            layers.push(layer);
        }

        // Output projection (ln_post, proj1, proj2 are at the root level)
        let output_proj = OutputProjection::new(
            config.d_model,
            config.output_dim,
            vb.clone(),
        )?;

        Ok(Self {
            conv1,
            conv2,
            conv3,
            conv_out,
            layers,
            output_proj,
        })
    }

    /// Forward pass through the audio encoder.
    ///
    /// Input: Mel spectrogram tensor of shape (batch, n_mels, n_frames)
    /// Output: Audio features of shape (batch, seq_len, output_dim)
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Input: (batch, n_mels=128, n_frames)
        // Reshape to conv input: (batch, 1, n_mels, n_frames)
        let x = mel.unsqueeze(1)?;

        // Conv2D downsampling (8x in mel and time dimensions)
        // After conv1 (stride 2): (batch, 480, 64, time/2)
        // After conv2 (stride 2): (batch, 480, 32, time/4)
        // After conv3 (stride 2): (batch, 480, 16, time/8)
        let x = self.conv1.forward(&x)?;
        let x = self.conv2.forward(&x)?;
        let x = self.conv3.forward(&x)?;

        // Reshape for linear: (batch, 480, 16, time/8) -> (batch, time/8, 480*16)
        let (batch_size, channels, mel_dim, time_dim) = x.dims4()?;
        let x = x.permute((0, 3, 1, 2))?; // (batch, time/8, 480, 16)
        let x = x.reshape((batch_size, time_dim, channels * mel_dim))?; // (batch, time/8, 7680)

        // Linear projection to d_model
        let x = self.conv_out.forward(&x)?; // (batch, time/8, d_model)

        // Transformer encoder layers
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // Output projection
        self.output_proj.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = AudioEncoderConfig::default();
        assert_eq!(config.d_model, 896);
        assert_eq!(config.encoder_layers, 18);
        assert_eq!(config.num_mel_bins, 128);
        assert_eq!(config.output_dim, 1024);
    }
}
