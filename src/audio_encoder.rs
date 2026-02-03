//! Qwen3 Audio Encoder (AuT - Audio Transformer).
//!
//! This module implements the audio encoder consisting of:
//! - Conv2D downsampling stack (8x total downsampling)
//! - Sinusoidal positional embeddings
//! - Transformer encoder layers
//! - Output projection to LLM hidden dimension

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    conv2d,
    linear,
    Activation,
    Conv2d,
    Conv2dConfig,
    Linear,
    VarBuilder,
};

use crate::config::AudioEncoderConfig;

/// Layer normalization with CUDA-compatible implementation.
///
/// Uses basic tensor operations instead of candle_nn::layer_norm
/// which lacks CUDA kernel support.
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        let bias = vb.get(size, "bias")?;
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();

        // Compute in f32 for numerical stability (skip if already f32)
        let x = if dtype == DType::F32 {
            x.clone()
        } else {
            x.to_dtype(DType::F32)?
        };

        // Layer norm: (x - mean) / sqrt(var + eps) * gamma + beta
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let variance =
            x_centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_norm =
            x_centered.broadcast_div(&(variance + self.eps)?.sqrt()?)?;

        // Apply weight and bias, convert back if needed
        let x_norm = if dtype != DType::F32 {
            x_norm.to_dtype(dtype)?
        } else {
            x_norm
        };
        x_norm
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
    }
}

/// Sinusoidal positional embeddings for audio features.
///
/// Matches the Python implementation in modeling_qwen3_asr.py:
/// - Uses sin/cos patterns with exponentially spaced frequencies
/// - max_timescale=10000 following the original Transformer paper
struct SinusoidalPositionEmbedding {
    embedding: Tensor,
}

impl SinusoidalPositionEmbedding {
    /// Create sinusoidal positional embeddings.
    ///
    /// Arguments:
    /// - length: Maximum sequence length (max_source_positions)
    /// - channels: Embedding dimension (d_model)
    /// - device: Device to create tensors on
    fn new(length: usize, channels: usize, device: &Device) -> Result<Self> {
        let max_timescale: f64 = 10000.0;
        let half_channels = channels / 2;

        // Compute inverse timescales: exp(-log(max_timescale) * i / (channels/2 - 1))
        let log_timescale_increment =
            max_timescale.ln() / (half_channels - 1) as f64;
        let inv_timescales: Vec<f32> = (0..half_channels)
            .map(|i| (-log_timescale_increment * i as f64).exp() as f32)
            .collect();

        // Create position indices [0, 1, 2, ..., length-1]
        let positions: Vec<f32> = (0..length).map(|i| i as f32).collect();

        // Compute scaled_time: positions[:, None] * inv_timescales[None, :]
        // Shape: (length, half_channels)
        let mut scaled_time = vec![0.0f32; length * half_channels];
        for (i, &pos) in positions.iter().enumerate() {
            for (j, &inv_ts) in inv_timescales.iter().enumerate() {
                scaled_time[i * half_channels + j] = pos * inv_ts;
            }
        }

        // Compute sin and cos, then concatenate: [sin(scaled_time), cos(scaled_time)]
        // Shape: (length, channels)
        let mut embedding = vec![0.0f32; length * channels];
        for i in 0..length {
            for j in 0..half_channels {
                let val = scaled_time[i * half_channels + j];
                embedding[i * channels + j] = val.sin();
                embedding[i * channels + half_channels + j] = val.cos();
            }
        }

        let embedding =
            Tensor::from_vec(embedding, (length, channels), device)?;

        Ok(Self { embedding })
    }

    /// Get positional embeddings for a given sequence length.
    ///
    /// Returns shape (1, seq_len, channels) for broadcasting with input.
    fn forward(&self, seq_len: usize) -> Result<Tensor> {
        self.embedding.narrow(0, 0, seq_len)?.unsqueeze(0)
    }
}

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
        // CUBLAS handles transposed inputs natively via transpose flags
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * self.scale)?;
        let attn_weights =
            candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;

        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to (batch, seq_len, d_model)
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.n_heads * self.head_dim))?;

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
        let self_attn_layer_norm = LayerNorm::new(
            config.d_model,
            1e-5,
            vb.pp("self_attn_layer_norm"),
        )?;
        // FFN uses fc1/fc2 directly at layer level, not nested
        let ffn = FeedForward::new(
            config.d_model,
            config.encoder_ffn_dim,
            vb.clone(),
        )?;
        let final_layer_norm =
            LayerNorm::new(config.d_model, 1e-5, vb.pp("final_layer_norm"))?;

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
        let ln_post = LayerNorm::new(d_model, 1e-5, vb.pp("ln_post"))?;
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
/// 3. Add sinusoidal positional embeddings
/// 4. Transformer encoder layers
/// 5. Output projection to LLM dimension
pub struct Qwen3AudioEncoder {
    conv1: ConvBlock,
    conv2: ConvBlock,
    conv3: ConvBlock,
    conv_out: Linear,
    positional_embedding: SinusoidalPositionEmbedding,
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

        // Sinusoidal positional embeddings (computed, not learned)
        let positional_embedding = SinusoidalPositionEmbedding::new(
            config.max_source_positions,
            config.d_model,
            vb.device(),
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
            positional_embedding,
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

        // Add sinusoidal positional embeddings
        let pos_embed = self
            .positional_embedding
            .forward(time_dim)?
            .to_dtype(x.dtype())?;
        let x = x.broadcast_add(&pos_embed)?;

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
        assert_eq!(config.max_source_positions, 1500);
    }

    #[test]
    fn test_sinusoidal_position_embedding() {
        let device = Device::Cpu;
        let length = 100;
        let channels = 64;

        let pos_embed =
            SinusoidalPositionEmbedding::new(length, channels, &device)
                .unwrap();

        // Test forward pass
        let seq_len = 50;
        let output = pos_embed.forward(seq_len).unwrap();
        assert_eq!(output.dims(), &[1, seq_len, channels]);

        // Verify values are in reasonable range (sin/cos are bounded by [-1, 1])
        let output_vec =
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for val in &output_vec {
            assert!(
                *val >= -1.0 && *val <= 1.0,
                "Position embedding value {} out of range",
                val
            );
        }

        // First position should have sin(0)=0 for first half, cos(0)=1 for second half
        let first_pos = pos_embed.forward(1).unwrap();
        let first_vec =
            first_pos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // sin(0) = 0 for the first element
        assert!(
            first_vec[0].abs() < 1e-5,
            "sin(0) should be ~0, got {}",
            first_vec[0]
        );
        // cos(0) = 1 for the element at half_channels
        assert!(
            (first_vec[channels / 2] - 1.0).abs() < 1e-5,
            "cos(0) should be ~1, got {}",
            first_vec[channels / 2]
        );
    }
}
