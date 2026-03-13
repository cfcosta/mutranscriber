//! Qwen3 Audio Encoder (AuT - Audio Transformer).
//!
//! This implementation follows the official Qwen3-ASR audio tower closely:
//! - 3 stride-2 Conv2D downsampling layers over mel frames
//! - chunked convolution over 100-frame windows
//! - ragged/local self-attention over flattened audio tokens
//! - LayerNorm + projection into the text decoder hidden size

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{
    conv2d,
    layer_norm,
    linear,
    linear_no_bias,
    Activation,
    Conv2d,
    Conv2dConfig,
    Linear,
    VarBuilder,
};

use crate::config::AudioEncoderConfig;

/// Layer normalization.
type LayerNorm = candle_nn::LayerNorm;

/// Sinusoidal positional embeddings for audio features.
struct SinusoidalPositionEmbedding {
    embedding: Tensor,
}

impl SinusoidalPositionEmbedding {
    fn new(length: usize, channels: usize, device: &Device) -> Result<Self> {
        let max_timescale: f64 = 10000.0;
        let half_channels = channels / 2;
        let log_timescale_increment =
            max_timescale.ln() / (half_channels - 1) as f64;

        let inv_timescales: Vec<f32> = (0..half_channels)
            .map(|i| (-log_timescale_increment * i as f64).exp() as f32)
            .collect();
        let positions: Vec<f32> = (0..length).map(|i| i as f32).collect();

        let mut scaled_time = vec![0.0f32; length * half_channels];
        for (i, &pos) in positions.iter().enumerate() {
            for (j, &inv_ts) in inv_timescales.iter().enumerate() {
                scaled_time[i * half_channels + j] = pos * inv_ts;
            }
        }

        let mut embedding = vec![0.0f32; length * channels];
        for i in 0..length {
            for j in 0..half_channels {
                let val = scaled_time[i * half_channels + j];
                embedding[i * channels + j] = val.sin();
                embedding[i * channels + half_channels + j] = val.cos();
            }
        }

        Ok(Self {
            embedding: Tensor::from_vec(embedding, (length, channels), device)?,
        })
    }

    fn forward(&self, seq_len: usize, dtype: DType) -> Result<Tensor> {
        self.embedding
            .narrow(0, 0, seq_len)?
            .to_dtype(dtype)?
            .unsqueeze(0)
    }
}

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
        Ok(Self {
            conv: conv2d(in_channels, out_channels, 3, config, vb)?,
            activation: Activation::Gelu,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        self.activation.forward(&x)
    }
}

/// Audio self-attention over ragged/local windows.
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
        Ok(Self {
            q_proj: linear(d_model, d_model, vb.pp("q_proj"))?,
            k_proj: linear(d_model, d_model, vb.pp("k_proj"))?,
            v_proj: linear(d_model, d_model, vb.pp("v_proj"))?,
            out_proj: linear(d_model, d_model, vb.pp("out_proj"))?,
            n_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn build_block_mask(
        total_seq: usize,
        cu_seqlens: &[usize],
        device: &Device,
    ) -> Result<Tensor> {
        let mut mask = vec![-10_000f32; total_seq * total_seq];
        for window in cu_seqlens.windows(2) {
            let start = window[0];
            let end = window[1];
            for row in start..end {
                let row_offset = row * total_seq;
                for col in start..end {
                    mask[row_offset + col] = 0.0;
                }
            }
        }
        Tensor::from_vec(mask, (1, total_seq, total_seq), device)
    }

    fn forward(&self, x: &Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
        let (seq_len, _) = x.dims2()?;

        let q = self.q_proj.forward(x)?.reshape((
            seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let k = self.k_proj.forward(x)?.reshape((
            seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let v = self.v_proj.forward(x)?.reshape((
            seq_len,
            self.n_heads,
            self.head_dim,
        ))?;

        #[cfg(feature = "cuda")]
        {
            if x.device().is_cuda() {
                let max_seqlen = cu_seqlens
                    .windows(2)
                    .map(|w| w[1] - w[0])
                    .max()
                    .unwrap_or(seq_len);
                let seqlens: Vec<u32> =
                    cu_seqlens.iter().map(|&v| v as u32).collect();
                let seqlens = Tensor::new(seqlens.as_slice(), x.device())?;
                let out = candle_flash_attn::flash_attn_varlen(
                    &q.contiguous()?,
                    &k.contiguous()?,
                    &v.contiguous()?,
                    &seqlens,
                    &seqlens,
                    max_seqlen,
                    max_seqlen,
                    self.scale as f32,
                    false,
                )?;
                let out =
                    out.reshape((seq_len, self.n_heads * self.head_dim))?;
                return self.out_proj.forward(&out);
            }
        }

        let q = q.transpose(0, 1)?.contiguous()?; // (heads, seq, dim)
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;

        let attn_weights = (q.matmul(&k.transpose(1, 2)?)? * self.scale)?;
        let mask = Self::build_block_mask(seq_len, cu_seqlens, x.device())?
            .to_dtype(attn_weights.dtype())?;
        let attn_weights = attn_weights.broadcast_add(&mask)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(0, 1)?
            .reshape((seq_len, self.n_heads * self.head_dim))?;

        self.out_proj.forward(&attn_output)
    }
}

struct FeedForward {
    fc1: Linear,
    fc2: Linear,
    activation: Activation,
}

impl FeedForward {
    fn new(d_model: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(d_model, ffn_dim, vb.pp("fc1"))?,
            fc2: linear(ffn_dim, d_model, vb.pp("fc2"))?,
            activation: Activation::Gelu,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = self.activation.forward(&x)?;
        self.fc2.forward(&x)
    }
}

struct EncoderLayer {
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: LayerNorm,
    ffn: FeedForward,
    final_layer_norm: LayerNorm,
}

impl EncoderLayer {
    fn new(config: &AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(
                config.d_model,
                config.encoder_attention_heads,
                vb.pp("self_attn"),
            )?,
            self_attn_layer_norm: layer_norm(
                config.d_model,
                1e-5,
                vb.pp("self_attn_layer_norm"),
            )?,
            ffn: FeedForward::new(
                config.d_model,
                config.encoder_ffn_dim,
                vb.clone(),
            )?,
            final_layer_norm: layer_norm(
                config.d_model,
                1e-5,
                vb.pp("final_layer_norm"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.self_attn_layer_norm.forward(x)?;
        let x = self.self_attn.forward(&x, cu_seqlens)?;
        let x = (residual + x)?;

        let residual = x.clone();
        let x = self.final_layer_norm.forward(&x)?;
        let x = self.ffn.forward(&x)?;
        residual + x
    }
}

struct OutputProjection {
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    activation: Activation,
}

impl OutputProjection {
    fn new(d_model: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln_post: layer_norm(d_model, 1e-5, vb.pp("ln_post"))?,
            proj1: linear(d_model, d_model, vb.pp("proj1"))?,
            proj2: linear(d_model, output_dim, vb.pp("proj2"))?,
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
pub struct Qwen3AudioEncoder {
    config: AudioEncoderConfig,
    conv1: ConvBlock,
    conv2: ConvBlock,
    conv3: ConvBlock,
    conv_out: Linear,
    positional_embedding: SinusoidalPositionEmbedding,
    layers: Vec<EncoderLayer>,
    output_proj: OutputProjection,
}

impl Qwen3AudioEncoder {
    /// Output time length after the three stride-2 conv layers for a single
    /// convolution chunk.
    fn conv_output_len(input_len: usize) -> usize {
        fn one_conv(len: usize) -> usize {
            if len == 0 {
                0
            } else {
                (len - 1) / 2 + 1
            }
        }
        one_conv(one_conv(one_conv(input_len)))
    }

    /// Output token length after chunked convolution, matching the official
    /// Qwen3-ASR processor helper.
    fn feature_output_len(input_len: usize, chunk_width: usize) -> usize {
        let full_chunks = input_len / chunk_width;
        let remainder = input_len % chunk_width;
        full_chunks * Self::conv_output_len(chunk_width)
            + if remainder == 0 {
                0
            } else {
                Self::conv_output_len(remainder)
            }
    }

    pub fn new(config: &AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let conv1 = ConvBlock::new(
            1,
            config.downsample_hidden_size,
            2,
            vb.pp("conv2d1"),
        )?;
        let conv2 = ConvBlock::new(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            2,
            vb.pp("conv2d2"),
        )?;
        let conv3 = ConvBlock::new(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            2,
            vb.pp("conv2d3"),
        )?;

        let downsampled_mel_bins = Self::conv_output_len(config.num_mel_bins);
        let conv_out = linear_no_bias(
            config.downsample_hidden_size * downsampled_mel_bins,
            config.d_model,
            vb.pp("conv_out"),
        )?;

        let positional_embedding = SinusoidalPositionEmbedding::new(
            config.max_source_positions,
            config.d_model,
            vb.device(),
        )?;

        let mut layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            layers.push(EncoderLayer::new(
                config,
                vb.pp(format!("layers.{}", i)),
            )?);
        }

        let output_proj = OutputProjection::new(
            config.d_model,
            config.output_dim,
            vb.clone(),
        )?;

        Ok(Self {
            config: config.clone(),
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
    /// `feature_len` must be the real mel-frame length before any right padding.
    pub fn forward(&self, mel: &Tensor, feature_len: usize) -> Result<Tensor> {
        let (batch, n_mels, _n_frames) = mel.dims3()?;
        if batch != 1 {
            candle_core::bail!("Qwen3AudioEncoder only supports batch size 1")
        }
        if n_mels != self.config.num_mel_bins {
            candle_core::bail!(
                "Expected {} mel bins, got {}",
                self.config.num_mel_bins,
                n_mels
            )
        }

        let mel = mel.i(0)?; // (n_mels, n_frames)
        let chunk_width = self.config.n_window * 2;
        let num_chunks = feature_len.div_ceil(chunk_width);
        let max_chunk_len = if num_chunks == 0 {
            0
        } else if num_chunks == 1 {
            feature_len
        } else {
            chunk_width
        };
        if max_chunk_len == 0 {
            candle_core::bail!("Cannot encode empty audio feature sequence")
        }

        let mut chunk_lengths = Vec::with_capacity(num_chunks);
        let mut chunk_aftercnn_lens = Vec::with_capacity(num_chunks);
        let mut chunk_tensors = Vec::with_capacity(num_chunks);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_width;
            let len = (feature_len - start).min(chunk_width);
            let mut chunk = mel.narrow(1, start, len)?;
            if len < max_chunk_len {
                let pad = Tensor::zeros(
                    (n_mels, max_chunk_len - len),
                    mel.dtype(),
                    mel.device(),
                )?;
                chunk = Tensor::cat(&[&chunk, &pad], 1)?;
            }
            chunk_lengths.push(len);
            chunk_aftercnn_lens.push(Self::conv_output_len(len));
            chunk_tensors.push(chunk);
        }

        let padded_feature = Tensor::stack(&chunk_tensors, 0)?.unsqueeze(1)?;

        let mut conv_outputs = Vec::new();
        for start in (0..num_chunks).step_by(self.config.conv_chunksize) {
            let len = (num_chunks - start).min(self.config.conv_chunksize);
            let chunk_batch = padded_feature.narrow(0, start, len)?;
            let chunk_batch = self.conv1.forward(&chunk_batch)?;
            let chunk_batch = self.conv2.forward(&chunk_batch)?;
            let chunk_batch = self.conv3.forward(&chunk_batch)?;
            conv_outputs.push(chunk_batch);
        }
        let padded_embed = Tensor::cat(&conv_outputs, 0)?;

        let (chunk_batch, channels, mel_dim, time_dim) =
            padded_embed.dims4()?;
        let padded_embed = padded_embed.permute((0, 3, 1, 2))?;
        let padded_embed = padded_embed.reshape((
            chunk_batch,
            time_dim,
            channels * mel_dim,
        ))?;
        let padded_embed = self.conv_out.forward(&padded_embed)?;
        let pos = self
            .positional_embedding
            .forward(time_dim, padded_embed.dtype())?;
        let padded_embed = padded_embed.broadcast_add(&pos)?;

        let mut pieces = Vec::with_capacity(num_chunks);
        for (chunk_idx, &len_aftercnn) in chunk_aftercnn_lens.iter().enumerate()
        {
            let piece =
                padded_embed.i(chunk_idx)?.narrow(0, 0, len_aftercnn)?;
            pieces.push(piece);
        }
        let mut hidden_states = Tensor::cat(&pieces, 0)?;

        let full_aftercnn_len =
            Self::feature_output_len(feature_len, chunk_width);
        let gathered_len = hidden_states.dim(0)?;
        if gathered_len != full_aftercnn_len {
            candle_core::bail!(
                "Audio token length mismatch after chunk gather: got {}, expected {}",
                gathered_len,
                full_aftercnn_len
            )
        }

        let window_scale = self.config.n_window_infer / chunk_width;
        let window_aftercnn = time_dim * window_scale;
        let mut cu_seqlens = vec![0usize];
        if window_aftercnn == 0 {
            cu_seqlens.push(full_aftercnn_len);
        } else {
            let mut offset = 0usize;
            while offset < full_aftercnn_len {
                let next = (offset + window_aftercnn).min(full_aftercnn_len);
                cu_seqlens.push(next);
                offset = next;
            }
        }

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &cu_seqlens)?;
        }

        let hidden_states = self.output_proj.forward(&hidden_states)?;
        hidden_states.unsqueeze(0)
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
        assert_eq!(config.downsample_hidden_size, 480);
        assert_eq!(config.output_dim, 1024);
        assert_eq!(config.max_source_positions, 1500);
        assert_eq!(config.n_window, 50);
        assert_eq!(config.n_window_infer, 800);
        assert_eq!(config.conv_chunksize, 500);
    }

    #[test]
    fn test_sinusoidal_position_embedding() {
        let device = Device::Cpu;
        let length = 100;
        let channels = 64;

        let pos_embed =
            SinusoidalPositionEmbedding::new(length, channels, &device)
                .unwrap();

        let seq_len = 50;
        let output = pos_embed.forward(seq_len, DType::F32).unwrap();
        assert_eq!(output.dims(), &[1, seq_len, channels]);

        let output_vec =
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for val in &output_vec {
            assert!(
                *val >= -1.0 && *val <= 1.0,
                "Position embedding value {} out of range",
                val
            );
        }

        let first_pos = pos_embed.forward(1, DType::F32).unwrap();
        let first_vec =
            first_pos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            first_vec[0].abs() < 1e-5,
            "sin(0) should be ~0, got {}",
            first_vec[0]
        );
        assert!(
            (first_vec[channels / 2] - 1.0).abs() < 1e-5,
            "cos(0) should be ~1, got {}",
            first_vec[channels / 2]
        );
    }

    #[test]
    fn test_conv_output_len() {
        assert_eq!(Qwen3AudioEncoder::conv_output_len(100), 13);
        assert_eq!(Qwen3AudioEncoder::conv_output_len(98), 13);
        assert_eq!(Qwen3AudioEncoder::conv_output_len(998), 125);
        assert_eq!(Qwen3AudioEncoder::conv_output_len(2998), 375);
    }

    #[test]
    fn test_feature_output_len() {
        assert_eq!(Qwen3AudioEncoder::feature_output_len(98, 100), 13);
        assert_eq!(Qwen3AudioEncoder::feature_output_len(998, 100), 130);
        assert_eq!(Qwen3AudioEncoder::feature_output_len(2998, 100), 390);
    }
}
