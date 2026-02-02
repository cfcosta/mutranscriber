//! Configuration structs for Qwen3-ASR model.

use candle_nn::Activation;
use candle_transformers::models::qwen3::Config as Qwen3Config;
use serde::Deserialize;

/// Audio encoder configuration (AuT - Audio Transformer).
#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncoderConfig {
    /// Model dimension (default: 1024)
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    /// Number of encoder layers (default: 24)
    #[serde(default = "default_encoder_layers")]
    pub encoder_layers: usize,
    /// Number of attention heads (default: 16)
    #[serde(default = "default_encoder_attention_heads")]
    pub encoder_attention_heads: usize,
    /// FFN dimension (default: 4096)
    #[serde(default = "default_encoder_ffn_dim")]
    pub encoder_ffn_dim: usize,
    /// Number of mel filterbank bins (default: 128)
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,
    /// Output dimension to LLM (default: 2048)
    #[serde(default = "default_output_dim")]
    pub output_dim: usize,
    /// Dropout rate (default: 0.0)
    #[serde(default)]
    pub dropout: f64,
}

fn default_d_model() -> usize {
    1024
}
fn default_encoder_layers() -> usize {
    24
}
fn default_encoder_attention_heads() -> usize {
    16
}
fn default_encoder_ffn_dim() -> usize {
    4096
}
fn default_num_mel_bins() -> usize {
    128
}
fn default_output_dim() -> usize {
    2048
}

impl Default for AudioEncoderConfig {
    fn default() -> Self {
        Self {
            d_model: default_d_model(),
            encoder_layers: default_encoder_layers(),
            encoder_attention_heads: default_encoder_attention_heads(),
            encoder_ffn_dim: default_encoder_ffn_dim(),
            num_mel_bins: default_num_mel_bins(),
            output_dim: default_output_dim(),
            dropout: 0.0,
        }
    }
}

/// Full Qwen3-ASR configuration combining audio encoder and text decoder.
#[derive(Debug, Clone)]
pub struct Qwen3ASRConfig {
    /// Audio encoder configuration
    pub audio_encoder: AudioEncoderConfig,
    /// Text decoder (Qwen3 LLM) configuration
    pub text_config: Qwen3Config,
    /// Audio start token ID (default: 151669)
    pub audio_start_token_id: u32,
    /// Audio end token ID (default: 151670)
    pub audio_end_token_id: u32,
}

/// Raw HuggingFace config format for parsing.
#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    audio_config: Option<AudioEncoderConfig>,
    text_config: Option<RawQwen3Config>,
    audio_start_token_id: Option<u32>,
    audio_end_token_id: Option<u32>,
}

/// Raw Qwen3 config from HuggingFace format.
#[derive(Debug, Clone, Deserialize)]
struct RawQwen3Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
    sliding_window: Option<usize>,
    #[serde(default = "default_max_window_layers")]
    max_window_layers: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
    head_dim: Option<usize>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    use_sliding_window: bool,
}

fn default_max_position_embeddings() -> usize {
    4096
}
fn default_max_window_layers() -> usize {
    28
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}

impl Qwen3ASRConfig {
    /// Parse config from HuggingFace JSON format.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let raw: RawConfig = serde_json::from_str(json)?;

        let audio_encoder = raw.audio_config.unwrap_or_default();

        let text_config = if let Some(tc) = raw.text_config {
            let head_dim = tc
                .head_dim
                .unwrap_or(tc.hidden_size / tc.num_attention_heads);
            Qwen3Config {
                vocab_size: tc.vocab_size,
                hidden_size: tc.hidden_size,
                intermediate_size: tc.intermediate_size,
                num_hidden_layers: tc.num_hidden_layers,
                num_attention_heads: tc.num_attention_heads,
                num_key_value_heads: tc.num_key_value_heads,
                max_position_embeddings: tc.max_position_embeddings,
                sliding_window: tc.sliding_window,
                max_window_layers: tc.max_window_layers,
                rope_theta: tc.rope_theta,
                rms_norm_eps: tc.rms_norm_eps,
                tie_word_embeddings: tc.tie_word_embeddings,
                head_dim,
                attention_bias: tc.attention_bias,
                use_sliding_window: tc.use_sliding_window,
                hidden_act: Activation::Silu,
            }
        } else {
            Self::default_qwen3_config()
        };

        Ok(Self {
            audio_encoder,
            text_config,
            audio_start_token_id: raw.audio_start_token_id.unwrap_or(151669),
            audio_end_token_id: raw.audio_end_token_id.unwrap_or(151670),
        })
    }

    /// Default Qwen3 0.6B configuration.
    fn default_qwen3_config() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 151936,
            hidden_size: 1024,
            intermediate_size: 2816,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            max_position_embeddings: 4096,
            sliding_window: None,
            max_window_layers: 28,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            head_dim: 64, // 1024 / 16
            attention_bias: false,
            use_sliding_window: false,
            hidden_act: Activation::Silu,
        }
    }

    /// Create config for Qwen3-ASR-0.6B model.
    pub fn mutranscriber_0_6b() -> Self {
        Self {
            audio_encoder: AudioEncoderConfig::default(),
            text_config: Self::default_qwen3_config(),
            audio_start_token_id: 151669,
            audio_end_token_id: 151670,
        }
    }

    /// Create config for Qwen3-ASR-1.7B model.
    pub fn mutranscriber_1_7b() -> Self {
        Self {
            audio_encoder: AudioEncoderConfig {
                output_dim: 2560,
                ..Default::default()
            },
            text_config: Qwen3Config {
                vocab_size: 151936,
                hidden_size: 2560,
                intermediate_size: 6912,
                num_hidden_layers: 28,
                num_attention_heads: 20,
                num_key_value_heads: 4,
                max_position_embeddings: 4096,
                sliding_window: None,
                max_window_layers: 28,
                rope_theta: 10000.0,
                rms_norm_eps: 1e-6,
                tie_word_embeddings: true,
                head_dim: 128, // 2560 / 20
                attention_bias: false,
                use_sliding_window: false,
                hidden_act: Activation::Silu,
            },
            audio_start_token_id: 151669,
            audio_end_token_id: 151670,
        }
    }
}
