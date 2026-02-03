//! Configuration structs for Qwen3-ASR model.

use candle_nn::Activation;
use candle_transformers::models::qwen3::Config as Qwen3Config;
use serde::Deserialize;

/// Special token IDs for Qwen3-ASR models.
///
/// These tokens are defined in the tokenizer_config.json from HuggingFace.
/// All special tokens have IDs >= 151643 (ENDOFTEXT).
pub mod special_tokens {
    // ChatML tokens
    /// `<|endoftext|>` - End of text / padding token
    pub const ENDOFTEXT: u32 = 151643;
    /// `<|im_start|>` - ChatML message start
    pub const IM_START: u32 = 151644;
    /// `<|im_end|>` - ChatML message end (also used as EOS)
    pub const IM_END: u32 = 151645;

    // Audio tokens
    /// `<|audio_start|>` - Audio segment start marker
    pub const AUDIO_START: u32 = 151669;
    /// `<|audio_end|>` - Audio segment end marker
    pub const AUDIO_END: u32 = 151670;
    /// `<|audio_pad|>` - Placeholder replaced by audio embeddings
    pub const AUDIO_PAD: u32 = 151676;

    // ASR-specific tokens
    /// `<non_speech>` - Marks non-speech audio content
    pub const NON_SPEECH: u32 = 151675;
    /// `<asr_text>` - ASR task marker, signals start of transcription output
    pub const ASR_TEXT: u32 = 151704;

    // Text formatting
    /// Newline token (GPT-2 style "Ċ")
    pub const NEWLINE: u32 = 198;

    /// First special token ID (all tokens >= this are special)
    pub const FIRST_SPECIAL: u32 = ENDOFTEXT;
}

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
    /// Maximum source positions for positional embeddings (default: 1500)
    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: usize,
    /// Dropout rate (default: 0.0)
    #[serde(default)]
    pub dropout: f64,
}

fn default_d_model() -> usize {
    896 // Actual Qwen3-ASR-0.6B value
}
fn default_encoder_layers() -> usize {
    18 // Actual Qwen3-ASR-0.6B value
}
fn default_encoder_attention_heads() -> usize {
    14 // Actual Qwen3-ASR-0.6B value
}
fn default_encoder_ffn_dim() -> usize {
    3584 // Actual Qwen3-ASR-0.6B value
}
fn default_num_mel_bins() -> usize {
    128
}
fn default_output_dim() -> usize {
    1024 // Actual Qwen3-ASR-0.6B value
}
fn default_max_source_positions() -> usize {
    1500 // From HuggingFace config
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
            max_source_positions: default_max_source_positions(),
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
    /// Audio token ID - placeholder that gets replaced by audio embeddings (default: 151676)
    pub audio_token_id: u32,
}

/// Raw HuggingFace config format for parsing.
#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    audio_config: Option<AudioEncoderConfig>,
    text_config: Option<RawQwen3Config>,
    audio_start_token_id: Option<u32>,
    audio_end_token_id: Option<u32>,
    audio_token_id: Option<u32>,
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
            audio_token_id: raw.audio_token_id.unwrap_or(151676),
        })
    }

    /// Default Qwen3 0.6B configuration.
    fn default_qwen3_config() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 151936,
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            max_position_embeddings: 65536,
            sliding_window: None,
            max_window_layers: 28,
            rope_theta: 1000000.0, // Qwen3 uses 1M, not 10K
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            head_dim: 128, // q_proj outputs 2048 (16 heads * 128 dim)
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
            audio_token_id: 151676,
        }
    }

    /// Create config for Qwen3-ASR-1.7B model.
    pub fn mutranscriber_1_7b() -> Self {
        Self {
            audio_encoder: AudioEncoderConfig {
                d_model: 1024,
                encoder_layers: 24,
                encoder_attention_heads: 16,
                encoder_ffn_dim: 4096,
                num_mel_bins: 128,
                output_dim: 2048, // Matches text hidden_size
                max_source_positions: 1500,
                dropout: 0.0,
            },
            text_config: Qwen3Config {
                vocab_size: 151936,
                hidden_size: 2048,
                intermediate_size: 6144,
                num_hidden_layers: 28,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                max_position_embeddings: 65536,
                sliding_window: None,
                max_window_layers: 28,
                rope_theta: 1000000.0, // Qwen3 uses 1M, not 10K
                rms_norm_eps: 1e-6,
                tie_word_embeddings: true,
                head_dim: 128,
                attention_bias: false,
                use_sliding_window: false,
                hidden_act: Activation::Silu,
            },
            audio_start_token_id: 151669,
            audio_end_token_id: 151670,
            audio_token_id: 151676,
        }
    }
}

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Temperature for sampling (None = greedy decoding).
    /// Higher values (e.g., 1.0) make output more random,
    /// lower values (e.g., 0.1) make it more deterministic.
    pub temperature: Option<f32>,
    /// Top-k sampling: only consider the k most likely tokens.
    pub top_k: Option<usize>,
    /// Top-p (nucleus) sampling: consider tokens with cumulative probability >= p.
    pub top_p: Option<f32>,
    /// Repetition penalty to discourage repeating tokens (1.0 = no penalty).
    pub repetition_penalty: Option<f32>,
    /// End-of-sequence token ID.
    pub eos_token_id: u32,
    /// Additional stop token IDs (e.g., <|im_end|>).
    pub stop_token_ids: Vec<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 4096,
            temperature: None, // Greedy decoding by default
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            eos_token_id: 151643, // <|endoftext|>
            // Note: Don't include <|im_end|> (151645) in stop tokens for ASR.
            // The model generates <|im_end|> after each ChatML response segment,
            // which would truncate transcription to just one sentence.
            // For ASR, we only stop on <|endoftext|> (eos_token_id).
            stop_token_ids: vec![],
        }
    }
}

impl GenerationConfig {
    /// Create config for greedy decoding (deterministic).
    pub fn greedy() -> Self {
        Self::default()
    }

    /// Create config for sampling with temperature.
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature: Some(temperature),
            ..Self::default()
        }
    }

    /// Set max new tokens.
    pub fn max_tokens(mut self, max: usize) -> Self {
        self.max_new_tokens = max;
        self
    }

    /// Set temperature (None for greedy).
    pub fn temperature(mut self, temp: Option<f32>) -> Self {
        self.temperature = temp;
        self
    }

    /// Set top-k sampling.
    pub fn top_k(mut self, k: Option<usize>) -> Self {
        self.top_k = k;
        self
    }

    /// Set top-p (nucleus) sampling.
    pub fn top_p(mut self, p: Option<f32>) -> Self {
        self.top_p = p;
        self
    }

    /// Set repetition penalty.
    pub fn repetition_penalty(mut self, penalty: Option<f32>) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Check if a token ID is a special token.
    ///
    /// Special tokens in Qwen3-ASR are those with IDs >= `eos_token_id` (151643).
    /// These include:
    /// - 151643: `<|endoftext|>` (EOS)
    /// - 151644: `<|im_start|>` (ChatML)
    /// - 151645: `<|im_end|>` (ChatML)
    /// - 151669: `<|audio_start|>`
    /// - 151670: `<|audio_end|>`
    /// - 151676: audio placeholder token
    /// - 151704: `<asr_text>` (ASR task marker)
    ///
    /// The vocab size is 151936, so special tokens span IDs 151643-151935.
    pub fn is_special_token(&self, token_id: u32) -> bool {
        token_id >= self.eos_token_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 4096);
        assert!(config.temperature.is_none());
        assert_eq!(config.eos_token_id, 151643);
    }

    #[test]
    fn test_special_token_detection() {
        let config = GenerationConfig::default();

        // Regular text tokens are not special
        assert!(!config.is_special_token(0));
        assert!(!config.is_special_token(1000));
        assert!(!config.is_special_token(151642)); // Just before EOS

        // EOS and beyond are special tokens
        assert!(config.is_special_token(151643)); // <|endoftext|>
        assert!(config.is_special_token(151644)); // <|im_start|>
        assert!(config.is_special_token(151645)); // <|im_end|>
        assert!(config.is_special_token(151669)); // <|audio_start|>
        assert!(config.is_special_token(151670)); // <|audio_end|>
        assert!(config.is_special_token(151676)); // audio placeholder
        assert!(config.is_special_token(151704)); // <asr_text>
        assert!(config.is_special_token(151935)); // Last valid vocab token
    }

    #[test]
    fn test_generation_config_builder() {
        let config = GenerationConfig::with_temperature(0.7)
            .max_tokens(512)
            .top_k(Some(50))
            .top_p(Some(0.9))
            .repetition_penalty(Some(1.1));

        assert_eq!(config.max_new_tokens, 512);
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.top_k, Some(50));
        assert_eq!(config.top_p, Some(0.9));
        assert_eq!(config.repetition_penalty, Some(1.1));
    }
}
