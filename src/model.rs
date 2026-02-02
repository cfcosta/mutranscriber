//! Full Qwen3-ASR model combining audio encoder with Qwen3 LLM decoder.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{Repo, RepoType, api::tokio::Api};
use tokenizers::{Tokenizer, models::bpe::BPE};

use crate::{
    audio_encoder::Qwen3AudioEncoder,
    config::Qwen3ASRConfig,
    mel::MelSpectrogram,
    qwen3_decoder::Qwen3Decoder,
};

/// Model variant for Qwen3-ASR.
#[derive(Debug, Clone, Copy, Default)]
pub enum ModelVariant {
    /// 0.6B parameter model (~2GB VRAM)
    #[default]
    Small,
    /// 1.7B parameter model (~4GB VRAM)
    Large,
}

impl ModelVariant {
    /// Get HuggingFace model ID.
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Small => "Qwen/Qwen3-ASR-0.6B",
            Self::Large => "Qwen/Qwen3-ASR-1.7B",
        }
    }

    /// Get the corresponding config.
    pub fn config(&self) -> Qwen3ASRConfig {
        match self {
            Self::Small => Qwen3ASRConfig::mutranscriber_0_6b(),
            Self::Large => Qwen3ASRConfig::mutranscriber_1_7b(),
        }
    }
}

/// Full Qwen3-ASR model for audio transcription.
pub struct Qwen3ASRModel {
    audio_encoder: Qwen3AudioEncoder,
    decoder: Qwen3Decoder,
    mel_extractor: MelSpectrogram,
    tokenizer: Tokenizer,
    config: Qwen3ASRConfig,
    device: Device,
}

impl Qwen3ASRModel {
    /// Load model from HuggingFace Hub.
    pub async fn from_pretrained(variant: ModelVariant, device: &Device) -> Result<Self> {
        let model_id = variant.model_id();
        let config = variant.config();

        tracing::info!("Loading Qwen3-ASR model: {}", model_id);

        // Download model files from HuggingFace
        let api = Api::new().map_err(|e| candle_core::Error::Msg(format!("HF Hub error: {}", e)))?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        // Get model file paths
        let config_path = repo
            .get("config.json")
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to get config.json: {}", e)))?;
        // Qwen3-ASR uses vocab.json + merges.txt instead of tokenizer.json
        let vocab_path = repo
            .get("vocab.json")
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to get vocab.json: {}", e)))?;
        let merges_path = repo
            .get("merges.txt")
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to get merges.txt: {}", e)))?;
        let model_path = repo.get("model.safetensors").await.map_err(|e| {
            candle_core::Error::Msg(format!("Failed to get model.safetensors: {}", e))
        })?;

        Self::load_from_parts(
            config,
            &config_path,
            &vocab_path,
            &merges_path,
            &model_path,
            device,
        )
    }

    /// Load model from local files with separate vocab and merges files.
    pub fn load_from_parts(
        config: Qwen3ASRConfig,
        _config_path: &Path,
        vocab_path: &Path,
        merges_path: &Path,
        model_path: &Path,
        device: &Device,
    ) -> Result<Self> {
        // Build tokenizer from vocab.json and merges.txt (GPT-2 style BPE)
        let bpe = BPE::from_file(
            vocab_path
                .to_str()
                .ok_or_else(|| candle_core::Error::Msg("Invalid vocab path".to_string()))?,
            merges_path
                .to_str()
                .ok_or_else(|| candle_core::Error::Msg("Invalid merges path".to_string()))?,
        )
        .build()
        .map_err(|e| candle_core::Error::Msg(format!("Failed to build BPE tokenizer: {}", e)))?;

        let tokenizer = Tokenizer::new(bpe);

        Self::load_with_tokenizer(config, tokenizer, model_path, device)
    }

    /// Load model from local files with a pre-built tokenizer.json.
    pub fn load(
        config: Qwen3ASRConfig,
        _config_path: &Path,
        tokenizer_path: &Path,
        model_path: &Path,
        device: &Device,
    ) -> Result<Self> {
        // Load tokenizer from tokenizer.json
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;

        Self::load_with_tokenizer(config, tokenizer, model_path, device)
    }

    /// Load model with a provided tokenizer.
    fn load_with_tokenizer(
        config: Qwen3ASRConfig,
        tokenizer: Tokenizer,
        model_path: &Path,
        device: &Device,
    ) -> Result<Self> {
        // Load model weights
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)? };

        // Model uses "thinker" prefix for all weights
        let vb = vb.pp("thinker");

        // Initialize audio encoder (weights under "thinker.audio_tower")
        let audio_encoder = Qwen3AudioEncoder::new(&config.audio_encoder, vb.pp("audio_tower"))?;

        // Initialize decoder with embedding injection support
        let decoder = Qwen3Decoder::new(&config.text_config, vb.clone())?;

        // Initialize mel spectrogram extractor
        let mel_extractor = MelSpectrogram::new();

        Ok(Self {
            audio_encoder,
            decoder,
            mel_extractor,
            tokenizer,
            config,
            device: device.clone(),
        })
    }

    /// Transcribe audio samples to text.
    ///
    /// Input: f32 audio samples at 16kHz
    /// Output: Transcribed text
    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        tracing::debug!(
            "Audio samples: {}, duration: {:.2}s",
            audio.len(),
            audio.len() as f64 / 16000.0
        );

        // Extract mel spectrogram
        let (mel_data, n_frames, n_mels) = self.mel_extractor.compute_2d(audio);
        tracing::debug!("Mel spectrogram: {} frames x {} mels", n_frames, n_mels);

        // Create tensor: (1, n_mels, n_frames)
        let mel = Tensor::from_vec(mel_data, (1, n_frames, n_mels), &self.device)?;
        let mel = mel.transpose(1, 2)?; // (1, n_mels, n_frames)
        tracing::debug!("Mel tensor shape: {:?}", mel.dims());

        // Encode audio
        let audio_features = self.audio_encoder.forward(&mel)?;
        tracing::debug!("Audio features shape: {:?}", audio_features.dims());

        // Check audio features stats
        let features_flat = audio_features.flatten_all()?;
        let mean = features_flat.mean(0)?.to_scalar::<f32>()?;
        let min = features_flat.min(0)?.to_scalar::<f32>()?;
        let max = features_flat.max(0)?.to_scalar::<f32>()?;
        tracing::debug!(
            "Audio features stats - mean: {:.4}, min: {:.4}, max: {:.4}",
            mean,
            min,
            max
        );

        // Generate transcription using LLM
        self.generate(audio_features)
    }

    /// Generate text from audio features.
    fn generate(&mut self, audio_features: Tensor) -> Result<String> {
        let device = audio_features.device().clone();
        let n_audio_frames = audio_features.dim(1)?;

        // ChatML-style prompt format:
        // <|im_start|>user\n<|audio_start|>[audio]<|audio_end|><|im_end|>\n<|im_start|>assistant\n
        let im_start_token: u32 = 151644; // <|im_start|>
        let im_end_token: u32 = 151645; // <|im_end|>
        let audio_start_token = self.config.audio_start_token_id; // 151669
        let audio_end_token = self.config.audio_end_token_id; // 151670

        // Encode the prompt tokens - use add_special_tokens=false since we're manually adding special tokens
        let user_tokens = self
            .tokenizer
            .encode("user", false)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {}", e)))?;
        let assistant_tokens = self
            .tokenizer
            .encode("assistant", false)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {}", e)))?;
        let newline_token: u32 = 198; // "Ċ" in GPT-style tokenizers

        tracing::debug!("user tokens: {:?}", user_tokens.get_ids());
        tracing::debug!("assistant tokens: {:?}", assistant_tokens.get_ids());

        // Encode the instruction
        let instruction = "Transcribe this audio to text.\n";
        let instruction_tokens = self
            .tokenizer
            .encode(instruction, false)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {}", e)))?;
        tracing::debug!("instruction tokens: {:?}", instruction_tokens.get_ids());

        // Build the prompt sequence before audio
        // Format: <|im_start|>user\n<instruction>\n<|audio_start|>
        let mut pre_audio_tokens: Vec<u32> = vec![im_start_token];
        pre_audio_tokens.extend(user_tokens.get_ids().iter().copied());
        pre_audio_tokens.push(newline_token);
        pre_audio_tokens.extend(instruction_tokens.get_ids().iter().copied());
        pre_audio_tokens.push(audio_start_token);

        // Build the prompt sequence after audio
        // Format: <|audio_end|><|im_end|>\n<|im_start|>assistant\n
        let mut post_audio_tokens: Vec<u32> =
            vec![audio_end_token, im_end_token, newline_token, im_start_token];
        post_audio_tokens.extend(assistant_tokens.get_ids().iter().copied());
        post_audio_tokens.push(newline_token);

        tracing::debug!("Pre-audio tokens: {:?}", pre_audio_tokens);
        tracing::debug!("Post-audio tokens: {:?}", post_audio_tokens);

        // Reset KV cache
        self.decoder.clear_kv_cache();

        // Get embeddings for all prompt parts
        let pre_tensor = Tensor::new(pre_audio_tokens.as_slice(), &device)?.unsqueeze(0)?;
        let post_tensor = Tensor::new(post_audio_tokens.as_slice(), &device)?.unsqueeze(0)?;

        let pre_embed = self.decoder.get_token_embeddings(&pre_tensor)?;
        let post_embed = self.decoder.get_token_embeddings(&post_tensor)?;

        // Concatenate: [pre_tokens, audio_features, post_tokens]
        let combined = Tensor::cat(&[pre_embed, audio_features, post_embed], 1)?;
        let total_prompt_len = pre_audio_tokens.len() + n_audio_frames + post_audio_tokens.len();

        tracing::debug!("Combined embedding shape: {:?}", combined.dims());

        // Process the combined sequence through the decoder
        let mut logits = self.decoder.forward_embeds(&combined, 0)?;

        // Generate with greedy decoding
        let max_new_tokens = 256;
        let eos_token_id = self
            .tokenizer
            .token_to_id("<|endoftext|>")
            .unwrap_or(151643);
        let im_end_id = im_end_token;

        tracing::debug!(
            "Starting generation, eos_token_id: {}, im_end_id: {}",
            eos_token_id,
            im_end_id
        );

        let mut generated_tokens = Vec::new();
        let mut position = total_prompt_len;

        for i in 0..max_new_tokens {
            // Get logits for the last token
            let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;

            // Greedy decoding: select token with highest probability
            let next_token = last_logits
                .argmax(candle_core::D::Minus1)?
                .to_vec1::<u32>()?[0];

            if i < 50 {
                tracing::debug!(
                    "Token {}: {} (decoded: {:?})",
                    i,
                    next_token,
                    self.tokenizer.decode(&[next_token], false).ok()
                );
            }

            // Stop on EOS or <|im_end|>
            if next_token == eos_token_id || next_token == im_end_id {
                tracing::debug!("Stop token reached at step {}", i);
                break;
            }

            // Skip special tokens in output but don't stop
            if next_token >= 151643 {
                tracing::debug!("Skipping special token {}", next_token);
                // Still update position and continue
                let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
                logits = self.decoder.forward(&next_input, position)?;
                position += 1;
                continue;
            }

            generated_tokens.push(next_token);

            // Prepare next input
            let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            logits = self.decoder.forward(&next_input, position)?;
            position += 1;
        }

        tracing::debug!(
            "Generated {} tokens: {:?}",
            generated_tokens.len(),
            generated_tokens
        );

        // Decode tokens to text
        let text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenizer decode error: {}", e)))?;

        // Clean up GPT-style tokenizer artifacts:
        // - Replace "Ġ" with space (this is how GPT tokenizers encode spaces)
        // - Remove "language <lang>" prefix if present
        // - Normalize whitespace
        let text = text.replace('Ġ', " ");

        // Normalize whitespace first, then remove language prefix
        let words: Vec<&str> = text.split_whitespace().collect();

        // Remove "language XYZ" prefix (the model outputs language detection first)
        // The format is: "language <lang_name> <transcription...>"
        let text = if words.len() >= 2 && words[0] == "language" {
            // Skip "language" and the language name
            words[2..].join(" ")
        } else {
            words.join(" ")
        };

        Ok(text)
    }

    /// Get the device the model is running on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the model configuration.
    pub fn config(&self) -> &Qwen3ASRConfig {
        &self.config
    }
}

/// Builder for creating Qwen3ASRModel with custom options.
pub struct Qwen3ASRModelBuilder {
    variant: ModelVariant,
    device: Option<Device>,
    use_gpu: bool,
}

impl Qwen3ASRModelBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            variant: ModelVariant::default(),
            device: None,
            use_gpu: true,
        }
    }

    /// Set the model variant.
    pub fn variant(mut self, variant: ModelVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Set whether to use GPU if available.
    pub fn use_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set a specific device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Build the model.
    pub async fn build(self) -> Result<Qwen3ASRModel> {
        let device = if let Some(d) = self.device {
            d
        } else if self.use_gpu {
            Device::cuda_if_available(0)?
        } else {
            Device::Cpu
        };

        Qwen3ASRModel::from_pretrained(self.variant, &device).await
    }
}

impl Default for Qwen3ASRModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_variant_ids() {
        assert_eq!(ModelVariant::Small.model_id(), "Qwen/Qwen3-ASR-0.6B");
        assert_eq!(ModelVariant::Large.model_id(), "Qwen/Qwen3-ASR-1.7B");
    }

    #[test]
    fn test_model_variant_config() {
        let small_config = ModelVariant::Small.config();
        assert_eq!(small_config.audio_encoder.output_dim, 1024);
        assert_eq!(small_config.audio_encoder.d_model, 896);

        let large_config = ModelVariant::Large.config();
        assert_eq!(large_config.audio_encoder.output_dim, 2560);
    }
}
