//! Full Qwen3-ASR model combining audio encoder with Qwen3 LLM decoder.

use std::{
    io::Write,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use tokenizers::{
    models::bpe::BPE,
    pre_tokenizers::byte_level::ByteLevel,
    Tokenizer,
};

use crate::{
    audio_encoder::Qwen3AudioEncoder,
    config::{special_tokens, Qwen3ASRConfig},
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
    generation_config: crate::config::GenerationConfig,
    device: Device,
}

impl Qwen3ASRModel {
    /// Load model from HuggingFace Hub.
    pub async fn from_pretrained(
        variant: ModelVariant,
        device: &Device,
    ) -> Result<Self> {
        let model_id = variant.model_id();
        let config = variant.config();

        tracing::info!("Loading Qwen3-ASR model: {}", model_id);

        // Download model files from HuggingFace
        let api = Api::new().map_err(|e| {
            candle_core::Error::Msg(format!("HF Hub error: {}", e))
        })?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        // Get model file paths (config.json is downloaded but we use built-in config)
        let _config_path = repo.get("config.json").await.map_err(|e| {
            candle_core::Error::Msg(format!("Failed to get config.json: {}", e))
        })?;
        // Qwen3-ASR uses vocab.json + merges.txt instead of tokenizer.json
        let vocab_path = repo.get("vocab.json").await.map_err(|e| {
            candle_core::Error::Msg(format!("Failed to get vocab.json: {}", e))
        })?;
        let merges_path = repo.get("merges.txt").await.map_err(|e| {
            candle_core::Error::Msg(format!("Failed to get merges.txt: {}", e))
        })?;
        // Try single model file first, fall back to sharded weights
        let model_paths = match repo.get("model.safetensors").await {
            Ok(path) => vec![path],
            Err(_) => {
                // Model uses sharded weights - download the index and all shards
                tracing::info!(
                    "Model uses sharded weights, downloading shards..."
                );

                let index_path = repo
                    .get("model.safetensors.index.json")
                    .await
                    .map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "Failed to get model.safetensors.index.json: {}",
                            e
                        ))
                    })?;

                // Parse index to get shard filenames
                let index_content = std::fs::read_to_string(&index_path)
                    .map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "Failed to read index file: {}",
                            e
                        ))
                    })?;
                let index: serde_json::Value =
                    serde_json::from_str(&index_content).map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "Failed to parse index JSON: {}",
                            e
                        ))
                    })?;

                // Extract unique shard filenames from weight_map
                let weight_map =
                    index["weight_map"].as_object().ok_or_else(|| {
                        candle_core::Error::Msg(
                            "Invalid index format".to_string(),
                        )
                    })?;

                let mut shard_files: Vec<String> = weight_map
                    .values()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                shard_files.sort();
                shard_files.dedup();

                tracing::info!(
                    "Downloading {} shard files...",
                    shard_files.len()
                );

                // Download all shards
                let mut shard_paths = Vec::with_capacity(shard_files.len());
                for shard_file in &shard_files {
                    let shard_path =
                        repo.get(shard_file).await.map_err(|e| {
                            candle_core::Error::Msg(format!(
                                "Failed to get {}: {}",
                                shard_file, e
                            ))
                        })?;
                    shard_paths.push(shard_path);
                }

                shard_paths
            }
        };

        Self::load_from_paths(
            config,
            &vocab_path,
            &merges_path,
            &model_paths,
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
        Self::load_from_paths(
            config,
            vocab_path,
            merges_path,
            &[model_path.to_path_buf()],
            device,
        )
    }

    /// Load model from local files with potentially multiple model weight files (sharded).
    fn load_from_paths(
        config: Qwen3ASRConfig,
        vocab_path: &Path,
        merges_path: &Path,
        model_paths: &[PathBuf],
        device: &Device,
    ) -> Result<Self> {
        // Build tokenizer from vocab.json and merges.txt (GPT-2 style BPE)
        let bpe = BPE::from_file(
            vocab_path.to_str().ok_or_else(|| {
                candle_core::Error::Msg("Invalid vocab path".to_string())
            })?,
            merges_path.to_str().ok_or_else(|| {
                candle_core::Error::Msg("Invalid merges path".to_string())
            })?,
        )
        .build()
        .map_err(|e| {
            candle_core::Error::Msg(format!(
                "Failed to build BPE tokenizer: {}",
                e
            ))
        })?;

        // Configure tokenizer with ByteLevel decoder to properly handle GPT-2 BPE
        // byte-to-unicode mapping (e.g., Ġ for space). Without this, subword tokens
        // get separated by spaces incorrectly.
        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.with_decoder(Some(ByteLevel::default()));

        Self::load_with_tokenizer(config, tokenizer, model_paths, device)
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
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e))
        })?;

        Self::load_with_tokenizer(
            config,
            tokenizer,
            &[model_path.to_path_buf()],
            device,
        )
    }

    /// Load model with a provided tokenizer.
    fn load_with_tokenizer(
        config: Qwen3ASRConfig,
        tokenizer: Tokenizer,
        model_paths: &[PathBuf],
        device: &Device,
    ) -> Result<Self> {
        // Load model weights (supports both single file and sharded weights)
        let path_refs: Vec<&Path> =
            model_paths.iter().map(|p: &PathBuf| p.as_path()).collect();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&path_refs, DType::F32, device)?
        };

        // Model uses "thinker" prefix for all weights
        let vb = vb.pp("thinker");

        // Initialize audio encoder (weights under "thinker.audio_tower")
        let audio_encoder = Qwen3AudioEncoder::new(
            &config.audio_encoder,
            vb.pp("audio_tower"),
        )?;

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
            generation_config: crate::config::GenerationConfig::default(),
            device: device.clone(),
        })
    }

    /// Set generation configuration.
    pub fn set_generation_config(
        &mut self,
        config: crate::config::GenerationConfig,
    ) {
        self.generation_config = config;
    }

    /// Get current generation configuration.
    pub fn generation_config(&self) -> &crate::config::GenerationConfig {
        &self.generation_config
    }

    /// Dump diagnostic tensor statistics to a JSON file for comparison with reference.
    ///
    /// This helps debug transcription accuracy issues by comparing intermediate
    /// tensor values with the official Python implementation.
    fn dump_diagnostics(
        &self,
        mel: &Tensor,
        audio_features: &Tensor,
        path: &std::path::Path,
    ) -> Result<()> {
        let mut file = std::fs::File::create(path).map_err(|e| {
            candle_core::Error::Msg(format!(
                "Failed to create diagnostics file: {}",
                e
            ))
        })?;

        // Mel spectrogram statistics
        let mel_flat = mel.flatten_all()?;
        let mel_vec = mel_flat.to_vec1::<f32>()?;
        let mel_mean: f32 = mel_vec.iter().sum::<f32>() / mel_vec.len() as f32;
        let mel_min = mel_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let mel_max = mel_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mel_std =
            (mel_vec.iter().map(|x| (x - mel_mean).powi(2)).sum::<f32>()
                / mel_vec.len() as f32)
                .sqrt();

        // First frame of mel (for detailed comparison)
        let mel_first_frame: Vec<f32> = mel
            .i((0, .., 0))?
            .to_vec1::<f32>()?
            .into_iter()
            .take(16)
            .collect();

        // Audio features statistics
        let af_flat = audio_features.flatten_all()?;
        let af_vec = af_flat.to_vec1::<f32>()?;
        let af_mean: f32 = af_vec.iter().sum::<f32>() / af_vec.len() as f32;
        let af_min = af_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let af_max = af_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let af_std =
            (af_vec.iter().map(|x| (x - af_mean).powi(2)).sum::<f32>()
                / af_vec.len() as f32)
                .sqrt();

        // First frame of audio features (for detailed comparison)
        let af_first_frame: Vec<f32> = audio_features
            .i((0, 0, ..))?
            .to_vec1::<f32>()?
            .into_iter()
            .take(16)
            .collect();

        // Write JSON
        writeln!(file, "{{").ok();
        writeln!(file, "  \"mel_spectrogram\": {{").ok();
        writeln!(file, "    \"shape\": {:?},", mel.dims()).ok();
        writeln!(file, "    \"mean\": {:.6},", mel_mean).ok();
        writeln!(file, "    \"std\": {:.6},", mel_std).ok();
        writeln!(file, "    \"min\": {:.6},", mel_min).ok();
        writeln!(file, "    \"max\": {:.6},", mel_max).ok();
        writeln!(file, "    \"first_frame_16\": {:?}", mel_first_frame).ok();
        writeln!(file, "  }},").ok();
        writeln!(file, "  \"audio_features\": {{").ok();
        writeln!(file, "    \"shape\": {:?},", audio_features.dims()).ok();
        writeln!(file, "    \"mean\": {:.6},", af_mean).ok();
        writeln!(file, "    \"std\": {:.6},", af_std).ok();
        writeln!(file, "    \"min\": {:.6},", af_min).ok();
        writeln!(file, "    \"max\": {:.6},", af_max).ok();
        writeln!(file, "    \"first_frame_16\": {:?}", af_first_frame).ok();
        writeln!(file, "  }}").ok();
        writeln!(file, "}}").ok();

        Ok(())
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
        tracing::debug!(
            "Mel spectrogram: {} frames x {} mels",
            n_frames,
            n_mels
        );

        // Create tensor: (1, n_mels, n_frames)
        let mel =
            Tensor::from_vec(mel_data, (1, n_frames, n_mels), &self.device)?;
        let mel = mel.transpose(1, 2)?; // (1, n_mels, n_frames)
        tracing::debug!("Mel tensor shape: {:?}", mel.dims());

        // Encode audio
        let audio_features = self.audio_encoder.forward(&mel)?;

        // Validate audio features shape: should be (batch, seq, hidden_size)
        let af_dims = audio_features.dims();
        if af_dims.len() != 3 {
            return Err(candle_core::Error::Msg(format!(
                "Expected 3D audio features, got shape {:?}",
                af_dims
            )));
        }
        let expected_hidden = self.config.audio_encoder.output_dim;
        if af_dims[2] != expected_hidden {
            return Err(candle_core::Error::Msg(format!(
                "Audio features hidden size mismatch: got {}, expected {}",
                af_dims[2], expected_hidden
            )));
        }
        tracing::debug!("Audio features shape: {:?}", af_dims);

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

        // Dump diagnostics if MUTRANSCRIBER_DIAGNOSTICS env var is set
        if let Ok(diag_path) = std::env::var("MUTRANSCRIBER_DIAGNOSTICS") {
            self.dump_diagnostics(
                &mel,
                &audio_features,
                std::path::Path::new(&diag_path),
            )?;
            eprintln!("Diagnostics written to: {}", diag_path);
        }

        // Generate transcription using LLM
        self.generate(audio_features)
    }

    /// Generate text from audio features.
    fn generate(&mut self, audio_features: Tensor) -> Result<String> {
        let device = audio_features.device().clone();
        let n_audio_frames = audio_features.dim(1)?;

        // ChatML-style prompt format (from chat_template.json):
        // <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>[audio]<|audio_end|><|im_end|>\n<|im_start|>assistant\n
        let im_start_token = special_tokens::IM_START;
        let im_end_token = special_tokens::IM_END;
        let audio_start_token = special_tokens::AUDIO_START;
        let audio_end_token = special_tokens::AUDIO_END;

        // Encode the prompt tokens
        let system_tokens =
            self.tokenizer.encode("system", false).map_err(|e| {
                candle_core::Error::Msg(format!("Tokenizer error: {}", e))
            })?;
        let user_tokens =
            self.tokenizer.encode("user", false).map_err(|e| {
                candle_core::Error::Msg(format!("Tokenizer error: {}", e))
            })?;
        let assistant_tokens =
            self.tokenizer.encode("assistant", false).map_err(|e| {
                candle_core::Error::Msg(format!("Tokenizer error: {}", e))
            })?;
        let newline_token = special_tokens::NEWLINE;

        tracing::debug!("system tokens: {:?}", system_tokens.get_ids());
        tracing::debug!("user tokens: {:?}", user_tokens.get_ids());
        tracing::debug!("assistant tokens: {:?}", assistant_tokens.get_ids());

        // Build the prompt sequence before audio
        // Format: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
        let mut pre_audio_tokens: Vec<u32> = vec![im_start_token];
        pre_audio_tokens.extend(system_tokens.get_ids().iter().copied());
        pre_audio_tokens.push(newline_token);
        pre_audio_tokens.push(im_end_token);
        pre_audio_tokens.push(newline_token);
        pre_audio_tokens.push(im_start_token);
        pre_audio_tokens.extend(user_tokens.get_ids().iter().copied());
        pre_audio_tokens.push(newline_token);
        pre_audio_tokens.push(audio_start_token);

        // Encode language hint tokens for priming the model
        let language_english_tokens = self
            .tokenizer
            .encode("language English", false)
            .map_err(|e| {
                candle_core::Error::Msg(format!("Tokenizer error: {}", e))
            })?;

        // ASR text marker token signals start of transcription output
        let asr_text_token = special_tokens::ASR_TEXT;

        // Build the prompt sequence after audio
        // Format: <|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>
        // Adding explicit language hint and ASR marker to prime the model for English transcription
        let mut post_audio_tokens: Vec<u32> =
            vec![audio_end_token, im_end_token, newline_token, im_start_token];
        post_audio_tokens.extend(assistant_tokens.get_ids().iter().copied());
        post_audio_tokens.push(newline_token);
        post_audio_tokens
            .extend(language_english_tokens.get_ids().iter().copied());
        post_audio_tokens.push(asr_text_token);

        tracing::debug!("Pre-audio tokens: {:?}", pre_audio_tokens);
        tracing::debug!("Post-audio tokens: {:?}", post_audio_tokens);

        // Reset KV cache
        self.decoder.clear_kv_cache();

        // Get embeddings for all prompt parts
        let pre_tensor =
            Tensor::new(pre_audio_tokens.as_slice(), &device)?.unsqueeze(0)?;
        let post_tensor =
            Tensor::new(post_audio_tokens.as_slice(), &device)?.unsqueeze(0)?;

        let pre_embed = self.decoder.get_token_embeddings(&pre_tensor)?;
        let post_embed = self.decoder.get_token_embeddings(&post_tensor)?;

        // Validate tensor shapes before concatenation
        let (_, _, pre_hidden) = pre_embed.dims3()?;
        let (_, _, audio_hidden) = audio_features.dims3()?;
        let (_, _, post_hidden) = post_embed.dims3()?;

        if pre_hidden != audio_hidden || audio_hidden != post_hidden {
            return Err(candle_core::Error::Msg(format!(
                "Hidden size mismatch: pre_embed={}, audio_features={}, post_embed={}. \
                 Expected audio_features to match text hidden_size={}",
                pre_hidden, audio_hidden, post_hidden, pre_hidden
            )));
        }

        // Concatenate: [pre_tokens, audio_features, post_tokens]
        let combined =
            Tensor::cat(&[pre_embed, audio_features, post_embed], 1)?;
        let total_prompt_len =
            pre_audio_tokens.len() + n_audio_frames + post_audio_tokens.len();

        tracing::debug!("Combined embedding shape: {:?}", combined.dims());

        // Process the combined sequence through the decoder
        let mut logits = self.decoder.forward_embeds(&combined, 0)?;

        // Validate logits shape: should be (batch, seq, vocab_size)
        let logits_dims = logits.dims();
        if logits_dims.len() != 3 {
            return Err(candle_core::Error::Msg(format!(
                "Expected 3D logits tensor, got shape {:?}",
                logits_dims
            )));
        }

        // Use generation config
        let gen_config = &self.generation_config;
        let eos_token_id = gen_config.eos_token_id;

        tracing::debug!(
            "Starting generation: max_tokens={}, temperature={:?}, eos={}",
            gen_config.max_new_tokens,
            gen_config.temperature,
            eos_token_id
        );

        let mut generated_tokens = Vec::new();
        let mut position = total_prompt_len;

        for i in 0..gen_config.max_new_tokens {
            // Get logits for the last token
            let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;

            // Sample next token based on generation config
            let next_token =
                self.sample_token(&last_logits, &generated_tokens)?;

            if i < 50 {
                tracing::debug!(
                    "Token {}: {} (decoded: {:?})",
                    i,
                    next_token,
                    self.tokenizer.decode(&[next_token], false).ok()
                );
            }

            // Stop on EOS or any configured stop token
            if next_token == eos_token_id
                || gen_config.stop_token_ids.contains(&next_token)
            {
                tracing::debug!(
                    "Stop token {} reached at step {}",
                    next_token,
                    i
                );
                break;
            }

            // Skip special tokens in output but still feed them to the decoder.
            // Special tokens (>= eos_token_id) are control tokens that shouldn't
            // appear in the transcription text, but they provide context to the
            // model for continued generation.
            if self.generation_config.is_special_token(next_token) {
                tracing::debug!("Skipping special token {}", next_token);
                let next_input =
                    Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
                logits = self.decoder.forward(&next_input, position)?;
                position += 1;
                continue;
            }

            generated_tokens.push(next_token);

            // Prepare next input
            let next_input =
                Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            logits = self.decoder.forward(&next_input, position)?;
            position += 1;
        }

        tracing::debug!(
            "Generated {} tokens: {:?}",
            generated_tokens.len(),
            generated_tokens
        );

        // Decode tokens to text - ByteLevel decoder handles byte-to-unicode mapping
        let text =
            self.tokenizer
                .decode(&generated_tokens, true)
                .map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "Tokenizer decode error: {}",
                        e
                    ))
                })?;

        // Normalize whitespace
        // Note: We prime the model with "language English<asr_text>" so it should
        // start generating the transcription directly. If it still outputs a language
        // prefix, strip it.
        let words: Vec<&str> = text.split_whitespace().collect();

        let text = if words.len() >= 2 && words[0] == "language" {
            // Skip any remaining "language <name>" prefix
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

    /// Sample a token from logits using the generation config.
    fn sample_token(
        &self,
        logits: &Tensor,
        generated_tokens: &[u32],
    ) -> Result<u32> {
        let gen_config = &self.generation_config;

        // Apply repetition penalty if configured
        let logits = if let Some(penalty) = gen_config.repetition_penalty {
            if penalty != 1.0 && !generated_tokens.is_empty() {
                let mut logits_vec = logits.squeeze(0)?.to_vec1::<f32>()?;
                for &token in generated_tokens {
                    let idx = token as usize;
                    if idx < logits_vec.len() {
                        if logits_vec[idx] > 0.0 {
                            logits_vec[idx] /= penalty;
                        } else {
                            logits_vec[idx] *= penalty;
                        }
                    }
                }
                Tensor::from_vec(logits_vec, logits.dims(), logits.device())?
            } else {
                logits.clone()
            }
        } else {
            logits.clone()
        };

        // Greedy decoding (no temperature)
        if gen_config.temperature.is_none() {
            let token =
                logits.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?[0];
            return Ok(token);
        }

        // Temperature scaling
        let temperature = gen_config.temperature.unwrap();
        let logits = if temperature != 1.0 {
            (logits / temperature as f64)?
        } else {
            logits
        };

        // Apply top-k filtering
        let logits = if let Some(k) = gen_config.top_k {
            self.top_k_filter(&logits, k)?
        } else {
            logits
        };

        // Apply top-p (nucleus) filtering
        let logits = if let Some(p) = gen_config.top_p {
            self.top_p_filter(&logits, p)?
        } else {
            logits
        };

        // Sample from the distribution
        let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;
        let probs_vec = probs.squeeze(0)?.to_vec1::<f32>()?;

        // Simple sampling using random selection weighted by probabilities
        let mut rng_val: f32 = 0.0;
        // Use a simple deterministic "random" for reproducibility in tests
        // In production, you'd want a proper RNG
        for (i, &p) in probs_vec.iter().enumerate() {
            rng_val += p;
            // Use cumulative sum approach with threshold at 0.5 for now
            // This is a simplified sampling that picks the median token
            if rng_val >= 0.5 {
                return Ok(i as u32);
            }
        }

        // Fallback to most likely token
        Ok(logits.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?[0])
    }

    /// Apply top-k filtering to logits.
    fn top_k_filter(&self, logits: &Tensor, k: usize) -> Result<Tensor> {
        let logits_vec = logits.squeeze(0)?.to_vec1::<f32>()?;
        let mut indexed: Vec<(usize, f32)> =
            logits_vec.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut filtered = vec![f32::NEG_INFINITY; logits_vec.len()];
        for (idx, val) in indexed.into_iter().take(k) {
            filtered[idx] = val;
        }

        Tensor::from_vec(filtered, logits.dims(), logits.device())
    }

    /// Apply top-p (nucleus) filtering to logits.
    fn top_p_filter(&self, logits: &Tensor, p: f32) -> Result<Tensor> {
        let probs = candle_nn::ops::softmax(logits, candle_core::D::Minus1)?;
        let probs_vec = probs.squeeze(0)?.to_vec1::<f32>()?;

        let mut indexed: Vec<(usize, f32)> =
            probs_vec.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumsum = 0.0;
        let mut filtered = vec![f32::NEG_INFINITY; probs_vec.len()];
        let logits_vec = logits.squeeze(0)?.to_vec1::<f32>()?;

        for (idx, prob) in indexed {
            if cumsum < p {
                filtered[idx] = logits_vec[idx];
                cumsum += prob;
            } else {
                break;
            }
        }

        Tensor::from_vec(filtered, logits.dims(), logits.device())
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
        assert_eq!(large_config.audio_encoder.output_dim, 2048);
        assert_eq!(large_config.audio_encoder.d_model, 1024);
    }
}
