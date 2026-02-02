//! High-level async transcription API.
//!
//! This module provides an easy-to-use interface for transcribing audio
//! files, including extraction from MP4 containers.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use candle_core::Device;
use tokio::sync::Mutex;

use crate::model::{ModelVariant, Qwen3ASRModel};

/// Error type for transcription operations.
#[derive(Debug, thiserror::Error)]
pub enum TranscriberError {
    #[error("Model error: {0}")]
    Model(String),
    #[error("GStreamer error: {0}")]
    Gstreamer(String),
    #[error("Audio extraction failed: {0}")]
    AudioExtraction(String),
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("GStreamer feature not enabled")]
    GstreamerNotEnabled,
}

#[cfg(feature = "gstreamer")]
impl From<gstreamer::glib::Error> for TranscriberError {
    fn from(e: gstreamer::glib::Error) -> Self {
        Self::Gstreamer(e.to_string())
    }
}

/// Result type for transcription operations.
pub type TranscriberResult<T> = std::result::Result<T, TranscriberError>;

/// Configuration for the transcriber.
#[derive(Debug, Clone)]
pub struct TranscriberConfig {
    /// Model variant to use
    pub variant: ModelVariant,
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// Sample rate for audio extraction (default: 16000)
    pub sample_rate: u32,
    /// Output directory for transcripts (default: same as input)
    pub output_dir: Option<PathBuf>,
}

impl Default for TranscriberConfig {
    fn default() -> Self {
        Self {
            variant: ModelVariant::Small,
            use_gpu: true,
            sample_rate: 16000,
            output_dir: None,
        }
    }
}

impl TranscriberConfig {
    /// Create config from environment variables.
    pub fn from_env() -> Self {
        let variant = std::env::var("QWEN_ASR_MODEL")
            .map(|v| match v.to_lowercase().as_str() {
                "large" | "1.7b" => ModelVariant::Large,
                _ => ModelVariant::Small,
            })
            .unwrap_or(ModelVariant::Small);

        let use_gpu = std::env::var("QWEN_ASR_CPU")
            .map(|v| v != "1" && v.to_lowercase() != "true")
            .unwrap_or(true);

        Self {
            variant,
            use_gpu,
            ..Default::default()
        }
    }
}

/// High-level transcriber for audio files.
///
/// Lazily loads the model on first transcription request.
pub struct Transcriber {
    config: TranscriberConfig,
    model: Arc<Mutex<Option<Qwen3ASRModel>>>,
}

impl Transcriber {
    /// Create a new transcriber with default configuration.
    pub fn new() -> Self {
        Self::with_config(TranscriberConfig::default())
    }

    /// Create a new transcriber with custom configuration.
    pub fn with_config(config: TranscriberConfig) -> Self {
        Self {
            config,
            model: Arc::new(Mutex::new(None)),
        }
    }

    /// Create a transcriber from environment variables.
    pub fn from_env() -> Self {
        Self::with_config(TranscriberConfig::from_env())
    }

    /// Ensure the model is loaded.
    async fn ensure_model(&self) -> TranscriberResult<()> {
        let mut model_guard = self.model.lock().await;
        if model_guard.is_none() {
            tracing::info!("Loading transcription model (lazy initialization)...");

            let device = if self.config.use_gpu {
                Device::cuda_if_available(0).map_err(|e| TranscriberError::Model(e.to_string()))?
            } else {
                Device::Cpu
            };

            let model = Qwen3ASRModel::from_pretrained(self.config.variant, &device)
                .await
                .map_err(|e| TranscriberError::Model(e.to_string()))?;

            *model_guard = Some(model);
            tracing::info!("Model loaded successfully");
        }
        Ok(())
    }

    /// Transcribe an audio or video file.
    ///
    /// For video files (MP4), audio is automatically extracted.
    /// Returns the path to the transcript file.
    ///
    /// Requires the `gstreamer` feature.
    #[cfg(feature = "gstreamer")]
    pub async fn transcribe(&self, input_path: &Path) -> TranscriberResult<PathBuf> {
        if !input_path.exists() {
            return Err(TranscriberError::FileNotFound(input_path.to_path_buf()));
        }

        // Extract audio samples
        let audio_samples = self.extract_audio(input_path).await?;

        // Ensure model is loaded
        self.ensure_model().await?;

        // Transcribe
        let text = {
            let mut model_guard = self.model.lock().await;
            let model = model_guard.as_mut().expect("Model should be loaded");
            model
                .transcribe(&audio_samples)
                .map_err(|e| TranscriberError::Model(e.to_string()))?
        };

        // Write transcript
        let transcript_path = self.transcript_path(input_path);
        tokio::fs::write(&transcript_path, &text).await?;

        tracing::info!("Transcript saved to: {}", transcript_path.display());
        Ok(transcript_path)
    }

    /// Transcribe an audio or video file (stub when gstreamer is not enabled).
    #[cfg(not(feature = "gstreamer"))]
    pub async fn transcribe(&self, _input_path: &Path) -> TranscriberResult<PathBuf> {
        Err(TranscriberError::GstreamerNotEnabled)
    }

    /// Transcribe audio samples directly.
    ///
    /// Input: f32 audio samples at 16kHz
    /// Returns: Transcribed text
    pub async fn transcribe_audio(&self, audio: &[f32]) -> TranscriberResult<String> {
        self.ensure_model().await?;

        let mut model_guard = self.model.lock().await;
        let model = model_guard.as_mut().expect("Model should be loaded");
        model
            .transcribe(audio)
            .map_err(|e| TranscriberError::Model(e.to_string()))
    }

    /// Extract audio from a file using GStreamer.
    #[cfg(feature = "gstreamer")]
    async fn extract_audio(&self, path: &Path) -> TranscriberResult<Vec<f32>> {
        let path = path.to_path_buf();
        let sample_rate = self.config.sample_rate;

        // Run in blocking task since GStreamer operations are synchronous
        tokio::task::spawn_blocking(move || extract_audio_sync(&path, sample_rate))
            .await
            .map_err(|e| TranscriberError::AudioExtraction(e.to_string()))?
    }

    /// Get the transcript output path for an input file.
    #[cfg(any(feature = "gstreamer", test))]
    fn transcript_path(&self, input_path: &Path) -> PathBuf {
        let output_dir = self
            .config
            .output_dir
            .as_deref()
            .unwrap_or_else(|| input_path.parent().unwrap_or(Path::new(".")));

        let stem = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("transcript");

        output_dir.join(format!("{}.txt", stem))
    }

    /// Check if the model is loaded.
    pub async fn is_model_loaded(&self) -> bool {
        self.model.lock().await.is_some()
    }

    /// Preload the model (useful for reducing first-transcription latency).
    pub async fn preload(&self) -> TranscriberResult<()> {
        self.ensure_model().await
    }

    /// Get the sample rate used for audio extraction.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

impl Default for Transcriber {
    fn default() -> Self {
        Self::new()
    }
}

/// Synchronous audio extraction using GStreamer.
#[cfg(feature = "gstreamer")]
fn extract_audio_sync(path: &Path, sample_rate: u32) -> TranscriberResult<Vec<f32>> {
    use gstreamer as gst;
    use gstreamer::prelude::*;
    use gstreamer_app as gst_app;

    gst::init().map_err(|e| TranscriberError::Gstreamer(e.to_string()))?;

    let path_str = path
        .to_str()
        .ok_or_else(|| TranscriberError::AudioExtraction("Invalid path encoding".to_string()))?;

    // Build GStreamer pipeline for audio extraction
    // filesrc -> decodebin -> audioconvert -> audioresample -> appsink
    let pipeline_str = format!(
        "filesrc location=\"{}\" ! decodebin ! audioconvert ! \
         audioresample ! audio/x-raw,format=F32LE,rate={},channels=1 ! appsink name=sink",
        path_str, sample_rate
    );

    let pipeline = gst::parse::launch(&pipeline_str)?;
    let pipeline = pipeline
        .dynamic_cast::<gst::Pipeline>()
        .map_err(|_| TranscriberError::Gstreamer("Failed to cast to Pipeline".to_string()))?;

    let appsink = pipeline
        .by_name("sink")
        .ok_or_else(|| TranscriberError::Gstreamer("Failed to find appsink".to_string()))?;
    let appsink = appsink
        .dynamic_cast::<gst_app::AppSink>()
        .map_err(|_| TranscriberError::Gstreamer("Failed to cast to AppSink".to_string()))?;

    // Collect audio samples
    let samples = Arc::new(std::sync::Mutex::new(Vec::new()));
    let samples_clone = samples.clone();

    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                let sample = sink.pull_sample().map_err(|_| gst::FlowError::Error)?;
                let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;
                let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;

                // Convert bytes to f32 samples (F32LE format)
                let bytes = map.as_slice();
                let float_samples: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                samples_clone.lock().unwrap().extend(float_samples);
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    // Start pipeline
    pipeline
        .set_state(gst::State::Playing)
        .map_err(|e| TranscriberError::Gstreamer(format!("Failed to start pipeline: {:?}", e)))?;

    // Wait for EOS or error
    let bus = pipeline
        .bus()
        .ok_or_else(|| TranscriberError::Gstreamer("Failed to get bus".to_string()))?;

    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Eos(..) => break,
            MessageView::Error(err) => {
                pipeline.set_state(gst::State::Null).ok();
                return Err(TranscriberError::Gstreamer(format!(
                    "Pipeline error: {}",
                    err.error()
                )));
            }
            _ => {}
        }
    }

    // Stop pipeline
    pipeline
        .set_state(gst::State::Null)
        .map_err(|e| TranscriberError::Gstreamer(format!("Failed to stop pipeline: {:?}", e)))?;

    // Return collected samples
    let samples = Arc::try_unwrap(samples)
        .map_err(|_| TranscriberError::AudioExtraction("Failed to unwrap samples".to_string()))?
        .into_inner()
        .map_err(|e| TranscriberError::AudioExtraction(e.to_string()))?;

    if samples.is_empty() {
        return Err(TranscriberError::AudioExtraction(
            "No audio samples extracted".to_string(),
        ));
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = TranscriberConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert!(config.use_gpu);
    }

    #[test]
    fn test_transcript_path() {
        let transcriber = Transcriber::new();
        let input = Path::new("/home/user/Videos/recording.mp4");
        let output = transcriber.transcript_path(input);
        assert_eq!(output, PathBuf::from("/home/user/Videos/recording.txt"));
    }

    #[test]
    fn test_transcript_path_with_output_dir() {
        let config = TranscriberConfig {
            output_dir: Some(PathBuf::from("/tmp/transcripts")),
            ..Default::default()
        };
        let transcriber = Transcriber::with_config(config);
        let input = Path::new("/home/user/Videos/recording.mp4");
        let output = transcriber.transcript_path(input);
        assert_eq!(output, PathBuf::from("/tmp/transcripts/recording.txt"));
    }
}
