//! Qwen3-ASR implementation for audio transcription using Candle.
//!
//! This crate provides native Rust audio transcription using the Qwen3-ASR model
//! running on Candle (Hugging Face's Rust ML framework).
//!
//! ## Features
//!
//! - `gstreamer` - Enable audio extraction from video/audio files using GStreamer
//! - `cli` - Enable CLI binary dependencies
//! - `cuda` - Enable CUDA GPU acceleration
//! - `metal` - Enable Metal GPU acceleration (macOS)
//!
//! ## Usage
//!
//! ```no_run
//! use mutranscriber::{Transcriber, TranscriberConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let transcriber = Transcriber::from_env();
//!
//!     // Transcribe audio samples directly
//!     let audio_samples: Vec<f32> = vec![]; // 16kHz f32 samples
//!     let text = transcriber.transcribe_audio(&audio_samples).await?;
//!
//!     println!("Transcription: {}", text);
//!     Ok(())
//! }
//! ```

mod audio_encoder;
mod config;
mod mel;
mod model;
mod qwen3_decoder;
mod transcriber;

pub use config::{AudioEncoderConfig, GenerationConfig, Qwen3ASRConfig};
pub use mel::{
    MelSpectrogram,
    CHUNK_LENGTH,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
};
pub use model::{ModelVariant, Qwen3ASRModel, Qwen3ASRModelBuilder};
pub use transcriber::{
    Transcriber,
    TranscriberConfig,
    TranscriberError,
    TranscriberResult,
};
