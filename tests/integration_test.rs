//! Integration tests for mutranscriber.
//!
//! These tests verify the full transcription pipeline works correctly.
//! The test audio is a 10-second segment from LibriVox's "The Art of War" by Sun Tzu.

use std::path::PathBuf;

use mutranscriber::{
    MelSpectrogram,
    ModelVariant,
    Transcriber,
    TranscriberConfig,
    HOP_LENGTH,
    N_MELS,
    SAMPLE_RATE,
};

/// Path to the test audio fixture.
fn test_audio_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("test_audio.wav")
}

/// Load test audio samples from the fixture file.
fn load_test_audio() -> Vec<f32> {
    let path = test_audio_path();
    assert!(path.exists(), "Test audio file not found: {:?}", path);

    // Read WAV file manually (16-bit PCM, 16kHz, mono)
    let data = std::fs::read(&path).expect("Failed to read test audio file");

    // Skip WAV header (44 bytes for standard PCM WAV)
    let audio_data = &data[44..];

    // Convert 16-bit PCM to f32 samples
    audio_data
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect()
}

#[test]
fn test_audio_file_exists() {
    let path = test_audio_path();
    assert!(path.exists(), "Test audio fixture missing: {:?}", path);

    let metadata =
        std::fs::metadata(&path).expect("Failed to read file metadata");
    assert!(metadata.len() > 0, "Test audio file is empty");
}

#[test]
fn test_load_audio_samples() {
    let samples = load_test_audio();

    // 10 seconds at 16kHz = 160,000 samples
    assert!(
        samples.len() > 150_000,
        "Expected ~160k samples, got {}",
        samples.len()
    );
    assert!(
        samples.len() < 170_000,
        "Audio too long, got {} samples",
        samples.len()
    );

    // Verify samples are in valid range
    for sample in &samples {
        assert!(
            *sample >= -1.0 && *sample <= 1.0,
            "Sample out of range: {}",
            sample
        );
    }

    // Verify audio is not silence (has some variation)
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs > 0.01,
        "Audio appears to be silence, max amplitude: {}",
        max_abs
    );
}

#[test]
fn test_mel_spectrogram_computation() {
    let samples = load_test_audio();
    let mel = MelSpectrogram::new();

    let (mel_data, n_frames, n_mels) = mel.compute_2d(&samples);

    // Verify dimensions
    assert_eq!(n_mels, N_MELS, "Expected {} mel bins", N_MELS);

    // Expected frames: (samples - n_fft) / hop_length + 1
    // For 320k samples: approximately 2000 frames
    let expected_frames = (samples.len() - 400) / HOP_LENGTH + 1;
    assert!(
        (n_frames as i64 - expected_frames as i64).abs() < 10,
        "Frame count mismatch: got {}, expected ~{}",
        n_frames,
        expected_frames
    );

    // Verify data size matches dimensions
    assert_eq!(mel_data.len(), n_frames * n_mels, "Mel data size mismatch");

    // Verify mel values are normalized (Whisper-style: roughly [-1, 1] range)
    let min_val = mel_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = mel_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Whisper normalization: (log_mel + 4.0) / 4.0, resulting in roughly [-1, 1]
    assert!(
        min_val >= -2.0,
        "Mel values should be >= -2.0, got min: {}",
        min_val
    );
    assert!(
        max_val <= 2.0,
        "Mel values should be <= 2.0, got max: {}",
        max_val
    );
}

#[test]
fn test_transcriber_config_defaults() {
    let config = TranscriberConfig::default();

    assert_eq!(config.sample_rate, SAMPLE_RATE as u32);
    assert!(config.use_gpu);
    assert!(matches!(config.variant, ModelVariant::Small));
}

#[test]
fn test_transcriber_creation() {
    let _transcriber = Transcriber::new();

    let config = TranscriberConfig {
        variant: ModelVariant::Small,
        use_gpu: false,
        sample_rate: 16000,
        output_dir: None,
    };
    let _transcriber = Transcriber::with_config(config);
}

/// Test that model can be loaded successfully.
///
/// This test verifies:
/// 1. Model weights can be downloaded from HuggingFace
/// 2. Audio encoder loads with correct tensor shapes
/// 3. LLM decoder loads with correct configuration
///
/// Run with: cargo test --test integration_test test_model_loading -- --ignored
#[tokio::test]
#[ignore] // Requires ~2GB model download
async fn test_model_loading() {
    let config = TranscriberConfig {
        variant: ModelVariant::Small,
        use_gpu: false, // Use CPU for CI compatibility
        sample_rate: 16000,
        output_dir: None,
    };

    let transcriber = Transcriber::with_config(config);

    // Preload model - this downloads and loads all weights
    transcriber
        .preload()
        .await
        .expect("Failed to preload model");

    assert!(
        transcriber.is_model_loaded().await,
        "Model should be loaded"
    );
}

/// Full transcription test - requires model download.
///
/// NOTE: The current text generation logic doesn't properly inject audio features
/// into the LLM. This test verifies the pipeline runs but transcription quality
/// is not yet validated.
///
/// Run with: cargo test --test integration_test test_full_transcription -- --ignored
#[tokio::test]
#[ignore] // Requires ~2GB model download and generation logic fixes
async fn test_full_transcription() {
    let samples = load_test_audio();

    let config = TranscriberConfig {
        variant: ModelVariant::Small,
        use_gpu: false, // Use CPU for CI compatibility
        sample_rate: 16000,
        output_dir: None,
    };

    let transcriber = Transcriber::with_config(config);

    // Preload model
    transcriber
        .preload()
        .await
        .expect("Failed to preload model");
    assert!(transcriber.is_model_loaded().await);

    // Transcribe - this exercises the full pipeline
    let transcript = transcriber
        .transcribe_audio(&samples)
        .await
        .expect("Transcription failed");

    // The audio is from "The Art of War" by Sun Tzu
    // Expected content: "The art of war is of vital importance to the State.
    //                    It is a matter of life and death, a road either to safety or to ruin."

    // Verify transcript is not empty
    assert!(!transcript.is_empty(), "Transcript should not be empty");

    // TODO: Once generation logic properly injects audio features, verify content:
    // let transcript_lower = transcript.to_lowercase();
    // let expected_keywords = ["war", "art", "state", "life", "death"];
    // Verify at least 3 keywords present

    println!("Transcription result: {}", transcript);
}

/// Test that model variant IDs are correct.
#[test]
fn test_model_variant_ids() {
    assert_eq!(ModelVariant::Small.model_id(), "Qwen/Qwen3-ASR-0.6B");
    assert_eq!(ModelVariant::Large.model_id(), "Qwen/Qwen3-ASR-1.7B");
}

/// Full transcription test with 1.7B model - requires larger model download.
///
/// Run with: cargo test --test integration_test test_full_transcription_large -- --ignored
#[tokio::test]
#[ignore] // Requires ~4GB model download
async fn test_full_transcription_large() {
    let samples = load_test_audio();

    let config = TranscriberConfig {
        variant: ModelVariant::Large,
        use_gpu: false, // Use CPU for CI compatibility
        sample_rate: 16000,
        output_dir: None,
    };

    let transcriber = Transcriber::with_config(config);

    // Preload model
    transcriber
        .preload()
        .await
        .expect("Failed to preload model");
    assert!(transcriber.is_model_loaded().await);

    // Transcribe - this exercises the full pipeline
    let transcript = transcriber
        .transcribe_audio(&samples)
        .await
        .expect("Transcription failed");

    // Verify transcript is not empty
    assert!(!transcript.is_empty(), "Transcript should not be empty");

    println!("Transcription result (1.7B): {}", transcript);
}
