//! Integration tests for mutranscriber.
//!
//! These tests verify the full transcription pipeline works correctly.
//! The test audio is a 20-second segment from LibriVox's "The Art of War" by Sun Tzu.

use std::path::PathBuf;

use mutranscriber::{
    HOP_LENGTH,
    MelSpectrogram,
    ModelVariant,
    N_MELS,
    SAMPLE_RATE,
    Transcriber,
    TranscriberConfig,
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

    let metadata = std::fs::metadata(&path).expect("Failed to read file metadata");
    assert!(metadata.len() > 0, "Test audio file is empty");
}

#[test]
fn test_load_audio_samples() {
    let samples = load_test_audio();

    // 20 seconds at 16kHz = 320,000 samples
    assert!(
        samples.len() > 300_000,
        "Expected ~320k samples, got {}",
        samples.len()
    );
    assert!(
        samples.len() < 350_000,
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

    // Verify mel values are normalized (0-1 range after log scaling)
    let min_val = mel_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = mel_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        min_val >= 0.0,
        "Mel values should be non-negative, got min: {}",
        min_val
    );
    assert!(
        max_val <= 1.0,
        "Mel values should be <= 1.0, got max: {}",
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

/// Full transcription test - requires model download.
///
/// This test is ignored by default because it requires:
/// 1. Downloading the model (~2GB)
/// 2. Significant computation time
///
/// Run with: cargo test --test integration_test test_full_transcription -- --ignored
#[tokio::test]
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

    // Transcribe
    let transcript = transcriber
        .transcribe_audio(&samples)
        .await
        .expect("Transcription failed");

    // The audio is from "The Art of War" by Sun Tzu
    // Expected content: "The art of war is of vital importance to the State.
    //                    It is a matter of life and death, a road either to safety or to ruin."

    // Verify transcript is not empty
    assert!(!transcript.is_empty(), "Transcript should not be empty");

    // Verify transcript contains expected keywords (case-insensitive)
    let transcript_lower = transcript.to_lowercase();

    // Check for key phrases that should appear
    let expected_keywords = ["war", "art", "state", "life", "death"];
    let mut found_keywords = 0;

    for keyword in &expected_keywords {
        if transcript_lower.contains(keyword) {
            found_keywords += 1;
        }
    }

    // Expect at least 3 of the 5 keywords to be present
    assert!(
        found_keywords >= 3,
        "Expected at least 3 keywords from {:?} in transcript: {}",
        expected_keywords,
        transcript
    );

    println!("Transcription result: {}", transcript);
}

/// Test that model variant IDs are correct.
#[test]
fn test_model_variant_ids() {
    assert_eq!(ModelVariant::Small.model_id(), "Qwen/Qwen3-ASR-0.6B");
    assert_eq!(ModelVariant::Large.model_id(), "Qwen/Qwen3-ASR-1.7B");
}
