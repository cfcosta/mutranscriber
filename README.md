# mutranscriber

Native Rust audio transcription using [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), powered by [Candle](https://github.com/huggingface/candle).

## Features

- Pure Rust implementation with no Python dependencies
- Automatic model download from HuggingFace Hub
- GPU acceleration via CUDA or Metal
- Audio extraction from video files via GStreamer
- Both CLI tool and library API

## Installation

### From source

```bash
# Standard build (includes GStreamer support)
cargo install --path .

# With GPU support
cargo install --path . --features cuda    # NVIDIA
cargo install --path . --features metal   # macOS

# Without GStreamer (library-only, no file loading)
cargo install --path . --no-default-features
```

### Requirements

- Rust 1.70+
- GStreamer development libraries (enabled by default)
- For `cuda` feature: CUDA toolkit
- For `metal` feature: macOS with Metal support

## Usage

### CLI

```bash
# Transcribe an audio file
mutranscriber recording.wav

# Transcribe a video file
mutranscriber video.mp4

# Use the larger model
mutranscriber audio.wav --model large

# Force CPU mode
mutranscriber audio.wav --cpu

# Print to stdout only
mutranscriber audio.wav --stdout-only

# Custom output path
mutranscriber audio.wav --output transcript.txt
```

### Library

```rust
use mutranscriber::{Transcriber, TranscriberConfig, ModelVariant};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create transcriber with default settings
    let transcriber = Transcriber::from_env();

    // Or with custom configuration
    let config = TranscriberConfig {
        variant: ModelVariant::Small,
        use_gpu: true,
        sample_rate: 16000,
        output_dir: None,
    };
    let transcriber = Transcriber::with_config(config);

    // Preload the model
    transcriber.preload().await?;

    // Transcribe raw audio samples (16kHz, f32)
    let audio_samples: Vec<f32> = load_audio_somehow();
    let text = transcriber.transcribe_audio(&audio_samples).await?;

    println!("{}", text);
    Ok(())
}
```

## Model Variants

| Variant | Parameters | VRAM | HuggingFace ID |
|---------|------------|------|----------------|
| Small   | 0.6B       | ~2GB | `Qwen/Qwen3-ASR-0.6B` |
| Large   | 1.7B       | ~4GB | `Qwen/Qwen3-ASR-1.7B` |

Models are automatically downloaded from HuggingFace Hub on first use and cached locally.

## Audio Requirements

- Sample rate: 16kHz
- Format: f32 mono samples
- The library handles padding to 30 seconds internally (matching WhisperFeatureExtractor)

When using the `gstreamer` feature, audio is automatically extracted and resampled from any format GStreamer supports.

## Build Features

| Feature | Description | Default |
|---------|-------------|---------|
| `gstreamer` | Audio extraction from video/audio files | Yes |
| `cuda` | NVIDIA GPU acceleration | No |
| `metal` | Apple Metal GPU acceleration | No |

## Development

```bash
# Enter dev environment (requires Nix)
nix develop

# Run tests
cargo test --lib

# Run integration tests (downloads model)
cargo test --test integration_test -- --ignored

# Format and lint
cargo fmt
cargo clippy
```

## Architecture

```
Audio Input (16kHz f32)
    │
    ▼
Mel Spectrogram (128 bins, 30s padded)
    │
    ▼
Audio Encoder (Qwen3-AuT Transformer)
    │
    ▼
Audio Features (projected to LLM dim)
    │
    ▼
Qwen3 LLM Decoder (with audio embeddings)
    │
    ▼
Text Output
```

## License

MIT OR Apache-2.0
