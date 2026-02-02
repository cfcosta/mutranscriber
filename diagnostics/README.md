# Diagnostic Tools for Audio Encoder Comparison

This directory contains tools for comparing the Rust implementation's intermediate
tensor values with the official Python implementation.

## Purpose

The transcription accuracy issues may stem from differences in:
1. Mel spectrogram computation
2. Audio encoder processing
3. Numerical precision or normalization

These tools help identify where divergence occurs.

## Quick Start

### Step 1: Generate Rust Diagnostics

```bash
cd /path/to/mutranscriber
MUTRANSCRIBER_DIAGNOSTICS=/tmp/rust_diagnostics.json \
  cargo test --test integration_test test_full_transcription -- --ignored
```

### Step 2: Generate Python Reference (requires Python environment)

```bash
# Create Python environment with dependencies
python -m venv .venv
source .venv/bin/activate
pip install torch transformers soundfile scipy

# Generate reference diagnostics
python diagnostics/reference_diagnostics.py
```

### Step 3: Compare Results

```bash
python diagnostics/compare_diagnostics.py /tmp/rust_diagnostics.json diagnostics/python_diagnostics.json
```

## Output Format

Both scripts generate JSON with the following structure:

```json
{
  "mel_spectrogram": {
    "shape": [1, 128, 998],
    "mean": -0.185353,
    "std": 0.418893,
    "min": -0.638210,
    "max": 1.361790,
    "first_frame_16": [...]
  },
  "audio_features": {
    "shape": [1, 125, 1024],
    "mean": 0.000536,
    "std": 0.015741,
    "min": -0.111874,
    "max": 0.100708,
    "first_frame_16": [...]
  }
}
```

## Current Rust Values (test_audio.wav)

| Tensor | Shape | Mean | Std | Min | Max |
|--------|-------|------|-----|-----|-----|
| Mel Spectrogram | [1, 128, 998] | -0.185 | 0.419 | -0.638 | 1.362 |
| Audio Features | [1, 125, 1024] | 0.001 | 0.016 | -0.112 | 0.101 |

## Interpretation

- **MATCH** (diff < 0.001): Values are essentially identical
- **CLOSE** (diff < 0.1): Minor numerical differences, likely acceptable
- **DIFFER** (diff >= 0.1): Significant difference, investigate further

If mel spectrogram values differ significantly, check:
- FFT window function
- Mel filterbank construction
- Log normalization

If audio features differ with similar mel input, check:
- Conv2D layer weights
- Layer normalization implementation
- Attention computation
