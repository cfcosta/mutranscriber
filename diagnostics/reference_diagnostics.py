#!/usr/bin/env python3
"""
Generate reference diagnostics from the official Qwen3-ASR implementation.

This script processes the test audio through the official HuggingFace implementation
and outputs statistics for comparison with the Rust implementation.

Usage:
    pip install torch transformers soundfile
    python reference_diagnostics.py

Output:
    python_diagnostics.json - Statistics for comparison with Rust output
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch


def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load WAV file and return samples as float32 array."""
    import wave

    with wave.open(path, "rb") as wav:
        assert wav.getnchannels() == 1, "Expected mono audio"
        assert wav.getsampwidth() == 2, "Expected 16-bit audio"
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        audio_bytes = wav.readframes(n_frames)

    # Convert to float32
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Resample if needed
    if sample_rate != target_sr:
        from scipy import signal

        audio = signal.resample(audio, int(len(audio) * target_sr / sample_rate))

    return audio


def compute_whisper_mel(audio: np.ndarray, n_mels: int = 128) -> np.ndarray:
    """Compute mel spectrogram using Whisper-style preprocessing."""
    from transformers import WhisperFeatureExtractor

    feature_extractor = WhisperFeatureExtractor(
        feature_size=n_mels,
        sampling_rate=16000,
        hop_length=160,
        n_fft=400,
        padding_value=0.0,
    )

    # Process audio
    inputs = feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt", padding=False
    )

    return inputs.input_features[0].numpy()  # Shape: (n_mels, n_frames)


def main():
    # Find test audio
    script_dir = Path(__file__).parent
    audio_path = script_dir.parent / "tests" / "fixtures" / "test_audio.wav"

    if not audio_path.exists():
        print(f"Error: Test audio not found at {audio_path}")
        sys.exit(1)

    print(f"Loading audio from: {audio_path}")
    audio = load_wav(str(audio_path))
    print(f"Audio samples: {len(audio)}, duration: {len(audio)/16000:.2f}s")

    # Compute mel spectrogram
    print("Computing mel spectrogram...")
    mel = compute_whisper_mel(audio)
    print(f"Mel shape: {mel.shape}")

    # Try to load the audio encoder for feature extraction
    try:
        from transformers import AutoProcessor, AutoModel

        print("Loading Qwen3-ASR model...")
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        # Process through audio encoder
        print("Extracting audio features...")
        inputs = processor(audio=[audio], sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            # Get audio features from the model's audio tower
            if hasattr(model, "thinker") and hasattr(model.thinker, "audio_tower"):
                audio_tower = model.thinker.audio_tower
                input_features = inputs.get("input_features", inputs.get("audio_values"))
                if input_features is not None:
                    audio_features = audio_tower(input_features)
                    if hasattr(audio_features, "last_hidden_state"):
                        audio_features = audio_features.last_hidden_state
                    audio_features = audio_features.numpy()
                else:
                    print("Warning: Could not extract input features")
                    audio_features = None
            else:
                print("Warning: Could not access audio_tower")
                audio_features = None

    except Exception as e:
        print(f"Warning: Could not load full model: {e}")
        print("Outputting mel spectrogram stats only")
        audio_features = None

    # Compute statistics
    diagnostics = {
        "mel_spectrogram": {
            "shape": list(mel.shape),
            "mean": float(np.mean(mel)),
            "std": float(np.std(mel)),
            "min": float(np.min(mel)),
            "max": float(np.max(mel)),
            "first_frame_16": mel[:16, 0].tolist(),  # First 16 mel bins, first frame
        }
    }

    if audio_features is not None:
        diagnostics["audio_features"] = {
            "shape": list(audio_features.shape),
            "mean": float(np.mean(audio_features)),
            "std": float(np.std(audio_features)),
            "min": float(np.min(audio_features)),
            "max": float(np.max(audio_features)),
            "first_frame_16": audio_features[0, 0, :16].tolist(),  # First 16 dims, first frame
        }

    # Write output
    output_path = script_dir / "python_diagnostics.json"
    with open(output_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\nDiagnostics written to: {output_path}")
    print("\nMel spectrogram stats:")
    print(f"  Shape: {diagnostics['mel_spectrogram']['shape']}")
    print(f"  Mean:  {diagnostics['mel_spectrogram']['mean']:.6f}")
    print(f"  Std:   {diagnostics['mel_spectrogram']['std']:.6f}")
    print(f"  Min:   {diagnostics['mel_spectrogram']['min']:.6f}")
    print(f"  Max:   {diagnostics['mel_spectrogram']['max']:.6f}")

    if "audio_features" in diagnostics:
        print("\nAudio features stats:")
        print(f"  Shape: {diagnostics['audio_features']['shape']}")
        print(f"  Mean:  {diagnostics['audio_features']['mean']:.6f}")
        print(f"  Std:   {diagnostics['audio_features']['std']:.6f}")
        print(f"  Min:   {diagnostics['audio_features']['min']:.6f}")
        print(f"  Max:   {diagnostics['audio_features']['max']:.6f}")


if __name__ == "__main__":
    main()
