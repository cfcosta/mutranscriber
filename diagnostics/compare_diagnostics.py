#!/usr/bin/env python3
"""
Compare Rust and Python diagnostic outputs.

Usage:
    python compare_diagnostics.py rust_diagnostics.json python_diagnostics.json
"""

import json
import sys
from pathlib import Path


def compare_arrays(name: str, rust: list, python: list) -> None:
    """Compare two arrays element-wise."""
    if len(rust) != len(python):
        print(f"  {name}: LENGTH MISMATCH - Rust {len(rust)} vs Python {len(python)}")
        return

    diffs = [abs(r - p) for r, p in zip(rust, python)]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)

    if max_diff < 0.001:
        status = "MATCH"
    elif max_diff < 0.1:
        status = "CLOSE"
    else:
        status = "DIFFER"

    print(f"  {name}: {status} (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")


def compare_scalar(name: str, rust: float, python: float) -> None:
    """Compare two scalar values."""
    diff = abs(rust - python)
    rel_diff = diff / max(abs(python), 1e-6) * 100

    if diff < 0.001:
        status = "MATCH"
    elif diff < 0.1:
        status = "CLOSE"
    else:
        status = "DIFFER"

    print(f"  {name}: Rust={rust:.6f}, Python={python:.6f}, diff={diff:.6f} ({rel_diff:.1f}%) [{status}]")


def main():
    if len(sys.argv) < 3:
        # Try default paths
        script_dir = Path(__file__).parent
        rust_path = Path("/tmp/rust_diagnostics.json")
        python_path = script_dir / "python_diagnostics.json"
    else:
        rust_path = Path(sys.argv[1])
        python_path = Path(sys.argv[2])

    if not rust_path.exists():
        print(f"Error: Rust diagnostics not found at {rust_path}")
        print("Run: MUTRANSCRIBER_DIAGNOSTICS=/tmp/rust_diagnostics.json cargo test ...")
        sys.exit(1)

    if not python_path.exists():
        print(f"Error: Python diagnostics not found at {python_path}")
        print("Run: python reference_diagnostics.py")
        sys.exit(1)

    with open(rust_path) as f:
        rust = json.load(f)

    with open(python_path) as f:
        python = json.load(f)

    print("=" * 60)
    print("DIAGNOSTIC COMPARISON: Rust vs Python")
    print("=" * 60)

    # Compare mel spectrogram
    print("\n[Mel Spectrogram]")
    if "mel_spectrogram" in rust and "mel_spectrogram" in python:
        r_mel = rust["mel_spectrogram"]
        p_mel = python["mel_spectrogram"]

        print(f"  Shape: Rust={r_mel['shape']}, Python={p_mel['shape']}")
        compare_scalar("mean", r_mel["mean"], p_mel["mean"])
        compare_scalar("std", r_mel["std"], p_mel["std"])
        compare_scalar("min", r_mel["min"], p_mel["min"])
        compare_scalar("max", r_mel["max"], p_mel["max"])
        compare_arrays("first_frame_16", r_mel["first_frame_16"], p_mel["first_frame_16"])
    else:
        print("  Missing data in one or both files")

    # Compare audio features
    print("\n[Audio Features]")
    if "audio_features" in rust and "audio_features" in python:
        r_af = rust["audio_features"]
        p_af = python["audio_features"]

        print(f"  Shape: Rust={r_af['shape']}, Python={p_af['shape']}")
        compare_scalar("mean", r_af["mean"], p_af["mean"])
        compare_scalar("std", r_af["std"], p_af["std"])
        compare_scalar("min", r_af["min"], p_af["min"])
        compare_scalar("max", r_af["max"], p_af["max"])
        compare_arrays("first_frame_16", r_af["first_frame_16"], p_af["first_frame_16"])
    elif "audio_features" in rust:
        print("  Audio features only in Rust output")
        r_af = rust["audio_features"]
        print(f"  Rust shape: {r_af['shape']}")
        print(f"  Rust mean:  {r_af['mean']:.6f}")
        print(f"  Rust std:   {r_af['std']:.6f}")
    else:
        print("  Audio features missing")

    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE:")
    print("  MATCH  = diff < 0.001 (essentially identical)")
    print("  CLOSE  = diff < 0.1 (minor numerical differences)")
    print("  DIFFER = diff >= 0.1 (significant difference)")
    print("=" * 60)


if __name__ == "__main__":
    main()
