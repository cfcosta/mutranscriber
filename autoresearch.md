# Autoresearch: GPU transcription throughput without quality loss

## Objective
Improve steady-state GPU transcription throughput for the CUDA path in `mutranscriber` without reducing transcription quality.

The benchmark uses the bundled `tests/fixtures/test_audio.wav` fixture and a synthetic 60-second workload formed by repeating that 10-second clip 6 times. The harness now parses the WAV as a real RIFF file instead of assuming a fixed 44-byte header, so the workload is the actual 10.0-second audio repeated 6 times rather than "audio plus stray metadata bytes".

## Metrics
- **Primary**: `wall_ms` (ms, lower is better) — median end-to-end transcription time for the 60-second repeated workload, excluding model preload and compilation.
- **Secondary**:
  - `throughput_x` — audio seconds transcribed per wall second
  - `audio_seconds` — sanity check for workload size
  - `short_exact` — exact normalized match on the original 10-second fixture
  - `long_exact` — exact normalized match on the repeated 60-second workload
  - `short_keyword_hits` — coarse quality guard from the existing integration test expectations

## How to Run
`./autoresearch.sh`

The script builds `src/bin/autoresearch_bench.rs`, runs a warm GPU benchmark via `nix develop`, and prints `METRIC ...` lines.

## Files in Scope
- `src/model.rs` — end-to-end inference path, chunking, prompt construction, generation loop
- `src/qwen3_decoder.rs` — decoder, KV cache, attention, prompt processing
- `src/audio_encoder.rs` — audio tower inference path and local attention
- `src/mel.rs` — CPU-side mel spectrogram extraction
- `src/transcriber.rs` — high-level transcription API if needed for runtime behavior
- `src/bin/autoresearch_bench.rs` — benchmark harness
- `autoresearch.sh` — benchmark driver
- `tests/fixtures/autoresearch_expected_short.txt` — normalized short reference transcript
- `tests/fixtures/autoresearch_expected_repeat6.txt` — normalized long reference transcript
- `autoresearch.ideas.md` — backlog for deferred ideas

## Off Limits
- Benchmark fixture audio contents in `tests/fixtures/test_audio.wav`
- Quality reference transcripts, unless the benchmark itself is demonstrably wrong and updated for a documented reason
- Changes that intentionally reduce generated text length, skip work, or otherwise game the benchmark
- CPU-only optimizations that do not help the CUDA path

## Constraints
- Do not overfit to the benchmark or cheat on the workload
- GPU path only: run the benchmark with CUDA enabled
- Keep quality at least as good as the current deterministic baseline on both short and repeated workloads
- No benchmark shortcuts that avoid real transcription work
- Keep the codebase buildable under `nix develop`

## What's Been Tried
- Initial harness setup: added a dedicated GPU benchmark binary and deterministic transcript references for the short fixture and a 6x repeated workload.
- Runtime quirk: the benchmark must include `/run/opengl-driver/lib` in `LD_LIBRARY_PATH` or Candle picks up the CUDA stub library instead of the real driver.
- Benchmark correctness fix: the original harness and integration test loader assumed a fixed 44-byte WAV header and accidentally treated extra RIFF metadata bytes as audio. The harness now parses RIFF chunks properly. This changes the workload slightly, so any post-fix measurements belong to a new baseline.
- On the original benchmark, profiling showed steady-state 30-second chunks spending roughly ~56ms in mel extraction, ~2-3ms in the audio encoder, and ~390-420ms in decoder generation. Generation is still the main bottleneck, but mel extraction was large enough to be worth optimizing.
- Discarded on the original benchmark: increasing `Qwen3ASRModel::CHUNK_SAMPLES` from 30s to 120s improved throughput materially, but it changed the long repeated transcript and dropped one repeated segment.
- Discarded on the original benchmark: folding a sub-1s tail chunk back into earlier chunks also improved throughput, but the shifted chunk boundary changed the long repeated transcript (`tsunzhu` -> `tsunzuo`).
- Discarded on the original benchmark: replacing CUDA flash-attn with a manual matmul/softmax path for single-token decode steps was much slower and changed the transcript.
- Kept on the original benchmark: `MelSpectrogram` now precomputes each mel filter's non-zero range and skips zero-weight bins during filterbank application.
- Kept on the original benchmark: after trimming each mel filter to its active range, replacing the iterator-heavy zip/map/sum accumulation with a tight indexed loop improved codegen further without changing transcripts.
- Discarded mel follow-ups on the original benchmark: precomputed FFT bit-reversal indices, a full-window Hann fast path, unsafe pointer walks, manual four-lane unrolling, forced inlining, and Rayon parallelization all failed to beat the current simple sparse-loop mel implementation.
- Discarded generation follow-ups on the original benchmark: prompt-embedding caching, segmented prompt/KV prefill, single-token manual attention, non-causal flash-attn for decode, debug-log guarding, shaped `Tensor::from_slice` token creation, and switching CUDA BF16 -> F16 all failed to improve throughput versus the kept baseline.
