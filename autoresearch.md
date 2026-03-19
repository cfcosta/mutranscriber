# Autoresearch: GPU transcription throughput without quality loss

## Objective
Improve steady-state GPU transcription throughput for the CUDA path in `mutranscriber` without reducing transcription quality.

The benchmark uses the bundled `tests/fixtures/test_audio.wav` fixture and a synthetic 60-second workload formed by repeating that 10-second clip 6 times. This gives a longer, more stable GPU workload while still checking output quality against deterministic reference transcripts.

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
- Baseline observation before the experiment loop: the current CUDA path processes the 60-second repeated workload at about 65x realtime, with exact transcript matches on both reference checks.
- Runtime quirk: the benchmark must include `/run/opengl-driver/lib` in `LD_LIBRARY_PATH` or Candle picks up the CUDA stub library instead of the real driver.
- Profiling insight from the harness: steady-state 30-second chunks spend roughly ~56ms in mel extraction, ~2-3ms in the audio encoder, and ~390-420ms in decoder generation. Generation is still the main bottleneck, but mel extraction is large enough to be worth optimizing.
- Discarded: increasing `Qwen3ASRModel::CHUNK_SAMPLES` from 30s to 120s improved throughput materially, but it changed the long repeated transcript and dropped one repeated segment.
- Discarded: folding a sub-1s tail chunk back into earlier chunks also improved throughput, but the shifted chunk boundary changed the long repeated transcript (`tsunzhu` -> `tsunzuo`).
- Discarded: replacing CUDA flash-attn with a manual matmul/softmax path for single-token decode steps was much slower and changed the transcript.
- Kept: `MelSpectrogram` now precomputes each mel filter's non-zero range and skips zero-weight bins during filterbank application. This preserves output exactly and cuts the primary metric by about 6-7% on the GPU benchmark because mel extraction was a meaningful share of total runtime.
- Kept: after trimming each mel filter to its active range, replacing the iterator-heavy zip/map/sum accumulation with a tight indexed loop improved codegen further and reduced the benchmark again without changing transcripts.
