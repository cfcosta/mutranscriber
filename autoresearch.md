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
- On the corrected workload, retrying larger chunks still caused a quality regression (`long_exact=0`), so chunk-size wins remain off the table unless transcription fidelity can be preserved.
- Discarded on the corrected workload: returning 2D logits from the decoder, a specialized greedy decode loop, direct `(1, 1)` token tensor construction, a `seq_len == 1` rotary fast path, cached decode-loop housekeeping, a hoisted attention scale constant, a direct `forward_token_id` embedding-table path, and downgrading hot-path shape checks to `debug_assert!` all regressed.
- Discarded on the corrected workload: two more targeted KV-prefill variants (`prefill` for prefix+audio and `prefill` for prefix-only) produced sizable speedups but changed the repeated-workload wording from `tsunzhu` to `sun zhu`, so these decoder refactors are not numerically equivalent enough to keep.
- Kept on the corrected workload: in `Qwen3Decoder::forward_embeds`, skip the redundant `narrow(1, seq_len - 1, 1)` when `seq_len == 1`. That hot path is used for every autoregressive decode step, preserves exact transcripts, and may shave a small amount off the corrected benchmark, but the current confidence is still low because reruns straddle the noise floor.
- Kept on the corrected workload: pipeline CPU mel extraction for chunked transcriptions so the next chunk's spectrogram is computed on a background thread while the GPU encodes/decodes the current chunk. This preserves the exact short and repeated transcripts while overlapping ~tens of milliseconds of CPU preprocessing with GPU inference on multi-chunk audio.
- Kept on the corrected workload: precompute the static ChatML prompt token IDs once at model load instead of re-tokenizing and rebuilding the same `system/user/assistant` prompt vectors for every chunk. The measured gain is tiny and close to the noise floor, but it preserves exact transcripts and trims a bit of repeated host-side setup from the generation path.
- Kept on the corrected workload: also materialize those static prompt token tensors once at model load so each chunk can reuse them directly when looking up prompt embeddings. This is another very small host-side win, but it remained transcript-exact and nudged the benchmark a bit lower again.
- Discarded on the corrected workload: fusing the decoder's separate `q_proj`/`k_proj`/`v_proj` matmuls into one combined `qkv` projection produced a large CUDA speedup, but it changed the repeated-workload wording from `tsunzhu` to `sun zhu`, so the naive fusion is not numerically equivalent enough to keep.
- Kept on the corrected workload: keep the original three-matmul path for prompt/prefill passes, but add a fused `qkv` projection only for the CUDA `seq_len == 1` decode hot path. That preserves exact transcripts while reducing per-token decoder launch overhead enough to beat the current baseline again.
- Kept on the corrected workload: derive the decoder's prompt-path `q_proj`/`k_proj`/`v_proj` views from the fused `qkv` weight instead of storing duplicate projection weights for the decode fast path. This keeps the exact output while trimming memory pressure and improved the benchmark a bit further.
- Kept on the corrected workload: fuse the decoder MLP's `gate_proj` + `up_proj` into one shared `gate_up` projection for the CUDA `seq_len == 1` decode hot path, while leaving the longer prompt path numerically identical via narrow-view `gate_proj`/`up_proj` wrappers. This preserved exact transcripts and delivered a much larger decoder-speed win than the earlier micro-optimizations.
- Discarded on the corrected workload: extending that fused `gate_up` MLP projection to all CUDA sequence lengths kept the transcript exact, but it was slightly slower than limiting the fusion to `seq_len == 1`, so the MLP fusion only pays off on the decode hot path.
- Discarded on the corrected workload: trying to squeeze the decode hot path down to 2D tensors before the fused QKV / fused `gate_up` matmuls was also slower than leaving Candle on its normal 3D linear fast path, even though the transcript stayed exact.
- Discarded on the corrected workload: attempting to merge the decoder's single-token Q/K RMS-norm work into one combined normalization launch catastrophically changed the output, so those norm paths are not interchangeable even though they look structurally similar.
- Discarded on the corrected workload: fusing only the prompt-side `k_proj` + `v_proj` matmuls, or re-enabling prompt-side QKV fusion for just the last few decoder layers, both kept the transcript exact but were still slower than the current decode-only QKV fusion baseline.
- Discarded on the corrected workload: combining Q and K into one single-token CUDA RoPE launch kept the transcript exact but was much slower, so the extra concat/split work outweighs the saved kernel launch.
- Discarded on the corrected workload: changing mel extraction to emit mel-major data so the GPU path could skip the Candle transpose was much slower than the current frame-major mel plus tensor transpose flow.
- Discarded on the corrected workload: replacing two tiny prompt embedding lookups with one larger lookup over a concatenated pre+post prompt tensor also regressed, so the current split prompt embedding path is already better.
- Discarded on the corrected workload: a single combined KV cache tensor for decoder decode steps also regressed badly, so reducing cache writes from two `slice_set`s to one is not enough to offset the extra K/V concat and split overhead.
