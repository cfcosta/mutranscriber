# Autoresearch ideas backlog

- Look for a faster decoder-generation path for single-token CUDA decode steps without changing numerics, ideally by reusing or upstreaming a better Candle/Qwen3 kernel instead of replacing flash-attn with a generic matmul path.
- If the corrected WAV benchmark still shows decoder generation dominating, investigate whether a more targeted KV-prefill API in `Qwen3Decoder` can avoid repeated static prompt work without introducing the overhead that sank the simple segmented-prefix experiment.
