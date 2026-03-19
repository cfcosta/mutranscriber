# Autoresearch ideas backlog

- Cache the static ChatML prompt work more aggressively in `Qwen3ASRModel::generate`: prompt token IDs, prompt embeddings, and possibly the decoder KV state after the pre-audio tokens, then replay only the audio features and post-audio tokens per chunk.
- Look for a faster decoder-generation path for single-token CUDA decode steps without changing numerics, ideally by reusing or upstreaming a better Candle/Qwen3 kernel instead of replacing flash-attn with a generic matmul path.
