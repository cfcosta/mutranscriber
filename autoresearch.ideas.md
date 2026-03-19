# Autoresearch ideas backlog

- Look for a faster decoder-generation path for single-token CUDA decode steps without changing numerics, ideally by reusing or upstreaming a better Candle/Qwen3 kernel instead of replacing flash-attn with a generic matmul path.
- If KV-prefill work is revisited, it likely needs stricter numerical equivalence testing first: both attempted prefix-prefill variants sped things up but changed wording on the repeated transcript, so a naïve segmented decoder API is not safe enough.
