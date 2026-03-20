# Autoresearch ideas backlog

- Look for a faster decoder-generation path for single-token CUDA decode steps without changing numerics, ideally by reusing or upstreaming a better Candle/Qwen3 kernel instead of replacing flash-attn with a generic matmul path.
- Investigate decoder KV-cache write overhead on the single-token CUDA hot path, but likely only via a better Candle/upstream append primitive. A naïve combined KV cache tensor regressed badly, so simple K/V concat+single-write rewrites are not promising.
- If KV-prefill work is revisited, it likely needs stricter numerical equivalence testing first: both attempted prefix-prefill variants sped things up but changed wording on the repeated transcript, so a naïve segmented decoder API is not safe enough.
- Broader prompt/prefill-side QKV fusion is now less attractive: full fusion changed the transcript, and exact partial variants (KV-only fusion, last-four-layer fusion) still regressed. Only revisit this if there is a more numerically stable kernel or layout strategy than the current naïve fused matmul.
