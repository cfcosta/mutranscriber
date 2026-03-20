# Autoresearch ideas backlog

- Look for a faster decoder-generation path for single-token CUDA decode steps without changing numerics, ideally by reusing or upstreaming a better Candle/Qwen3 kernel instead of replacing flash-attn with a generic matmul path.
- If KV-prefill work is revisited, it likely needs stricter numerical equivalence testing first: both attempted prefix-prefill variants sped things up but changed wording on the repeated transcript, so a naïve segmented decoder API is not safe enough.
- Broader decoder QKV fusion still looks promising: a full fused `qkv` projection was much faster on CUDA but changed the repeated transcript from `tsunzhu` to `sun zhu`, while a narrower `seq_len == 1` decode-only fusion preserved exact output and was keepable. If revisited, focus on making the prompt/prefill path numerically equivalent rather than the decode path, which now has a working fused variant.
