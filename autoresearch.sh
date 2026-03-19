#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"

cargo_cmd=(nix develop -c cargo)
bench_cmd=(nix develop -c target/release/autoresearch_bench --repeat 6 --warmup-runs 1 --measure-runs 5)

"${cargo_cmd[@]}" build --release --features cuda --bin autoresearch_bench >/dev/null

output=$("${bench_cmd[@]}")
printf '%s\n' "$output"

median_ms=$(awk '/^MEDIAN_MS / { print $2 }' <<<"$output")
throughput_x=$(awk '/^THROUGHPUT_X / { print $2 }' <<<"$output")
audio_seconds=$(awk '/^AUDIO_SECONDS / { print $2 }' <<<"$output")
short_norm=$(sed -n 's/^SHORT_TRANSCRIPT_NORM //p' <<<"$output")
long_norm=$(sed -n 's/^LONG_TRANSCRIPT_NORM //p' <<<"$output")

expected_short=$(tr -d '\n' < tests/fixtures/autoresearch_expected_short.txt)
expected_long=$(tr -d '\n' < tests/fixtures/autoresearch_expected_repeat6.txt)

short_exact=0
if [[ "$short_norm" == "$expected_short" ]]; then
  short_exact=1
fi

long_exact=0
if [[ "$long_norm" == "$expected_long" ]]; then
  long_exact=1
fi

short_keyword_hits=0
for kw in librevox recording art war translated lionel; do
  if [[ "$short_norm" == *"$kw"* ]]; then
    short_keyword_hits=$((short_keyword_hits + 1))
  fi
done

printf 'METRIC wall_ms=%s\n' "$median_ms"
printf 'METRIC throughput_x=%s\n' "$throughput_x"
printf 'METRIC audio_seconds=%s\n' "$audio_seconds"
printf 'METRIC short_exact=%s\n' "$short_exact"
printf 'METRIC long_exact=%s\n' "$long_exact"
printf 'METRIC short_keyword_hits=%s\n' "$short_keyword_hits"
