# Tokenizer Primitive Tracker

## Tokenizer Track Complete

`p1_fractal_hybrid_dyn-state-norm_v2` finished the tokenizer track as the stable leader. It passed the seeded proving-ground, short real-text follow-ups, motif-amplification benchmark, 20x repetition stress test, mixed-domain benchmark, SOTA sanity check against `tiktoken cl100k_base`, and round-trip reconstruction checks on both benchmark inputs while preserving the exact `d0:1, d1:2, d2:4, d3:8, d4:16, d5:32` recursive split throughout. The core thesis is now validated on this track: a single self-regulating lever inside the recursive rule can induce controlled cross-depth motif reuse on repetition-heavy text, remain lossless under reconstruction, and avoid spurious reuse on mixed-domain input without collapsing the hierarchy.

## Active / Promising Tokenizer Variants

| Variant Name                          | Base Primitive     | Lever Type                  | Status        | Token Count | Notes / Next Action |
|---------------------------------------|--------------------|-----------------------------|---------------|-------------|---------------------|
| p1_fractal_hybrid_dyn-state-norm_v2   | p1                 | self-regulating motif reuse | Stable Leader | 63          | Best across amplification, stress, and mixed-domain checks |

## Validation Snapshot

- Seeded proving-ground: all five revived squaring-family primitives survived at `dim=64`, `max_depth=6`, each producing `43` tokens
- Original motif-amplification input: static `p1_fractal_hybrid_v1` held the perfect split with `motif_reuse=0`; dynamic-v2 held the same split and reached `motif_reuse=4`
- 20x stress input: static `p1_fractal_hybrid_v1` held the perfect split with `motif_reuse=0`; dynamic-v2 held the same split and reached `motif_reuse=7`
- Mixed-domain benchmark: static `p1_fractal_hybrid_v1` and dynamic-v2 both preserved the perfect split with `motif_reuse=0`, showing no false cross-domain collisions
- Token-span debugging: final-depth span previews and grouped `REUSED MOTIFS (cross-depth)` output now expose the exact phrase blocks reused by dynamic-v2 under repetition-heavy input

## SOTA Sanity Check (tiktoken cl100k_base)

- 20x repetition stress input: `p1_fractal_hybrid_dyn-state-norm_v2` produced `63` hierarchical tokens at `31.32` chars/token with `motif_reuse=7`; `tiktoken cl100k_base` produced `569` tokens at `3.47` chars/token
- Stress reuse view: dynamic-v2 grouped repeated multi-sentence phrase blocks across depths, while `cl100k_base` remained a flat subword baseline with no cross-depth notion of reuse
- Mixed-domain input: `p1_fractal_hybrid_dyn-state-norm_v2` produced `63` hierarchical tokens at `11.49` chars/token with `motif_reuse=0`; `tiktoken cl100k_base` produced `138` tokens at `5.25` chars/token
- Mixed-domain note: v2 stayed coarse without introducing false positive reuse, which is the behavior we wanted from the self-regulating lever
- Added round-trip reconstruction check — v2 is lossless on both stress and mixed-domain inputs

## Quick OSS Model Benchmark (Llama 3.1 8B + Mistral 7B)

- Evaluation corpus: fixed `10,884`-char prose+code corpus built from repeated stress-text blocks, mixed-domain paragraphs, and a small Rust span-merging snippet
- `Llama 3.1 8B` with `p1_fractal_hybrid_dyn-state-norm_v2`: `63` tokens, `172.76` chars/token, `motif_reuse=5`, `rough_perplexity=N/A`, `wall_time_ms=63`, `hierarchy remains perfectly balanced`
- `Llama 3.1 8B` native tokenizer: `MODEL NOT FOUND — SKIPPED` because `LLAMA31_8B_GGUF_PATH` is not set in the local environment
- `Mistral 7B` with `p1_fractal_hybrid_dyn-state-norm_v2`: `63` tokens, `172.76` chars/token, `motif_reuse=5`, `rough_perplexity=N/A`, `wall_time_ms=45`, `hierarchy remains perfectly balanced`
- `Mistral 7B` native tokenizer: `MODEL NOT FOUND — SKIPPED` because `MISTRAL_7B_GGUF_PATH` is not set in the local environment
- OSS benchmark note: v2 stayed balanced and reused only large repeated phrase blocks on the 10k corpus, but the native comparison rows remain pending until local GGUF paths are configured

## Cloud GPU OSS Benchmark (Llama 3.1 8B + Mistral 7B)

- Live cloud GPU run completed on a secure RunPod `RTX 4090` pod using [oss_benchmark_v2.py](/Users/joseph/fractal-tokenizer-checkout/fractal-tokenizer/benchmarks/oss_benchmark_v2.py) with the `transformers` backend
- Models: `TroyDoesAI/Llama-3.1-8B-Instruct` and `mistralai/Mistral-7B-Instruct-v0.3`
- Stress input (`1973` chars):
  - `Llama 3.1 8B` `v2`: `63` tokens, `31.32` chars/token, `motif_reuse=7`, `wall_time_ms=482.31`, `simple_perplexity=N/A`
  - `Llama 3.1 8B` native: `569` tokens, `3.47` chars/token, `wall_time_ms=391.68`, `simple_perplexity=1.1548`
  - `Mistral 7B` `v2`: `63` tokens, `31.32` chars/token, `motif_reuse=7`, `wall_time_ms=1414.27`, `simple_perplexity=N/A`
  - `Mistral 7B` native: `590` tokens, `3.34` chars/token, `wall_time_ms=1324.36`, `simple_perplexity=1.1370`
- Mixed-domain input (`724` chars):
  - `Llama 3.1 8B` `v2`: `63` tokens, `11.49` chars/token, `motif_reuse=0`, `wall_time_ms=127.58`, `simple_perplexity=N/A`
  - `Llama 3.1 8B` native: `138` tokens, `5.25` chars/token, `wall_time_ms=36.78`, `simple_perplexity=47.2513`
  - `Mistral 7B` `v2`: `63` tokens, `11.49` chars/token, `motif_reuse=0`, `wall_time_ms=866.19`, `simple_perplexity=N/A`
  - `Mistral 7B` native: `157` tokens, `4.61` chars/token, `wall_time_ms=774.82`, `simple_perplexity=28.3476`
- Cloud benchmark note: on heavy repetition, `v2` compressed about `9x` relative to native tokenizers on both models while preserving the balanced hierarchy and exposing reusable phrase blocks; on mixed-domain text, `v2` still compressed about `2.2x–2.5x` while keeping `motif_reuse=0`, which means no false-positive reuse across unrelated domains

## Integration Tests

- Stress round-trip: input length `1973`, final token count `63`, avg chars/token `31.32`, `motif_reuse=7`, `hierarchy remains perfectly balanced`, `ROUNDTRIP: OK`
- Mixed-domain round-trip: input length `724`, final token count `63`, avg chars/token `11.49`, `motif_reuse=0`, `hierarchy remains perfectly balanced`, `ROUNDTRIP: OK`
- Streaming corpus: input length `6691`, final token count `126`, avg chars/token `53.10`, `motif_reuse=14`, `hierarchy remains perfectly balanced across streaming chunks`, `ROUNDTRIP: OK`
- Streaming reuse view: grouped `REUSED MOTIFS (cross-depth)` output stays concentrated in repeated multi-sentence phrase blocks while the mixed-domain portion remains non-reused

## Archived Experiments

| Variant Name                                 | Outcome Summary | Archive Reason |
|----------------------------------------------|-----------------|----------------|
| b1_fractal_gated_v1                          | Survived seeded baseline and real-text follow-up with balanced hierarchy | Useful comparator, but did not outperform the p1 line |
| p1_fractal_hybrid_v1                         | Strong static baseline; perfect hierarchy, `motif_reuse=0` on amplification, stress, and mixed-domain inputs | Archived as the final control baseline behind dynamic-v2 |
| p1_fractal_hybrid_dyn-state-norm_v1          | First dynamic lever; preserved hierarchy and induced reuse, but overshot to `motif_reuse=7` on the original amplification input | Replaced by self-regulating v2 after proving the lever concept |
| p1_fractal_hybrid_dyn-state-norm_v1 (tuned)  | Sensitivity tuning preserved hierarchy but collapsed reuse back to `motif_reuse=0` | Archived because the external sensitivity knob lost the desired signal |
| p2_mandelbrot_v1                             | Survived seeded baseline with expected recursive layout | Archived after no stronger tokenizer signal emerged than the p1 line |
| b3_fractal_hierarchical_v1                   | Survived seeded baseline with expected recursive layout | Archived after no stronger tokenizer signal emerged than the p1 line |
| b4_universal_v1                              | Survived seeded baseline with expected recursive layout | Archived after no stronger tokenizer signal emerged than the p1 line |

## Retired / Failed Tokenizer Variants

| Variant Name                  | Reason for Retirement                  |
|-------------------------------|----------------------------------------|
| (add any that fail later)     | -                                      |

## Bullpen (new tokenizer ideas to rotate in)

- (add new recursive tokenizer ideas here)
