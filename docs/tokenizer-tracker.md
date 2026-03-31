# Tokenizer Primitive Tracker

## Active / Promising Tokenizer Variants

| Variant Name                          | Base Primitive     | Lever Type                  | Status       | Token Count | Notes / Next Action |
|---------------------------------------|--------------------|-----------------------------|--------------|-------------|---------------------|
| b1_fractal_gated_v1                   | b1                 | static gated + squaring     | Promising    | 43          | Survived seeded test |
| p1_fractal_hybrid_v1                  | p1                 | static squaring             | Promising    | 43          | Strong performer     |
| p2_mandelbrot_v1                      | p2                 | pure squaring               | Testing      | 43          | Survived seeded test |
| b3_fractal_hierarchical_v1            | b3                 | squaring + hierarchy        | Testing      | 43          | Survived seeded test |
| b4_universal_v1                       | b4                 | full fusion                 | Testing      | 43          | Survived seeded test |

## Latest Seeded Baseline

- `seed=42`, `dim=64`, `max_depth=6`
- `b1_fractal_gated_v1`: survived, `43` tokens
- `p1_fractal_hybrid_v1`: survived, `43` tokens, strongest current performer
- `p2_mandelbrot_v1`: survived, `43` tokens
- `b3_fractal_hierarchical_v1`: survived, `43` tokens
- `b4_universal_v1`: survived, `43` tokens

## Latest Real-Text Follow-Up

- Comparison input:
  `"The quick brown fox jumps over the lazy dog and the cat sat on the mat while watching the birds."`
- `b1_fractal_gated_v1`: balanced depth profile `d0:1, d1:2, d2:4, d3:8, d4:16, d5:32`; no repeated digests by depth
- `p1_fractal_hybrid_v1`: balanced depth profile `d0:1, d1:2, d2:4, d3:8, d4:16, d5:32`; no repeated digests by depth; stronger current lead
- Longer `p1_fractal_hybrid_v1` input:
  `"The quick brown fox jumps over the lazy dog and the cat sat on the mat while watching the birds fly high above the old oak tree in the quiet meadow on a sunny afternoon."`
- Longer `p1_fractal_hybrid_v1` result: balanced depth profile still holds as `d0:1, d1:2, d2:4, d3:8, d4:16, d5:32`; no repeated digests by depth on this sample

## Latest Motif-Reuse Follow-Up

- Motif input:
  `"The cat sat on the mat. The dog sat on the mat. The bird sat on the mat. The fox sat on the mat."`
- `p1_fractal_hybrid_v1`: total depth profile held as `d0:1, d1:2, d2:4, d3:8, d4:16, d5:32`, while unique digests dipped to `d4:14, d5:31`
- Motif reuse signal: weak local reuse appeared at deeper levels, but no repeated motif digests appeared across depths
- Amplification input:
  `"The cat sat on the mat. The dog sat on the mat. The bird sat on the mat. The fox sat on the mat. The cat sat on the mat again."`
- Amplification result: total and unique depth profiles both returned to `d0:1, d1:2, d2:4, d3:8, d4:16, d5:32`
- Amplification note: motif reuse weakened compared to the previous motif sample, so repeated phrasing did not yet amplify hierarchical clustering on this run
- Dynamic lever candidate: `p1_fractal_hybrid_dyn-state-norm_v1`
- Lever description: rolling normalized state norm modulates a cross-depth motif-distance threshold, allowing digest reuse only when a prior motif is similar enough and still unused at the current depth
- Sensitivity knob: `lever_sensitivity` attenuates the state-norm similarity threshold as rolling norm falls, so lower values damp reuse less aggressively and higher values suppress reuse sooner
- Expected test output format: `static_unique_tokens_by_depth=...`, `dynamic_unique_tokens_by_depth=...`, `static_motif_reuse_count=...`, `dynamic_motif_reuse_count=...`, `amplification_note=...`
- Latest tuned result (`lever_sensitivity=0.6`): static `motif_reuse=0`, dynamic `motif_reuse=0`; both preserved `d0:1, d1:2, d2:4, d3:8, d4:16, d5:32`, so the tuned lever under-shot the target reuse window on this sample

## Retired / Failed Tokenizer Variants

| Variant Name                  | Reason for Retirement                  |
|-------------------------------|----------------------------------------|
| (add any that fail later)     | -                                      |

## Bullpen (new tokenizer ideas to rotate in)

- (add new recursive tokenizer ideas here)
