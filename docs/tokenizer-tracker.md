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
- Static amplification baseline: total and unique depth profiles both returned to `d0:1, d1:2, d2:4, d3:8, d4:16, d5:32`
- Static amplification note: motif reuse weakened compared to the previous motif sample, so repeated phrasing alone did not amplify hierarchical clustering on that baseline run
- Dynamic lever candidate: `p1_fractal_hybrid_dyn-state-norm_v2`
- Lever description: v2 self-regulates motif reuse from the primitive’s own rolling state norm, current recursion depth, and nearest-vs-local motif distance field, so reuse only opens when a prior cross-depth digest is both locally standout and still unused at the current depth
- Expected test output format: `static_unique_tokens_by_depth=...`, `dynamic_lever_type=v2-self-regulating`, `dynamic_unique_tokens_by_depth=...`, `static_motif_reuse_count=...`, `dynamic_motif_reuse_count=...`, `amplification_note=...`
- Live-confirmed dynamic-v2 result: static `motif_reuse=0`, dynamic `motif_reuse=4`; both preserved `d0:1, d1:2, d2:4, d3:8, d4:16, d5:32`, and the latest run printed `amplification_note=v2 self-regulating lever hit the target window (4 repeated motifs, static=0)`

## Retired / Failed Tokenizer Variants

| Variant Name                  | Reason for Retirement                  |
|-------------------------------|----------------------------------------|
| (add any that fail later)     | -                                      |

## Bullpen (new tokenizer ideas to rotate in)

- (add new recursive tokenizer ideas here)
