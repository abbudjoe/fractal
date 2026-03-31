# Primitive Tracker

## Active / Promising Variants

| Variant Name                          | Base Primitive     | Lever Type                  | Status                 | Last Fitness | Stability | Perplexity | ARC   | Notes / Next Action |
|---------------------------------------|--------------------|-----------------------------|------------------------|--------------|-----------|------------|-------|---------------------|
| p1_contractive_v1                     | p1                 | contractive gate            | Current truth leader   | 0.58         | 0.53      | 1.54       | 0.68  | Keep as ranking reference |
| p1_fractal_hybrid_v1                  | p1                 | static squaring             | **Co-Leader**          | **0.61**     | 1.24      | 1.73       | **0.74** | **NEW CO-LEADER** - strongest fractal signal on Fitness+ARC; next: compositing test |
| logistic_chaotic_map_v1               | logistic           | static r-clamp + residual   | Upper Cohort           | 0.46         | 1.18      | 3.49       | 0.44  | Promoted - ready for heavy stress or compositing |
| b2_stable_hierarchical_v1             | b2                 | stable hierarchical blend   | Upper Cohort           | 0.46         | 0.24      | 1.63       | 0.71  | Keep |
| p3_hierarchical_v1                    | p3                 | hierarchical compressor     | Upper Cohort           | 0.45         | 0.34      | 1.69       | 0.59  | Keep |

## Retired / Failed

| Variant Name                              | Reason for Retirement |
|-------------------------------------------|------------------------|
| p2_mandelbrot_dyn-gate-depth_v1           | Recovered proving-ground run still produced NaN metrics; only throughput changed |
| p2_mandelbrot_dyn-gate-norm_v1            | Minimal-baseline rerun stayed NaN and got slower than the depth-clamp variant |
| b1_fractal_gated_dyn-residual-norm_v1     | Minimal-baseline rerun stayed NaN despite the dynamic residual lever |
| b3_fractal_hierarchical_dyn-radius-depth_v1 | Proving-ground run stayed NaN and showed no learning signal |
| b4_universal_dyn-residual-norm_v1         | Proving-ground run stayed NaN and had extreme runtime cost (`4574.4s`) |
| generalized_mobius_dyn-jitter-norm_v1     | Superseded by v2 after instability in polish |
| generalized_mobius_dyn-jitter-norm_v2     | NaN failure even after contraction strengthening |
| ifs_dyn-radius-depth_v1                   | Repeated timeout on bounded minimal stress lane (demoted to tokenizer track only) |

## Bullpen (new ideas to rotate in)

- Compositing test for `p1_fractal_hybrid_v1` with `p1_contractive_v1`
- Heavy stress or compositing pass for `logistic_chaotic_map_v1`

## Latest Uncommitted Run Results

| Variant Name               | Preset               | Status                     | Last Fitness | Stability | Perplexity | ARC | tok/s | Notes / Next Action |
|---------------------------|----------------------|----------------------------|--------------|-----------|------------|-----|-------|---------------------|
| julia_recursive_escape_v1 | minimal_stress_lane  | Timed out                  | n/a          | n/a       | n/a        | n/a | n/a   | Current balanced-node rerun completed train (`1118.8s`) and stability (`17.3s`) but timed out in perplexity after `1200s`; next fix should target eval/runtime cost rather than training stability alone |
| p1_fractal_hybrid_composite_v1 | lighter_intermediate_stress | Completed         | 0.57         | 0.33      | 1.86       | 0.76 | 3    | First-ever compositing test completed cleanly; ARC improved to `0.76`, stability dropped below `0.80`, and fitness held near the co-leader band, so this is a real promotion candidate |
| p1_fractal_hybrid_dyn-gate_v1 | lighter_intermediate_stress | Timed out          | n/a          | n/a       | n/a        | n/a | n/a   | Current balanced-node rerun reached at least `36/48` train steps by `1462.4s` but still hit the bounded `1800s` timeout before reporting train completion; this lever remains too expensive on the present control surface |
| logistic_chaotic_map_v1 | full_medium_stress | Completed | 0.58 | 0.59 | 1.91 | 0.46 | 3 | Heavy-stress rerun completed cleanly; stability improved sharply from the earlier upper-cohort pass while fitness rose into co-leader range, so this variant now merits serious head-to-head consideration |
