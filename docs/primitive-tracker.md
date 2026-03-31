# Primitive Tracker

## Active / Promising Variants

| Variant Name                          | Base Primitive     | Lever Type                  | Status                 | Last Fitness | Stability | Perplexity | ARC   | Notes / Next Action |
|---------------------------------------|--------------------|-----------------------------|------------------------|--------------|-----------|------------|-------|---------------------|
| p1_contractive_v1                     | p1                 | contractive gate            | Truth leader (reference only) | 0.58         | 0.53      | 1.54       | 0.68  | Keep as benchmark |
| p1_fractal_hybrid_v1                  | p1                 | static squaring             | Co-Leader              | 0.61         | 1.24      | 1.73       | 0.74  | Still active |
| p1_fractal_hybrid_composite_v1        | p1                 | dynamic compositing lever   | Fractal Core Co-Leader | 0.57         | 0.33      | 1.86       | 0.76  | FIRST SUCCESSFUL COMPOSITING TEST — highest ARC ever + Stability below 0.80; lock as primary fractal primitive; next: stability-lever mutation |
| logistic_chaotic_map_v1               | logistic           | static r-clamp + residual   | Co-Leader Contender    | 0.58         | 0.59      | 1.91       | 0.46  | PROMOTED — survived full-medium-stress; next: compositing test with hybrid composite |
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
| p1_fractal_hybrid_dyn-gate_v1             | Timed out @1800s on train cost — too expensive under current harness |
| julia_recursive_escape_v1                 | Timed out @1200s on eval phase (train completed); eval bottleneck confirmed — demote to tokenizer track only |

## Bullpen (new ideas to rotate in)

- Stability-lever mutation on `p1_fractal_hybrid_composite_v1`
- Heavy-stress validation on `p1_fractal_hybrid_composite_v1`
- Compositing test for `logistic_chaotic_map_v1` inside the new fractal composite shell

## Latest Uncommitted Run Results

| Variant Name                               | Preset                   | Status    | Last Fitness | Stability | Perplexity | ARC  | tok/s | Notes / Next Action |
|--------------------------------------------|--------------------------|-----------|--------------|-----------|------------|------|-------|---------------------|
| mandelbox_recursive_dyn-escape-radius_v1   | proving_ground_baseline  | Completed | 0.13         | 0.00      | 11.18      | 0.00 | 37    | Fast proving-ground completion with no NaN, but learning signal was weak; keep in bullpen only if we want one more mutation pass, otherwise retire cleanly |
| p3_hierarchical_v1                         | minimal_stress_lane      | Completed | 0.58         | 0.55      | 1.66       | 0.44 | 34    | Clean bounded-stress completion with co-leader-range fitness, but weaker ARC than its earlier best run; likely merits promotion into a harder confirmation lane rather than retirement |
| b2_stable_hierarchical_v1                  | minimal_stress_lane      | Timed out | n/a          | n/a       | n/a        | n/a  | n/a   | Completed train (`1138.7s`) and stability (`25.0s`) but hit the bounded `1200s` timeout in perplexity; this looks like an eval-budget issue, not a collapse signal |
