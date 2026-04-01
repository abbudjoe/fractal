# Primitive Tracker

Use this tracker alongside [promotion-policy.md](/Users/joseph/fractal/docs/promotion-policy.md).

## Active / Promising Variants

`tok/s` is tracked here as a secondary production-readiness signal. Treat it as authoritative only for same-preset, same-backend comparisons.

| Variant Name                          | Base Primitive     | Lever Type                  | Status                 | Last Fitness | Stability | Perplexity | ARC   | tok/s | Notes / Next Action |
|---------------------------------------|--------------------|-----------------------------|------------------------|--------------|-----------|------------|-------|-------|---------------------|
| p1_fractal_hybrid_composite_v1        | p1                 | dynamic compositing lever   | Fractal Core Leader    | 0.66         | 0.47      | 1.53       | 0.81  | 3     | Won the authoritative winner-lane bakeoff on aggregate; advance as the primary next-phase fractal training core and preserve for the frozen canonical rerun |
| p1_fractal_hybrid_v1                  | p1                 | static squaring             | Fractal Core Co-Leader | 0.65         | 0.46      | 1.51       | 0.76  | 3     | Finished a close second in the authoritative bakeoff; keep in the top training cohort and compare directly against the composite winner at larger scale |
| p1_contractive_v1                     | p1                 | contractive gate            | Benchmark Reference    | 0.65         | 0.45      | 1.54       | 0.79  | 5     | Remains the strongest stable benchmark and throughput anchor; keep in every scaled phase as the control model |
| logistic_chaotic_map_v1               | logistic           | static r-clamp + residual   | Co-Leader Contender    | 0.58         | 0.59      | 1.91       | 0.46  | 3     | Earlier full-medium-stress run cleared the winner-lane gates, but it was not part of the final authoritative 3-way bakeoff; keep alive as the next alternate-family challenger |
| p3_hierarchical_v1                    | p3                 | hierarchical compressor     | Validation Leader      | 0.58         | 0.55      | 1.66       | 0.44  | 34    | Best non-winner validation candidate; merits one harder confirmation lane before it competes for the main training cohort |
| b2_stable_hierarchical_v1             | b2                 | stable hierarchical blend   | Eval-Constrained       | 0.46         | 0.24      | 1.63       | 0.71  | 104   | Latest validation run finished train and stability but timed out in perplexity; allow one reduced-eval rerun before retirement |

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
| mandelbox_recursive_dyn-escape-radius_v1  | Low-signal proving-ground result (`fitness 0.13`, `ARC 0.00`) — retire unless a new root-cause lever emerges |

## Bullpen (new ideas to rotate in)

- No new bullpen additions until the winner lane narrows.
- Frozen canonical rerun of the top three winner-lane variants from one identical commit
- Stability-lever mutation on `p1_fractal_hybrid_composite_v1`
- Reduced-eval confirmation rerun for `b2_stable_hierarchical_v1`
- Harder confirmation lane for `p3_hierarchical_v1`
