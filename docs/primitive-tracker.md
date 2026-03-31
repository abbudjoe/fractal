# Primitive Tracker

## Active / Promising Variants

| Variant Name                          | Base Primitive     | Lever Type                  | Status                 | Last Fitness | Stability | Perplexity | ARC   | Notes / Next Action |
|---------------------------------------|--------------------|-----------------------------|------------------------|--------------|-----------|------------|-------|---------------------|
| p1_contractive_v1                     | p1                 | contractive gate            | Current truth leader   | 0.58         | 0.53      | 1.54       | 0.68  | Best official Metal ranking so far; keep as ranking reference |
| p1_fractal_hybrid_v1                  | p1                 | static squaring             | Promising              | 0.57         | 1.49      | 1.67       | 0.67  | Won squaring-family proving ground; later medium-stress lane ran over 62m and was stopped, so it needs a lighter intermediate stress preset |
| logistic_chaotic_map_v1               | logistic           | static r-clamp + residual   | Promising              | 0.40         | 1.55      | 4.43       | 0.38  | Won bullpen polish; next step should be minimal stress before another heavy promotion |
| b2_stable_hierarchical_v1             | b2                 | stable hierarchical blend   | Promising              | 0.46         | 0.24      | 1.63       | 0.71  | Strong Gen 2 Metal result; keep in the upper cohort |
| p3_hierarchical_v1                    | p3                 | hierarchical compressor     | Promising              | 0.45         | 0.34      | 1.69       | 0.59  | Strong Gen 2 Metal result; keep in the upper cohort |

## Under Evaluation / Needs Another Pass

| Variant Name                          | Base Primitive     | Lever Type                  | Status                 | Last Fitness | Stability | Perplexity | ARC   | Notes / Next Action |
|---------------------------------------|--------------------|-----------------------------|------------------------|--------------|-----------|------------|-------|---------------------|
| ifs_dyn-radius-depth_v1               | ifs                | dynamic radius by depth     | Viable but slow        | 0.38         | 0.21      | 10.05      | 0.21  | Survived bullpen polish but had severe runtime cost; only promote after a bounded minimal stress lane |
| generalized_mobius_dyn-jitter-norm_v1| generalized_mobius | dynamic jitter by norm      | Unstable / needs work  | 0.13         | 935.92    | 62.04      | 0.18  | Numerically unstable in polish; do not promote until contraction/runtime are better bounded |

## Retired / Failed

| Variant Name                              | Reason for Retirement |
|-------------------------------------------|------------------------|
| p2_mandelbrot_dyn-gate-depth_v1           | Recovered proving-ground run still produced NaN metrics; only throughput changed |
| p2_mandelbrot_dyn-gate-norm_v1            | Minimal-baseline rerun stayed NaN and got slower than the depth-clamp variant |
| b1_fractal_gated_dyn-residual-norm_v1     | Minimal-baseline rerun stayed NaN despite the dynamic residual lever |
| b3_fractal_hierarchical_dyn-radius-depth_v1 | Proving-ground run stayed NaN and showed no learning signal |
| b4_universal_dyn-residual-norm_v1         | Proving-ground run stayed NaN and had extreme runtime cost (`4574.4s`) |

## Bullpen (new ideas to rotate in)

- New mid-stress preset for `p1_fractal_hybrid_v1` before another heavy CUDA run
- Runtime-bounded minimal stress lane for `ifs_dyn-radius-depth_v1`
- Stronger contraction / denominator control pass for `generalized_mobius_dyn-jitter-norm_v1`
