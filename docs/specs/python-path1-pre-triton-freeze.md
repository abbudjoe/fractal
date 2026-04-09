# Python Path 1 Pre-Triton Freeze

This note freezes the useful pre-Triton Path 1 primitive results before the
primitive line moves onto its own newer Triton stack.

## Frozen Baselines

The current benchmark truths are:

* official/native `A+M` surface (`official-mamba3`)
  * `A`: `2.5425`, `5723.59 tok/s`, `50.10 MB`
  * native `A+M`: `2.2892`, `3049.83 tok/s`, `53.46 MB`
  * uncompiled dense `A+P2.0 runtime`: `2.3280`, `351.55 tok/s`, `49.68 MB`

* compile-focused primitive optimization surface (`compile-safe`)
  * best quality pre-Triton `P2.0`:
    * dense packed-scan compiled `P2.0`: `2.3311`, `1361.50 tok/s`, `49.43 MB`
  * best speed pre-Triton `P2.0`:
    * `block-diagonal-4` compiled `P2.0`: `2.3418`, `1831.52 tok/s`, `48.74 MB`

These are the two pre-Triton primitive anchors we keep:

* `dense`
* `block-diagonal-4`

## Retired Variants

These variants are retired and are not part of the frozen surface:

* `P2.3` runtime
* `block-diagonal-4-low-rank-8`
* `identity-biased-block-diagonal-4`

They either lost to the dense baseline on quality without enough speed upside,
or lost to `block-diagonal-4` on both quality and speed.

## Control-Plane Consequence

The primitive line now carries three explicit CUDA env surfaces:

* `official-mamba3`
* `compile-safe`
* `primitive-triton`

The `primitive-triton` surface is the upgrade target for future custom primitive
Triton kernels. It is separate from both:

* native official Mamba benchmarking
* Torch-Inductor compile experiments

That separation is intentional and enforced in typed runtime config so the next
upgrade step does not blur unlike surfaces together.
