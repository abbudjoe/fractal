# P20 Packed In-Projection Freeze

This note freezes the first Path 1 primitive result that closes the gap to the
native Mamba baseline on the live CUDA benchmark surface.

## What Changed

`P20` previously assembled its four input-side projection streams by packing
separate layer weights at runtime. That was better than four totally separate
matmuls, but it still hid the real projection contract from the runtime and the
kernel stack.

The fixed surface now exposes one true packed input projection module:

* `/Users/joseph/fractal/python/runtime/recurrent.py`
* `/Users/joseph/fractal/python/models/primitives.py`

That change is architectural, not cosmetic. The runtime can now see one stable
`in_proj` surface instead of reconstructing it ad hoc inside the primitive.

## Frozen Result

Trusted confirmation run:

* env: `primitive-triton`
* backend: `triton`
* primitive: `P20`
* wrapper: `scaled + projected + pre-norm-only + standard`
* state transform: `block-diagonal-2`
* surface: `cuda-faithful-small-v1`
* seed: `42`

Trusted artifact:

* `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-0d67e0896d081bd9/20260409T181816Z_a02/remote/artifacts/v3a-python-path1/20260409T181816Z_a02/primitive-hybrid-p2-0-runtime-scaled-projected-pre-norm-only-standard-block-diagonal-2/report.json`

Frozen metrics:

* final loss: `2.2922`
* train throughput: `3722.22 tok/s`
* overall throughput: `4169.70 tok/s`
* CUDA peak: `50.06 MB`

## Comparison

Previous `block-diagonal-2` Triton baseline:

* final loss: `2.3039`
* train throughput: `1686.39 tok/s`
* CUDA peak: `50.06 MB`

Native `A+M` baseline on the same benchmark family:

* final loss: `2.2892`
* train throughput: `3049.83 tok/s`
* CUDA peak: `53.46 MB`

So the packed `in_proj` contract moved `P20 block-diagonal-2` to:

* essentially tied native Mamba quality
* higher train throughput than native Mamba on this surface
* lower CUDA memory than native Mamba

## Control-Plane Consequence

This result changes the optimization frontier.

The important missing contract was not another small kernel heuristic. It was a
real packed projection surface that the runtime could optimize against.

That means future reusable work should preserve this split:

* architecture-specific optimization taxonomy in:
  * `/Users/joseph/fractal/python/specs/runtime.py`
* reusable recurrent runtime substrate in:
  * `/Users/joseph/fractal/python/runtime/`
* experiment wiring in:
  * `/Users/joseph/fractal/python/models/`

Any later primitive experiments should reuse the explicit packed projection
surface rather than reconstruct packed inputs dynamically inside model code.
