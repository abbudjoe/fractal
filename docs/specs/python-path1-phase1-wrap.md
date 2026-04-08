# Python Path 1 Phase 1 Wrap

Status: Phase 1 complete, Phase 2 opened

## What This Checkpoint Means

Path 1 now has one coherent Python research substrate and one decision-grade
native CUDA trio on the shared `cuda-faithful-small-v1` surface.

This closes the migration phase where:

- Python became the architecture source of truth for Path 1
- the old one-off Mamba benchmark script stopped being an architecture island
- the first runtime-oriented primitive lane became real enough to benchmark
- the benchmark contract for `A`, native `A+M`, and `A+P` became shared

## Benchmark Surface

Benchmark:

- `cuda-faithful-small-v1`

Seed:

- `42`

Runtime policy:

- backend `cuda`
- dtype `bf16`
- `1` warmup eval batch
- `1` warmup train step
- full train pass
- full eval pass

Corpus:

- frozen FineWeb slice
- `961` train windows
- `94` eval windows

## Current Trio Result

Native Python CUDA results:

- `A`
  - final loss `2.5425`
  - train throughput `5723.59 tok/s`
  - CUDA peak `50.10 MB`
- native `A+M`
  - final loss `2.2892`
  - train throughput `3049.83 tok/s`
  - CUDA peak `53.46 MB`
- `A+P2.0 runtime`
  - final loss `2.3280`
  - train throughput `351.55 tok/s`
  - CUDA peak `49.68 MB`

Interpretation:

- `P2.0 runtime` is a real native-CUDA contender
- it is much closer to native Mamba on quality than the Rust reference stack
  suggested
- it is slightly lighter on CUDA memory than native `A+M`
- the remaining large gap is throughput, not viability

## What Finished In Phase 1

- shared Python package for Path 1 and future mini-MoE work
- first-class Python Path 1 CLI and runner surface
- typed benchmark, manifest, and reporting contracts
- typed primitive `reference` and `runtime` execution profiles
- explicit Python native reference-SSM adapter around official Mamba-3
- SDPA-based local attention path for the `A` baseline
- first runtime-oriented `P2.0` lane
- runtime/reference equivalence coverage across primitive families
- RunPod-native smoke, trio, and profiling runners

## What Remains Deferred

- custom-kernel or Triton work for `P`
- runtime treatment for `P2.3`
- larger-seed or larger-slice policy beyond the current faithful-small surface
- mini-MoE routing experiments
- consolidation of the dirty `/Users/joseph/fractal` checkout

## Immediate Next Step

Phase 2 should focus on the primitive scan core itself.

The main question is no longer whether `P2.0` belongs in the race. It does.
The main question is whether we can shrink the throughput gap by improving the
runtime shape of `scan_runtime` before reaching for custom kernels.
