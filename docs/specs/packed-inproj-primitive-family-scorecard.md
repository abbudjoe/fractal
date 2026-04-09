# Packed In-Projection Primitive Family Scorecard

This note records the first full primitive-family sweep after porting the shared
packed input-projection surface across the remaining Path 1 primitive families.

This is a dense, compile-safe comparison surface. It is useful for comparing
primitive-family quality/speed tradeoffs on one shared runtime contract, but it
is not the same surface as the `P20 + block-diagonal-2 + Triton` fast lane.

## Sweep Surface

Shared run contract:

* env: `compile-safe`
* backend: `torch`
* compile mode: `reduce-overhead`
* execution profile: `runtime`
* state transform: `dense`
* wrapper: `scaled + projected + pre-norm-only + standard`
* benchmark: `cuda-faithful-small-v1`
* seed: `42`

## Scorecard

| Primitive | Final loss | Train tok/s | Overall tok/s | CUDA peak MB |
| --- | ---: | ---: | ---: | ---: |
| `P1` | `2.3749` | `382.22` | `434.89` | `47.79` |
| `P2` | `2.3517` | `563.34` | `630.12` | `50.94` |
| `P2.1` | `2.3505` | `460.05` | `530.28` | `56.58` |
| `P2.2` | `2.2652` | `200.80` | `230.97` | `59.09` |
| `P2.3` | `2.3638` | `163.91` | `188.28` | `53.46` |

Artifacts:

* `P1`: `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-672b9aed866b4a4e/20260409T200608Z_a01/remote/artifacts/v3a-python-path1/20260409T200608Z_a01/primitive-hybrid-p1-runtime-scaled-projected-pre-norm-only-standard-dense/report.json`
* `P2`: `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-1feebd523be9ed8f/20260409T202511Z_a01/remote/artifacts/v3a-python-path1/20260409T202511Z_a01/primitive-hybrid-p2-runtime-scaled-projected-pre-norm-only-standard-dense/report.json`
* `P2.1`: `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-1feebd523be9ed8f/20260409T203944Z_a02/remote/artifacts/v3a-python-path1/20260409T203944Z_a02/primitive-hybrid-p2-1-runtime-scaled-projected-pre-norm-only-standard-dense/report.json`
* `P2.2`: `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-1feebd523be9ed8f/20260409T205338Z_a03/remote/artifacts/v3a-python-path1/20260409T205338Z_a03/primitive-hybrid-p2-2-runtime-scaled-projected-pre-norm-only-standard-dense/report.json`
* `P2.3`: `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-672b9aed866b4a4e/20260409T201541Z_a02/remote/artifacts/v3a-python-path1/20260409T201541Z_a02/primitive-hybrid-p2-3-runtime-scaled-projected-pre-norm-only-standard-dense/report.json`

## Ranking

By loss:

1. `P2.2`
2. `P2.1`
3. `P2`
4. `P2.3`
5. `P1`

By train throughput:

1. `P2`
2. `P2.1`
3. `P1`
4. `P2.2`
5. `P2.3`

## Readout

The sweep establishes three distinct roles:

* best dense-family throughput: `P2`
* best dense-family quality: `P2.2`
* best dense-family balance: `P2`, with `P2.1` very close on loss but slower and heavier

This sweep does not displace the current `P20` fast lane. The trusted packed
`in_proj` Triton fast lane remains:

* `P20 + block-diagonal-2 + Triton`
* final loss: `2.2922`
* train throughput: `3722.22 tok/s`
* CUDA peak: `50.06 MB`

So the follow-up question is not “which dense family is globally best?” It is
“which dense family deserves structured/Triton treatment next, if any?”
