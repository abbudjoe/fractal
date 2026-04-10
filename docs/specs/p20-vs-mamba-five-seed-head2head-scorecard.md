# P20 vs Native Mamba Five-Seed Head-to-Head

This note freezes the first multi-seed comparison between the current `P20`
fast lane and the native Mamba reference lane on the shared Path 1 benchmark
surface.

## Surface

Common benchmark contract:

* benchmark: `cuda-faithful-small-v1`
* shape: `d_model=128`, `layers=8`, `heads=4`, `ffn_multiplier=4`
* schedule: `A-P-A-P-A-P-A-P`
* dtype: `bf16`
* seed set: `42`, `7`, `13`, `123`, `256`

Compared lanes:

* native Mamba:
  * env: `official-mamba3`
  * variant: `reference-ssm-hybrid-mamba3-siso-runtime`
* `P20` fast lane:
  * env: `primitive-triton`
  * backend: `triton`
  * primitive: `P20`
  * wrapper: `scaled + projected + pre-norm-only + standard`
  * state transform: `block-diagonal-2`
  * packed input projection: enabled

## Scorecard

| Seed | Mamba loss | P20 loss | Loss winner | Mamba train tok/s | P20 train tok/s | Throughput winner | Mamba CUDA MB | P20 CUDA MB |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| `42` | `2.2892` | `2.2922` | Mamba | `3049.83` | `3722.22` | P20 | `53.46` | `50.06` |
| `7` | `2.2378` | `2.3321` | Mamba | `2651.67` | `3999.85` | P20 | `53.46` | `50.06` |
| `13` | `2.2551` | `2.3298` | Mamba | `1190.58` | `2149.41` | P20 | `53.46` | `50.06` |
| `123` | `2.2932` | `2.2354` | P20 | `1358.03` | `2014.76` | P20 | `53.46` | `50.06` |
| `256` | `2.3353` | `2.3109` | P20 | `1454.55` | `1080.66` | Mamba | `53.46` | `50.06` |

Five-seed averages:

* native Mamba:
  * mean loss: `2.2821`
  * mean train throughput: `1940.93 tok/s`
  * CUDA peak: `53.46 MB`
* `P20` fast lane:
  * mean loss: `2.3001`
  * mean train throughput: `2593.38 tok/s`
  * CUDA peak: `50.06 MB`

## Read

The five-seed result is now stable enough to summarize plainly:

* native Mamba is still the slight quality champion on average
* `P20` is still the systems champion on average
* `P20` wins memory on all five seeds
* `P20` wins train throughput on four of the five seeds
* native Mamba wins loss on three of the five seeds
* `P20` wins loss on two of the five seeds

That is a much tighter result than the single-seed story suggested.

The practical interpretation is:

* if the primary objective is mean eval loss on this surface, native Mamba still
  has the edge
* if the objective is the combined speed-memory frontier, `P20` is the better
  lane today

## Artifact Lineage

Seed `42` uses the previously frozen full-pass checkpoint in:

* `/Users/joseph/fractal/docs/specs/p20-packed-inproj-freeze.md`

Fresh local artifacts for the sweep seeds:

* seed `7` native Mamba:
  * `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-0fbc4bf3692e0cf8/20260410T014949Z_a01/remote/artifacts/v3a-python-path1/20260410T014949Z_a01/reference-ssm-hybrid-mamba3-siso-runtime/report.json`
* seed `7` `P20`:
  * `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-0fbc4bf3692e0cf8/20260410T020957Z_a02/remote/artifacts/v3a-python-path1/20260410T020957Z_a02/primitive-hybrid-p2-0-runtime-scaled-projected-pre-norm-only-standard-block-diagonal-2/report.json`
* seed `13` native Mamba:
  * `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-5f35e4ba47a90991/20260410T022354Z_a01/remote/artifacts/v3a-python-path1/20260410T022354Z_a01/reference-ssm-hybrid-mamba3-siso-runtime/report.json`
* seed `13` `P20`:
  * `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-5f35e4ba47a90991/20260410T024854Z_a02/remote/artifacts/v3a-python-path1/20260410T024854Z_a02/primitive-hybrid-p2-0-runtime-scaled-projected-pre-norm-only-standard-block-diagonal-2/report.json`
* seed `123` native Mamba:
  * `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-3a6c6af016712882/20260410T025809Z_a01/remote/artifacts/v3a-python-path1/20260410T025809Z_a01/reference-ssm-hybrid-mamba3-siso-runtime/report.json`
* seed `123` `P20`:
  * `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-3a6c6af016712882/20260410T031331Z_a02/remote/artifacts/v3a-python-path1/20260410T031331Z_a02/primitive-hybrid-p2-0-runtime-scaled-projected-pre-norm-only-standard-block-diagonal-2/report.json`
* seed `256` native Mamba:
  * `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-6357a4a8878fd09c/20260410T032205Z_a01/remote/artifacts/v3a-python-path1/20260410T032205Z_a01/reference-ssm-hybrid-mamba3-siso-runtime/report.json`
* seed `256` `P20` rerun:
  * `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-6357a4a8878fd09c/20260410T034459Z_a02/remote/artifacts/v3a-python-path1/20260410T034459Z_a02/primitive-hybrid-p2-0-runtime-scaled-projected-pre-norm-only-standard-block-diagonal-2/report.json`
