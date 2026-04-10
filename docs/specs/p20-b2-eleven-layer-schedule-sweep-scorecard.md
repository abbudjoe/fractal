# P20 vs B2 Eleven-Layer Schedule Sweep

This note freezes the first 11-layer schedule sweep on the shared Path 1 surface.

Surface:
- benchmark: `cuda-faithful-small-v1`
- seed: `42`
- shape: `d_model=128`, `head_count=4`, `ffn_multiplier=4`, `total_layers=11`
- schedule family:
  - `AAAAAAAAAAA`
  - `PAAAAAAAAAA`
  - `AAAPAAAAAAA`
  - `AAAAAPAAAAA`
  - `AAAAAAAAPAA`
  - `AAAAAAAAAAP`
  - `AAAAPPAAAAA`
  - `AAAAPAPAAAA`
  - `AAAAAPAAAAP`

P20 fast-lane contract:
- primitive: `P20`
- execution: `runtime`
- wrapper: `scaled + projected + pre-norm-only + standard`
- state transform: `block-diagonal-2`
- env/runtime: `primitive-triton` + `triton`

B2 comparison contract:
- primitive: `b2-stable-hierarchical`
- execution: `runtime`
- wrapper: `scaled + projected + pre-norm-only + standard`
- state transform: `dense`
- env/runtime: `compile-safe` + `torch`

Shared attention-only control:
- `AAAAAAAAAAA`: loss `2.5761`, train `1274.91 tok/s`, CUDA peak `60.12 MB`

P20 results:

| Schedule | Final loss | Train tok/s | CUDA peak MB |
| --- | ---: | ---: | ---: |
| `PAAAAAAAAAA` | `2.2853` | `1410.14` | `60.44` |
| `AAAAPPAAAAA` | `2.2992` | `1672.91` | `60.76` |
| `AAAAPAPAAAA` | `2.3199` | `1613.89` | `60.76` |
| `AAAAAPAAAAP` | `2.3334` | `1209.25` | `60.76` |
| `AAAAAAAAAAP` | `2.3522` | `1796.00` | `60.44` |
| `AAAPAAAAAAA` | `2.3547` | `1651.15` | `60.44` |
| `AAAAAPAAAAA` | `2.3761` | `1575.91` | `60.44` |
| `AAAAAAAAPAA` | `2.3810` | `1858.37` | `60.44` |

B2 results:

| Schedule | Final loss | Train tok/s | CUDA peak MB |
| --- | ---: | ---: | ---: |
| `AAAAAPAAAAP` | `2.3566` | `247.10` | `60.39` |
| `AAAAAPAAAAA` | `2.3711` | `368.66` | `59.75` |
| `AAAAPPAAAAA` | `2.3725` | `244.32` | `60.39` |
| `AAAPAAAAAAA` | `2.3727` | `344.60` | `59.75` |
| `PAAAAAAAAAA` | `2.3749` | `319.25` | `59.75` |
| `AAAAAAAAPAA` | `2.3764` | `374.19` | `59.75` |
| `AAAAPAPAAAA` | `2.3821` | `247.38` | `60.39` |
| `AAAAAAAAAAP` | `2.3850` | `413.30` | `59.75` |

Readout:
- Sparse primitive placement is the right regime for both families on this surface.
- The best-quality `P20` schedule is `PAAAAAAAAAA`.
- The best balance schedule for `P20` is `AAAAPPAAAAA`.
- `b2-stable-hierarchical` prefers sparse placement too, but it is not competitive with `P20` on either loss or throughput.
- `P20` remains the promotable family; `b2-stable-hierarchical` does not warrant backend/runtime follow-on work on this surface.

Artifact anchors:
- P20 best-quality: `.runpod-local-logs/runpod-results/exp-f95b51bf7166f66b/20260410T141826Z_a02`
- P20 best-balance: `.runpod-local-logs/runpod-results/exp-f95b51bf7166f66b/20260410T145859Z_a07`
- B2 best-quality: `.runpod-local-logs/runpod-results/exp-f95b51bf7166f66b/20260410T172848Z_a18`
- shared attention-only control: `.runpod-local-logs/runpod-results/exp-f95b51bf7166f66b/20260410T141041Z_a01`
