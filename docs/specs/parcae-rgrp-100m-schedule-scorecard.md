# Parcae/RGRP 100M Schedule Scorecard

Date: 2026-04-30

## Purpose

This note records the corrected 100M-class Parcae/RGRP comparison after the
bf16 launcher fix, the stabilized attention control, the official Mamba run,
and the first bounded layer-schedule scout.

The important correction relative to the older 2026-04-29 freeze is that the
seed-45 attention control no longer collapses when run with the stabilized
contract. Attention is currently the quality leader at 8192 steps, while RGRP is
close on quality and competitive on speed/memory. RGRP clearly beats the
official Mamba 130M baseline under the tested setup.

## Shared Corrected Contract

Unless otherwise noted:

```text
runner: SageMaker token-cache Path 1 CUDA scout
token cache: fineweb-cc-main-2024-10-openllama-750m
seed: 45
data_seed: 45
dtype: bf16
seq_len: 512
batch_size: 32
eval_batches: 4
train_loss_record_interval: 128
position_encoding_kind: learned
attention_position_contract: attention-only
final_norm_kind: layernorm
attention_kernel: flex-local
local_window: 512
optimizer: adam
learning_rate: 3e-4 for attention/RGRP
```

RGRP-specific common contract:

```text
variant: parcae-hourglass-p20-control-looped-attention
primitive_runtime_backend: triton
parcae_prelude_norm_kind: rmsnorm
parcae_loop_count: 2
parcae_backward_steps: 1
parcae_control_position_kind: learned
parcae_control_state_transform: trainable-block-diagonal-8
parcae_loop_update_backend: triton-loop-forward
parcae_band_block_contract: compiled-direct
parcae_band_prepare_backend: compiled
```

Official Mamba used the successful 1024-smoke learning-rate contract:

```text
learning_rate: 1e-3
wheelhouse: mamba-wheelhouse-cu124-l4h100-nodeps-0429a
```

## 8192-Step Corrected Head-To-Head

| Lane | Job | Params | Final loss | Tok/s | Peak CUDA | Train time |
|---|---|---:|---:|---:|---:|---:|
| attention fixed control | `attn704h11-flex8192-s45-lr3e4-layernorm-control-0429a` | 117.26M | 3.9747 | 38,024 | 11.10 GB | 3996s |
| RGRP quality | `rgrp768-b32322-loop256-stable8192-s45-0430a` | 111.09M | 3.9946 | 38,352 | 10.77 GB | 3956s |
| RGRP efficient | `rgrp704-b42123-loop320-stable8192-s45-0430a` | 100.05M | 4.0341 | 39,349 | 10.49 GB | 3875s |
| official Mamba 130M | `official-mamba130-wheelhouse8192-s45-0430a` | 115.10M | 4.0787 | 25,350 | 11.77 GB | 5754s |

Interpretation:

- Attention is the 8192-step quality leader by `0.0199` loss over the best RGRP
  lane.
- RGRP quality is slightly faster, smaller, and lower-memory than attention, but
  it is not a quality win.
- RGRP efficient is meaningfully smaller/faster and remains within `0.0594`
  loss of attention.
- RGRP beats official Mamba on loss, throughput, and memory in this setup.
- Mamba should get an LR-matched follow-up only if we need a stricter Mamba
  baseline; it used `1e-3`, while attention/RGRP used `3e-4`.

Current promoted 100M RGRP lanes:

```text
RGRP-quality:
  d_model: 768
  head_count: 12
  total_layers: 12
  band schedule: 3,2,3,2,2
  loop_d_model: 256
  loop_head_count: 4
  loop_count: 2

RGRP-efficient:
  d_model: 704
  head_count: 11
  total_layers: 12
  band schedule: 4,2,1,2,3
  loop_d_model: 320
  loop_head_count: 5
  loop_count: 2
```

## 1024-Step Schedule Scout

This scout tested new schedules only at the `d704 loop320x2` efficient scale.
The existing anchors are included below for comparison but were not rerun as
part of the new-schedule sweep.

Hardware note:

- Six schedules ran on `ml.g6.2xlarge`.
- Two overflow schedules used the additional approved SageMaker quota:
  `ml.g5.12xlarge` and `ml.g6e.24xlarge`.
- Loss is comparable across these runs; throughput and memory must be read with
  the instance type attached.

| Rank | Schedule / Lane | Job | Instance | Params | Final loss | Tok/s | Peak CUDA |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | incumbent quality `d768 3,2,3,2,2 loop256` | `rgrp768-b32322-loop256-stable1024-s45-0430a` | `ml.g6.2xlarge` | 111.09M | 5.1455 | 39,760 | 10.77 GB |
| 2 | `3,1,4,1,3` | `rgrp704-sched31413-loop320-stable1024-s45-0430a` | `ml.g6.2xlarge` | 109.50M | 5.1613 | 36,822 | 11.13 GB |
| 3 | `5,1,2,1,3` | `rgrp704-sched51213-loop320-stable1024-s45-g5x12-0430a` | `ml.g5.12xlarge` | 109.50M | 5.1642 | 50,820 | 11.13 GB |
| 4 | incumbent efficient `4,2,1,2,3` | `rgrp704-b42123-stable1024-s45-0430a` | `ml.g6.2xlarge` | 100.05M | 5.1685 | 40,366 | 10.49 GB |
| 5 | `4,2,2,2,2` | `rgrp704-sched42222-loop320-stable1024-s45-0430a` | `ml.g6.2xlarge` | 100.05M | 5.1714 | 39,313 | 10.49 GB |
| 6 | `2,3,3,1,3` | `rgrp704-sched23313-loop320-stable1024-s45-g6e24-0430a` | `ml.g6e.24xlarge` | 100.05M | 5.1715 | 134,141 | 10.49 GB |
| 7 | `3,2,4,2,1` | `rgrp704-sched32421-loop320-stable1024-s45-0430a` | `ml.g6.2xlarge` | 100.05M | 5.1737 | 40,182 | 10.49 GB |
| 8 | `2,2,4,2,2` | `rgrp704-sched22422-loop320-stable1024-s45-0430a` | `ml.g6.2xlarge` | 100.05M | 5.1741 | 39,611 | 10.49 GB |
| 9 | `3,2,2,3,2` | `rgrp704-sched32232-loop320-stable1024-s45-0430a` | `ml.g6.2xlarge` | 95.33M | 5.1761 | 41,649 | 10.17 GB |
| 10 | `2,3,2,2,3` | `rgrp704-sched23223-loop320-stable1024-s45-0430a` | `ml.g6.2xlarge` | 95.33M | 5.1805 | 41,405 | 10.17 GB |
| 11 | attention 1024 control | `attn704h11-flex1024-s45-lr3e4-layernorm-0429a` | `ml.g6.2xlarge` | 117.26M | 5.2034 | 39,090 | 11.10 GB |

Interpretation:

- No new 100M schedule beat the `d768 loop256` quality incumbent.
- `3,1,4,1,3` is the best new schedule by 1024-step loss, but it is larger and
  slower than the efficient incumbent and still behind the quality incumbent.
- `5,1,2,1,3` is interesting as a mixed-hardware loss read, but its speed cannot
  be compared directly with the L4 runs.
- The requested `3,2,4,2,1` schedule did not pop at 1024 steps.
- `4,2,1,2,3` remains the efficient RGRP lane.

## Current Four-Lane Organization

| Lane | Current state | Promotion rule |
|---|---|---|
| `A-control` | fixed seed-45 attention control at 8192: loss `3.9747` | rerun whenever seed, scale, data, or training contract changes |
| `RGRP-quality` | `d768`, `3,2,3,2,2`, loop `256x2`: loss `3.9946` | promote if it closes/beats attention at next scale |
| `RGRP-efficient` | `d704`, `4,2,1,2,3`, loop `320x2`: loss `4.0341` | promote if speed/memory Pareto remains strong |
| `Schedule-search` | new 100M schedules did not beat incumbents | pause unless a targeted hypothesis justifies another bounded scout |
| `Novel` | DeepSeek/Recurrent-Transformer ideas remain separate | do not mix into proof lane until isolated |

## Decision

Do not promote any new schedule from this scout to 8192.

Next main proof move:

```text
Scale the two RGRP incumbents and a matched attention control toward the
250-300M rung, starting with fit/speed scouts before long quality runs.
```

Keep as a side note:

```text
3,1,4,1,3 may be worth revisiting if we specifically want to test
"more exact middle, lighter recurrent bands" at larger width, but it is not
currently better than the quality incumbent.
```

## Deprecated / Diagnostic-Only Context

The 2026-04-29 `fp32` artifacts remain invalid for scorecards. The older
seed-44 freeze is preserved as historical context but no longer represents the
best current read because the seed-45 attention control has been repaired.

