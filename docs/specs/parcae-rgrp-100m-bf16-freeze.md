# Parcae/RGRP 100M bf16 Freeze

Date: 2026-04-29

## Purpose

This note freezes the first clean 100M-class Parcae/RGRP result after the
SageMaker dtype-contract correction.

The important correction is that the 2026-04-29 `fp32` smoke artifacts are
diagnostic-only. The promoted comparison below uses `bf16` on both lanes.

## Promoted Seed-44 Result

Shared setup:

```text
device: NVIDIA L4
dtype: bf16
seq_len: 512
batch_size: 32
steps: 8192
eval_batches: 1
seed: 44
data_seed: 44
token cache: fineweb-cc-main-2024-10-openllama-750m
attention kernel: flex-local
position_encoding_kind: none
attention_position_contract: attention-only
optimizer: adam-fused
head_loss_backend: compiled
ffn_backend: compiled
```

| Lane | Params | Final Loss | Tok/s | Peak CUDA |
|---|---:|---:|---:|---:|
| attention-only d704/h11/12L | 116.53M | 4.2735 | 43,966 | 3.00 GB |
| RGRP 3,2,3,2,2 loop320x2 BD8 | 99.00M | 4.2104 | 46,644 | 2.50 GB |

RGRP delta:

```text
loss: -0.0631
throughput: +6.1%
peak CUDA: -16.7%
parameters: -17.5M
```

## Promoted RGRP Shape

```text
variant: parcae-hourglass-p20-control-looped-attention
wide_d_model: 704
head_count: 11
total_layers: 12
band schedule: 3,2,3,2,2
loop_d_model: 320
loop_head_count: 5
loop_count: 2
backward_steps: 1
control_state_transform: trainable-block-diagonal-8
loop_update_backend: triton-loop-forward
band_block_contract: compiled-direct
band_prepare_backend: compiled
control_position_kind: learned
prelude_norm_kind: rmsnorm
```

## Invalidated Artifacts

The following artifacts are diagnostic-only and must not be used in scorecards:

```text
rgrp704-b32322-kcontract1024-s44-0429a
attn704h11-flex1024-s44-contractfix-0429a
attn704h11-flex8192-s44-control-0429b
```

Reason:

```text
They used dtype=fp32 due to the old SageMaker launcher default.
```

## Next Gate

Before changing architecture, run one matched seed replication:

```text
seed: 45
data_seed: 45
lanes: attention-only, promoted RGRP
same shape/runtime/data contract as seed 44
```

Promotion rule:

```text
If RGRP keeps a meaningful loss edge without losing the speed/memory advantage,
freeze the 100M rung as replicated and move to kernel-contract stabilization
before adding DeepSeek-style compressed memory.
```
