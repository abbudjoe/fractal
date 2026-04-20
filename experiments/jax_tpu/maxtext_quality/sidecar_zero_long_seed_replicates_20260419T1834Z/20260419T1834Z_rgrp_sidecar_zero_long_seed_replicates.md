# RGRP Layer-1 Zero-Init Sidecar 1000-Step Seed Replication

Validated on 2026-04-19.

## Purpose

This run replicated the 1000-step layer `1` zero-init RGRP MLP sidecar quality
signal across seeds. Seed `0` came from the first 1000-step long-quality run;
seeds `1` and `2` were added in this batch under the same MaxText contract.

The goal was to determine whether the seed `0` improvement was a one-off or a
robust short-rung quality signal.

## Execution Contract

- TPU VM for seeds `1` and `2`: `fractal-zero-long-seeds-202604191834`
- Hardware: `v5litepod-1` spot in `us-west4-a`
- Runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- Runner script: `scripts/run_maxtext_rgrp_sidecar_zero_long_seed_replication_tpu.sh`
- Dataset: `roneneldan/TinyStories`
- Tokenizer: Hugging Face GPT-2
- Shape: `d_model=256`, `mlp_dim=1024`, `layers=4`, `heads=4`,
  `head_dim=64`
- Sequence and batch: `max_target_length=512`, `per_device_batch_size=4`
- Budget: `steps=1000`, `eval_interval=250`, `eval_steps=10`
- Seeds: `0`, `1`, `2`, with `data_shuffle_seed` and `init_weights_seed`
  matched per seed
- RGRP lowering: `lax.scan`, `scan_unroll=3`,
  `block-diagonal-4-masked-dense`, sequence-wide packed projection,
  trig precompute
- Sidecar insertion: layer `1`, bottleneck `64`, side scale `0.1`,
  output init `zero`

Seed `0` raw logs are in:
`experiments/jax_tpu/maxtext_quality/sidecar_zero_long_20260419T1651Z/`.

Seeds `1` and `2` raw logs are in this artifact directory.

## Per-Seed Final Results

| Seed | Attention Final Eval | Layer1-Zero Final Eval | Delta | Attention Train Loss | Layer1-Zero Train Loss | Attention Median Tok/s | Layer1-Zero Median Tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | `2.952` | `2.921` | `-0.031` | `3.255` | `3.247` | `343,797` | `301,531` |
| `1` | `2.847` | `2.832` | `-0.015` | `3.383` | `3.360` | `334,258` | `301,531` |
| `2` | `2.930` | `2.910` | `-0.020` | `3.155` | `3.148` | `340,936` | `301,531` |

## Evaluation Ladder

| Seed | Eval Step | Attention Eval Loss | Layer1-Zero Eval Loss | Delta |
|---:|---:|---:|---:|---:|
| `0` | `249` | `3.529` | `3.521` | `-0.008` |
| `0` | `499` | `3.253` | `3.225` | `-0.028` |
| `0` | `749` | `3.046` | `3.019` | `-0.027` |
| `0` | `999` | `2.952` | `2.921` | `-0.031` |
| `1` | `249` | `3.504` | `3.505` | `+0.001` |
| `1` | `499` | `3.154` | `3.136` | `-0.018` |
| `1` | `749` | `2.915` | `2.904` | `-0.011` |
| `1` | `999` | `2.847` | `2.832` | `-0.015` |
| `2` | `249` | `3.537` | `3.525` | `-0.012` |
| `2` | `499` | `3.198` | `3.174` | `-0.024` |
| `2` | `749` | `3.014` | `2.995` | `-0.019` |
| `2` | `999` | `2.930` | `2.910` | `-0.020` |

## Aggregate Results

| Lane | Mean Final Eval Loss | Final Eval Std Dev | Mean Final Eval PPL | Mean Final Train Loss | Mean Median Tok/s/Device |
|---|---:|---:|---:|---:|---:|
| `attention` | `2.9097` | `0.0452` | `18.3747` | `3.2643` | `339,664` |
| `layer1-zero` | `2.8877` | `0.0396` | `17.9647` | `3.2517` | `301,531` |

Paired final-eval delta:

| Comparison | Mean Delta | Per-Seed Deltas | Wins |
|---|---:|---|---:|
| `layer1-zero - attention` | `-0.0220` | `[-0.031, -0.015, -0.020]` | `3 / 3` |

Mean median throughput delta:

| Comparison | Mean Delta |
|---|---:|
| `layer1-zero / attention - 1` | `-11.2%` |

## Interpretation

The 1000-step sidecar quality signal replicated across all three seeds. The
sidecar improved final eval loss by `-0.015` to `-0.031`, with a mean
improvement of `-0.0220`.

This upgrades the TPU finding from "promising on seed 0" to "repeatable on
TinyStories under this 30M-parameter MaxText rung." The effect is still not a
general architecture claim: it is a specific sparse FFN-side sidecar result on a
small TinyStories setup.

The throughput cost remains material. The sidecar averaged about `301k`
median tok/s/device versus attention's `340k`, an average `11.2%` slowdown under
the TPU `lax.scan` reference lowering.

## Decision

Promote `layer1-zero` as the current best faithful RGRP sidecar contract for
quality experiments.

Next gates:

- Run the same contract on a larger or more representative dataset before
  making broader architecture claims.
- Keep CUDA/H100 and TPU speed claims separate; this TPU reference lowering is
  not a speed win.
- Do not invest in TPU kernel work solely for speed until the larger-data
  quality signal survives.

The TPU VM for seeds `1` and `2` was deleted after copying logs; `gcloud compute
tpus tpu-vm list` reported no running TPU VMs.
