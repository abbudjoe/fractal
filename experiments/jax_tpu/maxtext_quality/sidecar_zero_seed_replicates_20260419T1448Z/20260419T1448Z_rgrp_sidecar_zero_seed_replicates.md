# RGRP Layer-1 Zero-Init MLP-Sidecar Seed Replication

Validated on 2026-04-19.

## Purpose

This run replicated the best single-seed ablation contract from the layer and
knob sweep: a layer `1` rotary gated recurrent state update primitive MLP
sidecar with zero-initialized output projection. It compares that lane against
the matched unscanned attention control and the prior layer `1` Xavier-init
sidecar.

The goal is to answer whether the zero-init sidecar result was a seed accident
or a repeatable, if small, quality signal.

## Execution Contract

- TPU VM: `fractal-zero-seeds-202604191448`
- Hardware: `v5litepod-1` spot in `us-west4-a`
- Runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- Runner script: `scripts/run_maxtext_rgrp_sidecar_zero_seed_replication_tpu.sh`
- Dataset: `roneneldan/TinyStories`
- Tokenizer: Hugging Face GPT-2
- Shape: `d_model=256`, `mlp_dim=1024`, `layers=4`, `heads=4`,
  `head_dim=64`
- Sequence and batch: `max_target_length=512`, `per_device_batch_size=4`
- Budget: `steps=200`, `eval_interval=100`, `eval_steps=5`
- Seeds: `0`, `1`, `2`, with `data_shuffle_seed` and `init_weights_seed`
  matched per seed
- RGRP lowering: `lax.scan`, `scan_unroll=3`,
  `block-diagonal-4-masked-dense`, sequence-wide packed projection,
  trig precompute
- Sidecar insertion: layer `1`, bottleneck `64`, side scale `0.1`

## Per-Seed Results

| Seed | Lane | Eval Loss @99 | Final Eval Loss @199 | Delta vs Attention | Final Train Loss | Median Tok/s/Device |
|---:|---|---:|---:|---:|---:|---:|
| `0` | `attention` | `4.019` | `3.756` | `0.000` | `3.839` | `364,932` |
| `0` | `layer1-default` | `4.026` | `3.754` | `-0.002` | `3.839` | `302,780` |
| `0` | `layer1-zero` | `4.019` | `3.751` | `-0.005` | `3.839` | `302,824` |
| `1` | `attention` | `4.041` | `3.753` | `0.000` | `4.047` | `363,121` |
| `1` | `layer1-default` | `4.041` | `3.754` | `+0.001` | `4.042` | `302,601` |
| `1` | `layer1-zero` | `4.040` | `3.750` | `-0.003` | `4.039` | `303,587` |
| `2` | `attention` | `3.984` | `3.744` | `0.000` | `3.851` | `352,556` |
| `2` | `layer1-default` | `3.983` | `3.743` | `-0.001` | `3.854` | `302,601` |
| `2` | `layer1-zero` | `3.984` | `3.747` | `+0.003` | `3.856` | `302,824` |

All lanes reported `0.030B` parameters and `0.37 GB` after parameter
initialization at MaxText's printed precision.

## Aggregate Results

| Lane | Mean Eval Loss @99 | Mean Final Eval Loss | Eval Loss Std Dev | Mean Final Train Loss | Mean Median Tok/s/Device | Wins vs Attention |
|---|---:|---:|---:|---:|---:|---:|
| `attention` | `4.0147` | `3.7510` | `0.0051` | `3.9123` | `360,203` | - |
| `layer1-default` | `4.0167` | `3.7503` | `0.0052` | `3.9117` | `302,660` | `2 / 3` |
| `layer1-zero` | `4.0143` | `3.7493` | `0.0017` | `3.9113` | `303,079` | `2 / 3` |

Paired final-eval deltas:

| Comparison | Mean Delta | Per-Seed Deltas |
|---|---:|---|
| `layer1-default - attention` | `-0.0007` | `[-0.002, +0.001, -0.001]` |
| `layer1-zero - attention` | `-0.0017` | `[-0.005, -0.003, +0.003]` |
| `layer1-zero - layer1-default` | `-0.0010` | `[-0.003, -0.004, +0.004]` |

## Interpretation

The zero-init sidecar did replicate directionally. It improved mean final eval
loss versus attention by about `0.0017` and versus the default sidecar by about
`0.0010`, winning `2 / 3` seeds in both paired comparisons.

This is still a tiny effect. It is not a decisive architecture win, and it does
not offset the TPU reference-lowering throughput cost. The attention control
averaged about `360k tok/s/device`; both sidecar lanes averaged about
`303k tok/s/device`, a roughly `16%` throughput regression in this run.

The most useful finding is contract-level: if we keep the sparse RGRP sidecar,
zero-initialized output projection is the better default than Xavier-init for
this MaxText seam. It makes the recurrent branch start as a dormant residual
path and lets training decide whether to use it.

## Decision

Promote `layer1-zero` as the current faithful TPU reference sidecar contract,
but do not claim a strong win. The next rung should only be run if we need a
quality-oriented TPU datapoint:

- longer run on the same TinyStories contract to see whether the small edge
  widens or vanishes
- larger-data run if the goal is publication-grade evidence
- no further TPU speed work on this `lax.scan` sidecar until the quality signal
  becomes materially larger

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.
