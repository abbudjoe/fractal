# RGRP Sidecar Matched-Control Rung

Validated on 2026-04-19.

## Purpose

This rung tests whether the layer `1`, zero-init sidecar gain is specific to
the rotary gated recurrent state update primitive, or whether a similarly small
FFN-side branch produces the same effect.

The run uses one same-day patched MaxText checkout and changes only
`fractal_rgrp_sidecar_type` after the attention baseline. This is intentionally
a narrow control-plane test, not a new architecture search.

## Execution Contract

- TPU VM: `fractal-sidecar-controls-202604191928`
- Hardware: `v5litepod-1` spot in `us-west4-a`
- Runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- Runner script: `scripts/run_maxtext_sidecar_control_rung_tpu.sh`
- Dataset: `roneneldan/TinyStories`
- Tokenizer: Hugging Face GPT-2
- Shape: `d_model=256`, `mlp_dim=1024`, `layers=4`, `heads=4`,
  `head_dim=64`
- Sequence and batch: `max_target_length=512`, `per_device_batch_size=4`
- Budget: `steps=1000`, `eval_interval=250`, `eval_steps=10`
- Seeds: `0`, `1`, `2`, with `data_shuffle_seed` and `init_weights_seed`
  matched per seed
- Sidecar insertion: layer `1`, bottleneck `64`, side scale `0.1`,
  output init `zero`
- RGRP lowering: `lax.scan`, `scan_unroll=3`,
  `block-diagonal-4-masked-dense`, sequence-wide packed projection,
  trig precompute
- Controls:
  - `tiny-mlp`: down projection, bottleneck MLP, zero-init up projection
  - `tiny-glu`: gated bottleneck projection, zero-init up projection
  - `binary-tree`: depth-2 differentiable tree over 4 learned scalar slots plus
    constants, zero-init up projection

A one-seed, five-step smoke was run first across all four sidecar operators;
all lanes compiled and trained.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## Per-Seed Final Results

| Lane | Seed | Final Eval | Delta vs Attention | Final Train | Median tok/s |
|---|---:|---:|---:|---:|---:|
| `attention` | `0` | `2.952` | `+0.000` | `3.255` | `350,389` |
| `rgrp` | `0` | `2.921` | `-0.031` | `3.247` | `297,601` |
| `tiny-mlp` | `0` | `2.942` | `-0.010` | `3.236` | `343,812` |
| `tiny-glu` | `0` | `2.948` | `-0.004` | `3.261` | `347,085` |
| `binary-tree` | `0` | `2.947` | `-0.005` | `3.231` | `352,238` |
| `attention` | `1` | `2.847` | `+0.000` | `3.383` | `361,426` |
| `rgrp` | `1` | `2.832` | `-0.015` | `3.360` | `308,056` |
| `tiny-mlp` | `1` | `2.841` | `-0.006` | `3.368` | `349,938` |
| `tiny-glu` | `1` | `2.849` | `+0.002` | `3.364` | `360,343` |
| `binary-tree` | `1` | `2.843` | `-0.004` | `3.388` | `344,321` |
| `attention` | `2` | `2.930` | `+0.000` | `3.155` | `368,193` |
| `rgrp` | `2` | `2.910` | `-0.020` | `3.148` | `300,713` |
| `tiny-mlp` | `2` | `2.921` | `-0.009` | `3.188` | `372,977` |
| `tiny-glu` | `2` | `2.925` | `-0.005` | `3.169` | `350,200` |
| `binary-tree` | `2` | `2.927` | `-0.003` | `3.169` | `376,480` |

## Aggregate Results

| Lane | Mean Final Eval | Std | Mean Delta vs Attention | Wins vs Attention | Mean Final Train | Mean Median tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `attention` | `2.9097` | `0.0452` | `+0.0000` | `0/3` | `3.2643` | `360,003` |
| `rgrp` | `2.8877` | `0.0396` | `-0.0220` | `3/3` | `3.2517` | `302,123` |
| `tiny-mlp` | `2.9013` | `0.0435` | `-0.0083` | `3/3` | `3.2640` | `355,576` |
| `tiny-glu` | `2.9073` | `0.0423` | `-0.0023` | `2/3` | `3.2647` | `352,543` |
| `binary-tree` | `2.9057` | `0.0451` | `-0.0040` | `3/3` | `3.2627` | `357,680` |

## Pairwise Versus RGRP

| Control | Per-Seed Delta vs RGRP | Mean Delta vs RGRP | Wins vs RGRP |
|---|---:|---:|---:|
| `tiny-mlp` | `[+0.021, +0.009, +0.011]` | `+0.0137` | `0/3` |
| `tiny-glu` | `[+0.027, +0.017, +0.015]` | `+0.0197` | `0/3` |
| `binary-tree` | `[+0.026, +0.011, +0.017]` | `+0.0180` | `0/3` |

## Interpretation

This strengthens the RGRP-specific case. Small sidecars do help a little:
`tiny-mlp`, `tiny-glu`, and `binary-tree` all land close to the attention
baseline, and `tiny-mlp` wins all three seeds by a small amount. But none of the
matched controls match the RGRP sidecar. RGRP wins every seed against every
control, with a mean advantage of `0.0137` to `0.0197` eval loss versus the
controls.

The cost is speed. RGRP is about `16.1%` slower than attention on mean median
tok/s/device in this same-checkout run, while the non-recurrent controls stay
near attention speed. That is the clearest remaining tradeoff: the recurrent
state carry appears to buy quality, but the TPU `lax.scan` reference lowering
pays for it.

## Decision

Promote the layer `1` zero-init RGRP sidecar as a real, control-tested
TinyStories quality signal. The next proof rung should use a larger or less
saturated dataset before spending engineering time on TPU/Pallas speed work.

Keep `tiny-mlp` as the matched extra-capacity control in future runs. It is the
strongest non-recurrent control and the cheapest way to check whether a future
RGRP win is still recurrent-state-specific.
