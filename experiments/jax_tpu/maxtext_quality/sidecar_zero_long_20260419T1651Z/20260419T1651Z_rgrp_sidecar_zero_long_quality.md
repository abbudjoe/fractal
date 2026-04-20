# RGRP Layer-1 Zero-Init Sidecar Long Quality Run

Validated on 2026-04-19.

## Purpose

This run tested whether the small 200-step layer `1` zero-init sidecar signal
widens, vanishes, or reverses under a longer matched quality budget.

It compares only two lanes: the unscanned attention baseline and the
zero-initialized layer `1` rotary gated recurrent state update primitive MLP
sidecar.

## Execution Contract

- TPU VM: `fractal-zero-long-202604191651`
- Hardware: `v5litepod-1` spot in `us-west4-a`
- Runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- Runner script: `scripts/run_maxtext_rgrp_sidecar_zero_long_quality_tpu.sh`
- Dataset: `roneneldan/TinyStories`
- Tokenizer: Hugging Face GPT-2
- Shape: `d_model=256`, `mlp_dim=1024`, `layers=4`, `heads=4`,
  `head_dim=64`
- Sequence and batch: `max_target_length=512`, `per_device_batch_size=4`
- Budget: `steps=1000`, `eval_interval=250`, `eval_steps=10`
- Seed: `0`, with `data_shuffle_seed=0` and `init_weights_seed=0`
- RGRP lowering: `lax.scan`, `scan_unroll=3`,
  `block-diagonal-4-masked-dense`, sequence-wide packed projection,
  trig precompute
- Sidecar insertion: layer `1`, bottleneck `64`, side scale `0.1`,
  output init `zero`

## Results

| Lane | Params | Memory After Init | Final Train Loss | Median Tok/s/Device | Mean Tok/s/Device |
|---|---:|---:|---:|---:|---:|
| `attention` | `0.030B` | `0.37 GB` | `3.255` | `343,797` | `342,806` |
| `layer1-zero` | `0.030B` | `0.37 GB` | `3.247` | `301,531` | `302,822` |

| Eval Step | Attention Eval Loss | Layer1-Zero Eval Loss | Delta |
|---:|---:|---:|---:|
| `249` | `3.529` | `3.521` | `-0.008` |
| `499` | `3.253` | `3.225` | `-0.028` |
| `749` | `3.046` | `3.019` | `-0.027` |
| `999` | `2.952` | `2.921` | `-0.031` |

Final eval perplexity:

| Lane | Final Eval PPL |
|---|---:|
| `attention` | `19.151` |
| `layer1-zero` | `18.553` |

## Interpretation

The zero-init sidecar edge widened under the longer run. At 200 steps this
contract was only a tiny seed-level signal. At 1000 steps, the same seed and
matched contract showed a final eval-loss improvement of `-0.031` versus the
attention control.

The sidecar was better at every evaluation checkpoint:

- `-0.008` at step `249`
- `-0.028` at step `499`
- `-0.027` at step `749`
- `-0.031` at step `999`

The speed cost remains real. Median throughput dropped from about `344k` to
about `302k tok/s/device`, a roughly `12.3%` regression under the TPU
`lax.scan` reference lowering. This is not a TPU speed win.

## Decision

Promote `layer1-zero` from "alive but marginal" to "quality-promising under
longer training." The next decision-relevant rungs are:

- replicate the 1000-step result across seeds before making a robustness claim
- test the same contract on a larger or more representative dataset before
  treating it as architecture evidence beyond TinyStories
- keep TPU speed claims out of scope until the quality signal is strong enough
  to justify kernel work

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.
