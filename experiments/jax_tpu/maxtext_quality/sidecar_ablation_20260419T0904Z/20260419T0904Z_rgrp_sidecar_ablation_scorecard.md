# RGRP MLP-Sidecar Location and Knob Ablation

Validated on 2026-04-19.

## Purpose

This run tested whether the faithful one-layer rotary gated recurrent state
update primitive sidecar depends on its layer location or small integration
knobs. The prior seed replication showed a near-tie against attention, so this
ablation intentionally stayed bounded: one seed, one 200-step MaxText contract,
and one changed axis at a time.

The lane keeps the attention and MLP backbone intact. It disables MaxText layer
scanning so the patch can see static layer identity, then attaches a bottlenecked
RGRP MLP sidecar to selected layer slots.

## Execution Contract

- TPU VM: `fractal-sidecar-ablate-202604190904`
- Hardware: `v5litepod-1` spot in `us-west4-a`
- Runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- Runner script: `scripts/run_maxtext_rgrp_sidecar_ablation_tpu.sh`
- Dataset: `roneneldan/TinyStories`
- Tokenizer: Hugging Face GPT-2
- Shape: `d_model=256`, `mlp_dim=1024`, `layers=4`, `heads=4`,
  `head_dim=64`
- Sequence and batch: `max_target_length=512`, `per_device_batch_size=4`
- Budget: `steps=200`, `eval_interval=100`, `eval_steps=5`
- Seeds: `data_shuffle_seed=0`, `init_weights_seed=0`
- RGRP lowering: `lax.scan`, `scan_unroll=3`,
  `block-diagonal-4-masked-dense`, sequence-wide packed projection,
  trig precompute

An earlier inline remote-shell attempt failed before producing a usable
ablation because MaxText overrides were lost through nested SSH quoting. That
failure is not counted. The durable fix is the checked-in TPU-side runner script,
which keeps overrides in real bash arrays inside the patched MaxText checkout.

## Results

| Lane | Sidecar Layer | Side Scale | Bottleneck | Output Init | Eval Loss @99 | Final Eval Loss @199 | Delta vs Attention | Final Train Loss | Median Tok/s/Device |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `layer1-zero` | `1` | `0.10` | `64` | `zero` | `4.019` | `3.751` | `-0.005` | `3.839` | `304,467` |
| `layer1-default` | `1` | `0.10` | `64` | `xavier` | `4.026` | `3.754` | `-0.002` | `3.839` | `303,925` |
| `layer1-bn128` | `1` | `0.10` | `128` | `xavier` | `4.022` | `3.755` | `-0.001` | `3.844` | `278,564` |
| `attention-control` | - | - | - | - | `4.019` | `3.756` | `0.000` | `3.839` | `331,044` |
| `layer1-scale020` | `1` | `0.20` | `64` | `xavier` | `4.029` | `3.756` | `0.000` | `3.843` | `304,649` |
| `layer0-default` | `0` | `0.10` | `64` | `xavier` | `4.023` | `3.757` | `+0.001` | `3.837` | `303,385` |
| `layer1-bn32` | `1` | `0.10` | `32` | `xavier` | `4.022` | `3.758` | `+0.002` | `3.847` | `339,101` |
| `layer3-default` | `3` | `0.10` | `64` | `xavier` | `4.021` | `3.758` | `+0.002` | `3.833` | `304,174` |
| `layer1-scale005` | `1` | `0.05` | `64` | `xavier` | `4.025` | `3.759` | `+0.003` | `3.844` | `303,318` |
| `layer2-default` | `2` | `0.10` | `64` | `xavier` | `4.025` | `3.759` | `+0.003` | `3.845` | `303,970` |

All lanes reported `0.030B` parameters and `0.37 GB` after parameter
initialization at MaxText's printed precision.

## Interpretation

The sidecar location does matter, but the effect is small at this rung. Among
the Xavier-init default sidecars, layer `1` remains the best slot. Layers `0`,
`2`, and `3` all lost to the attention control on final eval loss.

The best single-seed result was not a larger sidecar. It was the same layer `1`
sidecar with zero-initialized output projection. That is a useful contract
signal: the recurrent sidecar may need to start as a dormant residual path and
earn influence during training, rather than perturbing the MLP stream at step
zero.

The bottleneck and scale axes did not produce a clean monotonic improvement.
`bn32` recovered speed but lost quality. `bn128` was slightly better than the
attention control on loss but slower than the default sidecar. `side_scale=0.20`
tied the control, while `side_scale=0.05` was worse.

## Decision

Do not promote a longer run yet. The next cheapest decision-relevant move is a
seed replication of `layer1-zero` against the same unscanned attention control
and the prior `layer1-default` sidecar. If zero-init keeps the edge across seeds,
then it becomes the first candidate worth a longer MaxText run.

If zero-init does not replicate, the sparse sidecar should stay alive only as a
scientific note: it has repeatedly tied attention closely, but it has not earned
more TPU budget as a quality or efficiency winner.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.
