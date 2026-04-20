# MaxText RGRP MLP-Sidecar Faithful Probe

## Purpose

Run a more faithful TPU representation of the earlier positive signal for the
rotary gated recurrent state update primitive.

The previous MaxText quality run replaced every FFN seam with RGRP. That was a
harsh ablation, not the sparse/mid-layer hybrid shape that had looked alive in
the Python/CUDA experiments. This probe keeps the normal MaxText attention and
MLP backbone intact, then adds one bottlenecked RGRP side branch beside the MLP
in a selected middle layer.

## Contract

Both completed runs used:

- Date: 2026-04-19
- TPU VM: `fractal-sidecar-202604190512`
- Zone: `us-west4-a`
- Hardware: one spot `v5litepod-1`
- Runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`
- Dataset: `roneneldan/TinyStories`
- Tokenizer: Hugging Face `gpt2`
- Sequence length: `512`
- Per-device batch size: `4`
- Training steps: `200`
- Eval interval: `100`
- Eval steps: `5`
- Dtype: `bfloat16`
- Shape: `base_emb_dim=256`, `base_mlp_dim=1024`,
  `base_num_decoder_layers=4`, `base_num_query_heads=4`,
  `base_num_kv_heads=4`, `head_dim=64`
- Checkpointing disabled
- `scan_layers=false`

The unscanned control is necessary because selected-layer sidecar placement
needs an explicit layer identity.

Sidecar-specific knobs:

- `fractal_candidate=rotary-gated-recurrent-state-update`
- `fractal_rgrp_integration_mode=mlp-sidecar`
- `fractal_rgrp_layers=1`
- `fractal_rgrp_bottleneck_dim=64`
- `fractal_rgrp_state_transform=block-diagonal-4-masked-dense`
- `fractal_rgrp_scan_unroll=3`
- `fractal_rgrp_projection_mode=sequence`
- `fractal_rgrp_trig_mode=precompute`
- `fractal_rgrp_side_scale=0.1`
- `fractal_rgrp_output_init=xavier`

Raw logs:

- `experiments/jax_tpu/maxtext_quality/attention_unscanned_maxtext_20260419T0512Z.log`
- `experiments/jax_tpu/maxtext_quality/rgrp_mlp_sidecar_maxtext_20260419T0522Z.log`

The TPU VM was deleted after copying logs locally.

## Results

| Lane | Params | Memory After Init | Eval Loss @99 | Final Eval Loss @199 | Final Eval PPL | Final Train Loss | Median Tok/s/Device |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attention control, unscanned | `0.030B` | `0.37 GB` | `4.019` | `3.756` | `42.795` | `3.839` | `327,549` |
| RGRP one-layer MLP sidecar | `0.030B` | `0.37 GB` | `4.026` | `3.754` | `42.696` | `3.839` | `303,925` |

Delta, sidecar relative to unscanned control:

| Metric | Delta |
|---|---:|
| Final eval loss | `-0.002` better |
| Final eval perplexity | `-0.099` better |
| Final train loss | approximately tied |
| Median throughput | `0.93x` control |

For context, the earlier scanned attention baseline was faster and similar
quality: final eval loss `3.758`, final eval PPL `42.874`, median throughput
`375,987 tok/s/device`.

## Interpretation

This is the first MaxText run that matches the earlier positive-signal shape in
spirit: a transformer backbone with a sparse recurrent sidecar, not a full FFN
replacement. It is not a decisive win, but it is qualitatively different from
the all-layer replacement failure:

- The sidecar is stable and trainable.
- The quality is essentially tied with the matched unscanned attention control,
  with a tiny final-eval edge.
- The throughput cost is modest for this reference-lowering lane: about `7%`
  versus the matched unscanned control, not the `3.55x` slowdown seen in the
  full FFN replacement.
- The result is too small and too short to claim superiority. It is enough to
  say the faithful sparse sidecar hypothesis remains alive.

## Contract Fixes Found

Two patcher issues were fixed before the successful sidecar run:

- `fractal_rgrp_layers=1` is parsed by MaxText/Pydantic as an integer, so the
  patched config field now accepts `str | int`.
- Layer selection cannot depend on a traced `layer_idx` inside the Linen module.
  The sidecar enable flag is now computed during static module construction and
  passed as `enable_fractal_rgrp_sidecar`.

These are control-plane fixes, not model-quality tuning.

## Commands

Control:

```sh
python3 -m maxtext.trainers.pre_train.train \
  base_output_directory=gs://fractal-maxtext-runs-81f2add4 \
  run_name=attention-unscanned-maxtext-20260419T0512Z \
  dataset_type=hf \
  hf_path=roneneldan/TinyStories \
  hf_eval_split=validation \
  tokenizer_type=huggingface \
  tokenizer_path=gpt2 \
  train_split=train \
  steps=200 \
  log_period=10 \
  eval_interval=100 \
  eval_steps=5 \
  enable_checkpointing=false \
  save_checkpoint_on_completion=false \
  log_config=false \
  decoder_block=default \
  scan_layers=false \
  max_target_length=512 \
  vocab_size=50257 \
  base_emb_dim=256 \
  base_mlp_dim=1024 \
  base_num_decoder_layers=4 \
  base_num_query_heads=4 \
  base_num_kv_heads=4 \
  head_dim=64 \
  per_device_batch_size=4 \
  learning_rate=0.001 \
  dtype=bfloat16
```

Sidecar:

```sh
python3 -m maxtext.trainers.pre_train.train \
  base_output_directory=gs://fractal-maxtext-runs-81f2add4 \
  run_name=rgrp-mlp-sidecar-maxtext-20260419T0522Z \
  dataset_type=hf \
  hf_path=roneneldan/TinyStories \
  hf_eval_split=validation \
  tokenizer_type=huggingface \
  tokenizer_path=gpt2 \
  train_split=train \
  steps=200 \
  log_period=10 \
  eval_interval=100 \
  eval_steps=5 \
  enable_checkpointing=false \
  save_checkpoint_on_completion=false \
  log_config=false \
  decoder_block=default \
  scan_layers=false \
  fractal_candidate=rotary-gated-recurrent-state-update \
  fractal_adapter_module=python.jax_tpu.adapters.rotary_gated_recurrent_state_update \
  fractal_rgrp_integration_mode=mlp-sidecar \
  fractal_rgrp_layers=1 \
  fractal_rgrp_bottleneck_dim=64 \
  fractal_rgrp_state_transform=block-diagonal-4-masked-dense \
  fractal_rgrp_scan_unroll=3 \
  fractal_rgrp_projection_mode=sequence \
  fractal_rgrp_trig_mode=precompute \
  fractal_rgrp_side_scale=0.1 \
  fractal_rgrp_output_init=xavier \
  max_target_length=512 \
  vocab_size=50257 \
  base_emb_dim=256 \
  base_mlp_dim=1024 \
  base_num_decoder_layers=4 \
  base_num_query_heads=4 \
  base_num_kv_heads=4 \
  head_dim=64 \
  per_device_batch_size=4 \
  learning_rate=0.001 \
  dtype=bfloat16
```

## Recommendation

Do not claim a win from this single run. Do promote the sparse sidecar as the
correct TPU contract for further tests. The next sensible rung is either:

- repeat this sidecar/control pair across seeds, or
- run a longer same-contract quality read only if budget is available.

The full FFN replacement lane should remain demoted.
