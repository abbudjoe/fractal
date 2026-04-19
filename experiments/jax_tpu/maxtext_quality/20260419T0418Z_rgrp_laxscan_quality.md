# MaxText RGRP Quality-Only Run: 20260419T0418Z

## Purpose

Run a real-data MaxText training/eval smoke for the rotary gated recurrent
state update primitive using the TPU-friendly `jax.lax.scan` reference
lowering. This verifies the patched MaxText FFN seam and establishes a first
quality-only result before any apples-to-apples baseline claim.

## Environment

- Date: 2026-04-19
- TPU VM: `fractal-rgrp-mt-0419`
- Zone: `us-west4-a`
- Hardware: `v5litepod-1` spot
- Runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched in place with
  `scripts/patch_maxtext_rgrp.py`
- Raw log:
  `experiments/jax_tpu/maxtext_quality/rgrp_maxtext_quality_laxscan_20260419T0418Z.log`

The TPU VM was deleted after the run and after copying the log locally.

## Model And Data

- Candidate: `rotary-gated-recurrent-state-update`
- Integration: MaxText default decoder FFN seam
- Lowering: `jax.lax.scan`
- State transform: `block-diagonal-4-masked-dense`
- Scan unroll: `3`
- Projection mode: sequence-wide packed projection
- Trig mode: sequence-wide precompute
- Residual scale: `1.0`
- Dataset: `roneneldan/TinyStories`
- Train split: `train`
- Eval split: `validation`
- Tokenizer: Hugging Face `gpt2`
- Vocab size: `50257`
- Sequence length: `512`
- Per-device batch size: `4`
- Dtype: `bfloat16`
- Training steps: `200`
- Eval interval: `100`
- Eval steps: `5`
- Checkpointing: disabled

Shape:

| Field | Value |
|---|---:|
| `base_emb_dim` | `256` |
| `base_mlp_dim` | `1024` |
| `base_num_decoder_layers` | `4` |
| `base_num_query_heads` | `4` |
| `base_num_kv_heads` | `4` |
| `head_dim` | `64` |
| Reported parameters | `0.028B` |

## Command

```sh
python3 -m maxtext.trainers.pre_train.train \
  base_output_directory=gs://fractal-maxtext-runs-81f2add4 \
  run_name=rgrp-maxtext-quality-laxscan-20260419T0418Z \
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
  fractal_candidate=rotary-gated-recurrent-state-update \
  fractal_adapter_module=python.jax_tpu.adapters.rotary_gated_recurrent_state_update \
  fractal_rgrp_state_transform=block-diagonal-4-masked-dense \
  fractal_rgrp_scan_unroll=3 \
  fractal_rgrp_projection_mode=sequence \
  fractal_rgrp_trig_mode=precompute \
  fractal_rgrp_residual_scale=1.0 \
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

## Results

| Metric | Value |
|---|---:|
| TPU devices | `1` |
| TPU memory after params init | `0.33 / 15.75 GB` |
| Per-step total TFLOPs | `0.22` |
| Step 0 train loss | `11.355` |
| Eval after step 99 | loss `4.400`, perplexity `81.436` |
| Final eval after step 199 | loss `4.066`, perplexity `58.317` |
| Final train step 199 | loss `4.102`, perplexity `60.444` |
| Regular post-compile median throughput | `105,763 tok/s/device` |
| Regular post-compile mean throughput | `105,761 tok/s/device` |

Regular post-compile throughput was computed from logged train steps between
steps `4` and `197` whose duration was between `0.015s` and `0.025s`, excluding
the first compile-heavy step and the eval/pipeline timing artifacts.

## Interpretation

- The patched MaxText RGRP seam is operational on real data.
- The run stayed numerically stable through the full `200`-step smoke.
- The final validation loss dropped from `4.400` at step `99` to `4.066` at
  step `199`, so the candidate was still learning under this short budget.
- This is not yet an architecture win claim. It needs a matched unchanged
  MaxText attention baseline with the same data, shape, optimizer, and budget.
- The right next step is that matched attention baseline, followed by a longer
  paired run only if the baseline comparison is close enough to justify the
  extra TPU spend.
