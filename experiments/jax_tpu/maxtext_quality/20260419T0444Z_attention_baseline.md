# MaxText Attention Baseline: 20260419T0444Z

## Purpose

Run the matched unchanged MaxText transformer baseline for the first RGRP
quality-only run. This is the control needed before making any claim about the
rotary gated recurrent state update primitive in MaxText.

## Environment

- Date: 2026-04-19
- TPU VM: `fractal-attn-mt-0419`
- Zone: `us-west4-a`
- Hardware: `v5litepod-1` spot
- Runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, unpatched
- Raw log:
  `experiments/jax_tpu/maxtext_quality/attention_maxtext_baseline_20260419T0444Z.log`

The TPU VM was deleted after the run and after copying the log locally.

## Model And Data

- Candidate: unchanged MaxText transformer baseline
- Integration: `decoder_block=default`
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
| Reported parameters | `0.030B` |

## Command

```sh
python3 -m maxtext.trainers.pre_train.train \
  base_output_directory=gs://fractal-maxtext-runs-81f2add4 \
  run_name=attention-maxtext-baseline-20260419T0444Z \
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
| TPU memory after params init | `0.35 / 15.75 GB` |
| Per-step total TFLOPs | `0.22` |
| Step 0 train loss | `11.366` |
| Eval after step 99 | loss `4.030`, perplexity `56.265` |
| Final eval after step 199 | loss `3.758`, perplexity `42.874` |
| Final train step 199 | loss `3.817`, perplexity `45.474` |
| Regular post-compile median throughput | `375,987 tok/s/device` |
| Regular post-compile mean throughput | `372,388 tok/s/device` |

Regular post-compile throughput was computed from logged train steps between
steps `4` and `197` whose duration was between `0.004s` and `0.008s`, excluding
the first compile-heavy step and the eval/pipeline timing artifacts.

## Interpretation

- The unchanged MaxText transformer baseline is stable and substantially ahead
  of the RGRP FFN-seam run on this `200`-step TinyStories rung.
- The baseline has slightly more parameters, `0.030B` vs RGRP's `0.028B`, but
  the difference is too small to explain the full quality and speed gap.
- This result says the current RGRP `lax.scan` MaxText seam is not competitive
  with the default transformer block at this small quality-smoke scale.
- The next useful RGRP work should not be another identical quality-only run.
  Either improve the integration/training contract or test a longer/larger rung
  only when there is a concrete hypothesis that targets the observed gap.
