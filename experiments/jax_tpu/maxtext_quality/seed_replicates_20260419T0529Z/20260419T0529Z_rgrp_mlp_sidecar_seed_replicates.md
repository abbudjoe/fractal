# RGRP MLP-Sidecar Seed Replication

Validated on 2026-04-19 with:

- TPU VM: `fractal-sidecar-seeds-202604190529`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- dataset: `roneneldan/TinyStories`
- tokenizer: GPT-2 Hugging Face tokenizer
- model shape: `base_emb_dim=256`, `base_mlp_dim=1024`,
  `base_num_decoder_layers=4`, `base_num_query_heads=4`,
  `base_num_kv_heads=4`, `head_dim=64`
- sequence/batch: `max_target_length=512`, `per_device_batch_size=4`
- training budget: `steps=200`, `eval_interval=100`, `eval_steps=5`
- seed policy: `data_shuffle_seed` and `init_weights_seed` varied together

This rung replicates the faithful one-layer sidecar contract from the first
probe. The attention control and RGRP sidecar both run with `scan_layers=false`
so the decoder layer identity is static. The RGRP lane keeps attention and the
standard MLP intact, then adds a bottlenecked rotary gated recurrent state
update side branch beside the MLP in layer `1`.

The existing seed-0 pair from the first sidecar probe is included here for the
matched three-seed table. Seeds `1` and `2` were run in this replication batch.

## Command Shape

Common MaxText settings:

```sh
python3 -m maxtext.trainers.pre_train.train \
  base_output_directory=gs://fractal-maxtext-runs-81f2add4 \
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
  data_shuffle_seed=<seed> \
  init_weights_seed=<seed> \
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

RGRP sidecar extras:

```sh
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
fractal_rgrp_output_init=xavier
```

## Results

| Seed | Lane | Params | Memory After Init | Eval Loss @99 | Final Eval Loss @199 | Final Eval PPL | Final Train Loss | Median Tok/s/Device |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `0` | Attention control, unscanned | `0.030B` | `0.37 GB` | `4.019` | `3.756` | `42.795` | `3.839` | `327,549` |
| `0` | RGRP one-layer MLP sidecar | `0.030B` | `0.37 GB` | `4.026` | `3.754` | `42.696` | `3.839` | `303,925` |
| `1` | Attention control, unscanned | `0.030B` | `0.37 GB` | `4.041` | `3.753` | `42.634` | `4.047` | `333,279` |
| `1` | RGRP one-layer MLP sidecar | `0.030B` | `0.37 GB` | `4.041` | `3.754` | `42.696` | `4.042` | `304,309` |
| `2` | Attention control, unscanned | `0.030B` | `0.37 GB` | `3.984` | `3.744` | `42.249` | `3.851` | `342,303` |
| `2` | RGRP one-layer MLP sidecar | `0.030B` | `0.37 GB` | `3.983` | `3.743` | `42.223` | `3.854` | `302,779` |

## Matched Deltas

Negative loss deltas favor the RGRP sidecar.

| Seed | Delta Eval Loss @99 | Delta Final Eval Loss | Sidecar Tok/s Ratio |
|---:|---:|---:|---:|
| `0` | `+0.007` | `-0.002` | `0.928x` |
| `1` | `+0.000` | `+0.001` | `0.913x` |
| `2` | `-0.001` | `-0.001` | `0.885x` |

## Aggregate

| Lane | Mean Eval Loss @99 | Mean Final Eval Loss | Mean Final Eval PPL | Mean Final Train Loss | Mean Median Tok/s/Device |
|---|---:|---:|---:|---:|---:|
| Attention control, unscanned | `4.0147` | `3.7510` | `42.5593` | `3.9123` | `334,377` |
| RGRP one-layer MLP sidecar | `4.0167` | `3.7503` | `42.5383` | `3.9117` | `303,671` |
| Sidecar minus control | `+0.0020` | `-0.0007` | `-0.0210` | `-0.0007` | `-30,706` |

## Interpretation

- The sparse sidecar signal replicated as a near-tie across three seeds.
- The sidecar won final eval loss on `2 / 3` seeds, but the edge is tiny:
  about `-0.0007` mean final eval loss.
- The sidecar did not improve early eval loss at step `99`; mean step-99 loss
  was `+0.0020` worse.
- Throughput regressed consistently. Mean median throughput was about `9.2%`
  lower for the sidecar.
- Memory and rounded parameter count were unchanged at this reporting
  granularity.

Decision:

- The sidecar hypothesis remains alive, but this is not enough to claim a win.
- A longer run is scientifically reasonable only if framed as quality
  replication, not as an efficiency claim.
- Before spending a long rung, the most useful cheap tuning axis is still the
  sidecar contract itself: side scale, bottleneck width, selected layer, and
  output initialization.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.
