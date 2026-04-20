# Parcae 8-Layer FineWeb-EDU TPU Rung

Validated on 2026-04-19.

## Goal

Compare the faithful 8-layer Parcae proof-ladder shape against an attention
control on a larger, less toy-like text source than WikiText-103.

This rung keeps the parameter target fixed at the current proof-ladder scale:
`~10M` logged parameters. It is not a 30M-50M scaling test.

## Hardware And Runner

- TPU VM: `fractal-parcae-fw-202604192148`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `/Users/joseph/fractal/scripts/patch_maxtext_rgrp.py`
- runner: `/Users/joseph/fractal/scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- local artifacts:
  `/Users/joseph/fractal/experiments/jax_tpu/maxtext_quality/parcae8_fineweb_rung_20260419T2200Z-parcae8-fineweb-rung4096`

The TPU VM was deleted after copying logs. A follow-up `gcloud compute tpus
tpu-vm list --zone=us-west4-a` returned no running TPU VMs.

## Model Contract

- physical decoder layers: `8`
- loop count: `2`
- scaffold: `A0 A1 A2 [A3 A4] [A3 A4] A5 A6 A7`
- `d_model=128`
- attention heads: `4`
- head dim: `32`
- MLP dim: `512`
- context: `256`
- batch: `64`
- tokenizer: `openlm-research/open_llama_3b_v2`
- vocab size: `32000`
- seed: `42`

Lanes:

- `attention`
- `parcae-rgrp-control`
- `parcae-bx`

`parcae-rgrp-control` is the closest TPU/JAX port of the earlier H100
P20-control result: the full-width rotary gated recurrent state update primitive
runs over the prelude stream and controls the loop injection value/gate.

`parcae-bx` is the second control requested for this rung. It keeps the Parcae
looped scaffold but removes the RGRP control primitive.

## Data Contract

MaxText streamed FineWeb-EDU through the Hugging Face parquet loader:

- HF path: `parquet`
- train file:
  `https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/refs%2Fconvert%2Fparquet/CC-MAIN-2013-20/train/0000.parquet`
- eval file:
  `https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/refs%2Fconvert%2Fparquet/CC-MAIN-2013-20/train/0013.parquet`
- train split: `train`
- eval split: `train`

Configured train budget per lane:

```text
4096 steps * 64 batch * 256 context = 67,108,864 token positions
```

Packed/variable-length examples reduced this to `64,873,455` non-padding train
tokens per lane. Four eval passes consumed `4,046,440` non-padding eval tokens
per lane.

## Command

```sh
HF_PATH='parquet' \
HF_NAME='' \
HF_TRAIN_FILES='https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/refs%2Fconvert%2Fparquet/CC-MAIN-2013-20/train/0000.parquet' \
HF_EVAL_FILES='https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/refs%2Fconvert%2Fparquet/CC-MAIN-2013-20/train/0013.parquet' \
HF_EVAL_SPLIT='train' \
TRAIN_SPLIT='train' \
SEEDS='42' \
LANES='attention parcae-rgrp-control parcae-bx' \
STEPS=4096 \
EVAL_INTERVAL=1024 \
EVAL_STEPS=64 \
bash "$HOME/run_maxtext_parcae_proof_ladder_tpu.sh" \
  '20260419T2200Z-parcae8-fineweb-rung4096' \
  "$HOME/fractal_parcae_fineweb_rung_20260419T2200Z-parcae8-fineweb-rung4096"
```

## Results

Throughput below uses median fast training steps and excludes compile/eval
stalls. Duration is wall-clock log span and includes compile, training, and eval.

| Lane | Logged Params | Final Eval Loss | Delta vs Attention | Final Train Loss | Median Fast Tok/s | Mean Fast Tok/s | Train Nonpad | Eval Nonpad | Duration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `attention` | `0.010B` | `4.7420` | `+0.0000` | `4.7830` | `764,786` | `859,349` | `64,873,455` | `4,046,440` | `752s` |
| `parcae-rgrp-control` | `0.010B` | `4.7000` | `-0.0420` | `4.7400` | `573,168` | `672,441` | `64,873,455` | `4,046,440` | `784s` |
| `parcae-bx` | `0.010B` | `4.7110` | `-0.0310` | `4.7490` | `637,510` | `735,626` | `64,873,455` | `4,046,440` | `767s` |

Eval curves:

| Lane | Step 1023 | Step 2047 | Step 3071 | Step 4095 |
|---|---:|---:|---:|---:|
| `attention` | `5.3610` | `4.9770` | `4.8080` | `4.7420` |
| `parcae-rgrp-control` | `5.3520` | `4.9510` | `4.7710` | `4.7000` |
| `parcae-bx` | `5.3590` | `4.9640` | `4.7830` | `4.7110` |

## Interpretation

- Both Parcae lanes beat the attention control on final eval loss at the same
  logged `~10M` parameter target.
- `parcae-rgrp-control` was the best quality lane, improving final eval loss by
  `0.0420` versus attention and by `0.0110` versus `parcae-bx`.
- `parcae-bx` also beat attention, so the looped Parcae scaffold itself remains
  a live control. The RGRP control adds a further quality gain on this run.
- Attention remains materially faster under this TPU/JAX lowering. Compared to
  attention, `parcae-rgrp-control` ran at about `75%` of median fast tok/s and
  `parcae-bx` ran at about `83%`.
- This is stronger than the WikiText sanity rung because it uses a larger
  FineWeb-EDU source and a 4096-step budget, but it is still single-seed and
  small-model evidence.

## Decision

The next proof-ladder step should keep the faithful 8-layer shape and move to
either:

- a multi-seed repeat of `attention` vs `parcae-rgrp-control` on this FineWeb-EDU
  data contract, or
- a larger `30M-50M` rung if compute credits become available.

Do not claim general attention replacement, CUDA/H100 transfer, or frontier-scale
competitiveness from this rung alone.
