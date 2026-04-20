# Parcae Width-Scale 16k Seed Replication

Date: 2026-04-20

This run replicated the `d_model=192` Parcae width-scale rung on seeds `7` and
`123`, after seed `42` showed RGRP-control beating both attention and the
Parcae B(x) scaffold control.

The question was whether the width-scale RGRP-control result was a seed-42
accident or a stable signal.

## Environment

- TPU VM: `fractal-parcae-wrep-20260420T143015`
- zone: `us-west4-a`
- hardware: `v5litepod-1`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout on TPU: `~/maxtext-parcae-wrep`
- local artifact root:
  `experiments/jax_tpu/maxtext_quality/parcae_width_rep_20260420T143015Z`

The TPU VM was deleted after copying logs. A final `gcloud compute tpus tpu-vm
list --zone=us-west4-a` returned no active TPU VMs.

Note: the detached shell did not inherit the intended `RUN_DIR`, so the per-lane
logs landed in the runner default `fractal_parcae_proof_ladder_20260420T143015Z`
directory. The nohup log and all six lane logs were copied locally.

## Shared Shape And Data

- physical decoder layers: `8`
- Parcae loop count: `2`
- loop policy: `fixed`
- recurrence active depth: `2`
- backward active depth: `1`
- middle loop band: `layers 3..4`
- execution scaffold: `A0 A1 A2 [A3 A4] [A3 A4] A5 A6 A7`
- `d_model=192`
- `heads=6`
- `head_dim=32`
- `mlp_dim=768`
- MaxText reported parameter size: `0.017B`
- context length: `256`
- batch size: `64`
- vocab size: `32000`
- tokenizer: `openlm-research/open_llama_3b_v2`
- dataset: `HuggingFaceFW/fineweb-edu`, config dump `CC-MAIN-2013-20`
- train files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/train/*.parquet`
- eval files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/eval/*.parquet`
- new seeds: `7`, `123`
- prior comparison seed: `42`
- train steps: `16,384`
- eval interval: `2,048`
- eval steps: `64`
- configured train token positions per lane:
  `16384 * 64 * 256 = 268,435,456`

All Parcae lanes used:

- `PARCAE_DISCRETIZATION=stable-exp`
- `MU_REC=2`
- `MU_BWD=1`

## Command

```sh
STAMP='20260420T143015Z'
HF_PATH='parquet' HF_NAME='' \
HF_TRAIN_FILES='gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/train/*.parquet' \
HF_EVAL_FILES='gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/eval/*.parquet' \
HF_EVAL_SPLIT='train' TRAIN_SPLIT='train' \
SEEDS='7 123' LANES='attention parcae-bx parcae-rgrp-control' \
STEPS=16384 EVAL_INTERVAL=2048 EVAL_STEPS=64 \
PARCAE_DISCRETIZATION='stable-exp' MU_REC=2 MU_BWD=1 \
D_MODEL=192 HEADS=6 HEAD_DIM=32 MLP_DIM=768 LAYERS=8 BATCH_SIZE=64 SEQ_LEN=256 \
bash "$HOME/run_maxtext_parcae_proof_ladder_tpu.sh" \
  "$STAMP" \
  "$HOME/fractal_parcae_proof_ladder_$STAMP"
```

## New Seed Results

`Median steady tok/s` is the median `Tokens/s/device` for completed training
steps with `step >= 100` and `seconds < 0.5`.

| Seed | Lane | Final Eval Loss | Delta vs Attention | Delta vs BX | PPL | Final Train Loss | Median Steady Tok/s | Non-Padding Train Tokens | Log Duration |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `7` | `attention` | `3.995` | `+0.000` | `+0.054` | `54.316` | `3.888` | `474,967` | `259,416,359` | `2,594s` |
| `7` | `parcae-bx` | `3.941` | `-0.054` | `+0.000` | `51.445` | `3.830` | `386,972` | `259,416,359` | `2,679s` |
| `7` | `parcae-rgrp-control` | `3.907` | `-0.088` | `-0.034` | `49.740` | `3.804` | `354,878` | `259,416,359` | `2,712s` |
| `123` | `attention` | `3.987` | `+0.000` | `+0.054` | `53.893` | `3.846` | `474,961` | `259,428,995` | `2,569s` |
| `123` | `parcae-bx` | `3.933` | `-0.054` | `+0.000` | `51.050` | `3.794` | `387,027` | `259,428,995` | `2,630s` |
| `123` | `parcae-rgrp-control` | `3.904` | `-0.083` | `-0.029` | `49.610` | `3.764` | `354,878` | `259,428,995` | `2,670s` |

## Three-Seed Aggregate

This table includes the prior seed-42 width-scale run from
`parcae_width_20260420T115012Z`.

| Lane | Seed 7 | Seed 42 | Seed 123 | Mean Final Eval Loss | Loss Std | Mean PPL | Mean Median Steady Tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention` | `3.995` | `3.978` | `3.987` | `3.987` | `0.0069` | `53.871` | `474,928` |
| `parcae-bx` | `3.941` | `3.946` | `3.933` | `3.940` | `0.0054` | `51.404` | `387,009` |
| `parcae-rgrp-control` | `3.907` | `3.905` | `3.904` | `3.905` | `0.0012` | `49.675` | `354,898` |

Mean deltas:

- `parcae-bx` vs attention: `-0.047` eval loss
- `parcae-rgrp-control` vs attention: `-0.082` eval loss
- `parcae-rgrp-control` vs `parcae-bx`: `-0.035` eval loss

Per-seed deltas:

| Seed | BX vs Attention | RGRP vs Attention | RGRP vs BX |
|---:|---:|---:|---:|
| `7` | `-0.054` | `-0.088` | `-0.034` |
| `42` | `-0.032` | `-0.073` | `-0.041` |
| `123` | `-0.054` | `-0.083` | `-0.029` |

## Eval Curves

| Seed | Lane | 2047 | 4095 | 6143 | 8191 | 10239 | 12287 | 14335 | 16383 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `7` | `attention` | `4.953` | `4.552` | `4.348` | `4.230` | `4.138` | `4.070` | `4.020` | `3.995` |
| `7` | `parcae-bx` | `4.925` | `4.462` | `4.278` | `4.168` | `4.079` | `4.013` | `3.966` | `3.941` |
| `7` | `parcae-rgrp-control` | `4.896` | `4.464` | `4.260` | `4.142` | `4.051` | `3.982` | `3.933` | `3.907` |
| `123` | `attention` | `4.948` | `4.532` | `4.331` | `4.217` | `4.127` | `4.060` | `4.011` | `3.987` |
| `123` | `parcae-bx` | `4.907` | `4.458` | `4.272` | `4.161` | `4.075` | `4.008` | `3.957` | `3.933` |
| `123` | `parcae-rgrp-control` | `4.883` | `4.454` | `4.254` | `4.136` | `4.046` | `3.979` | `3.929` | `3.904` |

## Interpretation

- The `d_model=192` replication passed cleanly.
- RGRP-control beat both controls on all three seeds.
- The RGRP-over-B(x) margin was consistent: `0.029` to `0.041` eval loss.
- RGRP-control's aggregate loss variance was low at this rung: `0.0012`.
- B(x) remained a strong scaffold control and beat attention on all three seeds.
- RGRP-control still pays a throughput tax: about `74.7%` of attention median
  steady throughput and about `91.7%` of B(x).

## Decision

The first width-scale result is now replicated across three seeds. This is a
stronger quality signal than the smaller `d_model=128` rung:

- `d_model=128` three-seed mean RGRP-vs-B(x): `-0.014`
- `d_model=192` three-seed mean RGRP-vs-B(x): `-0.035`

The next proof-ladder rung can scale model size again. The cleanest next step is
to run a single-seed scout at a larger width, while still carrying all three
lanes:

- `d_model=256`
- `heads=8`
- `head_dim=32`
- `mlp_dim=1024`
- `layers=8`
- same Parcae middle-band schedule
- same data, tokenizer, context, batch if it fits

If `batch=64` does not fit at that shape, retry at `batch=32` and document the
token-budget difference explicitly.

## Artifacts

- `fractal_parcae_proof_ladder_20260420T143015Z/parcae8-attention-seed7-20260420T143015Z.log`
- `fractal_parcae_proof_ladder_20260420T143015Z/parcae8-parcae-bx-seed7-20260420T143015Z.log`
- `fractal_parcae_proof_ladder_20260420T143015Z/parcae8-parcae-rgrp-control-seed7-20260420T143015Z.log`
- `fractal_parcae_proof_ladder_20260420T143015Z/parcae8-attention-seed123-20260420T143015Z.log`
- `fractal_parcae_proof_ladder_20260420T143015Z/parcae8-parcae-bx-seed123-20260420T143015Z.log`
- `fractal_parcae_proof_ladder_20260420T143015Z/parcae8-parcae-rgrp-control-seed123-20260420T143015Z.log`
- `parcae_width_rep_20260420T143015Z.nohup.log`
