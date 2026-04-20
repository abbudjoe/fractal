# Parcae RGRP-Control Discretization Ablation

Date: 2026-04-20

This run isolated the Parcae discretization knob for the fixed
`parcae-rgrp-control` lane:

- `fractal_parcae_discretization=stable-exp`
- `fractal_parcae_discretization=zoh`

The prior Parcae training-contract retest used `zoh` for every Parcae lane, so
it could not tell whether the positive fixed-policy result depended on ZOH. This
ablation changes only the discretization policy.

## Environment

- TPU VM: `fractal-zoh-ablate-20260420033857`
- zone: `us-west4-a`
- hardware: `v5litepod-1`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout on TPU: `~/maxtext-zoh-ablate`
- local artifact root:
  `experiments/jax_tpu/maxtext_quality/parcae_discretization_ablation_20260420T035019Z/fractal_parcae_discretization_ablation_20260420T035019Z`

The TPU VM was deleted after copying logs.

## Shared Shape And Data

- candidate: `parcae-rgrp-control-looped-attention`
- physical decoder layers: `8`
- loop count: `2`
- loop policy: `fixed`
- recurrence active depth: `2`
- backward active depth: `1`
- middle loop band: `layers 3..4`
- execution scaffold: `A0 A1 A2 [A3 A4] [A3 A4] A5 A6 A7`
- `d_model=128`
- `heads=4`
- `head_dim=32`
- `mlp_dim=512`
- context length: `256`
- batch size: `64`
- vocab size: `32000`
- tokenizer: `openlm-research/open_llama_3b_v2`
- dataset: `HuggingFaceFW/fineweb-edu`, config dump `CC-MAIN-2013-20`
- train files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/train/*.parquet`
- eval files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/eval/*.parquet`
- seed: `42`
- train steps: `2048`
- eval interval: `512`
- eval steps: `64`
- configured train token positions per lane:
  `2048 * 64 * 256 = 33,554,432`
- observed non-padding train tokens per lane: `32,420,430`

## Command

```sh
for disc in stable-exp zoh; do
  RUN_STAMP="20260420T035019Z-${disc}"
  HF_PATH='parquet' HF_NAME='' \
  HF_TRAIN_FILES='gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/train/*.parquet' \
  HF_EVAL_FILES='gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/eval/*.parquet' \
  HF_EVAL_SPLIT='train' TRAIN_SPLIT='train' \
  SEEDS='42' LANES='parcae-rgrp-control' \
  STEPS=2048 EVAL_INTERVAL=512 EVAL_STEPS=64 \
  PARCAE_DISCRETIZATION="$disc" MU_REC=2 MU_BWD=1 \
  bash "$HOME/run_maxtext_parcae_proof_ladder_tpu.sh" \
    "$RUN_STAMP" \
    "$HOME/fractal_parcae_discretization_ablation_20260420T035019Z/$disc"
done
```

## Results

`Median steady tok/s` is the median `Tokens/s/device` for completed training
steps with `step >= 100` and `seconds < 0.5`.

| Discretization | Final Eval Loss | PPL | Final Train Loss | Median Steady Tok/s | Mean Steady Tok/s | Non-Padding Train Tokens | Log Duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stable-exp` | `5.015` | `150.584` | `4.972` | `548,988` | `669,811` | `32,420,430` | `467s` |
| `zoh` | `5.015` | `150.700` | `4.974` | `554,976` | `666,749` | `32,420,430` | `465s` |

Eval curves:

| Discretization | Step 511 | Step 1023 | Step 1535 | Step 2047 |
|---|---:|---:|---:|---:|
| `stable-exp` | `5.736` | `5.287` | `5.090` | `5.015` |
| `zoh` | `5.737` | `5.288` | `5.091` | `5.015` |

## Interpretation

- The discretization knob is effectively neutral at this `2048`-step rung.
- `stable-exp` had a tiny final perplexity edge, but both lanes rounded to the
  same final eval loss.
- `zoh` had a tiny median throughput edge, about `1.1%`, but the difference is
  small enough to treat as noise unless it repeats.
- The previous fixed-policy win is not obviously dependent on `zoh`.

## Decision

Use `stable-exp` as the proof-ladder default because it is simpler, has no extra
learned `dt_raw` parameter, and matched `zoh` on quality in this ablation.
Keep `zoh` as an explicit ablation knob for larger runs or if continuous-time
state tuning becomes a focus.

## Artifacts

- `stable-exp/parcae8-parcae-rgrp-control-seed42-20260420T035019Z-stable-exp.log`
- `zoh/parcae8-parcae-rgrp-control-seed42-20260420T035019Z-zoh.log`
- `parcae_discretization_ablation_20260420T035019Z.nohup.log`
