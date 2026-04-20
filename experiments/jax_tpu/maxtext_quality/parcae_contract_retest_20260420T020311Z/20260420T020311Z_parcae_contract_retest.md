# Parcae Training Contract Retest

Date: 2026-04-20

This run retested the Parcae/RGRP proof-ladder lanes after making the Parcae
training contract explicit in the MaxText patch surface:

- fixed vs stochastic per-sequence loop policy
- Poisson-style per-sequence depth sampling for the stochastic ablation
- separate recurrence and backward active-depth knobs
- explicit discretization policy
- the same outer loop policy available to both `parcae-bx` and
  `parcae-rgrp-control`

The goal was to answer whether the paper-contract pieces we had omitted should
become the default before moving to a longer quality run.

## Environment

- TPU VM: `fractal-parcae-contract-202604200132`
- zone: `us-west4-a`
- hardware: `v5litepod-1`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout on TPU: `~/maxtext-parcae-contract`
- local artifact root:
  `experiments/jax_tpu/maxtext_quality/parcae_contract_retest_20260420T020311Z/fractal_parcae_contract_rung_20260420T020311Z`

The TPU VM was deleted after copying logs. A final `gcloud compute tpus tpu-vm
list` check returned no active TPU VMs.

## Shared Shape And Data

- physical decoder layers: `8`
- loop count: `2`
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
- train steps: `4096`
- eval interval: `1024`
- eval steps: `64`
- configured train token positions per lane:
  `4096 * 64 * 256 = 67,108,864`
- observed non-padding train tokens per lane: `64,842,323`

## Command

```sh
STAMP='20260420T020311Z'
HF_PATH='parquet' HF_NAME='' \
HF_TRAIN_FILES='gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/train/*.parquet' \
HF_EVAL_FILES='gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/eval/*.parquet' \
HF_EVAL_SPLIT='train' TRAIN_SPLIT='train' \
SEEDS='42' LANES='attention parcae-bx parcae-bx-perseq parcae-rgrp-control parcae-rgrp-control-perseq' \
STEPS=4096 EVAL_INTERVAL=1024 EVAL_STEPS=64 \
PARCAE_DISCRETIZATION='zoh' MU_REC=2 MU_BWD=1 PERSEQ_MAX_LOOP_COUNT=4 \
bash "$HOME/run_maxtext_parcae_proof_ladder_tpu.sh" "$STAMP" "$HOME/fractal_parcae_contract_rung_$STAMP"
```

## Results

`Median steady tok/s` is the median `Tokens/s/device` for completed training
steps with `step >= 100` and `seconds < 0.5`.

| Lane | Final Eval Loss | Delta vs Attention | PPL | Final Train Loss | Median Steady Tok/s | Mean Steady Tok/s | Log Duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention` | `4.739` | `+0.000` | `114.369` | `4.796` | `765,250` | `856,939` | `747s` |
| `parcae-bx` | `4.705` | `-0.034` | `110.471` | `4.766` | `638,379` | `735,159` | `764s` |
| `parcae-rgrp-control` | `4.681` | `-0.058` | `107.828` | `4.738` | `555,051` | `652,234` | `778s` |
| `parcae-bx-perseq` | `4.805` | `+0.066` | `122.068` | `4.882` | `470,291` | `581,909` | `790s` |
| `parcae-rgrp-control-perseq` | `4.779` | `+0.040` | `118.946` | `4.853` | `437,794` | `542,845` | `806s` |

Eval curves:

| Lane | Step 1023 | Step 2047 | Step 3071 | Step 4095 |
|---|---:|---:|---:|---:|
| `attention` | `5.363` | `4.971` | `4.805` | `4.739` |
| `parcae-bx` | `5.365` | `4.950` | `4.771` | `4.705` |
| `parcae-rgrp-control` | `5.335` | `4.922` | `4.746` | `4.681` |
| `parcae-bx-perseq` | `5.490` | `5.063` | `4.877` | `4.805` |
| `parcae-rgrp-control-perseq` | `5.480` | `5.042` | `4.852` | `4.779` |

## Interpretation

- `parcae-rgrp-control` was the best lane in this retest: final eval loss
  `4.681`, `0.058` better than attention and `0.024` better than fixed
  `parcae-bx`.
- Fixed `parcae-bx` also beat attention, so the Parcae outer scaffold remains
  a live control even without RGRP.
- The stochastic per-sequence loop-depth policy was worse at this scale. It
  hurt both quality and speed for `parcae-bx` and `parcae-rgrp-control`.
- The per-sequence variants also paid a clear runtime tax. `parcae-rgrp-control`
  fixed policy reached about `72.5%` of attention median steady throughput;
  `parcae-rgrp-control-perseq` reached about `57.2%`.
- All Parcae lanes in this retest used `PARCAE_DISCRETIZATION=zoh`, so this
  confirms a positive fixed-policy RGRP-control result under the more explicit
  paper-contract surface, but it does not isolate `zoh` against the older
  `stable-exp` discretization.

## Decision

Keep `fractal_parcae_loop_policy=fixed` as the proof-ladder default. Keep
stochastic per-sequence loop depth as an explicit ablation only.

Before spending a longer TPU run, the clean next ablation is fixed
`parcae-rgrp-control` with `zoh` vs fixed `parcae-rgrp-control` with
`stable-exp` at the same seed and either this `4096`-step rung or a cheaper
`2048`-step rung.

## Artifacts

- `parcae8-attention-seed42-20260420T020311Z.log`
- `parcae8-parcae-bx-seed42-20260420T020311Z.log`
- `parcae8-parcae-bx-perseq-seed42-20260420T020311Z.log`
- `parcae8-parcae-rgrp-control-seed42-20260420T020311Z.log`
- `parcae8-parcae-rgrp-control-perseq-seed42-20260420T020311Z.log`
