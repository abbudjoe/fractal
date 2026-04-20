# Parcae/RGRP-Control 8-Layer MaxText Port Smoke

Run date: 2026-04-19

## Goal

Validate that the TPU/JAX MaxText surface can run the same architectural shape
as the H100 Parcae proof ladder, rather than the earlier 4-layer cheap smoke
shape.

This is a compile/contract smoke only. Five training steps are not quality
evidence.

## Ported Contract

- Physical decoder layers: `8`
- Loop count: `2`
- Middle loop band: `layers 3..4`, matching the Torch `_middle_loop_bounds`
  rule for 8 layers
- `d_model`: `128`
- heads: `4`
- head dim: `32`
- MLP dim: `512`
- context: `256`
- batch: `64`
- vocab size: `32000`
- tokenizer: `openlm-research/open_llama_3b_v2`
- dataset: `Salesforce/wikitext`, config `wikitext-103-raw-v1`
- train split: `train`
- eval split: `validation`
- dtype: `bfloat16`

The candidate profiles were:

- `attention`: unchanged MaxText transformer
- `parcae-looped`: prelude -> looped middle -> coda, direct loop injection
- `parcae-bx`: same scaffold with learned `B(x)` value/gate injection
- `parcae-rgrp-control`: same scaffold with a full-width RGRP scan controlling
  the injection value/gate

## Execution

- TPU VM: `fractal-parcae-port-202604192052`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- smoke stamp: `20260419T2057Z-parcae8-smoke`

Command:

```bash
SEEDS='42' \
LANES='attention parcae-looped parcae-bx parcae-rgrp-control' \
SMOKE_STEPS=5 \
EVAL_INTERVAL=5 \
EVAL_STEPS=1 \
bash "$HOME/run_maxtext_parcae_proof_ladder_tpu.sh" \
  20260419T2057Z-parcae8-smoke \
  "$HOME/fractal_parcae_proof_ladder_smoke_20260419T2057Z-parcae8-smoke"
```

Per lane, the configured train budget was `5 * 64 * 256 = 81,920` token
positions. WikiText padding reduced this to `74,181` non-padding train tokens
per lane. The single eval batch contained `15,302` non-padding tokens.

The TPU VM was deleted after copying logs.

## Smoke Results

| Lane | Logged Params | Eval Step | Eval Loss | Eval Tokens | Final Train Loss | Median Post-Step0 Tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `attention` | `0.010B` | `4` | `9.501` | `15,302` | `9.665` | `478,140` |
| `parcae-looped` | `0.010B` | `4` | `9.626` | `15,302` | `9.792` | `402,634` |
| `parcae-bx` | `0.010B` | `4` | `9.624` | `15,302` | `9.791` | `421,685` |
| `parcae-rgrp-control` | `0.010B` | `4` | `9.625` | `15,302` | `9.791` | `400,111` |

Interpretation:

- All four proof-ladder lanes compile, train, and evaluate under MaxText/JAX/TPU.
- The logged parameter scale matches the earlier H100 8-layer tiny-LM contract
  much more closely than the previous 4-layer GPT-2-vocab smoke.
- Do not read the 5-step loss ordering as a model result. The purpose was to
  verify the port and Flax parameter-sharing contract for the looped middle
  band.

## Next Rung

Run a bounded MaxText proof-ladder comparison with this same shape:

- seed `42`
- `attention`, `parcae-looped`, `parcae-bx`, `parcae-rgrp-control`
- same token budget per lane
- then promote finalists across seeds only if the seed-42 run shows a real
  quality signal

The data source remains the main unresolved difference from the H100 proof
ladder. The H100 run used the local 750M-token OpenLLaMA-tokenized FineWeb cache;
this smoke used WikiText text through MaxText's HF path.
