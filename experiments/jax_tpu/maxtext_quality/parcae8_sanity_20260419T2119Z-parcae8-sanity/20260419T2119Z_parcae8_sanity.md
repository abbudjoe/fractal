# Parcae/RGRP-Control 8-Layer MaxText Sanity Rung

Run date: 2026-04-19

## Goal

Run the first non-trivial TPU/JAX comparison on the 8-layer Parcae proof-ladder
shape, after the 5-step port smoke proved that the MaxText patch compiled and
trained.

This is still a bounded sanity rung, not a final quality claim. It uses
WikiText-103 through MaxText's Hugging Face input path, one seed, and 1,000
steps per lane. It does not yet match the earlier H100/FineWeb proof-ladder
data contract.

## Contract

- Physical decoder layers: `8`
- Loop count: `2`
- Middle loop band: `layers 3..4`
- Execution scaffold: `A0 A1 A2 [A3 A4] [A3 A4] A5 A6 A7`
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
- learning rate: `0.001`
- seed: `42`

The candidate profiles were:

- `attention`: unchanged MaxText transformer
- `parcae-looped`: prelude -> looped middle -> coda, direct loop injection
- `parcae-bx`: same scaffold with learned `B(x)` value/gate injection
- `parcae-rgrp-control`: same scaffold with a full-width RGRP scan controlling
  the injection value/gate

## Execution

- TPU VM: `fractal-parcae-sanity-202604192115`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- run stamp: `20260419T2119Z-parcae8-sanity`

Command:

```bash
SEEDS='42' \
LANES='attention parcae-looped parcae-bx parcae-rgrp-control' \
STEPS=1000 \
EVAL_INTERVAL=250 \
EVAL_STEPS=20 \
bash "$HOME/run_maxtext_parcae_proof_ladder_tpu.sh" \
  20260419T2119Z-parcae8-sanity \
  "$HOME/fractal_parcae_proof_ladder_sanity_20260419T2119Z-parcae8-sanity"
```

Per lane, the configured train budget was
`1000 * 64 * 256 = 16,384,000` token positions. WikiText padding reduced this
to `15,236,828` non-padding train tokens per lane. Four eval passes consumed
`1,074,268` non-padding eval tokens per lane.

The TPU VM was deleted after copying logs.

## Results

| Lane | Logged Params | Final Eval Loss | Delta vs Attention | Final Train Loss | Median Checkpoint Tok/s | Train Nonpad Tokens | Eval Nonpad Tokens | Log Duration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `attention` | `0.010B` | `4.923` | `+0.000` | `4.979` | `219,809` | `15,236,828` | `1,074,268` | `159s` |
| `parcae-looped` | `0.010B` | `4.911` | `-0.012` | `4.974` | `260,579` | `15,236,828` | `1,074,268` | `168s` |
| `parcae-bx` | `0.010B` | `4.910` | `-0.013` | `4.974` | `279,668` | `15,236,828` | `1,074,268` | `167s` |
| `parcae-rgrp-control` | `0.010B` | `4.894` | `-0.029` | `4.959` | `325,079` | `15,236,828` | `1,074,268` | `176s` |

Eval curves:

| Lane | Step 249 | Step 499 | Step 749 | Step 999 |
|---|---:|---:|---:|---:|
| `attention` | `5.589` | `5.199` | `5.006` | `4.923` |
| `parcae-looped` | `5.599` | `5.205` | `4.998` | `4.911` |
| `parcae-bx` | `5.596` | `5.207` | `4.998` | `4.910` |
| `parcae-rgrp-control` | `5.582` | `5.184` | `4.982` | `4.894` |

## Interpretation

- All three Parcae lanes beat the attention control by the final 1,000-step
  eval.
- `parcae-rgrp-control` was the clear winner in this rung, with a final eval
  delta of `-0.029` versus attention.
- The RGRP-control lane also had the best final train loss.
- MaxText's instantaneous checkpoint throughput is noisy, especially around
  eval boundaries. Treat the median checkpoint tok/s column as a lowering smoke
  signal, not as a cost-grade benchmark.
- The per-log duration is a better cost reminder: RGRP-control took `176s`
  from first logged event to final eval, versus `159s` for attention.

The useful conclusion is that the 8-layer TPU/JAX port is alive enough to climb
one more rung. The careful conclusion is that this does not yet prove general
superiority over attention, CUDA transferability, or H100/FineWeb parity.

## Next Rung

Promote `attention` and `parcae-rgrp-control` first, then optionally include
`parcae-bx` as the strongest non-RGRP Parcae control.

The next run should:

- keep the 8-layer Parcae/RGRP shape fixed
- use a larger time and token budget
- avoid looping over a small dataset where possible
- preferably move from WikiText-103 to the OpenLLaMA-tokenized FineWeb cache or
  a MaxText-compatible equivalent
- replicate seeds only after the larger single-seed rung still shows a signal
