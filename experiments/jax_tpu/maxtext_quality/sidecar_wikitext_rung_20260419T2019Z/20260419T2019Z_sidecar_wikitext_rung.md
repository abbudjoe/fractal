# RGRP Sidecar WikiText-103 Rung

Run date: 2026-04-19

## Goal

Move the current layer `1`, zero-init rotary gated recurrent state update
primitive sidecar off TinyStories and onto a broader language-modeling dataset
without changing the MaxText/JAX TPU harness. This rung tests whether the
quality signal survives against:

- an attention-only baseline
- the strongest matched extra-capacity control from the previous rung,
  `tiny-mlp`

This is still a small model and short training budget. It is not a scale claim.

## Execution

- TPU VM: `fractal-wikitext-rung-202604192010`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_sidecar_wikitext_rung_tpu.sh`
- dataset: `Salesforce/wikitext`
- dataset config: `wikitext-103-raw-v1`
- validation split: `validation`
- tokenizer: `gpt2`
- model: 4 layers, `d_model=256`, `mlp_dim=1024`, 4 heads, sequence length 512
- training: 1000 steps, eval every 250 steps, 20 eval steps
- seeds: `0 1 2`

Smoke command:

```bash
SEEDS='0' LANES='attention rgrp tiny-mlp' SMOKE_STEPS=5 EVAL_INTERVAL=5 EVAL_STEPS=1 \
  bash "$HOME/run_maxtext_sidecar_wikitext_rung_tpu.sh" \
  20260419T2015Z-wikitext-smoke \
  "$HOME/fractal_sidecar_wikitext_smoke_20260419T2015Z-wikitext-smoke"
```

Full command:

```bash
SEEDS='0 1 2' LANES='attention rgrp tiny-mlp' STEPS=1000 EVAL_INTERVAL=250 EVAL_STEPS=20 \
  bash "$HOME/run_maxtext_sidecar_wikitext_rung_tpu.sh" \
  20260419T2019Z \
  "$HOME/fractal_sidecar_wikitext_rung_20260419T2019Z"
```

The TPU VM was deleted after copying logs.

## Results

| Lane | Seed | Final Eval | Delta vs Attention | Final Train | Median tok/s |
|---|---:|---:|---:|---:|---:|
| `attention` | `0` | `5.941` | `+0.000` | `5.699` | `326,534` |
| `rgrp` | `0` | `5.935` | `-0.006` | `5.686` | `291,055` |
| `tiny-mlp` | `0` | `5.938` | `-0.003` | `5.699` | `329,332` |
| `attention` | `1` | `5.951` | `+0.000` | `5.817` | `344,204` |
| `rgrp` | `1` | `5.938` | `-0.013` | `5.821` | `314,086` |
| `tiny-mlp` | `1` | `5.940` | `-0.011` | `5.823` | `340,776` |
| `attention` | `2` | `5.939` | `+0.000` | `6.104` | `332,539` |
| `rgrp` | `2` | `5.929` | `-0.010` | `6.108` | `309,352` |
| `tiny-mlp` | `2` | `5.936` | `-0.003` | `6.114` | `340,745` |

| Lane | Mean Final Eval | Std | Mean Delta vs Attention | Wins vs Attention | Mean Final Train | Mean Median tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `attention` | `5.9437` | `0.0052` | `+0.0000` | `0/3` | `5.8733` | `334,426` |
| `rgrp` | `5.9340` | `0.0037` | `-0.0097` | `3/3` | `5.8717` | `304,831` |
| `tiny-mlp` | `5.9380` | `0.0016` | `-0.0057` | `3/3` | `5.8787` | `336,951` |

Pairwise against RGRP:

| Control | Per-Seed Delta vs RGRP | Mean Delta vs RGRP | Wins vs RGRP |
|---|---:|---:|---:|
| `tiny-mlp` | `[+0.003, +0.002, +0.007]` | `+0.0040` | `0/3` |

## Interpretation

- RGRP beat attention on all three WikiText-103 seeds.
- RGRP also beat the strongest matched extra-capacity control, `tiny-mlp`, on
  all three seeds.
- The effect is much smaller than on TinyStories: mean final eval improved by
  `-0.0097` versus attention and `-0.0040` versus `tiny-mlp`.
- The TPU `lax.scan` speed tax remains real: RGRP averaged `304,831`
  tok/s/device versus `334,426` for attention and `336,951` for `tiny-mlp`.

This rung strengthens the claim that the recurrent state carry is doing
something beyond simply adding a tiny side expert. It does not prove scaling,
CUDA throughput, tuned-Transformer superiority, or equal-wall-clock superiority.

## Next Rung

The next decision-relevant rung should keep this exact attention/RGRP/tiny-MLP
comparison and increase either:

- training budget on WikiText-103, to see whether the small RGRP margin widens
  or washes out, or
- model size/context, if the goal is to test whether the recurrent sidecar
  becomes more useful as the residual stream carries richer information.

Do not add new architectural variants until this same three-lane proof ladder is
stable at the next scale.
