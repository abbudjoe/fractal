# Parcae/RGRP-Control 8-Layer FineWeb-EDU GCS Rung

Validated on 2026-04-19 to 2026-04-20 with:

- TPU VM: `fractal-parcae-fw2-202604192316`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- local artifact directory:
  `experiments/jax_tpu/maxtext_quality/parcae8_fineweb_gcs_rung_20260419T2327Z-parcae8-fineweb-gcs-rung8192`

## Contract

This rung keeps the faithful 8-layer Parcae proof-ladder shape from the prior
FineWeb-EDU run, doubles the step budget, and broadens the train stream from one
remote parquet file to a GCS-staged wildcard over four train shards.

- physical decoder layers: `8`
- loop count: `2`
- middle loop band: `layers 3..4`
- execution scaffold: `A0 A1 A2 [A3 A4] [A3 A4] A5 A6 A7`
- `d_model=128`, `heads=4`, `head_dim=32`, `mlp_dim=512`
- context `256`, batch `64`
- vocab size `32000`, tokenizer `openlm-research/open_llama_3b_v2`
- dataset: `HuggingFaceFW/fineweb-edu`, config dump `CC-MAIN-2013-20`
- train files: `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/train/*.parquet`
- eval files: `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/eval/*.parquet`
- staged train shards: `0000.parquet`, `0001.parquet`, `0002.parquet`, `0003.parquet`
- staged eval shard: `0013.parquet`
- staged compressed bytes: `9,462,253,997` train, `2,284,902,248` eval, `11,747,156,245` total
- seed: `42`
- steps: `8192`
- eval interval: `2048`
- eval steps: `64`

Per lane, the configured train budget was
`8192 * 64 * 256 = 134,217,728` token positions. Packed variable-length
examples reduced this to `129,678,585` non-padding train tokens per lane. Four
eval passes consumed `4,046,440` non-padding eval tokens per lane.

## Results

| Lane | Logged Params | Final Eval Loss | Delta vs Attention | Final Train Loss | Median Fast Tok/s | Speed vs Attention | Log Duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention` | `0.010B` | `4.4840` | `+0.0000` | `4.3470` | `765,321` | `100.0%` | `1277s` |
| `parcae-rgrp-control` | `0.010B` | `4.4010` | `-0.0830` | `4.2500` | `573,489` | `74.9%` | `1316s` |
| `parcae-bx` | `0.010B` | `4.4050` | `-0.0790` | `4.2640` | `637,510` | `83.3%` | `1279s` |

## Eval Curves

| Lane | Step 2047 | Step 4095 | Step 6143 | Step 8191 |
|---|---:|---:|---:|---:|
| `attention` | `5.0260` | `4.7080` | `4.5500` | `4.4840` |
| `parcae-rgrp-control` | `5.0080` | `4.6660` | `4.4810` | `4.4010` |
| `parcae-bx` | `5.0160` | `4.6700` | `4.4810` | `4.4050` |

## Interpretation

- Both Parcae lanes beat the attention control on every eval checkpoint in this
  8192-step FineWeb-EDU GCS rung.
- `parcae-rgrp-control` remains the loss winner, but only narrowly: final eval
  loss was `0.0040` better than `parcae-bx`.
- `parcae-bx` captured most of the quality gain while running materially faster
  than `parcae-rgrp-control`: about `83.3%` of attention speed versus `74.9%`.
- The gap between `parcae-rgrp-control` and `parcae-bx` narrowed relative to the
  4096-step rung. At 4096 steps RGRP led BX by `0.0110`; here the final gap is
  `0.0040`.
- Attention is still the TPU throughput winner, but it is no longer the quality
  winner at this `~10M` proof-ladder shape and data budget.
- This is stronger evidence than the 4096-step remote-parquet run because the
  train stream is broader and the budget is doubled. It is still one seed, a
  small model, and a TPU/JAX lowering, so it should not be used to claim CUDA
  transfer, frontier scaling, or general attention replacement.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.
