# AWS SageMaker CUDA Port

Last updated: 2026-04-22

## Purpose

This is the smallest CUDA bridge for the Path 1 / Parcae-RGRP proof ladder. It is not a new training stack. The launcher packages the existing Python Path 1 runner, the shared `python/` modules, and the tiny built-in JSONL smoke corpus, then submits a SageMaker Training job on one small NVIDIA GPU.

## Current AWS Surface

Observed account/profile:

- AWS profile: `codex-eml`
- Region: `us-east-1`
- Approved SageMaker GPU quota: `ml.g6.2xlarge for training job usage = 4`
- S3 bucket: `fractal-sagemaker-806880856465-us-east-1`
- SageMaker execution role: `arn:aws:iam::806880856465:role/fractal-sagemaker-execution-role`

Observed restrictions:

- The IAM user initially could not list SageMaker jobs/domains/apps.
- The IAM user initially could not list or create IAM roles.
- The IAM user initially could not list or create S3 buckets.
- Temporary bootstrap permissions allowed the bucket and execution role to be created.
- The launcher now requires `FRACTAL_SAGEMAKER_ROLE_ARN` and `FRACTAL_SAGEMAKER_BUCKET`.
- Durable scoped runner policy `FractalSageMakerCudaRunnerScoped` has now been attached to `codex-eml-benchmark` after `iam:PutUserPolicy` was added.
- Remaining hardening caveat: the CLI still did not set bucket-level public access block because `s3:PutBucketPublicAccessBlock` was denied in the earlier bootstrap surface.

## Launcher

Script:

```bash
python scripts/sagemaker_path1_cuda_smoke.py
```

Required control-plane inputs:

```bash
export FRACTAL_SAGEMAKER_ROLE_ARN='arn:aws:iam::<account>:role/<sagemaker-execution-role>'
export FRACTAL_SAGEMAKER_BUCKET='<existing-writable-bucket>'
```

Default GPU/image:

- Instance: `ml.g6.2xlarge`
- Image: `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker`

Default lanes:

- `attention-only`
- `parcae-p20-control-looped-attention`

This maps the current CUDA name for the RGRP/P20 control scaffold. The JAX/TPU lane name `parcae-rgrp-control-looped-attention` is mapped to this Python/CUDA scaffold by the contract layer.

## Parcae CUDA/JAX Parity Knobs

The CUDA Path 1 runner now exposes the Parcae contract pieces that were previously implicit or JAX-only:

- `--parcae-prelude-norm-kind layernorm|rmsnorm`
- `--parcae-backward-steps <n>`
- `--parcae-discretization stable-exp|zoh`
- `--parcae-dt-raw-init <float>`

The TPU/JAX proof-ladder wins used the Parcae/RGRP loop as an explicit recurrent-depth scaffold, not just "two Python loops around attention." For CUDA parity scouts, use:

```bash
--parcae-loop-count 2 \
--parcae-backward-steps 1 \
--parcae-prelude-norm-kind rmsnorm \
--parcae-discretization stable-exp \
--primitive-runtime-backend triton
```

This keeps the old `parcae-p20-control-looped-attention` CUDA name for compatibility, but functionally treats it as the RGRP-control lane.

## Token Cache Ingress

Repeated token-cache runs should use a SageMaker input channel backed by S3, not per-job Hugging Face download/extract. This avoids burning GPU job wall time in the managed `Downloading` phase and avoids coupling private-HF auth to every training job.

Staged 250M cache:

```text
s3://fractal-sagemaker-806880856465-us-east-1/fractal/token-caches/fineweb-cc-main-2024-10-openllama-tokens-250m-v1/
```

Staged 750M cache:

```text
s3://fractal-sagemaker-806880856465-us-east-1/fractal/token-caches/fineweb-cc-main-2024-10-openllama-tokens-750m-v1/
```

Staging command:

```bash
aws s3 sync \
  experiments/stage0/assets/fineweb/fineweb-cc-main-2024-10-openllama-tokens-250m-v1/ \
  s3://fractal-sagemaker-806880856465-us-east-1/fractal/token-caches/fineweb-cc-main-2024-10-openllama-tokens-250m-v1/ \
  --only-show-errors \
  --region us-east-1 \
  --profile codex-eml
```

Verified staging result:

- Objects: `56`
- Total size: `991.9 MiB`
- Layout: extracted cache files at the channel root, including `manifest.json`

Verified 750M staging result:

- Objects: `763`
- Total size: `2.8 GiB`
- Train tokens: `750,000,544`
- Eval tokens: `10,029,403`
- No-repeat cap at context 256, batch 64: `45,598` steps, or `747,077,632` sampled train tokens

Use it in future CUDA parity/scaling runs:

```bash
--runner token-cache \
--token-cache-s3-uri s3://fractal-sagemaker-806880856465-us-east-1/fractal/token-caches/fineweb-cc-main-2024-10-openllama-tokens-250m-v1/ \
--token-cache-input-mode FastFile
```

When `--token-cache-s3-uri` is provided, the launcher mounts the cache as the `token_cache` SageMaker channel, sets `FRACTAL_SCOUT_DATA_ROOT=/opt/ml/input/data/token_cache`, sets `FRACTAL_SCOUT_TOKEN_CACHE_DIR=.`, and does not include `HF_TOKEN` in the training job environment.

Operational note: S3/FastFile is regional rather than AZ-volume-bound, so it avoids the RunPod-style failure mode where a persistent volume exists in a location without the desired GPU capacity. If we later need FSx for much larger repeated runs, we should treat that as a separate AZ/capacity planning decision.

### Lazy Token-Batch Contract

The 750M cache exposed a data-plane contract bug: the token-id loader still used the small-corpus path of eagerly materializing every `[input_ids, target_ids]` batch before training. At 750M tokens, this duplicates the corpus into pinned host batches and can make SageMaker report:

```text
ClientError: Please use an instance type with more memory, or reduce the size of job data processed on an instance.
```

That failure is not a trustworthy model-fit result. The token-id corpus path now keeps a sequence-like lazy batch collection: the runner can still use `len()`, indexing, slices, and deterministic shuffling, but only the requested batch is materialized and optionally pinned. Whole-split tokens are stored as CPU `int32` and converted to `long` only per batch for the embedding lookup.

Regression coverage:

```bash
.venv/bin/python -m pytest python/tests/test_data.py python/tests/test_sagemaker_path1_cuda_smoke.py -q
```

Validation job:

- Job: `fractal-path1-cuda-fastfile-smoke-20260422T044822Z`
- Result: `Completed`
- Channel proof: `token_cache_probe` reported `manifest_exists=true`
- Runner proof: `using existing token cache: /opt/ml/input/data/token_cache/manifest.json`
- Local artifact: `experiments/aws_sagemaker/path1_cuda_scout/fractal-path1-cuda-fastfile-smoke-20260422T044822Z`

| Lane | Params | Initial Loss | Final Loss | train tok/s | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|
| `attention-only` | 9,778,176 | 10.6005 | 9.9155 | 127,409.46 | 6,725.90 |

Interpretation: the S3/FastFile path is now operational. This was a two-step contract smoke, not a quality result. The important result is that the training job saw the S3-mounted manifest directly and skipped Hugging Face download/extract.

## Start-Small Command

Dry-run the generated SageMaker request:

```bash
.venv/bin/python scripts/sagemaker_path1_cuda_smoke.py --dry-run
```

Submit and wait:

```bash
.venv/bin/python scripts/sagemaker_path1_cuda_smoke.py --download-output
```

The first run intentionally uses a tiny smoke budget:

- `d_model=128`
- `total_layers=4`
- `seq_len=64`
- `batch_size=4`
- `steps=5`
- `eval_batches=2`
- `dtype=fp32`
- `primitive_runtime_backend=torch`

That is only meant to prove CUDA/container/SageMaker wiring. If it passes, the next rung should switch to the real token-cache runner and the 8-layer proof-ladder shape.

## Promotion Path

1. Confirm the tiny smoke launches and emits CUDA device metadata.
2. Run a slightly larger synthetic/tiny-corpus smoke with `dtype=bf16`.
3. Port the token-cache hydration path from `scripts/v3a_python_path1_parcae_h100_promotion.py` into the SageMaker launcher.
4. Run the H100-style lanes on `ml.g6.2xlarge` only for shape validation, not final quality.
5. If AWS exposes larger SageMaker GPU quota later, reuse the same request surface with a stronger instance type.

## First Smoke Result

Completed on 2026-04-21:

- Job: `fractal-path1-cuda-smoke-20260421Taws002`
- Instance: `ml.g6.2xlarge`
- CUDA device: `NVIDIA L4 (cc 8.9)`
- Local artifact: `experiments/aws_sagemaker/path1_cuda_smoke/fractal-path1-cuda-smoke-20260421Taws002`

| Lane | Params | Initial Loss | Final Loss | train tok/s | overall tok/s | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|---:|
| `attention-only` | 595,712 | 5.6860 | 3.6596 | 13,907.61 | 9,943.52 | 30.48 |
| `parcae-p20-control-looped-attention` | 691,392 | 5.5879 | 3.7036 | 2,514.11 | 2,775.62 | 34.27 |

Interpretation: this proves the AWS/SageMaker CUDA control plane and the current Python/CUDA RGRP-control scaffold. It is not a quality result; the budget is intentionally tiny and uses the built-in smoke JSONL corpus.

## 8-Layer Shape Smoke

Completed on 2026-04-21:

- Job: `fractal-path1-cuda-smoke-8layer-20260421Taws003`
- Runner: built-in tiny JSONL smoke corpus
- Instance: `ml.g6.2xlarge`
- CUDA device: `NVIDIA L4 (cc 8.9)`
- Shape: 8 layers, `d_model=128`, 4 heads, context 256, batch 64, loop count 2, BF16
- Local artifact: `experiments/aws_sagemaker/path1_cuda_smoke/fractal-path1-cuda-smoke-8layer-20260421Taws003`

| Lane | Params | Initial Loss | Final Loss | train tok/s | overall tok/s | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|---:|
| `attention-only` | 1,651,968 | 5.8996 | 3.7631 | 238,217.69 | 105,318.93 | 658.90 |
| `parcae-bx-looped-attention` | 1,685,504 | 5.6571 | 3.9831 | 217,285.58 | 87,511.07 | 876.21 |
| `parcae-p20-control-looped-attention` | 1,747,648 | 5.7743 | 4.0309 | 35,015.85 | 26,518.19 | 962.63 |

Interpretation: the real 8-layer proof-ladder shape fits on L4 at batch 64. This is still not a quality run because it uses the tiny smoke JSONL corpus and only 2 steps.

## Matched Token-Cache Scout

Completed on 2026-04-21:

- Job: `fractal-path1-cuda-scout-8layer-20260421Taws004`
- Runner: token-cache scout via `scripts/v3a_python_path1_parcae_h100_promotion.py`
- Token cache: `fineweb-cc-main-2024-10-openllama-tokens-250m-v1.tar.zst`
- Instance: `ml.g6.2xlarge`
- CUDA device: `NVIDIA L4 (cc 8.9)`
- Shape: 8 layers, `d_model=128`, 4 heads, context 256, batch 64, loop count 2, BF16
- Budget: 100 steps, 16 eval batches
- Local artifact: `experiments/aws_sagemaker/path1_cuda_scout/fractal-path1-cuda-scout-8layer-20260421Taws004`

| Lane | Params | Initial Loss | Final Loss | train tok/s | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|
| `attention-only` | 9,778,176 | 10.5847 | 6.7034 | 181,997.77 | 6,725.90 |
| `parcae-bx-looped-attention` | 9,811,712 | 10.4526 | 6.8218 | 168,666.84 | 6,943.23 |
| `parcae-p20-control-looped-attention` | 9,873,856 | 10.4995 | 6.7467 | 36,805.45 | 7,030.12 |

Interpretation: the matched CUDA scout is now operational. On this short L4 run, RGRP-control sits between attention and B(x) on final loss but remains roughly 5x slower than attention. This is a sensitivity/correctness scout, not a final architecture verdict.

## CUDA Parity Scout

Completed on 2026-04-22:

- Job: `fractal-path1-cuda-parity-scout-20260422T013908Z`
- Runner: token-cache scout
- Token cache: `fineweb-cc-main-2024-10-openllama-tokens-250m-v1.tar.zst`
- Instance: `ml.g6.2xlarge`
- CUDA device: `NVIDIA L4 (cc 8.9)`
- Shape: 8 layers, `d_model=128`, 4 heads, context 256, batch 64, loop count 2, BF16
- Budget: 100 steps, 16 eval batches
- Parcae parity knobs: `backward_steps=1`, `prelude_norm=rmsnorm`, `discretization=stable-exp`, `primitive_runtime_backend=triton`
- Local artifact: `experiments/aws_sagemaker/path1_cuda_scout/fractal-path1-cuda-parity-scout-20260422T013908Z`

This is the first AWS CUDA scout that attempts to match the successful TPU/JAX Parcae-RGRP training contract instead of the older Python Path 1 default loop contract.

| Lane | Params | Initial Loss | Final Loss | train tok/s | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|
| `attention-only` | 9,778,176 | 10.5847 | 6.7034 | 181,484.32 | 6,725.90 |
| `parcae-bx-looped-attention` | 9,811,584 | 10.4530 | 6.7803 | 173,863.17 | 6,790.16 |
| `parcae-p20-control-looped-attention` | 9,873,728 | 10.4998 | 6.7113 | 158,591.26 | 6,845.06 |

Interpretation: parity matters. The older AWS CUDA scout used the Torch scan backend plus the old Python Parcae loop contract and measured RGRP/P20-control at `36,805 tok/s`. With the parity knobs and Triton primitive backend, the same L4 scout measured `158,591 tok/s`, about 4.3x faster. The 100-step loss remains too early for architecture claims, but RGRP-control now lands very close to attention and ahead of B(x) under the matched scout.

## CUDA Parity 2048-Step Checkpoint

Completed on 2026-04-22:

- Job: `fractal-path1-cuda-parity-2048-20260422T020724Z`
- Runner: token-cache scout
- Token cache: `fineweb-cc-main-2024-10-openllama-tokens-250m-v1.tar.zst`
- Instance: `ml.g6.2xlarge`
- CUDA device: `NVIDIA L4 (cc 8.9)`
- Shape: 8 layers, `d_model=128`, 4 heads, context 256, batch 64, loop count 2, BF16
- Budget: 2048 steps, 64 eval batches
- Parcae parity knobs: `backward_steps=1`, `prelude_norm=rmsnorm`, `discretization=stable-exp`, `primitive_runtime_backend=triton`
- Local artifact: `experiments/aws_sagemaker/path1_cuda_scout/fractal-path1-cuda-parity-2048-20260422T020724Z`

| Lane | Params | Initial Loss | Final Loss | Delta vs Attention | Delta vs B(x) | train tok/s | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention-only` | 9,778,176 | 10.5876 | 5.2199 | +0.0000 | -0.0416 | 180,897.22 | 6,725.90 |
| `parcae-bx-looped-attention` | 9,811,584 | 10.4536 | 5.2615 | +0.0416 | +0.0000 | 171,663.43 | 6,790.16 |
| `parcae-p20-control-looped-attention` | 9,873,728 | 10.5014 | 5.1576 | -0.0623 | -0.1038 | 164,885.54 | 6,845.06 |

Interpretation: the missing CUDA loss edge was mostly an early-checkpoint artifact. At 100 steps, the parity lane was not yet ahead of attention. At the first TPU/JAX-style checkpoint scale, 2048 steps, RGRP/P20-control beats both matched controls on CUDA too. The throughput cost on L4 is about 8.9% vs attention and 3.9% vs B(x), far from the earlier 5x collapse.

This does not prove the full 16k-step TPU/JAX curve transfers all the way to CUDA, but it strongly supports the contract-level diagnosis: the old CUDA regression mixed an under-length quality run with a non-parity runtime path.

## CUDA Parity 8192-Step Rung

Completed on 2026-04-22:

- Job: `fractal-path1-cuda-parity-8192-20260422T023232Z`
- Runner: token-cache scout
- Token cache: `fineweb-cc-main-2024-10-openllama-tokens-250m-v1.tar.zst`
- Instance: `ml.g6.2xlarge`
- CUDA device: `NVIDIA L4 (cc 8.9)`
- Shape: 8 layers, `d_model=128`, 4 heads, context 256, batch 64, loop count 2, BF16
- Budget: 8192 steps, 64 eval batches
- Parcae parity knobs: `backward_steps=1`, `prelude_norm=rmsnorm`, `discretization=stable-exp`, `primitive_runtime_backend=triton`
- Local artifact: `experiments/aws_sagemaker/path1_cuda_scout/fractal-path1-cuda-parity-8192-20260422T023232Z`

| Lane | Params | Initial Loss | Final Loss | Delta vs Attention | Delta vs B(x) | train tok/s | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention-only` | 9,778,176 | 10.5876 | 4.6627 | +0.0000 | -0.0062 | 179,088.84 | 6,725.90 |
| `parcae-bx-looped-attention` | 9,811,584 | 10.4536 | 4.6689 | +0.0062 | +0.0000 | 169,919.24 | 6,790.16 |
| `parcae-p20-control-looped-attention` | 9,873,728 | 10.5014 | 4.6144 | -0.0482 | -0.0544 | 164,470.74 | 6,845.06 |

Interpretation: the TPU/JAX-style RGRP/P20-control advantage reproduces on CUDA at the 8192-step checkpoint. B(x) remains effectively tied with attention on this CUDA token-cache/data contract, while RGRP-control is clearly ahead of both. The L4 throughput cost is about 8.2% versus attention and 3.2% versus B(x), with about 119 MB more peak CUDA memory than attention.

This is now a real cross-backend signal: TPU/JAX and CUDA agree that the RGRP-control Parcae lane improves quality under the matched 8-layer proof-ladder shape. The remaining work is seed replication and a longer 16k/30k rung, not basic resurrection.

## CUDA Parity 8192-Step Seed Replication

Completed on 2026-04-22:

- Job: `fractal-path1-cuda-parity-8192-s43-20260422T033815Z`
- Runner: token-cache scout
- Token cache: `fineweb-cc-main-2024-10-openllama-tokens-250m-v1.tar.zst`
- Instance: `ml.g6.2xlarge`
- CUDA device: `NVIDIA L4 (cc 8.9)`
- Shape: 8 layers, `d_model=128`, 4 heads, context 256, batch 64, loop count 2, BF16
- Budget: 8192 steps, 64 eval batches
- Seed/data seed: `43`
- Parcae parity knobs: `backward_steps=1`, `prelude_norm=rmsnorm`, `discretization=stable-exp`, `primitive_runtime_backend=triton`
- Local artifact: `experiments/aws_sagemaker/path1_cuda_scout/fractal-path1-cuda-parity-8192-s43-20260422T033815Z`

| Lane | Params | Initial Loss | Final Loss | Delta vs Attention | Delta vs B(x) | train tok/s | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention-only` | 9,778,176 | 10.6593 | 4.6653 | +0.0000 | -0.0130 | 180,384.09 | 6,725.90 |
| `parcae-bx-looped-attention` | 9,811,584 | 10.5045 | 4.6783 | +0.0130 | +0.0000 | 172,099.13 | 6,790.16 |
| `parcae-p20-control-looped-attention` | 9,873,728 | 10.5122 | 4.6117 | -0.0536 | -0.0666 | 166,154.64 | 6,845.06 |

Two-seed aggregate for 8192-step CUDA parity (`seed=42,43`):

| Lane | Mean Final Loss | Loss Stdev | Mean Delta vs Attention | Mean tok/s |
|---|---:|---:|---:|---:|
| `attention-only` | 4.6640 | 0.0019 | +0.0000 | 179,736 |
| `parcae-bx-looped-attention` | 4.6736 | 0.0066 | +0.0096 | 171,009 |
| `parcae-p20-control-looped-attention` | 4.6131 | 0.0020 | -0.0509 | 165,313 |

Interpretation: the seed replication strengthens the signal. RGRP/P20-control beats attention and B(x) on both CUDA seeds, with low variance so far. The current mean loss advantage is about `0.051` over attention at roughly `92%` of attention throughput on L4.

## CUDA Full 250M-Token Rung

Completed on 2026-04-22:

- Job: `fractal-path1-cuda-parity-15199-s42-20260422T042837Z`
- Runner: token-cache scout
- Token cache: `fineweb-cc-main-2024-10-openllama-tokens-250m-v1`
- Instance: `ml.g6.2xlarge`
- CUDA device: `NVIDIA L4 (cc 8.9)`
- Shape: 8 layers, `d_model=128`, 4 heads, context 256, batch 64, loop count 2, BF16
- Budget: 15,199 steps, 64 eval batches
- Train tokens seen: `249,020,416`
- Parcae parity knobs: `backward_steps=1`, `prelude_norm=rmsnorm`, `discretization=stable-exp`, `primitive_runtime_backend=triton`
- Local artifact: `experiments/aws_sagemaker/path1_cuda_scout/fractal-path1-cuda-parity-15199-s42-20260422T042837Z`

| Lane | Params | Final Loss | Delta vs Attention | train tok/s | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|
| `attention-only` | 9,778,176 | 4.4905 | +0.0000 | 181,593.87 | 6,725.90 |
| `parcae-bx-looped-attention` | 9,811,584 | 4.4894 | -0.0011 | 173,256.67 | 6,790.16 |
| `parcae-p20-control-looped-attention` | 9,873,728 | 4.4346 | -0.0559 | 167,828.43 | 6,845.06 |

Interpretation: the RGRP-control edge survived the full 250M-token single-seed CUDA rung. B(x) remained essentially tied with attention, while RGRP-control improved loss by about `0.056` at roughly `92%` of attention throughput on L4.

## CUDA 50M Fit Gate

Completed on 2026-04-22:

- Initial failed jobs: `fractal-path1-cuda-50m-fit-smoke-20260422T061317Z`, `fractal-path1-cuda-50m-fit-smoke-b32-20260422T062713Z`
- Root cause: eager token-batch materialization duplicated the 750M token cache into host/pinned batches before training.
- Passing job after lazy loader fix: `fractal-path1-cuda-50m-fit-smoke-lazy-b64-20260422T064352Z`
- Runner: token-cache scout
- Token cache: `fineweb-cc-main-2024-10-openllama-tokens-750m-v1`
- Instance: `ml.g6.2xlarge`
- CUDA device: `NVIDIA L4 (cc 8.9)`
- Shape: 8 layers, `d_model=448`, 8 heads, context 256, batch 64, loop count 2, BF16
- Budget: 128 steps, 2 eval batches
- No-repeat cap at this shape/data contract: `45,598` steps, or `747,077,632` sampled train tokens
- Local artifact: `experiments/aws_sagemaker/path1_cuda_scout/fractal-path1-cuda-50m-fit-smoke-lazy-b64-20260422T064352Z`

| Lane | Params | Initial Loss | Final Loss | train tok/s | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|
| `attention-only` | 47,986,176 | 10.5812 | 6.2927 | 88,595.30 | 8,671.54 |
| `parcae-p20-control-looped-attention` | 49,144,928 | 10.4465 | 6.4359 | 17,854.50 | 9,102.99 |

Interpretation: the 50M 8L-wide shape fits on one L4 with comfortable CUDA memory after fixing the data plane. The 128-step quality read is too early to judge architecture quality and is worse for RGRP-control, but the fit result is sufficient to launch the 750M no-repeat rung. Expected L4 train time from this smoke is about `2.4h` for attention plus `11.7h` for RGRP-control, before evaluation and startup overhead.

Active follow-up jobs launched from this gate:

- Full 750M no-repeat A-vs-RGRP: `fractal-path1-cuda-50m-750m-b64-20260422T070014Z`
- Shape scout, 8L/d448 RGRP-only: `fractal-path1-cuda-50m-shape-8l-d448-rgrp512-20260422T070042Z`
- Shape scout, 12L/d400 RGRP-only: `fractal-path1-cuda-50m-shape-12l-d400-rgrp512-20260422T070046Z`
- Shape scout, 16L/d368 RGRP-only: `fractal-path1-cuda-50m-shape-16l-d368-rgrp512-20260422T070050Z`

## CUDA 50M Shape Allocation Scout

Completed on 2026-04-22:

- Runner: token-cache scout
- Token cache: `fineweb-cc-main-2024-10-openllama-tokens-750m-v1`
- Instance: `ml.g6.2xlarge`
- CUDA device: `NVIDIA L4 (cc 8.9)`
- Shared budget: 512 steps, 8 eval batches, context 256, batch 64, BF16
- Shared Parcae knobs: `loop_count=2`, `backward_steps=1`, `prelude_norm=rmsnorm`, `discretization=stable-exp`, `primitive_runtime_backend=triton`
- Scope: RGRP-control only, intended to compare parameter allocation before longer A-vs-RGRP rungs.

| Shape | Job | Params | Initial Loss | Final Loss | tok/s | Peak CUDA MB |
|---|---|---:|---:|---:|---:|---:|
| 8L wide, `d_model=448` | `fractal-path1-cuda-50m-shape-8l-d448-rgrp512-20260422T070042Z` | 49,144,928 | 10.4474 | 5.4847 | 18,173.87 | 9,102.99 |
| 12L medium, `d_model=400` | `fractal-path1-cuda-50m-shape-12l-d400-rgrp512-20260422T070046Z` | 49,626,600 | 10.5068 | 5.5353 | 17,151.64 | 10,027.63 |
| 16L narrow, `d_model=368` | `fractal-path1-cuda-50m-shape-16l-d368-rgrp512-20260422T070050Z` | 50,412,504 | 10.6208 | 5.5336 | 16,798.75 | 10,637.79 |

Interpretation: at the first bounded 50M allocation scout, 8L-wide wins on quality, throughput, and memory. The 12L/16L allocations fit, but they do not buy useful early loss and they increase the speed/memory tax. The current 50M RGRP allocation should stay `8L/d448` unless a longer run shows a reversal.
