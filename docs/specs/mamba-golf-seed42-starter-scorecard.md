# Mamba Golf Seed 42 Starter Scorecard

This note freezes the first public-facing Mamba Golf starter run.

## Timestamp

- local run window: `2026-04-11 09:49-10:29 MDT`
- UTC artifact window: `2026-04-11T15:49:42Z` through `2026-04-11T16:29:24Z`
- source base commit at launch: `8716da8f0a987489612f8e99a794d1e3ff9e91bc`
- launcher: `scripts/runpod-mamba-golf-path1-gauntlet.sh 42`

The run was launched from the Mamba Golf scaffold worktree before this note was
committed. This scorecard freezes the result and the scaffold together on
`main`.

## Contract

Common settings:

- benchmark: `cuda-faithful-small-v1`
- corpus: `fineweb-stage0-local-bench-9row-v1`
- seed: `42`
- dtype: `bf16`
- shape: `d_model=128`, `layers=8`, `heads=4`, `ffn_multiplier=4`
- local attention window: `256`
- vocab size: `257`
- sequence length: `32`
- window stride: `32`
- batch size: `1`
- train steps: `961`
- eval batches: `94`
- warmup train steps: `1`
- warmup eval batches: `1`

Lane contracts:

| Lane | Env | Schedule | Runtime |
| --- | --- | --- | --- |
| `attention-official` | `official-mamba3` | `A A A A A A A A` | PyTorch SDPA |
| `mamba` | `official-mamba3` | `A M A M A M A M` | native `mamba_ssm.Mamba3`, `mamba3-siso-runtime` |
| `attention-triton` | `primitive-triton` | `A A A A A A A A` | PyTorch SDPA |
| `p20-triton` | `primitive-triton` | `A P A P A P A P` | `P20` Triton runtime, `block-diagonal-2` |

The duplicated attention lane is an environment anchor. Native Mamba and `P20`
currently require different CUDA dependency stacks, so the anchor checks whether
the environments materially perturb the attention-only baseline.

## Results

| Lane | Initial Loss | Final Loss | Train tok/s | Final Eval ms | CUDA Peak MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attention-official` | `5.9138` | `2.5425` | `4204.37` | `233.82` | `47.78` |
| `mamba` | `5.7072` | `2.2692` | `2562.77` | `412.24` | `53.46` |
| `attention-triton` | `5.9138` | `2.5425` | `4146.10` | `210.44` | `48.78` |
| `p20-triton` | `5.8523` | `2.2922` | `3667.73` | `244.45` | `50.06` |

## Read

- The attention anchors match exactly on final loss: `2.5425` in both
  environments.
- Attention-only is the fastest lane on this short `seq_len=32` starter
  surface.
- Native Mamba wins final loss: `2.2692`.
- `P20` is close on loss at `2.2922`, while using less CUDA memory and training
  much faster than native Mamba.
- `P20` captures most of the recurrent quality gain over attention-only while
  keeping a materially better systems profile than native Mamba.

This is a healthy starter result, not a final claim. The decisive next test is a
sequence-length ladder. That ladder is deferred until sponsored or grant-backed
compute is available.

## Artifact Lineage

Report paths:

- `attention-official`: `.runpod-local-logs/runpod-results/exp-e617d6296f6856e1/20260411T154942Z_a01/remote/artifacts/v3a-python-path1/20260411T154942Z_a01/attention-only/report.json`
- `mamba`: `.runpod-local-logs/runpod-results/exp-e617d6296f6856e1/20260411T160949Z_a02/remote/artifacts/v3a-python-path1/20260411T160949Z_a02/reference-ssm-hybrid-mamba3-siso-runtime/report.json`
- `attention-triton`: `.runpod-local-logs/runpod-results/exp-e617d6296f6856e1/20260411T161250Z_a03/remote/artifacts/v3a-python-path1/20260411T161250Z_a03/attention-only/report.json`
- `p20-triton`: `.runpod-local-logs/runpod-results/exp-e617d6296f6856e1/20260411T162924Z_a04/remote/artifacts/v3a-python-path1/20260411T162924Z_a04/primitive-hybrid-p2-0-runtime-scaled-projected-pre-norm-only-standard-block-diagonal-2/report.json`

Both Mamba Golf pods used by this run were stopped after their lane group
completed.

## Deferred Ladder

Do not spend additional RunPod credits on the ladder until grant-backed compute
is available.

Queued command:

```bash
SEQ_LENS="32 128 512 1024 2048" \
scripts/runpod-mamba-golf-path1-gauntlet.sh 42
```

If the single-seed curve is interesting, expand to a multi-seed ladder using the
same script and scorecard template.
