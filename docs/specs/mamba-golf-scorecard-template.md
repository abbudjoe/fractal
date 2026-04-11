# Mamba Golf Scorecard Template

Use this template when freezing a Mamba Golf run.

## Summary

- date:
- commit:
- benchmark:
- corpus:
- tokenizer:
- seed set:
- sequence lengths:
- hardware:
- dtype:
- run script:

One-sentence read:

> TODO

## Contract

Common settings:

- model shape:
- layer schedule:
- optimizer:
- training budget:
- eval budget:
- batch size:
- window stride:

Lane contracts:

| Lane | Env | Variant | Runtime | Notes |
| --- | --- | --- | --- | --- |
| `attention-official` | `official-mamba3` | `attention-only` | PyTorch SDPA | environment anchor |
| `mamba` | `official-mamba3` | `reference-ssm-hybrid` | native Mamba | reference SSM lane |
| `attention-triton` | `primitive-triton` | `attention-only` | PyTorch SDPA | environment anchor |
| `p20-triton` | `primitive-triton` | `primitive-hybrid` | `P20` Triton scan | recurrent primitive lane |

## Results

Per-seed table:

| Seed | Seq Len | Lane | Loss | Train tok/s | Eval tok/s | CUDA MB | Status |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| TODO | TODO | `attention-official` | TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | `mamba` | TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | `attention-triton` | TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | `p20-triton` | TODO | TODO | TODO | TODO | TODO |

Aggregate table:

| Seq Len | Lane | Mean Loss | Mean Train tok/s | Mean Eval tok/s | Peak CUDA MB | Completed Seeds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| TODO | `attention-official` | TODO | TODO | TODO | TODO | TODO |
| TODO | `mamba` | TODO | TODO | TODO | TODO | TODO |
| TODO | `attention-triton` | TODO | TODO | TODO | TODO | TODO |
| TODO | `p20-triton` | TODO | TODO | TODO | TODO | TODO |

## Read

Answer these directly:

- Which lane wins loss?
- Which lane wins training throughput?
- Which lane wins eval throughput?
- Which lane wins memory?
- Does either attention anchor show environment drift?
- Does the recurrent result improve as sequence length increases?
- Did any lane fail or hit a memory/runtime wall?

## Artifact Lineage

List report paths:

- `attention-official`:
- `mamba`:
- `attention-triton`:
- `p20-triton`:

## Verdict

Choose one:

- promote `P20`
- native Mamba remains the better recurrent lane
- attention remains dominant on this surface
- benchmark is inconclusive and needs a sharper contract

Reason:

> TODO
