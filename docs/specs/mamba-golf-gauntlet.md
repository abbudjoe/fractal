# Mamba Golf Gauntlet

Mamba Golf is the public-facing proving ground for tiny, efficient recurrent
and state-space sequence primitives.

The first implementation reuses the existing Python Path 1 benchmark surface.
That keeps the benchmark grounded in code we already trust while making the
comparison explicit enough for other developers to reproduce or challenge.

## Purpose

The benchmark should answer a narrower question than Parameter Golf:

> Under constraints where recurrent and SSM-style models should be strong, can
> `P20` beat native Mamba, attention-only, or another community contender?

This is not a replacement for OpenAI Parameter Golf. Parameter Golf rewards the
full tiny-model system: tokenizer, quantization, compression, test-time
training, optimizer, recurrence, and runtime. Mamba Golf isolates the recurrent
primitive question first, then can grow toward a full artifact-size challenge.

## Starter Surface

The starter surface is Path 1:

- local causal attention blocks
- native Mamba reference blocks
- `P20` recurrent primitive blocks
- shared corpus, seed, dtype, shape, and schedule controls

Default benchmark:

- benchmark profile: `cuda-faithful-small-v1`
- dtype: `bf16`
- model seed: caller-provided
- sequence length: `32` unless `SEQ_LENS` overrides it
- report fields: loss, train throughput, eval throughput, and peak CUDA memory

Default lanes:

| Lane | Env | Variant | Runtime |
| --- | --- | --- | --- |
| `attention-official` | `official-mamba3` | `attention-only` | PyTorch SDPA |
| `mamba` | `official-mamba3` | `reference-ssm-hybrid` | native Mamba |
| `attention-triton` | `primitive-triton` | `attention-only` | PyTorch SDPA |
| `p20-triton` | `primitive-triton` | `primitive-hybrid` | `P20` Triton scan |

The duplicated attention lane is intentional. Native Mamba and `P20` currently
require different CUDA dependency stacks. Running attention in both environments
surfaces environment drift before we compare the recurrent lanes.

## Current P20 Contract

The initial `P20` contender is the frozen fast lane:

- primitive profile: `p2-0`
- execution profile: `runtime`
- residual profile: `scaled`
- readout profile: `projected`
- norm profile: `pre-norm-only`
- wrapper profile: `standard`
- state transform: `block-diagonal-2`
- runtime backend: `triton`

This is the same family used in the frozen five-seed `P20` vs native Mamba
scorecard.

## Run Command

Single seed:

```bash
scripts/runpod-mamba-golf-path1-gauntlet.sh 42
```

Seed and sequence ladder:

```bash
DRY_RUN=1 \
SEQ_LENS="32 64 128" \
scripts/runpod-mamba-golf-path1-gauntlet.sh 42 123 256
```

Long-context custom corpus:

```bash
BENCHMARK_PROFILE="" \
JSONL_TRAIN_PATH=/path/to/train.jsonl \
JSONL_EVAL_PATH=/path/to/eval.jsonl \
CORPUS_NAME=long-context-smoke-v1 \
SEQ_LENS="1024 2048 4096 8192" \
scripts/runpod-mamba-golf-path1-gauntlet.sh 42
```

## Like-for-Like Rules

A result should only be treated as a real head-to-head if it reports:

- exact commit SHA
- lane command arguments
- seed set
- corpus identity
- tokenizer identity, if applicable
- model shape and layer schedule
- dtype
- CUDA dependency environment
- loss
- train throughput
- eval throughput
- peak CUDA memory
- failure point for any sequence-length ladder

The benchmark must not hide mismatched contracts. If two lanes require different
environments, that difference belongs in the scorecard. If one lane uses a
custom kernel and another uses a reference implementation, that also belongs in
the scorecard.

## Phases

Phase 0: Path 1 starter gauntlet

- reproduce `A`, native Mamba, and `P20` on the existing small benchmark
- report both attention environment anchors
- freeze at least three seeds

Phase 1: long-context scaling curve

- run the same lanes at increasing sequence lengths
- record loss, tokens/sec, memory, and failures
- use a corpus large enough to make long contexts meaningful

Phase 2: streaming and incremental inference

- add a streaming update/decode benchmark
- report steady-state tokens/sec and recurrent state memory
- compare against attention KV-cache behavior

Phase 3: Mamba-native domains

- add at least one non-language long-sequence task
- candidate domains: DNA/genomics, audio, time series, event streams
- keep the same reporting contract

Phase 4: public challenge surface

- promote the gauntlet into a clean public README
- accept external contender lanes
- require reproducible scripts and result manifests

## Community Challenge Text

Mamba and SSM folks, Triton/CUDA kernel people, and the NVIDIA developer
community:

We want the fair home-court benchmark for Mamba-style models.

If Mamba should win anywhere, it should win under fixed memory, longer context,
streaming-friendly inference, and matched training constraints. This repo now
has a starter gauntlet with attention, native Mamba, and `P20` lanes. Help make
the benchmark fair, then beat it.

The benchmark is successful even if `P20` loses, provided we learn which
primitive actually wins under the right constraints.

## Non-Claims

The starter gauntlet does not prove:

- `P20` beats Mamba in general
- Mamba is weak
- Parameter Golf should replace transformer recurrence with SSM recurrence
- the current small Path 1 corpus is a long-context benchmark

It only gives us a reproducible place to start.
