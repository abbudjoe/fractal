# Mamba Golf

Mamba Golf is a small, reproducible proving ground for efficient recurrent and
state-space sequence primitives.

The goal is simple:

> If Mamba-style models should win anywhere, they should win here.

This repository already contains a Path 1 benchmark surface that can compare:

- `A`: local causal attention blocks
- `A+M`: local causal attention plus native Mamba
- `A+P20`: local causal attention plus the current `P20` recurrent primitive

The first scaffold is intentionally modest. It is not a claim that `P20` has
beaten Mamba in general. It is a way to make the comparison reproducible,
extend it toward longer contexts, and invite the Mamba, SSM, Triton, CUDA, and
NVIDIA developer communities to sharpen the benchmark before treating any
result as definitive.

## Why This Exists

OpenAI Parameter Golf has shown that parameter sharing and recurrence can be
extremely competitive under tiny-model constraints. The top leaderboard entries
mostly use recurrence by looping transformer layers, while still preserving
attention as the exact token-interaction path.

That leaves an open question:

> Can a recurrent or state-space primitive win when the benchmark is built for
> the things recurrent models are supposed to be good at?

Those things include:

- fixed parameter or artifact budgets
- fixed memory budgets
- long-context scaling
- streaming or incremental inference
- throughput-sensitive training
- domains with long sequential structure

## Current Lanes

The initial Path 1 gauntlet compares:

- `attention-official`: attention-only lane in the official Mamba environment
- `mamba`: native Mamba reference lane in the official Mamba environment
- `attention-triton`: attention-only lane in the primitive Triton environment
- `p20-triton`: `P20` fast lane in the primitive Triton environment

The duplicated attention lane is deliberate. It gives us a cheap environment
drift check before comparing native Mamba and `P20`, which currently require
different CUDA dependency stacks.

## Run The Starter Gauntlet

```bash
scripts/runpod-mamba-golf-path1-gauntlet.sh 42
```

Useful overrides:

```bash
DRY_RUN=1 \
SEQ_LENS="32 64 128" \
GPU_ID="NVIDIA GeForce RTX 4090" \
scripts/runpod-mamba-golf-path1-gauntlet.sh 42 123 256
```

The starter benchmark defaults to the existing `cuda-faithful-small-v1` corpus.
For a real long-context claim, provide a larger JSONL corpus and run a sequence
length ladder:

```bash
BENCHMARK_PROFILE="" \
JSONL_TRAIN_PATH=/path/to/train.jsonl \
JSONL_EVAL_PATH=/path/to/eval.jsonl \
CORPUS_NAME=long-context-smoke-v1 \
SEQ_LENS="1024 2048 4096 8192" \
scripts/runpod-mamba-golf-path1-gauntlet.sh 42
```

## Community Gauntlet

Mamba and SSM folks, Triton/CUDA kernel people, and the NVIDIA developer
community:

We want the fair home-court benchmark for Mamba-style models.

If the current benchmark is flawed, help fix it. If native Mamba needs a better
configuration, show it. If another recurrent primitive wins, even better.

The clean target is:

- same data
- same tokenizer
- same parameter or artifact budget
- same training budget
- same hardware class
- reported loss, throughput, memory, and sequence-length scaling

The first benchmark lives here because this repo already has the `A`, `A+M`,
and `A+P20` control plane. The long-term goal is a small public gauntlet where
Mamba, `P20`, transformers, and new community contenders can all compete under
the constraints where recurrent models are supposed to shine.

## Related Docs

- [Mamba Golf gauntlet spec](docs/specs/mamba-golf-gauntlet.md)
- [Mamba Golf scorecard template](docs/specs/mamba-golf-scorecard-template.md)
- [P20 vs native Mamba five-seed scorecard](docs/specs/p20-vs-mamba-five-seed-head2head-scorecard.md)
