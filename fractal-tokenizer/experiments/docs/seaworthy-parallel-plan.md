# Seaworthy Parallel Plan

## Purpose

This document splits the next tokenizer-seaworthiness work into three parallel
tracks that can proceed independently without stepping on each other.

The tracks are:

1. Edge-case hardening
2. Batch/runtime contract completion
3. Embedding-bridge scaffolding

Each track is intentionally test-driven, narrowly scoped, and assigned a
disjoint write surface.

## Shared Rules

- Start with focused failing tests or contract tests.
- Keep `NoveltyAware` unchanged unless a failing test proves a bug.
- No track may widen scope into a new frontier-policy experiment.
- If a track needs a shared contract change outside its owned files, stop and
  hand it back for orchestration.

## Ownership

### Track A: Edge-case hardening

Owns:

- `fractal-tokenizer/src/tests.rs`
- tokenizer/model-facing files only if a failing regression proves a concrete bug

### Track B: Batch/runtime contract completion

Owns:

- `fractal-tokenizer/src/model_face/mod.rs`
- `fractal-tokenizer/src/model_face/batch.rs`
- `fractal-tokenizer/src/model_face/native.rs`
- `fractal-tokenizer/src/lib.rs`
- `fractal-tokenizer/src/tests.rs`

### Track C: Embedding-bridge scaffolding

Owns:

- `fractal-tokenizer/src/model_face/bridge.rs`
- `fractal-tokenizer/src/model_face/mod.rs`
- `fractal-tokenizer/src/model_face/traits.rs`
- `fractal-tokenizer/src/lib.rs`
- `fractal-tokenizer/src/tests.rs`

## Merge Order

Merge in this order unless a blocking contract conflict is found:

1. Track A
2. Track B
3. Track C

Track C should treat Track B's current batch contract as the target boundary and
should avoid rewriting batch semantics.

## Success Condition

We are ready to move toward real downstream evaluation when:

- edge-case regressions are covered and green
- batch/collation semantics are explicit enough for training/inference code
- embedding-bridge scaffolding exists as a typed contract without model-family coupling
