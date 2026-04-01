# Adaptive Frontier: Span-Length-Aware

## What It Is

Use span size as a simple prior: long spans are allowed to stay coarse more often, while short or dense spans are forced finer.

## How It Works

- inspect byte span length at each node
- stop earlier for large repetitive spans
- recurse sooner for short spans that may hide detail

## Why It Is A Good Candidate

- simple
- deterministic
- easy to ablate
- could serve as a guardrail for more advanced adaptive policies

## Status

`Tried`

## Trial Outcome

Stress input: `SpanLengthAware` emitted `8` tokens versus `3` for `NoveltyAware`, while preserving exact round-trip and zero unknown/byte fallback.

Mixed-domain input: `SpanLengthAware` emitted `24` tokens versus `32` for `NoveltyAware`, again with exact round-trip and zero unknown/byte fallback.

Outcome: useful structural comparator, but not a promotion over `NoveltyAware` because it loses the stress-frontier race.

## Risks

- too heuristic if used as the main policy
- may reward span size rather than structural usefulness
