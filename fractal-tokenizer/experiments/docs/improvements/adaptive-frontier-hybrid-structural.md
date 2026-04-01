# Adaptive Frontier: Hybrid Structural

## What It Is

Combine several local signals to decide whether to emit a parent or recurse:

- reuse value
- novelty
- state coherence
- optional span-size guardrails

## How It Works

- compute a small deterministic stop/recurse score at each node
- emit when the parent is both stable and useful
- recurse when the children provide meaningfully better structure

## Why It Is A Good Candidate

- likely the highest-quality long-term frontier policy
- can balance compression and detail more intelligently than any single signal
- should handle repetition-heavy and mixed-domain text more gracefully

## Status

`Tried`

## Trial Outcome

- Stress: `NoveltyAware=3` tokens, `HybridStructural=1` token, exact round-trip
- Mixed-domain: `NoveltyAware=32` tokens, `HybridStructural=1` token, exact round-trip
- Fallback: `unknown=0`, `byte=0` for both
- Bottom line: the hybrid rule was lossless but too conservative, collapsing back to a single-root frontier on both benchmark inputs

## Risks

- easiest policy to overcomplicate
- the combined signal can over-constrain the frontier and collapse it to root emission
