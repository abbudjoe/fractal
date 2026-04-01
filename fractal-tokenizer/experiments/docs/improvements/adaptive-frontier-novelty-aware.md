# Adaptive Frontier: Novelty-Aware

## What It Is

Stop at coarse motifs when the current span is familiar, and recurse when the span appears novel enough to need finer representation.

## How It Works

- compute a novelty score for the current motif relative to vocab or local motif history
- emit familiar motifs at the parent level
- descend when the parent is too novel or heterogeneous

## Why It Is A Good Candidate

- complements reuse-aware stopping
- should keep mixed-domain and unfamiliar content detailed
- could reduce over-coarsening on new or rare spans

## Status

`Tried`

## Trial Outcome

Stress input: `NoveltyAware` emitted `3` tokens versus `32` for `FinestKnown`, while preserving exact round-trip and zero unknown/byte fallback.

Mixed-domain input: `NoveltyAware` matched `FinestKnown` at `32` tokens, again with exact round-trip and zero unknown/byte fallback.

Outcome: strong win on repetition-heavy text with no mixed-domain regression.

## Success Signal

- repeated or boilerplate text stays coarse
- novel spans receive finer-grained tokens without breaking round-trip
