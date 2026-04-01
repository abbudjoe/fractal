# Adaptive Frontier: Budgeted

## What It Is

Use a target token budget to decide how far to descend, so the emitted frontier fits a desired sequence size.

## How It Works

- pick a target token count or frontier budget
- recurse until the document meets the budget
- stop earlier in low-information regions and later in high-information regions

## Why It Is A Good Candidate

- directly useful for model training and inference
- gives explicit control over sequence length
- can help bridge tokenizer behavior to context-window constraints

## Status

`Tried`

## Trial Outcome

Stress input: `Budgeted` emitted `16` tokens versus `3` for `NoveltyAware`, while preserving exact round-trip and zero unknown/byte fallback.

Mixed-domain input: `Budgeted` emitted `16` tokens versus `32` for `NoveltyAware`, again with exact round-trip and zero unknown/byte fallback.

## Risks

- can turn the tokenizer into a packaging heuristic too early
- may hide a weak frontier policy instead of improving it

## Decision Note

Useful as a budget-shaping comparator, but it loses the stress-frontier race to `NoveltyAware` even though it is finer on mixed-domain text.
