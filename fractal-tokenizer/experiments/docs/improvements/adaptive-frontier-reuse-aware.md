# Adaptive Frontier: Reuse-Aware

## What It Is

Stop at a parent motif when it captures structure that is reused enough to be valuable as a single unit; recurse when the children carry the more reusable structure.

## How It Works

- estimate reuse value for the parent motif
- estimate reuse value for the child frontier
- emit the parent when reuse is stronger or more stable at the parent level
- recurse when reuse signal increases at finer depth

## Why It Is A Good Candidate

- aligns directly with the tokenizer thesis
- should keep repeated phrase blocks coarse
- should avoid forcing every region to the same leaf depth

## Status

`Tried`

## Trial Outcome

Observed in `faceoff_reuse_aware_vs_finest_known_side_by_side`:

- Stress input (`stress-20x-repetition`):
  - `FinestKnown`: `32` tokens, `61.66` avg chars/token
  - `ReuseAware`: `9` tokens, `219.22` avg chars/token
- Mixed-domain input:
  - `FinestKnown`: `32` tokens, `22.62` avg chars/token
  - `ReuseAware`: `32` tokens, `22.62` avg chars/token
- Round-trip: exact for both policies
- Fallback: `unknown=0`, `byte=0` for both policies

Result: `ReuseAware` wins on repetition-heavy input and ties on mixed-domain without introducing fallback regressions.

## Success Signal

- repetition-heavy input emits fewer tokens than `FinestKnown`
- mixed-domain input stays cautious and does not invent false-positive reuse

Observed: success signal met.
