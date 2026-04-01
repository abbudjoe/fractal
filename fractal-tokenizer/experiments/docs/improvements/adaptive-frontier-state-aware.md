# Adaptive Frontier: State-Aware

## What It Is

Use the primitive’s own recursive state to decide whether a node is coherent enough to emit or should be opened up into children.

## How It Works

- evaluate parent state coherence using state norm, readout stability, or parent-vs-child distance
- emit when the parent state looks stable and self-consistent
- recurse when the state suggests unresolved internal structure

## Why It Is A Good Candidate

- keeps the control signal inside the recursive rule
- extends the same self-regulating design that made `v2` succeed
- avoids relying on external heuristics too early

## Status

`Tried`

## Trial Outcome

State-aware adaptive frontier was implemented and compared directly against `FinestKnown` on the standard benchmark inputs.

Observed side-by-side results:

- Stress input (`stress-20x-repetition`)
  - `FinestKnown`: `32` tokens, `61.66` chars/token, fallback `motif_hits:32 unknown:0 recursed:31 byte:0`
  - `StateAware`: `1` token, `1973.00` chars/token, fallback `motif_hits:1 unknown:0 recursed:0 byte:0`
- Mixed-domain input
  - `FinestKnown`: `32` tokens, `22.62` chars/token, fallback `motif_hits:32 unknown:0 recursed:31 byte:0`
  - `StateAware`: `1` token, `724.00` chars/token, fallback `motif_hits:1 unknown:0 recursed:0 byte:0`
- Round-trip remained exact for both policies.

Interpretation:

- the current state-aware rule is deterministic and lossless
- it is too conservative and regresses to root-level emission
- it does not improve over `FinestKnown` on frontier usefulness

## Success Signal

- fewer tokens than `FinestKnown` on repetition-heavy text
- no false-positive reuse on mixed-domain text
- deterministic and lossless behavior maintained

## Decision

Do not promote this version. Keep `FinestKnown` as default and move to the next ranked candidate (`ReuseAware`).
