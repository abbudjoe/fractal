# Ranked Tokenizer Improvement Shortlist

This document turns the current improvement catalog into an explicit ordered trial list.

The goal is to keep future context disciplined:

- one ranked candidate list
- one standalone note per candidate
- one place to record why a candidate is next
- one place to record whether a candidate should be delayed or retired

## Ranking Criteria

Candidates are ranked by:

1. alignment with the tokenizer-track thesis
2. likelihood of improving over `FinestKnown`
3. determinism and contract clarity
4. implementation risk
5. ability to isolate experimental signal cleanly

## Latest Trial Snapshot

Most recent completed trial: held-out local bakeoff rerun after clustered
structural induction over coarse state buckets, length buckets, and lexical
shape.

- `BAKEOFF_DOCUMENTS=120`
- `BAKEOFF_INDUCTION_DOCUMENTS=63`
- `BAKEOFF_EVALUATION_DOCUMENTS=57`
- `BAKEOFF_VERDICT=GREEN`
- `byte_fallback_docs=0`
- hard-gate failures: `0`

Result:

- held-out byte collapse remains fixed
- clustered induction produced real held-out prototype hits
- `full` mode on `p1_fractal_hybrid_dyn-state-norm_v2` now shows:
  - `exact_motif_hit_docs=1`
  - `prototype_hit_docs=5`
  - `lexical_only_docs=42`
- `motif-only` mode still shows:
  - `exact_motif_hit_docs=1`
  - `prototype_hit_docs=5`
  - `lexical_only_docs=52`
- `jsonl.signals` remains a strong held-out win
- `code.rust`, `code.swift`, and `docs.spec` are still below native-tokenizer
  parity

New read:

- clustered induction is a real but modest lift
- the active bottleneck is now the primary motif identity surface
- the next disciplined step is to make prototype identity primary before
  considering any neighborhood matching

## Current Baseline

Current tried baselines:

- `GreedyKnown`
  - good contract proof
  - too coarse
- `FinestKnown`
  - current default model-facing policy
  - fixes root collapse
  - still too uniform because it resolves to a `32`-token frontier on both standard benchmark inputs
- `NoveltyAware` (trial)
  - deterministic and lossless
  - improved stress frontier size (`3` vs `32` tokens against `FinestKnown`)
  - matched `FinestKnown` on mixed-domain (`32` tokens)
  - meets promotion rule and is the leading adaptive-frontier candidate
- `Chunking / Model Packaging` (trial)
  - deterministic and lossless
  - stress: `3` frontier tokens -> `1` packaged chunk
  - mixed-domain: `32` frontier tokens -> `4` packaged chunks
  - validated packaging layer on top of the leading frontier
- `HybridStructural` (trial)
  - deterministic and lossless
  - regressed to a single-root frontier on both standard benchmark inputs
  - not promoted because it is too conservative to be useful
- `Budgeted` (trial)
  - deterministic and lossless
  - stress frontier (`16` tokens) is worse than `NoveltyAware`
  - mixed-domain frontier (`16` tokens) is finer than `NoveltyAware`
  - not promoted because it loses the stress-frontier race
- `SpanLengthAware` (trial)
  - deterministic and lossless
  - stress frontier (`8` tokens) is worse than `NoveltyAware`
  - mixed-domain frontier (`24` tokens) is finer than `NoveltyAware`
  - not promoted because it loses the stress-frontier race
- `ReuseAware` (trial)
  - deterministic and lossless
  - improved stress frontier size (`9` vs `32` tokens against `FinestKnown`)
  - matched `FinestKnown` on mixed-domain (`32` tokens)
  - strong earlier comparator, now behind NoveltyAware
- `StateAware` (trial)
  - deterministic and lossless
  - regressed to root-level emission (`1` token) on both standard benchmark inputs
  - not promoted

The ranking below applies to **next** frontier candidates. Packaging is now treated as validated integration work rather than a frontier-policy candidate.

## Ranked Candidates

### 1. Prototype-Primary Identity

Status:

- `Active`

Why it ranks first now:

- clustered induction (`#2`) produced real prototype hits
- but prototypes are still acting like a secondary rescue tier
- held-out code/docs did not materially move while lexical-only behavior stayed
  dominant

Expected upside:

- prototype hits become the main structural signal instead of a side effect
- held-out code/docs may finally move above the current ceiling
- the next bakeoff will tell us whether this control-plane line still has real
  headroom

Expected failure mode:

- prototype hits rise a little but code/docs remain flat
- which would strongly suggest the kill criterion is approaching

Decision:

- move here now

### 2. Primitive Comparison Pivot

Status:

- `Active`

Why it ranks second now:

- the one allowed serious rescue pass for `p1` has now been tried
- clustered induction still did not materially move the held-out code/docs
  outcome
- the experiment pipeline is strong enough to compare primitives honestly

Expected upside:

- we stop guessing whether `p1` is the right primitive
- we find out whether another primitive fits code/docs better without adding
  more local `p1` heuristics

Expected failure mode:

- no better primitive emerges, which would force a broader rethink

Decision:

- move here now

### 3. Boundary-Aware Split For `p1`

Status:

- `Tried`

Why it ranks second now:

- it was the strongest remaining structural rescue hypothesis for code/docs
- it directly targeted span-geometry mismatch instead of more fallback tuning

Expected upside:

- code/docs would recover materially more structural hits
- lexical fallback would stop fully dominating code/docs

Expected failure mode:

- no material change on held-out code/docs

Decision:

- tried and not promoted
- do not spend more cycles here before primitive comparison

### 4. Novelty-Aware Frontier

Status:

- `Tried`

Why it still matters:

- current best frontier policy on benchmark inputs
- remains the right baseline frontier while OOV behavior is hardened

Decision:

- keep as the current frontier baseline, but do not spend the next cycle on
  new frontier-policy candidates

## Recommended Trial Order

1. Typed lexical fallback above bytes
2. Compositional recurring-submotif vocab
3. Clustered structural induction
4. Prototype-primary identity
5. Held-out local bakeoff rerun
6. If still weak on code/docs, primitive comparison pivot
7. Hybrid bakeoff implementation

## Promotion Rule

A candidate should only move ahead of the current default if it:

- preserves exact round-trip
- preserves deterministic encoding
- improves over `FinestKnown` on at least one target dimension without regressing badly on the others
- keeps false-positive reuse near zero on mixed-domain input

## Outcome Logging Rule

When a candidate is tried, update:

1. this shortlist
2. the candidate’s standalone note
3. the tracker or experiment report if the result changes the current default
4. the held-out decision log if the current bottleneck changes again
