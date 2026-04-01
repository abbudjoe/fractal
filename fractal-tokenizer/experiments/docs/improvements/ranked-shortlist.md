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

Most recent completed trial: held-out local bakeoff scoring on the current
model-facing stack.

- `BAKEOFF_DOCUMENTS=120`
- `BAKEOFF_INDUCTION_DOCUMENTS=64`
- `BAKEOFF_EVALUATION_DOCUMENTS=56`
- `BAKEOFF_VERDICT=YELLOW`
- `byte_fallback_docs=56`
- hard-gate failures: `0`

Result:

- the earlier local-only `GREEN` does not survive held-out evaluation
- the active bottleneck is now held-out OOV behavior, not frontier selection
- the next ranked work should target the OOV contract directly

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

### 1. Typed Lexical Fallback Above Bytes

Status:

- `Proposed`

Why it ranks first now:

- fastest path to stopping total held-out byte collapse
- preserves meaningful local structure for novel text
- cleanly complements the existing model-facing contract

Expected upside:

- held-out docs stop degrading immediately to raw bytes
- model-facing batches retain words, identifiers, numbers, punctuation, and
  whitespace as typed units

Expected failure mode:

- lexical classes are either too coarse or too fine

Decision:

- implement as the first OOV hardening layer

### 2. Compositional Recurring-Submotif Vocab

Status:

- `Proposed`

Why it ranks second now:

- strongest structural fix for memorization-heavy induction
- should recover reusable descendant cover on held-out docs
- pairs naturally with typed lexical fallback

Expected upside:

- held-out docs reuse known internal structure instead of failing at the parent
  motif boundary

Expected failure mode:

- thresholds either over-admit giant memorized motifs or make the vocab too
  sparse

Decision:

- implement alongside lexical fallback as the structural recovery layer

### 3. Novelty-Aware Frontier

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
3. Held-out local bakeoff rerun
4. Hybrid bakeoff implementation

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
