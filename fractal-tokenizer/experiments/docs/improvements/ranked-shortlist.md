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

Most recent completed trial: `Chunking / Model Packaging` on top of `NoveltyAware`.

- Stress input:
  - frontier before packaging: `3` tokens
  - packaged chunks: `1`
- Mixed-domain input:
  - frontier before packaging: `32` tokens
  - packaged chunks: `4`
- Round-trip: exact for both
- Fallback stats: `unknown=0`, `byte=0` for both

Result:

- `Chunking / Model Packaging` is lossless and deterministic on the current winning frontier. It packages the `NoveltyAware` frontier into ordered model-sized windows without changing the frontier itself.

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

### 1. Novelty-Aware Frontier

Status:

- `Tried`

Why it ranks first now:

- still the best frontier policy
- keeps repeated structure coarse without collapsing mixed-domain text
- best current basis for packaging and model-facing integration

Expected upside:

- best current compression/detail tradeoff while preserving exact round-trip

Expected failure mode:

- if pushed further, it can become too aggressive on mixed-domain inputs

Decision:

- remains the leading frontier candidate

## Recommended Trial Order

1. Model packaging integration
2. Downstream model-facing evaluation

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
