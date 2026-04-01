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

Most recent completed trial: state-signature prototype induction on the stable
legacy control-plane path.

- `BAKEOFF_DOCUMENTS=120`
- `BAKEOFF_INDUCTION_DOCUMENTS=63`
- `BAKEOFF_EVALUATION_DOCUMENTS=57`
- `BAKEOFF_VERDICT=GREEN`
- `byte_fallback_docs=0`
- hard-gate failures: `0`

Result:

- held-out byte collapse remains fixed
- the typed state-signature surface materially increased held-out prototype
  hits
- `full` mode on `p1_fractal_hybrid_dyn-state-norm_v2` now shows:
  - `exact_motif_hit_docs=1`
  - `prototype_hit_docs=29`
  - `lexical_only_docs=27`
- `code.rust`, `code.swift`, and `docs.spec` all moved upward
- `jsonl.signals` overcollapsed badly enough to force a `YELLOW` verdict

New read:

- this is the first control-plane pass that produced a real held-out structural
  lift
- the active problem is no longer absence of prototype hits
- the active problem is prototype precision and false-positive overcollapse on
  structured non-log text

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

### 1. Prototype Precision Guardrails

Status:

- `Active`

Why it ranks first now:

- state-signature induction finally moved held-out structural reuse
- but it did so too aggressively on `jsonl.signals`
- the highest-value next move is to constrain false positives without giving
  back the new prototype hits

Expected upside:

- keep the gains on `docs.spec`, `code.rust`, and `code.swift`
- reduce structured JSONL overcollapse back into a sane range
- learn whether the current control-plane line can be made selective enough to
  survive the scorecard

Expected failure mode:

- guardrails kill the new prototype hits and drop us back to lexical-only
- or the false positives remain, showing this line is still too blunt

Decision:

- move here now

### 2. Primitive Comparison Pivot

Status:

- `Active`

Why it ranks second now:

- if precision guardrails cannot rescue the new state-signature line,
  primitive comparison is the next honest branch
- the pipeline is strong enough now to compare primitives against the same held-out
  control-plane contract

Expected upside:

- we stop guessing whether this ceiling is specific to `p1`
- we find out whether another primitive family breaks out under the same honest
  held-out bakeoff

Expected failure mode:

- no better primitive emerges, which would strengthen the case for a broader
  tokenizer-architecture pivot

Decision:

- keep ready as the next branch if the precision pass fails

### 3. State-Signature Prototype Induction

Status:

- `Tried`

Why it mattered:

- it replaced the hidden string-parsing contract with a typed state-signature
  surface on `TokenRecord`
- it is the first control-plane pass that materially improved held-out
  prototype reuse

Expected upside:

- more held-out structural hits
- less lexical-only behavior
- upward movement on docs/code

Observed failure mode:

- structured JSONL overcollapse became the dominant new regression

Decision:

- keep the typed state-signature surface
- do not treat it as seaworthy without precision guardrails

### 4. Boundary-Aware Split For `p1`

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

### 5. Novelty-Aware Frontier

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
5. State-signature prototype induction
6. Prototype precision guardrails
7. If still yellow, primitive comparison pivot
8. If no primitive breaks out, broader tokenizer-architecture pivot

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
