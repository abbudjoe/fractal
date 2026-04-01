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

Most recent completed trial: prototype emission gate on top of the adaptive
state-signature prototype surface.

- `BAKEOFF_DOCUMENTS=120`
- `BAKEOFF_INDUCTION_DOCUMENTS=63`
- `BAKEOFF_EVALUATION_DOCUMENTS=57`
- `BAKEOFF_VERDICT=YELLOW`
- `byte_fallback_docs=0`
- hard-gate failures: `0`

Result:

- the focused regression proved selective runtime emission works in isolation
- but the held-out bakeoff was identical between `direct` and `selective`
- `full` mode on `p1_fractal_hybrid_dyn-state-norm_v2` stayed at:
  - `exact_motif_hit_docs=1`
  - `prototype_hit_docs=19`
  - `lexical_only_docs=31`
  - `jsonl.signals=7.60`
  - `suspicious_nonlog_overcollapse_docs=10`

New read:

- runtime prototype emission is not the active bottleneck
- the remaining ceiling is upstream of emission
- another emission-side heuristic is not justified from this line

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

### 1. Primitive Comparison Pivot

Status:

- `Active`

Why it ranks first now:

- the admission -> granularity -> emission rescue line is now exhausted
- the pipeline is strong enough to compare primitives against the same held-out
  control-plane contract
- if no better primitive emerges, the result strengthens the case for a
  broader tokenizer-architecture pivot

Expected upside:

- we stop guessing whether the current held-out ceiling is specific to `p1`
- we find out whether another primitive family breaks out under the same honest
  bakeoff

Expected failure mode:

- no better primitive emerges, which would confirm that the current bottleneck
  is the tokenizer control plane or a deeper architectural limit

Decision:

- keep ready as the next branch after the emission no-op

### 2. Prototype Precision Guardrails

Status:

- `Tried`

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

- tried and not promoted
- candidate 2 was the justified replacement attempt after this failure

### 3. Targeted Prototype Precision

Status:

- `Tried`

Why it ranks second now:

- broad admission guardrails removed the JSONL regression
- but they also erased almost all of the structural gains
- the next honest move, if we stay on this line, is a narrower precision
  function instead of a blunt cutoff

Observed result:

- adaptive refinement improved over the coarse baseline:
  - `prototype_hit_docs=3 -> 19`
  - `lexical_only_docs=42 -> 31`
  - `code.rust=0.81 -> 0.83`
  - `code.swift=0.90 -> 0.91`
  - `docs.spec=0.76 -> 0.78`
- but it still failed the gate:
  - `jsonl.signals=7.60`
  - verdict: `YELLOW`

Decision:

- tried and not promoted
- keep adaptive granularity available as an explicit experiment mode
- keep coarse mode as the stable default
- candidate 3 was later tried as a runtime emission gate and was a field no-op

### 4. Prototype Emission Gate

Status:

- `Tried`

Why it mattered:

- adaptive granularity suggested one last local question:
  are prototypes over-emitted at runtime even when induction is already good
  enough?
- the experiment tested that directly without changing induction or identity

Observed result:

- focused regression: `Selective` recurses to structural children correctly
- held-out bakeoff: `Selective` and `Direct` were identical
- no metric moved:
  - `prototype_hit_docs=19`
  - `lexical_only_docs=31`
  - `code.rust=0.83`
  - `code.swift=0.91`
  - `docs.spec=0.78`
  - `jsonl.signals=7.60`

Decision:

- tried and not promoted
- revert the runtime experiment from code and keep only the logged result
- treat this as evidence that the remaining ceiling is upstream of emission

### 5. State-Signature Prototype Induction

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

### 6. Boundary-Aware Split For `p1`

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

### 6. Novelty-Aware Frontier

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
7. Targeted prototype precision
8. If still weak, primitive comparison pivot
9. If no primitive breaks out, broader tokenizer-architecture pivot

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
