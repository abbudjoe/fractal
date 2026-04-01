# Prototype Precision Guardrails Spec

## Goal

Keep the held-out gains from state-signature prototype induction while reducing
false-positive overcollapse on structured non-log text, especially
`jsonl.signals`.

## Why This Is Next

State-signature prototype induction was the first control-plane change that
materially improved held-out structural reuse:

- `prototype_hit_docs=29`
- `lexical_only_docs=27`
- `docs.spec`, `code.rust`, and `code.swift` all moved upward

But it also overcollapsed `jsonl.signals` badly enough to force a `YELLOW`
verdict.

That means the next move is not looser matching.
It is tighter admission.

## Proposed Change

Add a dynamic prototype admission policy that evaluates whether a cluster is
selective enough to become a reusable prototype.

For this first pass, the policy uses:

- distinct-text density
- repetition surplus
- span-size bucket

The policy is stricter on short spans and slightly looser on larger spans.

## TDD Plan

### Step 1. Positive / Negative Cluster Tests

Add focused regressions proving:

- a short cluster where every occurrence is text-unique is rejected
- a short cluster with real repeated support is kept

### Step 2. Integration Checks

Keep:

- `faceoff_`
- `model_face_`
- `local_bakeoff` parser tests

green.

### Step 3. Held-Out Bakeoff Gate

Run the held-out local bakeoff on the stable legacy path for
`p1_fractal_hybrid_dyn-state-norm_v2`.

## Success Criteria

This candidate passes the gate only if it:

- materially reduces structured overcollapse
- while preserving most of the new structural lift

Minimum bar:

- `jsonl.signals` comes down to a sane range
- `prototype_hit_docs` stays materially above the old pre-state-signature floor
- `docs.spec`, `code.rust`, and `code.swift` remain above the old pre-state-signature result
- hard gates remain perfect

## Failure Criteria

This candidate fails the gate if it removes the overcollapse by collapsing back
to the old lexical-only regime.

## Outcome

Status:

- `Tried`

Focused regression result:

- all-unique short clusters are rejected
- repeated short clusters are retained

Held-out local bakeoff result on the stable legacy path:

- `exact_motif_hit_docs=1`
- `prototype_hit_docs=3`
- `lexical_only_docs=42`
- `code.rust=0.81`
- `code.swift=0.90`
- `docs.spec=0.76`
- `jsonl.signals=3.45`
- verdict: `GREEN`

Read:

- the guardrail eliminated the JSONL overcollapse
- but it also erased almost all of the structural gains from state-signature
  induction
- the system fell back to essentially the old pre-guardrail profile

Decision:

- gate failed
- do not proceed to candidate 2 yet
- if this line continues, the next move should be a more targeted precision
  function, not this coarse admission rule
