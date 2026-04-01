# State-Signature Prototype Induction Spec

## Goal

Replace coarse prototype clustering based on:

- one state bucket
- one length bucket
- lexical shape

with a richer **typed state-signature** surface carried directly on
`TokenRecord`.

This experiment is meant to answer a narrower question than neighborhood
matching:

- if prototype induction uses a better exact signature surface,
  can held-out structural reuse rise materially without relaxing matching?

## Why This Is Next

Previous rescue passes established:

- typed lexical fallback fixed byte collapse
- clustered induction over coarse buckets produced a small structural lift
- prototype-primary identity did not materially improve held-out code/docs

The control-plane diagnosis pointed to a deeper hidden contract:

- primitive state was being reduced to a string token
- vocab induction then parsed structural identity back out of that string
- prototype clustering still depended heavily on lexical shape

So the next candidate is to fix the contract surface itself.

## Proposed Change

### 1. Add Typed State Signature

Each `TokenRecord` should carry a typed `StateSignature` derived from the
primitive readout.

Minimum fields:

- `state_bin`
- `norm_bin`
- `mean_abs_bin`
- `prefix_bins`

This keeps the exact digest token available for legacy paths, but stops the
prototype layer from reconstructing identity by string parsing.

### 2. Prototype Clustering Uses Typed Signature

Prototype cluster keys should be built from:

- `depth`
- coarse length bucket
- `StateSignature`

They should **not** include lexical shape in the primary key for this
experiment.

### 3. Matching Stays Exact

Held-out matching remains conservative:

- exact motif
- exact prototype membership
- recurse
- typed lexical fallback

No approximate neighborhood matching is introduced here.

## TDD Plan

### Step 1. Contract Regression

Add a focused regression that proves:

- two records with the same typed state signature
- but different lexical shapes

still induce one shared prototype cluster.

### Step 2. Integration Checks

Keep the existing faceoff and model-facing suites green.

### Step 3. Held-Out Bakeoff

Rerun the held-out local bakeoff on the stable legacy path for
`p1_fractal_hybrid_dyn-state-norm_v2`.

## Success Criteria

This experiment is a success if it materially improves the honest held-out
legacy bakeoff:

- `prototype_hit_docs` rises materially above `5`
- `lexical_only_docs` drops materially below `42`
- `code.rust`, `code.swift`, and `docs.spec` all move upward
- hard gates remain perfect

## Failure Criteria

The experiment fails if:

- prototype hits stay near the previous floor
- lexical-only behavior remains dominant
- code/docs remain flat

It also fails if it only improves by causing obvious false-positive
overcollapse on non-log buckets.

## Outcome

Status:

- `Tried`

Focused regression result:

- typed state-signature clustering now ignores lexical shape when the
  underlying state signature is the same

Held-out bakeoff result on the stable legacy path:

- `exact_motif_hit_docs=1`
- `prototype_hit_docs=29`
- `lexical_only_docs=27`
- `code.rust=0.83`
- `code.swift=0.93`
- `docs.spec=0.92`
- `jsonl.signals=113.26`
- `logs.operational_mixed=1.07`
- verdict: `YELLOW`

Motif-only diagnostic:

- `prototype_hit_docs=29`
- `lexical_only_docs=28`
- `jsonl.signals=27.45`
- verdict: `YELLOW`

Read:

- this is the first control-plane pivot that materially increased held-out
  structural reuse
- it also improved held-out docs/code modestly
- but it overcollapsed structured JSONL badly enough to fail the scorecard

Decision:

- keep the typed state-signature surface
- do not call it seaworthy
- if this line continues, the next move must be a precision / anti-overcollapse
  guardrail, not a looser matching step
