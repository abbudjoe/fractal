# Adaptive Signature Granularity Spec

## Goal

Keep the held-out structural gains from typed state-signature prototypes while
reducing false-positive overcollapse without falling back to the blunt rejection
used by prototype precision guardrails.

## Why This Is Next

Prototype precision guardrails solved the `jsonl.signals` regression only by
dropping almost all of the new prototype hits:

- `prototype_hit_docs=29 -> 3`
- `lexical_only_docs=27 -> 42`
- `code.rust`, `code.swift`, and `docs.spec` fell back to the old floor

That means the next honest move is not stricter rejection. It is finer
representation for risky prototype clusters.

## Proposed Change

Introduce an explicit `PrototypeGranularityMode`:

- `Coarse`
- `Adaptive`

In `Adaptive` mode:

1. induction starts from the existing coarse typed state-signature cluster
2. if the coarse cluster passes admission, keep it
3. if the coarse cluster is too broad, refine it into finer state-signature
   clusters instead of rejecting it outright

The refined clusters should use:

- exact `state_bin`
- the existing prefix bins
- new suffix bins

The stable default remains `Coarse`; `Adaptive` is an explicit experiment mode.

## TDD Plan

### Step 1. Focused Refinement Regression

Add a regression proving:

- a broad coarse cluster is rejected in `Coarse` mode
- the same source records produce multiple refined prototype clusters in
  `Adaptive` mode

### Step 2. Integration Checks

Keep:

- `faceoff_`
- `model_face_`
- `local_bakeoff` parser tests

green.

### Step 3. Bakeoff Gate

Run the held-out local bakeoff for `p1_fractal_hybrid_dyn-state-norm_v2` with:

- `--prototype-granularity adaptive`

and compare it directly to the explicit coarse baseline.

## Success Criteria

This candidate passes only if it:

- preserves a meaningful share of the state-signature recall lift
- materially reduces structured overcollapse relative to the raw
  state-signature pass
- keeps `docs.spec`, `code.rust`, and `code.swift` above the coarse baseline
- stays clear of the non-log overcollapse gate

## Failure Criteria

This candidate fails if it still trips the held-out overcollapse gate, even if
it is better than the coarse baseline.

## Outcome

Status:

- `Tried`

Focused regression result:

- broad short-span clusters are rejected in `Coarse` mode
- the same source records refine into multiple `fine::` prototype clusters in
  `Adaptive` mode

Held-out local bakeoff result for `p1_fractal_hybrid_dyn-state-norm_v2`:

- `coarse`
  - `prototype_hit_docs=3`
  - `lexical_only_docs=42`
  - `code.rust=0.81`
  - `code.swift=0.90`
  - `docs.spec=0.76`
  - `jsonl.signals=3.45`
  - verdict: `GREEN`
- `adaptive`
  - `prototype_hit_docs=19`
  - `lexical_only_docs=31`
  - `code.rust=0.83`
  - `code.swift=0.91`
  - `docs.spec=0.78`
  - `jsonl.signals=7.60`
  - verdict: `YELLOW`

Read:

- adaptive refinement is a real precision-by-representation improvement over
  the coarse baseline
- it recovers meaningful prototype recall without returning to the catastrophic
  `jsonl.signals=113.26` failure
- but it still trips the non-log overcollapse gate and does not move code/docs
  far enough

Decision:

- keep `Adaptive` available as an explicit experiment mode
- keep `Coarse` as the stable default
- gate failed
- do not proceed to candidate 3 from this line
