# Prototype Emission Gate Spec

## Goal

Test whether the remaining precision problem lives at runtime emission rather
than induction.

The hypothesis is:

- adaptive prototype induction has already recovered useful held-out prototypes
- some of those prototype hits may still be too eager to emit
- a selective runtime gate could recurse to children when the child frontier is
  already structurally recoverable

## Why This Is Next

Candidate 2 proved that adaptive refinement was real but still not selective
enough:

- `prototype_hit_docs=19`
- `lexical_only_docs=31`
- `code.rust=0.83`
- `code.swift=0.91`
- `docs.spec=0.78`
- `jsonl.signals=7.60`
- verdict: `YELLOW`

That leaves one last tightly scoped control-plane question:

- are prototypes over-emitted at runtime,
- or is the bottleneck still upstream in induction / identity?

## Proposed Change

Introduce a typed runtime mode:

- `PrototypeEmissionMode::Direct`
- `PrototypeEmissionMode::Selective`

In `Selective` mode:

- when a prototype hit is available
- and the child frontier fully covers the parent
- and every child is itself structurally recoverable (`exact motif` or
  `prototype`)
- recurse to children instead of emitting the parent prototype

The stable default remains `Direct`.

## TDD Plan

### Step 1. Focused Regression

Add a manual parent/children summary proving:

- `Direct` emits the parent prototype
- `Selective` recurses and emits the two child prototypes

### Step 2. Integration Checks

Keep:

- `faceoff_`
- `model_face_`
- `local_bakeoff` parser tests

green.

### Step 3. Bakeoff Gate

Run the held-out local bakeoff with:

- `--prototype-granularity adaptive --prototype-emission direct`
- `--prototype-granularity adaptive --prototype-emission selective`

## Success Criteria

This candidate passes only if `Selective` materially improves over `Direct` on
the held-out gate:

- fewer suspicious non-log overcollapse docs
- lower `jsonl.signals`
- without collapsing prototype recall back to the coarse floor

## Failure Criteria

This candidate fails if the held-out result is materially unchanged, even if the
focused regression proves the feature works in principle.

## Outcome

Status:

- `Tried`

Focused regression result:

- `Direct` emits one parent prototype
- `Selective` recurses and emits two child prototypes

Held-out local bakeoff result for `p1_fractal_hybrid_dyn-state-norm_v2` with
adaptive granularity:

- `direct`
  - `prototype_hit_docs=19`
  - `lexical_only_docs=31`
  - `code.rust=0.83`
  - `code.swift=0.91`
  - `docs.spec=0.78`
  - `jsonl.signals=7.60`
  - verdict: `YELLOW`
- `selective`
  - `prototype_hit_docs=19`
  - `lexical_only_docs=31`
  - `code.rust=0.83`
  - `code.swift=0.91`
  - `docs.spec=0.78`
  - `jsonl.signals=7.60`
  - verdict: `YELLOW`

Read:

- the selective emission gate is technically valid
- but it is a field no-op on the held-out corpus
- that means runtime prototype emission is not the active bottleneck

Decision:

- do not promote `Selective`
- do not keep pushing this rescue line based on runtime emission heuristics
- treat this as evidence that the remaining ceiling is upstream of emission
- keep this document as the experiment record and revert the runtime code path
