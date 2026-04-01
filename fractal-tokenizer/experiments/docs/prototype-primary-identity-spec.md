# Prototype-Primary Identity Spec

## Goal

Promote prototype clusters from a secondary recovery tier to the **primary
motif identity surface** for held-out encoding.

The purpose of this change is not to add fuzzier matching.
It is to test whether the current tokenizer can generalize better when its
model-facing reusable units are prototypes first, instead of exact text/digest
entries first.

## Why This Is Next

The previous two steps established:

1. typed lexical fallback prevents catastrophic byte collapse
2. clustered structural induction produces real held-out prototype hits

But the held-out local bakeoff still shows:

- code/docs remain below native-tokenizer parity
- lexical-only behavior is still dominant on most evaluation documents
- prototype recovery exists, but only as a small secondary effect

That means the remaining question is:

- does the tokenizer fail because prototype induction is too weak,
  or because prototype identity is not the primary runtime contract?

This spec isolates that question.

## Current Problem

Today the control plane still behaves roughly like this:

1. exact digest hit
2. exact text/literal hit
3. prototype/shape rescue
4. recurse
5. typed lexical fallback

Even after clustered induction, the primary reusable identity is still too
close to exact surface forms.

That makes prototypes look like a recovery mechanism instead of the actual
motif vocabulary.

## Proposed Change

Make prototype identity the primary motif contract.

In practical terms:

1. induce prototype clusters as first-class vocab entries
2. emit prototype IDs as the default motif identity for held-out encoding
3. treat exact digest/literal matches as diagnostics or optional aliases, not
   the primary surface
4. keep runtime matching exact:
   - exact prototype membership only
   - no nearest-neighbor or approximate matching

This is still a conservative change.
It strengthens `#1` without jumping to `#3`.

## Scope

This spec is for the tokenizer control plane only.

It does **not** include:

- neighborhood / similarity matching across prototypes
- primitive-side dynamic-lever tuning
- chunking changes
- model-facing adapter changes
- new primitive comparison work

## Design

### 1. Canonical Motif Identity

For the experimental path under this spec:

- `prototype cluster` becomes the canonical motif identity
- `prototype digest` becomes the canonical model-facing motif digest

The prototype cluster is still built from:

- coarse depth
- coarse byte-length bucket
- coarse state bucket
- lexical shape

But the important shift is:

- this cluster key is no longer just an auxiliary rescue lookup
- it is the unit the tokenizer is trying to emit and reuse

### 2. Vocab Induction

Induction should produce:

- prototype entries first
- exact text entries only if explicitly retained for diagnostics
- shape entries only if explicitly retained for diagnostics or fallback

The default bakeoff/faceoff path under this spec should score the prototype
surface, not the literal surface.

### 3. Runtime Matching

Held-out matching order for the prototype-primary experiment:

1. exact prototype membership
2. recurse to children
3. typed lexical fallback
4. bytes only for invalid UTF-8 spans

Notably absent:

- literal-text rescue
- free-standing shape rescue
- approximate prototype neighborhood matching

Those should remain off for this experiment so the result stays interpretable.

### 4. Telemetry

The bakeoff must distinguish:

- `prototype_hit_docs`
- `exact_motif_hit_docs`
- `literal_hit_docs`
- `shape_hit_docs`
- `lexical_only_docs`

For the prototype-primary run, the expected successful pattern is:

- `prototype_hit_docs` rises materially
- `lexical_only_docs` drops materially
- `literal_hit_docs` and `shape_hit_docs` become irrelevant or zero

### 5. Persistence / Versioning

If this mode changes the persisted vocab contract, bump the vocab format
version and persist the prototype-first layout explicitly.

The persisted artifact should make it clear whether a vocab was built in:

- legacy exact-first mode
- prototype-primary mode

This must not be implicit.

## TDD Plan

### Step 1. Failing Contract Tests

Add focused failing tests for:

1. prototype-primary encoding uses prototype hits on held-out shape-equivalent
   text without literal/shape rescue
2. prototype-primary encoding round-trips exactly
3. prototype-primary vocab persistence round-trips with explicit mode/version
4. bakeoff telemetry reports prototype hits as the dominant structural signal

### Step 2. Smallest Implementation

Implement the smallest change that makes those tests pass:

- canonical prototype identity in vocab/runtime
- exact prototype-only matching path
- no new similarity logic

### Step 3. Held-Out Bakeoff

Rerun:

- held-out local bakeoff in `full` mode under prototype-primary identity
- held-out local bakeoff in `motif-only` mode under prototype-primary identity

## Success Criteria

This experiment is a success if it shows **material** improvement over the
current clustered-induction result.

Minimum bar:

- `prototype_hit_docs` rises materially above `5`
- `lexical_only_docs` drops materially below `42` in full mode
- held-out code/docs bucket medians improve meaningfully
- hard gates remain perfect:
  - `roundtrip_failures=0`
  - `chunk_utf8_failures=0`
  - `collation_failures=0`
  - `byte_fallback_docs=0`

The most important held-out buckets are:

- `code.rust`
- `code.swift`
- `docs.spec`

If those remain essentially unchanged, this move did not save the primitive.

## Failure Criteria

This experiment fails if:

- prototype hits rise only a little while code/docs stay flat
- lexical-only behavior remains dominant on most held-out docs
- non-log bucket medians remain near:
  - `code.rust ~= 0.81`
  - `code.swift ~= 0.90`
  - `docs.spec ~= 0.76`

If that happens, the evidence becomes much stronger that:

- the current tokenizer control plane still cannot turn primitive state into
  reusable held-out motifs
- and further rescue work should stop in favor of pivoting the primitive or
  tokenizer architecture

## Expected Next Decision

After this experiment there are only two honest paths:

1. prototype-primary identity moves held-out code/docs materially
   - continue with the line and only then consider prototype-neighborhood
     matching
2. prototype-primary identity still does not move held-out code/docs materially
   - treat the kill criterion as close to firing
   - stop spending cycles on rescue passes

