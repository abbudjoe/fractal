# Boundary-Aware Split Spec

## Purpose

This spec defines one decisive rescue experiment for the current tokenizer
primitive:

- keep the recursive hierarchy
- change the segmentation policy from blind balanced splitting to
  boundary-aware splitting

This is not meant as a long sequence of local heuristics.
It is meant as one high-signal experiment to answer whether span geometry is
the main reason the primitive still fails on held-out code/docs.

## Problem

Current held-out code/docs behavior suggests that:

- the recursive tree rarely produces reusable structural spans
- exact motifs miss
- shape aliases miss
- the lexical layer takes over almost entirely

The strongest hypothesis is that the current split policy is cutting across the
wrong boundaries.

## Design Goal

Preserve the existing recursive primitive and hierarchy while improving the
stability of reusable spans on:

- markdown/spec prose
- Rust
- Swift
- structured operational text

The split policy should favor semantically plausible breakpoints before falling
back to the old balanced midpoint behavior.

## Proposed Policy

Introduce an explicit split policy layer with at least:

```rust
enum SplitPolicy {
    Balanced,
    BoundaryAware,
}
```

And add a tokenizer config seam or faceoff-only override that makes the chosen
split policy explicit and testable.

### Boundary-Aware Candidate Boundaries

For the current span, prefer the nearest useful boundary to the midpoint in
this priority order:

1. `\n\n` paragraph break
2. line break `\n`
3. markdown/list separator patterns
4. punctuation-delimited separators:
   - `;`
   - `:`
   - `,`
   - `.`
   - `{`
   - `}`
   - `(`
   - `)`
5. whitespace boundary
6. fallback to balanced midpoint

The exact order may adjust slightly, but the policy should remain:

- deterministic
- local
- cheap
- easy to reason about

## Guardrails

Boundary-aware splitting must not:

- produce empty child spans
- create extreme imbalance that collapses recursion usefulness
- depend on model family
- break exact round-trip

Suggested first-pass guardrails:

- boundary must land within a bounded window around the midpoint
- if no acceptable boundary is found, use the old balanced split
- if a chosen boundary creates a tiny child below a floor, reject it

## Why This Is Not A Bandaid

This is not a prompt-level or score-level tweak.

It changes a primitive contract:

- what counts as a reusable recursive span

That is exactly the right level for the current failure.

If the primitive is real, it should benefit from more stable boundaries.
If it still fails, that is strong evidence that the primitive itself is not
seaworthy for general tokenization.

## Required Tests

### Focused Unit Tests

1. boundary-aware splitting is deterministic
2. empty or degenerate spans are never produced
3. when no boundary is available, the policy falls back to balanced behavior
4. code-like text splits on stable punctuation/newline boundaries rather than
   arbitrary midpoint cuts
5. markdown/spec text prefers paragraph or line boundaries where possible

### Regression/Contract Tests

1. exact round-trip remains perfect
2. UTF-8-safe chunking remains intact
3. packaging and collation remain intact
4. previously passing faceoff/model-facing tests remain green

### Decisive Bakeoff Test

Rerun the held-out local bakeoff and compare against the current baseline.

Success means:

- code/docs recover materially more structural hits
- lexical fallback no longer dominates code/docs completely
- at least one of `docs.spec`, `code.rust`, or `code.swift` moves to parity or
  better
- `jsonl.signals` and logs do not regress badly

Failure means:

- code/docs are still almost entirely lexicalized
- structural hits remain near zero
- ratios stay below parity

If that happens, the primitive should be considered much closer to the kill
criterion.

## Telemetry To Add

The bakeoff runner should expose at least:

- `fallback_shape_hits`
- `fallback_lexical_fallback_tokens`
- maybe later:
  - `split_policy`
  - `boundary_split_count`
  - `balanced_fallback_split_count`

The first two are already enough to judge whether the rescue experiment worked.

## Scope Limit

This spec should be treated as one rescue pass, not an open-ended tuning lane.

If it fails, the next move should be:

- compare other primitives using the same bakeoff pipeline

not:

- add a long chain of split heuristics to `p1`

## Trial Outcome

Status:

- `Tried`

Observed result on the held-out local bakeoff:

- `BAKEOFF_SPLIT_POLICY=boundary_aware`
- hard gates stayed green
- bucket medians were materially unchanged from the prior shape-alias run
- `code.rust`, `code.swift`, and `docs.spec` remained below parity
- `jsonl.signals` remained the strongest held-out win

Interpretation:

- this boundary-aware split variant did not materially change the real-world
  held-out outcome
- it should be treated as a failed rescue pass for `p1`, not as a new default
