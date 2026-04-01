# Code/Docs Structural Diagnosis

## Purpose

This document records the first-principles diagnosis behind the current
held-out failure mode on:

- `docs.spec`
- `code.rust`
- `code.swift`

The goal is to avoid treating this as a threshold-tuning problem if the real
issue is the primitive's current segmentation geometry.

## Observed Held-Out Pattern

After:

- typed lexical fallback above bytes
- compositional recurring-submotif vocab
- structural shape aliases

the held-out local bakeoff now shows:

- `byte_fallback_docs=0`
- real shape-based motif hits on `jsonl.signals`
- only modest improvement on `logs.operational_mixed`
- little or no improvement on `docs.spec`, `code.rust`, and `code.swift`

Typical held-out code/docs rows still look like:

- `fallback_motif_hits=0`
- `fallback_shape_hits=0`
- `fallback_unknown_motifs=63`
- `fallback_recursed_to_children=31`
- `fallback_lexical_fallback_tokens == frontier_token_count`

Interpretation:

- held-out code/docs are not structurally tokenized in practice
- they are being carried almost entirely by the typed lexical rescue layer

## Why JSONL Signals Improved But Code/Docs Did Not

`jsonl.signals` is strongly templatic:

- repeated record shapes
- stable key/value boundaries
- regular punctuation layout
- similar local span geometry across documents

That means even the current recursion tree lands on sufficiently repeatable
subspans for structural aliases to fire.

Code/docs behave differently:

- reusable units are semantically meaningful, but not necessarily balanced
- the same concept may appear with different lexical surface forms
- useful reuse boundaries often follow:
  - newline and indentation
  - paragraph/list boundaries
  - punctuation-delimited clauses
  - signatures and block headers
  - markdown section structure

The current primitive still segments by balanced recursive subdivision.
That creates spans that are often:

- too arbitrary
- too cross-cutting
- unstable across files

So even when the same higher-level construct appears twice, it often does not
land in the same recursive span.

## First-Principles Read

This points to a deeper problem than OOV handling.

The issue is likely:

- not only "motif identity is too literal"
- but also "the spans themselves are badly chosen for code/docs"

Said differently:

- the recursive hierarchy may still be useful
- but blind balanced splitting is probably the wrong structural prior for
  code/docs

## Consequence

This means we should avoid spending many more cycles on:

- lexical class tuning
- threshold nudging
- more literal/shape alias variants

unless we first test whether the primitive improves when its segmentation
respects real code/docs boundaries.

## Most Important Next Question

Can the primitive survive if we keep the recursive hierarchy but make the split
policy boundary-aware?

That is the next decisive rescue experiment.

If boundary-aware splitting still fails to produce real held-out structural
reuse on code/docs, the case for pivoting away from this primitive becomes much
stronger.
