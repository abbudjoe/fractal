# Compositional Motif Vocab Spec

## Purpose

This spec defines the first half of the held-out OOV fix:

- replace exact memorized motif induction with reusable recurring submotif
  induction

The goal is to make held-out documents recover known structure from internal
pieces instead of dropping to bytes when a parent motif digest is novel.

## Problem

The current vocab behaves too much like a set of exact observed spans.

That creates a brittle contract:

- induction docs compress well
- held-out docs miss the exact digests
- fallback reaches bytes too often

The vocab needs to represent reusable structure, not just previously seen
frontiers.

## Design Goal

The vocab should prefer motifs that are:

- recurring
- internally reusable
- shared across documents or files
- not too large or document-specific

This is a compositional vocab, not a document memorization cache.

## Induction Pipeline

Run the recursive tokenizer over induction documents and collect motif
statistics for every node, not just emitted frontier tokens.

For each observed motif candidate, track at least:

- `digest`
- `depth`
- `byte_len`
- `child_digests`
- `occurrence_count`
- `doc_count`
- `source_count`
- `bucket_count`

Where:

- `doc_count` counts distinct induction documents
- `source_count` counts distinct parent source files
- `bucket_count` counts distinct induction buckets

## Admission Rules

Only admit motifs that clear recurrence and usefulness thresholds.

Required first-pass filters:

- minimum occurrence count
- minimum distinct document count
- minimum distinct source count
- maximum byte length
- reject motifs that occur only in one source unless they are very small and
  very frequent

Suggested first-pass defaults:

- `occurrence_count >= 3`
- `doc_count >= 2`
- `source_count >= 2`
- `byte_len <= 512`

These values should stay configurable, but the induction code must make the
selection logic explicit and testable.

## Vocab Entry Shape

The vocab should store reusable structural information, not only ids.

Suggested shape:

```rust
struct MotifEntry {
    id: u32,
    digest: String,
    child_ids: Vec<u32>,
    min_depth: u8,
    max_depth: u8,
    byte_len: u32,
    occurrence_count: u32,
    doc_count: u32,
    source_count: u32,
    bucket_count: u32,
}
```

And:

```rust
struct FaceoffVocab {
    format_version: u32,
    motifs: Vec<MotifEntry>,
    lexical: LexicalLayerConfig,
    byte_fallback_base: u32,
}
```

The exact types can vary, but the stored recurrence metadata should be
available for validation and debugging.

## Encode Behavior

Held-out encoding should use a stronger structural fallback ladder.

For a candidate span:

1. try current-node motif match
2. if unknown, recurse to children
3. keep any known child cover that exactly covers the span
4. only drop below that to lexical fallback if structural cover fails

That means the encoder should prefer a known cover assembled from descendants
over raw bytes.

## Required Invariants

- deterministic induction from the same induction corpus
- deterministic admission/rejection of motifs
- exact round-trip remains unchanged
- held-out docs may not use evaluation data for vocab induction
- no motif entry may reference unknown child ids

## Regression Surface

Required tests:

1. recurring internal motifs are admitted while one-off giant motifs are
   rejected
2. induction is deterministic across repeated runs
3. held-out docs recover known descendant cover instead of immediately hitting
   byte fallback
4. exact round-trip remains perfect
5. vocab persistence/versioning still works with the richer entry shape

## Telemetry To Add

The bakeoff runner should eventually expose:

- `known_parent_hits`
- `known_descendant_cover_docs`
- `known_descendant_cover_tokens`
- `motif_oov_docs`
- `motif_oov_tokens`

This telemetry will tell us whether the compositional vocab is actually doing
useful held-out recovery.

## Expected Win

If this works, held-out documents should:

- reuse internal motif structure more often
- produce fewer byte-fallback tokens
- keep stronger compression on logs and repetitive operational text
- avoid pathological collapse on code/docs/JSONL

## Expected Failure Mode

Possible failure modes:

- thresholds are too strict and the vocab becomes too sparse
- thresholds are too loose and the vocab reintroduces memorized giant motifs
- child-cover search becomes too expensive if not bounded carefully

## Recommended Order

This spec should be implemented alongside typed lexical fallback, but the two
layers should remain separate:

- compositional vocab for structural recovery
- lexical fallback for residual novel content
