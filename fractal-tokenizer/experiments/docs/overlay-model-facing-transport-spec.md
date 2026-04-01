# Overlay Model-Facing Transport Spec

## Purpose

This spec defines the first real model-facing integration slice for the
canonical-tokenizer + recursive-overlay pivot.

The overlay line already proved two things:

- exact reversible overlay transport works
- structured repetitive text is a real advantage surface

What remained missing was a typed adapter surface between:

- overlay discovery and transport in `overlay.rs`
- downstream model-facing batching and future runtime integration

This spec adds that seam without inventing a second tokenizer ABI.

## Contract

The first model-facing overlay slice must:

1. keep canonical token ids as the source of truth
2. accept only exact `RecursiveOverlayDocument` inputs
3. batch overlay documents through the existing pack/dictionary machinery
4. expose the resulting transport as a typed model-facing batch
5. preserve exact expansion back to canonical token ids

It must **not**:

- change canonical ids
- add fuzzy matching
- duplicate overlay packing logic in `model_face`
- require model runtimes to understand a new token vocabulary

## Data Model

The minimal owning types are:

```rust
struct OverlayModelFacingDocument {
    overlay: RecursiveOverlayDocument,
}

struct OverlayModelFacingBatch {
    documents: Vec<OverlayModelFacingDocument>,
}

struct OverlayTransportConfig {
    scope: OverlayDictionaryScope,
    sharing_policy: OverlaySharingPolicy,
    max_pack_docs: NonZeroUsize,
    strategy: OverlayBatchPackingStrategy,
}

struct OverlayTransportBatch {
    config: OverlayTransportConfig,
    pack: OverlayBatchPack,
}
```

Important ownership rule:

- `overlay.rs` still owns discovery, packing, factorization, and exact
  expansion
- `model_face::overlay` owns only validation, batching orchestration, and the
  model-facing adapter seam

## Adapter

The first adapter is:

```rust
struct OverlayTransportAdapter {
    config: OverlayTransportConfig,
}
```

It implements:

- `ModelAdapter<Input = OverlayModelFacingBatch, Output = OverlayTransportBatch>`

And it should do exactly one thing:

- convert validated overlay documents into a typed batch-local transport pack

## Initial Defaults

The first stable defaults should be conservative:

- `scope = BatchLocal`
- `sharing_policy = OverlaySharingPolicy::default()`
- `max_pack_docs = 16`
- `strategy = Sequential`

Reason:

- batch-local sharing already proved real
- fixed-pack structure-aware grouping is a useful runtime comparison, but not a
  clear default win on the current corpus

## Required Invariants

These must be enforced in code and tests:

1. invalid overlay documents are rejected before batching
2. prepared transport batches expand exactly back to canonical token ids
3. packing configuration is preserved explicitly in the prepared batch
4. document-local passthrough inputs remain neutral
5. batch-local transport never bypasses canonical exactness

## Success Criteria

This slice is successful if:

- the adapter is typed and local to `model_face`
- the overlay packer remains single-owned in `overlay.rs`
- exactness remains `100%`
- the adapter preserves the existing shared-dictionary and factorization wins
- the new path adds no regression to the existing model-facing tests

## Why This Slice Matters

This is the smallest clean bridge from:

- shadow overlay experiments

to:

- real model-facing transport

without overcommitting to a new runtime or token ABI too early.
