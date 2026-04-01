# Shared Overlay Dictionary Spec

## Purpose

This spec defines the next high-leverage improvement to the canonical-tokenizer
+ recursive-overlay pivot.

The current overlay line has now proven three important things:

- exact reversible overlay transport is feasible
- structured repetitive text is a real win surface
- the next bottleneck is not discovery, but transport overhead

The current record-aware overlay already performs well on the target buckets:

- `jsonl.signals`: `2.27`
- `logs.operational_mixed`: `1.54`
- `exact_failures`: `0`

But the transport summary shows the main remaining inefficiency:

- evaluation-set `transport_ratio = 1.27`
- `macro_definition_symbols = 32413`
- `macro_ref_symbols = 4769`
- `definition_overhead_rate = 0.19`
- bucket-level definition overhead is especially high on:
  - `jsonl.signals = 0.65`
  - `logs.operational_mixed = 0.56`

So the next job is not to discover more reuse. It is to stop paying to redefine
the same scaffolds document by document.

## Short Read

The next architecture step is:

1. keep the canonical tokenizer unchanged
2. keep exact record-aware local scaffold discovery
3. deduplicate identical scaffold definitions across documents in the same
   batch or evaluation session
4. emit per-document overlay streams that reference a shared batch dictionary
5. preserve exact expansion back to canonical token ids

This is not:

- a global vocabulary
- a new tokenizer ABI
- fuzzy cross-document matching
- another attempt to replace the canonical tokenizer

This is:

- transport-layer amortization for a proven structured-text overlay

## Why This Exists

The overlay line is now in a healthy place:

- the target buckets are winning
- the guardrail buckets are neutral
- exactness is intact

That means the next inefficiency is easy to name.

Today we do:

```text
document A -> discover scaffold S -> define S locally
document B -> discover scaffold S again -> define S locally again
document C -> discover scaffold S again -> define S locally again
```

That is correct, but wasteful.

The current results strongly suggest that many structured documents inside the
same batch share identical scaffolds. If we share those definitions, we should
improve the transport ratio without taking on more semantic risk.

## Hypothesis

If:

- structured overlays remain exact and conservative at the document level, and
- overlay definitions are deduplicated across documents within a batch or
  session,

then:

- transport efficiency should improve materially on repetitive structured text,
  while
- activation behavior, exactness, and neutral guardrails should remain stable.

## Non-Goals

This step must **not** do any of the following:

- create a persistent global scaffold vocabulary
- reuse macros across unrelated sessions silently
- allow approximate or neighborhood scaffold matches
- let cross-document sharing bypass exact token-id equality
- alter canonical tokenizer ids
- change model input ids yet

Those belong to later phases, if at all.

## Current Baseline

Measured on the hybrid held-out corpus with:

- canonical tokenizer: `qwen25`
- substrate: `lexical`
- split policy: `boundary-aware`
- overlay mode: `local-record-macro`

Current baseline:

```text
OVERLAY_TRANSPORT_SUMMARY split=evaluation scope=document_local
docs=117 canonical_tokens=218268 transport_symbols=172308
transport_ratio=1.27
base_slice_symbols=135126
macro_ref_symbols=4769
macro_definition_symbols=32413
definition_overhead_rate=0.19
```

This spec aims to improve that transport ratio primarily by reducing
`macro_definition_symbols`.

## Core Idea

### 1. Discovery Stays Local And Exact

Document-local overlay discovery does not change.

Each document still produces:

- canonical token ids
- exact record-aware local scaffold matches
- an exact overlay representation

The cross-document change happens only after discovery.

### 2. Transport Adds A Shared Dictionary Layer

Instead of sending each document’s macro definitions independently, we add a
batch/session overlay pack:

```text
batch
-> shared scaffold dictionary
-> document overlay streams referencing shared ids
```

This is a transport representation, not a new semantic layer.

### 3. Sharing Uses Exact Canonical Token Equality

Two scaffold definitions may be shared only if all of the following match:

- exact canonical token ids
- exact macro kind
- exact expansion span

No fuzzy similarity is allowed in this phase.

### 4. The Dictionary Scope Is Explicit

Phase 1.5 should support explicit scope values:

- `DocumentLocal`
- `BatchLocal`
- `SessionLocal`

Only `BatchLocal` needs to be implemented first.

`SessionLocal` may exist as a data-model placeholder but should not become the
default until the batch-local path is proven.

## Data Model

Illustrative shape:

```rust
struct OverlayPack {
    scope: OverlayDictionaryScope,
    shared_macros: Vec<SharedMacro>,
    documents: Vec<PackedOverlayDocument>,
}

enum OverlayDictionaryScope {
    DocumentLocal,
    BatchLocal,
    SessionLocal,
}

struct SharedMacro {
    shared_macro_id: u32,
    kind: MacroKind,
    token_ids: Vec<u32>,
    doc_ref_count: u32,
    total_use_count: u32,
}

struct PackedOverlayDocument {
    canonical_token_count: u32,
    segments: Vec<PackedOverlaySegment>,
}

enum PackedOverlaySegment {
    BaseSlice { start: u32, len: u32 },
    SharedMacroRef { shared_macro_id: u32, span_len: u32 },
}
```

Important invariants:

- every `SharedMacro` expands to canonical token ids
- every `PackedOverlayDocument` expands exactly to its original canonical token
  stream
- a document may only reference shared ids that are present in the pack

## Execution Model

### Step 1: Per-Document Overlay Discovery

Keep the current implementation:

- canonical tokenization
- record-aware scaffold detection
- exact local macro extraction

### Step 2: Shared Dictionary Construction

Across all held-out or batch documents:

- hash exact macro definitions by:
  - `kind`
  - canonical token ids
- assign one shared id per exact macro definition
- accumulate:
  - document count
  - total use count

### Step 3: Repack Documents

For each document:

- convert local `MacroRef` segments into `SharedMacroRef`
- remove duplicated document-local definitions from the transport cost model
- preserve exact expansion semantics

### Step 4: Report Both Views

The benchmark runner should report both:

- `document_local` transport metrics
- `batch_local` transport metrics

This matters because we want to isolate where the gain is coming from:

- better discovery
- better amortization

## Benchmark Additions

Add a new transport summary family:

```text
OVERLAY_TRANSPORT_SUMMARY split=evaluation scope=document_local ...
OVERLAY_TRANSPORT_SUMMARY split=evaluation scope=batch_local ...
```

Required fields:

- `docs`
- `canonical_tokens`
- `transport_symbols`
- `transport_ratio`
- `base_slice_symbols`
- `macro_ref_symbols`
- `macro_definition_symbols`
- `definition_overhead_rate`

And per-bucket transport summaries for both scopes.

## Success Criteria

This step is a success if all of the following hold:

- exact expansion remains `100%`
- document-local activation behavior does not regress materially
- batch-local transport ratio improves materially over document-local
- guardrail buckets remain neutral

Recommended success bars:

- overall hybrid evaluation:
  - `transport_ratio >= 1.40`
- target buckets:
  - `jsonl.signals` remains `>= 2.0`
  - `logs.operational_mixed` remains `>= 1.5`
- overhead reduction:
  - overall `definition_overhead_rate <= 0.12`
  - target-bucket median definition overhead should drop materially from the
    current `0.65` / `0.56`
- guardrails:
  - no more than `2%` median regression on code/prose/multilingual buckets

## Failure Criteria

This step should be treated as weak or failed if:

- the batch-local dictionary barely reduces definition overhead
- most of the current gain disappears when definitions are repacked
- implementation complexity rises without a meaningful transport-ratio gain
- the design starts drifting toward a second token ABI

## Why This Is The Right Next Layer

This layer is high leverage because it follows the evidence exactly:

- discovery already works on the target buckets
- exactness is already clean
- the current loss is in repeated definitions

So this is not another speculative feature. It is direct pressure relief on the
main measured bottleneck.

## After This

Only if batch-local dictionaries prove worthwhile should we consider:

- session-local scaffold reuse
- model-facing overlay transport
- overlay-aware batching/collation paths

Those are downstream opportunities, not prerequisites for this step.
