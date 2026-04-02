# Server-Side Overlay Envelope Spec

## Purpose

This spec defines the first runtime-facing integration step after the offline
overlay benchmark succeeded.

The offline benchmark already established that the current overlay line has a
real niche advantage:

- exact expansion remains perfect
- structured repetitive text gets strong transport wins
- neutral control buckets stay neutral

What remains uncertain is not whether the overlay works offline. It is whether
the overlay can cross a real process boundary without turning into dead weight.

This spec answers that question with the smallest complete runtime design.

## Short Read

The first runtime integration is:

1. client prepares a typed overlay transport batch
2. client sends that transport envelope to a local server
3. server validates and rematerializes the exact canonical prompt text
4. server forwards the exact canonical prompt text to the existing model
   runtime
5. outputs must match the plain-text baseline exactly for deterministic smoke
   cases

This is intentionally **not**:

- a new tokenizer ABI
- direct model ingestion of overlay symbols
- canonical token-id ingest by the runtime
- session-global dictionary lifecycle

It is a stateless, exact, process-boundary proof.

## Why This Slice Exists

The overlay line now has three proven pieces:

- discovery and packing work
- transport value is real on structured repetitive text
- local model-facing canaries work after exact rematerialization

But the current path is still effectively in-process:

- the client prepares the overlay
- the same process materializes it back to canonical text
- the model sees ordinary prompt text

That is enough for correctness, but not enough to answer the next systems
question:

- does the overlay still earn its keep once it crosses a real request
  boundary?

## Contract

The first server-side envelope must:

1. keep canonical prompt text as the only runtime input consumed by the model
2. send overlay transport as typed request data, not ad hoc strings
3. validate exactness before runtime dispatch
4. rematerialize prompt text on the server side only
5. preserve deterministic output equality for deterministic smoke prompts

It must **not**:

- mutate canonical token ids
- require the model runtime to understand overlay symbols
- add fuzzy matching or server-side heuristic repair
- add session state or shared dictionary persistence
- invent a second compatibility path that bypasses canonical rematerialization

## Ownership

Ownership must stay explicit:

- `overlay.rs` owns discovery, transport packing, factorization, and exact
  rematerialization primitives
- `model_face::overlay` owns typed transport adaptation between overlay docs
  and runtime-facing batches
- the new server-envelope module owns only:
  - request/response envelope types
  - validation
  - rematerialization orchestration
  - forwarding to an existing text-based runtime client

No overlay packing logic should be duplicated in the server layer.

## Data Model

The first envelope should be typed and minimal.

Illustrative shape:

```rust
struct OverlayServerRequest {
    documents: Vec<OverlayTransportDocument>,
    runtime: OverlayRuntimeTarget,
}

struct OverlayTransportDocument {
    document_id: String,
    transport: OverlayTransportBatchDocument,
}

enum OverlayRuntimeTarget {
    OllamaGenerate(OllamaGenerationRequest),
    OllamaEmbed(OllamaEmbeddingRequest),
}

struct OverlayServerPreparedDocument {
    document_id: String,
    prompt_text: String,
}

struct OverlayServerPreparedBatch {
    documents: Vec<OverlayServerPreparedDocument>,
}
```

The exact names may change, but the owning boundary should not:

- request envelope types live with the server seam
- overlay transport internals remain owned by overlay/model_face modules

## Minimal Runtime Flow

The first runtime path should be:

```text
client text
-> canonical tokenizer
-> overlay discovery
-> overlay transport batch
-> typed server request
-> server exact rematerialization
-> plain prompt text
-> existing runtime client
```

Important constraint:

- the runtime client must remain unaware of overlay symbols

That keeps the design reversible and low-risk.

## Validation Rules

The server-side envelope must reject:

- inexact overlay transports
- malformed batch/document ordering
- missing required runtime config
- zero-document requests

Validation should happen before runtime dispatch, not after a failed request.

## Initial Target Runtime

The first runtime target should stay local and simple:

- local Ollama generation and embedding requests

Reason:

- we already have a functioning local runtime seam
- this avoids introducing another serving stack while the protocol is still
  being proven

## Metrics

This slice should add runtime-boundary metrics, not replace the offline ones.

Required measurements:

- request payload bytes
- rematerialized prompt bytes
- server materialization latency
- server dispatch latency
- end-to-end latency
- exact prompt equality against the plain baseline
- deterministic output equality for generation smoke prompts

These should be reported separately for:

- base plain-text request
- overlay-envelope request

## Success Bars

### Required

- exact rematerialization remains perfect
- deterministic generation smoke output matches the plain baseline exactly
- embedding/generation requests succeed through the envelope path
- overlay-envelope requests preserve document order and document boundaries

### Strong

- request payload bytes are materially smaller on structured repetitive inputs
- server materialization cost stays modest relative to request latency
- control inputs stay near-neutral in total latency and payload size

## Failure Conditions

Treat the server-envelope slice as failing if any of the following appear:

- any prompt rematerialization mismatch
- any deterministic output mismatch on the smoke prompts
- payload savings are erased by oversized envelope overhead
- server orchestration duplicates logic already owned by `overlay.rs`
- runtime integration requires a second token ABI

## Why This Is The Right Next Step

This is the smallest clean runtime design because it answers the next unknown
without overcommitting:

- it proves the overlay can survive a real boundary
- it preserves the canonical tokenizer as the contract floor
- it does not claim model-side compute wins we have not earned yet

If this works, the next higher-upside steps become legitimate:

- server-side canonical token-id ingest
- session-local shared dictionaries

If this fails, we learn that the overlay’s current niche advantage is still
too fragile at the runtime boundary.
