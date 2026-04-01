# Document-Local Motif Cache Spec

## Purpose

This spec defines the next architectural attempt after the atom-first
substrate pass:

- keep the stable typed atom substrate
- add a strictly document-local reuse plane above lexical fallback

This is an experiment mode, not a default flip.

## Why This Is The Next Attempt

The current evidence says:

- global motif identity is still too weak on held-out code/docs
- typed lexical fallback prevents byte collapse but carries too much of the
  held-out path
- atom-first substrate materially reduced `lexical_only_docs`
- held-out code/docs still do not show strong reusable structure

The likely missing layer is contextual reuse inside a single held-out document.

Code, specs, and operational text often repeat exact spans locally:

- repeated log fragments
- repeated field names
- repeated type signatures
- repeated markdown phrases
- repeated code identifiers or clauses

Those repeats are honest reuse opportunities even when the global induction
vocab has never seen the span before.

## Hypothesis

If the tokenizer can assign document-local motif identities to exact repeated
spans after the first occurrence, then:

- held-out lexical-only pressure should drop further
- code/docs should gain local structural reuse without loosening global motif
  matching
- we can test whether a contextual reuse plane is the missing architectural
  layer

This should improve held-out code/docs more honestly than another global fuzzy
matching heuristic.

## Scope

### In Scope

- add an explicit document-local cache mode to the faceoff encoder
- cache exact repeated spans only
- keep cache lifetime limited to a single document encode
- add explicit telemetry for local cache hits and stores
- thread the mode through the local bakeoff runner
- validate with focused regressions and a held-out bakeoff rerun

### Out Of Scope

- global vocabulary changes
- approximate or neighborhood local matching
- document-type-specific parsing
- cache persistence across documents
- changing the stable default without clearing the bakeoff gate

## Chosen Design

Add an explicit local-cache mode:

```rust
enum FaceoffLocalCacheMode {
    Off,
    ExactSpan,
}
```

`Off` remains the default.

`ExactSpan` is the experiment mode.

## Architectural Shape

### 1. Cache Lives In The Faceoff Encoding Pass

The cache belongs in the faceoff fallback/control plane, not in:

- primitive state
- tokenizer induction
- global vocab persistence

Reason:

- the question is whether local repeated held-out spans should earn temporary
  reuse inside one document
- that is a document-scoped control-plane concern

### 2. Cache Key Is Exact UTF-8 Span Text

First pass stays intentionally strict:

- only exact repeated UTF-8 spans can hit the local cache
- no fuzzy matching
- no prototype clustering
- no lexical-shape equivalence

Suggested internal key:

```rust
struct LocalSpanKey {
    text: String,
    depth: usize,
}
```

The exact shape can vary, but the key must be typed and deterministic.

Including `depth` is preferred so we do not silently merge radically different
tree levels.

### 3. Cache Is Write-After-First-Lexicalization

The first occurrence of an unknown span should follow the normal path:

- try exact motif/prototype/global rescue
- recurse if appropriate
- lexical fallback if needed

Only after that first successful lossless encoding should the encoder register
the span in the local cache.

That means:

- no forward-looking prepopulation
- no hidden second pass
- no cheating before the first literal occurrence is paid for

### 4. Cache Hits Emit Temporary Document-Local Motifs

When the same exact span reappears later in the same document:

- emit a temporary local motif token instead of descending to lexical fallback
- preserve exact bytes for decode
- keep the identifier stable and deterministic within the document

Suggested token kind extension:

```rust
enum EncodedTokenKind {
    ...
    LocalMotif { digest: String },
}
```

Alternative:

- keep `Motif` as the emitted kind and reserve a local ID range

Either is acceptable, but the document-local status must be explicit somewhere
in typed state or telemetry. It must not masquerade as a persisted global motif.

### 5. Local IDs Must Be Deterministic And Non-Persistent

Local motif IDs must:

- be deterministic within a single encode
- never collide with global vocab IDs
- never be serialized into `FaceoffVocab`
- never survive across documents

Recommended shape:

```rust
struct LocalMotifEntry {
    id: FaceoffTokenId,
    digest: String,
    text: String,
}
```

and a reserved ID base computed from the current vocab size.

### 6. Telemetry Must Be Explicit

Add at least:

- `local_cache_hits`
- `local_cache_stores`

to `FaceoffFallbackStats`.

The bakeoff runner should surface:

- held-out docs with any local-cache hits
- held-out docs that stay lexical-only even with local cache enabled

## API Surface

### Faceoff Mode

Add a new explicit faceoff mode:

```rust
pub enum FaceoffLocalCacheMode {
    Off,
    ExactSpan,
}
```

Default:

```rust
FaceoffLocalCacheMode::Off
```

### Encode Path

Thread the mode through:

- `encode_summary_document`
- `encode_summary_with_policy_and_fallback_mode`
- `encode_text_with_factory_and_policy`

This should follow the existing pattern used for:

- fallback mode
- identity mode
- substrate mode

### Bakeoff CLI

Add an explicit flag:

```text
--local-cache off|exact
```

Default:

```text
off
```

The bakeoff summary should print:

```text
BAKEOFF_LOCAL_CACHE=off|exact
```

## TDD Plan

### Step 1. Focused Cache Regression

Add a regression proving:

- a repeated exact held-out span in the same document gets no local hit on the
  first occurrence
- the second occurrence is encoded as a local motif hit in `ExactSpan` mode
- round-trip stays exact

### Step 2. Isolation Regression

Add a regression proving:

- local cache state does not leak across documents
- encoding the same second document does not inherit local motif IDs from the
  first

### Step 3. CLI/Args Regression

Add parser tests proving:

- default is `off`
- `--local-cache exact` parses correctly
- invalid values fail loudly

### Step 4. Full Suite

Run:

```sh
cargo fmt --all
cargo test -p fractal-tokenizer faceoff_ -- --nocapture
cargo test -p fractal-tokenizer model_face_ -- --nocapture
cargo test -p fractal-tokenizer --bin local_bakeoff -- --nocapture
```

### Step 5. Held-Out Bakeoff Gate

Run the held-out local bakeoff against the current atom-first baseline.

Success means all hard gates remain clean and:

- `lexical_only_docs` drops materially below the current lexical-substrate
  baseline
- `code.rust`, `code.swift`, or `docs.spec` move upward without introducing
  suspicious non-log overcollapse

Failure means:

- no material movement
- or gains only come from overcollapse
- or local cache only helps logs/JSONL while code/docs remain flat

## Expected Interpretation

If this works:

- the missing layer is contextual reuse, not only global structural matching

If this fails:

- the tokenizer likely needs a deeper substrate or segmentation rethink
- or the primitive is nearing its architectural ceiling for general code/docs

## Observed Result

Held-out local bakeoff on the current atom-first lexical substrate:

- baseline `--local-cache off`
  - `prototype_hit_docs=2`
  - `local_cache_hit_docs=0`
  - `lexical_only_docs=21`
  - `code.rust=0.83`
  - `code.swift=0.96`
  - `docs.spec=0.77`
  - `jsonl.signals=5.16`
  - `logs.operational_mixed=1.10`
- experiment `--local-cache exact`
  - `prototype_hit_docs=2`
  - `local_cache_hit_docs=4`
  - `lexical_only_docs=20`
  - `code.rust=0.83`
  - `code.swift=0.96`
  - `docs.spec=0.77`
  - `jsonl.signals=5.16`
  - `logs.operational_mixed=1.10`

Hard gates stayed clean in both runs:

- `roundtrip_failures=0`
- `chunk_utf8_failures=0`
- `collation_failures=0`
- `byte_fallback_docs=0`

Interpretation:

- exact document-local cache hits are real
- they land in a few held-out documents across logs, JSONL, Rust, and Swift
- but they do not materially move the held-out bucket medians

Decision:

- keep `ExactSpan` available as an explicit experiment mode
- do not promote it to the default path
- treat this as a modest contextual signal, not a rescue
