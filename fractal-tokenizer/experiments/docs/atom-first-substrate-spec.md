# Atom-First Substrate Spec

## Purpose

This spec defines the next serious architectural attempt to save the tokenizer
primitive on held-out code/docs:

- stop building the recursive tree over raw byte spans
- build it over a canonical typed atom stream instead

This is an experiment mode, not a default flip.

## Why This Is The Next Attempt

The evidence now points to a substrate problem:

- primitive choice mostly washed out in the held-out bakeoff
- boundary-aware split was not enough
- typed lexical fallback fixed byte collapse, but only as a safety floor
- state-signature prototype induction improved recall, but precision collapsed on
  some structured buckets
- runtime emission gating was a no-op

The current control plane is trying to discover reusable structure over raw text
spans whose boundaries are still too arbitrary for code/docs.

But we already have a deterministic typed atomizer in
`src/faceoff/lexeme.rs`. The likely missed opportunity is:

- use typed atoms as the primary recursive substrate
- not only as the last fallback tier

## Hypothesis

If the recursive primitive operates over typed lexical atoms instead of raw byte
spans, then:

- code/docs spans will align more consistently across documents
- motif identity will be asked to compare more stable units
- held-out structural reuse should rise without immediately reintroducing byte
  collapse

This should improve held-out `code.rust`, `code.swift`, and `docs.spec` more
honestly than additional local matching heuristics.

## Scope

### In Scope

- add a typed tokenizer substrate mode
- keep the current raw-byte substrate as the stable default
- implement a lexeme-first recursive path as an explicit experiment mode
- thread that mode through the local bakeoff runner
- add focused regression tests and held-out bakeoff validation

### Out Of Scope

- full AST parsing
- document-type-specific parsers
- neighborhood / approximate motif matching
- model-training changes
- changing the stable default without passing the bakeoff gate

## Chosen Design

Add a new tokenizer substrate mode:

```rust
enum TokenizerSubstrateMode {
    RawBytes,
    LexicalAtoms,
}
```

`RawBytes` remains the default.

`LexicalAtoms` is an experiment mode.

## Architectural Shape

### 1. Canonical Atomization

For `LexicalAtoms` mode:

- scan the UTF-8 input with the existing typed lexical scanner
- produce an ordered atom stream
- each atom must preserve:
  - lexical kind
  - byte start
  - byte end
  - original bytes

Suggested internal shape:

```rust
struct AtomSpan {
    kind: FaceoffLexemeKind,
    start: usize,
    end: usize,
    bytes: Vec<u8>,
}
```

This may reuse or rename the current `LexemeSpan`, but the contract should stay
typed and explicit.

### 2. Recursive Segments Operate On Atom Ranges

Instead of recursive `Segment { bytes, start, depth }`, the lexeme-first path
should recurse over contiguous atom ranges:

```rust
struct AtomSegment<'a> {
    atoms: &'a [AtomSpan],
    byte_start: usize,
    byte_end: usize,
    depth: usize,
}
```

Rules:

- recursion boundaries must fall only on atom boundaries
- emitted `TokenRecord` values must still use byte offsets so the rest of the
  faceoff/model-facing stack remains compatible
- no atom may be split across recursive children

### 3. Atom Features Replace Raw-Byte Features

The primitive input features for `LexicalAtoms` mode should be derived from the
typed atom stream rather than raw bytes.

First-pass atom feature surface should be simple and deterministic:

- atom count
- byte length
- per-kind normalized counts
- newline-indent density
- punctuation density
- identifier/word/number ratios
- first atom kind
- last atom kind

The exact vector can be compressed into the existing `dim`, but it must be
derived from typed atoms, not reconstructed from raw byte histograms.

This should be implemented as a dedicated feature builder, not a hidden branch
inside the raw-byte path.

### 4. Atom-Aware Splitting

For `LexicalAtoms` mode, the recursive split should be chosen over atom
boundaries, not byte indices.

First-pass split priority:

1. double newline / paragraph boundary
2. newline-indent boundary
3. structural punctuation boundary
4. whitespace boundary
5. balanced atom midpoint

The important invariant is:

- no split inside an atom

### 5. Existing Downstream Contracts Stay Intact

The rest of the stack should continue to consume normal `TokenRecord` and
`EncodedDocument` values:

- faceoff vocab
- fallback ladder
- packaging
- model-facing adapters
- native retokenization
- collation

This experiment changes the tokenizer substrate, not the downstream ABI.

## API Surface

### Tokenizer Config

Add substrate mode to `TokenizerConfig`:

```rust
pub struct TokenizerConfig {
    ...
    pub substrate_mode: TokenizerSubstrateMode,
}
```

Default:

```rust
TokenizerSubstrateMode::RawBytes
```

### Bakeoff Runner

Add an explicit bakeoff flag:

```text
--substrate raw|lexical
```

Default:

```text
raw
```

The runner summary should print:

```text
BAKEOFF_SUBSTRATE=raw|lexical
```

## TDD Plan

### Step 1. Atom Boundary Regressions

Add focused tests proving:

- lexeme-first splitting never cuts through an identifier
- lexeme-first splitting never cuts through a newline-indent atom
- emitted `TokenRecord` byte spans still round-trip exactly

### Step 2. Tokenizer Surface Regressions

Add focused tests proving:

- `RawBytes` remains the default
- `LexicalAtoms` produces deterministic summaries on the same input
- `LexicalAtoms` preserves UTF-8-safe token boundaries

### Step 3. Bakeoff CLI Tests

Add parser tests proving:

- `--substrate` defaults to `raw`
- `--substrate lexical` is accepted
- unknown substrate values fail clearly

### Step 4. Full Validation

Keep these green:

- `cargo test -p fractal-tokenizer faceoff_ -- --nocapture`
- `cargo test -p fractal-tokenizer model_face_ -- --nocapture`
- `cargo test -p fractal-tokenizer --bin local_bakeoff -- --nocapture`

### Step 5. Held-Out Bakeoff Gate

Run the same held-out local bakeoff twice:

- control:
  - `--substrate raw`
- experiment:
  - `--substrate lexical`

Keep the same primitive and the same stable control-plane defaults unless a more
specific test is needed.

Recommended first comparison:

- `p1_fractal_hybrid_dyn-state-norm_v2`
- legacy identity mode
- coarse prototype granularity
- full fallback mode

This keeps the substrate as the main variable.

## Success Criteria

This experiment passes only if the lexeme-first substrate materially improves
held-out non-log buckets without breaking the hard gates.

Required hard gates:

- `roundtrip_failures=0`
- `chunk_utf8_failures=0`
- `collation_failures=0`
- `byte_fallback_docs=0`

Required directional win:

- `code.rust`, `code.swift`, and/or `docs.spec` move materially upward
- `lexical_only_docs` falls materially
- no catastrophic `jsonl.signals` overcollapse

Strong pass:

- at least one held-out non-log code/docs bucket reaches parity or better

## Failure Criteria

This experiment fails if any of the following happen:

- hard gates regress
- code/docs stay effectively unchanged
- JSONL/log buckets overcollapse again
- the effect is only cosmetic in focused tests and not visible in the held-out
  bakeoff

If it fails, we should treat that as evidence that the current tokenizer needs a
different architectural plane than “better units under the same recursive
primitive,” and narrow the remaining option space accordingly.

## Files Likely To Change

- `src/tokenizer.rs`
- `src/faceoff/lexeme.rs`
- `src/bin/local_bakeoff.rs`
- `src/tests.rs`

Possibly:

- `src/lib.rs`

## Implementation Notes

- prefer explicit new types over overloading the raw-byte path with booleans
- keep the experiment mode isolated and easy to compare against `RawBytes`
- do not silently change the stable default
- preserve downstream ABI wherever possible

## Outcome

Status:

- `Tried`

Validation:

- `cargo test -p fractal-tokenizer faceoff_ -- --nocapture`
- `cargo test -p fractal-tokenizer model_face_ -- --nocapture`
- `cargo test -p fractal-tokenizer --bin local_bakeoff -- --nocapture`

Held-out local bakeoff for `p1_fractal_hybrid_dyn-state-norm_v2` with stable
control-plane defaults:

- `raw`
  - `exact_motif_hit_docs=1`
  - `prototype_hit_docs=3`
  - `lexical_only_docs=42`
  - `code.rust=0.81`
  - `code.swift=0.90`
  - `docs.spec=0.76`
  - `jsonl.signals=3.45`
  - verdict: `GREEN`
- `lexical`
  - `exact_motif_hit_docs=0`
  - `prototype_hit_docs=2`
  - `lexical_only_docs=21`
  - `code.rust=0.83`
  - `code.swift=0.96`
  - `docs.spec=0.77`
  - `jsonl.signals=5.16`
  - verdict: `GREEN`

Read:

- the lexeme-first substrate is technically sound
- it materially reduces lexical-only held-out documents
- it improves held-out code buckets somewhat, especially `code.swift`
- it does not yet create a decisive non-log breakout
- it also raises `jsonl.signals`, though still within the current scorecard

Decision:

- keep `RawBytes` as the stable default
- keep `LexicalAtoms` as an explicit experiment mode
- treat this as real progress, but not a full rescue
