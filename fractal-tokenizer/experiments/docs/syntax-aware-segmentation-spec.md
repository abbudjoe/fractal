# Syntax-Aware Segmentation Spec

## Purpose

This spec defines the next tokenizer-internal rescue attempt after:

- atom-first lexical substrate
- exact document-local motif cache

The next question is whether the lexical substrate is still being grouped into
the wrong recursive spans for held-out code/docs.

This is an experiment mode, not a default flip.

## Why This Is The Next Attempt

The current read is:

- atom-first substrate was real progress
- exact document-local cache produced only a tiny additional lift
- held-out code/docs still do not show enough reusable structural motifs

That points back to the segmentation layer.

The current `BoundaryAware` split policy is still fairly generic:

- blank-line boundaries
- newline boundaries
- punctuation boundaries
- whitespace boundaries
- midpoint fallback

That is a safer baseline than pure balanced splitting, but it is still too
blunt for:

- Rust and Swift declaration boundaries
- markdown headings and list items
- structured docs with clause-level repetition

## Hypothesis

If the atom-first lexical substrate uses syntax-aware split priorities, then:

- code declarations and docs sections will land in more stable recursive spans
- the same constructs will align more consistently across held-out documents
- code/docs should gain reusable structure without loosening motif identity

This should improve held-out `code.rust`, `code.swift`, and `docs.spec` more
honestly than another local cache or matching heuristic.

## Scope

### In Scope

- add an explicit `SyntaxAware` split policy
- keep `BoundaryAware` as the stable default
- implement syntax-aware boundary ranking over lexical atoms
- thread the policy through the local bakeoff runner
- add focused regressions and a held-out bakeoff comparison

### Out Of Scope

- full AST parsing
- language-specific parsers
- fuzzy motif matching
- new primitive variants
- changing the default split policy without passing the bakeoff gate

## Chosen Design

Add a new split policy:

```rust
enum SplitPolicy {
    Balanced,
    BoundaryAware,
    SyntaxAware,
}
```

`SyntaxAware` should primarily operate on the lexical-atom substrate.

For raw-byte substrate:

- it may fall back to existing boundary-aware behavior
- the experiment is judged on lexical substrate, not raw bytes

## Architectural Shape

### 1. Syntax-Aware Operates On Atom Boundaries

The experiment should build on the existing atom-first substrate.

Split candidates remain atom boundaries only:

- never split inside a lexeme
- never violate UTF-8 boundaries

### 2. Syntax-Aware Uses Contextual Boundary Ranking

Replace the current simple boundary rank with a richer contextual classifier
over neighboring atoms.

First-pass preferred boundary classes:

1. blank-line / paragraph boundary
2. line-start declaration boundary
3. line-start markdown heading or list boundary
4. statement / block closure boundary
5. structural punctuation boundary
6. whitespace boundary
7. midpoint fallback

The important change is:

- do not treat all punctuation/whitespace boundaries as equivalent
- promote boundaries that likely start or end reusable code/docs units

### 3. Line-Start Declarations Should Be Explicit

At a line start, prefer boundaries before lexical atoms that look like
declarations or major statements.

Suggested initial keyword set:

- Rust / general code:
  - `fn`
  - `struct`
  - `enum`
  - `impl`
  - `trait`
  - `use`
  - `pub`
  - `const`
  - `let`
  - `match`
  - `if`
  - `for`
  - `while`
  - `return`
- Swift:
  - `struct`
  - `enum`
  - `class`
  - `actor`
  - `protocol`
  - `extension`
  - `func`
  - `var`
  - `let`
  - `import`

This must stay deterministic and typed. Do not hide it behind regex-only prose.

### 4. Docs Boundaries Should Be Explicit

Prefer boundaries before:

- markdown headings (`#`, `##`, etc.) at line start
- list bullets (`-`, `*`) at line start
- numbered list items at line start
- line-start code fence markers if present

This is important for `docs.spec`.

### 5. Statement / Block Closure Should Be Promoted

Prefer boundaries around:

- `}`
- `;`
- line breaks after clause-ending punctuation

This should help code bodies and repeated field/enum/struct clauses group more
consistently.

## API Surface

### Tokenizer Config

No new config object is needed. Extend `SplitPolicy`:

```rust
pub enum SplitPolicy {
    Balanced,
    BoundaryAware,
    SyntaxAware,
}
```

Stable default remains:

```rust
SplitPolicy::Balanced
```

The bakeoff path can still explicitly choose the experiment policy.

### Bakeoff CLI

Add:

```text
--split-policy balanced|boundary-aware|syntax-aware
```

Default in the runner remains:

```text
boundary-aware
```

The summary should print:

```text
BAKEOFF_SPLIT_POLICY=balanced|boundary_aware|syntax_aware
```

## TDD Plan

### Step 1. Focused Split Regressions

Add tests proving:

- syntax-aware split prefers a line-start declaration boundary in code
- syntax-aware split prefers a markdown heading/list boundary in docs
- syntax-aware split still preserves UTF-8-safe boundaries

### Step 2. CLI / Args Regressions

Add parser tests proving:

- default split policy is `boundary-aware` in the bakeoff runner
- `--split-policy syntax-aware` parses correctly
- invalid values fail loudly

### Step 3. Full Suite

Run:

```sh
cargo fmt --all
cargo test -p fractal-tokenizer faceoff_ -- --nocapture
cargo test -p fractal-tokenizer model_face_ -- --nocapture
cargo test -p fractal-tokenizer --bin local_bakeoff -- --nocapture
```

### Step 4. Held-Out Bakeoff Gate

Compare on the lexical substrate:

- baseline:
  - `--substrate lexical --split-policy boundary-aware`
- experiment:
  - `--substrate lexical --split-policy syntax-aware`

Keep the rest stable:

- `p1_fractal_hybrid_dyn-state-norm_v2`
- full fallback mode
- legacy identity mode
- coarse prototype granularity
- local cache off

## Success Criteria

Hard gates must remain perfect:

- `roundtrip_failures=0`
- `chunk_utf8_failures=0`
- `collation_failures=0`
- `byte_fallback_docs=0`

Directional success means:

- `code.rust`, `code.swift`, and/or `docs.spec` move materially upward
- `lexical_only_docs` falls materially
- `jsonl.signals` does not overcollapse

Strong pass:

- at least one held-out non-log code/docs bucket reaches parity or better

## Failure Criteria

This experiment fails if:

- hard gates regress
- code/docs stay effectively unchanged
- only logs/JSONL improve
- or non-log overcollapse reappears

If it fails, the current tokenizer is running very low on segmentation/substrate
options, and the primitive gets closer to its architectural ceiling.

## Outcome

Status:

- `Tried`

Validation:

- `cargo test -p fractal-tokenizer faceoff_ -- --nocapture`
- `cargo test -p fractal-tokenizer model_face_ -- --nocapture`
- `cargo test -p fractal-tokenizer --bin local_bakeoff -- --nocapture`

Important root-cause fix discovered during implementation:

- the lexical scanner's `NewlineIndent` atom was incorrectly consuming the first
  non-indentation character on lines with zero indent
- this was fixed before judging the syntax-aware split results

Held-out local bakeoff on the lexical substrate:

- baseline `--split-policy boundary-aware --local-cache off`
  - `exact_motif_hit_docs=0`
  - `prototype_hit_docs=0`
  - `lexical_only_docs=19`
  - `code.rust=0.80`
  - `code.swift=1.02`
  - `docs.spec=0.74`
  - `jsonl.signals=9.21`
  - `logs.operational_mixed=1.10`
- experiment `--split-policy syntax-aware --local-cache off`
  - `exact_motif_hit_docs=0`
  - `prototype_hit_docs=0`
  - `lexical_only_docs=17`
  - `code.rust=0.80`
  - `code.swift=0.98`
  - `docs.spec=0.74`
  - `jsonl.signals=6.94`
  - `logs.operational_mixed=1.07`

Hard gates stayed clean:

- `roundtrip_failures=0`
- `chunk_utf8_failures=0`
- `collation_failures=0`
- `byte_fallback_docs=0`

Read:

- syntax-aware segmentation is technically sound
- it reduces `lexical_only_docs` a bit
- it also reduces the `jsonl.signals` overcompression relative to the current
  lexical baseline
- but it does not materially improve held-out `code.rust` or `docs.spec`
- `code.swift` regresses slightly

Decision:

- keep `SyntaxAware` available as an explicit experiment mode
- do not promote it to the default path
- treat this as another near-miss, not a rescue
