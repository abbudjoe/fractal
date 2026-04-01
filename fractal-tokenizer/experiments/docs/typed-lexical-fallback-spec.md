# Typed Lexical Fallback Spec

## Purpose

This spec defines the second half of the held-out OOV fix:

- add a typed lexical fallback layer above raw bytes

This explicitly chooses **Option B**:

- typed lexical fallback
- not a closed memorized lexical vocab

The goal is to preserve meaningful local structure for novel held-out text even
when no reusable motif cover is available.

## Problem

The current fallback ladder bottoms out too early:

1. known motif
2. recurse to children
3. bytes

That is too harsh for held-out code, docs, JSONL, and logs, because a large
amount of novel text still has stable lexical structure:

- words
- identifiers
- numbers
- punctuation
- whitespace
- indentation

The model-facing path should not lose all of that structure immediately.

## Chosen Design

Unknown spans should degrade to typed lexical atoms before they degrade to raw
bytes.

New ladder:

1. known motif
2. known structural descendant cover
3. typed lexical atoms
4. bytes only as final floor

## Why Typed Lexical Fallback

We are **not** choosing a closed memorized lexical vocab as the main path.

Reason:

- a closed lexical vocab still pushes novel words and identifiers back to bytes
- it recreates the same held-out brittleness at a different layer

Typed lexical fallback is better because it keeps a deterministic typed surface
even when the exact string is novel.

## Lexical Atom Kinds

First-pass lexical kinds should be simple, explicit, and lossless.

Suggested kinds:

- `Word`
- `Identifier`
- `Number`
- `Whitespace`
- `NewlineIndent`
- `Punctuation`
- `SymbolRun`

The final set can change, but it should remain:

- deterministic
- UTF-8-safe
- reversible
- independent of downstream model family

## Lexeme Representation

Suggested shape:

```rust
enum LexemeKind {
    Word,
    Identifier,
    Number,
    Whitespace,
    NewlineIndent,
    Punctuation,
    SymbolRun,
}

struct Lexeme {
    kind: LexemeKind,
    text: Vec<u8>,
}
```

And on the encoded-token side:

```rust
enum EncodedTokenKind {
    Motif { digest: String },
    Lexical { kind: LexemeKind },
    Byte,
}
```

The exact storage may differ, but the kind must be explicit in the contract.

## Scanner Rules

The lexical scanner should be:

- deterministic
- lossless
- greedy within each lexical class
- stable under repeated runs

It should operate on UTF-8 text, not rewrite or normalize the original bytes.

Examples:

- `AuthProvider` -> `Identifier`
- `2026-03-31` -> split as `Number`, `Punctuation`, `Number`, `Punctuation`,
  `Number`
- `"    "` -> `Whitespace`
- `"\n        "` -> `NewlineIndent`
- `::git-push{...}` -> likely `SymbolRun` and punctuation-oriented segments

## Encode Behavior

When no known motif or descendant cover exists for a span:

1. scan the span into typed lexemes
2. emit lexical tokens for each lexeme
3. only fall to byte tokens for text the scanner cannot safely classify

The first implementation should aim for byte fallback to become rare on normal
UTF-8 local text.

## Required Invariants

- exact round-trip remains perfect
- lexical scan is deterministic
- lexical tokens preserve original byte order exactly
- packaging remains UTF-8-safe
- native retokenization and collation stay deterministic

## Regression Surface

Required tests:

1. identifiers, words, numbers, punctuation, and whitespace are classified
   deterministically
2. Unicode-heavy text remains lossless and UTF-8-safe
3. held-out local bakeoff no longer falls through to bytes on every evaluation
   doc
4. exact round-trip remains perfect
5. lexical fallback does not break packaging or native retokenization

## Telemetry To Add

The bakeoff runner should eventually expose:

- `lexical_fallback_docs`
- `lexical_fallback_tokens`
- `byte_fallback_docs`
- `byte_fallback_tokens`
- `lexeme_kind_counts`

This will let us distinguish:

- useful lexical recovery
- residual byte collapse

## Expected Win

If this works, held-out evaluation should:

- stop collapsing directly to bytes
- preserve meaningful structure for novel text
- improve model-facing sequence quality even when motifs are OOV
- remain lossless and deterministic

## Expected Failure Mode

Possible failure modes:

- scanner classes are too coarse and hide useful structure
- scanner classes are too fine and approach byte-level behavior again
- Unicode edge cases leak into byte fallback too often

## Relationship To Compositional Vocab

This spec complements, rather than replaces:

- [compositional-motif-vocab-spec.md](./compositional-motif-vocab-spec.md)

Compositional vocab should recover reusable structure.
Typed lexical fallback should catch the remaining novel surface content.

Both are needed for a seaworthy held-out OOV contract.
