# Model-Facing Contract

## Purpose

This document defines the stable intermediate representation that sits between the tokenizer track and any downstream model integration.

The goal is to keep the tokenizer abstract and reusable:

- existing LMs can consume the packaged frontier through adapters
- future fractal-native models can consume the same contract directly
- the tokenizer core does not need to know which model family is downstream

The current codebase already establishes the important tokenizer-side primitives:

- `NoveltyAware` is the current frontier policy leader
- `EncodedDocument` carries the exact token stream plus fallback statistics
- `FaceoffChunkLimits` and `FaceoffChunkedDocument` provide deterministic packaging
- `FaceoffChunkedDocument::reconstruct()` proves packaging is lossless

This contract formalizes those pieces as a model-facing ABI.

The current implementation now also includes file-backed HF tokenizer smoke
coverage and a versioned JSON persistence contract for `FaceoffVocab`, so the
contract is no longer purely in-memory.
The native compatibility path now also includes deterministic right-padded
collation with binary attention masks and stable multi-document chunk ordering.

For this phase, the canonical adapter input should be treated as a combined
wrapper concept:

- `ModelFacingDocument { encoded, chunked }`

That wrapper keeps token-level structure and packaged chunk structure together.
`FaceoffChunkedDocument` alone is not enough for all downstream adapters because
it does not preserve the full token-level frontier metadata by itself.

## Layer Model

The model-facing path should be split into four layers.

1. Tokenizer contract
2. Packaging contract
3. Model adapter
4. Model runtime

Each layer has a different responsibility and should not leak into the others.

### 1. Tokenizer Contract

This is the tokenizer-side output of `NoveltyAware` or any future frontier policy.

Required properties:

- deterministic encoding
- exact reconstruction of the raw input
- ordered tokens with byte spans
- explicit fallback accounting
- stable token identities within the current experiment run

The tokenizer contract is responsible for producing structural truth, not model inputs.
Vocabulary serialization, versioning, and train/val/test ownership are still
deferred as training-system concerns, but the codebase now has a versioned JSON
vocab persistence format for the current experiment phase.

### 2. Packaging Contract

Packaging takes an `EncodedDocument` and turns it into a model-sized sequence of ordered chunks.

The current implementation already exposes the minimum useful shape:

- `FaceoffChunkLimits`
- `FaceoffChunk`
- `FaceoffChunkedDocument`

The packaging contract should remain a typed transformation with explicit boundaries.
Chunk limits are best-effort around indivisible frontier tokens: the contract
should preserve deterministic ordering and reconstruction, but it should not
overstate byte limits as hard guarantees when a single frontier token is larger
than the nominal window.

Required properties:

- chunk ordering is deterministic
- chunk boundaries are explicit
- chunk reconstruction is lossless
- chunk payloads can be concatenated in order to recover the original raw text
- packaging does not invent new token semantics

### 3. Model Adapter

An adapter turns the packaged tokenizer output into whatever a particular model runtime expects.

This layer is where we support different downstream families:

- native-tokenizer compatibility mode for existing LMs
- embedding-bridge mode for learned integration with existing LMs
- fractal-native mode for a future tokenizer-aware model

The adapter is not the tokenizer and should not re-implement tokenizer logic.

### 4. Model Runtime

The runtime is the actual execution engine:

- a frozen OSS model
- a fine-tunable existing LM
- or a future fractal-native model

The runtime consumes adapter output and produces logits, hidden states, or training loss.

## Canonical Structures

The model-facing contract should stay close to the existing faceoff types.

### EncodedToken

An encoded token is the atomic unit of the tokenizer-side stream.

Required fields:

- stable token id
- token kind
- depth
- byte start
- byte end
- payload bytes

In the current implementation, `EncodedToken` already carries these fields.

### EncodedDocument

An encoded document is the ordered, lossless frontier stream.

Required invariants:

- tokens are ordered by byte span
- token spans are contiguous over the full document
- token bytes exactly match the underlying raw text slice
- `decode()` reconstructs the original text exactly

The document also carries fallback statistics so the model-facing layer can report whether it is consuming a clean frontier or a degraded fallback path.

### ModelFacingDocument

`ModelFacingDocument` is the canonical wrapper for this phase.

Required fields:

- `encoded`
- `chunked`

Required invariants:

- `encoded` remains the source of truth for exact frontier structure
- `chunked` remains the source of truth for model-sized windowing
- both views must agree on document order and reconstruction
- adapters can rely on the wrapper to keep token-level and chunk-level views together

### FaceoffChunk

A chunk is a packaging unit, not a new linguistic unit.

Required fields:

- chunk index
- token count
- byte count
- byte span
- payload bytes

Required invariants:

- indices are ordered
- spans are non-overlapping and cover the full document when concatenated
- payload bytes reconstruct the document exactly
- chunk limits are deterministic, but individual frontier tokens may force a
  chunk to exceed the nominal byte budget

### FaceoffChunkedDocument

A chunked document is the packaged form of an encoded document.

Required fields:

- input length
- frontier token count
- ordered chunks

Required invariants:

- reconstruction is exact
- chunk order is stable
- packaging parameters are explicit and deterministic

## Existing LMs vs Future Fractal-Native Models

The contract must support two downstream futures without mixing them together.

### Existing LMs

Existing LMs usually expect their own tokenization or embedding space.

Supported integration modes:

- native-tokenizer compatibility mode
  - chunk payload text is retokenized by the downstream model tokenizer
  - useful for immediate evaluation and prompting
- embedding-bridge mode
  - chunked fractal tokens are mapped into the model embedding space
  - useful for learned adapter experiments

The important point is that the tokenizer track does not embed model-specific assumptions into the core encoding contract.
Pad/mask semantics and embedding-bridge tensor contracts are intentionally
deferred to later phases.

### Future Fractal-Native Models

A future fractal-native model should be able to consume the same chunked IR directly.

That model will likely want:

- token id
- depth
- chunk index
- chunk span
- optional structural features derived from the same frontier

The ABI should therefore keep structural metadata explicit instead of hiding it in prose or string heuristics.

## Invariants

These invariants should be enforced in code and tests, not only described here.

1. Exact round-trip
   - raw text -> encoded document -> decoded text must be identical
2. Deterministic encoding
   - the same text and vocab must always produce the same encoded document
3. Deterministic packaging
   - the same encoded document and limits must always produce the same chunks
4. Explicit ordering
   - tokens and chunks must preserve document order
5. Explicit boundaries
   - token and chunk spans must be contiguous and non-overlapping
6. Clean fallback accounting
   - fallback reasons and counts must be visible
7. No hidden model coupling
   - the tokenizer package should not depend on a specific LM family

## What Belongs In Code And Tests

The following belong in code:

- typed data structures
- encode/decode/packaging functions
- deterministic adapter interfaces
- invariant checks and validation helpers

The following belong in tests:

- exact round-trip
- deterministic packaging
- fallback rate checks
- chunk reconstruction checks
- compatibility-mode smoke tests

The following belong in prose only:

- why the abstraction exists
- which model families are supported
- the order in which adapters should be implemented

Implementation status:

- file-backed HF tokenizer smoke tests exercise `HuggingFaceNativeTokenizer::from_file`
- `FaceoffVocab` persistence/versioning is implemented as a versioned JSON contract
- the remaining adapter phases still use the same abstract surface

## Practical Summary

The tokenizer track now has enough surface area to define a stable model-facing ABI:

- `EncodedDocument` captures the exact frontier
- `FaceoffChunkedDocument` packages it into model-sized windows
- `ModelFacingDocument` keeps the encoded and packaged views together for adapters
- adapters can translate that packaged form into either existing-LM inputs or future fractal-native inputs

The central design rule is simple:

- tokenizer code owns truth
- packaging owns windowing
- adapters own model-specific translation
- runtimes own execution

## Implementation Status

The first model-facing slice is now implemented and validated in the codebase:

- `ModelFacingDocument` owns the encoded and chunked views together
- `ModelFacingBatch` is the canonical batch wrapper for those documents
- the native compatibility adapter retokenizes chunk payloads deterministically
- focused tests cover exact reconstruction and benchmark-input batch order
- `FaceoffVocab` has a versioned JSON persistence contract for the current phase
