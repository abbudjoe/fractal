# Adapter Plan

## Objective

Turn the tokenizer-facing ABI into a small set of model adapters that can serve:

- existing LMs today
- future fractal-native models later

The design must stay abstract and typed. The tokenizer should expose structure, not assume a model family.

## Current Starting Point

The repository already has the important tokenizer-side pieces:

- `NoveltyAware` as the current frontier policy leader
- `EncodedDocument` as the exact frontier stream
- `FaceoffChunkLimits` and `FaceoffChunkedDocument` as the packaging layer
- `FaceoffChunkedDocument::reconstruct()` as the lossless packaging check

That means the next step is not to invent a new tokenizer. The next step is to add adapters that consume the packaged frontier.
For this phase, the adapter input should be treated as a combined wrapper such
as `ModelFacingDocument { encoded, chunked }`, because chunked output alone does
not preserve the full token-level frontier metadata required by all adapters.

## Recommended Adapter Set

### 1. Native-Tokenizer Compatibility Adapter

Purpose:

- make the packaged frontier usable with existing LMs immediately

Behavior:

- take each packaged chunk
- recover the raw text payload
- retokenize the payload using the downstream model tokenizer
- preserve chunk ordering and metadata outside the native token stream

Why it exists:

- easiest path to immediate evaluation
- good for prompt assembly, inference probes, and regression checks

Limitations:

- the downstream LM still sees its own tokenizer
- this path does not prove that fractal token ids are directly consumable by the model

### 2. Embedding-Bridge Adapter

Purpose:

- learn a bridge from fractal tokens/chunks into an existing LM embedding space

Behavior:

- accept the packaged fractal frontier
- map token ids and structural metadata into model-ready embeddings
- feed those embeddings into a frozen or partially frozen LM

Why it exists:

- this is the likely research bridge if we want to use the fractal tokenizer with current OSS models

Limitations:

- requires training
- needs a stable tokenizer ABI first

### 3. Fractal-Native Adapter

Purpose:

- provide the input contract for a future fractal-native model

Behavior:

- consume the packaged fractal frontier directly
- expose token ids, depth, and chunk metadata to the model runtime

Why it exists:

- future-proofing
- keeps the model architecture aligned with the tokenizer instead of forcing a retokenization detour

## Suggested Implementation Shape

The first implementation slice should stay small and composable.

Recommended module layout:

- `fractal-tokenizer/src/model_face/`
- `fractal-tokenizer/src/model_face/mod.rs`
- `fractal-tokenizer/src/model_face/native.rs`
- `fractal-tokenizer/src/model_face/bridge.rs`
- `fractal-tokenizer/src/model_face/fractal_native.rs`
- `fractal-tokenizer/src/model_face/batch.rs`

The model-facing layer should depend on the existing faceoff types rather than duplicating them.

Suggested flow:

1. build `EncodedDocument`
2. package it into `FaceoffChunkedDocument`
3. feed that chunked form into a `ModelAdapter`
4. obtain a model-specific batch
5. run the batch through a runtime

## Phase Plan

### Phase 1: Stable Adapter Interface + Native Compatibility

Deliverable:

- a typed adapter trait or set of traits
- a combined wrapper such as `ModelFacingDocument { encoded, chunked }`
- a minimal native-tokenizer compatibility adapter that consumes the packaged frontier
- batch types that can represent existing-LM and future-fractal inputs

Acceptance criteria:

- adapter traits compile against current faceoff types
- chunk order and reconstruction remain exact
- no tokenizer semantics leak into adapter code
- the native compatibility adapter is deterministic and validates against the current benchmark inputs
- serialization/versioning for vocabularies can remain deferred

### Phase 2: Embedding Bridge

Deliverable:

- a learned or learnable bridge interface for model embeddings

Acceptance criteria:

- clear tensor shapes
- explicit structural metadata inputs
- no hidden assumptions about a specific model family
- pad/mask semantics are explicitly specified
- train/eval ownership for vocabulary artifacts is explicitly specified

### Phase 3: Fractal-Native Runtime Contract

Deliverable:

- a model-facing batch shape for a future fractal-native model

Acceptance criteria:

- direct use of fractal token ids
- explicit chunk metadata
- same reconstruction and packaging invariants as the tokenizer side

## Validation Requirements

The adapter layer should be validated with the following checks:

- exact round-trip from raw text to encoded document and back
- deterministic packaging of encoded documents
- chunk reconstruction by concatenation
- stable adapter output for the same input
- no fallback corruption
- model-facing batch shape stability
- compatibility adapter output for existing LMs

The first adapter validation should use the existing benchmark inputs:

- stress repetition input
- mixed-domain input

## Acceptance Criteria For The First Slice

The first slice should be considered complete when:

- the adapter trait exists
- the combined wrapper exists and is the canonical handoff for the phase
- a native-tokenizer compatibility adapter exists
- the adapter consumes the current packaging layer
- tests prove exact reconstruction and stable ordering
- the abstraction does not depend on any single LM family

## Out Of Scope For The First Slice

Do not include these in the first implementation pass:

- large-scale training code
- full embedding-bridge training loops
- model-specific checkpointing
- inference orchestration for multiple backends
- production deployment plumbing

## Risks

1. Overfitting the adapter to one LM family
   - avoid by keeping the adapter input generic and typed
2. Mixing tokenizer semantics with model semantics
   - avoid by keeping the tokenizer contract separate from adapter runtime code
3. Losing exact reconstruction through packaging or batching
   - avoid by asserting reconstruction at every stage
4. Abstraction too broad for the first pass
   - avoid by starting with the native compatibility adapter and one batch shape

## Success Definition

This work is successful if the tokenizer track can hand downstream systems one stable packaged frontier and then support multiple consumers without changing the tokenizer core.

That means:

- tokenizer owns the frontier
- packaging owns chunking
- adapters own translation
- runtimes own execution

## Implementation Status

The first slice is landed and validated:

- the combined wrapper contract is implemented
- the native compatibility adapter retokenizes chunk payloads deterministically
- focused tests exercise stress and mixed-domain inputs with exact reconstruction
- the embedding bridge and fractal-native runtime remain later phases
