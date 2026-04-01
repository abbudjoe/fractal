# Embedding Bridge Scaffold Spec

## Goal

Create the smallest typed scaffolding for a future embedding bridge from fractal
model-facing documents into existing LM input space.

This track is contract work, not training work.

## Scope

In scope:

- a new `model_face/bridge.rs`
- bridge-related re-exports
- bridge trait/interface definitions
- typed bridge batch structures
- focused tests in `src/tests.rs`

Out of scope:

- training loops
- model execution
- backend-specific tensor code
- optimizer or checkpoint logic

## Failing Tests To Add First

Add these tests first, in this order:

1. `model_face_bridge_batch_from_document_preserves_order_and_spans`
   - bridge batch creation preserves document order
   - chunk order and byte spans remain explicit

2. `model_face_bridge_batch_preserves_structural_metadata`
   - bridge entries preserve token id, depth, chunk index, and span length metadata

3. `model_face_bridge_batch_is_deterministic_for_same_input`
   - repeated conversion of the same model-facing batch yields identical bridge batches

4. `model_face_bridge_contract_is_model_family_agnostic`
   - the bridge interface should not hardcode Llama/Mistral/Qwen specifics
   - compile-time/type-level contract only

## Required Design Shape

The bridge should stay abstract.

Suggested types:

- `BridgeFeatureToken`
- `BridgeFeatureChunk`
- `BridgeBatch`
- `EmbeddingBridgeAdapter` trait

Suggested preserved fields:

- fractal token id
- token kind
- depth
- byte span
- chunk index
- document index

If padding is needed, consume the existing batch/runtime contract rather than
inventing a second padding system here.

## Acceptance Criteria

- bridge scaffold compiles
- tests exist and pass
- no model-family-specific logic appears in the bridge layer
- no training/runtime implementation is added

## Notes

This track should stop at the boundary where a later experiment can plug in a
learned projection without redesigning the tokenizer or batch contracts.
