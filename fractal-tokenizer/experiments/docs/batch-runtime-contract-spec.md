# Batch Runtime Contract Spec

## Goal

Finish the minimum batch/runtime contract needed before real downstream model
work, without building training code.

This track exists to make the model-facing path explicit about:

- truncation
- overflow behavior
- multi-document collation rules
- mask semantics
- shape stability

## Scope

In scope:

- `model_face/native.rs`
- `model_face/batch.rs`
- `model_face/mod.rs`
- `lib.rs` re-exports
- focused contract tests in `src/tests.rs`

Out of scope:

- embedding-bridge logic
- new tokenizer policies
- pretrained tokenizer acquisition

## Failing Tests To Add First

Add these tests first, in this order:

1. `model_face_native_collation_empty_batch_is_stable`
   - empty batch collates deterministically
   - sequence length is well-defined

2. `model_face_native_collation_single_document_roundtrip_metadata_is_stable`
   - single-doc batch preserves document index, chunk index, masks, and valid lengths

3. `model_face_native_collation_rejects_overflow_without_truncation_policy`
   - introduce an explicit truncation/overflow contract
   - default behavior must reject overflow instead of silently truncating

4. `model_face_native_collation_truncates_with_explicit_policy`
   - add one explicit truncation policy
   - masks and metadata must remain consistent after truncation

5. `model_face_native_collation_multi_document_order_is_stable_under_truncation`
   - multi-doc order must remain deterministic
   - document/chunk provenance must survive collation

## Required Design Shape

Prefer typed policy over booleans:

- `NativeTruncationPolicy` enum
- `NativeOverflowPolicy` enum if needed
- explicit valid-length metadata after truncation

The contract must make it impossible to silently truncate without the caller
choosing that behavior.

## Acceptance Criteria

- the new tests exist and pass
- mask semantics stay explicit:
  - real token => `true`
  - padding => `false`
- truncation semantics are typed and deterministic
- no silent lossy behavior is introduced by default

## Notes

This track should improve the batch boundary used by later model code, not
rework tokenizer logic.
