# Edge-Case Hardening Spec

## Goal

Harden the tokenizer/model-facing path against realistic ugly inputs without
changing the winning frontier policy.

This track exists to answer:

- does the current pipeline stay lossless on harder text?
- does it avoid false-positive structural collapse on noisy inputs?
- do the current model-facing adapters stay deterministic on those inputs?

## Scope

In scope:

- new focused regression tests in `src/tests.rs`
- minimal bug fixes in tokenizer/model-facing code only when a new failing test
  proves a concrete defect

Out of scope:

- new frontier policies
- new pretrained model families beyond smoke coverage already present
- embedding-bridge work
- batch contract redesign

## Failing Tests To Add First

Add these tests first, in this order:

1. `faceoff_unicode_heavy_roundtrip_is_exact`
   - multilingual text
   - emoji
   - smart punctuation
   - exact `decode()` round-trip required

2. `faceoff_json_code_log_blend_avoids_false_reuse`
   - JSON + code + logs in one input
   - exact round-trip required
   - false-positive reuse must stay low or zero

3. `faceoff_near_repetition_does_not_overcollapse`
   - repeated templates with one or two changed fields
   - exact round-trip required
   - must not collapse distinct near-duplicate spans into the same frontier too aggressively

4. `model_face_pretrained_adapter_handles_edge_case_documents_deterministically`
   - reuse the current pretrained adapter path with one hard mixed input
   - deterministic output across repeated runs
   - chunk order preserved

## Allowed Fixes

Allowed:

- span/order validation fixes
- fallback accounting fixes
- payload reconstruction fixes
- deterministic ordering fixes

Not allowed:

- changing `NoveltyAware` behavior unless a new regression proves it is wrong
- changing batch semantics

## Acceptance Criteria

- all four new tests exist and pass
- `cargo test -p fractal-tokenizer faceoff_ -- --nocapture` stays green
- `cargo test -p fractal-tokenizer model_face_ -- --nocapture` stays green
- any code change must be directly justified by a failing regression

## Notes

If a test reveals a real product bug, add the smallest durable fix and keep the
behavioral explanation in the test name and assertion text.
