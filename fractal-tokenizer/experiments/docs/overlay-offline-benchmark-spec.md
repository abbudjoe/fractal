# Overlay Offline Benchmark Spec

## Purpose

This spec defines the primary evaluation path for the current canonical
tokenizer + recursive overlay line.

At this stage, the overlay still materializes back to the exact same canonical
prompt text before any model consumes it. That means the current question is
not:

- does the overlay improve model quality?
- does the overlay reduce model-side compute inside the serving runtime?

The real question is:

- does the overlay earn its keep as an exact structured-text transport layer?

This benchmark is designed to answer that question honestly.

## Core Principle

The current primary benchmark is **model-free**.

We do keep a tiny local model smoke as a required runtime gate, but the main
evaluation path should not depend on model outputs because:

- the model sees identical materialized text on both paths
- model-side request time mostly adds noise at this phase
- the real leverage is currently in transport efficiency and exactness

So the benchmark contract is:

1. prove exact expansion
2. measure transport benefit
3. measure client-side overhead
4. verify neutral behavior off-target
5. keep a small model smoke as an integration canary

## Questions This Benchmark Must Answer

### 1. Does the overlay win where it should?

On repetitive structured text, does the overlay produce meaningful net
transport savings?

### 2. Does it stay exact?

Do all prepared overlay transports expand back to canonical token ids without
error or drift?

### 3. Does it stay narrow?

On code, prose, and multilingual control buckets, does the overlay stay
neutral rather than turning into dead weight?

### 4. Is the client-side tax acceptable?

Do discovery, packing, and materialization overhead remain modest relative to
the transport win on the target buckets?

## Non-Goals

This benchmark must **not** claim any of the following yet:

- improved model quality
- reduced model-side prompt processing cost
- universal tokenization improvement
- general code/prose superiority
- serving/runtime gains after transport crosses the model boundary

Those claims need a later benchmark after deeper runtime integration.

## Benchmark Families

### Primary Win Buckets

These are the actual target domains for the current overlay architecture:

- `jsonl.signals`
- `logs.operational_mixed`

If available in the local corpus, we may add one more structured bucket such
as:

- config-like repetitive text

but only if the bucket is typed clearly and evaluated separately.

### Neutral Control Buckets

These guard against dead weight and accidental over-broad activation:

- `docs.spec`
- `external.prose.web`
- `external.code.python`
- `external.code.js_ts`
- `external.multilingual`

These are not optimization targets for the current phase.

## Required Runtime Gate

A tiny local Ollama smoke remains mandatory, but only as an integration gate.

Required smoke surfaces:

- embedding path:
  - exact overlay materialization to canonical text
  - successful local request
- instruct path:
  - exact overlay materialization to canonical text
  - deterministic output equality between base and overlay-materialized prompts

This smoke is a canary, not the main benchmark.

If the smoke fails, the offline benchmark result is not actionable because the
runtime seam is not trustworthy.

## Benchmark Config

The benchmark should use the current stable overlay line:

- canonical tokenizer: current production-proven native tokenizer path
- overlay mode: `local-record-macro`
- transport scope: `batch_local`
- profitability gating: enabled
- factorized shared definitions: enabled
- structure-aware batching:
  - measured optionally
  - not required as the primary score axis

This should reflect the current best exact structured-text overlay, not older
document-local-only baselines.

## Primary Metrics

### Exactness

- `exact_failures`
- `expansion_mismatch_docs`

Success bar:

- `exact_failures = 0`
- `expansion_mismatch_docs = 0`

### Transport Value

- `transport_ratio`
- `definition_overhead_rate`
- `macro_ref_symbols`
- `macro_definition_symbols`
- `transport_symbols`
- `canonical_tokens`

### Activation

- `activation_docs`
- `macro_hit_docs`
- `batch_local_hit_docs`

These are supporting metrics only.

Activation without net transport value is not a win.

### Client-Side Cost

- `overlay_discovery_ms`
- `overlay_pack_ms`
- `overlay_materialize_ms`
- `overlay_client_overhead_ms`

If measured as distributions:

- median
- p95

### Net Value

For the primary win buckets, the useful read is:

- transport gain stays high
- definition overhead stays bounded
- client overhead stays materially smaller than the value gained by the
  transport reduction

This does not need to be reduced to a single scalar yet.

## Success Bars

### Required

- exactness remains perfect
- `jsonl.signals` stays clearly above `2.0`
- `logs.operational_mixed` stays clearly above `1.5`
- neutral control buckets remain effectively neutral
- instruct smoke remains deterministic and exact

### Strong

- overall hybrid `batch_local transport_ratio >= 1.40`
- overall hybrid `definition_overhead_rate <= 0.10`
- target-bucket client overhead remains modest relative to warmed request time

### Failure Conditions

Treat the overlay as regressing if any of the following appear:

- any exact expansion failure
- control buckets materially worse than neutral
- transport ratio rises only because definition overhead or prep cost explodes
- activation broadens without corresponding structured-bucket gain

## Reporting

Every benchmark run should report:

- corpus/source family
- bucket
- canonical token count
- transport symbol count
- transport ratio
- definition overhead rate
- activation docs
- macro-hit docs
- median client overhead metrics

And a short summary:

- target buckets improved / stable / regressed
- control buckets neutral / regressed
- exactness pass/fail
- runtime smoke pass/fail

## Why This Is The Right Benchmark Now

This benchmark matches the architecture we actually have.

The overlay is currently:

- exact
- reversible
- narrow
- strongest on repetitive structured text

So the benchmark should judge it on those properties.

The mistake would be to benchmark it as if it were already:

- a universal tokenizer replacement, or
- a direct model-side compression format

That would repeat the same scope drift that hurt the earlier tokenizer line.

## Next Benchmark After This One

Only after the overlay crosses the runtime boundary without immediate
materialization should we promote a model-backed benchmark to primary status.

That later benchmark can ask:

- does the overlay reduce serving-side transport cost?
- does it improve effective context efficiency?
- does it reduce prompt processing cost in practice?

Until then, the offline benchmark in this spec is the primary truth test.
