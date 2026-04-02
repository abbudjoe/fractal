# Recurrent Projection Layout Checklist

This checklist defines the required design and validation gates before changing the
recurrent projection surface used by the p1 family.

This is not a permission slip to start patching.

It is a pre-change control document for:
- backward/layout bug repair
- checkpoint/export/load safety
- optimizer safety
- diagnostics continuity
- canary discipline

Use this alongside:
- [AGENTS.md](/Users/joseph/fractal/AGENTS.md)
- [ENGINEERING.md](/Users/joseph/fractal/ENGINEERING.md)
- [diagnostics-surface-contract.md](/Users/joseph/fractal/docs/specs/diagnostics-surface-contract.md)
- [export-load-failure-snapshot-contract.md](/Users/joseph/fractal/docs/specs/export-load-failure-snapshot-contract.md)
- [stage0-backward-oom-canary-4-2-26-20260402T092936-0600.md](/Users/joseph/fractal/docs/postmortems/stage0-backward-oom-canary-4-2-26-20260402T092936-0600.md)

## Why This Exists

The Stage 0 H200 debug canary did not fail because the model simply “does not fit.”

The current strongest evidence is:
- forward completes
- loss completes
- the crash happens at `backward_start`
- the failing operation is a BF16 matmul with a non-contiguous transposed RHS
- the recurrent graph burden is very large

That points to a layout/backward primitive failure in a hot recurrent projection seam.

The fix must not be a `p1_contractive`-local trick.

It must become a standalone projection abstraction with explicit ownership of:
- parameter storage
- layout policy
- execution policy
- diagnostics surface

## Scope

This checklist is for the first implementation pass on the p1 path:
- `p1_contractive`

The abstraction should be reusable by the other top p1 variants later, but the first
repair should stay scoped to the p1 recurrent projection seam and should not broaden into
an all-primitives refactor.

## Doctrine Requirements

This work must follow [AGENTS.md](/Users/joseph/fractal/AGENTS.md) and
[ENGINEERING.md](/Users/joseph/fractal/ENGINEERING.md).

That means:
- no bandaids
- no primitive-local tensor-layout hacks
- no hidden dtype changes
- no hidden batch/depth changes
- no silent checkpoint/load breakage
- no duplicated trainable params
- no diagnostic regression

Required design rules:
- the projection surface must be standalone and typed
- the primitive must depend on the abstraction, not embed execution/layout logic inline
- the abstraction must preserve current math unless a separate design decision says otherwise
- the abstraction must remain compatible with Burn module traversal
- the canary must not be the first place we learn whether the change broke the primitive

## High-Level Design Requirements

### 1. Standalone projection abstraction

The new structure should be a first-class reusable projection module, not a private helper
inside [p1_contractive.rs](/Users/joseph/fractal/fractal-primitives-private/src/primitives/p1_contractive.rs).

The primitive should declare intent such as:
- gate projection
- state-mix projection
- input-mix projection

The abstraction should own:
- parameter storage
- layout policy
- forward execution semantics
- backend-sensitive execution strategy

### 2. Fractal-like compositional pattern

The design should be compositional and reusable rather than special-cased.

That means:
- one reusable projection surface
- multiple concrete projection instances
- stable typed identity for each instance
- no ad hoc “special linear just for `w_h`” branches

The primitive remains the owner of recurrence math.

The projection abstraction becomes the owner of projection execution/layout semantics.

### 3. Programmable policy, not hard wiring

The abstraction should support explicit typed policy for:
- storage orientation
- effective execution orientation
- backend-specific strategy selection
- optional derived execution metadata

The policy must be visible in code as typed state, not spread across match arms and comments.

## Non-Negotiable Invariants

### 1. One canonical trainable weight

There must be exactly one canonical trainable parameter for a learned projection weight.

If an alternate orientation or cached form exists, it must not become a second trainable
parameter.

Checklist:
- [ ] no duplicate trainable params introduced
- [ ] any cached alternate layout is non-parameter state
- [ ] optimizer, grad clipping, and weight decay still see exactly the intended params

### 2. Preserve primitive math

The first implementation pass must preserve the current p1 recurrence semantics.

Checklist:
- [ ] output math remains equivalent within tolerance
- [ ] bias semantics are unchanged
- [ ] initialization semantics are unchanged unless intentionally and explicitly redesigned
- [ ] gate/state/input projection roles remain unchanged

### 3. Preserve module-tree compatibility

The runtime relies on Burn module traversal for:
- optimizer step
- gradient collection
- gradient clipping
- checkpoint save/load
- weight export/load
- failure-snapshot model-weight capture

Checklist:
- [ ] the new abstraction participates cleanly in `Module<B>`
- [ ] autodiff behavior still works through the module tree
- [ ] checkpoint save/load remains valid
- [ ] weight export/load remains valid
- [ ] failure snapshot model-weight capture remains valid when enabled

### 4. Preserve precision contract

The fix must not silently change:
- active compute precision
- optimizer precision
- reduction precision

Checklist:
- [ ] no hidden FP32 fallback
- [ ] no hidden BF16 disable
- [ ] no silent precision branching outside typed policy

### 5. Preserve diagnostics continuity

The new abstraction must not hide the exact signals we just added.

Checklist:
- [ ] projection identity remains typed and stable
- [ ] layout contract remains observable
- [ ] rule-level diagnostics still identify gate/state/input projections
- [ ] failure snapshots still preserve last diagnostic and last rule-projection event

## Blast-Radius Checklist

Before coding, confirm impact on all of the following:

- [ ] primitive fields and naming
- [ ] Burn module derivation / manual module implementation
- [ ] optimizer step semantics
- [ ] gradient clipping traversal
- [ ] checkpoint serialization paths
- [ ] weight export metadata and load paths
- [ ] failure snapshot artifact shape
- [ ] diagnostics event shape and summaries
- [ ] test fixtures that assume `Linear`
- [ ] proving-run comparability for `p1_contractive`

## Design Questions To Settle Before Implementation

These should be answered explicitly before writing code:

- [ ] Where does the abstraction live?
  - likely `fractal-core` if it is a durable runtime building block
  - not inside a single primitive file
- [ ] What is the public type name?
  - it should describe its ownership clearly
- [ ] Is the layout policy encoded in the type, in a typed config, or both?
- [ ] How are canonical parameter names preserved for checkpoint/export compatibility?
- [ ] If alternate execution layouts exist, how are they represented without becoming parameters?
- [ ] How does the abstraction emit layout diagnostics without making primitives know backend details?

## Minimum Validation Ladder

The implementation is not ready until it clears all of these gates.

### Gate 1: semantic/unit equivalence

- [ ] small CPU fixture compares old/new projection outputs within tolerance
- [ ] small CPU fixture compares old/new p1 recurrence outputs within tolerance
- [ ] load/save roundtrip still succeeds
- [ ] export/load roundtrip still succeeds

### Gate 2: narrow CUDA repro

Do not rerun a full canary first.

Use a one-step CUDA/H200 repro with the same essential semantics:
- same primitive
- same precision
- same batch size
- same recursion depth
- one train step only

And use a sequence-length staircase to find the shortest faithful reproducer.

Checklist:
- [ ] one-step repro exists
- [ ] repro confirms whether the backward/layout failure seam moved or disappeared
- [ ] diagnostics remain intact on the repro lane

### Gate 3: proving runs for `p1_contractive`

Before another canary:

- [ ] baseline proving run
- [ ] mid-stress proving run
- [ ] high-stress proving run

These are required because this is a low-level execution-structure change and we must prove
we did not accidentally degrade the primitive.

### Gate 4: canary eligibility review

Only after the above are green:

- [ ] compare stability/perplexity/ARC/throughput against the current `p1_contractive` expectation
- [ ] confirm no contract regressions in artifacts/checkpoints/exports/diagnostics
- [ ] freeze a new debug canary manifest
- [ ] rerun H200 canary

## What Not To Do

- [ ] do not mix this patch with TBPTT
- [ ] do not mix this patch with checkpointing changes
- [ ] do not mix this patch with router redesign
- [ ] do not mix this patch with precision changes
- [ ] do not mix this patch with primitive-math changes
- [ ] do not send it straight to canary from unit tests alone

## Success Condition

This work is successful only if all of the following are true:

1. the recurrent projection seam becomes a clean standalone abstraction
2. `p1_contractive` math is preserved
3. checkpoint/export/load/diagnostics remain healthy
4. the narrow backward/layout repro improves or moves meaningfully
5. `p1_contractive` survives baseline, mid-stress, and high-stress proving runs
6. only then do we spend another canary

## Bottom Line

The next implementation should not be:
- a local `p1_contractive` hack
- a hidden execution workaround
- a leap to a new backprop regime

It should be:
- one clean projection abstraction
- one clean p1 integration
- one disciplined proving ladder before another canary
