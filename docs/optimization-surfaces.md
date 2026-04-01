# Optimization Surfaces

This document defines the optimization seams we should build around the primitive harness
without changing primitive behavior or muddying the current winner-lane bakeoff.

Use this alongside:
- [experiment-interface.md](/Users/joseph/fractal/docs/experiment-interface.md)
- [program-state.md](/Users/joseph/fractal/docs/program-state.md)
- [harness-doctrine.md](/Users/joseph/fractal/docs/harness-doctrine.md)
- [harness-hardening-checklist.md](/Users/joseph/fractal/docs/harness-hardening-checklist.md)
- [promotion-policy.md](/Users/joseph/fractal/docs/promotion-policy.md)
- [primitive-tracker.md](/Users/joseph/fractal/docs/primitive-tracker.md)

## Purpose

The goal is to create optimization surfaces now so we can improve throughput later
without:
- changing primitive math
- changing leaderboard semantics
- blurring train and eval behavior
- forcing experimental and systems work into the same change

This is a control-plane spec, not permission to optimize the live bakeoff branch in place.

## Optimization Doctrine

These surfaces exist to answer one question:

How do we make the harness and eventual production path faster without accidentally
changing what the primitive is doing?

So every optimization surface must obey these rules:
- primitive math stays unchanged
- leaderboard semantics stay unchanged
- training and eval semantics stay separable
- systems wins must be attributable to a named runtime surface
- a faster run is not a better primitive unless it wins again on a controlled rerun

## Runtime Invariants

The following invariants are non-negotiable for all optimization work:
- the `FractalRule` contract does not change
- recursion depth policy does not silently change
- router semantics do not silently change
- train/eval dataset content does not silently change
- leaderboard scoring stays on the current scientific path unless explicitly revised
- any new optimization defaults remain disabled until validated on a frozen commit

## Surface Registry

| Surface | Purpose | Default State | Comparison Risk | First Rollout Priority |
|---------|---------|---------------|-----------------|------------------------|
| Eval backend split | remove autodiff overhead from eval | disabled | medium | 1 |
| Batching policy | reduce padding waste | disabled | medium | 2 |
| Forward execution policy | reduce token-loop overhead | disabled | medium-high | 3 |
| Buffer/allocation reuse | reduce allocator churn | disabled | low-medium | 4 |
| Throughput benchmark mode | separate science from deployment benchmarking | disabled | low | 5 |
| Backend policy | keep backend choices explicit and typed | disabled | low | 6 |
| Optimization metadata | preserve interpretability | enabled once surfaces exist | low | continuous |

## Discipline Rule

While a winner-lane bakeoff is live:
- surfaces may be specified and implemented on isolated branches
- defaults must remain unchanged on the shared branch
- no optimization should change the meaning of an active comparison
- no optimization result should be treated as authoritative until re-run on a frozen commit

## Scope

In scope:
- runtime and execution surfaces
- inference/eval backend separation
- batching policy surfaces
- memory reuse and output buffering seams
- throughput benchmarking surfaces
- backend policy seams

Out of scope:
- primitive mutations
- leaderboard scoring changes
- new bullpen species
- tokenizer-only work

## Rollout Model

Every surface should move through the same rollout states:

1. `specified`
   - document exists
   - contracts are explicit
2. `implemented-disabled`
   - code exists behind an explicit non-default policy
   - manifests record the surface when enabled
3. `validated`
   - same-preset reruns show no material metric drift
   - throughput or latency improvement is measurable
4. `candidate-default`
   - enough evidence exists to consider a default flip
5. `default`
   - only after a frozen-commit comparison confirms the behavior

No surface should skip directly from `specified` to `default`.

## Why These Surfaces Matter

Current throughput is measured in `arc_speed` and reflects:
- primitive compute
- runtime control overhead
- backend overhead
- padding waste
- batch utilization

We need explicit seams so we can improve the last four without accidentally changing the first.

## Surface 1: Eval Backend Split

### Goal

Train on the autodiff backend, but evaluate perplexity and ARC/speed on an inference-oriented backend.

### Why

This is the highest-value likely optimization that should preserve primitive behavior.

### Surface

Add an explicit eval backend policy:
- `shared_backend`
- `inner_backend`
- future:
  - `candle_inference`
  - `cuda_inference`

### Contract

- training semantics remain unchanged
- eval uses equivalent weights and output semantics
- eval backend selection is explicit in config and manifests
- current default remains the current behavior until we intentionally switch
- any numerical drift must be treated as a backend-policy effect, not a primitive win

### Acceptance

- same-preset eval metrics match current behavior within a tight tolerance
- phase timing shows improved perplexity and/or ARC/speed throughput
- manifests record which eval backend policy was used

### Primary risks

- hidden dtype / backend numerical drift
- accidental mismatch between trained and evaluated weights
- backend change being mistaken for a primitive quality change

### Primary files

- `fractal-core/src/registry.rs`
- `fractal-core/src/lifecycle.rs`
- `src/run_artifacts.rs`

## Surface 2: Batching Policy

### Goal

Separate primitive quality from padding and packing inefficiency.

### Why

`tok/s` counts valid tokens, so wasted padding hurts throughput without improving metrics.

### Surface

Add an explicit batching policy:
- `padded`
- `length_bucketed`
- future:
  - `packed`

### Contract

- example content does not change
- labels do not change
- ordering changes must be controlled and reproducible
- manifests record batching policy
- authoritative comparisons must record whether batching policy matches

### Acceptance

- same eval examples, same outputs, lower padding waste
- same-preset quality metrics stay materially unchanged
- throughput improves on the same backend

### Primary risks

- changing example order in a way that breaks determinism
- conflating batching policy wins with primitive wins
- introducing hidden data-loader heuristics outside config

### Primary files

- `fractal-core/src/data_generator.rs`
- `fractal-core/src/registry.rs`
- `fractal-core/src/lifecycle.rs`

## Surface 3: Forward Path Execution Policy

### Goal

Create a seam for optimizing the hot path in `forward_tokens` without changing the primitive contract.

### Why

The current path is intentionally simple, but it still pays:
- per-token loop overhead
- repeated reshapes/narrows
- output concatenation overhead

### Surface

Add an internal execution policy for token rollout:
- `simple_loop`
- `preallocated_output`
- future:
  - `captured_graph`
  - `fused_rollout`

### Contract

- primitive `apply` contract remains unchanged
- recursion behavior remains unchanged
- router semantics remain unchanged
- policy is runtime-owned, not primitive-owned
- execution policy must be visible in artifacts and manifests when enabled

### Acceptance

- no metric drift on the same seed/preset
- lower wall-clock in `perplexity` and `arc_speed`
- no new hidden contracts in primitive files

### Primary risks

- silent changes to token rollout semantics
- accidental buffer aliasing bugs
- speed wins that only appear on one backend or one sequence shape

### Primary files

- `fractal-core/src/model.rs`
- `fractal-core/src/registry.rs`

## Surface 4: Buffer and Allocation Reuse

### Goal

Reduce allocator churn and host/device setup overhead during eval.

### Why

At current scale, the harness is still partly control-plane bound.

### Surface

Add explicit reuse surfaces for:
- eval batch device buffers
- output tensor staging
- scratch allocations used by repeated eval loops

### Contract

- no mutation of logical batch contents
- reuse is explicit and owned by the runtime
- no stale state leaks between runs
- cache lifetime must be bounded by run ownership

### Acceptance

- reduced phase times without metric drift
- deterministic replay still works
- no hidden cache invalidation bugs

### Primary risks

- stale tensor reuse across runs or seeds
- device-specific invalidation bugs
- accidental train/eval state coupling

### Primary files

- `fractal-core/src/registry.rs`
- `fractal-core/src/data_generator.rs`

## Surface 5: Throughput Benchmark Mode

### Goal

Separate leaderboard metrics from deployment-style throughput benchmarking.

### Why

The current `tok/s` is useful, but it is not the same as deployment decode latency.

### Surface

Add a distinct benchmark mode that can measure:
- eval throughput
- decode-style throughput
- latency-oriented runs

with explicit benchmark contracts.

### Contract

- leaderboard scoring continues to use current scientific metrics unless intentionally revised
- benchmark mode records:
  - batch size
  - sequence length
  - recursion depth policy
  - backend
  - warmup behavior
- benchmark mode must never silently replace the leaderboard path

### Acceptance

- throughput experiments no longer need to overload leaderboard runs
- benchmark outputs are preserved in structured artifacts
- benchmark mode never silently becomes the leaderboard path

### Primary risks

- users comparing benchmark-mode speed directly against leaderboard rows
- deployment-style decode benchmarks silently inheriting eval semantics
- extra benchmark code path drifting from the real model path

### Primary files

- `examples/tournament.rs`
- `src/run_artifacts.rs`
- reporting surfaces under `src/`

## Surface 6: Backend Policy

### Goal

Make backend choices explicit so systems work can proceed without changing primitive files.

### Surface

Add a typed backend policy that can later express:
- training backend
- eval backend
- benchmark backend

### Contract

- backend choices are recorded in manifests
- backend changes do not masquerade as primitive improvements
- current default remains stable until an explicit comparison is run
- backend policy must stay typed and central, not scattered across CLI flags and wrappers

### Acceptance

- same primitive can be benchmarked across backend policies
- reports distinguish backend effects from primitive effects

### Primary risks

- overlapping backend controls at wrapper, CLI, and runtime layers
- "same run" results becoming incomparable because backend changed implicitly

### Primary files

- `fractal-core/src/lifecycle.rs`
- `examples/tournament.rs`
- `src/run_artifacts.rs`

## Surface 7: Optimization Metadata

### Goal

Record enough optimization context that later throughput comparisons remain interpretable.

### Required metadata

- eval backend policy
- batching policy
- execution policy
- explicit eval budget
- benchmark mode if any
- backend type
- commit SHA

### Acceptance

- a future speed improvement can be traced to the runtime surface that caused it
- we can distinguish primitive wins from runtime wins

## Validation Protocol

Any optimization surface that touches runtime behavior must pass all of:
- unit or integration coverage for config compatibility
- manifest/artifact preservation of the new surface metadata
- at least one same-preset rerun against a frozen commit
- explicit review of whether the result is authoritative or advisory

If the optimization affects only tooling or benchmark mode, it may skip the scientific rerun
but must still preserve manifests and artifacts correctly.

## Implementation Order

Recommended order:
1. eval backend split
2. batching policy
3. forward path execution policy
4. buffer/allocation reuse
5. throughput benchmark mode
6. backend policy cleanup

This order is chosen because it maximizes likely throughput gains before deeper runtime surgery.

## First Candidate Branches

When the live bakeoff is finished and classified, the first isolated implementation branches should be:

1. `codex/opt-eval-backend-split`
   - add typed eval backend policy
   - no default flip
2. `codex/opt-length-bucketing`
   - add `length_bucketed` policy
   - no default flip
3. `codex/opt-forward-prealloc`
   - add `preallocated_output` execution policy
   - no default flip

These three cover the most likely throughput gains without touching primitive math.

## Merge Policy

Before the live bakeoff finishes:
- spec work may merge
- tooling/docs may merge
- disabled or non-default surfaces may merge only if they cannot perturb active runs
- no default optimization switches should merge

After the live bakeoff finishes:
- merge one optimization surface at a time
- rerun same-preset comparisons on a frozen commit
- update the tracker only after the re-run, not from expectation

## Decision Gate After Each Surface

After each surface lands on an isolated branch, answer exactly one question:

- did throughput improve?
- did any scientific metric drift materially?
- is the result authoritative or only advisory?
- should the surface stay disabled, remain opt-in, or be considered for default?

If those questions cannot be answered cleanly, the surface is not ready to merge.

## Success Criteria

This spec is paying off when:
- we can make the harness faster without touching primitive math
- throughput improvements are attributable and reproducible
- the primitive leaderboard stays scientifically interpretable
- systems optimization stops competing with mutation design for the same control plane
