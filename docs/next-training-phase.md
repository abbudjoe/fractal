# Next Training Phase

This document defines the next major phase after the frozen winner-lane rerun.

It is the launch contract for the first scaled top-cohort training wave.

Use this alongside:
- [primitive-tracker.md](/Users/joseph/fractal/docs/primitive-tracker.md)
- [promotion-policy.md](/Users/joseph/fractal/docs/promotion-policy.md)
- [experiment-interface.md](/Users/joseph/fractal/docs/experiment-interface.md)
- [optimization-surfaces.md](/Users/joseph/fractal/docs/optimization-surfaces.md)
- [program-state.md](/Users/joseph/fractal/docs/program-state.md)

## Purpose

The primitive search phase has narrowed the field enough that the next question is no longer:
- which small-run primitive looks interesting?

The next question is:
- which top primitive remains the best training core under a larger, more serious, same-contract training budget?

This phase exists to answer that question without reopening broad primitive exploration.

## Current Locked Field

Scientific leader:
- `p1_fractal_hybrid_composite_v1`

Top challenger:
- `p1_fractal_hybrid_v1`

Benchmark reference:
- `p1_contractive_v1`

Reserve challenger, not in the initial top-cohort suite:
- `logistic_chaotic_map_v1`

Validation-only, not in the initial top-cohort suite:
- `p3_hierarchical_v1`
- `b2_stable_hierarchical_v1`

## Phase Principles

- The top cohort is frozen to 3 variants for this phase.
- No new bullpen species enter this phase.
- No benchmark mutation is allowed.
- Authoritative scientific comparisons must stay on one frozen commit.
- Runtime-surface changes must not be mixed into the authoritative scientific suite.
- Systems-speed reporting is required, but it is a separate output from the scientific leaderboard.

## Launch Prerequisites

The suite should launch only when all of the following are true:
- the shared branch is frozen to one commit for the entire suite
- the RunPod preservation collision fix is merged
- canonical manifests exist for the phase
- the runtime surface policy is fixed to `conservative-defaults`
- the backend is fixed across the suite
- seeds are fixed before launch

## Phase 1A: Top-Cohort Scientific Training Suite

### Exact Question

Under a larger and more serious training budget, which of the top 3 primitives is the best core on:
- scientific fitness
- stability
- perplexity
- ARC

while preserving apples-to-apples authority?

### Cohort

- `p1_fractal_hybrid_composite_v1`
- `p1_fractal_hybrid_v1`
- `p1_contractive_v1`

### Lane

- Lane: `training-top-cohort`
- Comparison class: `authoritative`

### Preset

Define a new canonical preset:
- `top_cohort_scaled_phase1`

Exact budget:
- `max_depth = 12`
- `stability_depth = 12`
- `train_batch_size = 16`
- `eval_batch_size = 8`
- `train_steps_per_species = 320`
- `eval_batches_per_family = 8`
- `perplexity_eval_batches = 8`
- `arc_eval_batches = 8`
- `learning_rate = 3e-4`
- `timeout = none`
- `execution_mode = sequential`
- `parallelism = 1`

Seeds:
- `42`
- `43`
- `44`

Backend:
- `cuda`

Runtime surface policy:
- `conservative-defaults`

### Required Outputs

Every completed row must emit:
- structured run artifact
- tournament run manifest
- wrapper manifest if cloud-run
- scientific leaderboard row
- systems-speed row
- ledger entry
- tracker-ready reminder block

### Scientific Leaderboard Fields

The authoritative scientific table for this suite is:
- `fitness`
- `stability`
- `perplexity`
- `ARC`
- `tok/s`

Primary ranking order:
1. higher `fitness`
2. higher `ARC`
3. lower `perplexity`
4. lower `stability`
5. higher `tok/s`

### Success Meaning By Variant

#### `p1_fractal_hybrid_composite_v1`

This variant succeeds if it does all of the following:
- completes all 3 seeds cleanly
- remains the mean fitness leader or within `0.01` of the mean fitness leader
- keeps `ARC >= 0.75`
- keeps mean `stability <= 0.60`

If it succeeds, it becomes:
- the primary scientific training core

#### `p1_fractal_hybrid_v1`

This variant succeeds if it does all of the following:
- completes all 3 seeds cleanly
- finishes within `0.02` mean fitness of the suite leader
- keeps `ARC >= 0.70`
- keeps mean `stability <= 0.60`

If it succeeds, it becomes:
- the primary non-composite fractal alternate core

#### `p1_contractive_v1`

This variant succeeds if it does all of the following:
- completes all 3 seeds cleanly
- finishes within `0.02` mean fitness of the suite leader
- preserves the best mean throughput in the suite
- keeps mean `stability <= 0.60`

If it succeeds, it becomes:
- the primary systems-reference training core

## Phase 1B: Systems-Speed Companion Suite

This runs as a separate reporting mode, not as the canonical scientific leaderboard.

### Exact Question

For the same top cohort and the same frozen commit:
- what is the deployment-style throughput ordering?

### Cohort

- `p1_fractal_hybrid_composite_v1`
- `p1_fractal_hybrid_v1`
- `p1_contractive_v1`

### Mode

- report mode: `systems-speed`
- authority: `advisory for scientific ranking`, `authoritative for systems-speed view`

### Rules

- same frozen commit as Phase 1A
- same backend family
- same seeds
- same runtime surface policy unless explicitly testing one validated surface

### Output

The systems-speed report must show:
- mean `tok/s`
- variance across seeds
- slowest seed
- fastest seed
- runtime surface policy used

This report is not allowed to change the scientific winner by itself.

## Decision Rules

### Promote To Primary Scientific Core

Promote one variant to `Primary Scientific Core` only if:
- it finishes all 3 seeds
- it ranks first on mean scientific fitness
- it does not lose the `ARC` comparison badly
- it is not classified as numerically fragile or runtime-cost dominated

### Keep Multiple Training Cores

Keep multiple variants active after this phase if any of the following are true:
- the top 2 variants finish within `0.02` mean fitness
- one variant wins scientific quality while another wins systems efficiency
- the top non-composite fractal core remains materially distinct from the composite shell

### Retire From Top Cohort

Remove a variant from the top cohort for the next phase if any of the following are true:
- repeated timeout on the same suite
- numeric failure
- mean fitness trails the leader by more than `0.04`
- `ARC` drops below `0.60`
- it no longer answers a distinct strategic question

## What Happens After Phase 1

If `p1_fractal_hybrid_composite_v1` still leads:
- it becomes the primary scientific core for the next scaling wave

If `p1_contractive_v1` stays within the leader band while keeping the clear throughput edge:
- it remains the primary systems-reference core

If `p1_fractal_hybrid_v1` stays close enough to the leader:
- it remains the primary non-composite fractal alternate

At the end of this phase, the project should move to:
- one primary scientific core
- one systems-reference core
- optionally one alternate fractal core

That is the maximum size of the active top cohort for the next scaling phase.

## Parallel Optimization Track

Optimization work is allowed in parallel, but only under these rules:
- it must run in a separate lane
- it must not alter the authoritative scientific suite mid-flight
- it must use explicit runtime surface labels
- it must be validated before entering the next authoritative training wave

Priority order:
1. `eval backend split`
2. `length-bucketed batching`
3. `forward-path preallocation`

## Explicit Non-Goals

This phase does not:
- reopen the bullpen
- mutate the benchmark
- admit new winner-lane species
- mix runtime-surface experiments into the authoritative scientific suite
- decide final long-term architecture from one run alone

## Expected Deliverables

By the end of this phase we should have:
- one authoritative top-cohort scientific leaderboard
- one systems-speed leaderboard
- one ledger slice describing what changed and why
- an updated tracker state
- a narrower training roadmap for the next scaling wave
