# DREEGMOR Recurrent Router Plan

Status: exploratory control-plane scaffold with a runnable CPU-only `A` lane

This surface is separate from:

* the main `v3a` / Path 1 proving line
* the current one-shot DREEGMOR experiment surface

All comparative reads on this surface should follow the shared rubric in
[`/Users/joseph/fractal-worktrees/goe-a-am-experiments/docs/specs/dreegmor-experiment-rubric.md`](/Users/joseph/fractal-worktrees/goe-a-am-experiments/docs/specs/dreegmor-experiment-rubric.md).

## Purpose

The question here is narrower than “do graphs help?”

The question is:

* does a tiny recurrent router improve expert control more than a one-shot
  router on the same frozen `A` / `A + M` backbone families?

This follows the recurrent-routing thesis directly:

* recurrence may be more useful in the routing/control plane than as a static
  graph prior

## Boundary

This plan treats recurrent routing as a **control primitive**, not a predictive
backbone primitive.

That is the key distinction from the current `v3a` primitive line:

* `A + M` and `A + P2` use recurrent/selective primitives to mix sequence
  content inside the predictive stack
* a recurrent router uses a small recurrent controller state to decide expert
  usage

So these surfaces may share primitive families, but they do **not** serve the
same role.

## Initial Contract

Held fixed for the first recurrent-router probe:

* same frozen `A` and `A + M` backbone surfaces already used by DREEGMOR
* same byte-level corpus contract
* same smoke-train harness style where possible
* same `2`-expert channel count
* no `P2` in the backbone
* no external memory sidecar
* no main `v3a` runner integration

Changed only in the controller:

* routing becomes multi-round instead of one-shot
* controller state is explicit and typed
* recurrence is in the control plane

## Minimal Design

The first recurrent-router design is intentionally narrow:

* a single virtual-node controller state
* `2` routing rounds by default
* dense softmax routing first
* `GRU` virtual node first
* optional expert-feedback path as a second recurrent ablation

Why this first:

* it is the smallest clean test of the recurrent-routing thesis
* it avoids immediately mixing sparse-top-`k`, expert-graph structure, and
  recurrent control into one opaque design

## Typed Core Surface

The initial typed scaffold lives in:

* [`/Users/joseph/fractal-worktrees/goe-a-am-experiments/fractal-core/src/hybrid_attention/recurrent_router.rs`](/Users/joseph/fractal-worktrees/goe-a-am-experiments/fractal-core/src/hybrid_attention/recurrent_router.rs)

Implemented typed contracts:

* `RecurrentRouterSpec`
* `VirtualNodeRecurrentRouter`
* `RecurrentRouterPrimitiveKind`
* `RecurrentRouterFeedbackMode`
* `RecurrentRouterSelectionMode`
* `minimal_recurrent_router_experiment_matrix()`

The initial narrow defaults are:

* `2` rounds
* state width `64`
* dense softmax selection
* `GRU` virtual node
* channel count fixed to the current DREEGMOR `2`-channel lane

## Minimal Experiment Matrix

The first comparison should stay tiny and falsifiable:

1. `dreegmor-one-shot-dense`
   Current one-shot dense controller baseline.

2. `dreegmor-recurrent-dense`
   Virtual-node recurrent router with controller-state-only recurrence.

3. `dreegmor-recurrent-dense-feedback`
   Same recurrent router family, but with aggregated expert-output feedback into
   the controller.

Do **not** add these yet:

* sparse top-`k`
* per-expert controller states
* explicit expert graph topology
* larger expert counts
* backbone changes

Those belong only after the dense recurrent controller proves something real.

## What Is Implemented Now

Implemented now:

* typed recurrent-router control-plane contract in core
* typed minimal experiment matrix helper in core
* validation tests that keep the initial design narrow
* runnable `dreegmor-recurrent-router-experiment` CPU bin for:
  * `A`
  * one-shot `DREEGMOR(A)`
  * recurrent `DREEGMOR-Recurrent(A)`
* isolated run directories per variant under a run-scoped artifact root
* round-by-round routing diagnostics on the shared routing summary seam
* explicit split between:
  * `--seed` for model/init RNG
  * `--data-seed` for optional train-order shuffling
  * fixed data order by default for apples-to-apples comparisons

Still stubbed:

* no `A + M` recurrent runtime lane yet
* no recurrent expert-feedback implementation
* no sparse top-`k` recurrent-routing lane

## Run

CPU smoke comparison:

```bash
cargo run --bin dreegmor-recurrent-router-experiment -- --variant all --steps 8 --eval-batches 2
```

Single recurrent lane only:

```bash
cargo run --bin dreegmor-recurrent-router-experiment -- --variant recurrent-a --steps 8 --eval-batches 2
```

For throughput or memory comparisons, do **not** use shared-process
`--variant all` runs. Use one fresh process per variant instead.

## Success Criteria

The recurrent-router line only earns further investment if it shows:

* better loss than the one-shot dense router
* non-trivial route changes across rounds
* stable behavior across seeds
* no expert collapse
* controller overhead small enough to justify the gain

If the one-shot router matches it, the recurrent-router design should not be
treated as the better controller by default.
