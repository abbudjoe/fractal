# Implementation Plan — Recursive Memory Kernel v1

## Status

Draft

## Purpose

This document turns the [Recursive Memory Kernel v1 RFC](./recursive-memory-kernel-v1_revised.md) into an execution plan for the first implementation series.

It is intentionally narrower than the full architecture spec.

Its purpose is to guide the first serious `fractal-v2` build toward something that is:

* causally correct
* observable
* ablatable
* narrow enough to debug
* hard to “decorate into success” without real evidence

## Related docs

* [Recursive Memory Kernel v1 RFC](./recursive-memory-kernel-v1_revised.md)
* [Recursive Memory Kernel v1 Checklist](./recursive-memory-kernel-v1_checklist.md)
* [Revision Suggestions](./revision_suggestions.md)
* [ENGINEERING](../../ENGINEERING.md)

Where this plan conflicts with broad ambition, prefer the narrower path.

---

## Implementation principles

### 1. Prove the tree-only design first

The first serious v1 should include:

* multi-root local processing
* sealed leaf blocks
* regular dyadic summaries
* sparse tree retrieval
* exact local reads within selected leaves

It should **not** initially include:

* side-memory bank
* learned merge scheduling
* routing early stop
* learned eviction
* dense fallback attention

### 2. Preserve falsifiability

Every major path must be individually removable or zeroable:

* extra roots
* tree summaries
* sparse routing
* exact leaf read

If a component cannot be ablated cleanly, it is not ready.

### 3. Prefer regular structure over learned policy

For the first runnable version:

* fixed leaf size
* regular dyadic tree
* fixed beam width
* no learned merge cadence
* no learned early stop

Learned policies may come later, but not before the base architecture is stable.

### 4. Favor backend-friendly storage

Hot paths should prefer:

* batched tensors
* flat stores
* level-major tree layouts
* explicit span metadata

Avoid:

* pointer-heavy recursive structures
* heap-heavy per-node ownership in hot loops
* hidden global mutable state

---

## Expected module map

The exact filenames may shift, but the ownership boundaries should land close to this:

* `fractal-core/src/v2/model.rs`
* `fractal-core/src/v2/state.rs`
* `fractal-core/src/v2/local_trunk.rs`
* `fractal-core/src/v2/leaf.rs`
* `fractal-core/src/v2/tree.rs`
* `fractal-core/src/v2/router.rs`
* `fractal-core/src/v2/read_fusion.rs`

Suggested exports:

* `FractalV2Model`
* `FractalV2State`
* `LocalTrunk`
* `LeafSummarizer`
* `TreeMergeCell`
* `FractalRouterHead`
* `ReadFusion`

If you keep different filenames, preserve the ownership boundaries.

---

## Execution phases

## Phase 1 — Scaffold v2 module boundaries

### Goal

Create explicit v2 architecture surfaces without disturbing `recursive-kernel-v1`.

### Deliverables

* `LocalTrunk`
* `LeafSummarizer`
* `TreeMergeCell`
* `FractalRouterHead`
* `ReadFusion`
* `FractalV2Model`

### Notes

* Keep this phase mostly structural.
* Avoid sneaking behavior policy into generic traits too early.
* Do not try to force everything through the current `FractalRule` abstraction.

### Suggested file targets

* `fractal-core/src/v2/mod.rs`
* `fractal-core/src/v2/model.rs`
* `fractal-core/src/v2/local_trunk.rs`
* `fractal-core/src/v2/leaf.rs`
* `fractal-core/src/v2/tree.rs`
* `fractal-core/src/v2/router.rs`
* `fractal-core/src/v2/read_fusion.rs`

### Exit criteria

* workspace compiles
* public type boundaries are explicit
* `recursive-kernel-v1` path still runs unchanged

---

## Phase 2 — Add typed v2 runtime state

### Goal

Define explicit state surfaces for the new architecture.

### Required state families

* multi-root recurrent state
* live leaf state
* sealed leaf summary store
* tree level summary stores
* exact leaf token cache
* retrieval policy

### Suggested ownership

`fractal-core/src/v2/state.rs`

### Design guidance

Architectural ownership is not the same thing as hot-path layout.

Conceptually, you need:

* roots
* leaves
* tree levels
* routing policy

Operationally, prefer:

* batched tensors
* flat buffers
* explicit spans and indices

### Exit criteria

* state transitions are testable in isolation
* shapes are explicit
* no hidden global state
* checkpoint surface is at least sketched

---

## Phase 3 — Multi-root local trunk baseline

### Goal

Prove that multi-root local processing works before adding global memory.

### Required behavior

* 2 to 4 roots
* leaf size 16
* simple local recurrent/selective trunk
* same token stream to all roots
* no tree retrieval yet
* no exact leaf read yet

### Suggested files

* `fractal-core/src/v2/local_trunk.rs`
* `fractal-core/src/v2/model.rs`
* `fractal-core/src/v2/state.rs`

### Diagnostics required now

* root similarity / collapse metric
* per-root norm tracking
* per-root activation statistics

### Exit criteria

* forward pass works
* training step works
* single-root vs multi-root baseline is runnable
* roots do not instantly collapse in smoke tests

---

## Phase 4 — Live leaf and sealed leaf mechanics

### Goal

Introduce the local block structure and sealing boundary.

### Required behavior

* append token-local state into a live leaf
* seal leaf at fixed size 16
* create sealed leaf summaries
* populate typed token-level cache for sealed leaves

### Rules

* only sealed leaves enter global memory
* live leaf remains local-only
* future-token leakage must be impossible

### Suggested files

* `fractal-core/src/v2/leaf.rs`
* `fractal-core/src/v2/state.rs`

### Tests to add

* incremental append correctness
* sealing boundary correctness
* span correctness
* no future-token leakage

### Exit criteria

* live/sealed split is stable
* sealed leaf summaries are deterministic
* leaf token cache is populated correctly

---

## Phase 5 — Regular dyadic summary tree

### Goal

Build the first global memory substrate.

### Required behavior

* insert sealed leaves into level 0
* deterministically merge parents
* store summaries level-major
* track span metadata at every level

### Rules

* regular dyadic tree only
* every sealed leaf participates
* no learned merge gating
* only prior sealed leaves are globally visible

### Suggested files

* `fractal-core/src/v2/tree.rs`
* `fractal-core/src/v2/state.rs`

### Tests to add

* incremental update equals reference recompute
* span metadata correctness at all levels
* parent and level counts are correct
* causal visibility holds

### Diagnostics

* nodes per level
* tree depth reached
* dead or unused node detection

### Exit criteria

* tree construction is correct
* incremental tree update path works
* no causal violations are detected

---

## Phase 6 — Sparse routing over the sealed tree

### Goal

Replace dense global reading with coarse-to-fine sparse retrieval.

### Required behavior

* 4 routing heads
* beam width 2
* top-down candidate scoring
* descent to selected sealed leaves
* normalization only over surviving candidates
* no early-stop in the first runnable version

### Suggested files

* `fractal-core/src/v2/router.rs`
* `fractal-core/src/v2/model.rs`

### Diagnostics

* routing depth histogram
* candidate entropy per head
* selected span distance histogram
* head agreement / disagreement rate

### Tests to add

* routing touches sealed nodes only
* beam width enforcement
* deterministic behavior under fixed seed

### Exit criteria

* routing is sparse
* routing is query-dependent
* heads do not all choose the same path by default

---

## Phase 7 — Exact leaf read

### Goal

Preserve copy and token-precise retrieval behavior.

### Required behavior

Choose one exact-read mechanism first:

* local attention over cached token-level K/V in a selected sealed leaf
* pointer-style read over cached token states in a selected sealed leaf
* copy-distribution read over token positions in a selected sealed leaf

Do not add multiple exact-read mechanisms in the first version.

### Rules

* exact read must target routed sealed leaves only
* summary-only approximation is not enough
* no dense global token cache fallback

### Suggested files

* `fractal-core/src/v2/leaf.rs`
* `fractal-core/src/v2/model.rs`
* optionally `fractal-core/src/v2/read_fusion.rs`

### Diagnostics

* fraction of steps using exact leaf read
* selected token-position distribution
* read concentration / entropy

### Tests to add

* exact reads target sealed leaves only
* routed leaf span matches read span
* token indices inside the leaf are correct

### Exit criteria

* exact read is real, not approximate
* copy/retrieval behavior improves relative to no-exact-read ablation

---

## Phase 8 — Read fusion and LM head wiring

### Goal

Wire all useful sources into the output path without hiding attribution.

### Required fusion sources

* per-root recurrent outputs
* routed tree values
* exact leaf read values

### Rules

* fusion logic must be explicit
* routing usefulness must remain ablatable
* exact-read usefulness must remain ablatable
* do not bury everything inside one giant opaque mixer

### Suggested files

* `fractal-core/src/v2/read_fusion.rs`
* `fractal-core/src/v2/model.rs`
* existing LM head integration point

### Tests to add

* zero routed values and verify behavior changes
* zero exact-read values and verify behavior changes
* zero extra roots and verify behavior changes

### Exit criteria

* end-to-end forward path works
* each source path is individually removable
* logits are stable enough for smoke training

---

## Phase 9 — Synthetic task harness

### Goal

Evaluate architectural usefulness before large LM training.

### Required probes

* copy
* associative recall
* induction
* noisy retrieval
* far-token comparison

### Suggested locations

* `experiments/`
* `examples/`
* or a dedicated `fractal-eval-private` path if that is where probes belong

### Rules

* tasks must run quickly
* metrics must be comparable across ablations
* do not wait for large training runs to judge whether the architecture helps

### Exit criteria

* each probe has a stable baseline
* each probe can compare:

  * no memory
  * tree only
  * tree + exact read

---

## Phase 10 — Benchmark and observability pass

### Goal

Verify that the implementation behaves like the design rather than merely sounding like it.

### Required benchmarks

* token append
* leaf sealing
* tree update
* routing
* exact leaf read
* end-to-end forward pass

### Sequence lengths

* 256
* 512
* 1k
* 2k
* 4k
* 8k

### Required metrics

* tokens/sec
* wall-clock per forward
* peak memory
* routing sparsity
* root collapse metrics
* exact-read usage
* retrieval distance

### Exit criteria

* behavior trends toward intended scaling
* hot paths are clearly identifiable
* no accidental quadratic fallback is hiding in the implementation

---

## Required ablations

Before calling v1 real, run all of these at equal total state or parameter budget:

1. single-root, no memory
2. multi-root, no memory
3. single-root, summaries only
4. single-root, sparse retrieval
5. single-root, sparse retrieval plus exact leaf read
6. multi-root, summaries only
7. multi-root, sparse retrieval
8. multi-root, sparse retrieval without exact leaf read
9. multi-root, sparse retrieval plus exact leaf read

These ablations exist to separate:

* root multiplicity effects
* multiscale summary effects
* sparse retrieval effects
* exact-read effects

Do not skip them.

---

## Observability requirements

The v2 implementation is incomplete without diagnostics for:

* root collapse or similarity
* routing depth histogram
* candidate entropy per head
* selected-span distance histogram
* fraction of steps reaching exact leaf reads
* leaf usage distribution
* dead-tree or unused-node behavior

If later work adds a side-memory bank, then add:

* slot utilization
* eviction churn
* owner-root dominance

---

## Deferred features

These are explicitly deferred until after the tree-only design is proven:

* bounded side-memory bank
* learned eviction
* learned merge scheduling
* routing early stop
* giant-scale training
* dense fallback attention

Do not add these to rescue weak early results.

---

## Definition of done

The first serious v1 exists only when all of the following are true:

* multi-root local trunk works
* leaf sealing is causal and correct
* dyadic tree is stable and incremental
* routing is sparse and query-dependent
* exact leaf read is implemented and ablatable
* synthetic retrieval and copy tasks run
* scaling benchmarks exist
* diagnostics expose collapse and dead-weight behavior
* the architecture can be falsified cleanly

If these are not true yet, the work is still infrastructure, not a validated architecture.

---

## Suggested PR sequence

1. scaffold v2 module boundaries
2. add typed v2 runtime state
3. implement multi-root local trunk baseline
4. add live leaf append and sealed leaf summary path
5. implement regular causal dyadic summary tree
6. add sparse fractal router over sealed tree
7. implement exact local read for selected sealed leaves
8. wire read fusion into existing language model head
9. add synthetic retrieval and copy probes for v2
10. add scaling benchmarks and v2 observability suite

---

## Recommendation

Proceed with a tree-only v1 first.

Do not introduce side-memory bank complexity until the tree-only architecture is:

* implemented
* instrumented
* ablated
* behaviorally justified

That is the cleanest path to determining whether recursive multiscale memory can replace most of the role currently played by dense attention.
