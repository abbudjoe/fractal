# Recursive Memory Kernel v1 — First PR Series Checklist

## Goal

Land the smallest tree-only v1 that is:

* causally correct
* easy to instrument
* easy to ablate
* narrow enough to debug

This checklist is ordered.
Do not skip ahead unless the earlier layer is stable.

---

## PR 1 — Carve out v2 module boundaries

### Deliverables

Create explicit module boundaries without changing the current v1 path:

* `LocalTrunk`
* `LeafSummarizer`
* `TreeMergeCell`
* `FractalRouterHead`
* `ReadFusion`
* `FractalV2Model`

### Requirements

* no behavior claims yet
* no side-memory bank
* no learned merge scheduling
* no routing early-stop
* no integration with existing tournament presets beyond stub wiring

### Exit criteria

* code compiles
* types are explicit
* module ownership is clear
* `recursive-kernel-v1` remains undisturbed

---

## PR 2 — Add typed v2 runtime state

### Deliverables

Introduce typed state/storage contracts for:

* multi-root recurrent state
* live leaf state
* sealed leaf summaries
* tree level summaries
* leaf token cache
* retrieval policy

### Requirements

* prefer backend-friendly tensor layouts in hot paths
* avoid pointer-heavy recursive state structures
* separate conceptual ownership from runtime storage layout

### Exit criteria

* state updates are testable in isolation
* shapes are explicit
* no hidden global state
* serialization/checkpoint surface is at least sketched

---

## PR 3 — Multi-root local trunk baseline

### Deliverables

Implement local token processing with:

* 2 to 4 roots
* leaf size 16
* simple local recurrent/selective trunk
* no tree retrieval yet

### Requirements

* strict autoregressive masking/causality
* same token stream goes to all roots
* root outputs remain separately inspectable

### Diagnostics to add now

* root similarity / collapse metric
* per-root norm tracking
* per-root activation statistics

### Exit criteria

* forward pass works
* training step works
* roots do not trivially collapse on first smoke tests
* single-root vs multi-root baseline is runnable

---

## PR 4 — Live leaf + sealed leaf mechanics

### Deliverables

Implement:

* live leaf append path
* leaf sealing at fixed size 16
* creation of sealed leaf summaries
* typed token-level cache for sealed leaves

### Requirements

* only sealed leaves enter global memory
* live leaf remains local-only
* exact leaf read target must be meaningful later

### Tests

* incremental append correctness
* leaf seal boundaries
* sealed leaf spans are correct
* no future-token leakage

### Exit criteria

* live/sealed split is stable
* leaf summaries are deterministic
* leaf token cache is populated correctly

---

## PR 5 — Regular dyadic summary tree

### Deliverables

Implement:

* level-0 insertion from sealed leaves
* deterministic parent merge
* level-major summary storage
* span metadata for each node

### Requirements

* regular dyadic tree only
* no learned merge gating
* every sealed leaf participates
* only prior sealed leaves are visible globally

### Tests

* tree build equals reference recompute
* spans are correct at all levels
* causal visibility holds
* parent count and level count are correct

### Diagnostics

* nodes per level
* tree depth reached
* dead/unused node detection

### Exit criteria

* tree construction is correct
* tree updates are incremental
* no causal violations

---

## PR 6 — Sparse routing over the sealed tree

### Deliverables

Implement routing heads with:

* 4 routing heads
* beam width 2
* top-down candidate scoring
* descent to selected sealed leaves

### Requirements

* candidate scoring is explicit
* normalization only over surviving candidates
* no early stop in first runnable version
* routing is inspectable per head

### Diagnostics

* routing depth histogram
* candidate entropy per head
* selected span distance histogram
* head agreement / disagreement rate

### Tests

* routing only touches sealed tree nodes
* beam width is enforced
* deterministic behavior under fixed seed

### Exit criteria

* routing is sparse
* routing is query-dependent
* heads do not all pick the same path by default

---

## PR 7 — Exact leaf read

### Deliverables

Implement one explicit exact-read mechanism:

* local attention over cached token-level K/V in selected leaf
  **or**
* pointer-style read over cached token states
  **or**
* copy-distribution read over leaf token positions

### Requirements

* choose one mechanism first
* keep it simple
* no dense global token cache fallback

### Diagnostics

* fraction of steps using exact leaf read
* selected token-position distribution
* read concentration / entropy

### Tests

* exact reads only target sealed leaves
* exact read span matches routed leaf
* local token indices are correct

### Exit criteria

* exact local read is real, not approximate
* retrieval behavior changes when this path is enabled
* copy/retrieval probes improve relative to no-exact-read

---

## PR 8 — Read fusion + LM head wiring

### Deliverables

Fuse:

* per-root recurrent outputs
* routed tree values
* exact leaf read values

Then connect to existing LM head.

### Requirements

* fusion logic is explicit and typed
* avoid burying routing usefulness inside a giant opaque mixer
* keep ablations easy

### Tests

* zeroing routed values changes behavior predictably
* zeroing exact leaf read changes behavior predictably
* zeroing extra roots changes behavior predictably

### Exit criteria

* full forward path works end-to-end
* each source path is individually ablatable
* logits are stable enough for smoke training

---

## PR 9 — Synthetic task harness

### Deliverables

Add first-class probes for:

* copy
* associative recall
* induction
* noisy retrieval
* far-token comparison

### Requirements

* tasks must run quickly
* tasks must produce comparable metrics across ablations
* do not wait for large LM training to evaluate architecture value

### Exit criteria

* each probe has a stable baseline
* each probe can compare:

  * no memory
  * tree only
  * tree + exact read

---

## PR 10 — Benchmark and observability pass

### Deliverables

Add benchmarks for:

* token append
* leaf sealing
* tree update
* routing
* exact leaf read
* full forward pass

At sequence lengths like:

* 256
* 512
* 1k
* 2k
* 4k
* 8k

### Metrics to log

* tokens/sec
* wall-clock per forward
* peak memory
* routing sparsity
* root collapse
* exact-read usage
* retrieval distance

### Exit criteria

* measured behavior trends toward intended scaling
* hot paths are identifiable
* no accidental quadratic fallback in implementation

---

## Required ablations before calling v1 “real”

Run at equal total state / parameter budget:

1. single-root, no memory
2. multi-root, no memory
3. single-root, summaries only
4. single-root, sparse retrieval
5. single-root, sparse retrieval + exact leaf read
6. multi-root, summaries only
7. multi-root, sparse retrieval
8. multi-root, sparse retrieval without exact leaf read
9. multi-root, sparse retrieval + exact leaf read

Do not skip these.

---

## Deferred until after this checklist

Do **not** add yet:

* side-memory bank
* learned eviction
* learned merge scheduling
* routing early-stop
* giant-scale training
* dense fallback attention
* extra complexity to “help” a weak first result

---

## Definition of done for the first serious v1

The first serious v1 exists only when all of these are true:

* multi-root local trunk works
* leaf sealing is causal and correct
* dyadic tree is stable and incremental
* routing is sparse and query-dependent
* exact leaf read is implemented and ablatable
* synthetic retrieval/copy tasks run
* scaling benchmarks exist
* diagnostics expose collapse and dead-weight behavior
* the architecture can be falsified cleanly

If those are not true yet, the work is still infrastructure, not a validated architecture.

