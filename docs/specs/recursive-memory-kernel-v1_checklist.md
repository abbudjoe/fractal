# Recursive Memory Kernel v1 — Implementation Checklist

> Purpose: land the smallest tree-only v1 that is causally correct, observable, ablatable, and narrow enough to debug.
>
> Rule of thumb: do **not** add side-memory bank, learned merge scheduling, routing early-stop, or dense fallback attention until this checklist is complete.

---

## PR 1 — Scaffold v2 module boundaries

**Title:** `scaffold fractal-v2 module boundaries`

### Deliverables
- [ ] Add `LocalTrunk`
- [ ] Add `LeafSummarizer`
- [ ] Add `TreeMergeCell`
- [ ] Add `FractalRouterHead`
- [ ] Add `ReadFusion`
- [ ] Add `FractalV2Model`

### Requirements
- [ ] Do not disturb `recursive-kernel-v1`
- [ ] No side-memory bank
- [ ] No learned merge scheduling
- [ ] No routing early-stop
- [ ] No broad tournament integration beyond stub wiring

### Merge criteria
- [ ] Workspace compiles
- [ ] Module ownership is explicit
- [ ] Public type boundaries are clear
- [ ] Existing v1 path still runs unchanged

---

## PR 2 — Add typed v2 runtime state

**Title:** `add typed recursive-memory-kernel v2 state surfaces`

### Deliverables
- [ ] Add multi-root recurrent state
- [ ] Add live leaf state
- [ ] Add sealed leaf summary store
- [ ] Add tree level summary stores
- [ ] Add exact leaf token cache
- [ ] Add retrieval policy surface

### Requirements
- [ ] Prefer backend-friendly tensor layouts in hot paths
- [ ] Avoid pointer-heavy recursive state structures
- [ ] Separate conceptual ownership from runtime storage
- [ ] Sketch checkpoint/serialization boundary

### Merge criteria
- [ ] State transitions are testable in isolation
- [ ] Shapes are explicit
- [ ] No hidden global state
- [ ] Types are stable enough for later model wiring

---

## PR 3 — Multi-root local trunk baseline

**Title:** `implement multi-root local trunk baseline`

### Deliverables
- [ ] Implement 2 to 4 root local processing
- [ ] Support leaf size 16
- [ ] Use a simple local recurrent/selective trunk
- [ ] Keep root outputs independently inspectable

### Requirements
- [ ] Strict autoregressive behavior
- [ ] Same token stream reaches all roots
- [ ] No tree retrieval yet
- [ ] No exact leaf read yet

### Diagnostics
- [ ] Root similarity / collapse metric
- [ ] Per-root norm tracking
- [ ] Per-root activation statistics

### Merge criteria
- [ ] Forward pass works
- [ ] Training step works
- [ ] Single-root vs multi-root baseline is runnable
- [ ] Roots do not immediately collapse in smoke tests

---

## PR 4 — Live leaf and sealed leaf mechanics

**Title:** `add live leaf append and sealed leaf summary path`

### Deliverables
- [ ] Implement live leaf append path
- [ ] Seal leaves at fixed size 16
- [ ] Create sealed leaf summaries
- [ ] Populate token-level cache for sealed leaves

### Requirements
- [ ] Only sealed leaves enter global memory
- [ ] Live leaf remains local-only
- [ ] Exact leaf read target is meaningful for later use

### Tests
- [ ] Incremental append correctness
- [ ] Leaf sealing boundaries
- [ ] Correct sealed leaf spans
- [ ] No future-token leakage

### Merge criteria
- [ ] Live/sealed split is stable
- [ ] Leaf summaries are deterministic
- [ ] Token cache contents are correct
- [ ] Causality is preserved

---

## PR 5 — Regular dyadic summary tree

**Title:** `implement regular causal dyadic summary tree`

### Deliverables
- [ ] Insert sealed leaves into level 0
- [ ] Add deterministic parent merge
- [ ] Store summaries level-major
- [ ] Track span metadata at each level

### Requirements
- [ ] Regular dyadic tree only
- [ ] No learned merge gating
- [ ] Every sealed leaf participates
- [ ] Only prior sealed leaves are globally visible

### Tests
- [ ] Incremental tree build matches reference recompute
- [ ] Span metadata is correct at all levels
- [ ] Causal visibility holds
- [ ] Parent and level counts are correct

### Diagnostics
- [ ] Nodes per level
- [ ] Tree depth reached
- [ ] Dead / unused node detection

### Merge criteria
- [ ] Tree construction is correct
- [ ] Incremental update path works
- [ ] No causal violations
- [ ] Summary tree is observable

---

## PR 6 — Sparse routing over the sealed tree

**Title:** `add sparse fractal router over sealed tree`

### Deliverables
- [ ] Add 4 routing heads
- [ ] Add beam width 2 routing
- [ ] Score candidates top-down
- [ ] Descend to selected sealed leaves

### Requirements
- [ ] Candidate scoring is explicit
- [ ] Normalize only over surviving candidates
- [ ] No early-stop in first runnable version
- [ ] Routing stays inspectable per head

### Diagnostics
- [ ] Routing depth histogram
- [ ] Candidate entropy per head
- [ ] Selected span distance histogram
- [ ] Head agreement / disagreement rate

### Tests
- [ ] Routing touches sealed nodes only
- [ ] Beam width is enforced
- [ ] Behavior is deterministic under fixed seed

### Merge criteria
- [ ] Routing is sparse
- [ ] Routing is query-dependent
- [ ] Heads do not all choose the same path by default
- [ ] Retrieval path is ablatable

---

## PR 7 — Exact leaf read

**Title:** `implement exact local read for selected sealed leaves`

### Deliverables
- [ ] Choose one exact-read mechanism:
  - [ ] local attention over token-level K/V inside the selected leaf
  - [ ] pointer-style read over cached token states
  - [ ] copy-distribution read over leaf token positions
- [ ] Wire exact read to selected routed leaves only

### Requirements
- [ ] Keep the first mechanism simple
- [ ] No dense global token cache fallback
- [ ] Exact means token-level local access, not summary-only approximation

### Diagnostics
- [ ] Fraction of steps using exact leaf read
- [ ] Selected token-position distribution
- [ ] Read concentration / entropy

### Tests
- [ ] Exact reads target sealed leaves only
- [ ] Exact read span matches routed leaf
- [ ] Local token indices are correct

### Merge criteria
- [ ] Exact read is real, not approximate
- [ ] Retrieval behavior changes when enabled
- [ ] Copy/retrieval probes improve relative to no-exact-read

---

## PR 8 — Read fusion and LM head wiring

**Title:** `wire read fusion into existing language model head`

### Deliverables
- [ ] Fuse per-root recurrent outputs
- [ ] Fuse routed tree values
- [ ] Fuse exact leaf read values
- [ ] Project fused output through LM head

### Requirements
- [ ] Fusion logic is explicit and typed
- [ ] Do not bury routing usefulness inside an opaque mixer
- [ ] Keep all major sources easy to zero out for ablations

### Tests
- [ ] Zeroing routed values changes behavior predictably
- [ ] Zeroing exact leaf read changes behavior predictably
- [ ] Zeroing extra roots changes behavior predictably

### Merge criteria
- [ ] Full forward path works end-to-end
- [ ] Each source path is individually ablatable
- [ ] Logits are stable enough for smoke training

---

## PR 9 — Causal Memory Auditor

**Title:** `add causal memory auditor for counterfactual memory credit`

### Deliverables
- [ ] Add full forward reference path
- [ ] Add no-tree-read intervention
- [ ] Add no-exact-leaf-read intervention
- [ ] Add next-best-span substitution
- [ ] Add root-drop intervention
- [ ] Add structured reporting of utility deltas

### Requirements
- [ ] Sampled, not always-on
- [ ] Cheap enough for evaluation runs
- [ ] Explicit intervention definitions
- [ ] No silent mutation of reference forward behavior

### Diagnostics
- [ ] Loss delta
- [ ] Target-logit delta
- [ ] KL divergence from full forward
- [ ] Utility by root
- [ ] Utility by routing depth
- [ ] Utility by task family

### Tests
- [ ] Each intervention preserves shape compatibility
- [ ] No-tree-read removes only tree-summary contributions
- [ ] No-exact-read removes only exact-read contributions
- [ ] Next-best substitution respects routing depth
- [ ] Root-drop removes only the selected root contribution

### Merge criteria
- [ ] Tree retrieval utility is measurable
- [ ] Exact leaf read utility is measurable
- [ ] Root utility is measurable
- [ ] Dead-weight tree behavior is detectable if present

---

## PR 10 — Synthetic task harness

**Title:** `add synthetic retrieval and copy probes for v2`

### Deliverables
- [ ] Add copy task
- [ ] Add associative recall task
- [ ] Add induction task
- [ ] Add noisy retrieval task
- [ ] Add far-token comparison task

### Requirements
- [ ] Tasks run quickly
- [ ] Metrics are comparable across ablations
- [ ] Do not wait for large LM training to evaluate architecture value

### Merge criteria
- [ ] Each probe has a stable baseline
- [ ] Probes can compare:
  - [ ] no memory
  - [ ] tree only
  - [ ] tree + exact read
- [ ] Results are logged in a repeatable format

---

## PR 11 — Benchmark and observability pass

**Title:** `add scaling benchmarks and v2 observability suite`

### Deliverables
- [ ] Benchmark token append
- [ ] Benchmark leaf sealing
- [ ] Benchmark tree update
- [ ] Benchmark routing
- [ ] Benchmark exact leaf read
- [ ] Benchmark end-to-end forward pass

### Sequence lengths
- [ ] 256
- [ ] 512
- [ ] 1k
- [ ] 2k
- [ ] 4k
- [ ] 8k

### Metrics
- [ ] Tokens/sec
- [ ] Wall-clock per forward
- [ ] Peak memory
- [ ] Routing sparsity
- [ ] Root collapse metrics
- [ ] Exact-read usage
- [ ] Retrieval distance

### Merge criteria
- [ ] Measured behavior trends toward intended scaling
- [ ] Hot paths are identifiable
- [ ] No accidental quadratic fallback is present

---

# Required ablations before calling v1 “real”

Run at equal total state / parameter budget:

- [ ] single-root, no memory
- [ ] multi-root, no memory
- [ ] single-root, summaries only
- [ ] single-root, sparse retrieval
- [ ] single-root, sparse retrieval + exact leaf read
- [ ] multi-root, summaries only
- [ ] multi-root, sparse retrieval
- [ ] multi-root, sparse retrieval without exact leaf read
- [ ] multi-root, sparse retrieval + exact leaf read

**Do not skip these.**

---

# Deferred until after this checklist

Do **not** add yet:

- [ ] side-memory bank
- [ ] learned eviction
- [ ] learned merge scheduling
- [ ] routing early-stop
- [ ] giant-scale training
- [ ] dense fallback attention
- [ ] complexity added only to rescue weak early results

---

# Definition of done for first serious v1

The first serious v1 exists only when all of these are true:

- [ ] multi-root local trunk works
- [ ] leaf sealing is causal and correct
- [ ] dyadic tree is stable and incremental
- [ ] routing is sparse and query-dependent
- [ ] exact leaf read is implemented and ablatable
- [ ] causal memory auditing is implemented and reports useful deltas
- [ ] synthetic retrieval/copy tasks run
- [ ] scaling benchmarks exist
- [ ] diagnostics expose collapse and dead-weight behavior
- [ ] the architecture can be falsified cleanly

If these are not true yet, the work is still infrastructure, not a validated architecture.
