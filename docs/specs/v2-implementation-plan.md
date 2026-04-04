# Implementation Plan — Recursive Memory Kernel v1

## Status
Draft

## Purpose
This document turns the `recursive-memory-kernel-v1` RFC into an execution plan for the first implementation series.

It is intentionally narrower than the full architecture spec.

Its purpose is to guide the first serious `fractal-v2` build toward something that is:

- causally correct
- observable
- ablatable
- narrow enough to debug
- hard to “decorate into success” without real evidence

## Related docs

- [Recursive Memory Kernel v1](./recursive-memory-kernel-v1.md)
- [Recursive Memory Kernel v1 RFC](./recursive-memory-kernel-v1-rfc.md)
- [Recursive Memory Kernel v1 Checklist](./recursive-memory-kernel-v1_checklist.md)
- [ENGINEERING](../../ENGINEERING.md)

Where this plan conflicts with broad ambition, prefer the narrower path.

---

## Implementation principles

### 1. Prove the tree-only design first
The first serious v1 should include:

- multi-root local processing
- sealed leaf blocks
- regular dyadic summaries
- sparse tree retrieval
- exact local reads within selected leaves

It should **not** initially include:

- side-memory bank
- learned merge scheduling
- routing early stop
- learned eviction
- dense fallback attention

### 2. Preserve falsifiability
Every major path must be individually removable or zeroable:

- extra roots
- tree summaries
- sparse routing
- exact leaf read

If a component cannot be ablated cleanly, it is not ready.

### 3. Prefer regular structure over learned policy
For the first runnable version:

- fixed leaf size
- regular dyadic tree
- fixed beam width
- no learned merge cadence
- no learned early stop

Learned policies may come later, but not before the base architecture is stable.

### 4. Favor backend-friendly storage
Hot paths should prefer:

- batched tensors
- flat stores
- level-major tree layouts
- explicit span metadata

Avoid:

- pointer-heavy recursive structures
- heap-heavy per-node ownership in hot loops
- hidden global mutable state

### 5. Measure causal usefulness, not just activation
This architecture must not only log what it routed or read.

It must also measure what actually changed the prediction.

That means the implementation must eventually support sampled counterfactual interventions over:

- tree summary retrieval
- exact leaf read
- selected spans
- root contributions

If a memory path looks structured but does not measurably affect outputs, it is probably decorative complexity.

---

## Expected module map

The exact filenames may shift, but the ownership boundaries should land close to this:

- `fractal-core/src/v2/model.rs`
- `fractal-core/src/v2/state.rs`
- `fractal-core/src/v2/local_trunk.rs`
- `fractal-core/src/v2/leaf.rs`
- `fractal-core/src/v2/tree.rs`
- `fractal-core/src/v2/router.rs`
- `fractal-core/src/v2/read_fusion.rs`
- `fractal-core/src/v2/auditor.rs`

Suggested exports:

- `FractalV2Model`
- `FractalV2State`
- `LocalTrunk`
- `LeafSummarizer`
- `TreeMergeCell`
- `FractalRouterHead`
- `ReadFusion`
- `CausalMemoryAuditor`

If you keep different filenames, preserve the ownership boundaries.

---

## Execution phases

## Phase 1 — Scaffold v2 module boundaries

### Goal
Create explicit v2 architecture surfaces without disturbing `recursive-kernel-v1`.

### Deliverables
- `LocalTrunk`
- `LeafSummarizer`
- `TreeMergeCell`
- `FractalRouterHead`
- `ReadFusion`
- `FractalV2Model`

### Notes
- Keep this phase mostly structural.
- Avoid sneaking behavior policy into generic traits too early.
- Do not try to force everything through the current `FractalRule` abstraction.

### Suggested file targets
- `fractal-core/src/v2/mod.rs`
- `fractal-core/src/v2/model.rs`
- `fractal-core/src/v2/local_trunk.rs`
- `fractal-core/src/v2/leaf.rs`
- `fractal-core/src/v2/tree.rs`
- `fractal-core/src/v2/router.rs`
- `fractal-core/src/v2/read_fusion.rs`

### Exit criteria
- workspace compiles
- public type boundaries are explicit
- `recursive-kernel-v1` path still runs unchanged

---

## Phase 2 — Add typed v2 runtime state

### Goal
Define explicit state surfaces for the new architecture.

### Required state families
- multi-root recurrent state
- live leaf state
- sealed leaf summary store
- tree level summary stores
- exact leaf token cache
- retrieval policy

### Suggested ownership
- `fractal-core/src/v2/state.rs`

### Design guidance
Architectural ownership is not the same thing as hot-path layout.

Conceptually, you need:
- roots
- leaves
- tree levels
- routing policy

Operationally, prefer:
- batched tensors
- flat buffers
- explicit spans and indices

### Exit criteria
- state transitions are testable in isolation
- shapes are explicit
- no hidden global state
- checkpoint surface is at least sketched

---

## Phase 3 — Multi-root local trunk baseline

### Goal
Prove that multi-root local processing works before adding global memory.

### Required behavior
- 2 to 4 roots
- leaf size 16
- simple local recurrent/selective trunk
- same token stream to all roots
- no tree retrieval yet
- no exact leaf read yet

### Suggested files
- `fractal-core/src/v2/local_trunk.rs`
- `fractal-core/src/v2/model.rs`
- `fractal-core/src/v2/state.rs`

### Diagnostics required now
- root similarity / collapse metric
- per-root norm tracking
- per-root activation statistics

### Exit criteria
- forward pass works
- training step works
- single-root vs multi-root baseline is runnable
- roots do not instantly collapse in smoke tests

---

## Phase 4 — Live leaf and sealed leaf mechanics

### Goal
Introduce the local block structure and sealing boundary.

### Required behavior
- append token-local state into a live leaf
- seal leaf at fixed size 16
- create sealed leaf summaries
- populate typed token-level cache for sealed leaves

### Rules
- only sealed leaves enter global memory
- live leaf remains local-only
- future-token leakage must be impossible

### Suggested files
- `fractal-core/src/v2/leaf.rs`
- `fractal-core/src/v2/state.rs`

### Tests to add
- incremental append correctness
- sealing boundary correctness
- span correctness
- no future-token leakage

### Exit criteria
- live/sealed split is stable
- sealed leaf summaries are deterministic
- leaf token cache is populated correctly

---

## Phase 5 — Regular dyadic summary tree

### Goal
Build the first global memory substrate.

### Required behavior
- insert sealed leaves into level 0
- deterministically merge parents
- store summaries level-major
- track span metadata at every level

### Rules
- regular dyadic tree only
- every sealed leaf participates
- no learned merge gating
- only prior sealed leaves are globally visible

### Suggested files
- `fractal-core/src/v2/tree.rs`
- `fractal-core/src/v2/state.rs`

### Tests to add
- incremental update equals reference recompute
- span metadata correctness at all levels
- parent and level counts are correct
- causal visibility holds

### Diagnostics
- nodes per level
- tree depth reached
- dead or unused node detection

### Exit criteria
- tree construction is correct
- incremental tree update path works
- no causal violations are detected

---

## Phase 6 — Sparse routing over the sealed tree

### Goal
Replace dense global reading with coarse-to-fine sparse retrieval.

### Required behavior
- 4 routing heads
- beam width 2
- top-down candidate scoring
- descent to selected sealed leaves
- normalization only over surviving candidates
- no early-stop in the first runnable version

### Suggested files
- `fractal-core/src/v2/router.rs`
- `fractal-core/src/v2/model.rs`

### Diagnostics
- routing depth histogram
- candidate entropy per head
- selected span distance histogram
- head agreement / disagreement rate

### Tests to add
- routing touches sealed nodes only
- beam width enforcement
- deterministic behavior under fixed seed

### Exit criteria
- routing is sparse
- routing is query-dependent
- heads do not all choose the same path by default

---

## Phase 7 — Exact leaf read

### Goal
Preserve copy and token-precise retrieval behavior.

### Required behavior
Choose one exact-read mechanism first:

- local attention over cached token-level K/V in a selected sealed leaf
- pointer-style read over cached token states in a selected sealed leaf
- copy-distribution read over token positions in a selected sealed leaf

Do not add multiple exact-read mechanisms in the first version.

### Rules
- exact read must target routed sealed leaves only
- summary-only approximation is not enough
- no dense global token cache fallback

### Suggested files
- `fractal-core/src/v2/leaf.rs`
- `fractal-core/src/v2/model.rs`
- optionally `fractal-core/src/v2/read_fusion.rs`

### Diagnostics
- fraction of steps using exact leaf read
- selected token-position distribution
- read concentration / entropy

### Tests to add
- exact reads target sealed leaves only
- routed leaf span matches read span
- token indices inside the leaf are correct

### Exit criteria
- exact read is real, not approximate
- copy/retrieval behavior improves relative to no-exact-read ablation

---

## Phase 8 — Read fusion and LM head wiring

### Goal
Wire all useful sources into the output path without hiding attribution.

### Required fusion sources
- per-root recurrent outputs
- routed tree values
- exact leaf read values

### Rules
- fusion logic must be explicit
- routing usefulness must remain ablatable
- exact-read usefulness must remain ablatable
- do not bury everything inside one giant opaque mixer

### Suggested files
- `fractal-core/src/v2/read_fusion.rs`
- `fractal-core/src/v2/model.rs`
- existing LM head integration point

### Tests to add
- zero routed values and verify behavior changes
- zero exact-read values and verify behavior changes
- zero extra roots and verify behavior changes

### Exit criteria
- end-to-end forward path works
- each source path is individually removable
- logits are stable enough for smoke training

---

## Phase 8.5 — Causal Memory Auditor

### Goal
Measure whether the memory architecture is doing useful causal work, rather than only producing interesting routing patterns.

This phase exists to answer questions like:

- does tree retrieval materially change predictions?
- does exact leaf read matter beyond diagnostics?
- do different roots contribute distinct utility?
- is deeper routing actually useful?
- is the tree helping, or is the local trunk carrying everything?

This phase should begin as soon as sparse retrieval and exact leaf read are both stable enough to run.

### Required capabilities
Implement a sampled counterfactual audit path that can run on selected batches, positions, or evaluation examples.

The auditor should support:

- full forward pass reference
- no-tree-read intervention
- no-exact-leaf-read intervention
- next-best-span substitution
- root-drop intervention

These interventions should be small, explicit, and easy to interpret.

Do not start with complicated attribution methods.

### Counterfactual interventions

#### 1. No tree read
Remove or zero retrieved tree-summary values while keeping the rest of the forward pass unchanged.

Purpose:
- test whether sparse tree retrieval is causally useful

#### 2. No exact leaf read
Remove the exact local read path while preserving routed coarse summaries.

Purpose:
- test whether exact leaf read is necessary
- detect whether exact leaf read dominates all useful retrieval

#### 3. Next-best span substitution
Replace the selected routed span with the next-best candidate at the same routing depth.

Purpose:
- test whether the chosen span was meaningfully better than nearby alternatives
- distinguish real routing quality from arbitrary sparse selection

#### 4. Root-drop intervention
Remove one root contribution at a time from the fused readout.

Purpose:
- measure differentiated root utility
- detect redundant roots

### Suggested file targets
- `fractal-core/src/v2/auditor.rs`
- `fractal-core/src/v2/model.rs`
- `fractal-core/src/v2/router.rs`
- `fractal-core/src/v2/read_fusion.rs`

If you prefer a different filename, preserve the ownership boundary:
- auditing is its own surface
- interventions are explicit
- reporting is structured

### Required outputs
For each sampled intervention, record deltas such as:

- loss difference
- target-token logit difference
- KL divergence from full forward
- retrieval-task accuracy delta
- perplexity delta on sampled LM spans

Aggregate results by:

- routing head
- root
- tree depth
- span distance
- selected leaf
- component family:
  - local trunk
  - tree summary retrieval
  - exact leaf read

### Minimum diagnostics
The auditor should emit at least:

- average utility of tree retrieval
- average utility of exact leaf read
- utility by root
- utility by routing depth
- utility by task family:
  - copy
  - associative recall
  - induction
  - noisy retrieval
  - ordinary LM evaluation

### Required tests
- intervention path preserves tensor shapes
- no-tree-read actually removes only tree-summary contributions
- no-exact-read actually removes only exact leaf-read contributions
- next-best substitution respects routing depth constraints
- root-drop removes only the selected root contribution
- full forward and counterfactual forward remain comparable on the same sampled position

### Engineering rules
- sampled, not always-on
- cheap enough to run during evaluation
- optionally enabled during training
- mandatory for architecture validation runs
- never silently mutate the reference forward path
- keep intervention definitions explicit and versioned

### Exit criteria
This phase is complete when the project can answer, with actual measurements:

- whether tree retrieval changes predictions
- whether exact leaf read is carrying most or only some of the retrieval benefit
- whether roots have differentiated utility
- whether deeper routing is useful on tasks that require it
- whether the architecture’s extra memory complexity is paying for itself

### Failure signals to watch for
Treat these as serious warnings:

- removing tree retrieval barely changes outputs
- exact leaf read is the only component with meaningful utility
- roots are nearly interchangeable
- deeper routing adds little or no causal value
- sparse retrieval looks structured but has negligible effect

These are not just diagnostics.  
They are evidence that the current architecture may need revision.

---

## Phase 9 — Synthetic task harness

### Goal
Evaluate architectural usefulness before large LM training.

### Required probes
- copy
- associative recall
- induction
- noisy retrieval
- far-token comparison

### Suggested locations
- `experiments/`
- `examples/`
- or a dedicated `fractal-eval-private` path if that is where probes belong

### Rules
- tasks must run quickly
- metrics must be comparable across ablations
- do not wait for large training runs to judge whether the architecture helps

### Exit criteria
- each probe has a stable baseline
- each probe can compare:
  - no memory
  - tree only
  - tree + exact read

---

## Phase 10 — Benchmark and observability pass

### Goal
Verify that the implementation behaves like the design rather than merely sounding like it.

### Required benchmarks
- token append
- leaf sealing
- tree update
- routing
- exact leaf read
- end-to-end forward pass

### Sequence lengths
- 256
- 512
- 1k
- 2k
- 4k
- 8k

### Required metrics
- tokens/sec
- wall-clock per forward
- peak memory
- routing sparsity
- root collapse metrics
- exact-read usage
- retrieval distance

### Exit criteria
- behavior trends toward intended scaling
- hot paths are clearly identifiable
- no accidental quadratic fallback is hiding in the implementation

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

- root multiplicity effects
- multiscale summary effects
- sparse retrieval effects
- exact-read effects

Do not skip them.

---

## Observability requirements

The v2 implementation is incomplete without diagnostics for:

- root collapse or similarity
- routing depth histogram
- candidate entropy per head
- selected-span distance histogram
- fraction of steps reaching exact leaf reads
- leaf usage distribution
- dead-tree or unused-node behavior

If later work adds a side-memory bank, then add:

- slot utilization
- eviction churn
- owner-root dominance

The Causal Memory Auditor extends these diagnostics by estimating which memory paths are causally useful, not merely active.

---

## Deferred features

These are explicitly deferred until after the tree-only design is proven:

- bounded side-memory bank
- learned eviction
- learned merge scheduling
- routing early stop
- giant-scale training
- dense fallback attention

Do not add these to rescue weak early results.

---

## Definition of done

The first serious v1 exists only when all of the following are true:

- multi-root local trunk works
- leaf sealing is causal and correct
- dyadic tree is stable and incremental
- routing is sparse and query-dependent
- exact leaf read is implemented and ablatable
- synthetic retrieval and copy tasks run
- scaling benchmarks exist
- diagnostics expose collapse and dead-weight behavior
- causal auditing shows whether memory paths materially affect predictions
- the architecture can be falsified cleanly

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
9. add Causal Memory Auditor  
10. add synthetic retrieval and copy probes for v2  
11. add scaling benchmarks and v2 observability suite

---

## Recommendation

Proceed with a tree-only v1 first.

Do not introduce side-memory bank complexity until the tree-only architecture is:

- implemented
- instrumented
- ablated
- causally audited
- behaviorally justified

That is the cleanest path to determining whether recursive multiscale memory can replace most of the role currently played by dense attention.
