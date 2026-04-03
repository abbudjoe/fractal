#  GPT 5.4 Pro Suggestions

Here’s a tightened rewrite plan for the spec, keeping your thesis but making v1 easier to implement and falsify.

## Main changes I recommend

The spec should narrow v1 to:

* local recurrent trunk
* sealed leaf blocks
* regular dyadic summary tree
* sparse tree routing
* exact leaf reads

And explicitly defer:

* separate bounded `MemoryBank`
* learned merge gating
* early-stop complexity in the first runnable version

That gives you a cleaner first proof target.

## What I would change section by section

### 1. Tighten the thesis

Your current thesis is good, but one line is still muddy: “explicit memory remains bounded” sits next to “O(n) persistent memory.”

I’d rewrite that part as:

```md
The new thesis:

- recursion remains the dominant local compute substrate
- explicit associative memory should not be collapsed into one recurrent state
- global memory should be organized causally and multiscale
- sparse query-specific retrieval should replace dense all-token attention where possible
- a persistent summary tree may grow with sequence length
- any additional associative side-memory should remain bounded and optional
```

That removes the contradiction.

---

### 2. Narrow the minimum v1 architecture

I would replace the current minimum executable section with this:

```md
## Minimum Executable v1

The first executable v2 should stay intentionally narrow.

Required components:

- 2 to 4 parallel roots
- leaf size 16
- local recurrent or selective trunk
- regular dyadic summary tree over sealed leaves
- 4 routing heads
- beam width 2
- top-2 selected leaf reads
- fused readout into the existing LM head

Deferred from v1:

- separate bounded MemoryBank
- learned merge gating
- adaptive early stop during routing
- learned eviction policy
- large-scale training ambitions

This version should optimize for:

- trainability
- observability
- ablatability
- causal correctness
```

That is a much better first implementation target.

---

### 3. Make causality fully explicit

Your causal block section is good, but I’d strengthen it.

Add this:

```md
### Causality contract

At step t:

- the live block may only contain tokens up to t
- unfinished future positions inside the live block are never visible
- the global summary tree contains only sealed prior leaf blocks
- sparse tree routing may only target sealed tree nodes
- exact leaf reads through the global memory path may only target sealed leaves
- short-range access inside the live block is handled only by the local causal trunk
```

That will save you a lot of implementation drift later.

---

### 4. Replace the vague exact leaf read contract

This is the biggest missing technical contract in the spec.

Right now `local_pointer_state` is too vague. I would rewrite that section to define exact leaf read more concretely:

```md
## Exact leaf read path

When a routing head selects a sealed leaf block, it may perform an exact read within that block.

v1 exact leaf read should mean one of the following, chosen explicitly in code:

- local attention over cached token-level keys/values within the selected leaf
- pointer-style selection over cached token-level hidden states within the selected leaf
- copy-distribution read over token positions within the selected leaf

This path exists to preserve:

- copy behavior
- associative retrieval
- token-precise recall

The implementation must define a typed leaf-token cache contract rather than a vague summary-only pointer state.
```

And the type sketch should become something more like:

```rust
pub struct LeafTokenCache<B: Backend> {
    pub keys: Tensor<B, 4>,   // [batch, leaves, tokens_per_leaf, d_k]
    pub values: Tensor<B, 4>, // [batch, leaves, tokens_per_leaf, d_v]
    pub mask: Tensor<B, 3>,   // [batch, leaves, tokens_per_leaf]
    pub span_start: Vec<usize>,
}
```

Even if you don’t use exactly this, the spec needs this level of clarity.

---

### 5. Remove `MemoryBank` from core v1

I would not delete it from the document entirely, but I would move it to a deferred section.

Right now it introduces too many memory systems at once:

* root recurrent state
* live leaf state
* summary tree
* side bank

That is too much for the first proving version.

I’d move it under something like:

```md
## Deferred v1.1 / v2 Features

The following are intentionally deferred until the tree-only architecture is stable:

- bounded associative MemoryBank
- owner-root slot policies
- learned eviction
- write-intent mediated bank insertion
```

That makes the proving version much cleaner.

---

### 6. Fix the typed runtime contracts

The current sketches mix conceptual ownership with runtime storage layout.

I’d change that section to explicitly separate the two.

Use wording like:

```md
These types describe architectural ownership, not necessarily final storage layout.

The runtime implementation should prefer batched tensor layouts over Vec-of-heap-objects in hot paths.
```

Then revise the examples to be more backend-friendly:

```rust
pub struct MultiRootState<B: Backend> {
    pub recurrent: Tensor<B, 3>,   // [batch, roots, d_state]
    pub read_intent: Tensor<B, 3>, // [batch, roots, d_q]
    pub write_intent: Tensor<B, 3>,
}

pub struct LeafSummaryStore<B: Backend> {
    pub keys: Tensor<B, 3>,   // [batch, leaves, d_k]
    pub values: Tensor<B, 3>, // [batch, leaves, d_v]
    pub spans: Vec<(usize, usize)>,
}

pub struct TreeLevelStore<B: Backend> {
    pub keys: Tensor<B, 3>,   // [batch, nodes_at_level, d_k]
    pub values: Tensor<B, 3>, // [batch, nodes_at_level, d_v]
    pub level: usize,
}
```

That is closer to how this should really run in Rust with Burn/Candle-like backends.

---

### 7. Make the tree regular in v1

I would change merge policy language so it does not imply learned irregular tree construction in the first version.

Replace:

```md
pub enum MergeCheckpointPolicy {
    EveryNTokens { n: usize },
    LearnedGate { threshold: f32 },
}
```

with:

```md
pub enum MergeCheckpointPolicy {
    FixedLeafSize { tokens_per_leaf: usize },
}
```

And add:

```md
v1 uses a regular dyadic tree over sealed leaves.
Learned merge scheduling is deferred.
```

This is important. A jagged learned hierarchy is much harder to reason about and benchmark.

---

### 8. Add a train-time routing semantics section

This is the biggest missing training detail.

I would add:

```md
## Train-time routing semantics

v1 routing should be sparse but trainable.

Default contract:

- candidate nodes are scored at each level
- top-k / beam pruning is applied per head
- normalization is performed only over the surviving candidate set
- early stop is disabled in the first runnable version
- gradients flow through scored surviving candidates
- pruned branches do not receive gradient
- optional teacher interval supervision may be added later

The first implementation should prefer simple, inspectable sparse routing over sophisticated policy learning.
```

That gives the implementer a clear default.

---

### 9. Strengthen the ablation ladder

Your ladder is already strong, but it needs one more control:
multi-root versus total-state-size.

I’d rewrite the ablations as:

```md
## Required Ablations

At equal total state / parameter budget, compare:

1. single-root, no memory
2. multi-root, no memory
3. single-root, sparse tree summaries, no retrieval
4. single-root, sparse tree retrieval
5. single-root, sparse tree retrieval, exact leaf read
6. multi-root, sparse tree retrieval
7. multi-root, sparse tree retrieval, no exact leaf read
8. multi-root, sparse tree retrieval, exact leaf read

Additional comparison:

- fixed merge cadence vs learned merge scheduling (deferred until after v1 stability)
```

That gives you much better attribution.

---

### 10. Add observability as a first-class section

You clearly care about diagnostics. Put that directly into the spec.

Add:

```md
## Observability requirements

The implementation should expose at least:

- root-state similarity / collapse metrics
- routing depth histogram per head
- candidate entropy per head
- selected-span distance histogram
- fraction of steps reaching exact leaf reads
- leaf usage distribution
- dead-tree / unused-node diagnostics
- if later enabled, memory-slot utilization and eviction churn

A v2 implementation without these diagnostics is incomplete.
```

This would make the spec much more implementation-ready.

---

### 11. Add a simple forward-pass pseudocode section

This would make the whole spec easier to build correctly.

Something like:

```md
## Forward pass sketch

For each new token:

1. embed token
2. update each root's local recurrent state
3. append token-local state into the live leaf cache
4. if the leaf seals:
   - summarize the leaf
   - insert leaf summary into level 0
   - recursively update dyadic parent summaries
5. form routing queries from current root states
6. route over sealed tree nodes with sparse beam search
7. descend to selected sealed leaves
8. perform exact local reads within selected leaves
9. fuse root state and retrieved values
10. produce LM logits through the existing head
```

That would remove a lot of ambiguity.

## Small doc fixes

Two smaller edits are worth making right away.

First, the links are malformed. They currently point to paths like:

* `/Users/joseph/fractal/fractal-core/src/state.rs`

Those should be repo-relative GitHub paths.

Second, I would rename either `recursive-kernel-v1` or `recursive-memory-kernel-v1` so the two lines do not sound almost identical in manifests and discussion.

## My recommended revised structure

If I were tightening the doc, I’d organize it like this:

1. Why v2 exists
2. New thesis
3. Non-goals for v1
4. Minimum executable v1
5. Core architecture

   * local trunk
   * sealed leaves
   * dyadic tree
   * sparse routing
   * exact leaf reads
6. Causality contract
7. Typed runtime/storage contracts
8. Train-time routing semantics
9. Observability requirements
10. Required ablations
11. Deferred features
12. Falsifiability
13. Implementation sequence

That would make it feel much sharper and more executable.

## Bottom line

I would **approve the direction**, but I would **tighten the spec before coding heavily against it**.

The most important edits are:

* remove separate `MemoryBank` from core v1
* make exact leaf read concrete
* make the tree regular in v1
* define train-time sparse routing behavior
* add observability and ablation controls
* fix the bounded-vs-persistent memory wording

If you want, I can now rewrite the spec into a cleaner v1 draft in one pass, with the sections and wording already tightened.
