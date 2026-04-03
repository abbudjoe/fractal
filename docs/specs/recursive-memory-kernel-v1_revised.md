# Fractal v2 Architecture

## Recursive Memory Kernel v1

This document defines the first serious successor architecture to the current single-root recursive language-model line.

It does **not** retroactively make `recursive-kernel-v1` successful.
It does **not** erase the current Stage 0 debugging history.

It states a narrower and more defensible thesis:

* keep recursion
* stop asking one recurrent state to replace all addressable memory
* move fractality from pointwise primitive math into **memory organization and routing geometry**

This architecture is intended for the `fractal-v2` branch family while reusing the strongest components of the current repository.

---

## Why v2 exists

The current v1 line made three large bets:

* one recurrent root can carry the whole sequence state
* inner recursive depth can stand in for explicit memory organization
* router-driven adaptive depth can supply most of the efficiency gains

Those bets produced useful work:

* strong diagnostics
* better projection ownership
* a cleaner LM head
* a reusable experiment control plane

They did **not** yet produce a convincing Stage 0 training path.

The strongest conceptual problem is now clearer:

* compressed recurrent state is likely useful as **primary memory**
* compressed recurrent state is unlikely to fully replace **content-addressable memory**

especially for:

* long-range retrieval
* selective copy
* distant comparison
* binding across far-apart regions

So v2 is **not**:

* abandoning recursion

v2 **is**:

* keeping recursion as the dominant local compute substrate
* adding explicit causal multiscale memory where the recurrent bottleneck is real
* making retrieval sparse, typed, observable, and falsifiable

---

## New thesis

`recursive-memory-kernel-v1` should be a language model where:

* local token processing is recurrent or selective-state-space-like
* tokens are grouped into fixed-size causal leaf blocks
* completed leaf blocks are recursively summarized into a regular dyadic tree
* multiple routing heads perform sparse coarse-to-fine retrieval over that tree
* selected leaves support exact local reads for copy and retrieval behavior
* recurrent state remains primary
* the persistent summary tree grows with sequence length
* any additional associative side-memory is **optional** and **deferred**
* dense all-token all-head attention is not the default substrate

In short:

**state first
memory second
attention only where it earns its cost**

---

## Non-goals for v1

v1 is **not** trying to optimize for:

* maximum novelty in primitive formulas
* exact imitation of transformer block structure
* a learned memory policy everywhere from day one
* giant-scale training
* hiding routing or memory behavior inside opaque heuristics
* proving that recurrence alone replaces all explicit memory

v1 is a proving architecture, not a final ideology.

---

## Minimum executable v1

The first runnable v2 should stay intentionally narrow.

Required components:

* **2 to 4 parallel roots**
* **leaf size 16**
* **local recurrent or selective trunk**
* **regular dyadic summary tree over sealed leaves**
* **4 routing heads**
* **beam width 2**
* **top-2 selected leaf reads**
* **fused readout into the existing LM head**

Deferred from core v1:

* separate bounded `MemoryBank`
* learned merge gating
* routing early-stop policy
* learned eviction
* large-scale training ambitions

This version should optimize for:

* trainability
* observability
* ablatability
* causal correctness

---

## Core architecture

### 1. Tokenizer and embeddings

Reuse the current tokenizer-backed input path and embedding path.

This remains:

* autoregressive
* token-based
* compatible with the current manifest and experiment surfaces

The LM head remains the output interface. It is **not** the internal reasoning substrate.

---

### 2. Causal leaf blocks

Sequence tokens are grouped into fixed-size leaf blocks.

Recommended v1 leaf size:

* `16`

A leaf is **live** while tokens are being appended to it.
A leaf becomes **sealed** once it reaches capacity.

Only sealed leaves are inserted into the global summary tree.

This gives:

* clean causality
* stable multiscale memory organization
* explicit boundaries for exact local reads
* predictable runtime structure

---

### 3. Causality contract

At autoregressive step `t`:

* the live block may contain only tokens up to `t`
* unfinished future positions inside the live block are never visible
* the global summary tree contains only sealed prior leaf blocks
* sparse routing over the global memory path may only target sealed tree nodes
* exact leaf reads through the global memory path may only target sealed leaves
* short-range access inside the live block is handled only by the local causal trunk

This contract must hold in both training and incremental inference.

---

### 4. Local causal trunk

Each live leaf block is processed by a local causal trunk.

Recommended v1 options:

* a narrowed p1-family recurrent cell
* a tiny selective SSM-like cell
* depthwise causal convolution plus recurrent gate

v1 should start with the easiest path to train and debug, not the most ambitious cell.

The key shift from v1 is:

* recursion lives primarily within the local processor and the summary tree
* not as an inner repeated fixed-point loop over a single token with no explicit memory geometry

---

### 5. Parallel roots

Replace the single-root hidden state with `H` parallel roots.

Recommended v1 range:

* `2`
* `4`

Each root:

* keeps its own recurrent state
* processes the same token stream
* can specialize
* may share cell family structure with other roots

These are **not** transformer heads.

They are parallel latent processors whose diversity should be measurable.

v1 must compare multi-root against single-root baselines at **equal total state budget**, not only at equal root width.

---

### 6. Regular dyadic summary tree

Sealed leaf blocks are recursively merged into a **regular causal dyadic tree**.

Each internal node stores a summary of its covered span.

Each summary should preserve:

* coarse semantic content
* enough key-like information for retrieval
* enough value-like information for downstream fusion

The merge family should be:

* shared across scales
* modulated by a scale embedding

This is the self-similarity contract:

* same merge family
* scale-aware modulation
* not rigid identical behavior across all depths

### v1 tree rule

v1 uses a **regular** dyadic tree over sealed leaves.

That means:

* every sealed leaf participates
* parent construction is deterministic
* tree structure is fixed by leaf order and leaf size

Learned merge scheduling is deferred.

---

### 7. Sparse fractal routing heads

Global retrieval happens through routing heads operating over the summary tree.

Each routing head:

* begins at the root
* scores only a small candidate set at each level
* keeps a small top-`B` beam
* descends level by level toward promising spans
* stops only when it reaches a selected leaf in v1

This gives:

* query-specific multiscale retrieval
* coarse-to-fine search
* bounded compute per query
* a direct substitute for global dense attention patterns

Recommended v1 values:

* `4` routing heads
* beam width `2`

---

### 8. Competitive retrieval

Retrieval should not collapse into additive soft gating only.

v1 should keep explicit competition:

* per-head competition over candidate nodes
* optional competition across retrieved spans during fusion

The goal is to preserve what efficient models often lose:

* sharp selection
* head diversity
* query-specific retrieval

---

### 9. Exact leaf read path

When a routing head reaches a selected sealed leaf block, it may perform an **exact local read** within that block.

This path is the architecture’s answer to:

* copy behavior
* associative retrieval
* token-precise recall

v1 exact leaf read must mean one explicit mechanism, chosen in code:

* local attention over cached token-level keys and values inside the selected leaf
* pointer-style selection over cached token-level hidden states inside the selected leaf
* copy-distribution read over token positions inside the selected leaf

A summary-only leaf object is not enough.

The implementation must define a typed token-level leaf cache.

---

### 10. Read fusion and LM head

The final readout fuses:

* per-root recurrent readouts
* retrieved tree values
* exact leaf read values
* optional coarse global summary features

That fused representation is then projected through the existing language-model head abstraction.

---

## Complexity and resource targets

Assume:

* local leaf size is constant `c`
* number of routing heads is constant `H`
* beam width is constant `B`
* sparse read count is constant `k`

Then:

* local processing is `O(n)`
* tree construction is `O(n)`
* routing over the tree is `O(H * B * n log(n / c))`

Treating `H`, `B`, `c`, and `k` as fixed constants gives the target:

* **sequence complexity `O(n log n)`**

Resource target:

* **persistent summary memory grows linearly with sequence length**
* **per-token routing/read budget remains bounded**
* **no dense global quadratic attention cache is required**

The important claim is not asymptotic purity on every implementation detail.

The important claim is:

**query-specific multiscale retrieval without global quadratic attention**

---

## Typed runtime and storage contracts

v2 should not hide its architecture in prose.

It needs explicit typed state surfaces.

These types describe **architectural ownership**, not necessarily final hot-path storage layout.
The runtime implementation should prefer **batched tensor layouts** over `Vec`-of-heap-objects in performance-critical code.

### Root state

```rust
pub struct MultiRootState<B: Backend> {
    pub recurrent: Tensor<B, 3>,   // [batch, roots, d_state]
    pub read_intent: Tensor<B, 3>, // [batch, roots, d_q]
    pub write_intent: Tensor<B, 3>,// [batch, roots, d_q]
}
```

### Leaf summaries

```rust
pub struct LeafSummaryStore<B: Backend> {
    pub keys: Tensor<B, 3>,   // [batch, leaves, d_k]
    pub values: Tensor<B, 3>, // [batch, leaves, d_v]
    pub spans: Vec<(usize, usize)>,
}
```

### Tree summaries

```rust
pub struct TreeLevelStore<B: Backend> {
    pub keys: Tensor<B, 3>,   // [batch, nodes_at_level, d_k]
    pub values: Tensor<B, 3>, // [batch, nodes_at_level, d_v]
    pub level: usize,
    pub spans: Vec<(usize, usize)>,
}
```

### Exact leaf token cache

```rust
pub struct LeafTokenCache<B: Backend> {
    pub keys: Tensor<B, 4>,   // [batch, leaves, tokens_per_leaf, d_k]
    pub values: Tensor<B, 4>, // [batch, leaves, tokens_per_leaf, d_v]
    pub mask: Tensor<B, 3>,   // [batch, leaves, tokens_per_leaf]
    pub spans: Vec<(usize, usize)>,
}
```

### Retrieval policy

```rust
pub struct RetrievalPolicy {
    pub beam_width: usize,
    pub top_k_reads: usize,
    pub allow_early_stop: bool,
}
```

### Merge policy

```rust
pub enum MergeCheckpointPolicy {
    FixedLeafSize { tokens_per_leaf: usize },
}
```

v1 uses only `FixedLeafSize`.

---

## New module families

v2 should be built from modules with clear ownership boundaries:

* `LocalTrunk`
  owns per-leaf token processing

* `LeafSummarizer`
  owns leaf key/value creation

* `TreeMergeCell`
  owns parent summary creation from two child summaries

* `FractalRouterHead`
  owns coarse-to-fine sparse routing

* `ReadFusion`
  owns fusion of root state and retrieved memory

* `FractalV2Model`
  owns the autoregressive forward contract

A separate `MemoryBank` module is deferred until the tree-only design is proven useful.

This is cleaner than trying to force one rule trait to express the entire architecture.

---

## What we reuse

This is not a new repository and not a total rewrite.

We should reuse:

* tokenizer-backed data path
* embedding path
* structured projection surfaces where they remain useful
* language-model head abstraction
* diagnostics and failure snapshots
* manifest, experiment, and tournament control surfaces
* skinny repro tooling
* narrowed p1-family cells as local trunk candidates

The reuse principle is:

* keep what is already helping observability, training hygiene, and output projection
* narrow old components into the roles they actually serve well

---

## What we replace

v2 should replace or retire:

* single-root-only model assumptions
* the current `FractalRule` shape as the main global abstraction
* current state layouts that assume one root is the whole model
* router-only adaptive depth as the main global memory strategy

v2 needs contracts for:

* local trunk processing
* leaf summarization
* tree merge
* sparse routing
* exact leaf read
* fusion

---

## Forward pass sketch

For each new token:

1. embed the token
2. update each root’s local recurrent state
3. append token-local state into the live leaf cache
4. if the leaf seals:

   * summarize the leaf
   * append its summary to tree level 0
   * recursively update dyadic parent summaries
5. form routing queries from current root states
6. route over sealed tree nodes using sparse beam search
7. descend to selected sealed leaves
8. perform exact local reads within selected leaves
9. fuse root state and retrieved values
10. produce vocabulary logits through the LM head

This sequence defines the intended v1 execution model.

---

## Train-time routing semantics

v1 routing should be sparse but trainable.

Default v1 contract:

* candidate nodes are scored at each level
* beam pruning is applied per head
* normalization is performed only over the surviving candidate set
* early stop is disabled in the first runnable version
* gradients flow through surviving scored candidates
* pruned branches do not receive gradient
* optional teacher interval supervision may be added later

The first implementation should prefer simple, inspectable sparse routing over sophisticated policy learning.

---

## Training strategy

v2 should not rely on next-token loss alone to invent routing.

Recommended recipe:

1. **start with next-token cross entropy**
2. add explicit synthetic behavior probes:

   * copy
   * induction
   * associative recall
   * far-token comparison
   * noisy retrieval
3. add retrieval supervision or distillation when possible
4. add optional discipline losses for:

   * root collapse
   * routing collapse
   * dead-head behavior

Best teacher option:

* distill interval-level retrieval behavior from a strong transformer or hybrid teacher

What we should **not** do first:

* giant-scale from-scratch training
* wavelet purity for its own sake
* a fully learned routing stack with no diagnostics

---

## Observability requirements

The implementation is incomplete without strong observability.

v1 should expose at least:

* root-state similarity / collapse metrics
* routing depth histogram per head
* candidate entropy per head
* selected-span distance histogram
* fraction of steps reaching exact leaf reads
* leaf usage distribution
* dead-tree / unused-node diagnostics

If later features add a side-memory bank, then also track:

* slot utilization
* eviction churn
* owner-root dominance

---

## Required ablations

At equal total state / parameter budget, the first proving ladder should compare:

1. **single-root, no memory**
2. **multi-root, no memory**
3. **single-root, summaries only, no retrieval**
4. **single-root, sparse tree retrieval**
5. **single-root, sparse tree retrieval, exact leaf read**
6. **multi-root, summaries only, no retrieval**
7. **multi-root, sparse tree retrieval**
8. **multi-root, sparse tree retrieval, no exact leaf read**
9. **multi-root, sparse tree retrieval, exact leaf read**

Additional deferred ablations:

* fixed merge cadence vs learned merge scheduling
* tree-only memory vs tree plus side-memory bank

This ladder is designed to separate:

* root multiplicity effects
* multiscale summary effects
* sparse retrieval effects
* exact leaf read effects

---

## Deferred features

The following are intentionally **not** part of core v1:

* bounded associative `MemoryBank`
* learned eviction
* owner-root slot policies
* learned merge gating
* routing early-stop policy
* giant-scale training
* dense global token memory fallback

These may be introduced later only after the tree-only architecture is stable, instrumented, and ablated.

---

## Falsifiability

This architecture should be considered wrong if it cannot demonstrate one or more of:

* better retrieval behavior than pure recurrence
* comparable or better perplexity than the v1 baseline at equal budget
* measurable head and root specialization
* meaningful sparse routing rather than collapse
* practical sequence scaling better than dense attention baselines at relevant lengths

We should not protect it with vague wins.

If:

* routing collapses
* roots collapse
* exact leaf read is the only thing that matters
* the local trunk does all the work while the tree becomes dead weight
* the sparse machinery behaves like decorative complexity

then the thesis needs revision.

---

## Implementation sequence

### Phase 1 — new architecture surfaces

Add new typed modules and state contracts without disturbing `recursive-kernel-v1`.

Deliverables:

* `FractalV2State`
* `LocalTrunk`
* `LeafSummarizer`
* `TreeMergeCell`
* `FractalRouterHead`
* `ReadFusion`

### Phase 2 — multi-root no-memory baseline

Prove:

* multi-root execution works
* diagnostics expose collapse and divergence correctly

### Phase 3 — dyadic summaries without retrieval

Prove:

* tree build is causal
* completed-leaf summarization is stable
* summaries are not dead artifacts

### Phase 4 — sparse retrieval

Prove:

* heads actually specialize
* routing remains sparse
* retrieval materially changes behavior

### Phase 5 — exact leaf read

Prove:

* copy and retrieval improve
* exact local access helps without collapsing into dense global memory

### Phase 6 — teacher-guided routing

Only after the above is instrumented and stable.

### Phase 7 — deferred side-memory experiments

Only after the tree-only architecture is clearly understood.

---

## Bottom line

This is the recommended successor direction for fractal.

It preserves the strongest part of the original intuition:

* recursion and state are still central

It gives up the weakest part:

* one recurrent state should replace all explicit memory

The most important design shift is:

**fractality should live in the memory tree and routing geometry, not only in the local primitive formula**

If v2 works, it should work because:

* local recurrent processing is cheap
* global retrieval is sparse and query-specific
* memory is multiscale and causal
* exact copy paths exist where they are needed

That is a much stronger and more defensible thesis than continuing to force the current v1 line to become transformer-equivalent through inner recursion alone.


