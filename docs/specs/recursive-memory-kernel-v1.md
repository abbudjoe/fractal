# Fractal v2 Architecture

## Recursive Memory Kernel v1

This document defines the first serious successor architecture to the current
single-root recursive language-model line.

It does not retroactively make `recursive-kernel-v1` successful.
It does not erase the current Stage 0 debugging history.

It states a new thesis:
- keep recursion
- stop asking one recurrent state to replace all addressable memory
- move fractality from pointwise primitive math into memory organization and routing geometry

This is intended to be implemented on the `fractal-v2` branch family while
reusing the strongest pieces of the current repository.

## Why v2 Exists

The current v1 line made three large bets:
- one recurrent root can carry the whole sequence state
- inner recursive depth can stand in for explicit memory organization
- router-driven early exit can supply most of the efficiency gains

Those bets produced useful work:
- strong diagnostics
- better projection ownership
- a cleaner LM head
- a reusable experiment control plane

They did not yet produce a convincing Stage 0 training path.

The strongest conceptual problem is now clearer:
- compressed recurrent state is likely useful as **primary memory**
- compressed recurrent state is unlikely to fully replace **content-addressable memory**
- especially for:
  - long-range retrieval
  - selective copy
  - distant comparison
  - binding across far-apart regions

So v2 is not:
- abandon recursion

It is:
- keep recursion as the dominant compute substrate
- add explicit, bounded, sparse, typed memory where the recurrent bottleneck is real

## The New Thesis

`recursive-memory-kernel-v1` should be a language model where:
- local token processing is recurrent or selective-state-space-like
- tokens are organized into completed causal leaf blocks
- completed blocks are recursively summarized into a dyadic tree
- multiple routing heads perform sparse coarse-to-fine retrieval over that tree
- selected leaves support exact local reads for copy/retrieval behavior
- recurrent state remains primary
- explicit memory remains bounded
- dense all-token all-head attention is not the default substrate

In short:
- state first
- memory second
- attention only where it earns its cost

## Design Goals

The architecture should optimize for:
- better long-range retrieval than pure single-state recurrence
- subquadratic sequence scaling
- explicit and typed memory/routing contracts
- high observability
- ablations that can falsify the thesis quickly

The architecture should **not** optimize for:
- maximum novelty in primitive formulas
- exact imitation of transformer block structure
- hiding routing or memory policy inside opaque heuristics

## Core Architectural Shape

### 1. Tokenizer and embeddings

Reuse the current tokenizer-backed input path and embedding path.

This remains:
- autoregressive
- token-based
- compatible with the current manifest and experiment surfaces

### 2. Causal leaf blocks

Sequence tokens are grouped into leaf blocks of fixed size.

Recommended v1 range:
- `16`
- `32`

The key causal contract is:
- the current block can only see tokens already present in that block
- only **completed** prior blocks may be summarized upward into the global tree

This gives clean causality and a natural multiscale memory structure.

### 3. Local causal trunk

Each leaf block is processed by a local causal trunk.

The local trunk is where we should first reuse current work.

Recommended v1 options:
- p1-family recurrent cell used across tokens inside the leaf
- tiny selective SSM-like cell
- depthwise causal conv plus recurrent gate

V1 should start with the easiest path to train and debug, not the most ambitious cell.

Important shift from v1:
- recursion lives primarily **within the local processor and the summary tree**
- not as an inner repeated fixed-point loop over a single token with no explicit memory geometry

### 4. Parallel roots

Replace the single-root hidden state with `H` parallel roots.

Recommended v1 range:
- `4`
- `8`

Each root:
- keeps its own recurrent state
- processes the same token stream
- can specialize
- can optionally share cell family structure with other roots

These are not transformer heads.

They are parallel latent processors whose diversity should be measurable.

### 5. Dyadic summary tree

Completed leaf blocks are recursively merged into a causal dyadic tree.

Each internal node stores a summary of its covered span.

Each summary should preserve:
- coarse semantic content
- enough key-like information for retrieval
- enough value-like information for downstream fusion

The merge cell should be shared across scales but modulated by a scale embedding.

That is the self-similarity contract:
- same merge family
- scale-aware modulation
- not rigid identical behavior across all depths

### 6. Sparse fractal routing heads

Global retrieval happens through routing heads operating over the summary tree.

Each head:
- begins at the root
- scores only a small candidate set at each level
- keeps top-`B` candidates
- descends until it stops early or reaches selected leaf blocks

This gives:
- coarse semantic read when early stopping is enough
- fine exact retrieval when descent is necessary

Recommended v1 range:
- `8` routing heads
- beam width `2` to `4`

### 7. Competitive retrieval

Do not let retrieval collapse into additive soft gating only.

V1 should keep explicit competition:
- per-head competition over candidate nodes
- optional competition across heads or scales

The goal is to preserve what efficient models often lose:
- sharp selection
- head diversity
- query-specific retrieval

### 8. Exact local read path

When a head descends to a leaf block, it should be able to read from that block exactly.

This is the architecture's answer to:
- copy behavior
- associative retrieval
- token-precise recall

This does not require a full global KV cache.

It does require:
- a typed leaf-memory contract
- exact read semantics inside the selected local block

### 9. Readout and LM head

Reuse the current LM head concept.

The readout should fuse:
- per-root readouts
- selected memory or tree values
- optional global summary features

Then project to vocabulary logits through the existing LM head abstraction.

The LM head remains:
- the output interface
- not the internal reasoning substrate

## Complexity Target

If:
- local leaf size is constant `c`
- number of routing heads is constant `H`
- beam width is constant `B`
- sparse read count is constant `k`

Then:
- local processing is `O(n)`
- tree construction is `O(n)`
- routing over the tree is `O(H * B * n log(n / c))`

Treating `H`, `B`, `c`, and `k` as fixed constants gives:
- sequence complexity target `O(n log n)`

Memory target:
- `O(n)` persistent memory

The important claim is not exact asymptotic purity on every implementation detail.

The important claim is:
- query-specific multiscale retrieval without global quadratic attention

## Typed Runtime Contracts

v2 should not hide its architecture in prose.

It needs explicit typed state surfaces.

### Root state

The single `FractalState` contract in [state.rs](/Users/joseph/fractal/fractal-core/src/state.rs)
is too narrow for v2.

v2 should introduce a state family more like:

```rust
pub struct RootState<B: Backend> {
    pub recurrent: Tensor<B, 2>,
    pub write_intent: Tensor<B, 2>,
    pub read_intent: Tensor<B, 2>,
}

pub struct MultiRootState<B: Backend> {
    pub roots: Vec<RootState<B>>,
}
```

### Leaf summaries

```rust
pub struct LeafSummary<B: Backend> {
    pub key: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
    pub local_pointer_state: Tensor<B, 3>,
    pub span_start: usize,
    pub span_len: usize,
}
```

### Tree node summaries

```rust
pub struct TreeNodeSummary<B: Backend> {
    pub key: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
    pub level: usize,
    pub span_start: usize,
    pub span_len: usize,
}
```

### Memory bank

```rust
pub struct MemorySlot<B: Backend> {
    pub key: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
    pub age: usize,
    pub owner_root: usize,
}

pub struct MemoryBank<B: Backend> {
    pub slots: Vec<MemorySlot<B>>,
}
```

### Retrieval and merge policy

```rust
pub enum MergeCheckpointPolicy {
    EveryNTokens { n: usize },
    LearnedGate { threshold: f32 },
}

pub struct RetrievalPolicy {
    pub beam_width: usize,
    pub top_k_reads: usize,
    pub allow_early_stop: bool,
}
```

These are sketches, not final APIs.

The point is:
- roots
- memory
- tree summaries
- retrieval
- merge cadence

must all become explicit owned surfaces.

## What We Reuse

This is not a new repository and not a total rewrite.

We should reuse:
- tokenizer-backed data path in [model.rs](/Users/joseph/fractal/fractal-core/src/model.rs)
- embeddings in [model.rs](/Users/joseph/fractal/fractal-core/src/model.rs)
- `StructuredProjection` in [projection.rs](/Users/joseph/fractal/fractal-core/src/projection.rs)
- `LanguageModelHead` in [language_model_head.rs](/Users/joseph/fractal/fractal-core/src/language_model_head.rs)
- diagnostics and failure snapshots in [diagnostics.rs](/Users/joseph/fractal/fractal-core/src/diagnostics.rs) and [lifecycle.rs](/Users/joseph/fractal/fractal-core/src/lifecycle.rs)
- manifest, experiment, and tournament surfaces in [examples/tournament.rs](/Users/joseph/fractal/examples/tournament.rs)
- skinny repro tooling in [skinny-matmul-repro.rs](/Users/joseph/fractal/src/bin/skinny-matmul-repro.rs)

We may also reuse:
- p1-family cells as local trunk candidates

But only in a narrowed role:
- local processor
- not the entire global memory substrate

## What We Replace

v2 should replace or retire:
- single-root-only model assumptions in [model.rs](/Users/joseph/fractal/fractal-core/src/model.rs)
- the current `FractalRule` contract in [rule_trait.rs](/Users/joseph/fractal/fractal-core/src/rule_trait.rs)
- the current `FractalState` layout family in [state.rs](/Users/joseph/fractal/fractal-core/src/state.rs)
- router-only adaptive depth as the main global memory strategy

The current `FractalRule` contract is too small because it only expresses:
- `state`
- `x`
- `apply`

v2 needs contracts for:
- local trunk processing
- tree merge
- memory read/write
- routing
- fusion

## New Module Families

v2 should be built from modules with clear ownership:

1. `LocalTrunk`
- owns per-leaf token processing

2. `LeafSummarizer`
- owns leaf key/value creation

3. `TreeMergeCell`
- owns parent summary creation from two children

4. `FractalRouterHead`
- owns coarse-to-fine routing

5. `MemoryBank`
- owns bounded slot storage and eviction

6. `ReadFusion`
- owns fusion of root state and retrieved memory

7. `FractalV2Model`
- owns the autoregressive forward contract

This is cleaner than trying to make one rule trait do everything.

## Training Strategy

v2 should not rely on next-token loss alone to invent routing.

### Recommended training recipe

1. Start with next-token cross entropy.

2. Add explicit synthetic behavior probes:
- copy
- induction
- associative recall
- far-token comparison
- noisy retrieval

3. Add retrieval supervision or distillation when possible.

Best option:
- distill interval-level retrieval behavior from a transformer or strong hybrid teacher

4. Add auxiliary memory discipline losses:
- discourage slot collapse
- discourage root collapse
- encourage routing diversity

5. Keep all auxiliary surfaces optional and typed in the manifest.

### What we should not do

Do not begin with:
- giant-scale from-scratch training
- wavelet purity for its own sake
- a fully learned routing stack with no diagnostics

## Minimum Executable v1

The first executable v2 should stay intentionally small.

Recommended first contract:
- `4` roots
- leaf size `16`
- local p1-style or simple selective trunk
- `32` memory slots
- dyadic summary tree over completed leaves
- `4` routing heads
- beam width `2`
- top-`2` reads
- merge every `16` tokens
- fused readout into the existing LM head

This version should optimize for:
- trainability
- observability
- ablatability

Not leaderboard ambitions.

## Required Ablations

The first proving ladder should compare:

1. single-root, no memory

This is the closest conceptual bridge to v1.

2. multi-root, no memory

Tests whether specialization alone helps.

3. multi-root, write-only tree summaries

Tests whether explicit multiscale summaries help before retrieval.

4. multi-root, sparse tree retrieval

This is the first real v2 architecture.

5. multi-root, sparse tree retrieval, no exact leaf read

Tests whether exact local read matters for copy/retrieval.

6. multi-root, sparse tree retrieval, exact leaf read

Tests the full v1 thesis.

7. fixed merge cadence vs learned merge gating

Tests whether learned scheduling is worth the complexity.

## Falsifiability

This architecture should be considered wrong if it cannot demonstrate one or more of:
- better retrieval behavior than pure recurrence
- comparable or better perplexity than the v1 baseline at equal budget
- strong head/root specialization
- meaningful sparse routing rather than collapse
- practical sequence scaling better than dense attention baselines

We should not protect it with vague wins.

If:
- routing collapses
- memory slots collapse
- exact leaf read becomes the only thing that matters
- or the local trunk does all the work while the tree becomes dead weight

then the thesis needs revision.

## Implementation Sequence

### Phase 1: new architecture surfaces

Add new typed modules and state contracts without disturbing `recursive-kernel-v1`.

Deliverables:
- `FractalV2State`
- `LocalTrunk`
- `LeafSummarizer`
- `TreeMergeCell`
- `FractalRouterHead`

### Phase 2: multi-root no-memory baseline

Prove:
- multi-root execution works
- diagnostics expose root collapse/divergence

### Phase 3: dyadic summaries without retrieval

Prove:
- tree build is causal
- completed-leaf summarization is stable

### Phase 4: sparse retrieval

Prove:
- heads actually specialize
- routing remains sparse

### Phase 5: exact leaf read

Prove:
- copy and retrieval improve
- without collapsing into dense token memory

### Phase 6: teacher-guided routing

Only after the above is instrumented and stable.

## Bottom Line

This is the recommended successor direction for `fractal`.

It preserves the strongest part of the original intuition:
- recursion and state are still central

It gives up the weakest part:
- one recurrent state should replace all explicit memory

The most important design shift is:
- fractality should live in the **memory tree and routing geometry**
- not only in the local primitive formula

If v2 works, it should work because:
- local recurrent processing is cheap
- global retrieval is sparse and query-specific
- memory is multiscale and causal
- exact copy paths still exist where they are needed

That is a much stronger and more defensible thesis than continuing to force the current v1 line to become transformer-equivalent through inner recursion alone.
