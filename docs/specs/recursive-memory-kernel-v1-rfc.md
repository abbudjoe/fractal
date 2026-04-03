# RFC: Recursive Memory Kernel v1

## Status
Draft

## Purpose
Define the first implementable v2 successor to the current single-root recursive language-model line.

This RFC is intentionally narrower than the full architecture spec.  
Its job is to make the first proving version easy to build, easy to instrument, and easy to falsify.

## Decision Summary

We are changing the architectural thesis from:

- one recurrent state should carry the whole sequence

to:

- recurrent state remains primary
- explicit global memory is organized as a causal multiscale tree
- retrieval is sparse and query-specific
- exact local reads are preserved where copy/retrieval behavior requires them

Fractality moves from pointwise primitive math into:

- memory organization
- multiscale summary construction
- routing geometry

## Why this exists

The current v1 line produced useful infrastructure and diagnostics, but it has not yet produced a convincing Stage 0 training path.

The main conceptual issue is now clear:

- compressed recurrent state is useful
- compressed recurrent state alone is unlikely to replace addressable memory for:
  - long-range retrieval
  - selective copy
  - distant comparison
  - far-apart binding

This RFC defines a narrower successor architecture that keeps recursion while adding explicit multiscale memory.

## Scope of v1

### Included
- local recurrent/selective token processing
- fixed-size causal leaf blocks
- regular dyadic summary tree over sealed leaves
- sparse coarse-to-fine retrieval over the tree
- exact local reads within selected sealed leaves
- reuse of existing tokenizer, embeddings, LM head, diagnostics, and experiment control surfaces
- causal auditing of whether memory paths materially affect predictions

### Excluded
- separate bounded side-memory bank
- learned merge scheduling
- routing early-stop policy
- learned eviction
- giant-scale training
- dense global token memory fallback

## Core architecture

### 1. Local trunk
Each token is processed by a local recurrent or selective trunk inside the current live leaf block.

Acceptable v1 starting points:
- narrowed p1-family cell
- tiny selective SSM-like cell
- causal conv plus recurrent gate

The v1 goal is trainability and observability, not maximum novelty.

### 2. Leaf blocks
Tokens are grouped into fixed-size causal leaf blocks.

v1 default:
- leaf size = 16

A leaf is:
- **live** while receiving tokens
- **sealed** once full

Only sealed leaves enter the global tree.

### 3. Dyadic summary tree
Sealed leaves are recursively merged into a regular causal dyadic tree.

Each node stores:
- retrieval key-like content
- downstream value-like content
- span metadata

The merge family is shared across levels and modulated by scale embedding.

v1 uses a **regular** tree:
- deterministic construction
- no learned merge gating
- every sealed leaf participates

### 4. Parallel roots
Replace the single-root state with 2 to 4 parallel roots.

Each root:
- maintains its own recurrent state
- processes the same token stream
- may specialize

These are not attention heads. They are parallel latent processors.

### 5. Sparse routing
Routing heads perform coarse-to-fine retrieval over the sealed tree.

Each head:
- starts at the root
- scores a small candidate set per level
- keeps a small beam
- descends to promising spans
- reaches selected leaves in v1

v1 default:
- 4 routing heads
- beam width 2

### 6. Exact leaf read
When a routing head selects a sealed leaf, it performs an exact local read within that leaf.

This exists to preserve:
- copy behavior
- associative retrieval
- token-precise recall

Allowed v1 mechanisms:
- local attention over cached token-level K/V inside the leaf
- pointer-style selection over cached token-level hidden states
- copy-distribution read over leaf token positions

Summary-only retrieval is not enough.

### 7. Read fusion
Final readout fuses:
- root recurrent readouts
- retrieved tree summaries
- exact leaf read values

The fused result is projected through the existing LM head.

## Causality contract

At autoregressive step `t`:

- the live leaf contains only tokens up to `t`
- unfinished future positions inside the live leaf are never visible
- the global tree contains only sealed prior leaves
- global routing may only target sealed tree nodes
- exact global-memory leaf reads may only target sealed leaves
- short-range access inside the live leaf is handled only by the local trunk

This must hold in training and incremental inference.

## Runtime shape

The implementation should expose typed architectural surfaces, but use backend-friendly batched storage in hot paths.

Minimum required state families:

- multi-root recurrent state
- leaf summary store
- tree level stores
- exact leaf token cache
- retrieval policy

Conceptual ownership is important, but hot paths should prefer flat tensor layouts over heap-heavy object graphs.

## Complexity target

Assuming constant:
- leaf size
- routing heads
- beam width
- sparse read count

Target behavior is:

- local processing: `O(n)`
- tree construction: `O(n)`
- sparse routing: `O(n log n)`

Overall target:
- **sequence complexity `O(n log n)`**

Persistent memory grows with sequence length through the summary tree.  
Per-token routing/read budget remains bounded.

## Train-time policy

v1 should not depend on next-token loss alone to invent good routing.

Default v1 training behavior:
- sparse beam pruning per head
- normalization over surviving candidates only
- no early stop in first runnable version
- gradients flow through surviving candidates
- pruned branches receive no gradient

Recommended training path:
1. next-token cross entropy
2. synthetic behavior probes
3. optional teacher-guided interval retrieval supervision
4. optional anti-collapse losses

## Observability requirements

The implementation is incomplete without diagnostics for:

- root collapse / similarity
- routing depth histogram
- candidate entropy per head
- selected-span distance histogram
- fraction of steps reaching exact leaf reads
- leaf usage distribution
- dead-tree / unused-node behavior

## Causal Memory Auditor

The implementation is also incomplete without a sampled counterfactual auditing path that measures whether the memory system is actually doing useful work.

Required first interventions:
- no-tree-read
- no-exact-leaf-read
- next-best-span substitution
- root-drop intervention

Required outputs:
- loss delta
- target-logit delta
- KL divergence from full forward
- utility by root
- utility by routing depth
- utility by task family

This exists to detect:
- decorative tree usage
- redundant roots
- exact-read domination
- routing that looks structured but does not affect outcomes

## Minimum executable v1

Ship the smallest version that can test the thesis:

- 2 to 4 roots
- leaf size 16
- simple local recurrent/selective trunk
- regular dyadic tree over sealed leaves
- 4 routing heads
- beam width 2
- top-2 leaf reads
- fused LM head readout
- sampled causal memory auditing

This version should optimize for:
- trainability
- observability
- ablatability
- causal correctness

## Required ablations

At equal total state / parameter budget, compare:

1. single-root, no memory
2. multi-root, no memory
3. single-root, summaries only
4. single-root, sparse tree retrieval
5. single-root, sparse tree retrieval + exact leaf read
6. multi-root, summaries only
7. multi-root, sparse tree retrieval
8. multi-root, sparse tree retrieval without exact leaf read
9. multi-root, sparse tree retrieval + exact leaf read

These ablations separate:
- root multiplicity effects
- multiscale summary effects
- sparse retrieval effects
- exact-read effects

## Success criteria

This RFC is supported if v1 shows one or more of:

- better retrieval behavior than pure recurrence
- comparable or better perplexity than v1 baseline at equal budget
- measurable root/head specialization
- meaningful sparse routing
- practical scaling better than dense attention baselines at relevant lengths
- causal auditing shows that memory paths materially affect predictions

## Failure criteria

This RFC should be reconsidered if:

- routing collapses
- roots collapse
- the tree becomes dead weight
- exact leaf read is the only useful component
- the local trunk does all meaningful work
- sparse retrieval adds complexity without measurable benefit

## Implementation phases

### Phase 1
Add new typed module boundaries:
- `LocalTrunk`
- `LeafSummarizer`
- `TreeMergeCell`
- `FractalRouterHead`
- `ReadFusion`
- `FractalV2Model`

### Phase 2
Prove multi-root no-memory baseline works.

### Phase 3
Prove causal dyadic summaries work without retrieval.

### Phase 4
Add sparse retrieval and verify specialization.

### Phase 5
Add exact leaf read and verify copy/retrieval gains.

### Phase 6
Add the Causal Memory Auditor and verify memory-path utility.

### Phase 7
Only then add teacher-guided routing or other deferred features.

## Recommendation

Proceed with a tree-only v1.

Do **not** add side-memory bank complexity until the tree-only architecture is:
- implemented
- instrumented
- ablated
- causally audited
- behaviorally justified

That is the cleanest path to testing whether recursive multiscale memory can replace most of the role currently played by dense attention.
