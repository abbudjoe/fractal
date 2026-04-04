# Fractal Hybrid v3

## Purpose

This document reframes the next architecture line around the two goals that
actually matter:

1. match frontier-style language-model accuracy more efficiently
2. extend useful context without paying full dense-attention cost everywhere

It is **not** another attempt to replace transformer behavior outright with a
pure recursive primitive.

The working thesis is:

* attention remains the best exact token-interaction primitive
* recurrent or selective-state compute is still valuable as a cheaper
  background processor
* the v2 tree memory is still valuable as a long-range retrieval subsystem
* the winning design is likely a **hybrid hot/warm/cold memory stack**

In short:

**attention for exact interaction**
**recurrence for cheap continuous processing**
**tree memory for long-range searchable storage**

---

## Why v3 exists

The v2 tree-only line produced useful infrastructure:

* typed state and module boundaries
* causal tree storage
* exact leaf-read plumbing
* synthetic retrieval probes
* causal memory auditing
* benchmark and ledger surfaces

But the core replacement thesis failed its fairest rescue attempts:

* next-token loss improved
* oracle routing did not rescue probe accuracy
* oracle exact-token selection did not rescue probe accuracy
* supervised synthetic retrieval did not rescue held-out behavior
* multi-root behavior collapsed
* retrieved evidence reached the model but did not become reliable output gain

That means the next architecture should **salvage the parts that help** and
**retire the claim that recursion alone can replace transformer-grade exact
token interaction**.

---

## New thesis

The next serious architecture should be a **hybrid language model** where:

* a transformer-like path handles exact copying, in-context retrieval, and
  token-to-token comparison
* a recurrent or selective-state path handles cheap continuous processing
* sealed-leaf tree memory extends effective context beyond the hot attention
  region
* retrieved memory is integrated through a small attention-style mechanism,
  not injected directly into logits
* efficiency comes from reducing how much full attention must stay live, not
  from deleting exact interaction entirely

This is a different target:

* not "beat transformers by eliminating attention"
* but "use less attention, less KV, and less high-precision compute while
  keeping transformer-level strengths where they matter"

---

## Architecture Overview

Think of the model as a three-tier memory system.

### 1. Hot memory — exact interaction path

This is the smallest part of the model, but the most important.

Responsibilities:

* next-token prediction
* local copying
* token-to-token comparison
* integrating retrieved evidence into the active token stream

Recommended form:

* transformer or latent-attention-style blocks
* local or sliding-window causal self-attention
* optional retrieved-memory cross-attention

This is the path that should remain closest to transformer behavior.

### 2. Warm memory — recurrent or selective-state path

This path runs continuously and cheaply.

Responsibilities:

* summarize recent context
* keep low-cost continuous state between sparse attention refreshes
* provide routing queries and compressed local context

Recommended form:

* single-root or per-layer recurrent state first
* selective SSM-like or narrowed recurrent trunk
* no multi-root by default in the first proving version

This is where recurrence is most likely to pay off.

### 3. Cold memory — searchable sealed-leaf tree

This is the long-range storage path.

Responsibilities:

* keep sealed span summaries
* keep exact token K/V snapshots for selected leaves
* expose sparse long-range retrieval over old context

Recommended form:

* fixed-size sealed leaves
* regular dyadic tree over leaf summaries
* sparse router over sealed nodes
* exact leaf reads only for selected leaves

This keeps the best part of the v2 memory thesis, but demotes it from
"replacement backbone" to "retrieval sidecar."

---

## Critical design change from v2

In v2, retrieved memory ultimately fed an additive read-fusion path that
projected toward logits.

That is not the right contract for exact evidence.

In v3:

* retrieved summaries and exact leaf reads should be converted into a small
  **retrieval attention context**
* the active token representation should attend to that context
* the output head should read from the updated hidden state, not from a
  directly fused memory delta

This means:

* memory helps by changing the model's internal token representation
* not by acting like a side-channel vote at the output

That is the single most important architectural correction.

---

## Minimal proving version

The first serious hybrid should stay narrow.

### Required components

* single-root recurrent warm path
* local/sliding hot attention path
* sealed leaves of size `16`
* regular dyadic tree over sealed leaves
* sparse routing over sealed memory
* exact reads from selected leaves
* retrieval cross-attention adapter into the hot path
* existing tokenizer and LM head surfaces
* existing synthetic probes, auditor, benchmark, and ledger surfaces

### Explicitly deferred

* multi-root recurrence
* side-memory bank
* learned eviction
* learned merge scheduling
* routing early stop
* giant-scale training
* dense fallback global attention

The first goal is **not** maximum cleverness.
It is to prove that hybrid retrieval can improve the real token predictor.

Before full scaffolding starts, run the narrower pre-validation in
[`hybrid-exact-attention-rescue-prevalidation.md`](./hybrid-exact-attention-rescue-prevalidation.md).
That experiment is the gate between the failed tree-only replacement line and
the full hybrid build.

---

## Keep, Reuse, Retire

### Keep and reuse

From v2, keep:

* typed state ownership in `fractal-core/src/v2/state.rs`
* sealed-leaf and tree storage logic in `fractal-core/src/v2/leaf.rs` and
  `fractal-core/src/v2/tree.rs`
* sparse routing diagnostics in `fractal-core/src/v2/router.rs`
* exact leaf-read storage contracts from `fractal-core/src/v2/exact_read.rs`
* causal auditor surfaces in `fractal-core/src/v2/auditor.rs`
* synthetic probe, benchmark, ablation, checkpoint, and ledger surfaces in
  `fractal-eval-private`

These are the most valuable artifacts from the v2 line.

### Retire as primary design bets

Retire:

* the claim that tree retrieval plus read fusion can replace transformer-grade
  exact token interaction
* multi-root recurrence as a default proving assumption
* additive routed-memory-to-logit fusion as the main integration contract
* "tree-only memory path as the main token predictor"

These were exactly the parts that failed hardest under oracle and supervised
tests.

### Keep only as optional later ablations

Keep available, but only as later ablations:

* multi-root warm path
* direct exact-read-to-output diagnostics
* alternative merge cells
* alternative routing heads

They are no longer part of the minimal thesis.

---

## Expected module map

The repo should treat this as a new architecture family, not a hidden rewrite
inside `v2`.

Suggested module boundary:

* `fractal-core/src/hybrid/mod.rs`
* `fractal-core/src/hybrid/model.rs`
* `fractal-core/src/hybrid/backbone.rs`
* `fractal-core/src/hybrid/state.rs`
* `fractal-core/src/hybrid/memory_tree.rs`
* `fractal-core/src/hybrid/router.rs`
* `fractal-core/src/hybrid/exact_read.rs`
* `fractal-core/src/hybrid/retrieval_adapter.rs`
* `fractal-core/src/hybrid/auditor.rs`

Suggested ownership:

* `HybridModel`
  - owns orchestration
* `HybridBackbone`
  - owns hot attention and warm recurrent blocks
* `HybridState`
  - owns recurrent state, hot-cache handles, and memory-tree handles
* `MemoryTree`
  - owns sealed leaves, summaries, and exact token K/V storage
* `MemoryRouter`
  - owns sparse selection over sealed memory
* `ExactLeafRead`
  - owns token-level retrieval within routed leaves
* `RetrievalAdapter`
  - converts retrieved memory into cross-attention-ready context
* `HybridAuditor`
  - owns causal interventions over hot/warm/cold paths

The key new surface is `RetrievalAdapter`.
That is where retrieved memory becomes something the hot attention path can
actually use.

---

## Forward pass sketch

At step `t`:

1. embed the current token
2. update the warm recurrent state
3. update the hot attention state over the recent active window
4. append current hidden state to the live leaf
5. if a leaf seals:
   * summarize it
   * snapshot exact token K/Vs
   * update the dyadic memory tree
6. form a retrieval query from the active hidden state and warm state
7. route over sealed tree nodes
8. perform exact read inside selected leaves
9. convert retrieved summaries and token reads into retrieval-attention K/Vs
10. run a small retrieval cross-attention step against the active hidden state
11. project the updated hidden state through the LM head

This is materially different from v2 because step `10` is now a model-internal
attention update, not an additive logit-side fusion.

---

## First proving ablations

These ablations should be run at equal total hidden size and as close to equal
parameter budget as practical.

### Backbone ablations

1. hot attention only
2. hot attention + warm recurrent path
3. hot attention + cold tree summaries
4. hot attention + cold tree summaries + exact leaf read
5. hot attention + warm recurrent path + cold tree summaries
6. hot attention + warm recurrent path + cold tree summaries + exact leaf read

### Integration ablations

7. retrieval disabled
8. retrieval summaries only
9. retrieval exact read enabled
10. retrieval cross-attention disabled but routing still logged
11. retrieval cross-attention enabled with oracle leaf
12. retrieval cross-attention enabled with oracle leaf + oracle exact token

### Warm-path ablations

13. single-root warm path
14. multi-root warm path

Do not skip the single-root control this time.

---

## Observability requirements

The hybrid implementation is incomplete without:

* hot-attention latency and cache footprint
* warm-state update latency
* tree update latency
* routing depth histogram
* candidate entropy per head
* selected-span distance histogram
* exact-read usage rate
* retrieval cross-attention usage and norm
* root collapse or similarity if multi-root is enabled
* dead-tree or unused-node behavior
* no-retrieval vs retrieval logit deltas
* no-cross-attention vs cross-attention deltas

The auditor must measure not only whether memory is active, but whether:

* retrieved memory changes hidden states
* changed hidden states improve prediction
* the improvement survives oracle forcing

---

## Quantization strategy

The quantization goal should be staged, not ideological.

### Likely easier to quantize aggressively

* warm recurrent path
* tree summaries
* sealed leaf metadata
* older cold-memory snapshots

### Likely more precision-sensitive

* hot attention projections
* retrieval cross-attention
* exact token K/Vs used for the live retrieval step
* LM head projection

That suggests a practical roadmap:

1. prove the hybrid at normal precision
2. quantize the warm and cold tiers first
3. keep the hot exact-interaction path less aggressively quantized
4. attempt 1-bit or near-1-bit only after the utility surfaces are stable

If the goal is "frontier accuracy with much less RAM," mixed-precision
hot/warm/cold tiers are more plausible than forcing everything to the same
extreme bit width.

---

## Definition of done for the first serious hybrid

The first serious hybrid exists only when all of these are true:

* hot attention baseline is stable and accurate on the probe suite
* warm recurrence reduces cost or improves throughput without breaking probe
  behavior
* cold tree memory improves long-range retrieval tasks over the hot-only
  baseline
* exact leaf read improves copy or pinpoint retrieval when enabled
* oracle retrieval improves output materially
* retrieval cross-attention beats routing-only logging
* single-root and multi-root comparisons are explicit
* benchmark and ledger outputs capture the full learned ablation matrix
* quantization experiments preserve the same evaluation surfaces

If the memory system is active but does not improve the hot token predictor,
the work is still infrastructure, not a successful hybrid.

---

## Anti-goals

Do not do any of these to rescue weak early results:

* reintroduce giant dense global attention
* add side-memory bank policy before the hybrid proves useful
* pile on more roots before single-root is understood
* hide retrieval in opaque fusion heuristics again
* change benchmark or probe tasks to make the result look better

If the minimal hybrid fails, it should fail clearly.

---

## Bottom line

The v2 line taught us something important:

**recurrence and tree memory are not enough to replace transformer-grade exact
interaction**

But it also left us valuable parts:

* searchable long-range memory
* exact-read storage contracts
* causal auditing
* ablation and benchmark discipline

The next attempt should use those pieces in a humbler role:

* transformer-like hot path for exact interaction
* recurrent warm path for cheap continuous compute
* tree-memory cold path for long-range retrieval

That architecture still aims at the original goals.
It just stops asking the wrong subsystem to do the impossible.
