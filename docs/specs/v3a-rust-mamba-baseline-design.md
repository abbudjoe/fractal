# v3A Rust Mamba Baseline Design

## Purpose

This note defines what “faithful enough Rust Mamba-style baseline” means for
Path 1.

It is a design contract, not a final performance claim.

Its job is to prevent the reference lane from drifting between:

* a toy proxy
* an under-specified “Mamba-like” block
* and a benchmark we later ask our primitives to beat

If Path 1 is going to be scientifically clean, the reference block must be
defined before we implement it.

---

## Baseline Role

The Rust Mamba baseline is the **reference SSM lane** inside the Path 1 hybrid
matrix.

It is not:

* a memory/index sidecar
* a retrieval system
* a MoE block
* a primitive contender

It is a **predictive-core sequence mixer** that fills the same architectural
slot that Mamba-family layers fill in hybrid attention models such as Jamba and
Nemotron-H.

The first target is:

* interleaved `A-S-A-S-A-S-A-S`
* where `A` is local exact causal attention
* and `S` is the Rust Mamba baseline block

---

## Required In The First Faithful Pass

The first faithful Rust pass must include all of the following:

1. **separate latent state and emitted output**
   The block may not collapse “new state” and “output token representation”
   into the same object.

2. **input-conditioned state update**
   The state transition must be selective and token-dependent.

3. **state transform before blending**
   The previous state must be transformed before interpolation with the current
   candidate.

4. **explicit scan contract**
   The block must define a clear recurrent step over sequence positions, not an
   opaque whole-sequence mixer.

5. **typed block config and shape validation**
   Width, state shape, head-compatible dimensions, and scan assumptions must be
   enforced in code.

6. **shared Path 1 integration surface**
   The block must plug into:
   * `fractal-core/src/hybrid_attention/`
   * `fractal-eval-private/src/hybrid_attention_training.rs`
   * `src/bin/v3a-hybrid-attention-matrix.rs`

7. **seeded reproducibility**
   Initialization and smoke training must run reproducibly under the shared
   Path 1 seed contract.

---

## Explicitly Deferred In The First Faithful Pass

These are intentionally deferred unless they become necessary for correctness:

* fused custom kernels
* CUDA-specific optimizations
* paper-faithful performance tuning
* multi-stream or multi-branch scan variants
* MoE
* external memory/index integration
* contender-specific ablations

This first pass is about **correct architectural role**, not peak efficiency.

---

## Block Contract

The baseline block must expose one explicit recurrent contract:

```text
given:
  x_t     current token representation
  h_{t-1} previous latent state

compute:
  transformed_state = T(h_{t-1}, x_t)
  candidate_state   = C(x_t)
  next_state        = U(transformed_state, candidate_state, x_t)
  output_t          = R(next_state, x_t)
```

Where:

* `T` is a learned state transform
* `C` is a learned candidate proposal from input
* `U` is a selective update/interpolation
* `R` is a distinct readout from state to emitted representation

The important non-negotiable property is:

* **`output_t` must not be defined as simply `next_state`**

That separation is one of the main lessons from the current proxy and from the
comparison against our simpler primitives.

---

## Parameter And Shape Contract

The first faithful pass must lock these structural rules:

* model width `d_model` is the emitted token representation width
* latent state width is explicit and validated
* if pairwise/rotary state transforms are used, width constraints must be
  checked in code
* readout projection dimensions must be explicit
* the block must be load-bearing inside the hybrid stack without any silent
  fallback to the proxy block

At the Path 1 phase-1 envelope, the block must remain compatible with:

* hidden width `128`
* head count `4`
* total layer count `8`
* local window `256`

Those values may change in a later proving round, but they are fixed for the
baseline graduation round.

---

## Initialization Contract

The first faithful pass must define initialization explicitly.

Required:

* projection initialization policy is named and consistent
* seeded runs produce reproducible model initialization
* the block does not rely on backend-specific random behavior

This is important because Path 1 now depends on:

* same-seed reruns
* multi-seed stability checks
* freezing one exact baseline artifact later

---

## Scan Contract

The scan behavior must be explicit.

Required:

* per-token recurrent step
* deterministic left-to-right scan
* same state update semantics on CPU and Metal
* test coverage for scan consistency across short sequences

Allowed in the first pass:

* straightforward sequential scan

Deferred:

* optimized parallel scan variants
* fused backend-specific scan kernels

This keeps the implementation honest before optimization begins.

---

## Backend Contract

The first faithful pass must be designed for this repo’s actual environment.

Required bring-up targets:

1. **CPU**
   Must compile, run, and pass the Path 1 smoke surface.

2. **Metal**
   Must not rely on zero-length tensor placeholders or backend-specific undefined
   behavior.

Not required for the first pass:

* CUDA-specific kernel parity with official Mamba implementations

That means the first baseline is a **Rust-native architectural baseline**, not a
claim of matching official kernel-level efficiency yet.

---

## What “Faithful Enough” Means

For Path 1, “faithful enough” means:

* the block clearly occupies the Mamba-family role in a hybrid stack
* the state/update/readout contract is materially richer than the current proxy
  and materially different from our primitive contender
* the block is stable, reproducible, and benchmarkable in this environment

It does **not** mean:

* exact paper/kernel parity on day one
* direct claim of matching official PyTorch/CUDA speed

Those are important later, but they are not the gating definition for the first
Rust baseline.

---

## Phase 1 Exit Criteria

Phase 1 design is done only when all of the following are true:

* a reader can identify the exact recurrent step contract
* required features vs deferred features are explicit
* backend expectations are explicit
* the block’s role inside the Path 1 hybrid stack is unambiguous
* the design is concrete enough to implement without inventing new rules during
  coding
