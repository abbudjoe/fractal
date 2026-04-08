# Mathematical Proof Program

This document defines the smallest clean proof program for the architecture
families we are comparing.

The goal is not to "prove benchmark wins."
The goal is to prove the parts that are actually provable:

- causal correctness
- equivalence to known execution paths
- reduction to simpler special cases
- strict architectural generalization where it is true
- asymptotic and control-plane cost bounds
- bounded-search behavior for search-native variants

Every statement in the eventual proof set should be marked as exactly one of:

- `Definition`
- `Lemma`
- `Theorem`
- `Proposition`
- `Proof sketch`
- `Conjecture`
- `Empirical claim`

If a claim depends on experiments, it is not a theorem.

## Scope

This proof program covers:

1. standard modern decoder LLMs
2. modern sub-variants:
   - `MoE`
   - linear-attention families
   - hybrid recurrent-attention families including `Kimi Linear`
3. our `A + P2` contender
4. the proposed Graph-of-Experts / thought-channel reasoning model

This document is the shared notation and theorem-outline layer.
It is intentionally narrower than a full paper draft.

## Common Abstract Machine

Use one causal machine for every family.

Given token sequence:

```text
x_1, x_2, ..., x_T
```

define:

- `h_t`
  - active latent state at step `t`
- `M_t`
  - memory object at step `t`
  - this may be a KV cache, recurrent state, expert-state bundle, or channel
    frontier
- `c_t`
  - control object at step `t`
  - this may be router state, gating scores, pruning decisions, or halt state
- `y_t`
  - emitted token representation used for prediction at step `t`

with generic update:

```text
(h_t, M_t, c_t) = U_theta(h_{t-1}, M_{t-1}, c_{t-1}, x_t)
y_t = R_theta(h_t, M_t, c_t, x_t)
p_theta(x_{t+1} | x_<=t) = Softmax(W y_t)
```

Architectures differ by:

- what `M_t` contains
- what `c_t` contains
- whether `U_theta` is attention-dominant, recurrent-dominant, hybrid, or
  search-native
- how much exact token interaction is available inside `U_theta`

## Shared Notation

- `T`
  - sequence length
- `d`
  - model width
- `w`
  - local exact-attention window
- `L`
  - total layer count
- `L_A`
  - exact-attention layer count
- `L_S`
  - sequence-mixer layer count
- `K`
  - number of thought channels or active latent hypotheses
- `E`
  - number of experts in an MoE layer
- `k`
  - number of active experts in sparse MoE
- `B`
  - bounded search budget
- `s_t`
  - recurrent latent state for one sequence-mixing block
- `q_t, k_t, v_t`
  - attention query, key, value objects

Use:

- `A`
  - exact-attention block
- `S`
  - reference sequence-mixing block
- `P2`
  - our contender sequence primitive
- `G`
  - graph-of-experts / thought-channel controller

## Shared Proof Obligations

Every architecture family should be evaluated against the same proof ladder.

### 1. Causality

Show that step `t` depends only on:

- `x_<=t`
- permitted state from steps `< t`

and never on future tokens.

### 2. Decode Equivalence

Where applicable, prove that incremental decode is equivalent to full-prefix
recomputation under the same weights and masks.

### 3. Reduction To Simpler Cases

Show how the architecture collapses to a simpler known class under explicit
parameter settings.

### 4. Strict Gain Or Separation

Where justified, prove that the new class strictly contains or can emulate the
older class.

### 5. Complexity And Resource Bounds

State train-time and decode-time bounds clearly.

### 6. Control-Plane Correctness

For routed, gated, or search-native models, prove that the control object
preserves the intended invariant:

- expert budget
- attention budget
- channel budget
- prune/halt budget

## Track 1: Standard Modern Decoder LLMs

### Core definitions

- masked causal self-attention
- residual stream
- feed-forward update
- KV-cache incremental decode

### Proof targets

#### Theorem 1.1: Causal Decoder Correctness

A masked decoder transformer defines a causal conditional distribution
`p(x_{t+1} | x_<=t)`.

#### Theorem 1.2: KV-Cache Decode Equivalence

Incremental decode with cached keys and values is equivalent to full-prefix
recomputation for the same prefix and weights.

#### Proposition 1.3: Exact Copy Capacity

Exact self-attention can implement token copying and pointer-style retrieval
over a visible context.

#### Proposition 1.4: Complexity Bound

Train-time dense attention cost is quadratic in visible sequence length.
Decode-time incremental cost is linear in visible cache length per attention
layer.

## Track 2: Modern Sub-Variants

## 2A. Sparse Mixture of Experts

### Proof targets

#### Proposition 2A.1: Dense Reduction

A sparse-MoE layer reduces to a dense feed-forward layer under a degenerate
gating and expert-sharing choice.

#### Proposition 2A.2: Expert Budget Invariant

If routing activates at most `k` experts per token, per-token expert compute is
bounded by `k` expert evaluations.

#### Conjecture 2A.3: Specialization Benefit

Sparse conditional activation may improve parameter-efficiency, but that is an
empirical claim rather than a theorem.

## 2B. Linear-Attention Families

### Proof targets

#### Theorem 2B.1: Associative Recurrent Form

For linear-attention kernels with associative feature maps, prefix processing
admits an equivalent recurrent update with linear-time sequence evolution.

#### Proposition 2B.2: Softmax Non-Equivalence Boundary

A fixed linear feature-map family does not in general exactly recover arbitrary
softmax attention over all sequences and parameter settings.

#### Proposition 2B.3: Decode Complexity Bound

Incremental decode cost is bounded by the recurrent-state update rather than a
full visible-context scan.

## 2C. Hybrid Recurrent-Attention Families

This track includes:

- Jamba-style interleaved attention + SSM hybrids
- Nemotron-H-style mostly-SSM hybrids
- `Kimi Linear`

### Proof targets

#### Theorem 2C.1: Hybrid Causal Composition

An interleaved stack of causal exact-attention and causal recurrent/SSM blocks
remains causal.

#### Proposition 2C.2: Attention-Fraction Bound

If only `L_A` of `L` layers use exact attention, then exact-attention compute
and exact-attention cache obligations are bounded by that fraction of the stack.

#### Proposition 2C.3: Reduction Edge Cases

The hybrid stack reduces to:

- attention-only when `L_S = 0`
- recurrent-only when `L_A = 0`

#### Proposition 2C.4: Periodic Exactness For Kimi-Style Schedules

For a fixed `3:1` KDA-to-MLA schedule, exact token interaction is guaranteed at
regular depth intervals while the recurrent path carries the cheaper continuous
background state between those refresh points.

#### Proposition 2C.5: Cache Pressure Reduction

If only the MLA or exact-attention layers require full high-fidelity visible
token state, total cache pressure scales with the exact-attention fraction
rather than the full layer count.

## Track 3: `A + P2`

`A + P2` is our Path 1 contender:

```text
A-P2-A-P2-A-P2-A-P2
```

### Required contract facts

- `P2` is predictive-core only
- `P2` is causal
- `P2` has transformed carry dynamics before blending
- `P2` emits an output distinct from latent state

### Proof targets

#### Theorem 3.1: `P2` Causal Scan Correctness

The `P2` step update and sequence scan define the same causal mapping under the
same weights and initial state.

#### Proposition 3.2: `P1` Reduction

Under explicit parameter restrictions, `P2` reduces to a simpler `P1`-style
contractive update family.

#### Proposition 3.3: Strict Structural Generalization Over `P1`

Because `P2` permits both transformed carry dynamics and a separate emitted
output map, it strictly enlarges the architectural contract class relative to
`P1`.

#### Proposition 3.4: Matched Slot Complexity

At fixed model width, `A + P2` occupies the same high-level hybrid slot as
`A + M`:

- exact-attention cost in `A`
- linear recurrent or sequence-mixer cost in `P2`

#### Conjecture 3.5: Better Efficiency-Quality Tradeoff

Whether `P2` matches or beats `A + M` is empirical.
The theorem surface should stop before that claim.

## Track 4: Graph-of-Experts / Thought-Channel Model

This track is the most speculative.

The model contains:

- a shared exact-attention trunk
- `K` active thought channels
- per-channel recurrent or SSM workspaces
- explicit compare, prune, merge, and halt control

### Proof targets

#### Proposition 4.1: Single-Channel Reduction

When `K = 1` and pruning is trivial, the model reduces to a single-stream
hybrid predictor.

#### Proposition 4.2: Bounded Channel Budget

If the controller keeps at most `K` live channels and at most `B` internal
refinement rounds, internal search compute is bounded by that explicit budget.

#### Lemma 4.3: Channel Identity Preservation Under Masked Separation

If channels maintain disjoint state slots and cross-channel interaction occurs
only through explicit compare/merge operators, branch identity can be preserved
across internal refinement steps.

#### Proposition 4.4: Bounded Latent Search Emulation

With expand, score, prune, and halt operators, the controller can emulate a
bounded beam-style latent search process inside a single model runtime.

#### Proposition 4.5: Token-Serialization Savings

Relative to an external GoT harness, a native channel model can avoid explicit
token serialization of intermediate branches and therefore has a lower token
overhead floor.

#### Proposition 4.6: Shared-Trunk Amortization

A shared exact-attention trunk amortizes common token-processing work across all
live channels rather than redoing the full backbone for each branch.

#### Conjecture 4.7: Better Search Efficiency Than External GoT

Whether that lower token-overhead floor turns into better end-to-end quality or
throughput is empirical.

## Active Split Docs

The first split docs now exist:

- [math/00-common-notation.md](/Users/joseph/fractal/docs/specs/math/00-common-notation.md)
- [math/01-standard-decoder-llms.md](/Users/joseph/fractal/docs/specs/math/01-standard-decoder-llms.md)
- [math/02-modern-subvariants.md](/Users/joseph/fractal/docs/specs/math/02-modern-subvariants.md)
- [math/03-a-plus-p2.md](/Users/joseph/fractal/docs/specs/math/03-a-plus-p2.md)
- [math/04-graph-of-experts.md](/Users/joseph/fractal/docs/specs/math/04-graph-of-experts.md)

These now cover Tracks 1 through 4, with the expected caveats:

- Track 3 is provisional while the final `P2` contract is still settling
- Track 4 remains partly speculative by design

## Planned Split

If this proof program expands, split it into:

- `docs/specs/math/00-common-notation.md`
- `docs/specs/math/01-standard-decoder-llms.md`
- `docs/specs/math/02-modern-subvariants.md`
- `docs/specs/math/03-a-plus-p2.md`
- `docs/specs/math/04-graph-of-experts.md`

Do not split earlier than needed.
The common notation should remain the single source of truth.

## Recommended Build Order

1. finalize the common notation and proof-label discipline
2. write Track 1 fully
3. write Track 2 fully, including `Kimi Linear`
4. write Track 3 only after the `P2` contract stops moving
5. keep Track 4 at theorem-statement and proof-sketch level until Path 1 is
   resolved

That keeps the proof program aligned with the repo's current experimental gates.
