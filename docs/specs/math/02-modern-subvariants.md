# Modern Sub-Variants

This note extends the proof program to modern architectural variants that sit
between standard dense transformers and our own contender lines.

The purpose is twofold:

- make the reduction relations explicit
- isolate what each variant changes mathematically

## 2A. Sparse Mixture Of Experts

Consider an MoE feed-forward block with:

- expert set `{E_1, ..., E_E}`
- routing scores `g_t`
- top-`k` expert activation

The block output can be written as:

```text
MoE(x_t) = sum_{i in Active_t} pi_{t,i} E_i(x_t)
```

where `|Active_t| <= k`.

### Proposition 2A.1: Dense Reduction

A sparse-MoE layer reduces to a dense feed-forward layer under a degenerate
expert-sharing and routing choice.

Proof sketch:

1. Set every expert function equal to the same dense function `F`.
2. Force the router to activate exactly one expert with weight `1`.
3. Then:

```text
MoE(x_t) = F(x_t)
```

So dense FFN is a special case of MoE.

### Proposition 2A.2: Expert Budget Invariant

If routing activates at most `k` experts per token, per-token expert compute is
bounded by `k` expert evaluations.

Proof sketch:

- by construction the router emits no more than `k` active experts
- therefore the expert branch cannot evaluate more than `k` experts for that
  token

This is the core formal benefit of sparse expert routing.

### Proposition 2A.3: Control-Plane Separation

Sparse MoE changes conditional compute allocation without changing the causal
prefix contract of the surrounding decoder.

Proof sketch:

- routing depends on the current causal token state
- expert evaluations consume only that current causal state
- no future-token dependence is introduced by expert selection alone

### Conjecture 2A.4: Better Parameter-Efficiency

Whether sparse expert specialization improves parameter-efficiency is empirical.
The theorem surface stops at the budget and reduction facts.

## 2B. Linear-Attention Families

For a linear-attention family with feature map `phi`, write:

```text
S_t = sum_{u <= t} phi(k_u) v_u^T
z_t = sum_{u <= t} phi(k_u)
output_t = (phi(q_t)^T S_t) / (phi(q_t)^T z_t)
```

### Theorem 2B.1: Associative Recurrent Form

If the attention kernel factors through an associative feature map `phi`, then
prefix processing admits an equivalent recurrent update with linear-time
sequence evolution.

Proof sketch:

1. `S_t` and `z_t` can be updated from `S_{t-1}` and `z_{t-1}` by adding the
   current token contribution only.
2. No explicit revisit of all prior tokens is needed once the prefix summaries
   are maintained.
3. Therefore the model can implement the visible-prefix read through a recurrent
   state update over `(S_t, z_t)`.

This is the central mathematical reason linear-attention families admit
recurrent implementations.

### Proposition 2B.2: Softmax Non-Equivalence Boundary

A fixed finite-dimensional linear feature-map family does not in general
represent arbitrary softmax attention exactly over all sequences and parameter
settings.

Proof sketch:

1. Linear-attention kernels factor through a fixed finite-dimensional feature
   representation.
2. Exact softmax attention defines a richer family of normalized similarity
   kernels over arbitrary inputs.
3. There exist softmax attention score patterns whose induced normalized kernel
   cannot be represented exactly by the fixed finite-dimensional factorization.

So linear attention trades exact expressive parity for a cheaper recurrent form.

### Proposition 2B.3: Decode Complexity Bound

If the model maintains recurrent prefix summaries of bounded size, incremental
decode cost is bounded by the recurrent-state update rather than a full
visible-cache scan.

At a high level:

```text
decode cost per token ~ O(state update)
```

instead of:

```text
decode cost per token ~ O(visible context scan)
```

## 2C. Hybrid Recurrent-Attention Families

This family includes:

- Jamba-style interleaved attention + SSM stacks
- Nemotron-H-style mostly-SSM stacks with occasional attention
- Kimi-style recurrent-memory plus periodic exact-attention stacks

Let the stack contain:

- `L_A` exact-attention layers
- `L_S` recurrent or SSM sequence-mixer layers

### Theorem 2C.1: Hybrid Causal Composition

An interleaved stack of causal exact-attention and causal recurrent or SSM
blocks remains causal.

Proof sketch:

1. each `A` block is causal by masked attention
2. each `S` block is causal by recurrent left-to-right update
3. causality is preserved under composition

So any `A/S` hybrid stack with causal blocks remains a valid autoregressive
predictor.

### Proposition 2C.2: Attention-Fraction Bound

If only `L_A` of `L = L_A + L_S` layers perform exact attention, then:

- exact-attention compute is bounded by those `L_A` layers
- exact-attention state or cache obligations are bounded by those `L_A` layers

This does not make the whole model linear automatically.
It does make the expensive exact-interaction fraction explicit.

### Proposition 2C.3: Reduction Edge Cases

The hybrid stack reduces to:

- attention-only when `L_S = 0`
- recurrent-only when `L_A = 0`

Proof sketch:

- setting one block family count to zero removes that family from the stack
- the remaining stack is exactly the edge-case class

### Proposition 2C.4: Hybrid Slot Interpretation

A recurrent or SSM block in a hybrid stack fills a different mathematical role
than an attention block:

- `A` provides exact token-to-token interaction
- `S` provides cheaper continuous sequence processing

This is not only a performance story.
It is a decomposition of model function by primitive type.

## 2D. Kimi-Style Periodic Exactness

Treat `Kimi Linear` as a hybrid recurrent-attention family with:

- `KDA`
  - recurrent-memory sequence-mixing layers
- `MLA`
  - exact-attention refresh layers

In the simplest schedule description:

```text
KDA-KDA-KDA-MLA
```

repeated through depth.

### Proposition 2D.1: Periodic Exactness

In a fixed `3:1` KDA-to-MLA schedule, exact token interaction is guaranteed at
regular depth intervals.

Proof sketch:

1. every fourth layer is an exact-attention layer
2. therefore any representation passing through depth encounters a periodic
   exact interaction surface
3. KDA layers carry cheaper recurrent memory between those exact refresh points

This is one clean mathematical reason hybrid schedules can retain high-value
precision without paying dense exact attention at every layer.

### Proposition 2D.2: Cache Pressure Reduction

If only MLA layers require full exact-interaction visible-token state, then
cache pressure scales with the MLA fraction rather than the full layer count.

Proof sketch:

1. KDA layers use recurrent memory instead of full exact-attention visible
   state
2. MLA layers retain the expensive exact-attention obligation
3. therefore total high-fidelity cache pressure is controlled by the exact
   layer density

### Proposition 2D.3: Reduction Edge Cases

The Kimi-style hybrid reduces to:

- a pure recurrent-memory model when MLA density is zero
- a pure exact-attention model when KDA density is zero

This makes it a genuine bridge family rather than a completely separate class.

## Why This Track Matters For Us

This note is the clean comparison surface for:

- standard attention-only baselines
- sparse expert scaling
- linear-attention compression
- hybrid recurrent-attention reference families
- our own `A + P2` contender

The key mathematical pattern is:

- attention keeps exact interaction
- recurrence or SSM layers cheapen the background sequence path
- hybrid schedules make the exact-interaction fraction an explicit design knob

That is why `Kimi Linear` belongs in this track rather than being treated as
"just another linear attention model."
