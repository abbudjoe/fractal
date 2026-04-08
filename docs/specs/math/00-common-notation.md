# Common Notation And Proof Discipline

This note defines the shared mathematical surface for the proof program.

The design goal is one common causal machine that can describe:

- standard decoder transformers
- sparse expert models
- linear-attention models
- hybrid recurrent-attention stacks
- `A + P2`
- future thought-channel search models

## Proof Labels

Every formal statement should be tagged as exactly one of:

- `Definition`
- `Lemma`
- `Theorem`
- `Proposition`
- `Proof sketch`
- `Conjecture`
- `Empirical claim`

Rule:

- if a statement needs benchmark evidence, it is not a theorem

## Base Sequence Objects

Let the input token sequence be:

```text
x_1, x_2, ..., x_T
```

At step `t`, define:

- `h_t`
  - active latent state
- `M_t`
  - memory object
  - examples:
    - KV cache
    - recurrent state
    - expert-state bundle
    - sealed-memory index
    - thought-channel frontier
- `c_t`
  - control object
  - examples:
    - router scores
    - expert-gating decisions
    - prune / halt state
- `y_t`
  - emitted token representation used by the output head

The common causal machine is:

```text
(h_t, M_t, c_t) = U_theta(h_{t-1}, M_{t-1}, c_{t-1}, x_t)
y_t = R_theta(h_t, M_t, c_t, x_t)
p_theta(x_{t+1} | x_<=t) = Softmax(W y_t)
```

Architectures differ by:

- what `M_t` stores
- what `c_t` controls
- how much exact token interaction `U_theta` performs
- whether the dominant substrate is attention, recurrence, a hybrid stack, or
  a bounded search controller

## Shared Symbols

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
- `E`
  - number of experts in an MoE layer
- `k`
  - number of active experts in a sparse-MoE layer
- `K`
  - number of active thought channels or live latent hypotheses
- `B`
  - bounded internal search budget
- `q_t, k_t, v_t`
  - attention query, key, value objects
- `s_t`
  - recurrent latent state in one sequence-mixing block

Architecture shorthands:

- `A`
  - exact-attention block
- `S`
  - reference sequence-mixing block
- `P2`
  - our contender sequence primitive
- `G`
  - Graph-of-Experts / thought-channel controller

## Causality

### Definition 0.1: Causal Update

An update rule is causal if step `t` depends only on:

- `x_<=t`
- state produced from steps `< t`

and never on any future token `x_u` with `u > t`.

### Lemma 0.2: Causal Composition

If `f` and `g` are causal update operators over the same prefix order, then the
composition `g(f(...))` is also causal.

Proof sketch:

- `f` depends only on allowed prefix information
- `g` only sees the output of `f` and the same allowed prefix information
- therefore no future token can enter through composition

This lemma is used repeatedly for stacked architectures.

## Decode Equivalence

### Definition 0.3: Incremental-Decode Equivalence

An incremental decode procedure is equivalent to full-prefix recomputation if,
for the same weights, masks, and prefix:

```text
y_t^incremental = y_t^full
```

up to deterministic arithmetic tolerance.

This matters because many practical systems replace full recomputation with:

- KV caching
- recurrent summary updates
- linear-attention accumulators

## Reductions

### Definition 0.4: Reduction To A Simpler Class

Architecture family `F` reduces to family `G` if there exists an explicit
parameter or control setting under which every member of the reduced subclass of
`F` behaves as a member of `G`.

Examples:

- sparse MoE reducing to dense FFN
- hybrid stacks reducing to attention-only when `L_S = 0`
- `P2` reducing to a `P1`-style contractive update under constrained settings

## Strict Structural Generalization

### Definition 0.5: Strict Structural Generalization

Family `F` strictly structurally generalizes family `G` if:

1. `F` reduces to `G` under an explicit restriction
2. there exist members of `F` that cannot be represented inside `G` without
   changing the declared contract class

This is stronger than saying "`F` is bigger" informally.
It is weaker than proving universal empirical superiority.

## Complexity

We use:

- sequence-length cost for train-time prefix processing
- per-step decode cost for incremental generation
- memory / cache obligations when they are architecture-defining

### Definition 0.6: Budget Invariant

A control plane satisfies a budget invariant if it preserves a declared bound on
some controlled resource at every step, for example:

- at most `k` active experts
- at most `K` live channels
- at most `B` internal refinement rounds
- at most `L_A` attention-heavy layers

## What This Proof Program Will Not Claim

These notes do not try to prove:

- benchmark superiority
- real-world wall-clock wins on every backend
- training stability in general
- emergent reasoning quality

Those remain empirical unless a narrower formal claim is stated.
