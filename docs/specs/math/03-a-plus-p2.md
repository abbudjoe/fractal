# `A + P2`

This note captures the theorem surface for the `A + P2` architecture family.

`A + P2` is the Path 1 contender in the current hybrid-attention program:

```text
A-P2-A-P2-A-P2-A-P2
```

where:

- `A`
  - local exact-attention block
- `P2`
  - predictive-core sequence primitive

Important boundary:

The current repo-level `P2` contract is still incomplete in
[v3a-p2-primitive-contract.md](/Users/joseph/fractal/docs/specs/v3a-p2-primitive-contract.md).
So this note is a theorem surface for the expected `P2` class, not a claim that
the final implementation has already frozen every parameterization detail.

## Current Expected `P2` Contract

This note assumes the current expected direction:

- predictive-core sequence primitive only
- no memory/index sidecar behavior
- transformed state dynamics before blending
- emitted output distinct from latent state
- no routed retrieval or Path 2 control-plane behavior

Let:

- `x_t`
  - current residual-stream token representation entering a `P2` block
- `s_{t-1}`
  - prior latent recurrent state
- `s_t`
  - next latent recurrent state
- `o_t`
  - emitted output returned to the residual stream

Write the `P2` step in abstract form as:

```text
u_t = T_theta(s_{t-1}, x_t)
c_t = C_theta(x_t)
s_t = U_theta(u_t, c_t, x_t)
o_t = R_theta(s_t, x_t)
```

where:

- `T_theta`
  - transformed carry path
- `C_theta`
  - candidate proposal from current input
- `U_theta`
  - selective update
- `R_theta`
  - emitted readout

Non-negotiable structural property:

```text
o_t != s_t
```

as an architectural contract class, even if some degenerate parameter setting
could make them numerically equal in a special case.

## Definition 3.1: `P2` Step / Scan Consistency

A `P2` implementation is step/scan consistent if:

- repeated `step(...)` over a sequence
- and one `scan(...)` over the same sequence

produce the same emitted outputs and final latent state, up to deterministic
arithmetic tolerance.

This is a required control-plane property for Path 1.

## Theorem 3.2: `P2` Causal Update Correctness

If `T_theta`, `C_theta`, `U_theta`, and `R_theta` depend only on:

- `x_t`
- `s_{t-1}`

then the `P2` block is causal.

Proof sketch:

1. `T_theta` sees only the prior latent state and current input.
2. `C_theta` sees only the current input.
3. `U_theta` combines only those causal objects.
4. `R_theta` reads only the resulting causal state and current input.

So no future token can enter the block computation.

## Theorem 3.3: `A + P2` Hybrid Causality

If:

- every `A` block is causal by masked local attention
- every `P2` block is causal by Theorem 3.2

then the interleaved `A + P2` stack is causal.

Proof sketch:

- apply causal composition layer by layer through the stack

This is the direct Path 1 validity condition for the contender line.

## Proposition 3.4: `P1`-Style Reduction

Under explicit restrictions, the `P2` family reduces to a simpler contractive
`P1`-style update family.

One sufficient reduction pattern is:

- choose `T_theta` to be the identity or a fixed affine transform
- choose `R_theta` to collapse to a direct state readout
- choose `U_theta` to reduce to a gated blend between transformed carry and
  fresh candidate

Then the `P2` block becomes a simpler contractive recurrent update class with a
single active latent trajectory.

This is a structural reduction, not a statement about identical parameter
counts.

## Proposition 3.5: Strict Structural Generalization Over `P1`

The expected `P2` class strictly structurally generalizes the simpler `P1`
family, provided the final `P2` contract keeps both:

- transformed carry dynamics before blending
- emitted output distinct from latent state

Proof sketch:

1. By Proposition 3.4, `P2` reduces to a `P1`-style family under explicit
   restrictions.
2. A `P1` contract that only blends state and treats latent memory as emitted
   output cannot represent the full two-object `latent state + emitted output`
   contract without changing the contract class.
3. Therefore the expected `P2` family strictly enlarges the structural class.

This proposition is conditional on the final implementation preserving the
current contract direction.

## Proposition 3.6: Matched Slot Complexity

At the high-level hybrid-stack slot, `A + P2` occupies the same asymptotic role
as `A + M`:

- `A` pays exact local-attention cost
- `P2` pays recurrent or sequence-mixer cost

If the local attention window is `w`, sequence length is `T`, and the sequence
mixer update is linear in prefix length through recurrent state rather than
context scan, then:

```text
A + P2 train-time cost ~ attention term + recurrent term
A + P2 decode-time cost ~ local attention term + recurrent step term
```

The exact constants depend on:

- model width
- readout width
- transform width
- gating structure

The important formal point is:

- `P2` is a slot contender for the sequence-mixer role
- not a different architecture family that secretly changes the Path 1 problem

## Proposition 3.7: Predictive-Core Discipline

If `P2` contains no:

- routed retrieval
- external memory write path
- sealed-memory sidecar
- direct memory-to-logit fusion

then `A + P2` remains inside the predictive-core discipline of Path 1.

Proof sketch:

- the only state transition is inside the sequence primitive and the surrounding
  residual stack
- no extra memory subsystem is introduced

This proposition matters because the current Path 1 comparison must stay clean.

## Conjecture 3.8: Better Efficiency-Quality Tradeoff

Whether `A + P2`:

- matches or beats `A`
- matches or beats `A + M`
- does so at a better memory / throughput point

is empirical.

This note does not and should not try to prove that.

## What This Note Gives Us

This note gives the clean theorem surface for `A + P2`:

- causal validity
- hybrid-stack validity
- reduction back to simpler recurrent families
- structural reason for expecting greater expressiveness than `P1`
- matched-slot interpretation against `A + M`

That is the most we should prove before the final `P2` implementation settles.
