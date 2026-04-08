# v3a Hybrid Attention Sketch

This note captures the current `v3a` hybrid-attention structure as defined by
the explicit Path 1 architecture documents on `codex/fractal-v3.0`.

This is no longer an inferred continuation of the older tree-first `v2` line.
`v3a` is now a predictive-core comparison surface.

## Working Thesis

`v3a` is a hybrid architecture because it separates two roles inside the token
predictor:

- local exact-attention blocks handle high-value token-to-token interaction
- sequence-mixing blocks handle cheaper continuous background processing

The immediate question is not whether tree memory can replace attention.
The immediate question is whether a recurrent or selective sequence mixer can
fill the non-attention slots in a modern hybrid stack under a matched budget.

## Fixed Path 1 Matrix

The phase-1 matrix is:

1. `A`
   - `8` local exact-attention layers
2. `A + M`
   - `A-S-A-S-A-S-A-S`
   - `S` is the Rust Mamba-style reference block
3. `A + P2`
   - `A-P2-A-P2-A-P2-A-P2`
   - `P2` is our contender primitive block

All three share:

- hidden width `128`
- head count `4`
- local attention window `256`
- the same training and evaluation budget

## Block Contract

Every sequence-mixing block in Path 1 must have:

- explicit latent recurrent state
- an input-conditioned state update
- an emitted token representation distinct from latent state
- a deterministic left-to-right scan contract

That is true for:

- the Rust Mamba reference lane
- the `P2` contender lane

## Architectural Contract

- exact attention remains in the predictive path
- the sequence-mixing lane is compared under the same attention budget
- tree retrieval is not part of the Path 1 baseline matrix
- sealed-leaf sidecars and routed remote gathers are excluded from the proving baseline
- direct memory-to-logit fusion is excluded

## Relationship To Broader v3

The broader `v3` architecture still treats tree memory as a possible cold-memory
sidecar and retrieval system.

But `v3a` Path 1 deliberately isolates the predictive-core question first:

- can a hybrid attention stack be validated in Rust
- and can our primitive compete for the sequence-mixer slot fairly

## Source Documents

This sketch is derived primarily from:

- [v3a-hybrid-attention-plan.md](/Users/joseph/fractal/docs/specs/v3a-hybrid-attention-plan.md)
- [v3a-rust-mamba-baseline-design.md](/Users/joseph/fractal/docs/specs/v3a-rust-mamba-baseline-design.md)
- [v3a-p2-primitive-contract.md](/Users/joseph/fractal/docs/specs/v3a-p2-primitive-contract.md)
- [fractal-hybrid-v3.md](/Users/joseph/fractal/docs/specs/fractal-hybrid-v3.md)
- [v3a-hybrid-attention.html](/Users/joseph/fractal/docs/visualizations/v3a-hybrid-attention.html)
