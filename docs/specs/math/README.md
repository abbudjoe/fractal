# Math Proof Notes

This folder contains the first split documents from the shared mathematical
proof program.

The purpose is to keep the proof surface:

- explicit
- scoped to claims that are actually provable
- separated from empirical benchmark claims

## Files

- [`architecture-proof-notes.tex`](./architecture-proof-notes.tex)
  - paper-style LaTeX source collecting the current theorem surface in one
    document
- [`architecture-proof-audit.md`](./architecture-proof-audit.md)
  - theorem-by-theorem verification audit labeling each formal statement as
    definition-only, verified-sketch, needs-stronger-assumptions, or
    conjectural / empirical
- [`00-common-notation.md`](./00-common-notation.md)
  - shared causal-machine notation and proof-label discipline
- [`01-standard-decoder-llms.md`](./01-standard-decoder-llms.md)
  - causal decoder, KV-cache equivalence, copy capacity, and cost bounds
- [`02-modern-subvariants.md`](./02-modern-subvariants.md)
  - `MoE`, linear-attention families, hybrid recurrent-attention models, and
    `Kimi Linear`
- [`03-a-plus-p2.md`](./03-a-plus-p2.md)
  - provisional theorem surface for the current expected `A + P2` contender
- [`04-graph-of-experts.md`](./04-graph-of-experts.md)
  - bounded formalism for thought-channel / Graph-of-Experts search models

## Current Boundary

This folder now covers the full first-pass split:

1. standard decoder LLMs
2. modern sub-variants
3. `A + P2`
4. Graph-of-Experts / thought-channel models

Important caveat:

- [`03-a-plus-p2.md`](./03-a-plus-p2.md) is intentionally provisional because
  the repo-level `P2` contract is still not fully frozen
- [`04-graph-of-experts.md`](./04-graph-of-experts.md) is intentionally heavy
  on propositions and conjectures because the family remains speculative
