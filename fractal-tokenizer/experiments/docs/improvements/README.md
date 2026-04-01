# Tokenizer Improvement Catalog

This directory tracks proposed tokenizer improvements as standalone experiment notes.

Each note should answer:

- what the idea is
- how it works
- why it is a good candidate
- whether it has been tried
- what happened when it was tried

Status legend:

- `Tried`: implemented and observed at least once
- `Proposed`: documented but not yet implemented
- `Deferred`: documented but intentionally not next in line

## Ranked Shortlist

- See [ranked-shortlist.md](./ranked-shortlist.md) for the current ordered trial list, including rationale, expected upside, and likely failure modes.

## Current Catalog

| Proposal | Status | Why it matters | Latest outcome |
|---|---|---|---|
| [Adaptive Frontier: Greedy Known](./adaptive-frontier-greedy-known.md) | Tried | Coarsest possible known motif emission | Lossless but collapses full-vocab inputs to one root token |
| [Adaptive Frontier: Finest Known](./adaptive-frontier-finest-known.md) | Tried | Finest deterministic non-overlapping known frontier | Lossless useful frontier, but currently resolves to a uniform 32-token leaf frontier |
| [Adaptive Frontier: Reuse-Aware](./adaptive-frontier-reuse-aware.md) | Tried | Keep repeated structure coarse where reuse is valuable | Wins on stress (9 vs 32 tokens), ties on mixed-domain (32 vs 32), exact round-trip |
| [Adaptive Frontier: Novelty-Aware](./adaptive-frontier-novelty-aware.md) | Tried | Push novel spans finer without flattening repeated structure | Wins on stress (3 vs 32), ties on mixed-domain (32 vs 32), exact round-trip |
| [Adaptive Frontier: State-Aware](./adaptive-frontier-state-aware.md) | Tried | Use recursive state coherence to decide stop vs recurse | Lossless but regressed to single-root frontier on stress and mixed-domain |
| [Adaptive Frontier: Budgeted](./adaptive-frontier-budgeted.md) | Tried | Target useful sequence length under model constraints | Stress: 16 tokens vs NoveltyAware's 3; mixed-domain: 16 vs 32; exact round-trip |
| [Adaptive Frontier: Span-Length-Aware](./adaptive-frontier-span-length-aware.md) | Tried | Use span size as a simple stop/recurse prior | Stress: 8 tokens vs NoveltyAware's 3; mixed-domain: 24 vs 32; exact round-trip |
| [Adaptive Frontier: Hybrid Structural](./adaptive-frontier-hybrid-structural.md) | Tried | Combine reuse, novelty, and state into one stop policy | Lossless but regressed to a single-root frontier on stress and mixed-domain |
| [Chunking / Model Packaging](./chunking-model-packaging.md) | Tried | Package the chosen frontier into model-sized windows | NoveltyAware packaged losslessly into 1 stress chunk and 4 mixed-domain chunks with exact reconstruction |

## Current Read

At the moment:

- `GreedyKnown` proved the stable contract but was too coarse
- `FinestKnown` fixed root collapse and preserved exact round-trip
- `ReuseAware` beat `FinestKnown` on stress (9 vs 32 tokens), tied on mixed-domain (32 vs 32), and kept exact round-trip with zero unknown/byte fallback
- `NoveltyAware` now beats `FinestKnown` even harder on stress (3 vs 32 tokens), ties on mixed-domain (32 vs 32), and preserves exact round-trip with zero unknown/byte fallback
- `SpanLengthAware` was tried against `NoveltyAware`: it kept exact round-trip and zero fallback, but lost on stress (8 vs 3 tokens) while producing a finer frontier on mixed-domain (24 vs 32 tokens)
- `Budgeted` was tried against `NoveltyAware`: it kept exact round-trip and zero fallback, but lost on stress (16 vs 3 tokens) while producing a finer frontier on mixed-domain (16 vs 32 tokens)
- `HybridStructural` was tried against `NoveltyAware`: it kept exact round-trip and zero fallback, but collapsed to a single-root frontier on both stress and mixed-domain inputs
- `StateAware` was tried and preserved round-trip, but collapsed back to a coarse one-token frontier
- `Chunking / Model Packaging` was tried on the current frontier (`NoveltyAware`): 8-token packaging windows preserved exact reconstruction, with stress `3` frontier tokens -> `1` chunk and mixed-domain `32` frontier tokens -> `4` chunks, zero unknown/byte fallback
- `NoveltyAware` remains the leading adaptive-frontier candidate
- the frontier-policy search is no longer the active bottleneck
- the first honest held-out local bakeoff invalidated the earlier local-only `GREEN`
- the next clean step is a held-out-safe OOV contract, not another frontier policy or broader bakeoff coverage

Chunking remains important, and it now has a validated packaging layer on top of the stronger frontier policy.

The decision record for that shift is:

- [Held-Out OOV Decision](../held-out-oov-decision.md)

And the next two architectural specs are:

- [Compositional Motif Vocab Spec](../compositional-motif-vocab-spec.md)
- [Typed Lexical Fallback Spec](../typed-lexical-fallback-spec.md)
