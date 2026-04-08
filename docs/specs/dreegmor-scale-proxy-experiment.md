# DREEGMOR Scale Proxy Experiment

## Status

Exploratory architecture lane only.

This surface exists to approximate the kind of expertized system we would plausibly build at scale more faithfully than the original whole-backbone DREEGMOR scaffold. It is not evidence for the main `v3a` / Path 1 proving line.

## Question

Does an internally expertized, shared-trunk miniature over the existing `A` backbone produce a more credible architecture probe than duplicating two whole `A` backbones and routing over their final outputs?

## Contract

- Base backbone: existing Path 1 `A` surface only
- Shared components:
  - token embedding
  - output head
  - all non-expertized attention blocks
- Expertized component:
  - exactly one internal attention block
  - two dense expert branches at that block
- Routing input:
  - hidden state at the expertized block boundary
- Routing modes:
  - one-shot hidden-state router
  - recurrent hidden-state router
- Channel count: fixed at `2`
- Execution:
  - dense for both experts at the expertized block
  - no sparse top-k yet
- Exclusions:
  - no `P2`
  - no external memory sidecar
  - no `A + M` scale-proxy lane yet
  - no multi-block expertization yet

## Why This Is Closer To Scale

Compared with whole-backbone DREEGMOR, this surface:

- does not duplicate embeddings and output heads
- does not duplicate the entire trunk
- expertizes an internal block instead of mixing whole-model logits
- routes on hidden states instead of token IDs alone

It is still a miniature, not a faithful sparse-MoE implementation, but it is a better scale proxy for internal expertization.

## Implemented

- `A` baseline on the shared byte-level smoke lane
- `DREEGMOR-ScaleProxy(A)` with one-shot hidden-state routing
- `DREEGMOR-ScaleProxy-Recurrent(A)` with the minimal dense GRU-style virtual-node router
- isolated output directories per variant
- quality, throughput, memory, and routing summaries in the report

Primary code:

- `/Users/joseph/fractal-worktrees/goe-a-am-experiments/fractal-core/src/hybrid_attention/scale_proxy.rs`
- `/Users/joseph/fractal-worktrees/goe-a-am-experiments/fractal-eval-private/src/hybrid_attention_scale_proxy.rs`
- `/Users/joseph/fractal-worktrees/goe-a-am-experiments/src/bin/dreegmor-scale-proxy-experiment.rs`

## Still Stubbed

- recurrent expert-feedback
- sparse/top-k routing
- uncertainty-gated extra rounds
- `A + M` scale-proxy surface
- more than one expertized block

## Metrics

Every run should be judged across:

- Quality: final eval loss / perplexity
- Cost: train tok/s, overall tok/s, process memory delta
- Explanation: channel usage, entropy, winner margin, round-to-round routing change

Use isolated single-variant runs for any memory or throughput comparison.

## How To Run

Smoke:

```bash
cargo run --bin dreegmor-scale-proxy-experiment -- --variant all --steps 1 --eval-batches 1 --output table
```

One-shot only:

```bash
cargo run --bin dreegmor-scale-proxy-experiment -- --variant scale-proxy-a --steps 16 --eval-batches 4
```

Recurrent only:

```bash
cargo run --bin dreegmor-scale-proxy-experiment -- --variant recurrent-scale-proxy-a --steps 16 --eval-batches 4
```

Isolated Metal run:

```bash
cargo run --bin dreegmor-scale-proxy-experiment -- --variant recurrent-scale-proxy-a --backend metal --steps 128 --eval-batches 16 --output-dir /absolute/path
```
