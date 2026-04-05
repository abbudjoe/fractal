# v3A Baseline Freeze and P2 Checklist

This checklist starts the next disciplined Path 1 sequence:

1. freeze the Rust `A` vs `A + M` baseline with recorded evidence
2. define the improved primitive `P2` as a typed predictive-core contract
3. run the first `A + P2` comparison on the same tracked surface

Nothing in this checklist counts unless it is backed by:

* a structured ledger entry in [`/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl`](/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl)
* and a matching documentation update under `docs/specs/`

---

## Fixed Tracking Contract

- [ ] all Path 1 matrix runs that inform a decision use `src/bin/v3a-hybrid-attention-matrix.rs`
- [ ] those runs append to `docs/v3a-results-ledger.jsonl`
- [ ] each recorded run uses a stable `run_label`
- [ ] each phase decision has a matching documentation artifact
- [ ] no baseline or `P2` claim relies only on terminal output or chat memory

Required documentation surfaces:

- [ ] [`v3a-baseline-freeze-record.md`](./v3a-baseline-freeze-record.md)
- [ ] [`v3a-p2-primitive-contract.md`](./v3a-p2-primitive-contract.md)

---

## Phase 1: Freeze the Path 1 Baseline

Goal:

- [ ] freeze `A` vs `A + M` before `P2` becomes a live contender

Required runs:

- [ ] run at least one full matrix with `--variant all`
- [ ] run at least one second full matrix with a different seed
- [ ] use the same budget across `A`, `A + M`, and the historical `primitive-hybrid` lane
- [ ] append both runs to `docs/v3a-results-ledger.jsonl`

Required documentation:

- [ ] update [`v3a-baseline-freeze-record.md`](./v3a-baseline-freeze-record.md) with:
  - [ ] commit hash
  - [ ] exact commands
  - [ ] run labels
  - [ ] ledger timestamps or references
  - [ ] artifact/report paths
  - [ ] final freeze decision

Exit gate:

- [ ] the frozen baseline is documented as `A` vs `A + M`
- [ ] the freeze point is backed by ledger entries, not only prose

---

## Phase 2: Define P2

Goal:

- [ ] define the first improved primitive as `P2`

`P2` must stay inside Path 1 discipline:

- [ ] predictive-core sequence primitive only
- [ ] no Path 2 memory/index sidecar behavior
- [ ] no routed retrieval or external memory write path

Minimum architectural upgrades relative to `P1Contractive`:

- [ ] transformed state dynamics before blending
- [ ] explicit emitted output distinct from latent state
- [ ] internal memory and output representation are not identical by default

Required documentation:

- [ ] update [`v3a-p2-primitive-contract.md`](./v3a-p2-primitive-contract.md) with:
  - [ ] state update contract
  - [ ] output readout contract
  - [ ] typed config surface
  - [ ] non-goals
  - [ ] expected comparison against `A + M`

Exit gate:

- [ ] `P2` is defined well enough to implement without changing the benchmark definition midstream

---

## Phase 3: First P2 Proving Run

Goal:

- [ ] give `P2` a fair standalone shot before any composite `A + M + P2` work

Required runs:

- [ ] implement `A + P2`
- [ ] run at least one seeded matrix including:
  - [ ] `A`
  - [ ] `A + M`
  - [ ] `A + P2`
- [ ] append the run to `docs/v3a-results-ledger.jsonl`

Required documentation:

- [ ] extend [`v3a-p2-primitive-contract.md`](./v3a-p2-primitive-contract.md) with:
  - [ ] actual implemented config
  - [ ] test surface
  - [ ] first recorded result references
  - [ ] whether `P2` earns deeper ablations

Exit gate:

- [ ] `P2` has one honest result-bearing comparison against the frozen baseline
- [ ] no composite `A + M + P2` work begins before this gate clears

---

## Explicit Prohibitions

- [ ] do not start `A + M + P2` during this checklist
- [ ] do not import Path 2 retrieval ideas into `P2`
- [ ] do not redefine the frozen `A + M` baseline while `P2` is under test
- [ ] do not count unlogged runs as evidence
