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

- [x] all Path 1 matrix runs that inform a decision use `src/bin/v3a-hybrid-attention-matrix.rs`
- [x] those runs append to `docs/v3a-results-ledger.jsonl`
- [x] each recorded run uses a stable `run_label`
- [x] each recorded run now carries an explicit execution backend (`cpu` or `metal`)
- [x] each phase decision has a matching documentation artifact
- [x] no baseline or `P2` claim relies only on terminal output or chat memory
- [x] the default `v3a` benchmark surface is now the frozen FineWeb stage0
  canary, not mutable repo docs
- [x] the CUDA-faithful small benchmark surface is now a named contract:
  `cuda-faithful-small-v1`
  - [x] Rust runner support exists via `--benchmark-profile`
  - [x] Python native Mamba runner records the same benchmark name
  - [x] RunPod helper scripts can select it without raw path/manual budget churn
- [ ] contender-freeze decisions must be replayed on the frozen FineWeb canary
  surface before they are treated as final
  - [x] first longer-budget replay on frozen FineWeb canary is complete for
    seeds `42` and `43`
  - [ ] contender-freeze decision remains open because the frozen FineWeb
    surface did not produce a stable `P2.3 > A + M` result

Required documentation surfaces:

- [x] [`v3a-baseline-freeze-record.md`](./v3a-baseline-freeze-record.md)
- [x] [`v3a-p2-primitive-contract.md`](./v3a-p2-primitive-contract.md)
- [x] [`v3a-p2-interface-ablation-plan.md`](./v3a-p2-interface-ablation-plan.md)
- [x] [`v3a-p2-primitive-quality-plan.md`](./v3a-p2-primitive-quality-plan.md)

---

## Phase 1: Freeze the Path 1 Baseline

Goal:

- [x] freeze `A` vs `A + M` before `P2` becomes a live contender

Required runs:

- [x] run at least one full matrix with `--variant all`
- [x] run at least one second full matrix with a different seed
- [x] use the same budget across `A`, `A + M`, and the historical `primitive-hybrid` lane
- [x] append both runs to `docs/v3a-results-ledger.jsonl`

Required documentation:

- [x] update [`v3a-baseline-freeze-record.md`](./v3a-baseline-freeze-record.md) with:
  - [x] commit hash
  - [x] exact commands
  - [x] run labels
  - [x] ledger timestamps or references
  - [x] artifact/report paths
  - [x] final freeze decision

Exit gate:

- [x] the frozen baseline is documented as `A` vs `A + M`
- [x] the freeze point is backed by ledger entries, not only prose

---

## Phase 2: Define P2

Goal:

- [x] define the first improved primitive as `P2`

`P2` must stay inside Path 1 discipline:

- [x] predictive-core sequence primitive only
- [x] no Path 2 memory/index sidecar behavior
- [x] no routed retrieval or external memory write path

Minimum architectural upgrades relative to `P1Contractive`:

- [x] transformed state dynamics before blending
- [x] explicit emitted output distinct from latent state
- [x] internal memory and output representation are not identical by default

Required documentation:

- [x] update [`v3a-p2-primitive-contract.md`](./v3a-p2-primitive-contract.md) with:
  - [x] state update contract
  - [x] output readout contract
  - [x] typed config surface
  - [x] non-goals
  - [x] expected comparison against `A + M`

Exit gate:

- [x] `P2` is defined well enough to implement without changing the benchmark definition midstream

---

## Phase 3: First P2 Proving Run

Goal:

- [x] give `P2` a fair standalone shot before any composite `A + M + P2` work

Required runs:

- [x] implement `A + P2`
- [x] run at least one seeded matrix including:
  - [x] `A`
  - [x] `A + M`
  - [x] `A + P2`
- [x] append the run to `docs/v3a-results-ledger.jsonl`

Required documentation:

- [x] extend [`v3a-p2-primitive-contract.md`](./v3a-p2-primitive-contract.md) with:
  - [x] actual implemented config
  - [x] test surface
  - [x] first recorded result references
  - [x] whether `P2` earns deeper ablations

Exit gate:

- [x] `P2` has one honest result-bearing comparison against the frozen baseline
- [x] no composite `A + M + P2` work begins before this gate clears

Recorded outcome:

- [x] `P2` has real positive signal and a credible efficiency story
- [x] `P2` has **not** yet beaten the frozen `A + M` baseline on the best tracked quality run
- [x] the next disciplined target is interface quality before primitive-internal redesign

---

## Phase 4: Interface Quality Ablations

Goal:

- [x] test whether the bottleneck is the `P2` <-> attention wrapper interface rather than the primitive core itself

Discipline:

- [x] hold the inner `P2RotaryReadout` primitive fixed
- [x] change only one interface contract at a time
- [x] keep `A` and frozen `A + M` unchanged
- [x] log every decision-bearing run to `docs/v3a-results-ledger.jsonl`

Required documentation:

- [x] define the ablation matrix and gating rules in [`v3a-p2-interface-ablation-plan.md`](./v3a-p2-interface-ablation-plan.md)
- [x] extend [`v3a-p2-primitive-contract.md`](./v3a-p2-primitive-contract.md) with the chosen interface incumbent and supporting evidence

Required ablation families:

- [x] norm placement around the primitive block
- [x] residual injection form:
  - [x] plain residual add
  - [x] scaled residual
  - [x] gated residual
- [x] readout handoff:
  - [x] direct emitted output
  - [x] projected emitted output
  - [x] renormalized emitted output
- [x] wrapper symmetry against the Rust Mamba-3 lane where meaningful

Exit gate:

- [x] no challenger materially beat the incumbent wrapper on the logged short tracked surface
- [x] the incumbent wrapper remains:
  - [x] `plain`
  - [x] `direct`
  - [x] `pre-norm-only`
  - [x] `standard`
- [x] `P2` still preserves a meaningful efficiency advantage over `A + M`
- [x] interface quality is no longer the leading ambiguity

---

## Phase 5: Primitive Quality Ablations

Goal:

- [x] improve `P2` itself only after the best currently known interface is frozen

Blocked until:

- [x] Phase 4 incumbent wrapper is documented
- [x] the interface question is no longer the leading ambiguity

Allowed primitive-quality targets:

- [x] stronger transform stage
- [x] richer update gate
- [x] stronger readout head
- [x] state width / projection width changes
- [x] ordered primitive-quality ladder defined in [`v3a-p2-primitive-quality-plan.md`](./v3a-p2-primitive-quality-plan.md)
- [x] `P2.1` defined as wider latent state only under a fixed wrapper

Prohibited during Phase 5:

- [x] changing the wrapper and the primitive internals in the same ablation
- [x] importing Path 2 memory/index behavior
- [x] composite `A + M + P2` work before standalone `A + P2` clears its next gate

Next active step:

- [x] implement `P2.1`
- [x] run the first tracked `A / A + M / A + P2.1` matrix
- [x] record that wider latent state alone is insufficient on the tracked short surface
- [x] implement the width/readout `2 x 2` primitive-family sweep:
  - [x] `P2.0` base-width direct-state
  - [x] `P2` base-width explicit-readout
  - [x] `P2.1` wide direct-state
  - [x] `P2.2` wide explicit-readout
- [x] run the short tracked width/readout sweep and record the winner candidates
- [x] record that the original `P2` corner remains incumbent and that width expansion should be deprioritized

Next active step:

- [x] define the `P2.3` state-dynamics hypothesis with the incumbent `P2` wrapper and width/readout settings held fixed
- [x] implement `P2.3` and run the first tracked short comparison against frozen `A`, frozen `A + M`, and incumbent `P2`
- [x] record that the first state-dynamics contender was a negative result and did not displace incumbent `P2`
- [x] add a resumable contender-only sweep surface over the full current `P2` family grid with frozen `A` and frozen `A + M`

Next active step:

- [x] launch and complete the full `270`-configuration contender sweep
- [x] record that the full sweep surfaced better combined contenders than the prior incumbent

Next active step:

- [ ] replay the short-listed sweep winners on:
  - [x] a second seed
  - [x] and a longer-budget tracked surface
- [ ] freeze a new contender short-list from those confirmation runs
  - blocked by the mixed seed-43 longer-budget result, where `P2.3` beat frozen
    `A + M` again but frozen `A` unexpectedly won overall

Recorded longer-budget confirmation:

- [x] frozen `A + M`
- [x] incumbent `P2`
- [x] `P2.3 + gated + projected-norm + residual-renorm + standard`
- [x] `P2.0 + scaled + projected + pre-norm-only + standard`
- [x] `P2.0 + plain + projected + pre-norm-only + standard`
- [x] frozen `A`
- [x] record that `P2.3` beat frozen `A + M` on the matched longer-budget surface
- [x] record that `P2.3` beat frozen `A + M` again on the second-seed
  longer-budget surface

---

## Explicit Prohibitions

- [x] do not start `A + M + P2` during this checklist
- [x] do not import Path 2 retrieval ideas into `P2`
- [x] do not redefine the frozen `A + M` baseline while `P2` is under test
- [x] do not count unlogged runs as evidence
