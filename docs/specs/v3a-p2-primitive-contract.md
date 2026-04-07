# v3A P2 Primitive Contract

Status: implemented P2, interface plateau recorded, primitive-quality phase active

This document defines the next Path 1 contender, `P2`.

`P2` is not a memory sidecar and not a retrieval primitive. It is a predictive-core sequence primitive that keeps the causal role of `P1Contractive`, but restores the missing expressiveness contracts that the frozen Rust Mamba-3 baseline made visible.

The contract is intentionally split in two layers:

1. a reusable **Base Sequence Primitive Contract** for any Path 1 primitive
2. a stricter **P2 Specialization** that defines the next hypothesis

---

## Why P2 Exists

The frozen Path 1 baseline established:

- `A + M` beats both `A` and `A + P1`
- `P1Contractive` is underpowered relative to the Rust Mamba-3 baseline

The strongest architectural lessons were:

- `P1` only blends state; it does not transform state before updating it
- `P1` collapses latent memory and emitted output into the same object
- richer recurrent blocks separate:
  - internal latent memory
  - emitted token representation

So `P2` is the first primitive revision that preserves the basic Path 1 role while fixing those missing contracts.

Current primitive-quality status:

- the interface plateau has been logged
- `P2.1` proved that wider latent state alone is insufficient
- the first width/readout `2 x 2` sweep kept the original `P2` corner as the incumbent
- first state-dynamics contender: `P2.3`, which keeps base width and explicit readout but enriches the transformed carry path before the update gate
- first `P2.3` tracked run was negative in isolation
- the full `270`-configuration contender sweep then changed the incumbent ordering
- current best-known quality contender:
  - family `P2.3`
  - `gated`
  - `projected-norm`
  - `residual-renorm`
  - `standard`
- current best-known efficiency-quality contender:
  - family `P2.0`
  - `scaled`
  - `projected`
  - `pre-norm-only`
  - `standard`
- the next gate is no longer broad exploration
- the next gate is confirming the short-listed winners on another seed and a longer budget
- first longer-budget confirmation is now complete:
  - `P2.3 + gated + projected-norm + residual-renorm + standard`
  - beat frozen `A + M` on the matched `64`-step, `8`-eval-batch surface
- second-seed longer-budget confirmation is now complete:
  - `P2.3 + gated + projected-norm + residual-renorm + standard`
  - beat frozen `A + M` again
  - but frozen `A` unexpectedly won the seed-43 longer-budget surface overall
- so the contender short-list is stronger than before, but not yet frozen
- after replay on the frozen FineWeb canary default surface:
  - `P2.3` remained the strongest hybrid-quality contender
  - but it no longer showed a stable win over frozen `A + M`
  - so no new contender baseline is frozen yet

---

## Base Sequence Primitive Contract

This section defines the generic interface that any Path 1 predictive-core primitive must satisfy.

### Scope

Allowed:

- causal sequence processing
- latent recurrent state
- emitted token-level outputs
- deterministic save/load
- integration inside the Path 1 hybrid attention matrix

Forbidden:

- routed retrieval
- external memory write paths
- tree/index traversal
- hidden side channels outside the predictive core
- Path 2 memory/index behavior in any form

### Typed Objects

Every primitive implementation must expose three typed surfaces:

- `latent_state_t`
  - the internal recurrent memory after step `t`
- `emitted_output_t`
  - the token representation emitted to the residual stream at step `t`
- `step_aux_t`
  - optional diagnostics needed for testing or debugging

### Required API Shape

Every primitive must support a deterministic contract equivalent to:

```rust
init_state(batch_size, device) -> LatentState

step(
    x_t: Tensor,
    state_prev: &LatentState,
    config: &PrimitiveConfig,
) -> StepResult {
    next_state: LatentState,
    emitted_output: Tensor,
    aux: StepAux,
}

scan(
    xs: Tensor,
    state0: &LatentState,
    config: &PrimitiveConfig,
) -> ScanResult {
    emitted_outputs: Tensor,
    final_state: LatentState,
    aux: ScanAux,
}
```

### Required Invariants

- Causal:
  - `step(x_t, ...)` may depend only on `x_<=t` and prior latent state
- Deterministic:
  - fixed seed + fixed weights + fixed backend path must produce stable outputs within test tolerance
- Step/scan equivalence:
  - repeated `step(...)` must match `scan(...)` under the same weights, inputs, and initial state
- Record/load reproducibility:
  - saved then reloaded primitive weights must preserve logits within test tolerance
- Residual compatibility:
  - `emitted_output_t` must match the model width expected by the surrounding residual stream

### Generic Config Surface

Every primitive config must be typed and validated. At minimum it must expose:

- `model_dim`
- `state_dim`
- `layer_role`
- `dtype` / backend-compatible precision
- any transform/readout knobs used by the primitive

Config must reject impossible or contradictory settings in code, not only in docs.

### Required Test Surface

Every primitive that implements this contract must include:

- shape validation tests
- deterministic init tests
- step vs scan equivalence tests
- save/load reproducibility tests
- minimal Path 1 matrix integration smoke

---

## P2 Specialization

`P2` is the first improved primitive profile that implements the base contract.

### First Implemented Profile

The first implemented `P2` block in the Rust Path 1 stack is:

- `P2RotaryReadout`

Current implementation characteristics:

- latent state width equals model width
- prior state is projected, then rotated by an input-conditioned angle transform
- candidate update is produced from the current token input
- next state is a gated blend of transformed prior state and fresh candidate
- emitted output is a gated projection read from latent state, not latent state by identity

This is intentionally the first `P2` implementation, not the final design space.

Implemented config on the tracked Path 1 surface:

- model width: `128`
- head count: `4`
- local window: `256`
- schedule: `A-P2-A-P2-A-P2-A-P2`
- variant label: `primitive-hybrid-p2`
- primitive id: `p2-rotary-readout`

### P2 Thesis

`P2` keeps the spirit of `P1`:

- simple
- causal
- predictive-core only

But it adds the missing expressive contracts:

1. transformed state dynamics before blending
2. explicit emitted output distinct from latent state
3. latent memory and emitted representation are not identical by default

### First Primitive-Quality Sweep Outcome

The first structured primitive-quality sweep tested the most likely coupled pair:

- latent width: `base | wide`
- explicit internal readout: `off | on`

Recorded short tracked result:

| Label | Meaning | Final loss | Train tok/s | RSS delta MB |
| --- | --- | ---: | ---: | ---: |
| `P2.0` | base width + no explicit internal readout | `4.1547` | `5.61` | `116.53` |
| `P2` | base width + explicit internal readout | `3.7060` | `4.82` | `128.97` |
| `P2.1` | wide latent only | `4.0410` | `3.55` | `163.86` |
| `P2.2` | wide latent + explicit internal readout | `4.3838` | `2.79` | `180.09` |

Interpretation:

- `P2` remains the best corner
- explicit internal readout helps at base width
- wider latent state hurt with and without explicit readout
- the width/readout complementarity hypothesis did not win on the tracked surface

So the active contender stays:

- `P2RotaryReadout`
- base latent width
- explicit internal readout
- incumbent wrapper:
  - `plain`
  - `direct`
  - `pre-norm-only`
  - `standard`

### Required Architectural Upgrades Over P1

#### 1. State Transform Stage

Before any blend/update, `P2` must transform prior state.

Allowed examples:

- rotation
- phase-like transform
- learned state mixing transform
- another typed transformation with equivalent expressive role

The contract is:

```text
state_transformed_t = T(state_{t-1}, x_t, params)
```

This transform is required. A plain interpolation between old state and candidate does not satisfy `P2`.

#### 2. Candidate Update Stage

`P2` must still produce a fresh input-conditioned candidate update:

```text
candidate_t = C(x_t, params)
```

#### 3. Controlled State Update Stage

The next latent state must combine:

- transformed prior state
- fresh candidate update
- explicit learned update control

The contract is:

```text
state_t = U(state_transformed_t, candidate_t, x_t, params)
```

This can be gating, interpolation, or another typed update rule, but it must be explicit and testable.

#### 4. Separate Output Readout Stage

`P2` must emit an output as a read from latent state, not by reusing latent state directly:

```text
output_t = R(state_t, x_t, params)
```

Required invariant:

- `output_t` must not be defined as `state_t` by identity

A linear or gated readout is allowed. An output gate is encouraged but not mandatory if the emitted representation is still structurally distinct from latent state.

### P2 Minimal Equation Sketch

The intended shape is:

```text
state_transformed_t = T(state_{t-1}, x_t)
candidate_t         = C(x_t)
state_t             = U(state_transformed_t, candidate_t, x_t)
output_t            = R(state_t, x_t)
```

This is intentionally generic so the implementation can remain ours rather than becoming a renamed Mamba copy.

### What Counts As “Mamba-Like” Here

`P2` may borrow lessons from Mamba-family blocks:

- richer transformed state dynamics
- rotation-friendly state evolution
- explicit readout from latent state

But `P2` is not required to copy the full Rust Mamba-3 baseline architecture.

That distinction is important:

- `A + M` remains the frozen baseline
- `A + P2` remains the contender

---

## Fair Comparison Contract

`P2` must be evaluated against the frozen baseline, not against an evolving target.

Frozen comparator:

- `A`
- `A + M`

Historical context lane:

- `A + P1`

The first proving run for `P2` must use:

- the same Path 1 matrix runner
- the same tracked ledger
- the same basic budget surface
- a fixed schedule with `A + P2` occupying the same structural role as `A + P1`

Initial expected schedule:

- `A-P2-A-P2-A-P2-A-P2`

No `A + M + P2` composite run is allowed before `A + P2` has one honest standalone comparison against frozen `A + M`.

## Current Best-Known P2 Wrapper

The interface-ablation phase did not produce a clean wrapper winner over the incumbent. So the current best-known `P2` wrapper is frozen by non-replacement as:

* residual: `plain`
* readout: `direct`
* norm: `pre-norm-only`
* wrapper symmetry: `standard`

This wrapper remains fixed for primitive-quality work unless a new larger architectural interface hypothesis is opened explicitly.

## Primitive-Quality Next Phase

The next Path 1 question is no longer "which simple wrapper should surround `P2`?"

It is now:

* can primitive-internal changes improve `A + P2` quality without giving back its systems advantage over `A + M`?

That primitive-quality work is governed by:

* [`v3a-p2-primitive-quality-plan.md`](./v3a-p2-primitive-quality-plan.md)

Ordered ladder:

1. latent/output separation strength
2. state dynamics
3. update rule
4. readout capacity

### P2.1

`P2.1` is the first primitive-quality revision.

Hypothesis:

* `P2` may still be underpowered because latent recurrent memory and emitted token representation are not separated strongly enough

`P2.1` must therefore change only one primitive axis:

* wider latent memory than model width

And in this first pass it must explicitly avoid a second simultaneous hypothesis:

* no learned latent-to-output readout bottleneck yet

And it must keep the frozen wrapper:

* `plain`
* `direct`
* `pre-norm-only`
* `standard`

Recorded result:

* `P2.1` width-only is **insufficient alone**

So the next active primitive-quality gate is not another single-axis wrapper adjustment. It is the width/readout factorial sweep documented in:

* [`v3a-p2-primitive-quality-plan.md`](./v3a-p2-primitive-quality-plan.md)

## Current Test Surface

The first implemented `P2` block is currently covered by:

- shape/config validation through the Path 1 config surface
- primitive scan equivalence test:
  - `P2RotaryReadoutSequenceMixer::scan(...)` matches a manual `step(...)` loop
- primitive-hybrid matrix integration smoke:
  - `build_primitive_hybrid_attention_model(...)` with the `P2` candidate variant returns logits of the expected shape
- existing Path 1 runner/eval surfaces:
  - `cargo test -p fractal-core hybrid_attention --quiet`
  - `cargo test -p fractal-eval-private hybrid_attention --quiet`
  - `cargo test -p fractal --quiet --bin v3a-hybrid-attention-matrix`
  - strict lint passes for the same surfaces

## Recorded Results

First tracked `P2` comparison:

- run label: `v3a-p2-first-proving-seed42`
- recorded at unix seconds: `1775409856`
- ledger: [`/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl`](/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl)
- command:

```bash
cargo run --quiet --bin v3a-hybrid-attention-matrix -- --steps 16 --eval-batches 4 --seed 42 --primitive-profile p2 --ledger-path default --run-label v3a-p2-first-proving-seed42 --output table
```

- artifact root: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775408694](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775408694)

Results:

- `A` / attention-only:
  - initial loss `5.8107`
  - final loss `4.1266`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775408694/attention-only/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775408694/attention-only/report.json)
- `A + M` / reference-ssm-hybrid:
  - initial loss `5.6956`
  - final loss `3.7429`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775408694/reference-ssm-hybrid/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775408694/reference-ssm-hybrid/report.json)
- `A + P2` / primitive-hybrid-p2:
  - initial loss `5.6871`
  - final loss `3.7060`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775408694/primitive-hybrid/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775408694/primitive-hybrid/report.json)

Current read:

- `A + P2` clearly beats `A`
- `A + P2` also edges `A + M` on this first tracked seed and budget
- this is real positive signal, but not yet enough to freeze a new contender ordering without the second seed

Second tracked `P2` comparison:

- run label: `v3a-p2-first-proving-seed43`
- recorded at unix seconds: `1775412327`
- ledger: [`/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl`](/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl)
- command:

```bash
cargo run --quiet --bin v3a-hybrid-attention-matrix -- --steps 16 --eval-batches 4 --seed 43 --primitive-profile p2 --ledger-path default --run-label v3a-p2-first-proving-seed43 --output table
```

- artifact root: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775411197](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775411197)

Results:

- `A` / attention-only:
  - initial loss `5.6271`
  - final loss `4.2000`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775411197/attention-only/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775411197/attention-only/report.json)
- `A + M` / reference-ssm-hybrid:
  - initial loss `5.7324`
  - final loss `3.6749`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775411197/reference-ssm-hybrid/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775411197/reference-ssm-hybrid/report.json)
- `A + P2` / primitive-hybrid-p2:
  - initial loss `5.4487`
  - final loss `3.9758`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775411197/primitive-hybrid/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775411197/primitive-hybrid/report.json)

Two-seed read:

- `A + P2` beat `A + M` on seed 42
- `A + M` beat `A + P2` on seed 43
- `A + P2` beat pure `A` on both tracked seeds
- `P2` therefore has real positive signal and has clearly cleared the old `P1` bar, but it has **not** yet demonstrated a stable win over the frozen Rust Mamba-3 baseline

Longer-budget tracked comparison:

- run label: `v3a-p2-longer-budget-seed42`
- recorded at unix seconds: `1775417002`
- ledger: [`/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl`](/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl)
- command:

```bash
cargo run --quiet --bin v3a-hybrid-attention-matrix -- --steps 64 --eval-batches 8 --seed 42 --primitive-profile p2 --ledger-path default --run-label v3a-p2-longer-budget-seed42 --output table
```

- artifact root: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775412864](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775412864)

Results:

- `A` / attention-only:
  - initial loss `5.8408`
  - final loss `3.5946`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775412864/attention-only/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775412864/attention-only/report.json)
- `A + M` / reference-ssm-hybrid:
  - initial loss `5.7159`
  - final loss `3.5186`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775412864/reference-ssm-hybrid/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775412864/reference-ssm-hybrid/report.json)
- `A + P2` / primitive-hybrid-p2:
  - initial loss `5.6460`
  - final loss `3.6133`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775412864/primitive-hybrid/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775412864/primitive-hybrid/report.json)

Current decision read:

- the longer-budget run cooled the early optimism
- `A + M` is currently the strongest lane on the best tracked budget we have run so far
- `A + P2` still appears materially better than the old `P1` line and competitive enough to justify further iteration
- `A + P2` has **not** yet earned a claim of beating the frozen Rust Mamba-3 baseline

Exhaustive contender sweep:

- run label prefix: `v3a-p2-contender-grid-s42-`
- completed configurations: `270`
- ledger: [`/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl`](/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl)
- sweep driver: [`/private/tmp/fractal-v3a-merge-G4mipf/scripts/v3a_p2_contender_sweep.sh`](/private/tmp/fractal-v3a-merge-G4mipf/scripts/v3a_p2_contender_sweep.sh)

Sweep read:

- `50 / 270` contender configs beat frozen `A + M` on the short tracked surface
- `12 / 270` contender configs beat the previous `A + P2` incumbent
- best overall quality contender:
  - `P2.3`
  - `gated`
  - `projected-norm`
  - `residual-renorm`
  - `standard`
  - loss `3.6006`
  - train throughput `3.07 tok/s`
  - RSS delta `154.78 MB`
- best efficiency-quality contender:
  - `P2.0`
  - `scaled`
  - `projected`
  - `pre-norm-only`
  - `standard`
  - loss `3.7001`
  - train throughput `6.73 tok/s`
  - RSS delta `108.14 MB`

Updated read:

- the broad logged sweep overturned the narrow one-axis read
- there are now real contender configurations that beat frozen `A + M` on the tracked short surface
- the next gate should be confirmation, not more broad exploration:
  - second-seed replay
  - longer-budget replay
  - then freeze a short-list

Isolated instrumented longer-budget comparison:

- run label: `v3a-p2-longer-budget-seed42-isolated`
- recorded at unix seconds: `1775430705`
- ledger: [`/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl`](/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl)
- command:

```bash
cargo run --quiet --bin v3a-hybrid-attention-matrix -- --steps 64 --eval-batches 8 --seed 42 --primitive-profile p2 --ledger-path default --run-label v3a-p2-longer-budget-seed42-isolated --output table
```

- artifact root: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775426451](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775426451)

Results:

- `A` / attention-only:
  - final loss `3.5946`
  - train throughput `39.71 tok/s`
  - RSS delta `56.58 MB`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775426451/attention-only/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775426451/attention-only/report.json)
- `A + M` / reference-ssm-hybrid:
  - final loss `3.5186`
  - train throughput `0.64 tok/s`
  - RSS delta `700.97 MB`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775426451/reference-ssm-hybrid/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775426451/reference-ssm-hybrid/report.json)
- `A + P2` / primitive-hybrid-p2:
  - final loss `3.6133`
  - train throughput `4.44 tok/s`
  - RSS delta `166.06 MB`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775426451/primitive-hybrid/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775426451/primitive-hybrid/report.json)

Updated read after isolation:

- the loss ordering did **not** change under fairer isolated measurement
- `A + P2` remains behind `A + M` on quality
- `A + P2` now has a trustworthy efficiency story:
  - much faster than `A + M`
  - much lower RSS growth than `A + M`
- this strengthens the case for further `P2` iteration, but not for replacing the frozen `A + M` baseline

## Next Gated Work

The next Path 1 target is **not** immediate primitive-internal redesign.

The next target is:

- improve the interface between `P2` and the attention/residual stack while holding the inner primitive fixed

Why this comes first:

- `P2` already has real positive signal
- `P2` already has a credible efficiency story
- the most plausible remaining bottleneck is now the wrapper contract rather than total primitive absence of value

So the next phase is:

1. interface quality ablations
2. then primitive-quality ablations only after the best interface is frozen

The dedicated plan for that phase is:

- [`v3a-p2-interface-ablation-plan.md`](./v3a-p2-interface-ablation-plan.md)

Current interface-status note:

- the first tracked residual-family ablation did **not** beat the incumbent `plain` `P2` wrapper
- `scaled` was close enough to remain plausible
- `gated` underperformed clearly on the first seed
- the first tracked readout-family ablation also did **not** beat the incumbent `direct` handoff
- `projected-norm` outperformed `projected`, but both trailed the current `plain residual + direct readout` wrapper
- the first tracked norm-family ablation also did **not** beat the incumbent `pre-norm-only` wrapper
- `residual-renorm` remained closer than `post-readout-norm`, but still trailed the current best wrapper
- the first tracked wrapper-symmetry ablation also did **not** beat the incumbent standard wrapper
- `mamba-rms` stayed close enough to remain plausible, but it still trailed the current best wrapper

Current best-known `P2` wrapper after the first four interface families:

- residual mode: `plain`
- readout handoff: `direct`
- norm placement: `pre-norm-only`
- wrapper symmetry profile: `standard`

---

## Non-Goals

`P2` is explicitly not:

- a Path 2 memory/index primitive
- a routed retrieval operator
- a tree memory write/read mechanism
- a direct Mamba-3 clone
- a MoE block
- a justification for changing the frozen `A + M` baseline

---

## Implementation Exit Gate

This contract is considered implementation-ready when the resulting primitive:

- satisfies the Base Sequence Primitive Contract
- includes a required transform stage `T`
- includes a required readout stage `R`
- preserves deterministic step/scan semantics
- integrates as `A + P2` without changing the Path 1 benchmark definition

The next decision-bearing artifact after implementation must be:

- one logged `A / A + M / A + P2` comparison in [`/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl`](/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl)
