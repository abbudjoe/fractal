# v3A P2 Interface Ablation Plan

Status: residual-, readout-, and norm-family control planes implemented

This document defines the next Path 1 question after the first `P2` proving runs:

* is the main remaining quality gap in the primitive itself,
* or in the interface between the primitive and the attention stack?

Current evidence says the next work should target the interface first.

Why:

* `A + P2` already has real positive signal
* `A + P2` has a materially better efficiency profile than the current Rust `A + M` lane
* `A + P2` has **not** yet matched the best tracked `A + M` quality result

That means the most likely next ambiguity is no longer “does `P2` do anything?”
It is:

* **are we handing `P2` to the surrounding stack in the right way?**

---

## Fixed Discipline

During this phase:

* the inner `P2RotaryReadout` primitive stays fixed
* `A` stays frozen
* `A + M` stays frozen
* the only allowed changes are the wrapper/interface contracts around `P2`

This is a one-variable-at-a-time phase.

Do **not** change:

* the primitive’s transform math
* the primitive’s readout math
* the local attention window
* model width or depth
* training budget
* eval suites

Do **not** test composite `A + M + P2` in this phase.

---

## Interface Hypotheses

The current hypothesis ladder is:

1. the primitive core is good enough to matter
2. the wrapper may be leaving quality on the table
3. fixing the wrapper may improve quality without surrendering the current speed/memory advantage

The main interface targets are:

1. **norm placement**
   - whether the primitive sees the right normalized input distribution
   - whether the emitted output is normalized appropriately before residual write-back

2. **residual injection form**
   - whether the primitive output is too strong, too weak, or too unstable when added back to the stream

3. **readout handoff**
   - whether the emitted output should be projected or renormalized before entering the residual stream

4. **wrapper symmetry**
   - whether the `P2` wrapper should look more like the successful Rust Mamba-3 block wrapper at the interface level

---

## Ablation Matrix

The initial ablation matrix should stay narrow.

### Family 1: Norm Placement

Hold the primitive fixed and compare:

* `pre-norm only`
* `pre-norm + post-readout norm`
* `pre-norm + residual-side renorm`

Question:

* does quality improve when the primitive output is better aligned with the residual stream distribution?

Current implementation status:

* this family is now wired into `src/bin/v3a-hybrid-attention-matrix.rs`
* runner flag:
  * `--primitive-norm-profile pre-norm-only|post-readout-norm|residual-renorm`
* current typed contender labels:
  * `primitive-hybrid-p2`
  * `primitive-hybrid-p2-post-readout-norm`
  * `primitive-hybrid-p2-residual-renorm`

First tracked seed-42 comparison:

Frozen quality references on the same `16`-step / `4`-eval-batch budget:

* `A`: final loss `4.1266`
* `A + M`: final loss `3.7429`

Tracked primitive-only norm-family runs:

* `pre-norm-only`
  * reused incumbent wrapper run label: `v3a-p2-interface-residual-plain-primitive-only-seed42`
  * final loss: `3.7060`
  * train throughput: `4.74 tok/s`
  * RSS delta: `125.28 MB`
* `post-readout-norm`
  * run label: `v3a-p2-interface-norm-post-readout-seed42`
  * final loss: `4.2352`
  * train throughput: `4.96 tok/s`
  * RSS delta: `123.41 MB`
* `residual-renorm`
  * run label: `v3a-p2-interface-norm-residual-renorm-seed42`
  * final loss: `3.7885`
  * train throughput: `5.01 tok/s`
  * RSS delta: `120.00 MB`

First read:

* `pre-norm-only` remains the best quality norm mode on the first tracked seed
* `residual-renorm` is the only non-default norm wrapper that stayed close enough to the incumbent to remain plausible
* `post-readout-norm` is a clear quality regression
* all three norm modes stay in the same rough throughput and RSS band

So the first norm-family result does **not** yet show a clear interface win over the incumbent `pre-norm-only` wrapper.

### Family 2: Residual Injection Form

Hold the primitive fixed and compare:

* `plain residual add`
* `scaled residual add`
* `gated residual add`

Question:

* is the primitive contribution currently underweighted, overweighted, or unstable?

Current implementation status:

* this family is now wired into `src/bin/v3a-hybrid-attention-matrix.rs`
* runner flag:
  * `--primitive-residual-profile plain|scaled|gated`
* current typed contender labels:
  * `primitive-hybrid-p2`
  * `primitive-hybrid-p2-scaled-residual`
  * `primitive-hybrid-p2-gated-residual`
* one smoke run has verified the gated lane executes end to end
  * this smoke is a control-plane check, not decision-bearing evidence

First tracked seed-42 comparison:

Frozen quality references on the same `16`-step / `4`-eval-batch budget:

* `A`: final loss `4.1266`
* `A + M`: final loss `3.7429`

Tracked primitive-only residual-family runs:

* `plain`
  * run label: `v3a-p2-interface-residual-plain-primitive-only-seed42`
  * final loss: `3.7060`
  * train throughput: `4.74 tok/s`
  * RSS delta: `125.28 MB`
* `scaled`
  * run label: `v3a-p2-interface-residual-scaled-seed42`
  * final loss: `3.7125`
  * train throughput: `4.49 tok/s`
  * RSS delta: `126.30 MB`
* `gated`
  * run label: `v3a-p2-interface-residual-gated-seed42`
  * final loss: `3.9872`
  * train throughput: `4.77 tok/s`
  * RSS delta: `120.69 MB`

First read:

* `plain` remains the best quality mode in the first tracked residual-family run
* `scaled` is very close on loss and may still be worth one longer-budget follow-up
* `gated` is clearly worse on quality on this first seed
* the three residual modes are in roughly the same throughput and RSS band

So the first residual-family result does **not** yet show a clear interface win over the incumbent `plain` wrapper.

### Family 3: Readout Handoff

Hold the primitive fixed and compare:

* `direct emitted output`
* `project emitted output before residual`
* `project + norm before residual`

Question:

* is the quality gap mostly a readout-to-residual alignment problem?

Current implementation status:

* this family is now wired into `src/bin/v3a-hybrid-attention-matrix.rs`
* runner flag:
  * `--primitive-readout-profile direct|projected|projected-norm`
* current typed contender labels:
  * `primitive-hybrid-p2`
  * `primitive-hybrid-p2-projected-readout`
  * `primitive-hybrid-p2-projected-norm-readout`
* one smoke run has verified the `projected` lane executes end to end
  * this smoke is a control-plane check, not decision-bearing evidence

First tracked seed-42 comparison:

Frozen quality references on the same `16`-step / `4`-eval-batch budget:

* `A`: final loss `4.1266`
* `A + M`: final loss `3.7429`

Tracked primitive-only readout-family runs:

* `direct`
  * reused incumbent wrapper run label: `v3a-p2-interface-residual-plain-primitive-only-seed42`
  * final loss: `3.7060`
  * train throughput: `4.74 tok/s`
  * RSS delta: `125.28 MB`
* `projected`
  * run label: `v3a-p2-interface-readout-projected-seed42`
  * final loss: `4.0360`
  * train throughput: `4.62 tok/s`
  * RSS delta: `120.38 MB`
* `projected-norm`
  * run label: `v3a-p2-interface-readout-projected-norm-seed42`
  * final loss: `3.9476`
  * train throughput: `4.67 tok/s`
  * RSS delta: `128.31 MB`

First read:

* `direct` remains the best quality readout mode on the first tracked seed
* `projected-norm` is better than `projected`, but both trail the incumbent `direct` handoff
* all three readout modes remain in the same rough throughput and RSS band
* this does **not** yet support the idea that the main quality gap is a simple readout-to-residual alignment mismatch

So the first readout-family result does **not** yet show a clear interface win over the incumbent `direct` wrapper.

### Family 4: Wrapper Symmetry Against A+M

Only where it is architecturally fair, compare a `P2` wrapper shaped more like the Rust Mamba-3 lane:

* matching norm placement
* matching residual ordering
* matching wrapper projection policy

Question:

* does the quality gap shrink when the wrapper contract is closer to the proven `A + M` lane?

Current implementation status:

* this family is now wired into `src/bin/v3a-hybrid-attention-matrix.rs`
* runner flag:
  * `--primitive-wrapper-profile standard|mamba-rms`
* current typed contender labels:
  * `primitive-hybrid-p2`
  * `primitive-hybrid-p2-mamba-rms-wrapper`
* one smoke run has verified the `mamba-rms` lane executes end to end
  * this smoke is a control-plane check, not decision-bearing evidence

First tracked seed-42 comparison:

Frozen quality references on the same `16`-step / `4`-eval-batch budget:

* `A`: final loss `4.1266`
* `A + M`: final loss `3.7429`

Tracked primitive-only wrapper-symmetry runs:

* `standard`
  * reused incumbent wrapper run label: `v3a-p2-interface-residual-plain-primitive-only-seed42`
  * final loss: `3.7060`
  * train throughput: `4.74 tok/s`
  * RSS delta: `125.28 MB`
* `mamba-rms`
  * run label: `v3a-p2-interface-wrapper-mamba-rms-seed42`
  * final loss: `3.7315`
  * train throughput: `4.45 tok/s`
  * RSS delta: `119.48 MB`

First read:

* the `mamba-rms` wrapper did **not** beat the incumbent standard wrapper on the first tracked seed
* it stayed reasonably close on quality and slightly reduced RSS, but it also gave up some throughput
* it remains a plausible interface-shaped alternative, but not a clear winner

So the first wrapper-symmetry result does **not** yet show a clean interface win over the incumbent standard wrapper.

---

## Run Protocol

Every decision-bearing interface ablation must:

* use `src/bin/v3a-hybrid-attention-matrix.rs`
* append to [`/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl`](/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl)
* use a stable `run_label`
* be documented back into:
  * this file
  * [`v3a-p2-primitive-contract.md`](./v3a-p2-primitive-contract.md)

Minimum evidence for an interface winner:

1. one short seeded matrix
2. one larger-budget follow-up
3. tracked comparison against:
   * `A`
   * `A + M`
   * the current best `A + P2` wrapper

---

## Winning Criteria

An interface change counts as a winner only if it does all of the following:

* improves `A + P2` quality on the tracked surface
* preserves a meaningful throughput advantage over `A + M`
* preserves a meaningful memory advantage over `A + M`
* does not collapse stability relative to the current `P2` wrapper

If a wrapper improves quality but destroys the efficiency thesis, it does not count as a clean Path 1 win.

---

## Failure Criteria

The interface phase fails if:

* no wrapper change materially improves `A + P2` quality
* the better wrappers only help by giving up the systems advantage
* the results remain too unstable to distinguish interface effects from noise

If that happens, the next step is primitive-quality work, not more wrapper churn.

---

## Exit Gate

This phase is complete when:

* one best-known `P2` interface is chosen on logged evidence
* the winning interface is documented in [`v3a-p2-primitive-contract.md`](./v3a-p2-primitive-contract.md)
* only then does Path 1 move on to primitive-internal `P2` revisions
