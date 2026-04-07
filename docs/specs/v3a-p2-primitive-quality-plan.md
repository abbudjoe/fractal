# v3A P2 Primitive-Quality Plan

Status: active after interface plateau

This document starts the next gated Path 1 phase:

* improve the inside of `P2`
* while holding the current best wrapper fixed
* and while keeping `A` and frozen `A + M` unchanged

The interface phase did useful work, but no wrapper family clearly dethroned the incumbent:

* residual form
* readout handoff
* norm placement
* wrapper symmetry

So primitive quality is now the leading ambiguity.

---

## Fixed Primitive-Quality Discipline

Hold fixed unless a primitive-quality ablation explicitly changes them:

* `A`
* frozen `A + M`
* `A + P2` wrapper:
  * residual: `plain`
  * readout: `direct`
  * norm: `pre-norm-only`
  * wrapper symmetry: `standard`
* local attention window
* layer schedule
* model width
* training budget for a given proving round
* ledger and reporting surface

Do not mix wrapper changes and primitive-internal changes in the same ablation.

Do not import Path 2 memory/index behavior.

Every decision-bearing primitive-quality run must:

* use `src/bin/v3a-hybrid-attention-matrix.rs`
* append to [`/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl`](/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl)
* use a stable `run_label`
* be documented back into:
  * this file
  * [`v3a-p2-primitive-contract.md`](./v3a-p2-primitive-contract.md)

Benchmark-surface note:

* earlier primitive-quality runs were recorded on the mutable repo-docs smoke
  corpus
* those runs remain useful historical screening signal
* but cross-seed contender-freeze decisions must now be replayed on the frozen
  FineWeb canary default surface

---

## Primitive-Quality Ladder

The primitive-quality work should proceed in this order:

1. latent/output separation strength
2. state dynamics
3. update rule
4. readout capacity

### Why This Order

1. **Latent/output separation strength**
   If latent memory and emitted output are still too similar, better dynamics and readout capacity can get masked. This is the cleanest first test because it changes the memory contract without reopening wrapper questions.

2. **State dynamics**
   Once latent/output separation is stronger, the next likely limitation is whether the primitive’s internal memory evolves richly enough over time.

3. **Update rule**
   After state dynamics improve, the next question is whether keep/write/overwrite decisions are expressive enough and stable enough.

4. **Readout capacity**
   This should come last because it is the easiest place to buy superficial quality improvements without actually improving the recurrent memory contract.

---

## Target Axes

### 1. Latent/Output Separation Strength

Goal:

* make internal memory and emitted token representation genuinely different jobs

Candidate levers:

* latent state width larger than model width
* direct output slice from a wider latent state
* later explicit readout from latent state to residual-stream width
* stronger diagnostic separation between latent memory and emitted output

Expected success signal:

* quality improves without a large throughput collapse
* the primitive remains meaningfully lighter than `A + M`

### 2. State Dynamics

Goal:

* make the recurrent memory evolve more richly than a single transform-and-blend step

Candidate levers:

* accumulated phase or angle state
* multi-timescale channels
* learned state mixing before or alongside rotation
* separate transforms for retained state and candidate state

Expected success signal:

* `A + P2.x` closes quality gap to frozen `A + M`
* state norms remain stable

### 3. Update Rule

Goal:

* improve how the primitive decides what to keep, what to overwrite, and how sharply to blend

Candidate levers:

* decoupled retain and write gates
* grouped or per-channel update controls
* smoother interpolation or trap-like updates
* explicit skip/carry path distinct from candidate write path

Expected success signal:

* better loss without obvious gate saturation or systems regression

### 4. Readout Capacity

Goal:

* improve the token-level representation emitted from latent state without making readout do all the work

Candidate levers:

* gated readout MLP
* bottleneck width sweeps
* dual-branch content/gating readout
* grouped or low-rank readout variants

Expected success signal:

* modest quality gain
* no large RSS blowup
* readout remains subordinate to the memory contract rather than replacing it

---

## P2.1: Stronger Latent/Output Separation

`P2.1` is the first primitive-quality revision.

Hypothesis:

* `P2` may still be underpowered because latent memory and emitted output are not separated strongly enough

`P2.1` must therefore change only one primitive axis:

* larger latent state than model width

The initial `P2.1` contract is:

* wider latent recurrent state than model width
* no learned latent-to-output readout yet
* emitted output taken from a direct designated slice of the widened latent state
* current best wrapper held fixed:
  * `plain`
  * `direct`
  * `pre-norm-only`
  * `standard`

Initial implementation target:

* latent width: `2 * d_model`
* emitted output width: `d_model`
* emitted output = direct leading slice of latent state, optionally modulated by the existing output gate
* schedule remains:
  * `A-P2.1-A-P2.1-A-P2.1-A-P2.1`

This keeps the first primitive-quality move interpretable:

* if quality improves, latent width alone mattered
* if it does not, the next likely bottleneck is deeper dynamics or update-rule quality

---

## P2.1 Run Gate

Minimum evidence for `P2.1`:

1. one short seeded matrix on the tracked surface
2. one longer-budget follow-up only if the short run is promising
3. tracked comparison against:
   * `A`
   * frozen `A + M`
   * incumbent `A + P2`

`P2.1` counts as promising only if it:

* improves `A + P2` quality on the tracked surface
* preserves a meaningful throughput advantage over `A + M`
* preserves a meaningful memory advantage over `A + M`

If `P2.1` fails that bar, the next move should be:

* a small factorial sweep over the most likely coupled pair:
  * latent width
  * internal explicit readout
* not wrapper churn

## Width/Readout Factor Sweep

Because `P2.1` failed as a single-axis move, the next active primitive-quality gate is a structured `2 x 2` sweep over the most likely coupled pair:

* latent width:
  * `base`
  * `wide`
* internal explicit readout:
  * `off`
  * `on`

Tracked corners:

| Label | Latent width | Internal explicit readout | Intended meaning |
| --- | --- | --- | --- |
| `P2.0` | base | off | transformed-state primitive without separate learned internal readout |
| `P2` | base | on | current incumbent primitive contender |
| `P2.1` | wide | off | wider latent memory only |
| `P2.2` | wide | on | wider latent memory plus separate learned internal readout |

Wrapper remains fixed for all four:

* `plain`
* `direct`
* `pre-norm-only`
* `standard`

This sweep is the smallest disciplined way to answer the complementarity question:

* does latent width only help when explicit readout is also present?
* does explicit readout only help once latent width is larger?
* or is the current `P2` corner already the strongest of the four?

### Sweep Protocol

Minimum first pass:

1. run the four corners on the same short seeded primitive-only surface
2. compare them against frozen references:
   * `A`
   * `A + M`
3. promote only promising corners to a larger-budget follow-up

### Current Read Before The Sweep

Recorded so far:

* `P2` has real positive signal and a systems advantage
* `P2.1` as width-only did **not** improve on `P2`

So the active question is no longer “does width alone help?”

It is:

* “does width plus readout help in combination, or is the current `P2` corner already the right part of this design space?”

### Width/Readout Sweep Result

Tracked `seed=42`, `16`-step, `4`-eval-batch primitive-only sweep:

| Label | Latent width | Internal explicit readout | Final loss | Train tok/s | RSS delta MB | Read |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `P2.0` | base | off | `4.1547` | `5.61` | `116.53` | cheaper, but clearly weaker on quality |
| `P2` | base | on | `3.7060` | `4.82` | `128.97` | incumbent and best quality corner |
| `P2.1` | wide | off | `4.0410` | `3.55` | `163.86` | width alone is insufficient |
| `P2.2` | wide | on | `4.3838` | `2.79` | `180.09` | worst corner; width+readout did not rescue width |

Frozen references on the same tracked surface:

* `A`: loss `4.1266`
* frozen `A + M`: loss `3.7429`

What this means:

* explicit internal readout is the useful ingredient in this pair
* widening latent state hurt both with and without explicit readout
* the width/readout complementarity hypothesis did **not** pay off on this first tracked surface
* current incumbent remains:
  * base latent width
  * explicit internal readout
  * `plain`
  * `direct`
  * `pre-norm-only`
  * `standard`

So this sweep resolves the immediate ambiguity:

* we should stop spending cycles on latent-width expansion for now
* the next primitive-quality target should move forward to `state dynamics`, not more width/readout churn

## State-Dynamics Family

The next active primitive-quality question is:

* can richer transformed carry dynamics improve `P2` quality without giving back its systems advantage?

The current incumbent stays fixed:

* base latent width
* explicit internal readout
* `plain`
* `direct`
* `pre-norm-only`
* `standard`

### P2.3: Blended Carry Dynamics

`P2.3` is the first state-dynamics contender.

Hypothesis:

* the current `P2` carry path is too narrow because it only rotates a single transformed prior state before the update gate
* a richer transformed carry that blends:
  * a rotary carry path
  * and a learned non-rotary carry path
  may preserve the good `P2` systems profile while improving quality

Contract:

* keep latent width at base `d_model`
* keep explicit internal readout on
* keep the current update rule shape
* change only the transformed-state stage

Intended equation sketch:

```text
rotated_carry_t   = rotate(W_rot state_{t-1}, angle(x_t))
carried_state_t   = tanh(W_carry state_{t-1})
carry_mix_t       = sigmoid(W_mix x_t)
state_transformed = carry_mix_t * rotated_carry_t
                  + (1 - carry_mix_t) * carried_state_t
candidate_t       = C(x_t)
state_t           = U(state_transformed, candidate_t, x_t)
output_t          = R(state_t, x_t)
```

### P2.3 Run Gate

Minimum first pass:

1. run one tracked short primitive comparison on the shared `seed=42`, `16`-step, `4`-eval-batch surface
2. compare against:
   * frozen `A`
   * frozen `A + M`
   * incumbent `P2`
3. only run a longer-budget follow-up if `P2.3` is promising

`P2.3` counts as promising only if it:

* improves on incumbent `P2` quality
* preserves a meaningful throughput advantage over `A + M`
* preserves a meaningful RSS advantage over `A + M`

### P2.3 Result

Tracked run:

* run label: `v3a-p2-3-first-proving-seed42`
* surface: `seed=42`, `16` steps, `4` eval batches
* artifact: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775447491/primitive-hybrid/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775447491/primitive-hybrid/report.json)

Result:

* `A + P2.3`: loss `4.1240`, `2.82 tok/s`, `152.83 MB`

Matched references:

* frozen `A`: loss `4.1266`
* frozen `A + M`: loss `3.7429`
* incumbent `A + P2`: loss `3.7060`, `4.82 tok/s`, `128.97 MB`

Read:

* `P2.3` did **not** improve on `P2`
* it also regressed on throughput and RSS versus incumbent `P2`
* richer blended carry dynamics, in this first form, do not look like the missing ingredient

So the first state-dynamics contender is a negative result.

## Exhaustive Contender Sweep Surface

With `A` and frozen `A + M` held fixed, the next broad search surface is the
contender grid only:

* primitive families:
  * `P2.0`
  * `P2`
  * `P2.1`
  * `P2.2`
  * `P2.3`
* residual modes: `3`
* readout handoff modes: `3`
* norm modes: `3`
* wrapper symmetry modes: `2`

Total contender configurations:

* `5 * 3 * 3 * 3 * 2 = 270`

The repo-owned sequential resumable sweep entrypoint is:

* [`scripts/v3a_p2_contender_sweep.sh`](/private/tmp/fractal-v3a-merge-G4mipf/scripts/v3a_p2_contender_sweep.sh)

Design rules for that sweep:

* do not rerun `A`
* do not rerun frozen `A + M`
* run only `primitive-hybrid` variants
* append every executed contender run to the shared v3a ledger
* skip any run label already present in the ledger so interrupted sweeps can resume safely

### Exhaustive Sweep Result

The full `270`-configuration contender sweep completed on the tracked short
surface:

* seed `42`
* `16` steps
* `4` eval batches
* frozen references:
  * `A`: loss `4.1266`
  * frozen `A + M`: loss `3.7429`
  * prior incumbent `A + P2`: loss `3.7060`

Breadth of signal:

* `50 / 270` contender configurations beat frozen `A + M`
* `12 / 270` contender configurations beat the prior incumbent `A + P2`

Best overall quality:

| Rank | Label | Loss | Train tok/s | RSS delta MB |
| --- | --- | ---: | ---: | ---: |
| 1 | `P2.3 gated + projected-norm + residual-renorm + standard` | `3.6006` | `3.07` | `154.78` |
| 2 | `P2.3 gated + projected-norm + residual-renorm + mamba-rms` | `3.6030` | `2.34` | `172.41` |
| 3 | `P2.3 gated + projected + residual-renorm + mamba-rms` | `3.6187` | `3.27` | `155.48` |
| 4 | `P2.3 gated + projected + residual-renorm + standard` | `3.6192` | `3.04` | `156.81` |

Best efficiency among contenders that still beat frozen `A + M`:

| Label | Loss | Train tok/s | RSS delta MB |
| --- | ---: | ---: | ---: |
| `P2.0 scaled + projected + pre-norm-only + standard` | `3.7001` | `6.73` | `108.14` |
| `P2.0 plain + projected + pre-norm-only + standard` | `3.6894` | `5.96` | `111.42` |

Best result per primitive family:

| Family | Best tracked config | Loss | Train tok/s | RSS delta MB |
| --- | --- | ---: | ---: | ---: |
| `P2.0` | `plain + projected + pre-norm-only + standard` | `3.6894` | `5.96` | `111.42` |
| `P2` | `gated + projected + residual-renorm + mamba-rms` | `3.7047` | `4.64` | `119.94` |
| `P2.1` | `plain + projected-norm + residual-renorm + mamba-rms` | `3.7358` | `3.53` | `168.39` |
| `P2.2` | `gated + projected-norm + residual-renorm + mamba-rms` | `3.7079` | `2.90` | `187.83` |
| `P2.3` | `gated + projected-norm + residual-renorm + standard` | `3.6006` | `3.07` | `154.78` |

What changed in our understanding:

* the earlier one-axis ablations were too conservative
* cross-axis interactions are real
* the best-quality contender is now a `P2.3` configuration, not the original
  `P2` incumbent
* the best efficiency-quality trade is now a `P2.0` projected-readout
  configuration

So the sweep resolved the key uncertainty:

* there **are** contender configurations worth promoting
* the search should now narrow from broad exploration to confirmation
* the next gate should be replaying the short-list on:
  * a second seed
  * and a longer-budget run

### Longer-Budget Leader Confirmation

The first longer-budget confirmation set replayed six leaders as isolated
top-level runs on the matched longer-budget surface:

* seed `42`
* `64` steps
* `8` eval batches

Tracked set:

1. frozen `A + M`
2. incumbent `P2`
3. `P2.3 + gated + projected-norm + residual-renorm + standard`
4. `P2.0 + scaled + projected + pre-norm-only + standard`
5. `P2.0 + plain + projected + pre-norm-only + standard`
6. frozen `A`

Results:

| Variant | Final loss | Train tok/s | RSS delta MB |
| --- | ---: | ---: | ---: |
| `P2.3 + gated + projected-norm + residual-renorm + standard` | `3.4519` | `2.91` | `205.61` |
| frozen `A + M` | `3.5186` | `0.64` | `675.45` |
| frozen `A` | `3.5946` | `38.55` | `56.48` |
| `P2.0 + plain + projected + pre-norm-only + standard` | `3.6050` | `5.56` | `158.67` |
| incumbent `P2` | `3.6133` | `4.64` | `180.78` |
| `P2.0 + scaled + projected + pre-norm-only + standard` | `3.6302` | `5.63` | `168.56` |

What changed:

* the best-quality contender from the short sweep, `P2.3`, **held up** at the
  longer budget and beat frozen `A + M`
* the two `P2.0` efficiency leaders remained fast and light, but did **not**
  close the quality gap at longer budget
* frozen `A` still remains far cheaper than all hybrid contenders, but it lost
  on quality to both `A + M` and `P2.3`

Current read after longer-budget confirmation:

* `P2.3 + gated + projected-norm + residual-renorm + standard` is the current
  quality leader
* it has now beaten frozen `A + M` on:
  * the short tracked sweep
  * and the first longer-budget confirmation run
* this is now strong enough to justify a second-seed confirmation before we
  freeze a new contender baseline

### Second-Seed Longer-Budget Confirmation

The same isolated leader set was then replayed on the matched longer-budget
surface for a second seed:

* seed `43`
* `64` steps
* `8` eval batches

Results:

| Variant | Final loss | Train tok/s | RSS delta MB |
| --- | ---: | ---: | ---: |
| frozen `A` | `3.3902` | `39.22` | `54.80` |
| `P2.3 + gated + projected-norm + residual-renorm + standard` | `3.4643` | `3.01` | `203.00` |
| frozen `A + M` | `3.5051` | `0.63` | `679.59` |
| `P2.0 + scaled + projected + pre-norm-only + standard` | `3.5306` | `5.48` | `136.30` |
| `P2.0 + plain + projected + pre-norm-only + standard` | `3.5505` | `5.64` | `171.05` |
| incumbent `P2` | `3.6251` | `4.61` | `183.05` |

What changed:

* `P2.3` beat frozen `A + M` again on the second-seed longer-budget run
* the two `P2.0` projected variants remained attractive efficiency contenders
* but frozen `A` unexpectedly won the second-seed longer-budget surface overall

Current read after second-seed confirmation:

* `P2.3` remains the strongest hybrid contender against frozen `A + M`
* but the seed-43 result is mixed enough that we should **not** freeze a new
  contender baseline yet
* the next clean gate is one more confirmation pass that explains or stabilizes
  the surprising `A` rebound on the longer-budget surface

### Frozen FineWeb Longer-Budget Confirmation

After discovering that the earlier docs-backed smoke surface was drifting, the
same leader set was replayed on the frozen FineWeb stage0 canary default
surface.

Surface:

* frozen FineWeb canary JSONL split
* isolated top-level runs
* `64` steps
* `8` eval batches

Seed `42`:

| Variant | Final loss | Train tok/s | RSS delta MB |
| --- | ---: | ---: | ---: |
| `P2.3 + gated + projected-norm + residual-renorm + standard` | `3.5490` | `2.98` | `220.78` |
| frozen `A + M` | `3.5502` | `0.61` | `638.25` |
| `P2.0 + plain + projected + pre-norm-only + standard` | `3.5678` | `5.50` | `161.11` |
| incumbent `P2` | `3.5737` | `4.13` | `187.70` |
| `P2.0 + scaled + projected + pre-norm-only + standard` | `3.6047` | `4.77` | `174.86` |
| frozen `A` | `3.6288` | `38.76` | `56.92` |

Seed `43`:

| Variant | Final loss | Train tok/s | RSS delta MB |
| --- | ---: | ---: | ---: |
| frozen `A` | `3.5161` | `38.63` | `58.66` |
| frozen `A + M` | `3.5195` | `0.61` | `696.06` |
| `P2.3 + gated + projected-norm + residual-renorm + standard` | `3.5424` | `2.85` | `221.66` |
| `P2.0 + scaled + projected + pre-norm-only + standard` | `3.5460` | `5.39` | `171.22` |
| `P2.0 + plain + projected + pre-norm-only + standard` | `3.5814` | `5.56` | `164.72` |
| incumbent `P2` | `3.6011` | `3.58` | `202.05` |

What changed:

* the frozen FineWeb surface is much tighter than the drifting docs surface
* `P2.3` no longer shows a stable quality win over frozen `A + M`
* `P2.0` projected variants remain attractive efficiency contenders
* frozen `A` is much more competitive on this tiny frozen corpus than it looked
  on the mutable docs surface

Current read on the frozen benchmark surface:

* no contender is yet frozen as the new Path 1 winner
* `P2.3` remains the strongest hybrid-quality contender
* `P2.0 + plain/projected` and `P2.0 + scaled/projected` remain the strongest
  efficiency contenders
* but the benchmark story is now “promising but unresolved,” not “P2.3 has
  clearly beaten frozen `A + M`”

---

## Exit Gate

This phase remains active until:

* one primitive-quality winner is chosen on logged evidence
* or the primitive-quality ladder plateaus and a larger architectural interface hypothesis becomes the leading ambiguity
