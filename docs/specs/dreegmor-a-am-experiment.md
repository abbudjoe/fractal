# Minimal DREEGMOR over `A` / `A + M`

Status: exploratory scaffold

This experiment is a separate surface from the main `v3a` / Path 1 proving line.

Comparative reads on this surface should follow the shared rubric in
[`/Users/joseph/fractal-worktrees/goe-a-am-experiments/docs/specs/dreegmor-experiment-rubric.md`](/Users/joseph/fractal-worktrees/goe-a-am-experiments/docs/specs/dreegmor-experiment-rubric.md).

`DREEGMOR` is the shorthand name for this current minimal Graph-of-Experts-style
scaffold:

* dense routed expert ensemble with graph-mixed router

It exists to answer one narrow question:

* does a tiny typed DREEGMOR controller over the already-frozen `A` and `A + M`
  backbones show any sign of usefulness on the shared byte-level smoke lane,
  without touching `P2` or introducing external memory?

## Non-goals

This experiment is **not**:

* evidence for Path 1 promotion
* a replacement for the frozen `A` vs `A + M` baseline story
* a `P2` experiment
* a sparse-MoE or external-memory architecture
* a parameter-matched comparison

## Fixed Contract

Held fixed against the existing `A` / `A + M` setup:

* same checked-in FineWeb canary default corpus
* same byte-level vocabulary contract
* same hidden width, head count, local window, and layer schedules as the
  frozen base backbones
* same smoke-train harness, eval surface, and report shape where possible
* no `P2`
* no memory sidecar

Intentionally changed only for the DREEGMOR variants:

* channel count is fixed at `2`
* each channel is a full copy of the selected base backbone
* routing is dense, not sparse: both experts execute on every token
* the controller structure is restricted to a small typed ladder:
  * `uniform-average`
  * `routed-no-graph-mix`
  * `routed-graph-mix`
* a tiny token-local controller mixes expert logits when routing is enabled
* graph smoothing, when enabled, is one bounded step across the two expert
  channels

Because both experts always execute, this is **not** an apples-to-apples
parameter or compute match to the single-backbone baselines. That is why this
surface is exploratory only.

## Variants

The runner supports four fixed variants:

* `A`
  * existing `attention-only` baseline
* `A + M`
  * existing `reference-ssm-hybrid` baseline
* `DREEGMOR(A)`
  * two attention-only experts plus the minimal controller
* `DREEGMOR(A + M)`
  * two reference-SSM experts plus the minimal controller

## Current Implementation

Implemented:

* typed GoE variant/controller contract in
  [`/Users/joseph/fractal-worktrees/goe-a-am-experiments/fractal-core/src/hybrid_attention/goe.rs`](/Users/joseph/fractal-worktrees/goe-a-am-experiments/fractal-core/src/hybrid_attention/goe.rs)
* `DREEGMOR(A)` and `DREEGMOR(A + M)` model builders over the existing frozen backbones
* typed DREEGMOR structure ladder:
  * `uniform-average`
  * `routed-no-graph-mix`
  * `routed-graph-mix`
* dedicated DREEGMOR smoke-train/report/ledger surface in
  [`/Users/joseph/fractal-worktrees/goe-a-am-experiments/fractal-eval-private/src/hybrid_attention_goe.rs`](/Users/joseph/fractal-worktrees/goe-a-am-experiments/fractal-eval-private/src/hybrid_attention_goe.rs)
* separate experiment binary in
  [`/Users/joseph/fractal-worktrees/goe-a-am-experiments/src/bin/dreegmor-a-am-experiment.rs`](/Users/joseph/fractal-worktrees/goe-a-am-experiments/src/bin/dreegmor-a-am-experiment.rs)
* separate artifact root: `artifacts/dreegmor-a-am-experiment/`
* separate optional ledger: `docs/dreegmor-a-am-results-ledger.jsonl`
* routing summaries in DREEGMOR reports:
  * mean pre-graph channel weights
  * mean channel weights
  * winner counts
  * active channel count
  * route entropy
  * winner margin
  * graph-adjustment L1
  * learned edge-mix fraction

Still intentionally omitted:

* sparse top-k routing
* expert-specific message passing beyond the single bounded smoothing step
* parameter matching to `A` or `A + M`
* any `P2` lane
* any external memory or retrieval path
* integration into the main `v3a-hybrid-attention-matrix` binary

## How To Run

Minimal four-variant smoke run:

```bash
cargo run --bin dreegmor-a-am-experiment -- --steps 1 --eval-batches 1 --output table
```

Slightly less tiny run with a separate ledger:

```bash
cargo run --bin dreegmor-a-am-experiment -- --steps 8 --eval-batches 2 --ledger-path default --run-label dreegmor-a-am-smoke-seed42 --output table
```

Full-pass sizing on the shared corpus split:

```bash
cargo run --bin dreegmor-a-am-experiment -- --variant dreegmor-a --full-train-pass --full-eval-pass --output table
```

Structure ablations over `DREEGMOR(A)`:

```bash
cargo run --bin dreegmor-a-am-experiment -- --variant dreegmor-a --steps 8 --eval-batches 2 --goe-routing-mode uniform-average --goe-graph-topology none
cargo run --bin dreegmor-a-am-experiment -- --variant dreegmor-a --steps 8 --eval-batches 2 --goe-routing-mode token-local-router --goe-graph-topology none
cargo run --bin dreegmor-a-am-experiment -- --variant dreegmor-a --steps 8 --eval-batches 2 --goe-routing-mode token-local-router --goe-graph-topology two-node-line
```

Single-variant runs:

```bash
cargo run --bin dreegmor-a-am-experiment -- --variant a --steps 8 --eval-batches 2
cargo run --bin dreegmor-a-am-experiment -- --variant a-plus-m --steps 8 --eval-batches 2
cargo run --bin dreegmor-a-am-experiment -- --variant dreegmor-a --steps 8 --eval-batches 2
cargo run --bin dreegmor-a-am-experiment -- --variant dreegmor-a-plus-m --steps 8 --eval-batches 2
```

Artifacts land under:

* `artifacts/dreegmor-a-am-experiment/<unix-seconds>/`

Detailed per-variant reports are written as:

* `a/report.json`
* `a-plus-m/report.json`
* `dreegmor-over-a-<structure>/report.json`
* `dreegmor-over-a-plus-m-<structure>/report.json`

## Falsifiable Read

The narrow hypothesis is:

* if `DREEGMOR(A)` or `DREEGMOR(A + M)` does not beat its corresponding
  single-backbone baseline often enough to justify its extra compute and
  parameter cost, this line should be discarded quickly

The experiment should therefore be judged on:

* final loss vs its own base family
* routing behavior actually using both channels
* whether any gain survives the obvious extra-cost penalty

For memory and throughput comparisons, use isolated single-variant runs rather
than shared-process `--variant all` sweeps.
