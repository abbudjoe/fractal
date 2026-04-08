# DREEGMOR Experiment Rubric

Status: required evaluation contract for exploratory DREEGMOR controller runs

This rubric exists to keep exploratory controller work empirically disciplined
without contaminating the main `v3a` / Path 1 proving line.

Use it for:

* `DREEGMOR(A)` / `DREEGMOR(A + M)` controller experiments
* recurrent-router follow-on experiments over the same frozen backbones

Do not use it to promote an exploratory result into the main Path 1 story.

## Core Rule

Each experiment may answer **one** primary question at a time.

Examples:

* does recurrent routing beat one-shot routing on the same frozen experts?
* does expert feedback improve recurrent routing over state-only recurrence?
* does a load-balancing term stabilize recurrent routing enough to justify its
  cost?

An experiment may not simultaneously claim:

* a controller win
* a systems win
* a scale win

unless each claim was part of the pre-run contract and measured directly.

## Pre-run Contract

Before a run, write down:

* the single changed axis
* the fixed axes
* the acceptance rule
* the disallowed conclusions

Every run should therefore answer this template:

1. Hypothesis
   Example: recurrent routing improves final loss over one-shot routing at the
   same expert set and budget.
2. Changed axis
   Example: controller changes from one-shot to recurrent state-only routing.
3. Held fixed
   Example: backbone family, corpus split, data order, optimizer, learning
   rate, step budget, eval budget, backend, channel count.
4. Acceptance rule
   Example: recurrent must beat one-shot on final loss across at least two
   seeds without a disproportionate cost increase.
5. Disallowed conclusions
   Example: this run does not prove sparse expertization, does not prove Path 1
   promotion, and does not prove better large-scale scaling.

## Judgment Columns

Every comparative read must be reported in three columns:

* `Quality`
  * final loss
  * loss delta from initial to final
  * perplexity
* `Cost`
  * train tokens per second
  * overall tokens per second
  * process-memory metric kind
  * process-memory delta
* `Explanation`
  * route entropy
  * winner margin
  * active channel count
  * collapse signs
  * round-to-round controller adjustment

No architecture should be called a win from quality alone if cost regresses
materially.

## Run Discipline

### Smoke Runs

Use tiny runs only to answer:

* does the control plane execute?
* do routing summaries move?
* are the configs and reports honest?

Allowed shortcuts:

* shared-process `--variant all`
* single-seed runs

Disallowed conclusions:

* memory-footprint claims
* throughput claims
* scale claims

### Comparative Runs

When comparing quality, throughput, or memory between variants:

* run one variant per fresh process
* use separate output roots
* keep backend fixed
* keep `--data-seed` fixed unless the experiment is explicitly about data-order
  sensitivity
* compare the same seed set across all variants

This rule is required for footprint claims. Shared-process runs are not valid
for cross-variant memory interpretation because allocator warmup and reused
buffers distort deltas.

### Scaling Runs

Do not increase budget until the mechanism has already earned the next rung.

Recommended ladder:

1. smoke: liveness only
2. short comparative rung: multiple seeds, isolated processes
3. medium comparative rung: same seeds, larger budget
4. controller change only after the mechanism is understood
5. scale increase only after the controller earns the right to scale

## Stop/Go Rules

Stop scaling a variant if:

* it loses on quality and costs more across the chosen seed set
* it wins only through obvious extra-compute tax without a durable quality edge
* it collapses routing or fails to use the intended controller mechanism

Keep investigating, but change the controller instead of scaling, if:

* quality is mixed across seeds
* routing is active but unstable
* cost is clearly worse while quality is only marginally better

Scale the variant only if:

* quality survives multiple seeds
* the mechanism is visibly active
* the result is still competitive after cost is included

## Allowed Conclusions

The strongest allowed conclusion should match the evidence:

* `mechanism live`
  * the control plane executes and measurably changes routing
* `competitive but not justified`
  * quality is near baseline but cost is worse
* `quality tradeoff`
  * quality improves, but cost regresses enough that the result is a tradeoff,
    not a win
* `earned next rung`
  * quality is stable enough across seeds and cost is understandable enough to
    justify the next budget rung

Avoid stronger wording unless the evidence supports it:

* do not say `better architecture` when the result is only a single-seed win
* do not say `more efficient` from shared-process memory numbers
* do not say `scales better` from one budget rung

## Current Branch Defaults

For the current recurrent-router line, the default comparative contract is:

* compare `A`, one-shot `DREEGMOR(A)`, and recurrent `DREEGMOR-Recurrent(A)`
* use isolated processes for throughput and memory reads
* keep backend fixed per rung
* keep `data_seed=fixed` unless explicitly studying data-order sensitivity

The current branch should treat controller changes as the next lever before
larger budget increases when:

* recurrent is real but not yet clearly worth its memory and throughput tax

