# Stage 0 Training Contract

This document defines the immediate next launch surface for scaled training.

It is the de-risking bridge between:
- tournament-scale primitive selection
- real top-cohort pretraining investment

Use this alongside:
- [next-training-phase.md](/Users/joseph/fractal/docs/next-training-phase.md)
- [primitive-tracker.md](/Users/joseph/fractal/docs/primitive-tracker.md)
- [promotion-policy.md](/Users/joseph/fractal/docs/promotion-policy.md)
- [experiment-interface.md](/Users/joseph/fractal/docs/experiment-interface.md)
- [program-state.md](/Users/joseph/fractal/docs/program-state.md)

## Purpose

Stage 0 exists to prove that the training harness, evaluation path, and top-cohort ordering survive the first real FineWeb pretraining pass before larger OPEX is committed.

It is not a flagship training run.

It is a gate.

## Locked Cohort

- `p1_fractal_hybrid_composite_v1`
- `p1_fractal_hybrid_v1`
- `p1_contractive_v1`

No additional variants are admitted to Stage 0.

## Locked Data Contract

Training corpus:
- `FineWeb`

Rules:
- one frozen shared corpus for all 3 variants
- one frozen tokenizer for all 3 variants
- one frozen validation slice for all 3 variants
- no per-variant curriculum
- no domain-specific augmentation
- no ARC-style training data mixed into pretraining

## Locked Fairness Rule

Equality is defined as:
- equal `tokens seen`

Stage 0 does not equalize by wall-clock or cost.

Those remain secondary systems measurements.

## Locked Hardware Contract

GPU class:
- `H200`

Topology:
- `3` concurrent single-GPU runs
- `1` GPU per variant
- no distributed training in Stage 0

Rationale:
- strong memory and bandwidth headroom
- lower risk of memory-bound instability during first real pretraining
- keeps the comparison simple by avoiding multi-GPU training variables

## Locked Budget

Token budget:
- `100M tokens per variant`

Total Stage 0 suite:
- `300M tokens`

Training budget is fixed before launch and may not be widened mid-run.

## Model Contract

All 3 variants must share:
- the same executable model architecture
- the same parameter budget target
- the same tokenizer
- the same context length
- the same optimizer family
- the same precision policy
- the same checkpoint cadence
- the same evaluation cadence

Only the recursive core may differ.

### Locked Stage 0 Runtime Architecture

Stage 0 must describe the runtime we can actually execute today, not a desired future shell.

Executable architecture:
- `recursive-kernel-v1`
- shared hidden state width `d_model = 1024`
- shared recursion budget `max_recursion_depth = 16`
- router enabled
- current executable path is `embedding -> recursive rule -> router -> output`

Rules:
- the runtime architecture is identical for all 3 variants
- the primitive may self-organize only within this fixed executable envelope
- no manifest, doc, or launch wrapper may claim a shared outer scaffold that the runtime does not implement
- any future shared-shell experiment must land as a new typed architecture kind before it can become canon

This keeps Stage 0 focused on primitive behavior without pretending the control plane can execute a scaffold it does not yet own.

### Locked Context Contract

Training context length:
- `2048`

Context probe evaluation windows:
- `256`
- `512`
- `1024`
- `2048`

Rules:
- the training allowance is fixed at `2048`
- shorter context windows are used as evaluation probes, not as the main training budget
- Stage 0 should test whether the top cohort uses context efficiently, not starve it artificially

### Locked Tokenizer Contract

Tokenizer baseline:
- one frozen shared `32k` SentencePiece-style reference tokenizer

Rules:
- the tokenizer is shared across all 3 variants
- the tokenizer is treated as infrastructure, not as a Stage 0 research variable
- the tokenizer bridge path may be exercised and preserved, but tokenizer-specific features may not change primitive semantics in Stage 0
- bridge packaging is a frozen experiment-owned artifact, not a live corpus-derived side effect
- tokenizer-track work remains a parallel stream and does not redefine this contract mid-run

### Locked Optimization Contract

Optimizer:
- `AdamW`

Schedule:
- linear warmup to peak LR
- cosine decay to a small floor

Exact Stage 0 defaults:
- peak learning rate: `2e-4`
- warmup: `2%` of total tokens
- decay floor: `10%` of peak LR (`2e-5`)
- weight decay: `0.05`
- gradient clipping: global norm `1.0`
- AdamW betas: `(0.9, 0.95)`
- epsilon: `1e-8`

Rules:
- the external optimizer and schedule remain fixed and boring
- the experiment may observe internal primitive self-management, but Stage 0 does not allow primitives to change global optimizer or schedule policy
- if the trainer cannot express these values yet, the control plane must be extended explicitly before launch

## Required Outputs

Every Stage 0 run must emit:
- structured run artifact
- experiment manifest
- wrapper manifest if cloud-run
- checkpoint metadata
- evaluation records
- scientific leaderboard row
- systems-speed row
- ledger entry

## Evaluation Contract

Stage 0 must evaluate:
- perplexity on the frozen validation slice
- ARC on the frozen eval path
- throughput as a separate systems-speed readout

Evaluation cadence:
- at fixed token intervals
- identical across all 3 variants
- plus one final eval at end of run

## What Stage 0 Must Prove

Stage 0 is successful only if it proves all of the following:

1. Training harness stability
- all 3 runs complete end-to-end
- no numeric failure
- no checkpoint corruption
- no artifact/preservation failure

2. Evaluation integrity
- eval cadence runs on schedule
- final metrics are preserved cleanly
- scientific leaderboard and systems-speed outputs both generate correctly

3. Operational predictability
- observed throughput and wall-clock stay within a plausible range of the planning envelope
- no hidden control-plane regressions appear under real training load

4. Scientific continuity
- the top-cohort ranking remains coherent enough to justify a larger comparison spend

## Graduation Criteria To Stage 1

Graduate the cohort to Stage 1 only if:
- all 3 variants complete Stage 0 cleanly
- no variant is classified as `numeric-failure`, `train-timeout`, or `infra-failure`
- checkpoint and resume surfaces are judged reliable enough for a longer run
- evaluation and preservation surfaces remain intact end-to-end
- the top-cohort ranking remains scientifically interpretable

Additional graduation rules by variant:

### `p1_fractal_hybrid_composite_v1`

Must remain one of:
- the Stage 0 scientific leader
- or within `0.02` fitness of the Stage 0 leader

### `p1_fractal_hybrid_v1`

Must remain:
- within `0.03` fitness of the Stage 0 leader
- and still clearly stronger than validation-lane alternates justify displacing it

### `p1_contractive_v1`

Must remain:
- the systems-reference winner on throughput
- and within `0.03` fitness of the Stage 0 scientific leader

If any of these fail, Stage 1 should pause for review before launch.

## Failure Meaning

Stage 0 failures are not embarrassing. They are exactly why Stage 0 exists.

If Stage 0 fails, the result should be one of:
- fix the harness
- fix the control plane
- narrow the top cohort
- revise the scale assumptions

It should not automatically trigger a larger spend.

## Stage 1 Target If Stage 0 Passes

If Stage 0 passes cleanly, the next approved phase is:
- Stage 1
- same top cohort
- same `H200` hardware class
- `500M tokens per variant`

Stage 1 is therefore a graduation, not a fresh debate.

## Explicit Non-Goals

Stage 0 does not:
- decide the final flagship model
- reopen the bullpen
- mutate benchmark behavior
- admit logistic into the top cohort
- mix optimization-surface experiments into the scientific contract

## Success Output

At the end of Stage 0, we should have:
- one authoritative Stage 0 scientific leaderboard
- one Stage 0 systems-speed leaderboard
- one Stage 0 ledger slice
- one clear go/no-go decision for Stage 1
