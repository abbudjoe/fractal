# Harness Hardening Checklist

This document defines the hardening work required before the next winner-lane bakeoff is considered authoritative.

It is a control-plane document, not a backlog dump.

The goal is simple:
- every run produces durable structured results
- every failure is classified
- every ranking can be traced back to an apples-to-apples manifest
- no cloud result is lost because a pod exits before local capture finishes

Use this alongside:
- [harness-doctrine.md](/Users/joseph/fractal/docs/harness-doctrine.md)
- [promotion-policy.md](/Users/joseph/fractal/docs/promotion-policy.md)
- [primitive-tracker.md](/Users/joseph/fractal/docs/primitive-tracker.md)

## Scope

This checklist applies to:
- local tournament runs
- RunPod CUDA runs
- single-species runs
- full tournament runs

It does not apply to tokenizer-track-only work unless tokenizer work consumes the same shared run artifacts.

## Hardening Goals

### 1. Durable Run Artifact

Every completed or failed run must produce one structured artifact with:
- variant name
- species id
- preset
- lane
- backend
- execution mode
- seed
- commit SHA
- pod id if any
- timeout policy
- config snapshot
- per-phase timings
- final metrics if available
- failure classification if not successful

The artifact must be written even when the run fails after training begins.

Acceptance:
- one local JSON artifact exists per run
- the artifact survives pod exit
- artifact writing does not depend on tracker updates

### 2. Failure Classification

Every non-success run must be classified as exactly one of:
- `numeric-failure`
- `train-timeout`
- `eval-constrained`
- `low-signal`
- `runtime-cost`
- `infra-failure`

Classification must be explicit in structured output, not inferred later from logs.

Acceptance:
- no failed run is emitted without a classification
- timeout paths distinguish train timeout from eval timeout
- infra/setup failures do not masquerade as primitive failures

### 3. Phase-Level Observability

Every run must emit structured phase progress for:
- training start
- training progress checkpoints
- training done
- stability start/done
- perplexity start/done
- ARC/speed start/done
- final completion or failure

Acceptance:
- the logs are sufficient to identify the slow phase
- phase timing is present in the durable artifact

### 4. Run Manifest / Reproducibility

Every run must emit a manifest that captures the exact experimental contract:
- variant identity and version
- lever type
- preset name
- lane
- backend
- seed
- config values
- commit SHA
- host or pod metadata

Acceptance:
- every ranked row can be traced to a manifest
- repeated runs with the same manifest are reproducible in principle

### 5. Apples-to-Apples Ranking Discipline

Rankings must distinguish:
- authoritative same-preset comparisons
- advisory cross-preset comparisons

This distinction must be visible in reporting output and/or the saved artifact.

Acceptance:
- the harness can mark whether a result is authoritative for leaderboard use
- cross-preset results are not silently treated as equivalent

### 6. Tracker Integration

The tracker remains human-reviewed, but result capture should not depend on copying chat text.

At minimum the harness must produce tracker-ready rows with:
- variant name
- preset
- status
- metrics
- failure classification
- short next-action note

Acceptance:
- a completed run prints or writes a tracker-ready block
- tracker-ready output is derived from the structured artifact, not duplicated logic

### 7. RunPod Preservation

Cloud runs must preserve:
- remote stdout/stderr
- final structured artifact
- latest manifest

before the pod exits or is stopped.

Acceptance:
- a local watcher or sync step copies final artifacts off the pod
- losing the final streamed lines does not lose the result
- `--keep-pod` is not required to preserve results

### 8. Eval Budget Control

Eval-heavy failures must be controllable without mutating the primitive.

The harness must keep train and eval budgets distinct enough to classify:
- train failure
- eval-constrained failure

Acceptance:
- config surface exposes eval budget distinctly from train budget
- tracker and artifact record whether a timeout happened during train or eval

## Implementation Workstreams

### Workstream A: Core Runtime Artifact Contract

Own:
- structured run artifact type
- manifest type
- failure classification type
- phase timing capture
- emission from the core run path

Primary files:
- `fractal-core/src/registry.rs`
- shared artifact types in `fractal-core/src/lifecycle.rs`

### Workstream B: Reporting / Tracker Surface

Own:
- tracker-ready output derived from artifacts
- same-preset vs cross-preset labeling
- CLI/reporting integration

Primary files:
- `examples/tournament.rs`
- `src/primitive_tracker.rs`
- `src/lib.rs`

### Workstream C: Cloud Preservation

Own:
- RunPod wrapper preservation logic
- final artifact/log sync
- local storage layout for preserved artifacts

Primary files:
- `scripts/runpod-tournament.sh`

## Minimum Bar Before Next Winner Bakeoff

The next 3-seed winner bakeoff should not start until all of the following are true:
- durable structured artifacts exist
- failures are classified
- phase timings are preserved
- authoritative same-preset labeling exists
- RunPod preservation is automatic

## Out of Scope For This Pass

- new tokenizer features
- new bullpen primitives
- model architecture changes unrelated to result capture or evaluation discipline

## Current Priority Order

1. Durable artifact + manifest
2. Failure classification
3. RunPod preservation
4. Tracker-ready output
5. Authoritative ranking labels
