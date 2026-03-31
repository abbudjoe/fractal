# Experiment Interface v1

This document defines the typed experiment control plane we should build next.

Its purpose is to keep variant identity, runtime surfaces, execution target, and
comparison authority inside one explicit contract instead of scattering them across:
- CLI flags
- presets
- wrapper behavior
- tracker notes
- pod metadata

Use this alongside:
- [harness-doctrine.md](/Users/joseph/fractal/docs/harness-doctrine.md)
- [harness-hardening-checklist.md](/Users/joseph/fractal/docs/harness-hardening-checklist.md)
- [optimization-surfaces.md](/Users/joseph/fractal/docs/optimization-surfaces.md)
- [promotion-policy.md](/Users/joseph/fractal/docs/promotion-policy.md)
- [primitive-tracker.md](/Users/joseph/fractal/docs/primitive-tracker.md)

## Purpose

The experiment interface exists to answer one question cleanly:

What exactly was this run, and is its result allowed to influence the canonical leaderboard?

At current scale, the answer is no longer determined only by the primitive. It also depends on:
- the lane and preset
- the runtime surface policy
- the backend and execution target
- the timeout and preservation contract
- whether the comparison is authoritative or only advisory

So the harness needs one typed experiment object that owns the full run contract.

## Design Goals

The experiment interface must:
- make run identity explicit
- make optimization surfaces explicit
- make comparison authority explicit
- make execution target explicit
- preserve reproducibility and artifact ownership
- prevent accidental apples-to-oranges leaderboard comparisons

It must not:
- replace the primitive tracker
- replace promotion policy
- hide meaningful runtime choices behind wrapper-only defaults
- allow the same experiment to be described differently by CLI, wrapper, and artifact

## Core Rule

One experiment spec must be the canonical answer to:
- what question is being asked
- what primitive or composite is under test
- what budget applies
- what runtime surfaces are active
- where it will execute
- how the result is allowed to be interpreted

If any of those answers must be reconstructed later from multiple files, the control plane is still incomplete.

## Proposed Top-Level Type

The control plane should revolve around a single top-level type:

`ExperimentSpec`

Suggested shape:

```rust
pub struct ExperimentSpec {
    pub experiment_id: ExperimentId,
    pub question: ExperimentQuestion,
    pub variant: VariantSpec,
    pub budget: BudgetSpec,
    pub runtime: RuntimeSurfaceSpec,
    pub comparison: ComparisonContract,
    pub execution: ExecutionTarget,
    pub artifacts: ArtifactPolicy,
}
```

This object should be serializable, preserved in manifests, and constructible from CLI or a manifest file.

## ExperimentSpec Sections

### 1. `experiment_id`

Purpose:
- give every run a stable identity separate from pod id or wrapper session id

Required fields:
- logical name
- generated run id
- branch
- commit SHA
- creation timestamp

Rules:
- pod id is execution metadata, not experiment identity
- retries must preserve logical experiment identity but get distinct run ids

### 2. `question`

Purpose:
- force every run to answer one question only

Suggested fields:
- human-readable question
- lane intent: `benchmark`, `bullpen`, `validation`, `winner`
- decision intent:
  - `promote`
  - `hold`
  - `retire`
  - `benchmark`
  - `optimize`

Rules:
- if the question cannot fit in one sentence, the run is too broad
- tracker notes should derive from this section, not from ad hoc wrapper prose

### 3. `variant`

Purpose:
- define exactly what model hypothesis is under test

Suggested fields:
- variant name
- base primitive
- lever type
- lever version
- composite inner variant if any
- composite depth policy if any

Rules:
- variant naming must still obey the current naming convention
- composite inner variants must be explicit here, not inferred from wrapper strings
- benchmark variants stay frozen even when wrapped in ExperimentSpec

### 4. `budget`

Purpose:
- define the scientific workload

Suggested fields:
- lane
- preset name
- seed
- train budget
  - epochs or train steps
  - train batch size
  - learning rate
- eval budget
  - perplexity eval batches
  - ARC eval batches
- recursion budget
  - max depth
  - stability depth
- timeout policy

Rules:
- presets may populate this section, but the manifested values are the final truth
- explicit overrides must be captured here, not left in wrapper state

### 5. `runtime`

Purpose:
- define the optimization surfaces that were active

This section should directly own the surfaces from [optimization-surfaces.md](/Users/joseph/fractal/docs/optimization-surfaces.md):
- eval backend policy
- batching policy
- execution policy
- buffer reuse policy
- benchmark mode
- backend policy

Rules:
- all runtime surfaces default to the current conservative behavior
- any enabled non-default surface must appear here
- if runtime differs, the comparison contract must account for it

### 6. `comparison`

Purpose:
- define whether the result can influence the canonical leaderboard

Suggested fields:
- authority:
  - `authoritative`
  - `advisory`
- same-preset requirement
- same-runtime-surface requirement
- frozen-commit requirement
- same-backend requirement
- promotion policy target

Rules:
- this section is the guard against apples-to-oranges comparisons
- authoritative runs should require matched preset, backend, and runtime surfaces
- mixed-commit runs should automatically downgrade to `advisory` unless explicitly approved otherwise

### 7. `execution`

Purpose:
- define where and how the run executes

Suggested fields:
- target:
  - `local`
  - `runpod`
- backend:
  - `metal`
  - `cuda`
  - future typed backends
- pod class or local hardware class
- preservation mode
- retry policy

Rules:
- wrapper concerns belong here, not in free-form wrapper arguments
- execution target must never silently change the scientific meaning of the run

### 8. `artifacts`

Purpose:
- define what must be preserved

Suggested fields:
- manifest required
- structured artifact required
- final log required
- tracker-ready output required
- sync policy

Rules:
- a completed run without artifacts is a control-plane failure
- an infra failure must still preserve the experiment spec and failure class

## Typed Supporting Objects

Suggested supporting types:

```rust
pub enum ComparisonAuthority {
    Authoritative,
    Advisory,
}

pub enum LaneIntent {
    Benchmark,
    Bullpen,
    Validation,
    Winner,
}

pub enum DecisionIntent {
    Promote,
    Hold,
    Retire,
    Benchmark,
    Optimize,
}

pub enum ExecutionBackend {
    Metal,
    Cuda,
}

pub enum ExecutionTargetKind {
    Local,
    RunPod,
}
```

The point is not the exact enum names. The point is that these choices become typed and manifest-visible.

## Non-Negotiable Invariants

The experiment interface must enforce:
- one experiment spec maps to one scientific question
- one run artifact maps back to one experiment spec
- one leaderboard row knows whether it is authoritative or advisory
- one runtime surface change cannot masquerade as a primitive change
- one preset name cannot hide different effective budgets across runs

If any of these fail, the interface has not solved the control-plane problem.

## Relationship to Existing Documents

This interface does not replace current docs. It binds them together.

- [promotion-policy.md](/Users/joseph/fractal/docs/promotion-policy.md)
  defines whether a result earns promotion or retirement
- [primitive-tracker.md](/Users/joseph/fractal/docs/primitive-tracker.md)
  records the reviewed state of the field
- [optimization-surfaces.md](/Users/joseph/fractal/docs/optimization-surfaces.md)
  defines runtime knobs the experiment can activate
- [harness-hardening-checklist.md](/Users/joseph/fractal/docs/harness-hardening-checklist.md)
  defines the artifact and observability minimums
- [next-run-manifests.md](/Users/joseph/fractal/docs/next-run-manifests.md)
  should eventually be expressible directly as `ExperimentSpec` instances

## Canonical Examples

### Example A: Winner-Lane Bakeoff Row

This is the ideal fully controlled case:
- variant: `p1_fractal_hybrid_v1`
- lane: `winner`
- preset: `full_medium_stress`
- seed: `42`
- runtime surfaces: all defaults
- backend: `cuda`
- comparison authority: `authoritative`
- frozen commit: required

Interpretation:
- this row is allowed to influence the canonical leaderboard

### Example B: Eval-Constrained Rerun

This is a controlled exception:
- variant: `b2_stable_hierarchical_v1`
- lane: `validation`
- preset: `minimal_stress_lane`
- explicit reduced eval budget
- runtime surfaces otherwise default
- comparison authority: `advisory` unless matched against the same reduced-eval contract

Interpretation:
- useful for classifying the failure mode
- not directly interchangeable with normal validation rows

### Example C: Optimization Validation Run

This is a systems experiment, not a primitive bakeoff:
- variant: `p1_contractive_v1`
- lane: `benchmark`
- preset: `full_medium_stress`
- runtime surface: `length_bucketed`
- comparison authority: `advisory` until matched same-preset reruns confirm no scientific drift

Interpretation:
- this measures runtime surface effect, not primitive superiority

## Recommended Storage Model

The interface should exist in two equivalent forms:

1. typed Rust structs in the shared control plane
2. serialized manifest form in the preserved run output

Likely implementation homes:
- `fractal-core/src/lifecycle.rs`
- `src/run_artifacts.rs`
- `examples/tournament.rs`

Wrapper scripts should consume or emit this spec, not invent parallel state.

## CLI Relationship

The CLI should remain usable, but it should become a constructor for `ExperimentSpec`, not a second control plane.

That means:
- flags populate `ExperimentSpec`
- presets populate `BudgetSpec`
- runtime flags populate `RuntimeSurfaceSpec`
- execution flags populate `ExecutionTarget`
- printed manifests should show the fully resolved spec

The manifest, not the original flag list, should be the authoritative run contract.

## Minimum v1 Implementation Scope

The first implementation pass should do only this:

1. add typed `ExperimentSpec` and supporting structs
2. emit them in manifests and artifacts
3. mark comparison authority explicitly
4. record active runtime surfaces explicitly
5. keep existing preset and CLI behavior working

Do not try to solve:
- GUI orchestration
- scheduler design
- multi-run workflow automation
- new primitive logic

## Success Criteria

This interface is successful when:
- every run can be described by one manifest object
- the same run is no longer described differently by CLI, wrapper, and tracker prose
- authoritative vs advisory status is visible without detective work
- optimization surfaces become manageable without corrupting primitive comparisons
- a future 1.5B-scale training run can be reasoned about as one explicit experiment contract

## Recommended Next Step

After the live bakeoff is fully preserved and classified:

1. implement `ExperimentSpec` as typed shared state
2. thread it through manifest emission first
3. keep default runtime surfaces unchanged
4. only then use it to power the first optimization branch:
   - `codex/opt-eval-backend-split`

This keeps the interface architectural, additive, and disciplined.
