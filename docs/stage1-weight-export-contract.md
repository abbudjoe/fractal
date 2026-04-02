# Stage 1 Weight Export Contract

This document defines the required model weight export contract that must exist before any `Stage 1` training run is authorized.

This is not optional housekeeping.

It is a control-plane requirement for scaling, debugging, reproducibility, and post-run evaluation.

Use this alongside:
- [stage-0-training-contract.md](/Users/joseph/fractal/docs/stage-0-training-contract.md)
- [next-training-phase.md](/Users/joseph/fractal/docs/next-training-phase.md)
- [experiment-interface.md](/Users/joseph/fractal/docs/experiment-interface.md)
- [program-state.md](/Users/joseph/fractal/docs/program-state.md)

## Why This Exists

Today the runtime has internal checkpoint persistence.

That is not the same thing as a durable export contract.

Before `Stage 1`, the system must support explicit model artifact export with:
- typed format selection
- typed phase selection
- explicit manifest/artifact recording
- explicit load/import semantics

Without that, longer runs remain too hard to debug, promote, compare, and recover.

## Doctrine Requirements

This feature must follow the repo doctrine in [ENGINEERING.md](/Users/joseph/fractal/ENGINEERING.md) and [AGENTS.md](/Users/joseph/fractal/AGENTS.md).

That means:
- no bandaids
- no hidden checkpoint/export coupling
- no “export if a side effect happened to exist” behavior
- no stringly typed format flags scattered across wrapper, runtime, and reporting code
- no duplicated policy surfaces that can silently diverge

Required design rules:
- export policy must be a typed runtime contract
- checkpoint persistence and export persistence must be separate owned concerns
- artifact reporting must consume the typed export contract, not reconstruct it from paths
- unsupported formats must fail explicitly during config validation
- every bug fix must land with regression coverage

## Non-Goals

This contract does not require:
- changing the current checkpoint system into a different storage engine
- adding quantized export immediately
- exporting every intermediate checkpoint by default
- remote model registry publishing

It does require explicit structured model export for promoted training phases.

## Required Outcomes Before Stage 1

Before `Stage 1`, the runtime must support all of the following:

1. Explicit export policy
- the experiment/runtime contract declares which export artifacts are required
- export is not inferred from checkpoint retention

2. Structured model weight artifacts
- model weights can be exported as a proper structured file
- at minimum, one portable structured format must be supported as the canonical Stage 1 export

3. Explicit import/load path
- exported artifacts can be loaded intentionally
- load behavior is typed and validated against model contract

4. Artifact lineage
- every exported model artifact is recorded in structured run artifacts with:
  - format
  - phase
  - path
  - producing commit
  - experiment identity
  - compatibility contract

5. Promotion safety
- `best` and `final` model exports are available as first-class artifacts for promoted runs

## Required Type Shape

The runtime must introduce a typed export surface.

Recommended shape:

```rust
enum WeightExportFormat {
    BurnBin,
    SafeTensors,
}

enum WeightExportPhase {
    Latest,
    Best,
    Final,
    FailureSnapshot,
}

struct WeightExportPolicy {
    format: WeightExportFormat,
    phases: Vec<WeightExportPhase>,
    required: bool,
}

struct WeightExportArtifact {
    format: WeightExportFormat,
    phase: WeightExportPhase,
    path: String,
    contract: WeightExportContract,
}
```

The exact names may differ, but the architecture must preserve the same ownership:
- config owns policy
- runtime owns execution
- artifacts own reporting

## Ownership Rules

### Checkpointing owns resumable training state

Checkpointing should continue to own:
- model resume state
- optimizer state
- runtime counters
- restart lineage

### Export owns portable model artifacts

Weight export should own:
- explicit model artifact creation
- export format selection
- export file naming
- export compatibility metadata

### Reporting owns structured surfacing

Artifact reporting should own:
- recording exported artifact metadata
- marking required exports as present or missing
- surfacing export lineage in the run artifact

No other layer should reconstruct export semantics from filenames.

## Minimum Stage 1 Contract

Before `Stage 1`, the system must support:
- `best` export
- `final` export
- structured artifact recording for both
- typed format selection
- typed validation

My recommendation:
- keep current internal Burn checkpoint persistence
- add canonical external export support for:
  - `BurnBin` as the immediate compatibility baseline
  - `SafeTensors` as the preferred portable structured format if Burn integration is stable enough

If `SafeTensors` cannot be implemented cleanly before Stage 1, then:
- `BurnBin` may be the temporary canonical export
- but the contract must still be typed as a format abstraction, not hardwired to Burn internals

That keeps the architecture extensible for future:
- `SafeTensors`
- quantized export
- `int8`
- `int4`
- `1-bit`

## Failure Rules

If a run declares a required weight export and export fails:
- the run must be marked as artifact-incomplete
- the failure must be explicit in the structured run artifact
- promotion to Stage 1 or beyond must be blocked automatically if required exports are missing

Do not silently degrade:
- from export failure to “checkpoint exists, so good enough”

That would violate the contract.

## Manifest And Artifact Requirements

The experiment/control plane must be able to declare and record export policy.

Required manifest-visible fields:
- export format
- requested phases
- whether export is required

Required artifact-visible fields:
- exported files
- missing required files
- export contract metadata
- load compatibility metadata

## Load Compatibility Contract

An exported weight artifact must carry enough metadata to validate:
- architecture kind
- hidden dimension
- vocab size
- max recursion depth
- primitive variant
- precision/export format assumptions
- producing commit and experiment identity

Loading must fail explicitly on contract mismatch.

Do not allow best-effort ambiguous loads.

## Testing Requirements

This feature is not complete unless it lands with focused tests.

Minimum required coverage:

1. Export policy validation
- invalid format rejected
- invalid phase rejected
- required export with unsupported backend rejected

2. Export execution
- `best` export writes a real file
- `final` export writes a real file
- artifact metadata records both

3. Load compatibility
- exported weights can be loaded when contract matches
- load fails cleanly when contract mismatches

4. Failure behavior
- required export failure marks the run artifact incomplete
- missing export blocks promotion

5. Regression test
- a run with checkpoints but missing required exports must not be treated as export-complete

## Acceptance Gate

Stage 1 is blocked until all of the following are true:
- export policy is typed and manifest-visible
- export execution is separate from checkpoint retention
- structured run artifacts record export lineage explicitly
- `best` and `final` exports are produced for promoted runs
- load compatibility is validated in code
- tests cover the contract and failure modes

## Implementation Guidance

Preferred implementation order:

1. add typed export policy to the runtime/experiment contract
2. add export artifact reporting types
3. implement one clean export execution seam
4. implement load validation
5. add promotion gating on required export completeness

Do not:
- bolt export flags into wrapper scripts first
- treat checkpoint directories as the export API
- scatter format assumptions across the runtime

The right shape is:
- one typed policy
- one execution seam
- one reporting seam
- one load seam

## Bottom Line

Before `Stage 1`, the project must have:
- explicit model weight export
- explicit model weight import/validation
- explicit artifact lineage

Internal checkpointing is necessary.

It is not sufficient.
