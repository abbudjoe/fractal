# Export, Load, And Failure-Snapshot Contract

This spec defines the next buildout of model export, model load, and failure-snapshot
behavior for `fractal`.

This is not a convenience feature.

It is a control-plane requirement for:
- reproducibility
- post-run evaluation
- promotion safety
- forensic debugging
- safe scaling into Stage 1 and beyond

Use this alongside:
- [AGENTS.md](/Users/joseph/fractal/AGENTS.md)
- [ENGINEERING.md](/Users/joseph/fractal/ENGINEERING.md)
- [stage1-weight-export-contract.md](/Users/joseph/fractal/docs/stage1-weight-export-contract.md)
- [diagnostics-surface-contract.md](/Users/joseph/fractal/docs/specs/diagnostics-surface-contract.md)

## Why This Exists

The project now has a typed Stage 1 export/load baseline for:
- `BurnBin`
- `best`
- `final`

That is necessary, but not sufficient.

The next step is to round out the contract so the runtime can distinguish clearly between:
- checkpoint state for resumability
- exported model artifacts for promotion and reuse
- failure snapshots for crash forensics

These are different owned concerns and must remain different owned concerns.

## Doctrine Requirements

This feature must follow [AGENTS.md](/Users/joseph/fractal/AGENTS.md) and
[ENGINEERING.md](/Users/joseph/fractal/ENGINEERING.md).

That means:
- no bandaids
- no “checkpoint directory is the API” shortcuts
- no checkpoint/export/failure-snapshot conflation
- no wrapper-only failure artifact hacks
- no best-effort ambiguous loads
- no hidden artifact completeness rules

Required design rules:
- export policy, load policy, and failure-snapshot policy must be typed
- checkpointing, export, and failure snapshots must remain separate owned surfaces
- unsupported formats or phases must fail explicitly during validation
- required artifacts must be surfaced explicitly in structured run artifacts
- every failure-path feature must land with regression coverage

## Non-Goals

This spec does not require:
- cloud registry publication
- every format to be executable immediately
- quantized training support
- generic debugger integration

It does require a clean architecture that can support those later without rewrites.

## Current Baseline

Current executable baseline:
- typed `WeightExportPolicy`
- typed `WeightExportArtifact`
- production `BurnBin` export/load seam
- `best` and `final` export phases
- collision-safe export roots

Current missing contract:
- typed failure snapshots
- explicit snapshot artifact lineage
- clear runtime behavior on panic/OOM/fault
- broader format/phase expansion under one durable architecture

## Ownership Rules

### Checkpointing owns resumable training state

Checkpointing should continue to own:
- model resume state
- optimizer state
- runtime counters
- restart lineage

Checkpointing is for restart.

It is not the promotion/export API.

### Export owns portable promoted model artifacts

Export should own:
- explicit artifact creation
- export phase selection
- export format selection
- promotion-facing lineage
- typed import/load validation

Export is for promotion and reuse.

It is not the crash-forensics API.

### Failure snapshots own crash-time forensics

Failure snapshots should own:
- best-effort capture on panic/fault/error
- snapshot metadata explaining capture completeness
- optional model/runtime/diagnostic artifact preservation
- failure lineage and last-known boundary

Failure snapshots are for forensics.

They are not resumable checkpoints and not promoted model artifacts.

## Required Type Shape

The exact names may differ, but the architecture must preserve this shape:

```rust
enum WeightExportFormat {
    BurnBin,
    SafeTensors,
    Quantized { precision: QuantizedPrecisionKind },
}

enum WeightExportPhase {
    Best,
    Final,
    Latest,
    FailureSnapshot,
}

struct WeightExportPolicy {
    format: WeightExportFormat,
    phases: Vec<WeightExportPhase>,
    required: bool,
}

struct FailureSnapshotPolicy {
    enabled: bool,
    required: bool,
    capture_model_weights: bool,
    capture_runtime_state: bool,
    capture_diagnostics_tail: bool,
}

struct FailureSnapshotArtifact {
    kind: FailureSnapshotKind,
    completeness: SnapshotCompleteness,
    path: String,
    metadata_path: String,
    contract: FailureSnapshotContract,
}
```

The key rule is that failure snapshots may reuse shared serialization primitives,
but they must not be modeled as just another checkpoint slot.

## Export And Load Buildout

The next implementation slice must preserve the current clean baseline and add:

1. Typed export completeness signaling
- required exports missing must be first-class artifact state
- promotion gates must read typed artifact state, not infer from file existence

2. Typed import API
- import/load must remain runtime-owned
- validation must happen before any model load attempt
- mismatch behavior must be explicit and deterministic

3. Stable artifact metadata
- every export artifact must carry:
  - producing commit
  - experiment identity
  - model contract
  - precision assumptions
  - export format
  - export phase

4. Future-safe format abstraction
- `SafeTensors` and quantized formats may remain unsupported for now
- but they must remain represented as typed, explicitly rejected branches

## Failure Snapshot Contract

The runtime must introduce a distinct failure-snapshot surface.

Required behavior:

1. Typed declaration
- failure-snapshot behavior is manifest-visible and validated

2. Best-effort capture with explicit completeness
- snapshots must record what was attempted and what succeeded
- a partial snapshot is allowed if it is labeled explicitly

3. Failure lineage
- snapshot metadata must include:
  - experiment identity
  - producing commit
  - species
  - last successful diagnostic boundary
  - error class
  - whether capture happened before or after panic propagation

4. Separation from export
- a failure snapshot is not automatically a promoted weight export
- if model weights are captured in a failure snapshot, they must still be labeled as failure
  artifacts, not `best` or `final`

## Immediate Failure-Snapshot Scope

The next clean minimum should support:
- runtime-state metadata capture
- diagnostics tail capture
- artifact lineage capture
- optional model weight capture if and only if it can be done through a typed runtime seam
  without pretending that panic-time capture is always guaranteed

For the first implementation:
- runtime metadata snapshot: required
- diagnostics tail snapshot: required
- model weight capture on failure: optional, best-effort, explicitly labeled

## SafeTensors, Quantized Formats, And Deferred Branches

These branches should stay typed and explicit, but do not need to become executable in the
same slice unless they can be implemented cleanly.

That means:
- `SafeTensors`: may remain validation-rejected until there is a real recorder/load path
- quantized export/load: may remain validation-rejected until there is a true quantization
  contract
- `latest` export phase: may remain validation-rejected until lifecycle semantics are nailed down
- `failure-snapshot` as an export phase: should not be enabled by abusing the export pipeline;
  it should become executable only when the separate failure-snapshot contract is real

The design requirement is:
- typed now
- executable only when honest

## Artifact Requirements

Run artifacts must record all of the following:

1. Export state
- requested export policy
- completed exports
- missing required exports

2. Load compatibility metadata
- compatibility contract for each export artifact

3. Failure snapshot state
- whether failure snapshots were enabled
- whether a snapshot was attempted
- whether it succeeded fully, partially, or not at all
- paths to preserved failure artifacts

## Failure Rules

If required export fails:
- the run is artifact-incomplete
- promotion is blocked automatically

If required failure snapshot capture fails:
- the run is failure-snapshot-incomplete
- the run artifact must say exactly which snapshot pieces are missing

Do not silently degrade from:
- required typed artifact capture
to:
- whatever files happened to be on disk

That would violate the contract.

## Testing Requirements

Minimum required coverage:

1. Export/load regression coverage
- matching artifact loads succeed
- mismatched artifact loads fail cleanly
- required export completeness is surfaced explicitly

2. Failure-snapshot validation
- invalid snapshot policy rejected
- unsupported required snapshot branch rejected

3. Failure-snapshot execution
- simulated failure writes snapshot metadata
- snapshot completeness state is recorded correctly
- diagnostics tail is preserved when requested

4. Separation invariants
- checkpoint presence does not satisfy export requirements
- export presence does not satisfy failure-snapshot requirements

## Acceptance Gate

This spec is complete only when:
- export/load remains typed and production-owned
- failure snapshots exist as a separate typed runtime surface
- artifact completeness is explicit for both
- failure preservation no longer depends on wrapper improvisation
- tests prove the separation between checkpointing, export, and failure snapshots

## Bottom Line

`fractal` needs three cleanly separated artifact families:
- checkpoints to resume
- exports to promote and reuse
- failure snapshots to debug crashes

If those concerns are blurred, the control plane becomes ambiguous exactly when the system is
under stress.

This spec exists to keep those boundaries explicit before scaling further.
