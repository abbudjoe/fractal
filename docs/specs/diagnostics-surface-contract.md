# Diagnostics Surface Contract

This spec defines the required runtime diagnostics surface for `fractal`.

This is not a logging cleanup task.

It is a control-plane contract for:
- root-cause debugging
- performance forensics
- launch safety
- deterministic reproduction

Use this alongside:
- [AGENTS.md](/Users/joseph/fractal/AGENTS.md)
- [ENGINEERING.md](/Users/joseph/fractal/ENGINEERING.md)
- [stage-0-training-contract.md](/Users/joseph/fractal/docs/stage-0-training-contract.md)
- [stage1-weight-export-contract.md](/Users/joseph/fractal/docs/stage1-weight-export-contract.md)

## Why This Exists

The current Stage 0 debug loop proved the need for a real diagnostics surface.

The RunPod H200 canary did not fail at launch or batch allocation. It failed deep inside
the first training step after progressing through most of the forward token loop, and the
useful narrowing came from typed probes added to the runtime.

That is the core lesson:
- diagnostics are not incidental printouts
- diagnostics are part of the executable contract

If the system can only explain failures after ad hoc probe edits, the diagnostics surface
is still incomplete.

## Doctrine Requirements

This feature must follow [AGENTS.md](/Users/joseph/fractal/AGENTS.md) and
[ENGINEERING.md](/Users/joseph/fractal/ENGINEERING.md).

That means:
- no bandaids
- no probe behavior hidden only in wrapper scripts
- no `println!` soup as the diagnostics API
- no stringly probe names scattered across runtime layers
- no probe semantics reconstructed from free-form log text
- no diagnostic path that changes model semantics

Required design rules:
- diagnostics policy must be typed and manifest-visible
- probe emission must be owned by the runtime, not wrapper glue
- human-readable logs and structured diagnostics must come from the same typed events
- probe sinks must preserve experiment identity and phase identity explicitly
- every new probe class must land with focused regression coverage

## Non-Goals

This spec does not require:
- a full observability stack
- live remote dashboards
- third-party telemetry systems
- always-on verbose tracing for every run

It does require a typed runtime surface for targeted diagnostics and failure forensics.

## Required Outcomes

The runtime must support all of the following:

1. Typed diagnostics policy
- diagnostics configuration is declared in the experiment/runtime contract
- policy is validated in code
- policy is visible in manifests and artifacts

2. Typed diagnostic events
- the runtime emits structured events rather than relying on prose-only log lines
- events carry stable identity fields
- event kinds are owned by enums/structs, not free-form strings

3. Dual sinks from one source of truth
- the same typed event stream can drive:
  - readable phase/debug logs
  - structured per-run diagnostic artifacts

4. Failure-oriented phase boundaries
- the runtime can show where a step died:
  - before forward
  - inside forward
  - between forward and loss materialization
  - inside backward
  - inside optimizer step

5. Deterministic replay support
- a failing run artifact preserves enough typed diagnostic context to reproduce the same lane

## Ownership Rules

### Launch policy owns declaration

The experiment/runtime contract owns:
- whether diagnostics are enabled
- which probe families are enabled
- probe cadence / sampling
- where structured output should be written

### Runtime owns emission

The runtime owns:
- when probes fire
- which typed events are emitted
- phase ordering and lifecycle
- event correlation with active experiment state

### Artifacts own persistence

Artifact/reporting code owns:
- recording structured diagnostic outputs
- listing which probe families ran
- surfacing missing required diagnostics

No wrapper or post-processor should infer diagnostics state from raw log text.

## Required Type Shape

The exact names may differ, but the architecture must preserve this shape:

```rust
enum DiagnosticProbeKind {
    TrainBatch,
    ForwardBoundary,
    ForwardPosition,
    LossBoundary,
    BackwardBoundary,
    OptimizerBoundary,
    CudaMemory,
    CheckpointBoundary,
    ExportBoundary,
}

struct DiagnosticsPolicy {
    required: bool,
    probes: Vec<DiagnosticProbeRequest>,
    structured_output: StructuredDiagnosticsOutput,
}

struct DiagnosticProbeRequest {
    kind: DiagnosticProbeKind,
    cadence: ProbeCadence,
}

struct DiagnosticEvent {
    experiment_run_id: String,
    species: String,
    phase: RunPhase,
    step: Option<usize>,
    tokens_seen: Option<usize>,
    event: DiagnosticEventKind,
}
```

The important architectural rule is:
- config owns policy
- runtime owns event emission
- artifacts own recording

## Minimum Probe Set

The minimum useful runtime surface must cover:

1. Train batch summary
- step index
- planned steps
- tokens seen
- input/target shapes
- effective batch token count

2. Forward boundaries
- forward start
- forward complete

3. Forward position tracing
- optional position-level probes for selected steps
- includes token position and key tensor shapes

4. Loss boundary
- loss materialization start
- loss materialization complete

5. Backward boundary
- backward start
- backward complete

6. Optimizer boundary
- optimizer step start
- optimizer step complete

7. CUDA memory snapshot
- used/free/total memory at relevant boundaries

8. Checkpoint/export boundaries
- checkpoint start/complete/failure
- export start/complete/failure

## Required Phase Forensics

For any training-step failure, the runtime must be able to answer:
- did the run enter training?
- did the step start?
- how far did the forward path progress?
- was loss materialized?
- did backward begin?
- did optimizer update begin?
- what was the last successful boundary?

This must come from typed event emission, not guesswork from surrounding prose.

## Artifact Requirements

Each run artifact must record:
- diagnostics policy
- emitted diagnostic event file(s)
- whether required probes completed
- the last recorded diagnostic event per species if the run failed

Preferred artifact form:
- structured JSONL event stream per run
- plus summary fields in the main run artifact

## Failure Rules

If diagnostics are declared `required` and the runtime cannot emit them:
- the run must be marked diagnostics-incomplete
- the run artifact must say which required probe families are missing

Do not silently degrade from:
- required typed diagnostics
to:
- best-effort printlns

That would violate the contract.

## Stage 0 Immediate Scope

The next implementation slice should at minimum cover the current Stage 0 hotspot:
- `train_step_start`
- `forward_start`
- `forward_position`
- `forward_complete`
- `loss_start`
- `loss_complete`
- `backward_start`
- `backward_complete`
- `optimizer_step_start`
- `optimizer_step_complete`
- `cuda_memory_snapshot`

The current debug-only forward probes should be absorbed into this typed diagnostics surface.

## Testing Requirements

Minimum coverage:

1. Validation
- invalid probe cadences rejected
- unsupported required diagnostics rejected

2. Emission
- enabled probes emit typed events
- disabled probes do not emit stray events

3. Persistence
- structured diagnostic output is written and referenced by the run artifact

4. Failure behavior
- a simulated failure records the last successful boundary
- required diagnostics missing marks the run diagnostics-incomplete

## Acceptance Gate

This spec is complete only when:
- diagnostics are manifest-visible and typed
- runtime emits typed events for the minimum probe set
- structured outputs are recorded in artifacts
- failure boundaries are explicit
- the current Stage 0 debug probes are no longer ad hoc one-offs

## Bottom Line

`fractal` needs a real diagnostics surface.

Not because logging is nice to have.

Because scalable debugging and launch confidence require the runtime to expose where it is,
what it was doing, and exactly where it failed in a typed, durable, machine-visible way.
