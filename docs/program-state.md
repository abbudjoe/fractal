# Program State v1

This document defines the typed program-level state we should build around the
primitive harness.

Use this alongside:
- [experiment-interface.md](/Users/joseph/fractal/docs/experiment-interface.md)
- [optimization-surfaces.md](/Users/joseph/fractal/docs/optimization-surfaces.md)
- [promotion-policy.md](/Users/joseph/fractal/docs/promotion-policy.md)
- [primitive-tracker.md](/Users/joseph/fractal/docs/primitive-tracker.md)
- [harness-hardening-checklist.md](/Users/joseph/fractal/docs/harness-hardening-checklist.md)

## Purpose

The harness now owns more than one moving part:
- primitive variants
- canonical benchmark suites
- optimization surfaces
- promotion and retirement decisions
- long-running training and validation tracks

We need one typed state object that can answer:

What is the current research program state, what changed, and what is the next allowed move?

This is the layer that should generate:
- the primitive tracker
- the scientific leaderboard
- the systems-speed leaderboard
- the experiment ledger
- the roadmap

## Core Rule

`ProgramState` is the living source of truth for the research program.

It must be:
- updated automatically from preserved run evidence
- explicit about authority and review state
- append-only for run history
- able to derive planning views without hand-maintained prose

It must not:
- silently promote or retire variants without a recorded decision
- overwrite raw run evidence
- treat advisory and authoritative evidence as equivalent
- let runtime-surface changes masquerade as primitive wins

## Design Goals

`ProgramState` should:
- centralize the current state of the field
- encode the allowed next moves
- separate evidence from interpretation
- make training and optimization tracks visible at the same time
- generate derived views rather than asking humans to keep several docs in sync

## Top-Level Type

Suggested shape:

```rust
pub struct ProgramState {
    pub metadata: ProgramMetadata,
    pub variants: BTreeMap<String, VariantRecord>,
    pub suites: BTreeMap<String, SuiteRecord>,
    pub surfaces: BTreeMap<String, SurfaceRecord>,
    pub runs: BTreeMap<String, RunRecord>,
    pub decisions: Vec<DecisionRecord>,
    pub roadmap: RoadmapView,
}
```

This object should be serializable and reconstructable from manifests, artifacts,
policy, and derived summaries.

## State Ownership

### 1. `metadata`

Purpose:
- identify the current program state snapshot

Suggested fields:
- schema version
- generated at timestamp
- branch
- current commit
- state generation source

Rules:
- this is state metadata, not experiment metadata
- state generation should be reproducible from preserved artifacts

### 2. `variants`

Purpose:
- hold one current record per variant

Suggested shape:

```rust
pub struct VariantRecord {
    pub variant_name: String,
    pub base_primitive: String,
    pub lever_type: String,
    pub lane: Lane,
    pub status: VariantStatus,
    pub last_authoritative_result: Option<ResultRef>,
    pub last_advisory_result: Option<ResultRef>,
    pub failure_class: Option<FailureClass>,
    pub next_eligible_action: NextAction,
    pub review_state: ReviewState,
}
```

Rules:
- lane and status must be typed, not prose
- the tracker should derive from this section
- benchmark variants remain frozen here by rule, not by custom tracker wording

### 3. `suites`

Purpose:
- track canonical benchmark suites as first-class program objects

Suggested shape:

```rust
pub struct SuiteRecord {
    pub suite_name: String,
    pub suite_kind: SuiteKind,
    pub manifest_paths: Vec<String>,
    pub target_rows: usize,
    pub completed_rows: usize,
    pub failed_rows: usize,
    pub pending_rows: usize,
    pub authority: ComparisonAuthority,
    pub frozen_branch: String,
    pub frozen_commit: Option<String>,
    pub status: SuiteStatus,
}
```

Examples:
- canonical winner rerun
- systems-speed suite
- validation confirmation suite

Rules:
- suite completion must be computed from manifests plus artifacts
- frozen commit is captured from the launch, not guessed later
- roadmap status should derive from suite status

### 4. `surfaces`

Purpose:
- track optimization surfaces independently from primitive variants

Suggested shape:

```rust
pub struct SurfaceRecord {
    pub surface_name: String,
    pub rollout_state: SurfaceRolloutState,
    pub default_enabled: bool,
    pub validated_on_commit: Option<String>,
    pub validation_suite: Option<String>,
    pub residual_risk: Option<String>,
}
```

Rules:
- surfaces must not be hidden behind CLI-only flags
- their rollout state must be visible in the roadmap
- a validated surface is still not a primitive win

### 5. `runs`

Purpose:
- preserve append-only run history in one indexed view

Suggested shape:

```rust
pub struct RunRecord {
    pub run_id: String,
    pub logical_experiment_id: String,
    pub attempt_id: String,
    pub variant_name: Option<String>,
    pub suite_name: Option<String>,
    pub benchmark_mode: BenchmarkMode,
    pub authority: ComparisonAuthority,
    pub build_commit: Option<String>,
    pub backend: String,
    pub outcome: RunOutcomeClass,
    pub metrics: Option<RunMetrics>,
}
```

Rules:
- this section is append-only
- raw evidence is never edited in place
- the ledger should derive from here

### 6. `decisions`

Purpose:
- record interpretation and review separately from raw evidence

Suggested shape:

```rust
pub struct DecisionRecord {
    pub decision_id: String,
    pub target: DecisionTarget,
    pub proposed_action: ProposedAction,
    pub rationale: String,
    pub evidence_refs: Vec<String>,
    pub review_state: ReviewState,
    pub decided_at: Option<String>,
}
```

Examples:
- promote `p1_fractal_hybrid_composite_v1` to leader
- hold `p3_hierarchical_v1` for one harder validation lane
- retire `mandelbox_recursive_dyn-escape-radius_v1`
- validate `eval_backend_split` on the next frozen commit

Rules:
- the system may propose decisions automatically
- canonical promotion/retirement remains reviewable
- tracker status changes should be traceable back to one decision record

### 7. `roadmap`

Purpose:
- expose the next work as a derived planning view

This is the key architectural rule:

The roadmap is a derived view over program state, not hand-maintained prose.

Suggested shape:

```rust
pub struct RoadmapView {
    pub current_phase: ProgramPhase,
    pub training_track: Vec<RoadmapItem>,
    pub optimization_track: Vec<RoadmapItem>,
    pub validation_track: Vec<RoadmapItem>,
    pub blocked: Vec<RoadmapItem>,
    pub rejoin_gates: Vec<RejoinGate>,
}
```

Rules:
- roadmap items should be generated from suites, surfaces, variant records, and policy
- blocked items should cite the missing suite/surface/decision
- the roadmap must show training and optimization together without conflating them

## Derived Views

The following views should be generated from `ProgramState`, not maintained separately:

1. Primitive Tracker
- derives from `variants` plus reviewed `decisions`

2. Scientific Leaderboard
- derives from authoritative completed `runs`

3. Systems-Speed Leaderboard
- derives from `runs` where `benchmark_mode = systems-speed`

4. Experiment Ledger
- derives from append-only `runs`

5. Roadmap
- derives from `variants`, `suites`, `surfaces`, and `decisions`

## Sources of Truth

`ProgramState` should be rebuilt from:
- experiment manifests
- run artifacts
- wrapper manifests
- promotion policy
- canonical suite definitions

It should not treat tracker prose as the primary source of truth.
The tracker becomes one rendered view, not the root state.

## Update Model

The update loop should look like this:

1. ingest new manifests and artifacts
2. update append-only `runs`
3. recompute suite completion state
4. recompute variant last-result references
5. apply policy rules to propose decisions
6. refresh derived roadmap
7. render tracker/leaderboards/ledger views

This gives us a self-updating control plane without silent self-governance.

## Review Model

The program state may:
- auto-ingest evidence
- auto-generate proposals
- auto-mark blocked/unblocked roadmap items

The program state must not:
- silently finalize promotions
- silently retire variants
- silently flip optimization surfaces to default

So we need typed review state such as:
- `proposed`
- `approved`
- `rejected`
- `superseded`

## Invariants

The following invariants must hold:
- run evidence is append-only
- authoritative and advisory evidence remain distinct
- mixed-commit suites cannot silently count as canonical
- surface rollout state is separate from variant status
- benchmark variants remain frozen
- roadmap items must cite a typed source, not prose inference

## Current Program Mapping

If we projected today’s field into `ProgramState`, the major records would be:

- winner-lane suite:
  - canonical frozen rerun of
    - `p1_contractive_v1`
    - `p1_fractal_hybrid_v1`
    - `p1_fractal_hybrid_composite_v1`
- current leader:
  - `p1_fractal_hybrid_composite_v1`
- top cohort:
  - `p1_fractal_hybrid_composite_v1`
  - `p1_fractal_hybrid_v1`
  - `p1_contractive_v1`
- alternate-family contender:
  - `logistic_chaotic_map_v1`
- validation queue:
  - `p3_hierarchical_v1`
  - `b2_stable_hierarchical_v1`
- first optimization surfaces:
  - `eval_backend_split`
  - `batching_policy`
  - `forward_execution_policy`

The roadmap view should make that state obvious without manual rewriting.

## Implementation Order

`ProgramState v1` should land in stages:

1. `specified`
- this document exists

2. `implemented-readonly`
- build the typed state from manifests/artifacts
- no tracker writing yet

3. `derived-views`
- tracker, leaderboard, ledger, and roadmap are rendered from state

4. `proposal-engine`
- promotion/retirement/hold proposals are generated from policy

5. `reviewable-control-plane`
- decisions can be explicitly approved/rejected and the rendered views follow that reviewed state

## Primary Files

Likely implementation homes:
- `src/program_state.rs`
- `src/bin/program-state.rs`
- `src/bin/bakeoff-summary.rs`
- `src/primitive_tracker.rs`
- `src/run_artifacts.rs`
- `fractal-core/src/lifecycle.rs`

## Strategic Value

If nothing else came from the project, this layer would still be valuable.

It gives us:
- a typed research operating system
- reproducible experiment governance
- a roadmap that updates from evidence instead of memory
- a clean separation between primitive discovery and systems optimization

That is exactly the layer we need if the harness is going to scale into a real training program.
