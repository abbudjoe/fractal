# Codebase Inspection Plan

This inspection exists because a lazy hardwired execution path reached the launch surface.

That is a doctrine violation.

The goal is not to find style nits. The goal is to find places where the codebase violates:
- [ENGINEERING.md](/Users/joseph/fractal/ENGINEERING.md)
- [harness-doctrine.md](/Users/joseph/fractal/docs/harness-doctrine.md)
- [experiment-interface.md](/Users/joseph/fractal/docs/experiment-interface.md)

## Audit Rules

Every slice should inspect for:
- hidden or duplicated contracts
- hardwired behavior behind typed config
- central dispatch growth
- mutation-specific logic leaking into shared runtime
- stringly or ad hoc control-plane surfaces where types should own behavior
- persistence or artifact semantics that can drift from runtime truth
- test gaps for previously observed regressions
- hot-path inefficiencies that are now structural rather than incidental

## Scope

Inspect the first-party codebase and first-party integration seams:
- `examples/`
- `fractal-core/src/`
- `fractal-primitives-private/src/`
- `fractal-tokenizer/src/`
- `src/`
- `scripts/`
- `experiments/`

The vendored backend under `vendor/` is out of scope except where first-party code relies on assumptions about it.

## Partitions

### Slice A: Control Plane And Launch Orchestration

Files:
- `examples/tournament.rs`
- `fractal-core/src/lifecycle.rs`
- `fractal-core/src/registry.rs`
- `src/run_artifacts.rs`
- `src/primitive_tracker.rs`
- `src/bin/bakeoff-summary.rs`
- `scripts/runpod-tournament.sh`
- `scripts/runpod-audit.sh`
- `experiments/`

Questions:
- Does typed config actually own runtime behavior?
- Are manifests, CLI, wrapper, and artifacts aligned?
- Is any benchmark or launch behavior still hidden behind fallback logic or duplicated policy?

### Slice B: Core Runtime, Model Kernel, And Primitive Registration

Files:
- `fractal-core/src/model.rs`
- `fractal-core/src/router.rs`
- `fractal-core/src/state.rs`
- `fractal-core/src/rule_trait.rs`
- `fractal-core/src/data_generator.rs`
- `fractal-core/src/fitness.rs`
- `fractal-core/src/primitives/mod.rs`
- `fractal-primitives-private/src/primitives/`
- `fractal-primitives-private/src/lib.rs`
- `fractal-primitives-private/src/tests.rs`

Questions:
- Is mutation behavior local to the mutation implementation?
- Are shared contracts explicit and reusable?
- Is registration clean, or is the runtime accumulating mutation-specific knowledge?

### Slice C: Tokenizer, Stage 0 Training Input, And Bridge Seams

Files:
- `src/tokenizer_training.rs`
- `src/lib.rs`
- `fractal-tokenizer/src/model_face/`
- `fractal-tokenizer/src/tokenizer.rs`
- `fractal-tokenizer/src/faceoff/`
- `fractal-tokenizer/src/primitives/`
- `fractal-tokenizer/src/lib.rs`
- `fractal-tokenizer/src/tests.rs`

Questions:
- Is tokenizer-backed training a real typed surface or a translation layer with hidden assumptions?
- Are pad/tokenizer semantics explicit all the way through?
- Does the bridge preserve clear ownership boundaries?

## Ownership

- `Hypatia`: Slice A
- `Rawls`: Slice C
- Main thread: Slice B and final synthesis across all slices

## Output Contract

Each slice should return:
- findings ordered by severity
- precise file references
- which doctrine rule is being violated
- whether the issue is architectural or local
- the smallest durable fix direction

If a slice finds no issues, it should say that explicitly and list residual risk.
