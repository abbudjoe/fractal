# Cloud Workflow Policy

## Context

All current development flows through cloud frontier models (Codex as orchestrator, cloud LLMs for generation). Full local sovereignty is not achievable yet. This policy is structural hygiene: compartmentalize novelty now so that future local-compute migration is straightforward.

## Principle

Reduce unnecessary exposure of novel experimental logic to cloud-hosted tooling. The harness (runner, config, CLI, persistence, evaluation framework) is cloud-safe. The concrete experimental primitives (mutation implementations, fitness functions, selection algorithms) are novel-bearing and should be isolated.

## Compartmentalization Rules

### Cloud-Safe (stays in `fractal-core`)
- `FractalRule` trait and `FractalState` enum (the contract surface)
- `StateLayout`, `FractalModel`, `EarlyExitRouter`
- `Tournament`, `TournamentConfig` (orchestration)
- `SimpleHierarchicalGenerator`, `TokenBatch`, task families (data pipeline)
- `SpeciesId` enum, `SpeciesDefinition`, registry infrastructure
- `ComputeBackend`, `ExecutionMode`, backend type aliases
- Error types, shared tensor helpers (`one_minus`, `gated_sigmoid`, `complex_square`)
- Examples, CLI, persistence, config

### Novel-Bearing (extract to `fractal-primitives-private`)
- All 7 concrete primitive implementations:
  - `p1_contractive`, `p2_mandelbrot`, `p3_hierarchical`
  - `b1_fractal_gated`, `b2_stable_hierarchical`, `b3_fractal_hierarchical`, `b4_universal`
- The specific fitness scoring weights and formulas (`aggregate_results`, `stability_score`, `perplexity_score`, `speed_score`)
- Any future selection/evolution algorithms beyond basic tournament ranking

### Novel-Bearing (extract to `fractal-eval-private`)
- Tournament scoring logic (the weighted fitness formula)
- Selection algorithms
- Any future meta-learning or adaptive tournament strategies

## Workflow Guidelines

1. **Codex prompts for cloud-safe code are unrestricted.** Harness plumbing, trait design, orchestration, data generation, test infrastructure.

2. **Codex prompts touching novel-bearing code should be minimal and task-scoped.** Do not paste full primitive implementations into prompts when only the trait contract is needed.

3. **When adding a new primitive**, create it in `fractal-primitives-private`. The cloud-safe layer should only see it through the `FractalRule` trait registration seam.

4. **Fitness formula tuning** happens in `fractal-eval-private`. The harness sees only the `RankedSpeciesResult` output shape.

5. **Future local-compute migration**: when DGX Spark class hardware is available, novel-bearing crates can move to fully local tooling while the harness continues using cloud orchestration.

## What This Does NOT Do

- This is not secrecy. Cloud models have already seen the current implementations.
- This does not prevent Codex from working on primitives when needed.
- This is forward-looking hygiene: stop accumulating novel logic in cloud-exposed layers as the work progresses.
