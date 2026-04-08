# V2 Scorecard

## Purpose

This document turns the `fractal-v2` KPI set into a decision surface.

It defines what the metrics look like when:

1. the narrow v2 proving architecture is ready to be called stable enough to build on
2. tuning is still needed, but the design is on track
3. a major architectural issue or regression is present and the design needs reconsideration

This is intentionally stricter than “the code runs.”

## Overall principle

`fractal-v2` should not earn more complexity just because the system is implemented.
Deferred features stay deferred until the current tree-only design is both stable and useful.

Deferred features include:

* side-memory bank
* learned eviction
* learned merge scheduling
* routing early-stop
* giant-scale training
* dense fallback attention
* rescue complexity added only to cover weak early results

## Scorecard

### 1. Learning Signal

#### Definition of done

* eval loss improves reliably over random-init across repeated smoke runs
* perplexity trends in the right direction under fixed token budgets
* training does not diverge, spike, or produce unstable losses

#### Tuning still needed, but on track

* eval loss improves, but only modestly
* runs are stable
* gains are real but still small

#### Major architectural issue or regression

* no consistent eval-loss improvement
* repeated flat or worsening curves
* unstable training or divergence

### 2. Synthetic Probe Utility

#### Definition of done

On retrieval-sensitive probes, the mode ordering is consistently useful:

* `TreePlusExactRead >= TreeOnly >= SummariesOnly >= NoMemory`

And:

* exact read helps specifically on copy / retrieval tasks
* gains repeat across checkpoints, not just one run

#### Tuning still needed, but on track

* some tasks separate while others remain flat
* gains are small but directionally right
* exact read helps on at least a subset of tasks

#### Major architectural issue or regression

* all memory modes remain effectively identical after meaningful training
* tree and exact read never beat `NoMemory`
* the local trunk appears to do all the work

### 3. Causal Memory Auditor Deltas

#### Definition of done

* removing tree read measurably hurts retrieval tasks
* removing exact read measurably hurts copy / retrieval tasks
* next-best substitution is meaningfully worse than chosen retrieval
* root drop shows nontrivial contribution from more than one root

#### Tuning still needed, but on track

* interventions matter, but only weakly
* effects exist, though not sharply yet

#### Major architectural issue or regression

* interventions barely change behavior
* next-best substitution performs almost as well as the chosen route
* root-drop deltas imply redundant roots

### 4. Root Specialization / Collapse

#### Definition of done

* roots are clearly not near-identical
* root-drop causes real degradation
* per-root usage and contribution are differentiated

Practical target:

* similarity meaningfully below near-collapse
* a healthier zone is roughly `0.4 - 0.8` if utility is also improving

#### Tuning still needed, but on track

* similarity is still somewhat high but moving downward with training
* roots are not fully redundant, but specialization is weak

#### Major architectural issue or regression

* similarity remains near `1.0`
* root-drop barely matters
* multi-root behaves like a wider single root rather than multiple functional processors

### 5. Routing Health

#### Definition of done

* routing is sparse and query-dependent
* heads do not all choose the same path
* selected span distance is meaningfully distributed
* selected leaves are not concentrated into one trivial hotspot

#### Tuning still needed, but on track

* routing works but remains too concentrated
* head agreement is still higher than desired
* depth behavior is repetitive but not fully collapsed

#### Major architectural issue or regression

* all heads choose the same path nearly all the time
* routing depth is trivial in practice
* selected leaves collapse to a tiny subset regardless of query

### 6. Systems / Scaling Health

#### Definition of done

* `dead_nodes=false`
* tree updates remain stable and incremental
* benchmark runs complete across the intended scale sweep
* tree, routing, and exact-read costs stay bounded and interpretable
* causal correctness invariants hold

#### Tuning still needed, but on track

* implementation is stable
* some surfaces are slower than desired
* optimization is needed, but correctness is intact

#### Major architectural issue or regression

* tree state breaks under scale
* dead or unused node behavior becomes common
* routing or exact-read cost becomes pathological
* causal or tree invariants fail

## Current assessment

### Definition of done

Not yet.

### Tuning still needed, but on track

This is the best current overall description.

Why:

* the implementation foundation is built
* the systems surfaces are healthy
* smoke-scale training is real
* live learned-checkpoint evaluation is now real

### Major architectural issue or regression

Not proven, but there are two watch items:

* memory-mode probe gains are still too weak to interpret
* root similarity is still far too high in the current smoke-trained checkpoint

## Current bottom line

`fractal-v2` is now at the point where the next phase should be driven by trained-checkpoint evidence, not more architectural surface area.

The next decision gate is:

1. train a larger checkpoint
2. rerun synthetic probes
3. rerun causal memory auditor interventions
4. rerun benchmark and observability
5. judge the scorecard again before adding deferred complexity
