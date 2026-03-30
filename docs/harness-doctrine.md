# Harness Doctrine

This harness exists to support learning experiments on top of a Rust runtime that stays understandable, deterministic, and extensible.

The architecture should make the next experiment easier, not harder.

## Product Thesis

The harness should provide:
- a clear primitive contract
- multiple local mutations of that primitive
- reliable execution
- comparable outputs
- strong enough structure that learning experiments do not deform the runtime

The harness is a tool for experiments. The harness itself should stay disciplined.

## Core Doctrine

### 1. Primitive first
Define the primitive clearly before multiplying implementations.

Each primitive should have:
- a clear contract
- explicit inputs
- explicit outputs
- explicit failure modes
- explicit invariants

Mutations exist to vary behavior within that contract.

### 2. Mutations are local implementations
A mutation should be an implementation of a shared contract, not a pile of conditionals sprayed across the system.

Adding a mutation should feel like:
- add implementation
- register it at one seam
- inherit shared tests and evaluation flow

### 3. Shared plumbing belongs in shared code
Anything repeated across mutations should be extracted into:
- shared helpers
- shared traits
- shared execution utilities
- shared evaluation utilities

Mutation files should contain what is unique about that mutation.

### 4. Runtime owns orchestration
The runtime should own:
- selection
- scheduling
- evaluation sequencing
- persistence boundaries
- experiment coordination

A mutation should not need to know the whole system to do its job.

### 5. Determinism matters
Learning experiments need reproducibility.

Record and control:
- config
- seed
- mutation identity
- version or commit
- input set
- relevant runtime flags

If a result cannot be reproduced, it has limited value.

### 6. Observability is part of the design
Every experiment path should be inspectable.

The harness should make it easy to answer:
- which mutation ran?
- with what config?
- on what input?
- with what result?
- what failed?
- what changed between runs?

### 7. No hidden contracts
Do not rely on:
- magic strings
- implied ordering
- side effects in distant modules
- undocumented assumptions between runtime and mutation code

If behavior matters, model it explicitly.

### 8. Keep the hot path simple
Experiment loops amplify small inefficiencies.

Optimize for:
- narrow data movement
- minimal repeated work
- low-overhead execution
- simple persistence
- cheap replay and comparison

## Mutation Model

Each mutation should describe itself through a typed interface.

A mutation should declare:
- identity
- configuration surface
- preconditions if any
- execution entry point
- output shape
- optional metadata used by evaluation

Preferred shape:
- one trait or small trait family for the primitive
- one implementation per mutation
- one registration seam

Avoid:
- giant switch statements
- mutation behavior keyed off raw strings
- one file that “knows” every mutation’s quirks

## State Doctrine

State should be:
- explicit
- typed
- owned by the right layer
- easy to reset between runs

Avoid:
- duplicated derived state
- stale caches without ownership
- mutation-specific state leaking into shared runtime state
- silent fallback when state is missing or inconsistent

## Persistence Doctrine

Persist only what is useful for:
- replay
- comparison
- debugging
- learning analysis

Persisted records should be:
- structured
- versionable
- ordered correctly
- easy to validate

Do not rely on UI rendering assumptions as the source of truth.

## Evaluation Doctrine

Evaluation should be separate from implementation.

The harness should distinguish:
- executing a mutation
- scoring or comparing its result
- storing experiment outcomes
- selecting future runs

That separation keeps the primitive contract clean.

## Extensibility Doctrine

The system is extensible when adding a new mutation requires:
- little or no change to existing implementations
- little or no change to runtime orchestration
- no changes to unrelated persistence or evaluation code
- clear inheritance of shared tests

If adding one mutation requires editing many unrelated files, the architecture needs tightening.

## Engineering Heuristics

Before adding an abstraction, ask:
1. Does a second real use case require it?
2. Does it reduce repeated logic across mutations?
3. Does it make the next mutation easier to add?
4. Does it preserve clear ownership?

Before accepting a design, ask:
1. Where does this behavior belong?
2. Is that ownership visible in the types?
3. Would the next person know where to add the next mutation?
4. Would a failed experiment be easy to diagnose?

## Quality Gates

A change is healthy when:
- the primitive contract is clearer after the change
- the next mutation is easier to add
- tests got stronger
- state ownership got simpler
- execution became easier to reason about

A change is unhealthy when:
- routing got more implicit
- mutation behavior spread across multiple unrelated modules
- persistence got more fragile
- evaluation and execution became tangled
- the runtime needs more special-case knowledge to function

## Default Direction

Bias toward:
- typed contracts
- local implementations
- shared plumbing
- deterministic execution
- explicit persistence
- narrow seams
- reproducible experiments

That is the path that keeps the harness useful for learning without letting the architecture drift.
