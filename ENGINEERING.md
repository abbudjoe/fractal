# ENGINEERING.md

This repo is for a Rust agent harness that should stay small, clear, testable, and extensible.

The goal is to build useful experimental capability without accumulating avoidable technical debt.

## Core Rules

1. Solve the requested problem with the smallest complete design.
2. Search for existing helpers, traits, and utilities before adding new code.
3. Prefer explicit types over strings and ad hoc booleans.
4. Keep behavior with the module or type that owns it.
5. Extract shared plumbing. Keep mutations local.
6. Fix root causes, not symptoms.
7. Add regression tests for every bug fix.
8. Keep hot paths lean.
9. Leave the codebase easier to extend after each change.

## Design Standards

### Prefer types over strings
Use:
- enums over string flags
- structs over loose parameter bundles
- traits over central name-based dispatch
- explicit state machines over implicit state

### Keep ownership clear
- Runtime owns orchestration.
- A primitive or mutation owns its own behavior.
- Shared infrastructure owns shared concerns.
- No hidden cross-module knowledge.

### Keep mutation-specific logic local
A new mutation should usually require:
- one new implementation unit
- minimal shared plumbing updates
- no broad switch expansion across unrelated modules

### Avoid central dispatch growth
Do not build systems where every new mutation requires editing many unrelated match statements or string routers.

Preferred shape:
- shared contract
- local implementation
- registration at one clear seam

## Rust Rules

1. No `.unwrap()` or `.expect()` outside tests.
2. Functions should stay under 40 lines when practical.
3. If a function needs more than 5 parameters, introduce a struct.
4. Prefer `Result` with context over silent fallback.
5. Prefer private-by-default visibility.
6. Prefer small focused types over giant bags of state.
7. Delete dead code instead of parking it.
8. No TODO/FIXME without a linked issue.

## Testing Rules

### Every bug fix gets a regression test
If a bug happened once, add the test that would have caught it.

### Every new behavior gets focused tests
Cover:
- main path
- edge cases
- failure behavior where relevant

### Tests should be
- deterministic
- independent
- narrow in purpose
- named by behavior

### Contract tests
If multiple mutations implement the same primitive contract, add or update shared contract tests so every mutation is held to the same baseline behavior.

## Efficiency Rules

Check for:
- repeated file reads
- repeated parsing
- repeated allocation or cloning
- unnecessary persistence churn
- no-op state updates
- broad scans where narrow reads would do
- work happening on every turn that could be cached or avoided

Performance matters most on:
- startup
- per-turn execution
- persistence and replay
- evaluation loops
- experiment batch runs

## Change Workflow

Before coding:
1. Read the relevant files.
2. Restate the intended behavior.
3. Search for existing helpers and abstractions.
4. Identify the smallest clean seam for the change.

While coding:
1. Keep the change scoped.
2. Extract shared logic when duplication appears.
3. Keep mutation-specific logic local.
4. Add tests as you go.

Before finishing:
1. Review the diff for duplication, leaks, and unnecessary complexity.
2. Run:
   - `cargo fmt --all`
   - `cargo clippy --workspace --tests -- -D warnings`  
     or, for a single crate repo,  
     `cargo clippy --tests -- -D warnings`
   - `cargo test --workspace`  
     or, for a single crate repo,  
     `cargo test`
3. Summarize:
   - what changed
   - tests run
   - any remaining risk or follow-up

## Review Expectations

A change is only done when it is:
- correct
- typed clearly
- tested
- locally coherent
- easy to extend

Do not call code done if you already know where the debt is growing.

## Final Heuristic

Before you finish, ask:

- Is this the smallest clean design?
- Did I reuse what already exists?
- Does the right type own this behavior?
- Would adding the next mutation feel straightforward?
- Did I add the test that would catch this regression next time?

If the answer to any of those is no, keep going.
