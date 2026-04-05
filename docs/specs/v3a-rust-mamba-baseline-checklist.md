# v3A Rust Mamba Baseline Checklist

This checklist turns Path 1 into a concrete implementation program:

* build a real Rust Mamba-style hybrid baseline first
* validate it in the same environment, tooling, and eval surface we will later
  use for the contender
* freeze it as the Path 1 benchmark before primitive redesign resumes

---

## Goal

Produce a **validated Rust hybrid-attention baseline** that plays the same role
as a modern Mamba-style hybrid model inside this repo.

“Validated” does not mean “state of the art.”

It means:

* the model is real, not a proxy
* it trains reproducibly in this environment
* it runs on the shared Path 1 control plane
* it beats or matches the attention-only baseline on the agreed smoke surface
* it becomes the frozen benchmark our primitive line must later beat

---

## Current Status

- [x] attention-only baseline exists
- [x] Path 1 matrix runner exists
- [x] typed proxy reference lane exists as `mamba3-proxy-v1`
- [x] seeded smoke runs are reproducible
- [ ] faithful Rust Mamba-style baseline exists
- [ ] faithful Rust baseline is frozen as the Path 1 benchmark

The proxy is useful for bring-up, but it is **not** the benchmark we want to
make contender claims against.

---

## Non-Goals

- [ ] Do not redesign our primitives during this checklist
- [ ] Do not import Path 2 memory/index ideas
- [ ] Do not add MoE, retrieval sidecars, or extra hybrid complexity
- [ ] Do not claim “beats Mamba-3” until the faithful Rust reference exists

---

## Fixed Contract

These must stay fixed during the baseline build unless a checklist item says
otherwise:

- [ ] tokenizer
- [ ] byte-level smoke corpus slice
- [ ] local attention window
- [ ] hidden width
- [ ] head count
- [ ] total layer count
- [ ] output head contract
- [ ] eval suites and report schema

The baseline must land on the same Path 1 surfaces:

- [ ] `fractal-core/src/hybrid_attention/`
- [ ] `fractal-eval-private/src/hybrid_attention.rs`
- [ ] `fractal-eval-private/src/hybrid_attention_training.rs`
- [ ] `src/bin/v3a-hybrid-attention-matrix.rs`

---

## Phase 0: Proxy Bring-Up

This phase is already mostly complete.

- [x] prove the Path 1 matrix can execute three lanes
- [x] add seeded runs
- [x] make the reference lane explicit as a proxy, not an implicit Mamba claim
- [x] collect the first smoke result showing proxy vs attention-only vs
  primitive-hybrid

Exit criterion:

- [x] the control plane is real enough to accept a faithful reference block

---

## Phase 1: Faithful Reference Design

Define what “faithful enough Rust Mamba-style baseline” means before writing the
full block.

- [x] write a short design note describing the chosen Mamba-style contract
- [x] list which paper features are required in the first faithful pass
- [x] list which paper features are intentionally deferred
- [x] specify the exact state update, readout, and scan contract
- [x] specify parameter shapes and initialization rules
- [x] specify backend constraints for CPU and Metal bring-up

Design note:

* [`v3a-rust-mamba-baseline-design.md`](./v3a-rust-mamba-baseline-design.md)
* [`v3a-rust-mamba3-porting-note.md`](./v3a-rust-mamba3-porting-note.md)

Required outcome:

- [x] a reader can point to one typed block contract and say “this is the Path 1
  baseline block”

---

## Phase 2: Core Block Implementation

Implement the faithful reference block in `fractal-core`.

- [ ] add a dedicated baseline block type under `fractal-core/src/hybrid_attention/`
- [ ] keep the block separate from our primitive line
- [ ] give it typed config and shape validation
- [ ] keep state update and output readout explicitly separate
- [ ] add focused unit tests for:
  - shape correctness
  - deterministic seeded initialization
  - step-by-step state update behavior
  - scan consistency over sequences

Exit criteria:

- [ ] the block compiles cleanly
- [ ] the block has focused regression tests
- [ ] the block is clearly distinct from `mamba3-proxy-v1`

---

## Phase 3: Hybrid Model Integration

Replace the proxy lane with the faithful baseline in the Path 1 hybrid stack.

- [ ] add a real `ReferenceSsmHybridAttentionModel`
- [ ] wire the faithful block into the interleaved `A-S-A-S-A-S-A-S` schedule
- [ ] remove any code path that silently falls back from the faithful baseline
  to the proxy
- [ ] keep the proxy only if it remains explicitly labeled as historical bring-up

Exit criteria:

- [ ] the matrix runner executes `attention-only`
- [ ] the matrix runner executes `reference-ssm-hybrid` with the faithful block
- [ ] the matrix runner can still execute `primitive-hybrid`

---

## Phase 4: Training and Report Surface

Make the baseline observable and comparable.

- [ ] ensure the faithful reference lane uses the same seeded training surface
- [ ] record training throughput
- [ ] record decode throughput or an equivalent decode-time proxy
- [ ] record activation-memory and exact-attention-fraction metrics if available
- [ ] keep report fields aligned across all Path 1 lanes

Exit criteria:

- [ ] one report schema covers attention-only, faithful reference, and primitive
- [ ] no lane has hidden extra metrics or missing key metrics

---

## Phase 5: Reproducible Validation

Run the baseline on a serious enough smoke budget to become the benchmark.

- [ ] run at least one reproducible seeded smoke matrix beyond the tiny 4-step pass
- [ ] rerun the same seed to confirm stable ordering
- [ ] run at least one second seed to check for gross brittleness
- [ ] confirm the faithful baseline matches or beats attention-only on the smoke
  surface
- [ ] confirm the result is not just a one-off from the proxy-era lane

Exit criteria:

- [ ] the faithful Rust reference is stable and reproducible
- [ ] it has a coherent evaluation story
- [ ] it is strong enough to freeze as the Path 1 benchmark

---

## Phase 5A: CUDA / RunPod Benchmark Gate

Do not benchmark Path 1 on RunPod until the backend gap is closed explicitly.

- [ ] add a CUDA execution path for the Rust Mamba-3 reference lane
- [ ] validate CUDA outputs against the existing parity ladder where possible
- [ ] run at least one seeded CUDA smoke benchmark on the shared Path 1 matrix
- [ ] confirm the CUDA report surface matches the local CPU/Metal report schema
- [ ] document any remaining CUDA-vs-local numerical tolerance policy

Exit criteria:

- [ ] RunPod benchmarking is no longer blocked by a missing CUDA backend
- [ ] CUDA is treated as a validated backend, not an aspirational future target

---

## Phase 6: Freeze The Baseline

Do not start contender redesign until this happens.

- [ ] mark the faithful Rust reference lane as the official Path 1 baseline
- [ ] record the frozen baseline config in docs
- [ ] record the seed/budget/report artifact used to define the freeze point
- [ ] explicitly retire proxy-only claims from the main Path 1 story

Exit criteria:

- [ ] future contender work can say exactly what baseline it is trying to beat
- [ ] the baseline no longer shifts while contender work is underway

---

## Allowed Next Step After Completion

Only after all phases above are done:

- [ ] resume primitive contender design
- [ ] compare our primitive against the frozen Rust Mamba-style baseline
- [ ] run matched ablations without changing the baseline definition midstream

---

## Definition Of Done

This checklist is complete only when all of the following are true:

- [ ] the Path 1 reference lane is no longer a proxy
- [ ] the Rust baseline is reproducible
- [ ] the Rust baseline is validated on the shared Path 1 surface
- [ ] the CUDA / RunPod benchmark gate is either complete or explicitly marked
  out of scope for the current proving round
- [ ] the baseline is frozen in docs and artifacts
- [ ] contender work can proceed against a real apples-to-apples benchmark
