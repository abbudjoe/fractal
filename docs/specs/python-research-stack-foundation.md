# Python Research Stack Foundation

Status: active architecture substrate for Path 1 and future mini-MoE work

## Purpose

This note records the package shape that now serves as the Python source of
truth for model-architecture research in this repo.

The goal is not to create a second disconnected experiment stack beside `v3a`.
The goal is to move the live Path 1 hybrid line and the future mini-MoE line
onto one coherent Python substrate with explicit typed seams.

## Package Layout

The shared package lives directly under:

- `/private/tmp/fractal-v3a-merge-G4mipf/python/`

Main modules:

- `/private/tmp/fractal-v3a-merge-G4mipf/python/specs/`
  - typed benchmark, Path 1, and mini-MoE config surfaces
- `/private/tmp/fractal-v3a-merge-G4mipf/python/data/`
  - byte-level JSONL corpus loading, windowing, batching, and explicit data-seed wiring
- `/private/tmp/fractal-v3a-merge-G4mipf/python/runtime/`
  - runtime/device resolution, seed handling, warmup, train/eval execution
- `/private/tmp/fractal-v3a-merge-G4mipf/python/reporting/`
  - typed report schema, ledger append, and table rendering
- `/private/tmp/fractal-v3a-merge-G4mipf/python/models/`
  - shared transformer blocks, Path 1 sequence blocks, primitive contracts, official reference-SSM adapter, mini-MoE seams
- `/private/tmp/fractal-v3a-merge-G4mipf/python/runners/`
  - runner entrypoints that bind manifests, models, data, and reports together

## Path 1 Plug-In Points

Path 1 now lives on the shared substrate through:

- `/private/tmp/fractal-v3a-merge-G4mipf/python/specs/path1.py`
- `/private/tmp/fractal-v3a-merge-G4mipf/python/models/path1.py`
- `/private/tmp/fractal-v3a-merge-G4mipf/python/models/reference_ssm.py`
- `/private/tmp/fractal-v3a-merge-G4mipf/python/models/primitives.py`
- `/private/tmp/fractal-v3a-merge-G4mipf/python/runners/path1.py`

Current Path 1 support includes:

- attention-only baseline
- reference-ssm-hybrid schedule
- primitive-hybrid schedule
- fixed 8-layer envelope
- local causal attention blocks
- typed sequence-primitive boundary:
  - `init_state`
  - `step`
  - `scan`
- current primitive family/config surface:
  - `P1`
  - `P20`
  - `P2`
  - `P21`
  - `P22`
  - `P23`
  - residual/readout/norm/wrapper variants
- explicit reference-family distinction:
  - MIMO reference
  - SISO reference
  - SISO runtime-oriented lane

The old script:

- `/private/tmp/fractal-v3a-merge-G4mipf/scripts/v3a_pytorch_mamba3_hybrid.py`

is now only a thin wrapper over the package runner instead of a one-off
architecture island.

The Mamba parity helper:

- `/private/tmp/fractal-v3a-merge-G4mipf/scripts/mamba3_pytorch_reference.py`

now wraps reusable package code in:

- `/private/tmp/fractal-v3a-merge-G4mipf/python/models/mamba3_reference_math.py`

## Mini-MoE Plug-In Points

The future mini-MoE line plugs into the same substrate through:

- `/private/tmp/fractal-v3a-merge-G4mipf/python/specs/mini_moe.py`
- `/private/tmp/fractal-v3a-merge-G4mipf/python/models/mini_moe.py`
- `/private/tmp/fractal-v3a-merge-G4mipf/python/models/transformer.py`

The important seam is explicit:

- repeated local-attention transformer blocks
- pluggable FFN seam inside each block
- typed router boundary
- typed dispatch boundary
- typed observability sink

That means future one-shot and recurrent mini-MoE routers can land as narrow
model-layer deltas instead of forcing a backbone rewrite.

## Intentionally Deferred

Still intentionally deferred:

- full Python port of every historical Rust `v3a` runner surface
- recurrent mini-MoE router implementation
- route-plan vs dispatch-plan experiment harness for mini-MoE
- kernel/performance optimization work as the primary task
- final large-scale benchmark policy

The current priority remains:

1. Path 1 first-class Python architecture
2. mini-MoE substrate on the same seams
3. future routing experiments as narrow additions on top of that foundation
