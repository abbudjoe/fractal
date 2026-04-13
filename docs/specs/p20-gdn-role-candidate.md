# P20 GDN-Role Candidate

Date: 2026-04-12

## Purpose

This note freezes the first Fractal-side candidate that treats the recurrent primitive as a Gated-DeltaNet-role block rather than a naive attention replacement.

The Parameter Golf smoke result showed that direct P20 substitution works best as a sparse middle replacement but does not beat the pure-attention control. A literal `P20 x5 -> SWA -> P20 x5 -> shared SWA` topology also failed in the Parameter Golf harness. The working interpretation is a topology and training-contract mismatch: current P20 lacked the update, local-mixing, ramp, and optimizer contracts expected by the GDN role.

## Sources

- Parameter Golf PR #1564: https://github.com/openai/parameter-golf/pull/1564
- Gated DeltaNet paper: https://arxiv.org/abs/2412.06464
- Gated DeltaNet reference repo: https://github.com/NVlabs/GatedDeltaNet
- Fractal P20 fast-lane freeze: `/Users/joseph/fractal/docs/specs/p20-packed-inproj-freeze.md`
- Parameter Golf handoff note: `/Users/joseph/parameter-golf-p20-nonrecord/records/track_non_record_16mb/2026-04-12_Fractal_RecurrentPrimitive_SP1024_1xH100/GDN_P20_CONTRACT_NOTES.md`

## Extracted Contract

Gated DeltaNet is not only a recurrence primitive. It combines:

- gated decay/erase through an input-dependent alpha control
- delta-rule targeted write strength through beta
- matrix-valued associative state rather than a single vector state
- short causal token mixing on q/k/v paths
- L2-normalized q/k paths for stability
- output normalization plus output gate before projection
- hybrid macro topology with periodic sliding-window attention
- stable stack behavior through residual/readout scaling and optimizer separation

The paper reports that short convolution and output gate are important ablation components, and frames the gated delta rule as combining rapid memory erasure with targeted key-value update. The PR #1564 record uses a `[GDN x5] -> SWA -> [GDN x5] -> SWA_shared` macro shape, but this Fractal candidate does not copy that implementation.

## Implemented Candidate

Profile:

- `PrimitiveProfile.P20_GDN_ROLE`
- CLI value: `p2-0-gdn-role`
- implementation: `/Users/joseph/fractal/python/models/primitives.py`
- runner surface: existing Path 1 `primitive-hybrid`

Block internals:

- packed q/k/v input projection via `PackedLinearProjection(d_model, (d_model, d_model, d_model), bias=False)`
- separate packed control projection for optimizer isolation: alpha, beta, output gate
- causal depthwise q/k/v local mixers with identity-start kernels
- q/k L2 normalization
- matrix recurrent state shaped `[batch, heads, value_dim, key_dim]`
- gated delta update:

```text
old = S @ k
S' = alpha * (S - beta * old outer k) + beta * v outer k
read = S' @ q
```

- per-head RMS readout norm
- output gate, output projection, and readout ramp
- wrapper `scaled` residual init special-cased to `0.1` for this profile

This is intentionally torch-runtime first. Triton/chunkwise kernels are deferred until the semantics survive the Fractal 5-step and 120s gates.

## Optimizer And Ramp Contract

The Path 1 model now exposes `optimizer_parameter_groups(base_lr)` and the runner uses it.

For `p2-0-gdn-role`, parameters are separated into:

- `p20_gdn_recurrent`: q/k/v projection and short causal mixers at `0.5x` base LR
- `p20_gdn_gates`: alpha/beta/output-gate control projection at `0.5x` base LR
- `p20_gdn_readout`: readout norm/projection at `1.0x` base LR
- `p20_gdn_scalars`: readout ramp and wrapper residual scale at `1.0x` base LR
- `default`: embedding, attention, FFN, and other non-primitive parameters at `1.0x` base LR

Initialization contract:

- alpha gate bias starts high enough to retain memory by default
- beta gate bias starts low enough to avoid aggressive early writes
- output gate bias starts low
- readout ramp starts at sigmoid logit equivalent of roughly `0.1`
- scaled residual starts at `0.1` instead of the P20 default `0.5`

## Local Smoke Results

Environment:

- local CPU
- `.venv/bin/python`
- explicit JSONL corpus: `experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1`
- `seq_len=16`, `window_stride=16`, `batch_size=1`, `steps=5`, `eval_batches=1`
- `dtype=fp32`

Results:

| lane | initial loss | final loss | train tok/s | report |
|---|---:|---:|---:|---|
| `p2-0-gdn-role` | `5.6964` | `4.1782` | `283.38` | `/Users/joseph/fractal/artifacts/v3a-python-path1/primitive-hybrid-p2-0-gdn-role-runtime-scaled-direct-pre-norm-only-mamba-rms-dense/report.json` |
| frozen `P20` torch comparator | `5.8739` | `4.8264` | `855.32` | `/Users/joseph/fractal/artifacts/v3a-python-path1/primitive-hybrid-p2-0-runtime-scaled-projected-pre-norm-only-standard-block-diagonal-2/report.json` |
| attention-only CPU comparator | `5.7008` | `5.2439` | `2054.09` | `/Users/joseph/fractal/artifacts/v3a-python-path1/attention-only/report.json` |

Interpretation:

- The candidate is shape-safe, causal, trainable, and runner-wired.
- CPU throughput is not competitive yet, which is expected for a Python matrix-state scan.
- The local smoke is a stability gate only. It is not a CUDA performance claim.

## RTX 5090 CUDA Gates

Environment:

- RunPod RTX 5090 secure pod
- CUDA compute capability: `12.0`
- `compile-safe` Python env
- arch-aware bootstrap selected `torch 2.10.0+cu128` with bundled `triton 3.6.0`
- explicit JSONL corpus: `experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1`
- `seq_len=16`, `window_stride=16`, `batch_size=1`, `eval_batches=1`
- `dtype=bf16`, `primitive_runtime_backend=torch`

The first RTX 5090 attempt failed before model execution because the old
`compile-safe` default installed `torch 2.4.1+cu124`, which only supported
CUDA kernel images through `sm_90`. The bootstrap now treats `sm_120+` as an
explicit Blackwell-or-newer contract and selects the repo's cu128 Torch line
before reusing any base-image Torch install.

Results:

| gate | initial loss | final loss | train tok/s | CUDA peak | local artifact |
|---|---:|---:|---:|---:|---|
| 5-step smoke | `5.6953` | `4.1772` | `121.04` | `48.91 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-176f8d9241091ad9/20260412T204009Z_a02/remote/artifacts/v3a-python-path1/20260412T204009Z_a02/primitive-hybrid-p2-0-gdn-role-runtime-scaled-direct-pre-norm-only-mamba-rms-dense/report.json` |
| 512-step stability | `5.6953` | `2.9397` | `179.79` | `48.91 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-176f8d9241091ad9/20260412T205048Z_a03/remote/artifacts/v3a-python-path1/20260412T205048Z_a03/primitive-hybrid-p2-0-gdn-role-runtime-scaled-direct-pre-norm-only-mamba-rms-dense/report.json` |

Interpretation:

- The block clears CUDA forward/backward and trains stably through the bounded
  512-step gate.
- The current torch implementation is not a speed claim. Throughput is dominated
  by a Python per-step matrix-state scan and tiny batch/sequence shape.
- The successful CUDA gate justifies a head-to-head Fractal comparison and,
  later, a sequence/chunk kernel if the quality signal survives.

## Validation

Passing local tests:

```bash
.venv/bin/python -m unittest \
  python.tests.test_models.Path1ModelTests \
  python.tests.test_specs.Path1SpecTests \
  python.tests.test_runtime.CudaSetupPatchTests
```

Result:

```text
Ran 44 tests in 2.112s
OK
```

The first attempted smoke with `--benchmark-profile cuda-faithful-small-v1` was stopped because that profile forces full corpus passes and is too slow on local CPU for a bounded candidate check.

## Deferred

- Triton or chunkwise sequence kernel for the matrix-state update
- RunPod execution of the corrected GDN topology head-to-head, including
  `RRRRRARRRRRS` and the optional FLA `gated-deltanet-fla` lane
- P20-GDN-role topology test shaped like
  `P20-GDN-role x5 -> SWA -> P20-GDN-role x5 -> shared SWA`
- Parameter Golf port

## Head-to-Head Gate

The CUDA head-to-head is complete and frozen in
`/Users/joseph/fractal/docs/specs/p20-gdn-head2head-scorecard.md`.

Result summary:

- Fractal-native GDN torch won quality: `final_loss=2.7272`, `160.46 tok/s`,
  `96.95 MB`
- native Mamba3 was second on quality: `final_loss=2.8989`, `460.29 tok/s`,
  `101.20 MB`
- frozen P20 Triton block-diagonal-2 was the best speed/quality recurrent lane:
  `final_loss=2.9082`, `515.60 tok/s`, `97.80 MB`
- P20-GDN-role trained stably but did not beat frozen P20:
  `final_loss=2.9717`, `167.35 tok/s`, `95.66 MB`
- attention-only remained fastest but lowest-quality in this bounded run:
  `final_loss=2.9895`, `950.68 tok/s`, `95.52 MB`

Interpretation:

- The GDN control validates the target role as a quality teacher.
- The GDN control used the alternating Path 1 schedule, not the full
  `[5xGDN] -> SWA -> [5xGDN] -> SWA_shared` topology.
- The current P20-GDN-role implementation is not yet the right translation of
  that contract.
- Frozen P20 Triton remains the fast recurrent baseline until a new GDN-informed
  P20 block can beat it.

## Historical Gate Command

The single-lane candidate gate used this command shape before the wider
head-to-head launcher was added:

```bash
scripts/runpod-tournament.sh \
  --pod-name fractal-p20-gdn-role-h100 \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --cloud-type SECURE \
  --binary-kind python \
  --binary-name scripts/v3a_python_path1.py \
  --python-requirements scripts/requirements-v3a-python-mamba3.txt \
  --python-install-mode compile-safe \
  --stop-after-run \
  -- \
  --variant primitive-hybrid \
  --primitive-profile p2-0-gdn-role \
  --primitive-execution-profile runtime \
  --primitive-residual-profile scaled \
  --primitive-readout-profile direct \
  --primitive-norm-profile pre-norm-only \
  --primitive-wrapper-profile mamba-rms \
  --primitive-state-transform-profile dense \
  --backend cuda \
  --dtype bf16 \
  --env-kind compile-safe \
  --primitive-runtime-backend torch
```

The wider H100 head-to-head is now complete. The GDN reference block is alive as
a quality teacher, but P20-GDN-role is not strong enough yet to justify a
Parameter Golf port or SWA seam topology test.
