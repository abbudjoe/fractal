# P20 / Mamba3 / GDN Head-to-Head Scorecard

Date: 2026-04-12

## Purpose

This scorecard freezes the first Fractal Path 1 head-to-head that includes a
Gated-DeltaNet-role competitor alongside the existing attention, native Mamba3,
frozen P20 fast lane, and P20-GDN-role candidate lanes.

Important scope correction: the GDN lane in this scorecard is a Fractal-native
torch reference block-control lane on the default alternating Path 1 schedule.
It is not the full Parameter Golf PR #1564 topology. The PR topology is closer
to `[5xGDN] -> SWA -> [5xGDN] -> SWA_shared`, and its recurrent blocks are FLA
`GatedDeltaNet(..., mode="chunk")` blocks rather than this internal torch
reference scan.

This scorecard therefore shows that the GDN update contract is promising, not
that Fractal has reproduced the record GDN hybrid.

Sources:

- Parameter Golf PR #1564: https://github.com/openai/parameter-golf/pull/1564
- Gated DeltaNet paper: https://arxiv.org/abs/2412.06464
- P20-GDN-role candidate note: `/Users/joseph/fractal/docs/specs/p20-gdn-role-candidate.md`
- P20 packed-inproj freeze note: `/Users/joseph/fractal/docs/specs/p20-packed-inproj-freeze.md`

## Run Contract

- launcher: `/Users/joseph/fractal/scripts/runpod-v3a-python-path1-gdn-head2head.sh`
- RunPod GPU: `NVIDIA H100 80GB HBM3`
- cloud type: secure
- seed: `42`
- steps: `512`
- eval batches: `1`
- sequence length: `16`
- window stride: `16`
- batch size: `1`
- dtype: `bf16`
- train/eval corpus: `experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1`
- official lanes env: `official-mamba3`
- frozen P20 fast lane env: `primitive-triton`

The official pod ran attention-only, native Mamba3, Fractal-native GDN, and
P20-GDN-role. The launcher then stopped that pod and ran the frozen P20 Triton
lane on a separate primitive-triton pod. `runpodctl pod list` returned `[]`
after completion.

## Results

| rank by loss | lane | implementation | initial loss | final loss | train tok/s | CUDA peak MB | local report |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `A + GDN torch` | `python_reference_ssm_gated_deltanet_torch` | `5.6211` | `2.7272` | `160.46` | `96.95` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260412T213106Z_a03/remote/artifacts/v3a-python-path1/20260412T213106Z_a03/reference-ssm-hybrid-gated-deltanet-torch/report.json` |
| 2 | `A + Mamba3 native` | `python_reference_ssm_native_runtime` | `5.9082` | `2.8989` | `460.29` | `101.20` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260412T212824Z_a02/remote/artifacts/v3a-python-path1/20260412T212824Z_a02/reference-ssm-hybrid-mamba3-siso-runtime/report.json` |
| 3 | `A + P20 Triton block-diagonal-2` | `python_primitive_triton_runtime` | `5.8750` | `2.9082` | `515.60` | `97.80` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260412T213354Z_a05/remote/artifacts/v3a-python-path1/20260412T213354Z_a05/primitive-hybrid-p2-0-runtime-scaled-projected-pre-norm-only-standard-block-diagonal-2/report.json` |
| 4 | `A + P20-GDN-role torch` | `python_primitive_runtime` | `5.6953` | `2.9717` | `167.35` | `95.66` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260412T213225Z_a04/remote/artifacts/v3a-python-path1/20260412T213225Z_a04/primitive-hybrid-p2-0-gdn-role-runtime-scaled-direct-pre-norm-only-mamba-rms-dense/report.json` |
| 5 | `A` | `python_attention_sdpa` | `5.6992` | `2.9895` | `950.68` | `95.52` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260412T211253Z_a01/remote/artifacts/v3a-python-path1/20260412T211253Z_a01/attention-only/report.json` |

## Interpretation

- The GDN-role control is alive: it beat native Mamba3, frozen P20 Triton, and
  attention-only on this bounded 512-step Fractal surface.
- The GDN control is not a speed claim. It is the slowest quality winner because
  the matrix-state scan is a torch/Python reference implementation.
- Frozen P20 Triton remains the stronger speed/quality recurrent lane:
  `515.60 tok/s` at `2.9082` loss versus native Mamba3's `460.29 tok/s` at
  `2.8989` loss.
- P20-GDN-role did not inherit the GDN control's quality advantage yet. The
  current candidate has the right broad contracts but is still missing enough
  of the GDN update/topology behavior to beat frozen P20 or native Mamba3.
- Attention-only remains the throughput ceiling on this tiny sequence shape.
  Its quality is worst here, but it is still roughly `1.84x` faster than frozen
  P20 Triton and `5.92x` faster than torch GDN.

## Decision

Do not port this GDN control directly into Parameter Golf yet. The immediate
correction is to run a topology-faithful head-to-head before drawing stronger
conclusions:

- torch GDN topology control with schedule `RRRRRARRRRRS`
- `S` as a typed shared-SWA seam, sharing the first SWA attention module
- explicit `--local-window` so SWA is not hidden behind a default
- optional FLA GDN topology lane using `flash-linear-attention`

After that corrected run, the useful Fractal-side work is to extract why the
GDN reference block won on quality, then translate only the durable mechanism
into a faster P20-compatible runtime:

- tighter GDN-style targeted write semantics
- short causal local mixing inside the recurrent block
- output-gated readout with stable ramping
- optimizer isolation for recurrent matrices, gates, readout, and scalars
- eventual chunk/sequence kernel only after the torch contract is stable

The frozen P20 Triton block-diagonal lane remains the fast recurrent baseline.
The Fractal-native GDN block is now the quality teacher/control for the next
P20 redesign loop.

## Follow-Up Surface

Implemented after this scorecard:

- `ReferenceSsmProfile.GATED_DELTANET_FLA` with CLI value `gated-deltanet-fla`
- shared-SWA schedule token `S`, so the PR-like topology is `RRRRRARRRRRS`
- CLI `--local-window`
- optional FLA requirements file:
  `/Users/joseph/fractal/scripts/requirements-v3a-python-gdn-fla.txt`
  with `flash-linear-attention==0.4.2`
- launcher support in:
  `/Users/joseph/fractal/scripts/runpod-v3a-python-path1-gdn-head2head.sh`

Local smoke passed for the torch topology lane:

```text
reference-ssm-hybrid-gated-deltanet-torch-schedule-rrrrrarrrrrs
initial_loss=5.9298
final_loss=5.6067
train_tok_s=74.97
cuda_peak_mb=0.00
```

The FLA topology lane is intentionally opt-in with
`RUN_FLA_GDN_TOPOLOGY=1` because it introduces a new native/Triton dependency
surface.

## Torch GDN Topology Probe

Date: 2026-04-12

A single H100 lane was run for the PR-like torch topology only:

- profile: `gated-deltanet-torch`
- schedule: `RRRRRARRRRRS`
- topology: `[5xGDN] -> SWA -> [5xGDN] -> SWA_shared`
- local window: `256`
- seed: `42`
- steps: `512`
- sequence length: `16`
- batch size: `1`
- dtype: `bf16`
- execution note: run directly on the RunPod base image's system
  `torch 2.4.1+cu124`; the wrapper venv bootstrap was paused because it was
  spending H100 time unpacking the same torch stack.

Result:

| lane | initial loss | final loss | train tok/s | CUDA peak MB | local report |
|---|---:|---:|---:|---:|---|
| `A + GDN torch topology RRRRRARRRRRS` | `5.7539` | `2.8584` | `80.32` | `112.95 MB` | `/Users/joseph/fractal/.runpod-local-logs/manual-results/gdn-topology-h100-s42/report.json` |

Interpretation:

- The topology-faithful torch GDN probe still beats native Mamba3, frozen P20
  Triton, and attention-only on this bounded loss surface.
- It does not beat the earlier alternating torch GDN block-control result
  (`2.7272`), so the SWA topology is not automatically better at this tiny
  sequence/window scale.
- The result is strong enough to justify one P20 topology ablation:
  `PPPPPAPPPPPS` with the current best P20 fast lane, measured as a quality
  probe first rather than a throughput claim.
- If P20 quality improves under the same SWA seam, then a faster shared-SWA or
  chunked recurrent runtime becomes worth considering. If it does not improve,
  the GDN advantage is likely coming from the GDN update law more than the macro
  topology.

## Parallel Composite SWA Probe

Date: 2026-04-12

This probe tested the explicit parallel-hybrid hypothesis:

- `GDN + P20`
- `P20 + Mamba3`
- `GDN + Mamba3`
- `GDN + P20 + Mamba3`

All four used the same PR-like SWA macro-topology:

- schedule: `RRRRRARRRRRS`
- topology: `[5xR] -> SWA -> [5xR] -> SWA_shared`
- local window: `256`
- seed: `42`
- RTX 5090 smoke steps: `128`
- H100 promotion steps: `512`
- sequence length: `16`
- batch size: `1`
- dtype: `bf16`
- corpus: `experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1`

The composite recurrent block is parallel, not serial: each branch reads the
same normalized hidden state, branch outputs are independently normalized, and
a learned per-channel softmax gate mixes branch outputs. This keeps the
experiment disciplined: the probe asks whether the memory mechanisms complement
one another at the same recurrent seam, not whether one giant serial block can
overfit a tiny run.

Environment note: the first RTX 5090 attempt failed before model execution
because `causal-conv1d` tried to compile `sm_120` with the RunPod image's CUDA
12.4 `nvcc`, which does not support `compute_120`. The bootstrap now separates
device arch from source-extension build arch. On Blackwell, it still installs
the cu128 Torch line, but source extensions fall back to the highest
compiler-supported `+PTX` target when necessary. The successful RTX 5090 run
patched both `causal-conv1d` and `mamba_ssm` to `sm_90+PTX`.

RTX 5090 smoke results:

| rank by loss | lane | initial loss | final loss | train tok/s | CUDA peak MB | local report |
|---:|---|---:|---:|---:|---:|---|
| 1 | `GDN + P20` | `5.6758` | `3.1824` | `26.17` | `78.92 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260412T225748Z_a08/remote/artifacts/v3a-python-path1/20260412T225748Z_a08/reference-ssm-hybrid-gated-deltanet-p20-torch-schedule-rrrrrarrrrrs/report.json` |
| 2 | `GDN + P20 + Mamba3` | `5.7598` | `3.2029` | `22.44` | `105.83 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260412T232342Z_a11/remote/artifacts/v3a-python-path1/20260412T232342Z_a11/reference-ssm-hybrid-gated-deltanet-p20-mamba3-torch-schedule-rrrrrarrrrrs/report.json` |
| 3 | `GDN + Mamba3` | `5.6836` | `3.2510` | `39.98` | `93.16 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260412T232129Z_a10/remote/artifacts/v3a-python-path1/20260412T232129Z_a10/reference-ssm-hybrid-gated-deltanet-mamba3-torch-schedule-rrrrrarrrrrs/report.json` |
| 4 | `P20 + Mamba3` | `5.7637` | `3.4058` | `51.00` | `89.57 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260412T231757Z_a09/remote/artifacts/v3a-python-path1/20260412T231757Z_a09/reference-ssm-hybrid-p20-mamba3-torch-schedule-rrrrrarrrrrs/report.json` |

H100 promotion result:

| lane | env | initial loss | final loss | train tok/s | CUDA peak MB | local report |
|---|---|---:|---:|---:|---:|---|
| `GDN + P20` | `primitive-triton` | `5.6758` | `2.6819` | `47.35` | `126.67 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260412T232737Z_a12/remote/artifacts/v3a-python-path1/20260412T232737Z_a12/reference-ssm-hybrid-gated-deltanet-p20-torch-schedule-rrrrrarrrrrs/report.json` |

Interpretation:

- `GDN + P20` won the four-lane RTX 5090 smoke and improved strongly when
  promoted to a 512-step H100 run.
- Adding Mamba3 did not help this small surface. `GDN + P20 + Mamba3` was close
  in smoke loss but slower and higher-memory; `P20 + Mamba3` was the weakest
  quality lane.
- The result argues for a focused `GDN + P20` design loop rather than a
  three-way omnibus block. The useful combination appears to be targeted
  GDN-style writes plus P20's state transform/readout path, not "all recurrent
  ideas at once."
- Do not overread this as proof that the current P20 adapter is the right final
  contract. The adapter preserves the frozen P20 scan/readout interface, which
  is useful for controlled comparison, but it may also be forcing P20 through a
  shape that is awkward for GDN-style scan composition. The next design loop
  should explicitly compare three options: keep the adapter and improve its
  contract, change P20 itself to expose cleaner scan-native hooks, or promote a
  nearby primitive that naturally composes with GDN-style targeted writes.
- The H100 promotion beat the prior torch GDN topology probe (`2.6819` versus
  `2.8584`) on the same 512-step bounded surface, but it is still a torch
  reference/composite implementation and not a throughput-optimized runtime.

Reproduction:

```bash
GPU_ID="NVIDIA GeForce RTX 5090" \
CLOUD_TYPE=SECURE \
RUN_TIMEOUT_SECONDS=14400 \
scripts/runpod-v3a-python-path1-composite-swa-smoke.sh \
  42 \
  128 \
  1 \
  v3a-python-path1-composite-swa-smoke
```

```bash
scripts/runpod-tournament.sh \
  --pod-name fractal-v3a-composite-swa-h100-s42-gdn-p20 \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --cloud-type SECURE \
  --binary-kind python \
  --binary-name scripts/v3a_python_path1.py \
  --python-requirements scripts/requirements-v3a-python-mamba3.txt \
  --python-install-mode primitive-triton \
  --run-timeout-seconds 14400 \
  --stop-after-run \
  -- \
  --variant reference-ssm-hybrid \
  --reference-ssm-profile gated-deltanet-p20-torch \
  --layer-schedule RRRRRARRRRRS \
  --backend cuda \
  --cuda-device 0 \
  --dtype bf16 \
  --env-kind primitive-triton \
  --primitive-runtime-backend torch \
  --seed 42 \
  --warmup-eval-batches 1 \
  --warmup-train-steps 1 \
  --local-window 256 \
  --jsonl-train-path /Users/joseph/fractal/experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl \
  --jsonl-eval-path /Users/joseph/fractal/experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/eval.jsonl \
  --seq-len 16 \
  --window-stride 16 \
  --batch-size 1 \
  --steps 512 \
  --eval-batches 1 \
  --run-label v3a-python-path1-composite-swa-h100-s42-gated-deltanet-p20-torch-schedule-rrrrrarrrrrs \
  --output table
```

## GDN / P20 Diagnostic Ablation

Date: 2026-04-13

This follow-up tested whether the `GDN + P20` quality bump looked like true
complementarity or mostly like an expensive two-branch ensemble. It also added
report-level diagnostics for composite branch weights.

Implementation additions:

- `diagnostics` field in Path 1 JSON reports
- composite branch weight summaries per trained report
- `ReferenceSsmProfile.P20_TORCH`
- `ReferenceSsmProfile.P20_THIN_TORCH`
- `ReferenceSsmProfile.GATED_DELTANET_P20_THIN_TORCH`
- diagnostic launcher:
  `/Users/joseph/fractal/scripts/runpod-v3a-python-path1-gdn-p20-diagnostic.sh`

Run contract:

- RunPod GPU: `NVIDIA GeForce RTX 5090`
- env: `primitive-triton`
- seed: `42`
- steps: `128`
- eval batches: `1`
- sequence length: `16`
- batch size: `1`
- dtype: `bf16`
- schedule: `RRRRRARRRRRS`
- topology: `[5xR] -> SWA -> [5xR] -> SWA_shared`
- local window: `256`

Results:

| rank by loss | lane | initial loss | final loss | train tok/s | CUDA peak MB | branch summary | local report |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | `GDN + P20` | `5.6758` | `3.1824` | `45.57` | `78.92 MB` | `gdn=0.4992`, `p20=0.5008` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T022047Z_a15/remote/artifacts/v3a-python-path1/20260413T022047Z_a15/reference-ssm-hybrid-gated-deltanet-p20-torch-schedule-rrrrrarrrrrs/report.json` |
| 2 | `GDN only` | `5.7539` | `3.2019` | `63.31` | `66.20 MB` | n/a | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T021236Z_a13/remote/artifacts/v3a-python-path1/20260413T021236Z_a13/reference-ssm-hybrid-gated-deltanet-torch-schedule-rrrrrarrrrrs/report.json` |
| 3 | `P20 only` | `5.6582` | `3.2075` | `99.38` | `62.61 MB` | n/a | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T022002Z_a14/remote/artifacts/v3a-python-path1/20260413T022002Z_a14/reference-ssm-hybrid-p20-torch-schedule-rrrrrarrrrrs/report.json` |
| 4 | `GDN + thin-P20` | `5.9863` | `3.2439` | `41.42` | `72.62 MB` | `gdn=0.4999`, `p20_thin=0.5001` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T022158Z_a16/remote/artifacts/v3a-python-path1/20260413T022158Z_a16/reference-ssm-hybrid-gated-deltanet-p20-thin-torch-schedule-rrrrrarrrrrs/report.json` |

Interpretation:

- The full `GDN + P20` branch still wins this 128-step smoke, but only by a
  small margin over GDN-only and P20-only.
- The branch gate stayed almost exactly `50/50`, so the smoke does not show the
  composite learning a sparse branch preference.
- Thin P20 did not preserve the quality bump and was not meaningfully faster
  than full `GDN + P20` in this reference implementation. That weakens the
  "P20 as cheap auxiliary adapter" hypothesis, at least in this simple
  bottleneck form.
- The result points away from optimizing the current parallel composite.
  The better next target is a fused or redesigned scan law that shares
  projections/state between GDN-style targeted writes and a P20-like transform,
  or a nearby primitive that composes more naturally with the GDN update.

Reproduction:

```bash
GPU_ID="NVIDIA GeForce RTX 5090" \
CLOUD_TYPE=SECURE \
RUN_TIMEOUT_SECONDS=14400 \
scripts/runpod-v3a-python-path1-gdn-p20-diagnostic.sh \
  42 \
  128 \
  1 \
  v3a-python-path1-gdn-p20-diagnostic
```

## Fused GDN/P20 Role Candidate

Date: 2026-04-13

This pass tested a Fractal-native fused block rather than another parallel
composite. The block keeps a typed recurrent state:

- vector state `h_t`: P20-like block-diagonal rotary transition
- matrix state `M_t`: GDN-style delta-KV associative memory

The P20-like vector transition shapes the value written into the GDN-style
matrix state. This is closer to "P20 in the GDN role" than
`mix(GDN(x), P20(x))`.

Run contract:

- RunPod GPU: `NVIDIA GeForce RTX 5090`
- env: `primitive-triton`
- seed: `42`
- steps: `128`
- eval batches: `1`
- sequence length: `16`
- batch size: `1`
- dtype: `bf16`
- schedule: `RRRRRARRRRRS`
- topology: `[5xR] -> SWA -> [5xR] -> SWA_shared`
- profile: `gated-deltanet-p20-fused-torch`

Results:

| lane | initial loss | final loss | train tok/s | CUDA peak MB | local report |
|---|---:|---:|---:|---:|---|
| fused GDN/P20, per-step readout | `5.7207` | `3.1716` | `25.33` | `78.87 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T023134Z_a17/remote/artifacts/v3a-python-path1/20260413T023134Z_a17/reference-ssm-hybrid-gated-deltanet-p20-fused-torch-schedule-rrrrrarrrrrs/report.json` |
| fused GDN/P20, sequence-level readout | `5.7207` | `3.1677` | `60.41` | `78.87 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T024358Z_a18/remote/artifacts/v3a-python-path1/20260413T024358Z_a18/reference-ssm-hybrid-gated-deltanet-p20-fused-torch-schedule-rrrrrarrrrrs/report.json` |
| fused GDN/P20, sequence-level readout, H100 promotion | `5.7227` | `2.6879` | `54.53` | `126.62 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T031229Z_a19/remote/artifacts/v3a-python-path1/20260413T031229Z_a19/reference-ssm-hybrid-gated-deltanet-p20-fused-torch-schedule-rrrrrarrrrrs/report.json` |

Comparison against the diagnostic ablation:

| lane | final loss | train tok/s | CUDA peak MB |
|---|---:|---:|---:|
| `P20 only` | `3.2075` | `99.38` | `62.61 MB` |
| `GDN only` | `3.2019` | `63.31` | `66.20 MB` |
| `GDN + P20` parallel composite | `3.1824` | `45.57` | `78.92 MB` |
| fused GDN/P20 sequence-level readout | `3.1677` | `60.41` | `78.87 MB` |

H100 promotion comparison:

| lane | steps | final loss | train tok/s | CUDA peak MB | local report |
|---|---:|---:|---:|---:|---|
| `GDN + P20` parallel composite | `512` | `2.6819` | `47.35` | `126.67 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260412T232737Z_a12/remote/artifacts/v3a-python-path1/20260412T232737Z_a12/reference-ssm-hybrid-gated-deltanet-p20-torch-schedule-rrrrrarrrrrs/report.json` |
| fused GDN/P20 sequence-level readout | `512` | `2.6879` | `54.53` | `126.62 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T031229Z_a19/remote/artifacts/v3a-python-path1/20260413T031229Z_a19/reference-ssm-hybrid-gated-deltanet-p20-fused-torch-schedule-rrrrrarrrrrs/report.json` |

Interpretation:

- The fused block beat the parallel composite on both loss and throughput in
  this 128-step RTX 5090 smoke.
- Moving output norm/projection out of the per-step loop more than doubled
  fused throughput, from `25.33` to `60.41` tok/s, with a small loss improvement.
- On the 512-step H100 promotion, fused was faster than the parallel composite
  (`54.53` vs `47.35` tok/s) with essentially identical memory, but slightly
  worse loss (`2.6879` vs `2.6819`).
- Peak memory did not move, which suggests the remaining memory cost is not the
  readout loop itself. It is more likely activation/autograd storage around the
  recurrent scan, local convolutions, and matrix-state intermediates.
- The fused contract is alive as a systems improvement, but the current fusion
  law has not yet proven a quality win at 512 H100 steps. The next iteration
  should target the update/readout law, not the already-retired parallel
  composite.

Reproduction:

```bash
scripts/runpod-tournament.sh \
  --pod-name fractal-v3a-gdnp-fused-readout-5090-s42 \
  --gpu-id "NVIDIA GeForce RTX 5090" \
  --cloud-type SECURE \
  --binary-kind python \
  --binary-name scripts/v3a_python_path1.py \
  --python-requirements scripts/requirements-v3a-python-mamba3.txt \
  --python-install-mode primitive-triton \
  --run-timeout-seconds 14400 \
  --stop-after-run \
  -- \
  --variant reference-ssm-hybrid \
  --reference-ssm-profile gated-deltanet-p20-fused-torch \
  --layer-schedule RRRRRARRRRRS \
  --backend cuda \
  --cuda-device 0 \
  --dtype bf16 \
  --env-kind primitive-triton \
  --primitive-runtime-backend torch \
  --seed 42 \
  --warmup-eval-batches 1 \
  --warmup-train-steps 1 \
  --local-window 256 \
  --jsonl-train-path /Users/joseph/fractal/experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl \
  --jsonl-eval-path /Users/joseph/fractal/experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/eval.jsonl \
  --seq-len 16 \
  --window-stride 16 \
  --batch-size 1 \
  --steps 128 \
  --eval-batches 1 \
  --run-label v3a-python-path1-gdnp-fused-readout-s42-gated-deltanet-p20-fused-torch-schedule-rrrrrarrrrrs \
  --output table
```

## Fused Law Ablation

Date: 2026-04-13

This pass tested whether the fused block needed a better update/readout law
rather than more parallel composition. It also changed the attention seam so
full-window causal attention can use mask-free SDPA (`is_causal=True`) instead
of forcing an additive local mask when `local_window >= seq_len`.

Run contract:

- RunPod GPU: `NVIDIA GeForce RTX 5090`
- env: `primitive-triton`
- seed: `42`
- steps: `128`
- eval batches: `1`
- sequence length: `16`
- batch size: `1`
- dtype: `bf16`
- schedule: `RRRRRARRRRRS`
- topology: `[5xR] -> SWA -> [5xR] -> SWA_shared`
- launcher: `/Users/joseph/fractal/scripts/runpod-v3a-python-path1-gdnp-fused-law-ablation.sh`

Candidates:

- `value`: current fused law, `value_base + h_t`
- `beta`: P20 state modulates the GDN beta/write gate
- `qkv`: P20 state modulates GDN query/key/value channels
- `residual-readout`: matrix read and vector read are separate readout paths
- `multi-read`: GDN reads with both the base query and a P20-derived auxiliary query
- `all`: combined beta + qkv + residual readout + multi-read

Results:

| rank by loss | law | profile | initial loss | final loss | train tok/s | CUDA peak MB | local report |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `multi-read` | `gated-deltanet-p20-fused-multi-read-torch` | `5.7305` | `2.9136` | `49.55` | `82.02 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T035458Z_a24/remote/artifacts/v3a-python-path1/20260413T035458Z_a24/reference-ssm-hybrid-gated-deltanet-p20-fused-multi-read-torch-schedule-rrrrrarrrrrs/report.json` |
| 2 | `residual-readout` | `gated-deltanet-p20-fused-residual-readout-torch` | `5.6699` | `3.1226` | `38.56` | `78.92 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T035349Z_a23/remote/artifacts/v3a-python-path1/20260413T035349Z_a23/reference-ssm-hybrid-gated-deltanet-p20-fused-residual-readout-torch-schedule-rrrrrarrrrrs/report.json` |
| 3 | `qkv` | `gated-deltanet-p20-fused-qkv-torch` | `5.7207` | `3.1370` | `38.71` | `78.95 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T035240Z_a22/remote/artifacts/v3a-python-path1/20260413T035240Z_a22/reference-ssm-hybrid-gated-deltanet-p20-fused-qkv-torch-schedule-rrrrrarrrrrs/report.json` |
| 4 | `beta` | `gated-deltanet-p20-fused-beta-torch` | `5.7227` | `3.1775` | `41.59` | `78.90 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T035136Z_a21/remote/artifacts/v3a-python-path1/20260413T035136Z_a21/reference-ssm-hybrid-gated-deltanet-p20-fused-beta-torch-schedule-rrrrrarrrrrs/report.json` |
| 5 | `value` | `gated-deltanet-p20-fused-torch` | `5.7207` | `3.1802` | `51.82` | `78.87 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T034249Z_a20/remote/artifacts/v3a-python-path1/20260413T034249Z_a20/reference-ssm-hybrid-gated-deltanet-p20-fused-torch-schedule-rrrrrarrrrrs/report.json` |
| 6 | `all` | `gated-deltanet-p20-fused-all-torch` | `5.7754` | `3.1838` | `33.87` | `82.19 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T035555Z_a25/remote/artifacts/v3a-python-path1/20260413T035555Z_a25/reference-ssm-hybrid-gated-deltanet-p20-fused-all-torch-schedule-rrrrrarrrrrs/report.json` |

Interpretation:

- `multi-read` is the clear winner in the 128-step RTX 5090 smoke. It improves
  final loss by `0.2666` over the fresh `value` fused control.
- The quality gain is not free: `multi-read` adds about `3.15 MB` peak memory
  and is slower than `value`, but it is still faster than the older parallel
  composite's 5090 diagnostic run.
- `residual-readout` and `qkv` both improve loss but cost substantial
  throughput. They may be useful as second-stage ingredients, but not stacked
  blindly.
- `all` failed to combine the wins. The law changes interact destructively in
  this simple composition, so the next iteration should build outward from
  `multi-read`, not from the combined profile.
- The H100 promotion candidate is `gated-deltanet-p20-fused-multi-read-torch`.

H100 promotion comparison:

| lane | profile | steps | final loss | train tok/s | CUDA peak MB | local report |
|---|---|---:|---:|---:|---:|---|
| `GDN + P20` parallel composite | `gated-deltanet-p20-torch` | `512` | `2.6819` | `47.35` | `126.67 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260412T232737Z_a12/remote/artifacts/v3a-python-path1/20260412T232737Z_a12/reference-ssm-hybrid-gated-deltanet-p20-torch-schedule-rrrrrarrrrrs/report.json` |
| fused value readout | `gated-deltanet-p20-fused-torch` | `512` | `2.6879` | `54.53` | `126.62 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T031229Z_a19/remote/artifacts/v3a-python-path1/20260413T031229Z_a19/reference-ssm-hybrid-gated-deltanet-p20-fused-torch-schedule-rrrrrarrrrrs/report.json` |
| fused multi-read | `gated-deltanet-p20-fused-multi-read-torch` | `512` | `2.5156` | `49.45` | `129.77 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T035743Z_a26/remote/artifacts/v3a-python-path1/20260413T035743Z_a26/reference-ssm-hybrid-gated-deltanet-p20-fused-multi-read-torch-schedule-rrrrrarrrrrs/report.json` |

H100 interpretation:

- `multi-read` promoted cleanly. It improved final loss by `0.1663` versus the
  earlier parallel composite and by `0.1723` versus the fused value-readout
  profile.
- `multi-read` was also faster than the parallel composite on this surface
  (`49.45` vs `47.35` tok/s), though slower than fused value-readout.
- Peak memory increased by about `3.1 MB` versus the earlier H100 lanes. The
  memory increase is consistent with storing/reading a second matrix-read stream.
- This is the strongest evidence so far that the right role for P20 inside a
  GDN-like block is to provide an auxiliary read/query path over the recurrent
  matrix state, not merely to perturb the written value.

Reproduction:

```bash
GPU_ID="NVIDIA GeForce RTX 5090" \
CLOUD_TYPE=SECURE \
RUN_TIMEOUT_SECONDS=14400 \
scripts/runpod-v3a-python-path1-gdnp-fused-law-ablation.sh \
  42 \
  128 \
  1 \
  v3a-python-path1-gdnp-fused-law-ablation
```

H100 promotion reproduction:

```bash
scripts/runpod-tournament.sh \
  --pod-name fractal-v3a-gdnp-fused-multiread-h100-s42 \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --cloud-type SECURE \
  --binary-kind python \
  --binary-name scripts/v3a_python_path1.py \
  --python-requirements scripts/requirements-v3a-python-mamba3.txt \
  --python-install-mode primitive-triton \
  --run-timeout-seconds 14400 \
  --stop-after-run \
  -- \
  --variant reference-ssm-hybrid \
  --reference-ssm-profile gated-deltanet-p20-fused-multi-read-torch \
  --layer-schedule RRRRRARRRRRS \
  --backend cuda \
  --cuda-device 0 \
  --dtype bf16 \
  --env-kind primitive-triton \
  --primitive-runtime-backend torch \
  --seed 42 \
  --warmup-eval-batches 1 \
  --warmup-train-steps 1 \
  --local-window 256 \
  --jsonl-train-path /Users/joseph/fractal/experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl \
  --jsonl-eval-path /Users/joseph/fractal/experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/eval.jsonl \
  --seq-len 16 \
  --window-stride 16 \
  --batch-size 1 \
  --steps 512 \
  --eval-batches 1 \
  --run-label v3a-python-path1-gdnp-fused-multiread-h100-s42-gated-deltanet-p20-fused-multi-read-torch-schedule-rrrrrarrrrrs \
  --output table
```

## FLA Kernel Contract Probe

Date: 2026-04-13

This pass tested whether the GDN/P20 quality winner could be accelerated by
mapping it onto the Flash Linear Attention gated-delta-rule kernel contract
before investing in a bespoke Fractal Triton kernel.

Run contract:

- RunPod GPU: `NVIDIA H100 80GB HBM3`
- env: `primitive-triton`
- seed: `42`
- steps: `512`
- eval batches: `1`
- sequence length: `16`
- batch size: `1`
- dtype: `bf16`
- schedule: `RRRRRARRRRRS`
- topology: `[5xR] -> SWA -> [5xR] -> SWA_shared`

Results:

| lane | profile | implementation | final loss | train tok/s | CUDA peak MB | local report |
|---|---|---|---:|---:|---:|---|
| Torch fused multi-read control | `gated-deltanet-p20-fused-multi-read-torch` | `python_reference_ssm_gdnp_fused_multi_read_torch` | `2.5156` | `49.45` | `129.77 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T035743Z_a26/remote/artifacts/v3a-python-path1/20260413T035743Z_a26/reference-ssm-hybrid-gated-deltanet-p20-fused-multi-read-torch-schedule-rrrrrarrrrrs/report.json` |
| pure FLA GDN topology | `gated-deltanet-fla` | `python_reference_ssm_gated_deltanet_fla` | `2.5503` | `235.84` | `113.93 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T052537Z_a28/remote/artifacts/v3a-python-path1/20260413T052537Z_a28/reference-ssm-hybrid-gated-deltanet-fla-schedule-rrrrrarrrrrs/report.json` |
| FLA GDN plus P20 conditioning | `gated-deltanet-fla-p20-compatible` | `python_reference_ssm_gdnp_fla_compatible` | `2.5642` | `275.60` | `139.20 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T053650Z_a29/remote/artifacts/v3a-python-path1/20260413T053650Z_a29/reference-ssm-hybrid-gated-deltanet-fla-p20-compatible-schedule-rrrrrarrrrrs/report.json` |
| FLA GDN plus P20 multi-read | `gated-deltanet-fla-p20-multi-read` | `python_reference_ssm_gdnp_fla_compatible_multi_read` | `2.5994` | `207.88` | `145.45 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T054454Z_a30/remote/artifacts/v3a-python-path1/20260413T054454Z_a30/reference-ssm-hybrid-gated-deltanet-fla-p20-multi-read-schedule-rrrrrarrrrrs/report.json` |

Environment note:

- The first FLA attempt used the older requirements-only install lane and failed
  before training with `ImportError: cannot import name 'Replicate' from
  'torch.distributed.tensor'`.
- Rerunning under the `primitive-triton` env fixed the FLA dependency contract.
  FLA work should stay on that stack unless the requirements file is revised.

Interpretation:

- Pure FLA GDN is the strongest off-the-shelf kernel result so far: it is about
  `4.77x` faster than the Torch fused multi-read control while losing only
  `0.0347` final loss on this short H100 surface.
- The single-read FLA-compatible GDN/P20 adapter is faster than pure FLA, but it
  is worse on quality. That means "P20 as conditioner around FLA GDN" is not
  preserving the winning law.
- The FLA-compatible multi-read variant preserved the high-level multi-read
  shape but did not preserve the win. It was slower than single-read and worse
  than pure FLA on loss.
- The decision gate is therefore closed for the FLA-compatible adapter path.
  The next acceleration target should preserve the Torch fused multi-read law
  directly, either through a bespoke Fractal Triton scan or a staged kernel path
  that keeps the matrix-state update/readout semantics unchanged.

## Fractal Triton Vector-Scan Probe

Date: 2026-04-13

After the FLA-compatible adapter failed the quality gate, this pass tested the
smallest faithful Fractal Triton acceleration step: keep the Torch fused
multi-read GDN/P20 matrix update/readout law unchanged, but route the embedded
P20 vector-state scan through the existing Triton block-diagonal sequence-scan
kernel.

Run contract:

- RunPod GPU: `NVIDIA H100 80GB HBM3`
- env: `primitive-triton`
- seed: `42`
- steps: `512`
- eval batches: `1`
- sequence length: `16`
- batch size: `1`
- dtype: `bf16`
- schedule: `RRRRRARRRRRS`
- topology: `[5xR] -> SWA -> [5xR] -> SWA_shared`

Result:

| lane | profile | implementation | final loss | train tok/s | CUDA peak MB | local report |
|---|---|---|---:|---:|---:|---|
| Torch fused multi-read control | `gated-deltanet-p20-fused-multi-read-torch` | `python_reference_ssm_gdnp_fused_multi_read_torch` | `2.5156` | `49.45` | `129.77 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T035743Z_a26/remote/artifacts/v3a-python-path1/20260413T035743Z_a26/reference-ssm-hybrid-gated-deltanet-p20-fused-multi-read-torch-schedule-rrrrrarrrrrs/report.json` |
| staged Fractal Triton vector scan | `gated-deltanet-p20-fused-multi-read-torch` | `python_reference_ssm_gdnp_fused_multi_read_triton_vector` | `2.6560` | `79.97` | `129.77 MB` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-908dac1101b0fa89/20260413T060008Z_a31/remote/artifacts/v3a-python-path1/20260413T060008Z_a31/reference-ssm-hybrid-gated-deltanet-p20-fused-multi-read-torch-schedule-rrrrrarrrrrs/report.json` |

Interpretation:

- The staged Triton vector scan improved throughput by about `1.62x` versus the
  Torch fused multi-read control.
- It did not improve memory, because the matrix-state recurrence and multi-read
  activations still dominate this tiny surface.
- The quality regression is too large to promote: final loss worsened by
  `0.1404`.
- This result closes the "easy staged vector-scan only" path for now. The next
  Triton work should start with an explicit forward/backward parity harness for
  the fused GDN/P20 recurrence under matrix-read supervision, then move the
  matrix-state update/readout itself into the kernel if parity holds.
- A CUDA parity harness now exists at
  `/Users/joseph/fractal/scripts/check_gdnp_triton_vector_parity.py`. It
  compares the Torch fused GDN/P20 path against the Triton runtime path with the
  same weights/input and checks forward, input-gradient, and parameter-gradient
  deltas.

## Fractal Triton Matrix Kernel Parity Gate

Date: 2026-04-13

This pass extended the staged Triton lane from vector-scan only to a first
matrix-state kernel slice. The new path keeps the Torch fused GDN/P20
multi-read law as the teacher, routes the P20 vector scan through Triton, and
routes the GDN-style matrix-state update plus primary/auxiliary matrix
forward/backward through Triton sequence kernels.

Current implementation contract:

- profile: `gated-deltanet-p20-fused-multi-read-torch`
- runtime backend: `triton`
- implementation kind:
  `python_reference_ssm_gdnp_fused_multi_read_triton_vector_matrix`
- matrix forward: Triton sequence kernel with fp32 state history workspace
- matrix backward: Triton reverse sequence kernel
- purpose: parity-gated training runtime candidate

Local validation:

- `py_compile` passed for the Triton runtime, GDN/P20 mixer, CLI, tests, and
  parity harness.
- `python.tests.test_models.Path1ModelTests.test_gdnp_fused_triton_policy_routes_vector_scan_to_sequence_kernel`
  passed with both vector and matrix backend calls asserted.
- `python.tests.test_specs.Path1SpecTests` and
  `python.tests.test_models.Path1ModelTests` passed locally.
- `git diff --check` passed for the touched Triton/parity files.

H100 parity result:

| gate | dtype | shape | result | forward abs | forward rel | input-grad abs | input-grad rel | param-grad abs | param-grad rel | worst param | local artifact |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Torch teacher vs Triton vector+matrix forward/backward | `fp32` | `batch=2, seq=16, d_model=128, heads=4` | passed | `3.7252903e-08` | `0.0013201747` | `1.4551915e-11` | `8.6970431e-06` | `5.8207661e-10` | `0.0001658691` | `output_projection.weight` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-69b8d45944e0f440/20260413T074152Z_a05/remote/logs/latest.log` |
| Torch teacher vs Triton vector+matrix forward/backward | `bf16` | `batch=2, seq=16, d_model=128, heads=4` | passed | `0.0009765625` | `1.4171429` | `7.1525574e-07` | `0.34971163` | `1.335144e-05` | `1.9728507` | `output_projection.weight` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-69b8d45944e0f440/20260413T074733Z_a06/remote/logs/latest.log` |

Interpretation:

- The matrix forward/backward Triton slice is numerically faithful in fp32
  under allclose-style tolerances. The maximum relative forward delta is caused
  by near-zero outputs; the maximum absolute forward delta is only `3.7e-08`.
- The BF16 check also passes under BF16-appropriate absolute tolerances. The
  large max-relative numbers again come from near-zero values; the absolute
  output and gradient deltas remain small.
- The parity harness now uses the same tolerance shape as `torch.allclose`
  instead of independently failing on maximum relative deltas. It still reports
  max-relative values as diagnostics.
- This promotes the lane from forward scaffold to a parity-gated training
  runtime candidate. It still needs a short CUDA train smoke before any
  throughput claim.
- The next gates are: short CUDA train smoke, then 120s H100 benchmark only if
  the smoke remains stable.

## Fractal Triton Matrix Kernel Smoke And Benchmark

Date: 2026-04-13

After fp32 and BF16 parity passed, this pass ran the Triton matrix-kernel lane
against the Torch fused multi-read control on the same H100 pod. The benchmark
profile resolved to the full local faithful-small surface (`1925` train steps,
`189` eval batches, `seq_len=16`, `batch_size=1`), so the nominal smoke became
a full tiny-corpus train/eval pass.

Run contract:

- RunPod GPU: `NVIDIA H100 80GB HBM3`
- env: `primitive-triton`
- seed: `42`
- dtype: `bf16`
- benchmark profile: `cuda-faithful-small-v1`
- schedule: `RRRRRARRRRRS`
- topology: `[5xR] -> SWA -> [5xR] -> SWA_shared`
- profile: `gated-deltanet-p20-fused-multi-read-torch`

Result:

| lane | implementation | train steps | initial loss | final loss | train tok/s | CUDA peak MB | local report |
|---|---|---:|---:|---:|---:|---:|---|
| Triton matrix-kernel smoke | `python_reference_ssm_gdnp_fused_multi_read_triton_vector_matrix` | `1925` | `5.7545` | `2.2968` | `317.69` | `129.77` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-589e3e101a2e8613/20260413T081152Z_a02/remote/artifacts/v3a-python-path1/20260413T081152Z_a02/reference-ssm-hybrid-gated-deltanet-p20-fused-multi-read-torch-schedule-rrrrrarrrrrs/report.json` |
| Torch fused multi-read control | `python_reference_ssm_gdnp_fused_multi_read_torch` | `1925` | `5.7545` | `2.2640` | `45.25` | `129.77` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-589e3e101a2e8613/20260413T081431Z_a03/remote/artifacts/v3a-python-path1/20260413T081431Z_a03/reference-ssm-hybrid-gated-deltanet-p20-fused-multi-read-torch-schedule-rrrrrarrrrrs/report.json` |
| Triton matrix-kernel repeat | `python_reference_ssm_gdnp_fused_multi_read_triton_vector_matrix` | `1925` | `5.7545` | `2.2968` | `324.13` | `129.77` | `/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-589e3e101a2e8613/20260413T082737Z_a04/remote/artifacts/v3a-python-path1/20260413T082737Z_a04/reference-ssm-hybrid-gated-deltanet-p20-fused-multi-read-torch-schedule-rrrrrarrrrrs/report.json` |

Interpretation:

- The Triton matrix-kernel lane is now a real training candidate, not just a
  parity scaffold: it trains end-to-end on the local faithful-small surface.
- Throughput improved from `45.25 tok/s` to `324.13 tok/s`, about `7.16x`
  faster than the Torch fused multi-read control at the same CUDA peak memory.
- Quality still trails the Torch teacher on this tiny surface by about `0.033`
  loss, so the acceleration path is promising but not yet a final promotion.
- The lane no longer appears to be slower than the previously measured FLA GDN
  lanes, so the immediate "profile because it still lags FLA badly" gate is
  not triggered.
- Next work should compare this Triton fused multi-read lane against pure FLA
  GDN and FLA GDN + P20 single-read on the same benchmark surface, then decide
  whether to spend effort on quality recovery or Parameter Golf integration.

## Reproduction

```bash
GPU_ID="NVIDIA H100 80GB HBM3" \
CLOUD_TYPE=SECURE \
RUN_TIMEOUT_SECONDS=14400 \
scripts/runpod-v3a-python-path1-gdn-head2head.sh \
  42 \
  512 \
  1 \
  v3a-python-path1-gdn-head2head
```
