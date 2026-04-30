# CUDA SDPA Under-Hood Note

Timestamp: 2026-04-24

## Why this exists

The long-context Parcae/RGRP lane showed a real speed tax after positional and
shape tuning. Earlier evidence showed the RGRP scan itself was small, so this
note isolates the CUDA attention path instead of treating end-to-end throughput
as a black box.

## Microbench surface

Artifact:

- `experiments/aws_sagemaker/cuda_sdpa_microbench/fractal-cuda-sdpa-l4-20260424T-underhood/extracted/sdpa-microbench/summary.md`

Hardware/runtime:

- SageMaker `ml.g6.2xlarge`
- NVIDIA L4, compute capability 8.9
- PyTorch `2.5.1+cu124`
- bf16, batch 32, seq len 512, local window 128
- forward + backward SDPA timing

Shapes tested:

| Shape | d_model | heads | head_dim | Why |
|---|---:|---:|---:|---|
| `attn_d448_h7` | 448 | 7 | 64 | prior attention control |
| `rgrp_d448_h8` | 448 | 8 | 56 | speed-recovered RGRP under budget |
| `attn_d456_h19` | 456 | 19 | 24 | matched-parameter attention control |
| `rgrp_d480_h10` | 480 | 10 | 48 | current under-50M RGRP candidate |
| `rgrp_d512_h8` | 512 | 8 | 64 | over-budget quality scout |
| `slow_d472_h8` | 472 | 8 | 59 | known slow-path shape |

## Key result

PyTorch SDPA flash attention is available for causal full attention, but the
current dense local additive/bool mask cannot use flash attention. CUDA-friendly
head dimensions fall back to efficient attention. The bad `head_dim=59` shape
falls all the way to math attention for local masks.

| Shape | causal auto ms | local-additive auto ms | local/causal | local auto backend |
|---|---:|---:|---:|---|
| `attn_d448_h7` | 2.510 | 3.351 | 1.33x | efficient |
| `rgrp_d448_h8` | 2.632 | 3.774 | 1.43x | efficient |
| `attn_d456_h19` | 2.869 | 6.323 | 2.20x | efficient |
| `rgrp_d480_h10` | 2.998 | 4.314 | 1.44x | efficient |
| `rgrp_d512_h8` | 3.019 | 4.068 | 1.35x | efficient |
| `slow_d472_h8` | 3.719 | 29.503 | 7.93x | math |

## Interpretation

The earlier `d472/h8` slowdown was not an RGRP scan problem. It was a hidden
attention-kernel contract violation: `head_dim=59` plus a dense local mask forced
math SDPA.

The remaining d480 speed tax is more subtle. Its head dimension is CUDA-friendly,
but local masking still disables flash attention. At seq len 512, "local"
attention is not a real windowed flash kernel in this path; it is dense masked
efficient attention. That means local attention can be slower than full causal
flash despite seeing fewer logical tokens.

## Immediate consequences

- Keep the head-dimension guard: CUDA attention head dimensions must be multiples
  of 8 unless an experiment explicitly opts into a slow-path probe.
- Treat dense local attention masks as a runtime risk. They preserve semantics,
  but they do not express a true windowed CUDA kernel contract.
- Do not prioritize the RGRP scan kernel first. The scan is already small in the
  timing profile; attention mask/backend selection is the larger speed lever.

## Next runtime lanes

1. Run model-level CUDA timing for the current d480 RGRP lane with
   `local_window=512` so it uses full causal flash attention. This is a speed
   diagnostic, not a final architecture claim.
2. If full causal flash materially improves throughput, run a short quality
   ablation at seq len 512 to see whether full attention is acceptable or even
   better at this rung.
3. For longer context, implement a real local/windowed CUDA attention path
   instead of dense additive masks. Candidate surfaces are FlashAttention with a
   window contract if available, or a small Triton banded attention kernel.
4. Only after the attention path is honest should we revisit custom RGRP/Parcae
   fusion.

## Model-level timing follow-up

Artifacts:

- `experiments/aws_sagemaker/path1_cuda_scout/fractal-lctx-rgrp480-timing-w512-0424-underhood/extracted/path1-cuda-scout/timing-parcae-hourglass-p20-control-looped-attention/cuda_timing.json`
- `experiments/aws_sagemaker/path1_cuda_scout/fractal-lctx-rgrp480-timing-w128-0424-underhood/extracted/path1-cuda-scout/timing-parcae-hourglass-p20-control-looped-attention/cuda_timing.json`

Both jobs used the current under-50M d480 RGRP hourglass candidate:

- `d_model=480`
- `head_count=10`
- `total_layers=8`
- `parcae_loop_d_model=256`
- `parcae_loop_head_count=8`
- `parcae_loop_count=2`
- `parcae_backward_steps=1`
- `seq_len=512`
- `batch_size=32`
- `bf16`
- Triton RGRP scan backend

| Attention window | step ms | approx tok/s | forward ms | backward ms | SDPA mean ms | attention total ms | Parcae total ms | primitive total ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_window=512` | 207.789 | 78,849 | 66.381 | 124.746 | 0.303 | 516.5 | 827.9 | 28.9 |
| `local_window=128` | 227.339 | 72,069 | 76.342 | 134.002 | 0.660 | 583.8 | 914.6 | 27.5 |

Interpretation:

- Full causal flash is about 9.4 percent faster end-to-end than the dense local
  mask path for the same d480 RGRP model shape.
- The local-mask path roughly doubles the measured SDPA region per attention
  call (`0.660 ms` vs `0.303 ms`).
- The RGRP primitive remains tiny in both cases (`~28 ms` inclusive over all
  timed steps). The speed tax is not coming from the scan.
- For seq len 512, full causal flash is the better runtime contract. For longer
  context, we need a true windowed attention kernel before treating
  `local_window` as a speed optimization.

## FlexAttention windowed-kernel rung

After the SDPA diagnosis, we added an explicit `attention_kernel` model contract:

- `sdpa`: existing PyTorch `scaled_dot_product_attention` path.
- `flex-local`: PyTorch FlexAttention with a causal local block mask.
- `flash-local`: optional Dao-AILab FlashAttention local-window path when the
  `flash-attn` package is installed.

This is not the final custom CUDA layer, but it is the first deeper kernel rung:
the model now expresses the local-window contract as a block mask instead of as a
dense additive mask.

The second rung is `flash-local`: it uses FlashAttention's native
`window_size=(local_window - 1, 0)` causal-local API. That lets us test whether a
non-PyTorch local FlashAttention package already covers the d480/head_dim48
quality-favored shape before writing our own Triton backward.
The package metadata checked before adding the install hook is minimal:
`flash-attn==2.8.3` declares only `torch` and `einops` runtime dependencies.

FlexAttention microbench artifact:

- `experiments/aws_sagemaker/cuda_sdpa_microbench/fractal-cuda-flex-local-l4-20260424-underhood/extracted/sdpa-microbench/summary.md`

Microbench finding:

| Shape | SDPA local ms | Flex local ms | Result |
|---|---:|---:|---|
| `attn_d448_h7`, head_dim 64 | 3.355 | 1.887 | Flex wins |
| `rgrp_d480_h10`, head_dim 48 | 4.219 | unsupported | Flex requires power-of-two head dim |
| `rgrp_d512_h8`, head_dim 64 | 3.985 | 59.276 | Flex path regressed on this shape/runtime |
| `slow_d472_h8`, head_dim 59 | 29.481 | unsupported | non-power-of-two unsupported |

FlashAttention package probe:

- first install attempt failed because the `flash-attn` setup path found the
  prebuilt wheel URL but attempted an invalid cross-device rename while caching
  the wheel;
- the SageMaker microbench/scout install hooks now place pip cache, temp, and
  cwd on the same mounted volume before installing `flash-attn`;
- retry artifact:
  `experiments/aws_sagemaker/cuda_sdpa_microbench/fractal-cuda-flash-local-probe-l4-20260424-r2/extracted/sdpa-microbench/summary.md`.

| Shape | Dense local SDPA ms | `flash-local` ms | Result |
|---|---:|---:|---|
| `rgrp_d480_h10`, head_dim 48 | 4.336 | 3.668 | `flash-local` wins by about 15% at kernel level |
| `attn_d448_h7`, head_dim 64 | 3.376 | 3.147 | `flash-local` wins by about 7% at kernel level |

Interpretation:

- FlexAttention is not a universal replacement in this PyTorch/runtime stack.
- It creates an architectural constraint: local-window FlexAttention wants
  power-of-two head dimensions.
- This is now an explicit model and SageMaker launcher contract, not a hidden
  runtime failure: `attention_kernel=flex-local` rejects non-power-of-two
  `head_dim` before a cloud training job is submitted.
- The current d480 candidate (`480/10 = 48`) cannot use this path.
- The old d448 attention-friendly geometry (`448/7 = 64`) can use it.

Model-level timing artifacts:

- `experiments/aws_sagemaker/path1_cuda_scout/fractal-lctx-rgrp448h7-sdpa-timing-0424-underhood/extracted/path1-cuda-scout/timing-parcae-hourglass-p20-control-looped-attention/cuda_timing.json`
- `experiments/aws_sagemaker/path1_cuda_scout/fractal-lctx-rgrp448h7-flex-timing-0424-underhood/extracted/path1-cuda-scout/timing-parcae-hourglass-p20-control-looped-attention/cuda_timing.json`

Both used:

- `d_model=448`
- `head_count=7`
- `parcae_loop_d_model=256`
- `parcae_loop_head_count=4`
- `local_window=128`
- `seq_len=512`
- `batch_size=32`
- `parcae_loop_count=2`
- `parcae_backward_steps=1`
- Triton RGRP scan backend

| Lane | Params | Kernel | step ms | approx tok/s | attention call mean | attention total ms | Parcae total ms |
|---|---:|---|---:|---:|---:|---:|---:|
| d448/h7 local SDPA | 45.61M | `sdpa` | 212.050 | 77,265 | 0.458 | 496.7 | 807.1 |
| d448/h7 local Flex | 45.61M | `flex-local` | 202.980 | 80,717 | 0.250 | 454.2 | 760.7 |
| d480/h10 local SDPA | 49.81M | `sdpa` | 227.339 | 72,069 | 0.660 | 583.8 | 914.6 |
| d480/h10 local FlashAttention | 49.81M | `flash-local` | 218.292 | 75,056 | 0.236 | 495.3 | 893.9 |
| d480/h10 full causal flash | 49.81M | `sdpa` | 207.789 | 78,849 | 0.303 | 516.5 | 827.9 |

Current interpretation:

- Kernel-aware shape co-design matters. The d448/h7 + loop256/h4 shape keeps
  head dimensions at 64 and lets FlexAttention express the local window.
- Flex local gives a further roughly 4.3 percent step-time improvement over the
  exact same d448/h7 SDPA local-mask model.
- FlashAttention local windows recover about 4.0 percent model-level step time
  for the d480/h10 shape, but do not erase the full speed tax: after the
  attention call is fixed, attention is only about 11 percent of the timed step.
- The best timed local-window RGRP path is now d448/h7 `flex-local`, not d480/h10
  `sdpa`.
- This restores local-window speed without giving up the local-window semantics,
  but it also shows the deeper custom-kernel requirement: a hand-owned
  Triton/CUDA kernel should not have to reject head_dim 48.

Next runtime decision:

1. Run a short quality comparison for d448/h7 `flex-local` against the prior
   d448/h8 and d480/h10 candidates. The math should match local attention
   semantics, but the head schedule changed, so quality must be checked.
2. If d448/h7 quality holds, promote it as the fast local-window lane.
3. If d480 quality remains necessary, write a custom Triton/CUDA banded attention
   path that supports head_dim 48 instead of contorting the architecture around
   FlexAttention.

## First quality check

Artifact:

- `experiments/aws_sagemaker/path1_cuda_scout/fractal-lctx-rgrp448h7-flex-8192-s42-0424/extracted/path1-cuda-scout/summary.md`
- `experiments/aws_sagemaker/path1_cuda_scout/fractal-lctx-attn448h7-flex-8192-s42-0424/extracted/path1-cuda-scout/summary.md`

Run shape:

- `d_model=448`
- `head_count=7`
- `attention_kernel=flex-local`
- `local_window=128`
- `parcae_loop_d_model=256`
- `parcae_loop_head_count=4`
- `seq_len=512`
- `batch_size=32`
- `steps=8192`
- `eval_batches=2`
- `seed=42`

Result:

| Lane | Params | Final loss | tok/s | Peak CUDA MB |
|---|---:|---:|---:|---:|
| d448/h7 attention-only flex-local | 48.44M | 4.2756 | 81,376.42 | 8,676.94 |
| d448/h7 RGRP flex-local | 45.61M | 4.2449 | 76,971.79 | 8,711.26 |

Interpretation:

- Quality did not collapse under the kernel-aware d448/h7 geometry.
- Under the same d448/h7 `flex-local` attention geometry, RGRP improves final
  loss by `0.0307` while using fewer parameters (`45.61M` vs `48.44M`).
- The cost is a real but bounded speed tax: `76,971.79` tok/s vs `81,376.42`
  tok/s, about `5.4%` slower. This is no longer the earlier 5x speed trap.
- RGRP remains slightly behind the prior d448/h8 RGRP (`4.2427`) and meaningfully
  behind the d480/h10 RGRP quality scout (`4.2226`), so the quality-favored shape
  still wants a local-window kernel that supports `head_dim=48`.
- The next kernel question is therefore narrow and concrete: can `flash-local`
  or an owned `triton-window` path recover d480/h10 local-window speed without
  sacrificing its better loss?

## Synchronized wall-clock timing rung

Artifact:

- `experiments/aws_sagemaker/path1_cuda_scout/fractal-lctx-rgrp480-flash-timing-syncwall-0424/path1-cuda-scout/timing-parcae-hourglass-p20-control-looped-attention/cuda_timing.json`

Why this rung mattered:

- CUDA event timing showed where GPU kernels spend time, but Python wall-clock
  timing initially underreported forward/backward regions because CUDA launches
  are asynchronous.
- The timing harness now includes a `path1.train.synchronize_step` boundary, so
  wall-clock `step_total` matches CUDA event `step_total`; any queued GPU work
  that has not completed is exposed as synchronization wait instead of being
  hidden outside the measured step.

Run shape:

- `d_model=480`
- `head_count=10`
- `attention_kernel=flash-local`
- `local_window=128`
- `parcae_loop_d_model=256`
- `parcae_loop_head_count=8`
- `seq_len=512`
- `batch_size=32`
- `timing_steps=20`

Key synchronized result:

| Region | CUDA event mean ms | Wall mean ms | Interpretation |
|---|---:|---:|---|
| full step | 219.621 | 219.734 | timing is now synchronized and trustworthy |
| forward | 77.263 | 21.906 | wall time is mostly launch time; GPU completion lands at sync |
| backward | 126.114 | 10.530 | same asynchronous launch pattern |
| optimizer | 15.547 | 10.409 | optimizer has more visible CPU/wall work |
| synchronize step | 0.036 | 176.029 | queued GPU work is being waited on here |
| materialize batch | 0.315 | 0.330 | data movement is not the bottleneck |
| flash-local attention call | 0.239 | 0.168 | attention kernel call is no longer the main bottleneck |
| Triton RGRP scan | 1.227 | 0.214 | RGRP scan is not the speed tax |

Current root-cause read:

- The large synchronized wait is not a separate model operation. It is the point
  where Python waits for GPU work launched by forward/backward/loss/optimizer.
- Data materialization is effectively noise at this shape.
- The RGRP scan is also small.
- The remaining cost is mostly dense transformer work from the Parcae schedule:
  3 prelude wide blocks, 2 recurrent narrow blocks run twice, and 3 coda wide
  blocks. In other words, the lane behaves like 10 transformer block executions
  per step rather than 8, even though the controller itself is cheap.

Additional hot-loop fix:

- Parcae diagnostics were already fixed to avoid `.item()` inside forward.
- The main training benchmark also materialized train loss every step, forcing a
  scalar readback and creating huge long-run reports.
- `BenchmarkBudgetSpec.train_loss_record_interval` now makes this explicit.
  Default is `1` for backward compatibility; scale runs can set a larger value
  and still record the first and final step.

Next runtime target:

- Before scaling again, run matched quality/perf with sparse train-loss
  recording enabled, for example `--train-loss-record-interval 128` or `256`.
- If the speed tax remains around 5-10 percent, the next genuine model-side
  lever is reducing extra virtual dense block work, not rewriting the RGRP scan.
- If d480 quality is still needed, the kernel work remains a true local-window
  attention path that supports `head_dim=48`.
- `parcae_loop_layer_count` now exposes the recurrent-loop band width directly.
  The first target ablation is d448/h7 flex-local with one recurrent loop block
  instead of the default two. A SageMaker timing job for this shape was submitted
  as `fractal-lctx-rgrp448h7-flex-looplayers1-timing-0424`, but AWS kept it in
  `Pending` waiting for capacity; it was stopped before training started.

Follow-up retry:

- EC2 was not usable for this smoke even though IAM could dry-run instance
  creation: the active EC2 `Running On-Demand G and VT instances` quota in
  `us-east-1` was still `0` vCPUs, and the real `g6.2xlarge` launch failed
  with `VcpuLimitExceeded`.
- The same timing surface was retried on SageMaker `ml.g6.2xlarge` and completed.
- Current-script timing showed that setting `parcae_loop_layer_count=1` inside
  the same 8-layer shell is not the desired speed lever. It changes the schedule
  from `3 + 2x2 + 3` to `3 + 1x2 + 4`: two narrow recurrent block executions are
  removed, but one wide coda block is added.
- The corrected reduced-virtual-depth smoke is therefore `total_layers=7` with
  `parcae_loop_layer_count=1`, giving `3 + 1x2 + 3`.

| Lane | Params | Schedule | Dense block execs | Wide execs | Narrow execs | Step ms | Tok/s |
|---|---:|---|---:|---:|---:|---:|---:|
| 8L default loop band | 45.35M | `3 + 2x2 + 3` | 10 | 6 | 4 | 206.952 | 79,168 |
| 8L loop band 1 | 46.97M | `3 + 1x2 + 4` | 9 | 7 | 2 | 212.369 | 77,149 |
| 7L loop band 1 | 44.56M | `3 + 1x2 + 3` | 8 | 6 | 2 | 200.590 | 81,679 |

Interpretation:

- The first loop-band knob worked mechanically, but it exposed a hidden
  allocation contract: fixed total depth silently reassigns removed loop depth to
  the coda.
- The speed tax is not the RGRP scan. It is the number and placement of dense
  transformer block executions around the loop.
- The next quality experiment should not promote the 8L loop-band-1 shape. If
  we test this lever, use a true reduced-virtual-depth shape or add explicit
  prelude/loop/coda ownership knobs so the runtime contract cannot be inferred
  from `total_layers` alone.

8192-step quality check:

- The true reduced-virtual-depth shape was run as
  `fractal-lctx-rgrp448h7-flex-7l-looplayers1-8192-s42-0424`.
- It used the 750M-token cache and saw `134,217,728` train tokens.

| Lane | Params | Final loss | Tok/s | Peak CUDA GB |
|---|---:|---:|---:|---:|
| attention-only d448/h7 flex | 48.44M | 4.2756 | 81,376 | 8.47 |
| RGRP d448/h7 flex, `3 + 2x2 + 3` | 45.61M | 4.2449 | 76,972 | 8.51 |
| RGRP d448/h7 flex, `3 + 1x2 + 3` | 44.82M | 4.2658 | 82,301 | 8.34 |

Read:

- The reduced-virtual-depth lane kept a small quality edge over attention while
  becoming faster and lighter.
- It did not keep the full 8L RGRP quality edge. This makes it a promising
  shape-allocation target, not a finished winner.

## Sparse-loss CUDA verification

Artifact:

- `experiments/aws_sagemaker/path1_cuda_scout/fractal-lctx-rgrp448h7-flex-sparse-1024-s42-0424/path1-cuda-scout/summary.md`

Run shape:

- `d_model=448`
- `head_count=7`
- `attention_kernel=flex-local`
- `local_window=128`
- `parcae_loop_d_model=256`
- `parcae_loop_head_count=4`
- `seq_len=512`
- `batch_size=32`
- `steps=1024`
- `train_loss_record_interval=128`

Result:

| Lane | Params | Final loss | tok/s | Peak CUDA MB | Train-loss records |
|---|---:|---:|---:|---:|---:|
| d448/h7 RGRP flex-local sparse-loss smoke | 45.35M | 5.1410 | 78,491.96 | 8,708.41 | 9 |

Interpretation:

- The CUDA training path now honors sparse train-loss recording. The report
  records steps `1, 128, 256, 384, 512, 640, 768, 896, 1024` instead of every
  step.
- This confirms that long-run reports no longer require a per-step scalar
  readback or a full per-step JSON ledger.
- The throughput is in the same band as the earlier d448/h7 flex-local timing
  and quality runs. This is not an apples-to-apples quality comparison because
  it only ran 1024 steps, but it validates the corrected runtime contract before
  the next scale rung.
