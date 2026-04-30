# Parcae/RGRP Native Kernel Superspec

Date: 2026-04-29

## Purpose

This superspec defines the native-kernel dependency ladder for the
Parcae/RGRP CUDA lane. It is intentionally broader than any one Triton kernel.
The goal is to stop treating PyTorch as a mysterious runtime tax and instead
name each PyTorch-owned contract that the architecture currently depends on,
then replace those contracts in an order that lets later kernels stack on top
of earlier ones.

This is not a new architecture proposal. It preserves the current champion
model recipe while preparing the runtime surface for native ownership.

Current champion family:

```text
Parcae hourglass RGRP control
bands 3,2,3,2,2
loop320x2
trainable-block-diagonal-8 controller state transform
triton-loop-forward
compiled-direct recurrent block contract
compiled band prepare
compiled FFN / compiled head loss
fused Adam
```

## Current PyTorch Ownership Map

The current model path is explicit enough to map ownership:

```text
input tokens
-> embedding / optional position features
-> wide prelude blocks
-> per-band prepare:
     norm(hidden)
     wide -> loop projection
     controller scan / control projection
     scalar decay / injection / nonlinear controls
-> Parcae loop region:
     state mix
     recurrent transformer block(s)
     residual/update mix
     loop repeat / truncated BPTT detach
-> loop -> wide projection and output mix
-> coda blocks
-> final norm + LM head + cross entropy
-> optimizer update
```

Today, PyTorch still owns most of the real runtime contract:

| Region | Current owner | Why this matters |
|---|---|---|
| Linear/GEMM projections | `torch.nn.Linear` / `F.linear` / cuBLAS through PyTorch | Tensor layout, dtype conversion, GEMM dispatch, weight-gradient accumulation. |
| Norms | PyTorch layer norm / RMS norm expressions | Reductions, eps behavior, saved activations, backward reductions. |
| Attention | PyTorch SDPA / FlexAttention or `flash-attn` wrapper | Mask semantics, causal/window semantics, backend dispatch, backward workspace. |
| FFN | PyTorch GEMMs plus GELU/backward, sometimes `torch.compile` | Large repeated inner-loop cost and activation-save policy. |
| Loop glue | Triton forward for parts of state/residual/update | Forward is partially native, backward mostly returns to PyTorch reductions/autograd. |
| Broadcast control gradients | PyTorch autograd, `sum_to_size`-style reductions | Small-looking control tensors expand over `[batch, seq, width]` and must reduce correctly. |
| Detach/truncated BPTT | PyTorch autograd tape construction and `.detach()` | Owns which loop passes keep backward history. |
| LM head/loss | PyTorch final norm, output projection, CE, sometimes compiled | Large vocab projection and CE memory/workspace. |
| Optimizer | PyTorch AdamW / local Triton Adam prototype | Parameter traversal, moment buffers, dtype and launch granularity. |

The promoted native surface is therefore still a hybrid:

```text
native-ish forward glue
+ PyTorch GEMM/norm/attention/FFN/autograd/loss/optimizer
```

The long-term target is:

```text
typed Parcae/RGRP loop-region contract
+ native projection/control layout
+ native recurrent block components
+ native full loop-region forward/backward/recompute
+ eventually native optimizer
```

## Primary External References

These are the reference surfaces to audit before replacing PyTorch contracts:

- PyTorch SDPA docs:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch SDPA tutorial:
  https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial
- PyTorch attention module / FlexAttention entrypoint:
  https://docs.pytorch.org/docs/stable/nn.attention.html
- PyTorch CUDA graphs docs:
  https://docs.pytorch.org/docs/stable/cuda
- PyTorch CUDA graphs blog:
  https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- NVIDIA CUTLASS overview:
  https://docs.nvidia.com/cutlass/latest/overview.html
- NVIDIA CUTLASS GEMM API:
  https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html
- NVIDIA CUDA Programming Guide:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- AdaExplore repository:
  https://github.com/StigLidu/AdaExplore
- The Recurrent Transformer: Greater Effective Depth and Efficient Decoding:
  https://arxiv.org/abs/2604.21215

Audit rule:

```text
Before replacing a PyTorch-owned boundary, identify what the PyTorch boundary
currently guarantees: layout, dtype, broadcasting, reduction shape, saved
activation policy, determinism expectations, and backward workspace.
```

## AdaExplore-Inspired Kernel Search Discipline

AdaExplore is not a kernel we can reuse directly for Parcae/RGRP. Its useful
lesson is procedural: kernel work should be treated as an explicit search over
correctness-preserving candidates, not as a sequence of hopeful one-off edits.

Relevant ideas to borrow:

- Separate correctness from speed. Every native candidate must first pass a
  shape/dtype/numerical parity gate, then a warmup-stabilized performance gate.
- Preserve diversity. Keep several candidate families alive when they represent
  genuinely different contracts, such as Triton glue, explicit reducer kernels,
  CUTLASS-style GEMM packing, or full loop-region ownership.
- Convert failures into rules. Repeated compile/runtime/numerical failures
  should update this spec as constraints before the next candidate is launched.
- Use local refinement and structural jumps differently. Tune tile sizes,
  block sizes, and save/recompute policy as small steps; change ownership
  boundaries or tensor layout only as deliberate large steps.
- Run candidate kernels in isolated subprocesses where possible. Illegal memory
  access, bad CUDA state, or extension-build failures should not poison the
  whole experiment runner.

Concrete adaptation for this repo:

```text
candidate kernel
-> isolated parity test
-> CUDA-event microbench
-> short training smoke
-> 2048-step paired control
-> 8192-step paired control only after the above passes
```

The reward is not raw speed alone. A candidate only promotes when it preserves
or improves loss under matched controls while moving a named timing bucket in
the intended direction.

## Recurrent Transformer Extraction

The Recurrent Transformer paper is useful because it studies a nearby failure
mode: adding temporal recurrence to a transformer-like block gives extra
effective depth, but naive training/prefill becomes launch-heavy and
memory-traffic-heavy unless the runtime is redesigned around the recurrence.

What to steal:

- Output-derived persistent memory. Recurrent Transformer uses temporary
  key/value pairs from the layer input for the current token, then writes
  persistent key/value pairs from the layer output for future tokens. For
  Parcae/RGRP, the closest analogue is: compute normal shell attention locally,
  then write recurrent state/control summaries from the post-block output, not
  only from the pre-block input.
- Exact tiled accumulation. Their forward algorithm relies on all queries being
  known early while persistent key/value pairs are revealed causally. Newly
  revealed key/value tiles update future query accumulators in power-of-two
  windows using online-softmax statistics. For us, the corresponding idea is a
  tiled loop-region kernel that updates future control/attention accumulators
  rather than replaying one-token/one-band work through many tiny launches.
- CUDA Graphs as a first-class recurrent runtime primitive. Their measurements
  show launch overhead dominating at smaller batches, with CUDA Graph replay
  turning the recurrent forward/backward into a materially different runtime
  class. Our loop-region timing contract must therefore distinguish kernel
  math from launch overhead before declaring a model shape too slow.
- Position-major state/cache layout. Their implementation stores recurrent
  key/value cache with the position dimension first to match the tiled access
  pattern. Our native loop/control state layout should be chosen for the access
  pattern of the loop kernel, not inherited from PyTorch convenience layouts.
- Checkpoint/recompute around revealed state. They preallocate persistent
  memory, write in place, and use custom backward because PyTorch autograd does
  not naturally own that contract. This reinforces that Parcae/RGRP full-loop
  ownership must include backward/recompute, not just forward glue.
- Stability defaults. RMS-style normalization before persistent memory writes
  and depth-wise residual scaling are not cosmetic. They are part of the
  stability contract for deeper-in-time computation.

What not to steal yet:

- Do not replace the current Parcae/RGRP champion with a full output-derived KV
  recurrent transformer before the matched attention control and current kernel
  contract tests land.
- Do not treat their speed tax as acceptable for us. Their reported recurrent
  transformer is still materially slower than vanilla attention at 150M-300M.
  The lesson is that recurrence can win quality, but the runtime must be owned
  more deeply before scaling claims are credible.

Concrete additions to the dependency ladder:

```text
loop-region timing
-> typed loop/control layout
-> tiled future-accumulator prototype
-> CUDA graph full-loop capture/replay
-> custom backward/recompute for persistent state
-> optional output-derived memory write ablation
```

## Dependency Ladder

## Implementation Progress

As of 2026-04-29:

- `python.runtime.parcae_loop_region` exposes stable loop-region timing names via
  `PARCAE_LOOP_REGION_TIMING_NAMES`.
- `ParcaeLoopRegionControls.validate_against(...)` now validates the explicit
  `[batch, sequence, width]` state/control layout, dtype, device, and broadcast
  ownership before loop kernels run.
- The fused first-state-mix path now expands broadcast injection controls to the
  real state shape before entering recurrent blocks.
- Triton loop-glue backward no longer calls `Tensor.sum_to_size` directly. It
  now routes through `_sum_to_broadcast_owner(...)`.
- `_sum_to_broadcast_owner(...)` owns a Triton two-pass width-owner reducer for
  the Parcae control-gradient shapes we actually use, such as
  `[batch, sequence, width] -> [1, 1, width]` and
  `[batch, sequence, width] -> [width]`.
- Unsupported broadcast reductions still use the compatibility path; they are
  not considered native-owned until a shape-specific reducer exists.

## PyTorch `sum_to_size` Source Audit

The relevant upstream implementation is not a unique CUDA kernel named
`sum_to_size`. It is a shape contract plus the standard reduction stack:

```text
Tensor.sum_to_size(...)
-> ExpandUtils::is_expandable_to(shape, tensor.sizes())
-> ExpandUtils::sum_to(...)
-> _sum_to(...)
-> tensor.sum(reduce_dims, keepdim=true)
-> ReduceOps.cpp sum_out(...)
-> TensorIterator reduction
-> CUDA sum_stub
-> ReduceSumProdKernel.cu sum_kernel_cuda(...)
-> Reduce.cuh gpu_reduce_kernel(...)
```

Important details from the audit:

- `_sum_to` chooses leading dimensions plus dimensions where the owner shape is
  `1` and the runtime tensor dimension is not `1`.
- The result is kept reduced with `keepdim=true`, then viewed back to the owner
  shape when leading dimensions were reduced.
- CUDA sum dispatch uses TensorIterator and `gpu_reduce_kernel`, including type
  handling for half/bfloat16 and fp32 accumulation paths.
- Replacing this with ordinary `gradient.sum(...).reshape(...)` is not an
  atomic-equivalent replacement. It routes through PyTorch differently and can
  produce worse launch/materialization behavior, which our 1024-step smoke did.

Primary source anchors:

- `aten/src/ATen/ExpandUtils.h`
- `aten/src/ATen/native/ReduceOps.cpp`
- `aten/src/ATen/native/cuda/ReduceSumProdKernel.cu`
- `aten/src/ATen/native/cuda/Reduce.cuh`

### 1. Loop-Region Timing Contract

Why first:

- Native work without stable timing will optimize the wrong boundary.
- CUDA event timings must distinguish launch-time, GPU-time, and sync points.
- The same timing names must survive later implementation swaps.

Current PyTorch-owned pieces to expose:

```text
band prepare
loop input projection
controller scan
controller/control projection
state mix
recurrent block attention
recurrent block FFN
residual/update mix
loop output projection
output mix
LM head/loss
optimizer step
```

Native implementation implication:

- Add a durable timing schema, not ad hoc `record_function` names.
- Use CUDA events for GPU work and separate wall-time buckets for host launch
  and synchronization.
- Treat timing names as part of the runtime contract. Kernel variants should
  plug into the same timing slots.

Acceptance gate:

```text
One champion-recipe timing run can answer:
- which loop-region subregion dominates GPU time;
- which subregion dominates wall time;
- whether the bottleneck is GEMM, attention, FFN, reduction, optimizer, or host launch.
```

Do not proceed to large native rewrites until this exists.

### 2. Projection/Control Packing

Why second:

- Projection/control tensors are inputs to all deeper native kernels.
- Changing them after writing block or loop kernels forces downstream rewrites.

Current PyTorch-owned pieces:

```text
parcae_prelude_norm(hidden)
hourglass_down_projection(hidden)
P20/RGRP controller scan(control_input)
control_norm(control)
control_projection(control).chunk(2)
sigmoid(control_gate_logits)
position embedding add
optional stride slicing / repeat_interleave
softplus/exp/sigmoid scalar controls
```

Important hidden contracts:

- `Linear` output layout is contiguous enough for downstream `.chunk`.
- `.chunk(2, dim=-1)` creates views with stride semantics that downstream ops
  can consume.
- `repeat_interleave` materializes expanded control values when stride is not 1.
- Scalar controls such as decay/nonlinear are broadcast over batch and sequence,
  and gradients must reduce back to width-sized parameters.

Native direction:

```text
PackedControlProjection:
  input: loop_input [B, T, loop_d]
  optional position/time features
  optional controller output
  output:
    injection_value [B, T, loop_d]
    injection_gate [B, T, loop_d]
    decay [1, 1, loop_d] or [B, T, loop_d]
    nonlinear [1, 1, loop_d] or [B, T, loop_d]
```

Implementation notes:

- Keep stride-1 champion path first. Strided control can stay deferred.
- Prefer one packed projection for control value/gate where possible.
- Make broadcast ownership explicit:

```text
control_layout = per_width | per_token | per_chunk
gradient_reduce = width_sum | token_sum | no_reduce
```

Acceptance gate:

```text
Packed controls produce numerically equivalent forward values on the champion path
and expose contiguous tensors for loop-region kernels.
```

### 3. Loop Input/Output Seam Fusion

Why third:

- This is the stable boundary between wide model space and recurrent loop space.
- Full loop fusion needs a single contract for entering and leaving loop space.

Current PyTorch-owned pieces:

```text
input seam:
  norm(hidden)
  wide -> loop Linear

output seam:
  loop -> wide Linear
  sigmoid(residual_logit)
  loop_anchor + gate * loop_delta
```

Important hidden contracts:

- Norm backward owns reductions over hidden width.
- Linear backward owns `dW = grad^T @ input`, `dInput = grad @ W`.
- Output mix gate is width-broadcast and needs reduction back to one parameter
  vector.

Native direction:

```text
LoopInputSeam:
  RMSNorm/LayerNorm + GEMM + optional epilogue

LoopOutputSeam:
  GEMM + gated residual epilogue
```

Implementation options:

- Triton first for output mix and lightweight epilogues.
- CUTLASS/CuTe for GEMM + epilogue fusion if the seam GEMMs are large enough.
- Avoid writing custom matmul unless the kernel can use Tensor Cores or delegate
  to a tuned GEMM path.

Acceptance gate:

```text
Seam fusion reduces materialization or launch count without changing loss on
512/2048-step smokes.
```

### 4. Recurrent FFN / Native Block Pieces

Why fourth:

- FFN is repeated inside the recurrent band and often dominates once attention
  is no longer the primary bottleneck.
- This depends on stable loop tensor layout from steps 2 and 3.

Current PyTorch-owned pieces:

```text
norm
fc1 GEMM
GELU
fc2 GEMM
residual add
backward:
  GELU derivative
  activation saves or recompute
  two weight-gradient GEMMs
  norm backward reductions
```

Existing local experiments:

- manual autograd and recompute variants own the formula but still use PyTorch
  GEMMs and reductions.
- Triton GELU alone is not sufficient as a runtime lane.

Native direction:

```text
RecurrentFFNBlock:
  forward:
    norm -> GEMM1 -> activation -> GEMM2 -> residual
  backward:
    choose activation-save or recompute policy
    own activation derivative
    use cuBLAS/CUTLASS for GEMMs
    own reductions for norm/bias
```

Research questions:

- Is the bottleneck GEMM time, activation materialization, norm reduction, or
  Python/autograd launch fragmentation?
- Does recompute reduce memory enough to matter for 100M+ scaling?
- Are loop widths such as 320 Tensor-Core friendly enough after packing?

Acceptance gate:

```text
Native/recompute FFN preserves quality and improves either tok/s or peak memory
on the champion recipe.
```

### 5. Complete Recurrent Block Native Path

Why fifth:

- Once norm, attention, FFN, and residual seams are explicit, a whole recurrent
  block can be owned as one native boundary.
- Doing this before steps 2-4 would freeze the wrong layouts.

Current PyTorch-owned pieces:

```text
position feature add
attention norm
qkv projection
head reshape / transpose / contiguous
local attention kernel
output projection
FFN norm
FFN
residual routing
```

Important hidden contracts:

- PyTorch reshapes/transposes create layout changes that are cheap as views until
  a downstream kernel demands contiguous memory.
- FlexAttention owns block-mask representation and enforces shape/head-dim
  constraints.
- Attention and FFN backward each have their own workspace and saved activation
  policy.

Native direction:

```text
NativeRecurrentBlock:
  packed qkv projection
  local attention
  output projection
  FFN
  residuals
  optional position feature handling
```

Implementation strategy:

- Start with a block-level wrapper that still calls vendor GEMMs/attention but
  owns tensor layout and saved activation policy.
- Replace the heaviest subpart only after timing proves it.
- Use CUTLASS for GEMM + epilogue opportunities; use custom Triton/CUDA only
  where the operation is not a standard GEMM.

Acceptance gate:

```text
One recurrent block native path matches PyTorch forward/backward within tolerance
and improves the loop-region timing bucket.
```

### 6. Full Loop-Region Forward/Backward/Recompute

Why sixth:

- This is the real prize, but it should sit on stable tensor and block
  contracts.
- Tiny native backward kernels failed because they replaced small fragments
  while leaving PyTorch to own surrounding reductions/autograd.

Current PyTorch-owned pieces:

```text
Python loop over loop_count
Python loop over recurrent blocks
torch.no_grad truncation for early loop passes
detach boundary
autograd tape for final loop passes
state/control broadcast reductions
activation saves inside recurrent blocks
```

Native direction:

```text
NativeParcaeLoopRegion:
  owns:
    loop iteration schedule
    BPTT/recompute policy
    recurrent block dispatch
    state/control reductions
    saved activation workspace
    optional checkpoint/recompute
```

Backward choices:

1. save activations for final `backward_steps`;
2. recompute recurrent blocks during backward;
3. hybrid: save cheap controls, recompute heavy block activations;
4. no-BPTT / ES-style side experiments stay separate from the champion path.

Implementation warning:

```text
Do not reintroduce standalone native backward for state_mix/residual_mix unless
the full loop-region kernel owns the reduction and accumulation policy.
```

Acceptance gate:

```text
Full loop-region native path beats the champion runtime at 2048 steps without
quality drift, then survives an 8192 promotion.
```

### 7. Custom Local Attention / Sparse Attention

Why seventh:

- Attention is already less dominant at the current `seq_len=512` champion
  shape.
- It becomes more important at longer context or if recurrent block fusion makes
  everything else cheaper.

Current PyTorch-owned pieces:

```text
SDPA backend selection
FlexAttention block mask
FlashAttention wrapper
q/k/v layout transforms
attention backward workspace
```

Native direction:

```text
WindowAttentionKernel:
  q/k/v layout chosen by recurrent block contract
  causal local window
  head_dim not restricted to PyTorch FlexAttention power-of-two constraints
  forward + backward
```

Implementation options:

- Start by making layout compatible with FlashAttention local-window when
  possible.
- Write owned Triton/CUDA only if:
  - head_dim/window constraints block desired model shapes, or
  - timing shows attention is the recurrent-loop bottleneck.

Acceptance gate:

```text
Native local attention either unlocks a better shape such as non-power-of-two
head dimensions or improves long-context speed/memory.
```

### 8. Native Optimizer / Muon

Why eighth:

- Mostly orthogonal to Parcae loop correctness.
- Worth doing after the model runtime path is stable, because optimizer wins
  stack with all architectures and should not mask loop-region regressions.

Current PyTorch-owned pieces:

```text
parameter iteration
state allocation
Adam moment updates
weight decay
Muon Newton-Schulz orthogonalization for 2D matrices
fallback optimizer for embeddings/head/non-2D tensors
```

Native direction:

```text
NativeOptimizer:
  grouped parameter buckets by shape/dtype/update law
  fused Adam for fallback tensors
  native Muon for selected 2D hidden matrices
  explicit exclusion rules for embeddings, heads, norms, scalar controls
```

Implementation notes:

- Current local Muon reference is PyTorch-based and was not yet a win.
- A native Muon path must own grouped 2D matrix updates and Newton-Schulz GEMMs,
  not just wrap per-parameter Python loops.
- Use this after the champion model runtime is stable.

Acceptance gate:

```text
Native optimizer improves wall-clock without degrading the already-promoted
model/loss surface.
```

## Research Backlog By PyTorch Contract

### Linear/GEMM

Current PyTorch behavior to audit:

- `F.linear` dispatch path for bf16 CUDA.
- Weight-gradient accumulation path.
- Bias gradient reductions.
- Layout requirements for Tensor Core use.

Native options:

- cuBLASLt for standard GEMM with epilogues.
- CUTLASS/CuTe for custom epilogue fusion and grouped/persistent GEMMs.
- Triton only for small custom epilogues or nonstandard tensor layouts.

### Norms

Current PyTorch behavior to audit:

- LayerNorm/RMSNorm reduction precision.
- Saved tensors for backward.
- Bias/weight gradient reductions.

Native options:

- Triton RMSNorm/LayerNorm with explicit fp32 reduction.
- Fuse norm with projection only after reduction semantics are locked.

### Broadcast Controls

Current PyTorch behavior to audit:

- Expanded tensor gradient semantics.
- `sum_to_size` reductions from `[B,T,D]` back to `[1,1,D]` or `[D]`.
- Whether reductions are bandwidth-bound or launch-bound.

Native options:

- Keep PyTorch reductions until full loop-region backward owns them.
- If owned, reduce per block over `[B,T]` into fp32 accumulators, then write one
  control-gradient vector.

### Attention

Current PyTorch behavior to audit:

- SDPA backend selection and constraints.
- FlexAttention block mask representation and compile path.
- FlashAttention local-window wrapper layout requirements.

Native options:

- Use FlashAttention where it already owns the right local-window semantics.
- Write custom local-window attention only for unsupported shapes or confirmed
  bottleneck cases.

### Autograd / Recomputation

Current PyTorch behavior to audit:

- Saved tensors for recurrent blocks.
- Detach/truncated BPTT boundary.
- Compile interaction with loop boundaries.

Native options:

- Explicit activation workspace.
- Recompute backward for recurrent FFN/block.
- Full loop-region custom autograd only after block contracts are stable.

## Immediate Work Items

1. Add a SageMaker wrapper regression test for the primitive-triton compile
   contract:
   - `FRACTAL_SCOUT_PRIMITIVE_RUNTIME_BACKEND=triton` with
     `FRACTAL_SCOUT_COMPILE_MODE` set must skip global `--compile-mode`;
   - recurrent compilation must remain controlled by
     `FRACTAL_SCOUT_PARCAE_RECURRENT_COMPILE_MODE`;
   - this prevents wrapper helpers from violating the spec-level
     `env_kind=primitive-triton` / `compile_mode=None` invariant.
2. Extend timing to emit a stable loop-region breakdown for the champion recipe.
3. Add a `ProjectionControlLayout` spec that records:

```text
loop_d_model
control_stride
control_layout
decay_layout
injection_layout
nonlinear_layout
contiguity expectations
gradient reduction owner
```

4. Add parity tests for packed projection/control tensors.
5. Add a microbenchmark for loop input/output seams:

```text
current PyTorch seam
compiled seam
Triton epilogue seam
CUTLASS/cuBLASLt candidate, if available
```

6. Only after the seam layout is fixed, start native recurrent FFN/block work.

## Non-Goals For The Next Slice

- Do not write a full custom local attention kernel before timing says attention
  is the bottleneck.
- Do not reattempt tiny standalone native backward kernels for scalar loop glue.
- Do not switch optimizer families while validating loop-region kernel changes.
- Do not change the champion training recipe during runtime-contract tests.

## Decision Rule

Every new kernel change must answer:

```text
What PyTorch-owned contract did this replace?
What exact tensor layout does the native path now own?
What backward/reduction semantics moved with it?
What timing bucket should improve?
What parity test protects the model quality surface?
```

If those cannot be answered, the change is not ready to become a runnable lane.
