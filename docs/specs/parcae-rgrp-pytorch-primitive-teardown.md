# Parcae/RGRP PyTorch Primitive Teardown

Date: 2026-04-29

## Purpose

This note is the source-level teardown companion to
[`parcae-rgrp-native-kernel-superspec.md`](./parcae-rgrp-native-kernel-superspec.md).
The superspec names the native-kernel dependency ladder; this document records
what the PyTorch primitives in that ladder actually do before we replace them.

The rule for this lane is:

```text
Do not replace a PyTorch boundary until its layout, dtype, broadcasting,
reduction, saved-activation, and backward contract are explicit.
```

Runtime context:

| Surface | Version / source |
|---|---|
| SageMaker CUDA runtime observed in logs | PyTorch 2.5.1+cu124 |
| Local inspection environment | PyTorch 2.11.0, CPU-only venv |
| Source tag inspected | `pytorch/pytorch@v2.5.1` |

Primary source URLs:

- PyTorch `Linear.cpp`: https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/Linear.cpp
- PyTorch LayerNorm CUDA kernel: https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/layer_norm_kernel.cu
- PyTorch GELU CUDA kernel: https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/ActivationGeluKernel.cu
- PyTorch cross entropy / NLL: https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/LossNLL.cpp
- PyTorch CUDA NLL kernels: https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/Loss.cu
- PyTorch SDPA frontend: https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/transformers/attention.cpp
- PyTorch CUDA SDPA backend selection: https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
- PyTorch FlexAttention API: https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/attention/flex_attention.py
- PyTorch AdamW: https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/adamw.py

## Repo Boundary Being Replaced

The current Parcae/RGRP champion path still relies on these PyTorch-owned
boundaries:

```text
embedding / position features
-> wide prelude attention blocks
-> loop input norm + down projection
-> RGRP controller scan and control projection
-> Parcae loop state mix
-> recurrent attention + FFN block
-> residual/update mix
-> loop output projection and output gate
-> coda attention blocks
-> final norm + LM head + cross entropy
-> AdamW / fused optimizer step
```

Some forward glue is already Triton-owned, but backward, reduction, GEMM, norm,
attention, and optimizer behavior are still mostly PyTorch-owned.

## Primitive 1: Linear / Projection

Used by:

- QKV projection and output projection in `LocalCausalSelfAttention`.
- FFN `fc1` and `fc2`.
- Parcae loop down/up projections.
- Controller/control projections around the RGRP seam.
- LM head projection when not using a fused head/loss path.

Source behavior:

- `torch.nn.functional.linear` dispatches to `aten::linear`.
- For 2D input with bias, PyTorch uses fused `addmm(bias, input, weight.t())`.
- For contiguous 3D input with bias, PyTorch flattens all but the last dimension,
  runs `addmm`, then views back to the original leading dimensions.
- For other layouts, it falls back to `matmul(input, weight.t())`, then adds
  bias in place when safe.
- There is an environment-controlled fallback, `TORCH_LINEAR_FLATTEN_3D`, that
  can force 3D flattening after a contiguous copy.

Hidden contract:

```text
input [B, T, Din] contiguous + bias
-> reshape [B*T, Din]
-> addmm with weight.t()
-> view [B, T, Dout]
```

Native replacement requirements:

- Own the flatten policy explicitly. Do not rely on accidental `view`/`reshape`
  compatibility.
- Make the weight layout explicit: PyTorch stores `weight` as `[Dout, Din]` and
  multiplies by `weight.t()`.
- Include the bias epilogue in the GEMM path where possible.
- Preserve output contiguity assumptions for downstream `.chunk`, `.view`, and
  head reshaping.
- Backward must produce:
  - `grad_input = grad_output @ weight`
  - `grad_weight = grad_output_flat.T @ input_flat`
  - `grad_bias = grad_output_flat.sum(dim=0)`

First native target:

```text
Packed linear family for fixed [B, T, D] tensors:
  qkv: one GEMM with 3D output, no post-hoc tiny projections
  controller/control: packed value/gate/decay candidates
  ffn: stable cublasLt/CUTLASS GEMM + activation epilogue
```

Parity tests:

- Contiguous `[B,T,D]` with bias.
- Non-contiguous transposed or sliced input should either fail loudly or copy
  intentionally.
- Weight/bias gradients compared against PyTorch at bf16 forward / fp32
  accumulation tolerance.

## Primitive 2: LayerNorm / RMSNorm

Used by:

- Standard transformer blocks through `nn.LayerNorm`.
- Parcae loop input/output normalization.
- Manual-autograd and recompute FFN paths.
- PR5-style paths use repo-local `SimpleRmsNorm`.

PyTorch LayerNorm source behavior:

- CUDA LayerNorm computes row-wise mean and reciprocal standard deviation with
  Welford-style reductions.
- Forward applies:

```text
y = (x - mean) * rstd * gamma + beta
```

- CUDA has scalar and vectorized paths. The vectorized path relies on alignment
  and width divisibility.
- Backward needs reductions over the normalized width plus separate reductions
  for gamma and beta gradients.

Repo RMSNorm behavior:

```python
denom = sqrt(mean(x * x, dim=-1, keepdim=True) + eps)
out = x / denom * weight
```

Hidden contract:

- Standard `nn.LayerNorm` subtracts mean; `SimpleRmsNorm` does not.
- LayerNorm stores or recomputes row statistics depending on the path.
- Norm gradients are not local elementwise ops; they are row reductions plus
  parameter reductions.

Native replacement requirements:

- Keep LayerNorm and RMSNorm as separate native contracts. Do not collapse them.
- Decide whether the native backward saves `mean/rstd` or recomputes.
- Use fp32 accumulation for bf16/half inputs unless deliberately changing the
  numerical contract.
- Own gamma/beta or RMS weight reductions directly; otherwise PyTorch
  `sum_to_size` will sneak back into the loop.

First native target:

```text
RMSNorm first for RGRP/PR5-like paths, then LayerNorm for GPT-shaped controls.
```

Parity tests:

- Forward and backward against PyTorch for `[B,T,D]` at the exact loop widths.
- Zero variance / near-constant rows.
- bf16 input, fp32 master weights, and native output dtype round-trip.

## Primitive 3: GELU / Activation

Used by:

- Standard `PositionWiseFeedForward`: `fc2(gelu(fc1(x)))`.
- Manual-autograd and recompute FFN paths.
- Existing Triton GELU kernels.

PyTorch source behavior:

- Default `F.gelu(x)` uses exact erf mode.
- Optional tanh approximation is a different formula.
- CUDA kernels use `opmath_type`, so bf16/half math promotes internally for the
  formula.
- Backward exact mode uses:

```text
cdf = 0.5 * (1 + erf(x / sqrt(2)))
pdf = exp(-0.5 * x^2) / sqrt(2*pi)
grad = dy * (cdf + x * pdf)
```

Hidden contract:

- Our repo paths call `F.gelu(preactivation)` without `approximate="tanh"`.
  The native path must therefore match exact erf GELU unless a config knob
  explicitly changes the model.

Native replacement requirements:

- Use exact erf by default.
- If a faster tanh approximation is tested, treat it as a model-quality
  ablation, not a pure kernel optimization.
- Avoid saving the full preactivation unless the chosen backward contract needs
  it; recompute is a valid FFN memory tradeoff.

First native target:

```text
Fused FFN boundary:
  norm -> fc1 -> exact GELU -> fc2 -> residual
with either saved activation or recompute backward.
```

Parity tests:

- Exact GELU forward/backward versus PyTorch.
- Ensure tanh/erf modes cannot silently switch.

## Primitive 4: Cross Entropy / LM Head

Used by:

- Final training loss after LM head projection.
- Compiled/fused head-loss experiments.

PyTorch source behavior:

- Class-index `cross_entropy` calls:

```text
log_softmax(logits, class_dim)
-> nll_loss_nd(log_probs, target, weight, reduction, ignore_index)
```

- With label smoothing, PyTorch computes a smoothed loss path and combines it
  with NLL.
- CUDA NLL kernels skip `ignore_index`, check target bounds, accumulate weighted
  negative log probability, and divide by total weight for mean reduction.

Hidden contract:

- The default class dimension is `1` unless logits are 1D. LM code commonly
  flattens `[B,T,V]` to `[B*T,V]`, so class dim is `1`.
- Mean reduction divides by non-ignored tokens or total target weight, not by
  raw `B*T` when ignored labels exist.
- A fused head/loss path must match logsumexp numerics without materializing the
  full `[B,T,V]` logits if the goal is memory reduction.

Native replacement requirements:

- Stream or tile the vocab projection and logsumexp.
- Preserve `ignore_index` semantics and denominator.
- Backward must emit the equivalent of:

```text
grad_logits = (softmax(logits) - one_hot(target)) / denominator
```

- If tying embeddings and LM head, gradient accumulation must respect shared
  storage.

First native target:

```text
Fused LM head + CE for fixed vocab and fixed [B,T,D]:
  tiled logits
  row-wise max / sumexp
  target logit gather
  optional gradient recompute
```

Parity tests:

- No ignored labels.
- Some ignored labels.
- bf16 hidden, fp32 accumulation.
- Exact loss and gradient versus PyTorch within tolerance.

## Primitive 5: SDPA / FlexAttention / Local Attention

Used by:

- Pure attention baseline.
- Prelude/coda blocks in Parcae/RGRP.
- Recurrent attention blocks inside the loop band.

Repo behavior:

- `LocalCausalSelfAttention` projects packed QKV, reshapes to
  `[B, heads, T, head_dim]`, and dispatches to one of:
  - `flash_attn_func` with `window_size=(local_window - 1, 0)`;
  - PyTorch FlexAttention with a cached local-causal `BlockMask`;
  - `F.scaled_dot_product_attention`.
- Flex local currently has a repo guard requiring power-of-two head dimension.

PyTorch SDPA source behavior:

- CUDA SDPA chooses a backend in priority order:

```text
flash_attention -> efficient_attention -> math -> cudnn_attention
```

- Flash attention requires Q/K/V to have the same last dimension and head dim
  less than or equal to 256.
- Memory-efficient attention has alignment requirements tied to dtype/device.
- On sm86-sm89, flash backward has constraints for training with head dim above
  192.
- Causal non-square sequence lengths can reject flash.
- If fused backends fail and math is enabled, PyTorch falls back to math.

PyTorch FlexAttention source behavior:

- FlexAttention is a prototype API.
- It consumes Q/K/V shaped `[B, H, L, E]` / `[B, H, S, E]`.
- `BlockMask` stores block-sparse metadata, not a dense mask.
- Default sparse block size is 128.
- `create_block_mask` rounds Q/KV lengths up to block multiples.
- It requires Dynamo support and internally uses a higher-order op.

Hidden contract:

- Attention speed can change because PyTorch silently changes backend.
- Local-window semantics are different across the three paths:
  - `flash_attn_func` receives a native window tuple;
  - FlexAttention receives block metadata;
  - SDPA receives an additive bias mask or causal flag.
- A custom local attention kernel must define one canonical local-causal
  contract rather than inheriting three subtly different call conventions.

Native replacement requirements:

- Fixed local-causal semantics:

```text
key_index <= query_index
key_index >= query_index - (local_window - 1)
```

- Explicit QKV layout:

```text
qkv packed projection -> [B,T,3,H,Dh] or [B,H,T,Dh]
```

- Decide whether the native kernel owns QKV projection fusion or starts at
  already-projected Q/K/V.
- Preserve softmax stability with max subtraction.
- Backward must own gradients for Q/K/V and optionally projection weights.

First native target:

```text
Do not start with attention unless profiling shows attention dominates again.
At seq512, old flash-local/FlexAttention made attention call time small.
At longer context, implement local-window attention with fixed block contract.
```

Parity tests:

- Compare local window 128 at seq512 and longer contexts.
- Confirm the same visible key set for every query position.
- Compare gradients against SDPA/Flex reference using a dense additive mask.

## Primitive 6: Broadcast, Expand, `sum_to_size`

Used by:

- Width-sized scalar controls broadcast over `[B,T,D]`.
- Residual gates and loop decay/nonlinear parameters.
- Position/control features with optional batch broadcast.

PyTorch source behavior:

- `expand` creates an `as_strided` view with expanded sizes/strides; it does not
  necessarily materialize.
- Backward through expanded tensors reduces with `sum_to_size`.
- `sum_to_size` checks expandability and reduces back to the original shape.

Hidden contract:

- Scalar-looking controls are not free in backward. A `[D]` control expanded
  over `[B,T,D]` needs a reduction over `B*T`.
- Previous tiny custom backward attempts can pass local smoke tests and still
  lose to PyTorch if they punt reductions back to autograd.

Native replacement requirements:

- Every native loop-region kernel must declare each control tensor as:

```text
per_width: [D], gradient reduce over B,T
per_token: [B,T,D], no broadcast reduce
per_chunk: [B,C,D], reduce over tokens within chunk
```

- Gradients for per-width controls should be accumulated in-kernel or in a
  dedicated reduction kernel, not through PyTorch autograd.

First native target:

```text
Loop-region backward with explicit reductions for decay, nonlinear, residual
gate, and controller gate/value projections.
```

Parity tests:

- Broadcasted `[D]` controls versus materialized `[B,T,D]` controls.
- Gradient equality for decay/nonlinear/gate parameters.

## Primitive 7: AdamW / Optimizer

Used by:

- Current baseline optimizer path.
- Local fused Adam prototype.
- Future Muon replacement.

PyTorch source behavior:

- AdamW supports single-tensor, foreach/multi-tensor, and fused paths.
- Default selection prefers foreach when neither `foreach` nor `fused` is set;
  fused is not enabled by default.
- Single-tensor AdamW does:

```text
step += 1
param *= (1 - lr * weight_decay)
exp_avg = lerp(exp_avg, grad, 1 - beta1)
exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
param += -lr / bias_correction1 * exp_avg / (sqrt(exp_avg_sq / bias_correction2) + eps)
```

- Foreach groups tensors by device/dtype and applies multi-tensor ops.
- Fused AdamW calls `torch._fused_adamw_` after grouping tensors by device/dtype.

Hidden contract:

- AdamW weight decay is decoupled: parameter multiply happens before moment
  update.
- Step counters may live on CPU unless capturable/fused is active.
- Tensor grouping affects launch count and behavior by dtype/device.
- Fused PyTorch AdamW is a useful reference but not the same as a native Muon
  optimizer.

Native replacement requirements:

- Decide parameter grouping up front:
  - large 2D matrices;
  - embeddings/head;
  - norm/scalar controls;
  - recurrent state/control matrices.
- Preserve decoupled weight decay if comparing to AdamW.
- For Muon, treat orthogonalization as a new optimizer contract, not an AdamW
  kernel tweak.

First native target:

```text
Keep fused Adam for now as a stable baseline.
Implement native Muon only after the recurrent loop kernel contract is stable,
unless optimizer timing again dominates.
```

Parity tests:

- Single parameter tensor AdamW against PyTorch.
- Multiple dtype/device groups if supported.
- Norm/scalar params excluded or assigned to a non-Muon fallback.

## Primitive 8: Parcae/RGRP Loop Region

Used by:

- The main architecture under investigation.

Current repo behavior:

- `python/runtime/parcae_loop_region.py` defines a clearer loop-region boundary,
  but it still delegates most math to PyTorch modules and autograd.
- `python/runtime/triton_primitives.py` owns some forward elementwise glue:
  - state mix;
  - residual mix;
  - state+residual loop update;
  - output mix;
  - exact-erf GELU forward/backward;
  - RGRP/P20 primitive kernels.
- Existing native pieces are not yet a full block-level forward/backward
  contract.

Hidden contract:

- The recurrent loop is not one operation yet. It is a Python/PyTorch sequence
  of:

```text
prepare controls
state mix
attention block
FFN block
residual/update mix
optional detach
repeat
```

- Any one tiny fusion can be erased by materialization, autograd graph nodes, or
  reductions at the boundary.

Native replacement requirements:

- Stabilize the loop tensor layout before deeper fusion:

```text
wide hidden: [B,T,Dwide]
loop state: [B,T,Dloop]
loop controls: typed per_width/per_token tensors
block output: [B,T,Dloop]
```

- Own block-local saved tensors and recompute policy.
- Own backward for state update and gate/control reductions.
- Make loop count and truncated-BPTT detach points explicit.

First native target:

```text
Full loop-region forward/backward for one recurrent block shape:
  loop320x2 BD8 first
  then 3,2,3,2,2 double-hourglass if it remains promising
```

Parity tests:

- Single loop pass.
- Two loop passes.
- Detach between passes versus full BPTT.
- Same final hidden and gradients as PyTorch reference.

## Replacement Order

The dependency order remains:

1. Timing contract.
2. Projection/control packing.
3. Loop input/output seam fusion.
4. Recurrent FFN/native block pieces.
5. Complete recurrent block native path.
6. Full loop-region forward/backward/recompute.
7. Custom local attention / sparse attention.
8. Native optimizer / Muon.

The important refinement from this teardown is that each rung now has a source
contract. Native work that does not preserve the contract must be labeled as a
model ablation, not a pure runtime optimization.

## Immediate Next Native Work

Recommended next implementation work while the 8192-step run is pending:

1. Add a targeted SageMaker wrapper regression test:
   - `primitive_runtime_backend=triton` plus global `compile_mode` must not emit
     global `--compile-mode`;
   - the run must route recurrent compilation through
     `--parcae-recurrent-compile-mode` / `FRACTAL_SCOUT_PARCAE_RECURRENT_COMPILE_MODE`;
   - this protects the `env_kind=primitive-triton` runtime contract enforced by
     `DeviceRuntimeSpec`.
2. Add a loop-region timing schema that isolates:
   - projection/control packing;
   - controller scan;
   - state/update mix;
   - recurrent attention;
   - recurrent FFN;
   - output seam;
   - head/loss;
   - optimizer.
3. Turn controller/control tensors into a typed layout object, even if still
   backed by PyTorch tensors.
4. Implement the first native backward boundary around state/update mix with
   explicit per-width reductions.
5. Only then fuse the larger recurrent FFN/block boundary.
