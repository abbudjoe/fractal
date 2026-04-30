# Parcae/RGRP Native Loop-Region Contract

Date: 2026-04-28

## Status

The promoted CUDA fast lane remains:

```text
parcae_loop_update_backend=triton-loop-forward
parcae_band_block_contract=compiled-direct
parcae_band_prepare_backend=compiled
parcae_output_mix_backend=standard
```

This keeps the loop-update forward path native while leaving the backward reductions to the current PyTorch autograd path.

The first typed boundary now exists as `python.runtime.parcae_loop_region.run_parcae_loop_region`.
It preserves the current math while making loop count, truncation, state carry, recurrent block dispatch,
residual/update application, diagnostics, and timing ownership explicit. This is not yet a fully native
loop-region kernel; it is the contract that future native kernels should replace.

## Negative Ablations

Two attempted native-backward variants were removed from the runnable surface:

```text
triton-loop-forward-native-bwd
triton-loop-forward-frozen-control-bwd
```

Both were too slow before reaching useful step output. A prior Python/Dynamo loop reshaping attempt,
`parcae_band_loop_contract=two-block-direct`, was also removed because it timed out and did not test the
native runtime contract we actually care about.

## Lesson

The problem is not solved by writing custom kernels for tiny PyTorch-shaped fragments. The failed variants
cut the graph at the loop-update scalar glue:

```text
mixed = decay * state + injection
output = mixed + nonlinear * (block_out - mixed)
```

That boundary still leaves the architecture shaped like:

```text
state mix
-> recurrent block 0
-> residual/update mix
-> recurrent block 1
-> residual/update mix
-> loop repeat
```

The real native target is the whole loop region, not a single update op.

## Source-Audit Finding

The failed native-backward attempts replaced a deceptively large PyTorch contract. In our backward path,
the control gradients are broadcast reductions:

```text
grad_decay = (grad_mixed * state).sum_to_size(decay_shape)
grad_injection = grad_mixed.sum_to_size(injection_shape)
grad_nonlinear = (grad * (block_out - mixed)).sum_to_size(nonlinear_shape)
```

In PyTorch, `sum_to_size` first validates that the smaller control tensor shape can expand to the larger
runtime tensor shape, then lowers to `sum_to`. The CUDA sum path is not just a loop: it builds a
`TensorIterator` reduction, chooses vectorization along input/output dimensions, can split reductions across
CTAs, allocates staging buffers/semaphores for global reductions, and handles low-precision accumulation
rules. Autograd also treats expanded/broadcast views with layout-agnostic gradient semantics, where expanded
dimensions must be explicitly summed back to the owner tensor.

That means the local gradients above are not "scalar glue" from the runtime's point of view. They are a
broadcast/reduction ownership contract. Replacing them with tiny per-fragment kernels loses PyTorch's mature
reduction scheduling while still paying Python/autograd/materialization costs around the rest of the loop.

The durable fix is therefore:

```text
keep PyTorch reductions for the tiny loop-update backward
or
replace the complete loop-region backward/recompute boundary
```

Do not reintroduce a standalone native backward for `state_mix`, `residual_mix`, or `loop_update` unless it
is part of a larger loop-region kernel that owns the reduction accumulation policy.

## Required Native Boundary

The next implementation should introduce an explicit `ParcaeLoopRegion` runtime contract that owns:

```text
loop_input/state allocation
loop decay/injection/nonlinear controls
per-loop state carry
per-band recurrent block dispatch
loop-output projection seam
saved activation policy
backward/recompute policy
```

The contract should not rely on TorchDynamo to discover recurrent structure. It should receive typed tensors
and explicit metadata:

```text
batch_size
seq_len
wide_d_model
loop_d_model
loop_count
band_count
blocks_per_band
local_window
attention_kernel_contract
ffn_contract
state_layout
```

## Implementation Gate

Do not add another runnable backend unless it can satisfy at least one of these:

```text
1. Own the entire loop-region backward/recompute boundary.
2. Own one complete recurrent block including norm, attention, FFN, and residual seams.
3. Replace PyTorch attention/FFN calls inside the recurrent loop with native kernels.
```

Tiny custom backward kernels for `state_mix`, `residual_mix`, or `loop_update` are not enough. They can remain
microbenchmarks, but they should not be promoted as model runtime lanes.

## Near-Term Plan

1. Preserve the current `triton-loop-forward` fast lane.
2. Add a CUDA-event microbenchmark for the full Parcae loop region, separating:
   - band prepare
   - loop update glue
   - recurrent attention
   - recurrent FFN
   - output projection
3. Implement the next native kernel only after the benchmark identifies a complete architecture-shaped region.
4. Prefer contracts that can later port to Rust/CUDA ownership.
