from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F

from python.runtime.cuda_timing import timed_region

try:  # pragma: no cover - import availability depends on runtime environment
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import availability depends on runtime environment
    triton = None
    tl = None


def triton_runtime_available() -> bool:
    return triton is not None and tl is not None


def ensure_triton_runtime_available() -> None:
    if not triton_runtime_available():
        raise RuntimeError(
            "primitive_runtime_backend=triton requires the primitive-triton CUDA env with a working Triton install"
        )


def _next_power_of_two(value: int) -> int:
    power = 1
    while power < value:
        power <<= 1
    return power


def _p20_atomic_transform_grad_enabled(block_pair_width: int) -> bool:
    raw = os.environ.get("FRACTAL_P20_TRITON_ATOMIC_TRANSFORM_GRAD", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    # The non-atomic path materializes one transform-gradient workspace per
    # batch item, then reduces it. Once rotary pair tiles widen past 32, that
    # workspace path falls off a severe backward-kernel cliff on L4/H100-class
    # CUDA runs. Default the wider-tile contract to in-kernel accumulation while
    # leaving the 256-wide fast lane unchanged.
    return block_pair_width > 32


def _sum_to_broadcast_owner(gradient: torch.Tensor, target_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    """Reduce an expanded gradient back to its broadcast owner shape.

    This is the explicit contract hidden behind Tensor.sum_to_size in the loop
    glue backward path. Keeping it in one helper lets native reductions replace
    this boundary later without rediscovering each broadcast rule.
    """

    shape = torch.Size(target_shape)
    if gradient.shape == shape:
        return gradient
    native_reduce_disabled = os.environ.get("FRACTAL_NATIVE_BROADCAST_OWNER_REDUCE", "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }
    if not native_reduce_disabled and _triton_width_owner_reduce_supported(gradient, shape):
        return _triton_sum_to_width_owner(gradient, shape)
    if len(shape) > gradient.ndim:
        raise RuntimeError(
            "cannot reduce gradient to a higher-rank broadcast owner: "
            f"gradient_shape={tuple(gradient.shape)} target_shape={tuple(shape)}"
        )
    leading_dims = gradient.ndim - len(shape)
    reduction_dims: list[int] = list(range(leading_dims))
    for shape_index, target_size in enumerate(shape):
        grad_dim = leading_dims + shape_index
        grad_size = gradient.shape[grad_dim]
        if target_size == 1 and grad_size != 1:
            reduction_dims.append(grad_dim)
        elif target_size != grad_size:
            raise RuntimeError(
                "gradient shape is not broadcast-compatible with target shape: "
                f"gradient_shape={tuple(gradient.shape)} target_shape={tuple(shape)}"
            )
    if reduction_dims:
        gradient = gradient.sum(dim=tuple(reduction_dims), keepdim=True)
    return gradient.reshape(shape)


def _triton_width_owner_reduce_supported(
    gradient: torch.Tensor,
    shape: torch.Size,
) -> bool:
    if not triton_runtime_available() or gradient.device.type != "cuda" or gradient.ndim < 2:
        return False
    if len(shape) == 1:
        return shape[0] == gradient.shape[-1]
    if len(shape) != gradient.ndim:
        return False
    if shape[-1] != gradient.shape[-1]:
        return False
    return all(size == 1 for size in shape[:-1])


def _fallback_sum_to_size(gradient: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    try:
        return gradient.sum_to_size(shape)
    except RuntimeError as exc:
        raise RuntimeError(
            "gradient shape is not broadcast-compatible with target shape: "
            f"gradient_shape={tuple(gradient.shape)} target_shape={tuple(shape)}"
        ) from exc


def _triton_sum_to_width_owner(gradient: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    ensure_triton_runtime_available()
    if gradient.numel() == 0:
        return torch.zeros(shape, device=gradient.device, dtype=gradient.dtype)
    contiguous = gradient.contiguous()
    width = int(contiguous.shape[-1])
    rows = int(contiguous.numel() // width)
    block_rows = 256
    block_d = min(64, _next_power_of_two(width))
    row_blocks = triton.cdiv(rows, block_rows)
    partial = torch.empty((row_blocks, width), device=contiguous.device, dtype=torch.float32)
    output = torch.empty((width,), device=contiguous.device, dtype=contiguous.dtype)
    _sum_to_width_owner_partial_kernel[(row_blocks, triton.cdiv(width, block_d))](
        contiguous,
        partial,
        rows,
        width,
        BLOCK_ROWS=block_rows,
        BLOCK_D=block_d,
    )
    chunk_block = _next_power_of_two(row_blocks)
    _sum_to_width_owner_finish_kernel[(triton.cdiv(width, block_d),)](
        partial,
        output,
        row_blocks,
        width,
        BLOCK_CHUNKS=chunk_block,
        BLOCK_D=block_d,
    )
    return output.reshape(shape)


if triton_runtime_available():  # pragma: no branch

    @triton.jit
    def _sum_to_width_owner_partial_kernel(
        gradient_ptr,
        partial_ptr,
        rows,
        width,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        width_block = tl.program_id(1)
        row_offsets = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        width_offsets = width_block * BLOCK_D + tl.arange(0, BLOCK_D)
        offsets = row_offsets[:, None] * width + width_offsets[None, :]
        mask = (row_offsets[:, None] < rows) & (width_offsets[None, :] < width)
        values = tl.load(gradient_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        reduced = tl.sum(values, axis=0)
        tl.store(
            partial_ptr + row_block * width + width_offsets,
            reduced,
            mask=width_offsets < width,
        )


    @triton.jit
    def _sum_to_width_owner_finish_kernel(
        partial_ptr,
        output_ptr,
        row_blocks,
        width,
        BLOCK_CHUNKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        width_block = tl.program_id(0)
        chunk_offsets = tl.arange(0, BLOCK_CHUNKS)
        width_offsets = width_block * BLOCK_D + tl.arange(0, BLOCK_D)
        offsets = chunk_offsets[:, None] * width + width_offsets[None, :]
        mask = (chunk_offsets[:, None] < row_blocks) & (width_offsets[None, :] < width)
        values = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        reduced = tl.sum(values, axis=0)
        tl.store(output_ptr + width_offsets, reduced, mask=width_offsets < width)

    @triton.jit
    def _parcae_state_mix_forward_kernel(
        state_ptr,
        decay_ptr,
        injection_ptr,
        output_ptr,
        n_elements,
        width,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        width_offsets = offsets % width
        state = tl.load(state_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        decay = tl.load(decay_ptr + width_offsets, mask=mask, other=0.0).to(tl.float32)
        injection = tl.load(injection_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        output = decay * state + injection
        tl.store(output_ptr + offsets, output, mask=mask)


    @triton.jit
    def _parcae_residual_mix_forward_kernel(
        mixed_ptr,
        block_out_ptr,
        nonlinear_ptr,
        output_ptr,
        n_elements,
        width,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        width_offsets = offsets % width
        mixed = tl.load(mixed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        block_out = tl.load(block_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        nonlinear = tl.load(nonlinear_ptr + width_offsets, mask=mask, other=0.0).to(tl.float32)
        output = mixed + nonlinear * (block_out - mixed)
        tl.store(output_ptr + offsets, output, mask=mask)


    @triton.jit
    def _parcae_loop_update_forward_kernel(
        state_ptr,
        decay_ptr,
        injection_ptr,
        block_out_ptr,
        nonlinear_ptr,
        output_ptr,
        n_elements,
        width,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        width_offsets = offsets % width
        state = tl.load(state_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        decay = tl.load(decay_ptr + width_offsets, mask=mask, other=0.0).to(tl.float32)
        injection = tl.load(injection_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        block_out = tl.load(block_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        nonlinear = tl.load(nonlinear_ptr + width_offsets, mask=mask, other=0.0).to(tl.float32)
        mixed = decay * state + injection
        output = mixed + nonlinear * (block_out - mixed)
        tl.store(output_ptr + offsets, output, mask=mask)


    @triton.jit
    def _parcae_output_mix_forward_kernel(
        anchor_ptr,
        delta_ptr,
        gate_ptr,
        output_ptr,
        n_elements,
        width,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        width_offsets = offsets % width
        anchor = tl.load(anchor_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        delta = tl.load(delta_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + width_offsets, mask=mask, other=0.0).to(tl.float32)
        output = anchor + gate * delta
        tl.store(output_ptr + offsets, output, mask=mask)


    @triton.jit
    def _gelu_forward_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
        tl.store(output_ptr + offsets, y, mask=mask)


    @triton.jit
    def _gelu_backward_kernel(
        grad_output_ptr,
        input_ptr,
        grad_input_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        cdf = 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
        pdf = 0.3989422804014327 * tl.exp(-0.5 * x * x)
        grad_input = grad_output * (cdf + x * pdf)
        tl.store(grad_input_ptr + offsets, grad_input, mask=mask)


    @triton.jit
    def _p20_forward_kernel(
        update_gate_ptr,
        retain_gate_ptr,
        transformed_state_ptr,
        candidate_ptr,
        output_gate_ptr,
        next_state_ptr,
        emitted_output_ptr,
        input_row_stride,
        output_row_stride,
        width,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block_index = tl.program_id(1)
        offsets = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < width
        input_row_offsets = batch_index * input_row_stride + offsets
        output_row_offsets = batch_index * output_row_stride + offsets

        update_gate = tl.load(update_gate_ptr + input_row_offsets, mask=mask, other=0.0)
        retain_gate = tl.load(retain_gate_ptr + input_row_offsets, mask=mask, other=0.0)
        transformed_state = tl.load(transformed_state_ptr + input_row_offsets, mask=mask, other=0.0)
        candidate = tl.load(candidate_ptr + input_row_offsets, mask=mask, other=0.0)
        output_gate = tl.load(output_gate_ptr + input_row_offsets, mask=mask, other=0.0)

        next_state = update_gate * transformed_state + retain_gate * candidate
        emitted_output = output_gate * next_state

        tl.store(next_state_ptr + output_row_offsets, next_state, mask=mask)
        tl.store(emitted_output_ptr + output_row_offsets, emitted_output, mask=mask)


    @triton.jit
    def _p20_backward_kernel(
        grad_next_state_ptr,
        grad_emitted_output_ptr,
        update_gate_ptr,
        retain_gate_ptr,
        transformed_state_ptr,
        candidate_ptr,
        output_gate_ptr,
        next_state_ptr,
        grad_update_gate_ptr,
        grad_retain_gate_ptr,
        grad_transformed_state_ptr,
        grad_candidate_ptr,
        grad_output_gate_ptr,
        grad_row_stride,
        input_row_stride,
        next_state_row_stride,
        output_row_stride,
        width,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block_index = tl.program_id(1)
        offsets = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < width
        grad_row_offsets = batch_index * grad_row_stride + offsets
        input_row_offsets = batch_index * input_row_stride + offsets
        next_state_row_offsets = batch_index * next_state_row_stride + offsets
        output_row_offsets = batch_index * output_row_stride + offsets

        grad_next_state = tl.load(grad_next_state_ptr + grad_row_offsets, mask=mask, other=0.0)
        grad_emitted_output = tl.load(grad_emitted_output_ptr + grad_row_offsets, mask=mask, other=0.0)
        update_gate = tl.load(update_gate_ptr + input_row_offsets, mask=mask, other=0.0)
        retain_gate = tl.load(retain_gate_ptr + input_row_offsets, mask=mask, other=0.0)
        transformed_state = tl.load(transformed_state_ptr + input_row_offsets, mask=mask, other=0.0)
        candidate = tl.load(candidate_ptr + input_row_offsets, mask=mask, other=0.0)
        output_gate = tl.load(output_gate_ptr + input_row_offsets, mask=mask, other=0.0)
        next_state = tl.load(next_state_ptr + next_state_row_offsets, mask=mask, other=0.0)

        total_grad_next_state = grad_next_state + grad_emitted_output * output_gate

        grad_update_gate = total_grad_next_state * transformed_state
        grad_retain_gate = total_grad_next_state * candidate
        grad_transformed_state = total_grad_next_state * update_gate
        grad_candidate = total_grad_next_state * retain_gate
        grad_output_gate = grad_emitted_output * next_state

        tl.store(grad_update_gate_ptr + output_row_offsets, grad_update_gate, mask=mask)
        tl.store(grad_retain_gate_ptr + output_row_offsets, grad_retain_gate, mask=mask)
        tl.store(grad_transformed_state_ptr + output_row_offsets, grad_transformed_state, mask=mask)
        tl.store(grad_candidate_ptr + output_row_offsets, grad_candidate, mask=mask)
        tl.store(grad_output_gate_ptr + output_row_offsets, grad_output_gate, mask=mask)


class _P20FusedUpdateReadout(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        update_gate: torch.Tensor,
        retain_gate: torch.Tensor,
        transformed_state: torch.Tensor,
        candidate: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ensure_triton_runtime_available()
        if update_gate.device.type != "cuda":
            raise RuntimeError("P20 Triton fused update requires CUDA tensors")
        for tensor in (
            retain_gate,
            transformed_state,
            candidate,
            output_gate,
        ):
            if tensor.device != update_gate.device:
                raise RuntimeError("P20 Triton fused update requires all tensors on the same CUDA device")
            if tensor.shape != update_gate.shape:
                raise RuntimeError("P20 Triton fused update requires matching tensor shapes")

        batch_size, width = update_gate.shape
        input_row_stride = update_gate.stride(0)
        block_size = min(1024, _next_power_of_two(width))
        next_state = torch.empty_like(update_gate)
        emitted_output = torch.empty_like(update_gate)
        output_row_stride = next_state.stride(0)

        grid = (batch_size, triton.cdiv(width, block_size))
        _p20_forward_kernel[grid](
            update_gate,
            retain_gate,
            transformed_state,
            candidate,
            output_gate,
            next_state,
            emitted_output,
            input_row_stride,
            output_row_stride,
            width,
            BLOCK_SIZE=block_size,
        )
        ctx.block_size = block_size
        ctx.input_row_stride = input_row_stride
        ctx.output_row_stride = output_row_stride
        ctx.width = width
        ctx.save_for_backward(
            update_gate,
            retain_gate,
            transformed_state,
            candidate,
            output_gate,
            next_state,
        )
        return next_state, emitted_output

    @staticmethod
    def backward(ctx, grad_next_state: torch.Tensor, grad_emitted_output: torch.Tensor):  # type: ignore[override]
        (
            update_gate,
            retain_gate,
            transformed_state,
            candidate,
            output_gate,
            next_state,
        ) = ctx.saved_tensors
        grad_update_gate = torch.empty_like(update_gate)
        grad_retain_gate = torch.empty_like(retain_gate)
        grad_transformed_state = torch.empty_like(transformed_state)
        grad_candidate = torch.empty_like(candidate)
        grad_output_gate = torch.empty_like(output_gate)
        grad_next_state_contiguous = grad_next_state.contiguous()
        grad_emitted_output_contiguous = grad_emitted_output.contiguous()

        batch_size = update_gate.shape[0]
        grid = (batch_size, triton.cdiv(ctx.width, ctx.block_size))
        _p20_backward_kernel[grid](
            grad_next_state_contiguous,
            grad_emitted_output_contiguous,
            update_gate,
            retain_gate,
            transformed_state,
            candidate,
            output_gate,
            next_state,
            grad_update_gate,
            grad_retain_gate,
            grad_transformed_state,
            grad_candidate,
            grad_output_gate,
            grad_next_state_contiguous.stride(0),
            ctx.input_row_stride,
            ctx.output_row_stride,
            grad_update_gate.stride(0),
            ctx.width,
            BLOCK_SIZE=ctx.block_size,
        )
        return (
            grad_update_gate,
            grad_retain_gate,
            grad_transformed_state,
            grad_candidate,
            grad_output_gate,
        )


class _ParcaeTritonStateMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state: torch.Tensor, decay: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ensure_triton_runtime_available()
        if state.device.type != "cuda":
            raise RuntimeError("Parcae Triton state mix requires CUDA tensors")
        if state.shape != injection.shape:
            raise RuntimeError("Parcae Triton state mix requires state and injection shapes to match")
        if decay.shape[-1] != state.shape[-1]:
            raise RuntimeError("Parcae Triton state mix requires decay width to match state width")
        state_contiguous = state.contiguous()
        injection_contiguous = injection.contiguous()
        decay_contiguous = decay.reshape(-1, decay.shape[-1]).contiguous()
        output = torch.empty_like(state_contiguous)
        n_elements = output.numel()
        width = output.shape[-1]
        block_size = 256
        grid = (triton.cdiv(n_elements, block_size),)
        _parcae_state_mix_forward_kernel[grid](
            state_contiguous,
            decay_contiguous,
            injection_contiguous,
            output,
            n_elements,
            width,
            BLOCK_SIZE=block_size,
        )
        ctx.decay_shape = decay.shape
        ctx.injection_shape = injection.shape
        ctx.save_for_backward(state_contiguous, decay_contiguous)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        state, decay = ctx.saved_tensors
        grad = grad_output.contiguous()
        grad_state = grad * decay.view(*((1,) * (grad.ndim - 1)), decay.shape[-1])
        grad_decay = _sum_to_broadcast_owner(grad * state, ctx.decay_shape)
        grad_injection = _sum_to_broadcast_owner(grad, ctx.injection_shape)
        return grad_state, grad_decay, grad_injection


class _ParcaeTritonResidualMix(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        mixed: torch.Tensor,
        block_out: torch.Tensor,
        nonlinear: torch.Tensor,
    ) -> torch.Tensor:
        ensure_triton_runtime_available()
        if mixed.device.type != "cuda":
            raise RuntimeError("Parcae Triton residual mix requires CUDA tensors")
        if mixed.shape != block_out.shape:
            raise RuntimeError("Parcae Triton residual mix requires mixed and block_out shapes to match")
        if nonlinear.shape[-1] != mixed.shape[-1]:
            raise RuntimeError("Parcae Triton residual mix requires nonlinear width to match mixed width")
        mixed_contiguous = mixed.contiguous()
        block_out_contiguous = block_out.contiguous()
        nonlinear_contiguous = nonlinear.reshape(-1, nonlinear.shape[-1]).contiguous()
        output = torch.empty_like(mixed_contiguous)
        n_elements = output.numel()
        width = output.shape[-1]
        block_size = 256
        grid = (triton.cdiv(n_elements, block_size),)
        _parcae_residual_mix_forward_kernel[grid](
            mixed_contiguous,
            block_out_contiguous,
            nonlinear_contiguous,
            output,
            n_elements,
            width,
            BLOCK_SIZE=block_size,
        )
        ctx.nonlinear_shape = nonlinear.shape
        ctx.save_for_backward(mixed_contiguous, block_out_contiguous, nonlinear_contiguous)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        mixed, block_out, nonlinear = ctx.saved_tensors
        grad = grad_output.contiguous()
        nonlinear_view = nonlinear.view(*((1,) * (grad.ndim - 1)), nonlinear.shape[-1])
        grad_mixed = grad * (1.0 - nonlinear_view)
        grad_block_out = grad * nonlinear_view
        grad_nonlinear = _sum_to_broadcast_owner(grad * (block_out - mixed), ctx.nonlinear_shape)
        return grad_mixed, grad_block_out, grad_nonlinear


class _ParcaeTritonLoopUpdate(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        state: torch.Tensor,
        decay: torch.Tensor,
        injection: torch.Tensor,
        block_out: torch.Tensor,
        nonlinear: torch.Tensor,
    ) -> torch.Tensor:
        ensure_triton_runtime_available()
        if state.device.type != "cuda":
            raise RuntimeError("Parcae Triton loop update requires CUDA tensors")
        if state.shape != injection.shape or state.shape != block_out.shape:
            raise RuntimeError("Parcae Triton loop update requires state, injection, and block_out shapes to match")
        if decay.shape[-1] != state.shape[-1] or nonlinear.shape[-1] != state.shape[-1]:
            raise RuntimeError("Parcae Triton loop update requires decay/nonlinear widths to match state width")
        state_contiguous = state.contiguous()
        injection_contiguous = injection.contiguous()
        block_out_contiguous = block_out.contiguous()
        decay_contiguous = decay.reshape(-1, decay.shape[-1]).contiguous()
        nonlinear_contiguous = nonlinear.reshape(-1, nonlinear.shape[-1]).contiguous()
        output = torch.empty_like(state_contiguous)
        n_elements = output.numel()
        width = output.shape[-1]
        block_size = 256
        grid = (triton.cdiv(n_elements, block_size),)
        _parcae_loop_update_forward_kernel[grid](
            state_contiguous,
            decay_contiguous,
            injection_contiguous,
            block_out_contiguous,
            nonlinear_contiguous,
            output,
            n_elements,
            width,
            BLOCK_SIZE=block_size,
        )
        ctx.decay_shape = decay.shape
        ctx.injection_shape = injection.shape
        ctx.nonlinear_shape = nonlinear.shape
        ctx.save_for_backward(state_contiguous, decay_contiguous, injection_contiguous, block_out_contiguous, nonlinear_contiguous)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        state, decay, injection, block_out, nonlinear = ctx.saved_tensors
        grad = grad_output.contiguous()
        decay_view = decay.view(*((1,) * (grad.ndim - 1)), decay.shape[-1])
        nonlinear_view = nonlinear.view(*((1,) * (grad.ndim - 1)), nonlinear.shape[-1])
        grad_mixed = grad * (1.0 - nonlinear_view)
        grad_state = grad_mixed * decay_view
        grad_decay = _sum_to_broadcast_owner(grad_mixed * state, ctx.decay_shape)
        grad_injection = _sum_to_broadcast_owner(grad_mixed, ctx.injection_shape)
        grad_block_out = grad * nonlinear_view
        mixed = decay_view * state + injection
        grad_nonlinear = _sum_to_broadcast_owner(grad * (block_out - mixed), ctx.nonlinear_shape)
        return grad_state, grad_decay, grad_injection, grad_block_out, grad_nonlinear


class _ParcaeTritonOutputMix(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        anchor: torch.Tensor,
        delta: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        ensure_triton_runtime_available()
        if anchor.device.type != "cuda":
            raise RuntimeError("Parcae Triton output mix requires CUDA tensors")
        if anchor.shape != delta.shape:
            raise RuntimeError("Parcae Triton output mix requires anchor and delta shapes to match")
        if gate.shape[-1] != anchor.shape[-1]:
            raise RuntimeError("Parcae Triton output mix requires gate width to match anchor width")
        anchor_contiguous = anchor.contiguous()
        delta_contiguous = delta.contiguous()
        gate_contiguous = gate.reshape(-1, gate.shape[-1]).contiguous()
        output = torch.empty_like(anchor_contiguous)
        n_elements = output.numel()
        width = output.shape[-1]
        block_size = 256
        grid = (triton.cdiv(n_elements, block_size),)
        _parcae_output_mix_forward_kernel[grid](
            anchor_contiguous,
            delta_contiguous,
            gate_contiguous,
            output,
            n_elements,
            width,
            BLOCK_SIZE=block_size,
        )
        ctx.gate_shape = gate.shape
        ctx.save_for_backward(delta_contiguous, gate_contiguous)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        delta, gate = ctx.saved_tensors
        grad = grad_output.contiguous()
        gate_view = gate.view(*((1,) * (grad.ndim - 1)), gate.shape[-1])
        grad_anchor = grad
        grad_delta = grad * gate_view
        grad_gate = _sum_to_broadcast_owner(grad * delta, ctx.gate_shape)
        return grad_anchor, grad_delta, grad_gate


class _TritonGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ensure_triton_runtime_available()
        if inputs.device.type != "cuda":
            raise RuntimeError("Triton GELU requires CUDA tensors")
        contiguous = inputs.contiguous()
        output = torch.empty_like(contiguous)
        n_elements = output.numel()
        block_size = 256
        grid = (triton.cdiv(n_elements, block_size),)
        _gelu_forward_kernel[grid](
            contiguous,
            output,
            n_elements,
            BLOCK_SIZE=block_size,
        )
        ctx.save_for_backward(contiguous)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (inputs,) = ctx.saved_tensors
        grad_output_contiguous = grad_output.contiguous()
        grad_input = torch.empty_like(inputs)
        n_elements = grad_input.numel()
        block_size = 256
        grid = (triton.cdiv(n_elements, block_size),)
        _gelu_backward_kernel[grid](
            grad_output_contiguous,
            inputs,
            grad_input,
            n_elements,
            BLOCK_SIZE=block_size,
        )
        return grad_input


if triton_runtime_available():  # pragma: no branch

    @triton.jit
    def _p20_dense_sequence_forward_kernel(
        update_gate_ptr,
        retain_gate_ptr,
        angle_cos_ptr,
        angle_sin_ptr,
        candidate_ptr,
        output_gate_ptr,
        initial_state_ptr,
        transform_weight_ptr,
        transform_bias_ptr,
        state_history_ptr,
        emitted_output_ptr,
        tensor_batch_stride,
        tensor_seq_stride,
        pair_batch_stride,
        pair_seq_stride,
        state_batch_stride,
        history_batch_stride,
        history_seq_stride,
        output_batch_stride,
        output_seq_stride,
        weight_row_stride,
        weight_col_stride,
        SEQ_LEN: tl.constexpr,
        PAIR_WIDTH: tl.constexpr,
        BLOCK_PAIR_WIDTH: tl.constexpr,
    ):
        batch_index = tl.program_id(0)

        pair_offsets = tl.arange(0, BLOCK_PAIR_WIDTH)
        pair_mask = pair_offsets < PAIR_WIDTH
        even_offsets = pair_offsets * 2
        odd_offsets = even_offsets + 1
        matrix_mask = pair_mask[:, None] & pair_mask[None, :]
        row_even = even_offsets[:, None]
        row_odd = odd_offsets[:, None]
        col_even = even_offsets[None, :]
        col_odd = odd_offsets[None, :]

        weight_even_even = tl.load(
            transform_weight_ptr + row_even * weight_row_stride + col_even * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        weight_even_odd = tl.load(
            transform_weight_ptr + row_even * weight_row_stride + col_odd * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        weight_odd_even = tl.load(
            transform_weight_ptr + row_odd * weight_row_stride + col_even * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        weight_odd_odd = tl.load(
            transform_weight_ptr + row_odd * weight_row_stride + col_odd * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        bias_even = tl.load(transform_bias_ptr + even_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        bias_odd = tl.load(transform_bias_ptr + odd_offsets, mask=pair_mask, other=0.0).to(tl.float32)

        state_base = batch_index * state_batch_stride
        state_even = tl.load(initial_state_ptr + state_base + even_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        state_odd = tl.load(initial_state_ptr + state_base + odd_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )

        history_batch_base = batch_index * history_batch_stride
        tl.store(state_history_ptr + history_batch_base + even_offsets, state_even, mask=pair_mask)
        tl.store(state_history_ptr + history_batch_base + odd_offsets, state_odd, mask=pair_mask)

        tensor_batch_base = batch_index * tensor_batch_stride
        pair_batch_base = batch_index * pair_batch_stride
        output_batch_base = batch_index * output_batch_stride

        for position in range(SEQ_LEN):
            projected_even = (
                tl.sum(weight_even_even * state_even[None, :], axis=1)
                + tl.sum(weight_even_odd * state_odd[None, :], axis=1)
                + bias_even
            )
            projected_odd = (
                tl.sum(weight_odd_even * state_even[None, :], axis=1)
                + tl.sum(weight_odd_odd * state_odd[None, :], axis=1)
                + bias_odd
            )

            pair_step_base = pair_batch_base + position * pair_seq_stride
            cos = tl.load(angle_cos_ptr + pair_step_base + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            sin = tl.load(angle_sin_ptr + pair_step_base + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            rotated_even = projected_even * cos - projected_odd * sin
            rotated_odd = projected_even * sin + projected_odd * cos

            tensor_step_base = tensor_batch_base + position * tensor_seq_stride
            update_even = tl.load(update_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            update_odd = tl.load(update_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            retain_even = tl.load(retain_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            retain_odd = tl.load(retain_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            candidate_even = tl.load(candidate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            candidate_odd = tl.load(candidate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            output_even = tl.load(output_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            output_odd = tl.load(output_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )

            next_even = update_even * rotated_even + retain_even * candidate_even
            next_odd = update_odd * rotated_odd + retain_odd * candidate_odd
            emitted_even = output_even * next_even
            emitted_odd = output_odd * next_odd

            output_step_base = output_batch_base + position * output_seq_stride
            tl.store(emitted_output_ptr + output_step_base + even_offsets, emitted_even, mask=pair_mask)
            tl.store(emitted_output_ptr + output_step_base + odd_offsets, emitted_odd, mask=pair_mask)

            history_step_base = history_batch_base + (position + 1) * history_seq_stride
            tl.store(state_history_ptr + history_step_base + even_offsets, next_even, mask=pair_mask)
            tl.store(state_history_ptr + history_step_base + odd_offsets, next_odd, mask=pair_mask)

            state_even = next_even
            state_odd = next_odd


    @triton.jit
    def _p20_dense_sequence_backward_kernel(
        grad_emitted_output_ptr,
        grad_final_state_ptr,
        update_gate_ptr,
        retain_gate_ptr,
        angle_cos_ptr,
        angle_sin_ptr,
        candidate_ptr,
        output_gate_ptr,
        state_history_ptr,
        transform_weight_ptr,
        transform_bias_ptr,
        grad_update_gate_ptr,
        grad_retain_gate_ptr,
        grad_angle_cos_ptr,
        grad_angle_sin_ptr,
        grad_candidate_ptr,
        grad_output_gate_ptr,
        grad_initial_state_ptr,
        grad_transform_weight_ptr,
        grad_transform_bias_ptr,
        tensor_batch_stride,
        tensor_seq_stride,
        pair_batch_stride,
        pair_seq_stride,
        state_batch_stride,
        history_batch_stride,
        history_seq_stride,
        weight_row_stride,
        weight_col_stride,
        grad_weight_batch_stride,
        grad_weight_row_stride,
        grad_weight_col_stride,
        grad_bias_batch_stride,
        SEQ_LEN: tl.constexpr,
        PAIR_WIDTH: tl.constexpr,
        BLOCK_PAIR_WIDTH: tl.constexpr,
    ):
        batch_index = tl.program_id(0)

        pair_offsets = tl.arange(0, BLOCK_PAIR_WIDTH)
        pair_mask = pair_offsets < PAIR_WIDTH
        even_offsets = pair_offsets * 2
        odd_offsets = even_offsets + 1
        matrix_mask = pair_mask[:, None] & pair_mask[None, :]
        row_even = even_offsets[:, None]
        row_odd = odd_offsets[:, None]
        col_even = even_offsets[None, :]
        col_odd = odd_offsets[None, :]

        weight_even_even = tl.load(
            transform_weight_ptr + row_even * weight_row_stride + col_even * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        weight_even_odd = tl.load(
            transform_weight_ptr + row_even * weight_row_stride + col_odd * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        weight_odd_even = tl.load(
            transform_weight_ptr + row_odd * weight_row_stride + col_even * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        weight_odd_odd = tl.load(
            transform_weight_ptr + row_odd * weight_row_stride + col_odd * weight_col_stride,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        bias_even = tl.load(transform_bias_ptr + even_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        bias_odd = tl.load(transform_bias_ptr + odd_offsets, mask=pair_mask, other=0.0).to(tl.float32)

        grad_state_even = tl.load(
            grad_final_state_ptr + batch_index * state_batch_stride + even_offsets,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        grad_state_odd = tl.load(
            grad_final_state_ptr + batch_index * state_batch_stride + odd_offsets,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)

        grad_weight_even_even = tl.zeros((BLOCK_PAIR_WIDTH, BLOCK_PAIR_WIDTH), dtype=tl.float32)
        grad_weight_even_odd = tl.zeros((BLOCK_PAIR_WIDTH, BLOCK_PAIR_WIDTH), dtype=tl.float32)
        grad_weight_odd_even = tl.zeros((BLOCK_PAIR_WIDTH, BLOCK_PAIR_WIDTH), dtype=tl.float32)
        grad_weight_odd_odd = tl.zeros((BLOCK_PAIR_WIDTH, BLOCK_PAIR_WIDTH), dtype=tl.float32)
        grad_bias_even = tl.zeros((BLOCK_PAIR_WIDTH,), dtype=tl.float32)
        grad_bias_odd = tl.zeros((BLOCK_PAIR_WIDTH,), dtype=tl.float32)

        tensor_batch_base = batch_index * tensor_batch_stride
        pair_batch_base = batch_index * pair_batch_stride
        history_batch_base = batch_index * history_batch_stride

        for position in range(SEQ_LEN - 1, -1, -1):
            state_history_base = history_batch_base + position * history_seq_stride
            next_state_history_base = history_batch_base + (position + 1) * history_seq_stride
            state_even = tl.load(state_history_ptr + state_history_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            state_odd = tl.load(state_history_ptr + state_history_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            next_even = tl.load(
                state_history_ptr + next_state_history_base + even_offsets,
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)
            next_odd = tl.load(
                state_history_ptr + next_state_history_base + odd_offsets,
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)

            projected_even = (
                tl.sum(weight_even_even * state_even[None, :], axis=1)
                + tl.sum(weight_even_odd * state_odd[None, :], axis=1)
                + bias_even
            )
            projected_odd = (
                tl.sum(weight_odd_even * state_even[None, :], axis=1)
                + tl.sum(weight_odd_odd * state_odd[None, :], axis=1)
                + bias_odd
            )

            pair_step_base = pair_batch_base + position * pair_seq_stride
            cos = tl.load(angle_cos_ptr + pair_step_base + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            sin = tl.load(angle_sin_ptr + pair_step_base + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            rotated_even = projected_even * cos - projected_odd * sin
            rotated_odd = projected_even * sin + projected_odd * cos

            tensor_step_base = tensor_batch_base + position * tensor_seq_stride
            update_even = tl.load(update_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            update_odd = tl.load(update_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            retain_even = tl.load(retain_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            retain_odd = tl.load(retain_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            candidate_even = tl.load(candidate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            candidate_odd = tl.load(candidate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            output_even = tl.load(output_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            output_odd = tl.load(output_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )

            grad_output_base = tensor_batch_base + position * tensor_seq_stride
            grad_emitted_even = tl.load(
                grad_emitted_output_ptr + grad_output_base + even_offsets,
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)
            grad_emitted_odd = tl.load(
                grad_emitted_output_ptr + grad_output_base + odd_offsets,
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)

            total_grad_next_even = grad_state_even + grad_emitted_even * output_even
            total_grad_next_odd = grad_state_odd + grad_emitted_odd * output_odd
            grad_output_even_step = grad_emitted_even * next_even
            grad_output_odd_step = grad_emitted_odd * next_odd

            grad_update_even = total_grad_next_even * rotated_even
            grad_update_odd = total_grad_next_odd * rotated_odd
            grad_retain_even = total_grad_next_even * candidate_even
            grad_retain_odd = total_grad_next_odd * candidate_odd
            grad_candidate_even = total_grad_next_even * retain_even
            grad_candidate_odd = total_grad_next_odd * retain_odd

            grad_rotated_even = total_grad_next_even * update_even
            grad_rotated_odd = total_grad_next_odd * update_odd
            grad_projected_even = grad_rotated_even * cos + grad_rotated_odd * sin
            grad_projected_odd = -grad_rotated_even * sin + grad_rotated_odd * cos
            grad_cos = grad_rotated_even * projected_even + grad_rotated_odd * projected_odd
            grad_sin = -grad_rotated_even * projected_odd + grad_rotated_odd * projected_even

            tl.store(grad_update_gate_ptr + grad_output_base + even_offsets, grad_update_even, mask=pair_mask)
            tl.store(grad_update_gate_ptr + grad_output_base + odd_offsets, grad_update_odd, mask=pair_mask)
            tl.store(grad_retain_gate_ptr + grad_output_base + even_offsets, grad_retain_even, mask=pair_mask)
            tl.store(grad_retain_gate_ptr + grad_output_base + odd_offsets, grad_retain_odd, mask=pair_mask)
            tl.store(grad_candidate_ptr + grad_output_base + even_offsets, grad_candidate_even, mask=pair_mask)
            tl.store(grad_candidate_ptr + grad_output_base + odd_offsets, grad_candidate_odd, mask=pair_mask)
            tl.store(grad_output_gate_ptr + grad_output_base + even_offsets, grad_output_even_step, mask=pair_mask)
            tl.store(grad_output_gate_ptr + grad_output_base + odd_offsets, grad_output_odd_step, mask=pair_mask)
            tl.store(grad_angle_cos_ptr + pair_step_base + pair_offsets, grad_cos, mask=pair_mask)
            tl.store(grad_angle_sin_ptr + pair_step_base + pair_offsets, grad_sin, mask=pair_mask)

            grad_weight_even_even += grad_projected_even[:, None] * state_even[None, :]
            grad_weight_even_odd += grad_projected_even[:, None] * state_odd[None, :]
            grad_weight_odd_even += grad_projected_odd[:, None] * state_even[None, :]
            grad_weight_odd_odd += grad_projected_odd[:, None] * state_odd[None, :]
            grad_bias_even += grad_projected_even
            grad_bias_odd += grad_projected_odd

            grad_state_even = (
                tl.sum(weight_even_even * grad_projected_even[:, None], axis=0)
                + tl.sum(weight_odd_even * grad_projected_odd[:, None], axis=0)
            )
            grad_state_odd = (
                tl.sum(weight_even_odd * grad_projected_even[:, None], axis=0)
                + tl.sum(weight_odd_odd * grad_projected_odd[:, None], axis=0)
            )

        grad_initial_base = batch_index * state_batch_stride
        tl.store(grad_initial_state_ptr + grad_initial_base + even_offsets, grad_state_even, mask=pair_mask)
        tl.store(grad_initial_state_ptr + grad_initial_base + odd_offsets, grad_state_odd, mask=pair_mask)

        grad_bias_base = batch_index * grad_bias_batch_stride
        tl.store(grad_transform_bias_ptr + grad_bias_base + even_offsets, grad_bias_even, mask=pair_mask)
        tl.store(grad_transform_bias_ptr + grad_bias_base + odd_offsets, grad_bias_odd, mask=pair_mask)

        grad_weight_base = batch_index * grad_weight_batch_stride
        tl.store(
            grad_transform_weight_ptr + grad_weight_base + row_even * grad_weight_row_stride + col_even * grad_weight_col_stride,
            grad_weight_even_even,
            mask=matrix_mask,
        )
        tl.store(
            grad_transform_weight_ptr + grad_weight_base + row_even * grad_weight_row_stride + col_odd * grad_weight_col_stride,
            grad_weight_even_odd,
            mask=matrix_mask,
        )
        tl.store(
            grad_transform_weight_ptr + grad_weight_base + row_odd * grad_weight_row_stride + col_even * grad_weight_col_stride,
            grad_weight_odd_even,
            mask=matrix_mask,
        )
        tl.store(
            grad_transform_weight_ptr + grad_weight_base + row_odd * grad_weight_row_stride + col_odd * grad_weight_col_stride,
            grad_weight_odd_odd,
            mask=matrix_mask,
        )


class _P20DenseSequenceScan(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        update_gate: torch.Tensor,
        retain_gate: torch.Tensor,
        angle_cos: torch.Tensor,
        angle_sin: torch.Tensor,
        candidate: torch.Tensor,
        output_gate: torch.Tensor,
        initial_state: torch.Tensor,
        transform_weight: torch.Tensor,
        transform_bias: torch.Tensor,
        identity_transform: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ensure_triton_runtime_available()
        tensors = (
            update_gate,
            retain_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
        )
        if any(tensor.device.type != "cuda" for tensor in tensors):
            raise RuntimeError("P20 Triton dense sequence scan requires CUDA tensors")

        update_gate = update_gate.contiguous()
        retain_gate = retain_gate.contiguous()
        angle_cos = angle_cos.contiguous()
        angle_sin = angle_sin.contiguous()
        candidate = candidate.contiguous()
        output_gate = output_gate.contiguous()
        initial_state = initial_state.contiguous()
        transform_weight = transform_weight.contiguous()
        transform_bias = transform_bias.contiguous()

        batch_size, seq_len, width = update_gate.shape
        if retain_gate.shape != update_gate.shape or candidate.shape != update_gate.shape or output_gate.shape != update_gate.shape:
            raise RuntimeError("P20 Triton dense sequence scan requires matching [batch, seq, width] tensor shapes")
        if initial_state.shape != (batch_size, width):
            raise RuntimeError("P20 Triton dense sequence scan requires initial_state shape [batch, width]")
        if angle_cos.shape != angle_sin.shape:
            raise RuntimeError("P20 Triton dense sequence scan requires matching cosine and sine runtime plans")
        if angle_cos.shape[:2] != update_gate.shape[:2]:
            raise RuntimeError("P20 Triton dense sequence scan requires angle tensors aligned with [batch, seq]")
        if transform_weight.shape != (width, width):
            raise RuntimeError("P20 Triton dense sequence scan requires dense transform weight shape [width, width]")
        if transform_bias.shape != (width,):
            raise RuntimeError("P20 Triton dense sequence scan requires dense transform bias shape [width]")
        if width % 2 != 0:
            raise RuntimeError("P20 Triton dense sequence scan requires even state width")
        pair_width = width // 2
        if angle_cos.shape[2] != pair_width:
            raise RuntimeError("P20 Triton dense sequence scan requires angle width equal to state_width // 2")
        if pair_width > 64:
            raise RuntimeError("P20 Triton dense sequence scan currently supports widths up to 128")
        block_pair_width = _next_power_of_two(pair_width)
        emitted_outputs = torch.empty_like(update_gate)
        state_history = torch.empty(
            batch_size,
            seq_len + 1,
            width,
            device=update_gate.device,
            dtype=update_gate.dtype,
        )

        grid = (batch_size,)
        _p20_dense_sequence_forward_kernel[grid](
            update_gate,
            retain_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
            state_history,
            emitted_outputs,
            update_gate.stride(0),
            update_gate.stride(1),
            angle_cos.stride(0),
            angle_cos.stride(1),
            initial_state.stride(0),
            state_history.stride(0),
            state_history.stride(1),
            emitted_outputs.stride(0),
            emitted_outputs.stride(1),
            transform_weight.stride(0),
            transform_weight.stride(1),
            SEQ_LEN=seq_len,
            PAIR_WIDTH=pair_width,
            BLOCK_PAIR_WIDTH=block_pair_width,
        )

        ctx.seq_len = seq_len
        ctx.block_pair_width = block_pair_width
        ctx.pair_width = pair_width
        ctx.save_for_backward(
            update_gate,
            retain_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
            state_history,
        )
        return emitted_outputs, state_history[:, -1, :].contiguous()

    @staticmethod
    def backward(ctx, grad_emitted_outputs: torch.Tensor | None, grad_final_state: torch.Tensor | None):  # type: ignore[override]
        (
            update_gate,
            retain_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
            state_history,
        ) = ctx.saved_tensors
        if grad_emitted_outputs is None:
            grad_emitted_outputs = torch.zeros_like(update_gate)
        else:
            grad_emitted_outputs = grad_emitted_outputs.contiguous()
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(initial_state)
        else:
            grad_final_state = grad_final_state.contiguous()

        batch_size, _seq_len, width = update_gate.shape
        grad_update_gate = torch.empty_like(update_gate)
        grad_retain_gate = torch.empty_like(retain_gate)
        grad_angle_cos = torch.empty_like(angle_cos)
        grad_angle_sin = torch.empty_like(angle_sin)
        grad_candidate = torch.empty_like(candidate)
        grad_output_gate = torch.empty_like(output_gate)
        grad_initial_state = torch.empty_like(initial_state)
        grad_transform_weight_contrib = torch.empty(
            batch_size,
            width,
            width,
            device=transform_weight.device,
            dtype=transform_weight.dtype,
        )
        grad_transform_bias_contrib = torch.empty(
            batch_size,
            width,
            device=transform_bias.device,
            dtype=transform_bias.dtype,
        )
        grid = (batch_size,)
        _p20_dense_sequence_backward_kernel[grid](
            grad_emitted_outputs,
            grad_final_state,
            update_gate,
            retain_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            state_history,
            transform_weight,
            transform_bias,
            grad_update_gate,
            grad_retain_gate,
            grad_angle_cos,
            grad_angle_sin,
            grad_candidate,
            grad_output_gate,
            grad_initial_state,
            grad_transform_weight_contrib,
            grad_transform_bias_contrib,
            update_gate.stride(0),
            update_gate.stride(1),
            angle_cos.stride(0),
            angle_cos.stride(1),
            initial_state.stride(0),
            state_history.stride(0),
            state_history.stride(1),
            transform_weight.stride(0),
            transform_weight.stride(1),
            grad_transform_weight_contrib.stride(0),
            grad_transform_weight_contrib.stride(1),
            grad_transform_weight_contrib.stride(2),
            grad_transform_bias_contrib.stride(0),
            SEQ_LEN=ctx.seq_len,
            PAIR_WIDTH=ctx.pair_width,
            BLOCK_PAIR_WIDTH=ctx.block_pair_width,
        )
        grad_transform_weight = grad_transform_weight_contrib.sum(dim=0)
        grad_transform_bias = grad_transform_bias_contrib.sum(dim=0)
        return (
            grad_update_gate,
            grad_retain_gate,
            grad_angle_cos,
            grad_angle_sin,
            grad_candidate,
            grad_output_gate,
            grad_initial_state,
            grad_transform_weight,
            grad_transform_bias,
        )


if triton_runtime_available():  # pragma: no branch

    @triton.jit
    def _p20_block_diagonal_sequence_forward_kernel(
        update_gate_ptr,
        retain_gate_ptr,
        angle_cos_ptr,
        angle_sin_ptr,
        candidate_ptr,
        output_gate_ptr,
        initial_state_ptr,
        transform_weight_ptr,
        transform_bias_ptr,
        state_history_ptr,
        emitted_output_ptr,
        tensor_batch_stride,
        tensor_seq_stride,
        pair_batch_stride,
        pair_seq_stride,
        state_batch_stride,
        history_batch_stride,
        history_seq_stride,
        output_batch_stride,
        output_seq_stride,
        weight_block_stride,
        weight_row_stride,
        weight_col_stride,
        block_width,
        pair_width,
        SEQ_LEN: tl.constexpr,
        BLOCK_PAIR_WIDTH: tl.constexpr,
        IDENTITY_TRANSFORM: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block_index = tl.program_id(1)

        pair_offsets = tl.arange(0, BLOCK_PAIR_WIDTH)
        pair_mask = pair_offsets < pair_width
        even_offsets = pair_offsets * 2
        odd_offsets = even_offsets + 1
        block_base = block_index * block_width
        global_even_offsets = block_base + even_offsets
        global_odd_offsets = block_base + odd_offsets

        matrix_mask = pair_mask[:, None] & pair_mask[None, :]
        row_even = even_offsets[:, None]
        row_odd = odd_offsets[:, None]
        col_even = even_offsets[None, :]
        col_odd = odd_offsets[None, :]
        if not IDENTITY_TRANSFORM:
            bias_even = tl.load(transform_bias_ptr + global_even_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            bias_odd = tl.load(transform_bias_ptr + global_odd_offsets, mask=pair_mask, other=0.0).to(tl.float32)

            weight_block_ptr = transform_weight_ptr + block_index * weight_block_stride
            weight_even_even = tl.load(
                weight_block_ptr + row_even * weight_row_stride + col_even * weight_col_stride,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            weight_even_odd = tl.load(
                weight_block_ptr + row_even * weight_row_stride + col_odd * weight_col_stride,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            weight_odd_even = tl.load(
                weight_block_ptr + row_odd * weight_row_stride + col_even * weight_col_stride,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            weight_odd_odd = tl.load(
                weight_block_ptr + row_odd * weight_row_stride + col_odd * weight_col_stride,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)

        state_base = batch_index * state_batch_stride
        state_even = tl.load(initial_state_ptr + state_base + global_even_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        state_odd = tl.load(initial_state_ptr + state_base + global_odd_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )

        history_batch_base = batch_index * history_batch_stride
        tl.store(state_history_ptr + history_batch_base + global_even_offsets, state_even, mask=pair_mask)
        tl.store(state_history_ptr + history_batch_base + global_odd_offsets, state_odd, mask=pair_mask)

        tensor_batch_base = batch_index * tensor_batch_stride
        pair_batch_base = batch_index * pair_batch_stride
        output_batch_base = batch_index * output_batch_stride

        for position in range(SEQ_LEN):
            if IDENTITY_TRANSFORM:
                projected_even = state_even
                projected_odd = state_odd
            else:
                projected_even = (
                    tl.sum(weight_even_even * state_even[None, :], axis=1)
                    + tl.sum(weight_even_odd * state_odd[None, :], axis=1)
                    + bias_even
                )
                projected_odd = (
                    tl.sum(weight_odd_even * state_even[None, :], axis=1)
                    + tl.sum(weight_odd_odd * state_odd[None, :], axis=1)
                    + bias_odd
                )

            pair_step_base = pair_batch_base + position * pair_seq_stride + block_index * pair_width
            cos = tl.load(angle_cos_ptr + pair_step_base + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            sin = tl.load(angle_sin_ptr + pair_step_base + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            rotated_even = projected_even * cos - projected_odd * sin
            rotated_odd = projected_even * sin + projected_odd * cos

            tensor_step_base = tensor_batch_base + position * tensor_seq_stride + block_base
            update_even = tl.load(update_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            update_odd = tl.load(update_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            retain_even = tl.load(retain_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            retain_odd = tl.load(retain_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            candidate_even = tl.load(candidate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            candidate_odd = tl.load(candidate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            output_even = tl.load(output_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            output_odd = tl.load(output_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )

            next_even = update_even * rotated_even + retain_even * candidate_even
            next_odd = update_odd * rotated_odd + retain_odd * candidate_odd
            emitted_even = output_even * next_even
            emitted_odd = output_odd * next_odd

            output_step_base = output_batch_base + position * output_seq_stride + block_base
            tl.store(emitted_output_ptr + output_step_base + even_offsets, emitted_even, mask=pair_mask)
            tl.store(emitted_output_ptr + output_step_base + odd_offsets, emitted_odd, mask=pair_mask)

            history_step_base = history_batch_base + (position + 1) * history_seq_stride + block_base
            tl.store(state_history_ptr + history_step_base + even_offsets, next_even, mask=pair_mask)
            tl.store(state_history_ptr + history_step_base + odd_offsets, next_odd, mask=pair_mask)

            state_even = next_even
            state_odd = next_odd


    @triton.jit
    def _p20_block_diagonal_sequence_backward_kernel(
        grad_emitted_output_ptr,
        grad_final_state_ptr,
        update_gate_ptr,
        retain_gate_ptr,
        angle_cos_ptr,
        angle_sin_ptr,
        candidate_ptr,
        output_gate_ptr,
        state_history_ptr,
        transform_weight_ptr,
        transform_bias_ptr,
        grad_update_gate_ptr,
        grad_retain_gate_ptr,
        grad_angle_cos_ptr,
        grad_angle_sin_ptr,
        grad_candidate_ptr,
        grad_output_gate_ptr,
        grad_initial_state_ptr,
        grad_transform_weight_ptr,
        grad_transform_bias_ptr,
        tensor_batch_stride,
        tensor_seq_stride,
        pair_batch_stride,
        pair_seq_stride,
        state_batch_stride,
        history_batch_stride,
        history_seq_stride,
        weight_block_stride,
        weight_row_stride,
        weight_col_stride,
        grad_weight_batch_stride,
        grad_weight_block_stride,
        grad_weight_row_stride,
        grad_weight_col_stride,
        grad_bias_batch_stride,
        block_width,
        pair_width,
        SEQ_LEN: tl.constexpr,
        BLOCK_PAIR_WIDTH: tl.constexpr,
        COMPUTE_TRANSFORM_GRAD: tl.constexpr,
        ATOMIC_TRANSFORM_GRAD: tl.constexpr,
        IDENTITY_TRANSFORM: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        block_index = tl.program_id(1)

        pair_offsets = tl.arange(0, BLOCK_PAIR_WIDTH)
        pair_mask = pair_offsets < pair_width
        even_offsets = pair_offsets * 2
        odd_offsets = even_offsets + 1
        block_base = block_index * block_width
        global_even_offsets = block_base + even_offsets
        global_odd_offsets = block_base + odd_offsets

        matrix_mask = pair_mask[:, None] & pair_mask[None, :]
        row_even = even_offsets[:, None]
        row_odd = odd_offsets[:, None]
        col_even = even_offsets[None, :]
        col_odd = odd_offsets[None, :]
        if not IDENTITY_TRANSFORM:
            weight_block_ptr = transform_weight_ptr + block_index * weight_block_stride
            weight_even_even = tl.load(
                weight_block_ptr + row_even * weight_row_stride + col_even * weight_col_stride,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            weight_even_odd = tl.load(
                weight_block_ptr + row_even * weight_row_stride + col_odd * weight_col_stride,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            weight_odd_even = tl.load(
                weight_block_ptr + row_odd * weight_row_stride + col_even * weight_col_stride,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            weight_odd_odd = tl.load(
                weight_block_ptr + row_odd * weight_row_stride + col_odd * weight_col_stride,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            bias_even = tl.load(transform_bias_ptr + global_even_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            bias_odd = tl.load(transform_bias_ptr + global_odd_offsets, mask=pair_mask, other=0.0).to(tl.float32)

        grad_state_even = tl.load(
            grad_final_state_ptr + batch_index * state_batch_stride + global_even_offsets,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        grad_state_odd = tl.load(
            grad_final_state_ptr + batch_index * state_batch_stride + global_odd_offsets,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)

        if COMPUTE_TRANSFORM_GRAD:
            grad_weight_even_even = tl.zeros((BLOCK_PAIR_WIDTH, BLOCK_PAIR_WIDTH), dtype=tl.float32)
            grad_weight_even_odd = tl.zeros((BLOCK_PAIR_WIDTH, BLOCK_PAIR_WIDTH), dtype=tl.float32)
            grad_weight_odd_even = tl.zeros((BLOCK_PAIR_WIDTH, BLOCK_PAIR_WIDTH), dtype=tl.float32)
            grad_weight_odd_odd = tl.zeros((BLOCK_PAIR_WIDTH, BLOCK_PAIR_WIDTH), dtype=tl.float32)
            grad_bias_even = tl.zeros((BLOCK_PAIR_WIDTH,), dtype=tl.float32)
            grad_bias_odd = tl.zeros((BLOCK_PAIR_WIDTH,), dtype=tl.float32)

        tensor_batch_base = batch_index * tensor_batch_stride
        pair_batch_base = batch_index * pair_batch_stride
        history_batch_base = batch_index * history_batch_stride

        for position in range(SEQ_LEN - 1, -1, -1):
            state_history_base = history_batch_base + position * history_seq_stride + block_base
            next_state_history_base = history_batch_base + (position + 1) * history_seq_stride + block_base
            state_even = tl.load(state_history_ptr + state_history_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            state_odd = tl.load(state_history_ptr + state_history_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            next_even = tl.load(
                state_history_ptr + next_state_history_base + even_offsets,
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)
            next_odd = tl.load(
                state_history_ptr + next_state_history_base + odd_offsets,
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)

            if IDENTITY_TRANSFORM:
                projected_even = state_even
                projected_odd = state_odd
            else:
                projected_even = (
                    tl.sum(weight_even_even * state_even[None, :], axis=1)
                    + tl.sum(weight_even_odd * state_odd[None, :], axis=1)
                    + bias_even
                )
                projected_odd = (
                    tl.sum(weight_odd_even * state_even[None, :], axis=1)
                    + tl.sum(weight_odd_odd * state_odd[None, :], axis=1)
                    + bias_odd
                )

            pair_step_base = pair_batch_base + position * pair_seq_stride + block_index * pair_width
            cos = tl.load(angle_cos_ptr + pair_step_base + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            sin = tl.load(angle_sin_ptr + pair_step_base + pair_offsets, mask=pair_mask, other=0.0).to(tl.float32)
            rotated_even = projected_even * cos - projected_odd * sin
            rotated_odd = projected_even * sin + projected_odd * cos

            tensor_step_base = tensor_batch_base + position * tensor_seq_stride + block_base
            update_even = tl.load(update_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            update_odd = tl.load(update_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            retain_even = tl.load(retain_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            retain_odd = tl.load(retain_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            candidate_even = tl.load(candidate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            candidate_odd = tl.load(candidate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            output_even = tl.load(output_gate_ptr + tensor_step_base + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            output_odd = tl.load(output_gate_ptr + tensor_step_base + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )

            grad_output_base = tensor_batch_base + position * tensor_seq_stride + block_base
            grad_emitted_even = tl.load(
                grad_emitted_output_ptr + grad_output_base + even_offsets,
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)
            grad_emitted_odd = tl.load(
                grad_emitted_output_ptr + grad_output_base + odd_offsets,
                mask=pair_mask,
                other=0.0,
            ).to(tl.float32)

            total_grad_next_even = grad_state_even + grad_emitted_even * output_even
            total_grad_next_odd = grad_state_odd + grad_emitted_odd * output_odd
            grad_output_even_step = grad_emitted_even * next_even
            grad_output_odd_step = grad_emitted_odd * next_odd

            grad_update_even = total_grad_next_even * rotated_even
            grad_update_odd = total_grad_next_odd * rotated_odd
            grad_retain_even = total_grad_next_even * candidate_even
            grad_retain_odd = total_grad_next_odd * candidate_odd
            grad_candidate_even = total_grad_next_even * retain_even
            grad_candidate_odd = total_grad_next_odd * retain_odd

            grad_rotated_even = total_grad_next_even * update_even
            grad_rotated_odd = total_grad_next_odd * update_odd
            grad_projected_even = grad_rotated_even * cos + grad_rotated_odd * sin
            grad_projected_odd = -grad_rotated_even * sin + grad_rotated_odd * cos
            grad_cos = grad_rotated_even * projected_even + grad_rotated_odd * projected_odd
            grad_sin = -grad_rotated_even * projected_odd + grad_rotated_odd * projected_even

            tl.store(grad_update_gate_ptr + grad_output_base + even_offsets, grad_update_even, mask=pair_mask)
            tl.store(grad_update_gate_ptr + grad_output_base + odd_offsets, grad_update_odd, mask=pair_mask)
            tl.store(grad_retain_gate_ptr + grad_output_base + even_offsets, grad_retain_even, mask=pair_mask)
            tl.store(grad_retain_gate_ptr + grad_output_base + odd_offsets, grad_retain_odd, mask=pair_mask)
            tl.store(grad_candidate_ptr + grad_output_base + even_offsets, grad_candidate_even, mask=pair_mask)
            tl.store(grad_candidate_ptr + grad_output_base + odd_offsets, grad_candidate_odd, mask=pair_mask)
            tl.store(grad_output_gate_ptr + grad_output_base + even_offsets, grad_output_even_step, mask=pair_mask)
            tl.store(grad_output_gate_ptr + grad_output_base + odd_offsets, grad_output_odd_step, mask=pair_mask)
            tl.store(grad_angle_cos_ptr + pair_step_base + pair_offsets, grad_cos, mask=pair_mask)
            tl.store(grad_angle_sin_ptr + pair_step_base + pair_offsets, grad_sin, mask=pair_mask)

            if COMPUTE_TRANSFORM_GRAD:
                grad_weight_even_even += grad_projected_even[:, None] * state_even[None, :]
                grad_weight_even_odd += grad_projected_even[:, None] * state_odd[None, :]
                grad_weight_odd_even += grad_projected_odd[:, None] * state_even[None, :]
                grad_weight_odd_odd += grad_projected_odd[:, None] * state_odd[None, :]
                grad_bias_even += grad_projected_even
                grad_bias_odd += grad_projected_odd

            if IDENTITY_TRANSFORM:
                grad_state_even = grad_projected_even
                grad_state_odd = grad_projected_odd
            else:
                grad_state_even = (
                    tl.sum(weight_even_even * grad_projected_even[:, None], axis=0)
                    + tl.sum(weight_odd_even * grad_projected_odd[:, None], axis=0)
                )
                grad_state_odd = (
                    tl.sum(weight_even_odd * grad_projected_even[:, None], axis=0)
                    + tl.sum(weight_odd_odd * grad_projected_odd[:, None], axis=0)
                )

        grad_initial_base = batch_index * state_batch_stride + block_base
        tl.store(grad_initial_state_ptr + grad_initial_base + even_offsets, grad_state_even, mask=pair_mask)
        tl.store(grad_initial_state_ptr + grad_initial_base + odd_offsets, grad_state_odd, mask=pair_mask)

        if COMPUTE_TRANSFORM_GRAD:
            if ATOMIC_TRANSFORM_GRAD:
                grad_bias_base = block_base
                tl.atomic_add(grad_transform_bias_ptr + grad_bias_base + even_offsets, grad_bias_even, sem="relaxed", mask=pair_mask)
                tl.atomic_add(grad_transform_bias_ptr + grad_bias_base + odd_offsets, grad_bias_odd, sem="relaxed", mask=pair_mask)

                grad_weight_block_ptr = grad_transform_weight_ptr + block_index * grad_weight_block_stride
                tl.atomic_add(
                    grad_weight_block_ptr + row_even * grad_weight_row_stride + col_even * grad_weight_col_stride,
                    grad_weight_even_even,
                    sem="relaxed",
                    mask=matrix_mask,
                )
                tl.atomic_add(
                    grad_weight_block_ptr + row_even * grad_weight_row_stride + col_odd * grad_weight_col_stride,
                    grad_weight_even_odd,
                    sem="relaxed",
                    mask=matrix_mask,
                )
                tl.atomic_add(
                    grad_weight_block_ptr + row_odd * grad_weight_row_stride + col_even * grad_weight_col_stride,
                    grad_weight_odd_even,
                    sem="relaxed",
                    mask=matrix_mask,
                )
                tl.atomic_add(
                    grad_weight_block_ptr + row_odd * grad_weight_row_stride + col_odd * grad_weight_col_stride,
                    grad_weight_odd_odd,
                    sem="relaxed",
                    mask=matrix_mask,
                )
            else:
                grad_bias_base = batch_index * grad_bias_batch_stride + block_base
                tl.store(grad_transform_bias_ptr + grad_bias_base + even_offsets, grad_bias_even, mask=pair_mask)
                tl.store(grad_transform_bias_ptr + grad_bias_base + odd_offsets, grad_bias_odd, mask=pair_mask)

                grad_weight_block_ptr = (
                    grad_transform_weight_ptr
                    + batch_index * grad_weight_batch_stride
                    + block_index * grad_weight_block_stride
                )
                tl.store(
                    grad_weight_block_ptr + row_even * grad_weight_row_stride + col_even * grad_weight_col_stride,
                    grad_weight_even_even,
                    mask=matrix_mask,
                )
                tl.store(
                    grad_weight_block_ptr + row_even * grad_weight_row_stride + col_odd * grad_weight_col_stride,
                    grad_weight_even_odd,
                    mask=matrix_mask,
                )
                tl.store(
                    grad_weight_block_ptr + row_odd * grad_weight_row_stride + col_even * grad_weight_col_stride,
                    grad_weight_odd_even,
                    mask=matrix_mask,
                )
                tl.store(
                    grad_weight_block_ptr + row_odd * grad_weight_row_stride + col_odd * grad_weight_col_stride,
                    grad_weight_odd_odd,
                    mask=matrix_mask,
                )


class _P20BlockDiagonalSequenceScan(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        update_gate: torch.Tensor,
        retain_gate: torch.Tensor,
        angle_cos: torch.Tensor,
        angle_sin: torch.Tensor,
        candidate: torch.Tensor,
        output_gate: torch.Tensor,
        initial_state: torch.Tensor,
        transform_weight: torch.Tensor,
        transform_bias: torch.Tensor,
        identity_transform: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ensure_triton_runtime_available()
        tensors = (
            update_gate,
            retain_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
        )
        if any(tensor.device.type != "cuda" for tensor in tensors):
            raise RuntimeError("P20 Triton sequence scan requires CUDA tensors")

        update_gate = update_gate.contiguous()
        retain_gate = retain_gate.contiguous()
        angle_cos = angle_cos.contiguous()
        angle_sin = angle_sin.contiguous()
        candidate = candidate.contiguous()
        output_gate = output_gate.contiguous()
        initial_state = initial_state.contiguous()
        transform_weight = transform_weight.contiguous()
        transform_bias = transform_bias.contiguous()

        batch_size, seq_len, width = update_gate.shape
        if retain_gate.shape != update_gate.shape or candidate.shape != update_gate.shape or output_gate.shape != update_gate.shape:
            raise RuntimeError("P20 Triton sequence scan requires matching [batch, seq, width] tensor shapes")
        if initial_state.shape != (batch_size, width):
            raise RuntimeError("P20 Triton sequence scan requires initial_state shape [batch, width]")
        if angle_cos.shape != angle_sin.shape:
            raise RuntimeError("P20 Triton sequence scan requires matching cosine and sine runtime plans")
        if angle_cos.shape[:2] != update_gate.shape[:2]:
            raise RuntimeError("P20 Triton sequence scan requires angle tensors aligned with [batch, seq]")

        blocks, block_width, block_width_cols = transform_weight.shape
        if block_width != block_width_cols:
            raise RuntimeError("P20 Triton sequence scan requires square block-diagonal transform weights")
        if blocks * block_width != width:
            raise RuntimeError("P20 Triton sequence scan requires block-diagonal weight width to match state width")
        if block_width % 2 != 0:
            raise RuntimeError("P20 Triton sequence scan requires an even per-block width for rotary pairs")
        pair_width = block_width // 2
        expected_angle_width = width // 2
        if angle_cos.shape[2] != expected_angle_width:
            raise RuntimeError("P20 Triton sequence scan requires angle width equal to state_width // 2")
        block_pair_width = _next_power_of_two(pair_width)

        emitted_outputs = torch.empty_like(update_gate)
        state_history = torch.empty(
            batch_size,
            seq_len + 1,
            width,
            device=update_gate.device,
            dtype=update_gate.dtype,
        )

        grid = (batch_size, blocks)
        _p20_block_diagonal_sequence_forward_kernel[grid](
            update_gate,
            retain_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
            state_history,
            emitted_outputs,
            update_gate.stride(0),
            update_gate.stride(1),
            angle_cos.stride(0),
            angle_cos.stride(1),
            initial_state.stride(0),
            state_history.stride(0),
            state_history.stride(1),
            emitted_outputs.stride(0),
            emitted_outputs.stride(1),
            transform_weight.stride(0),
            transform_weight.stride(1),
            transform_weight.stride(2),
            block_width,
            pair_width,
            SEQ_LEN=seq_len,
            BLOCK_PAIR_WIDTH=block_pair_width,
            IDENTITY_TRANSFORM=bool(identity_transform),
        )

        ctx.seq_len = seq_len
        ctx.block_pair_width = block_pair_width
        ctx.block_width = block_width
        ctx.pair_width = pair_width
        ctx.identity_transform = bool(identity_transform)
        ctx.save_for_backward(
            update_gate,
            retain_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
            state_history,
        )
        return emitted_outputs, state_history[:, -1, :].contiguous()

    @staticmethod
    def backward(ctx, grad_emitted_outputs: torch.Tensor | None, grad_final_state: torch.Tensor | None):  # type: ignore[override]
        (
            update_gate,
            retain_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
            state_history,
        ) = ctx.saved_tensors
        if grad_emitted_outputs is None:
            grad_emitted_outputs = torch.zeros_like(update_gate)
        else:
            grad_emitted_outputs = grad_emitted_outputs.contiguous()
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(initial_state)
        else:
            grad_final_state = grad_final_state.contiguous()

        batch_size, _seq_len, width = update_gate.shape
        blocks, block_width, _block_width_cols = transform_weight.shape
        pair_width = ctx.pair_width

        grad_update_gate = torch.empty_like(update_gate)
        grad_retain_gate = torch.empty_like(retain_gate)
        grad_angle_cos = torch.empty_like(angle_cos)
        grad_angle_sin = torch.empty_like(angle_sin)
        grad_candidate = torch.empty_like(candidate)
        grad_output_gate = torch.empty_like(output_gate)
        grad_initial_state = torch.empty_like(initial_state)
        compute_transform_grad = bool((ctx.needs_input_grad[7] or ctx.needs_input_grad[8]) and not ctx.identity_transform)
        atomic_transform_grad = (
            compute_transform_grad
            and _p20_atomic_transform_grad_enabled(ctx.block_pair_width)
        )
        if compute_transform_grad:
            if atomic_transform_grad:
                grad_transform_weight_contrib = torch.zeros_like(transform_weight)
                grad_transform_bias_contrib = torch.zeros_like(transform_bias)
            else:
                grad_transform_weight_contrib = torch.empty(
                    batch_size,
                    blocks,
                    block_width,
                    block_width,
                    device=transform_weight.device,
                    dtype=transform_weight.dtype,
                )
                grad_transform_bias_contrib = torch.empty(
                    batch_size,
                    width,
                    device=transform_bias.device,
                    dtype=transform_bias.dtype,
                )
            grad_weight_batch_stride = grad_transform_weight_contrib.stride(0)
            grad_weight_block_stride = (
                grad_transform_weight_contrib.stride(0)
                if atomic_transform_grad
                else grad_transform_weight_contrib.stride(1)
            )
            grad_weight_row_stride = (
                grad_transform_weight_contrib.stride(1)
                if atomic_transform_grad
                else grad_transform_weight_contrib.stride(2)
            )
            grad_weight_col_stride = (
                grad_transform_weight_contrib.stride(2)
                if atomic_transform_grad
                else grad_transform_weight_contrib.stride(3)
            )
            grad_bias_batch_stride = grad_transform_bias_contrib.stride(0)
        else:
            grad_transform_weight_contrib = transform_weight
            grad_transform_bias_contrib = transform_bias
            grad_weight_batch_stride = 0
            grad_weight_block_stride = 0
            grad_weight_row_stride = 0
            grad_weight_col_stride = 0
            grad_bias_batch_stride = 0

        grid = (batch_size, blocks)
        with timed_region("path1.primitive.runtime.triton_sequence_scan_backward_kernel"):
            _p20_block_diagonal_sequence_backward_kernel[grid](
                grad_emitted_outputs,
                grad_final_state,
                update_gate,
                retain_gate,
                angle_cos,
                angle_sin,
                candidate,
                output_gate,
                state_history,
                transform_weight,
                transform_bias,
                grad_update_gate,
                grad_retain_gate,
                grad_angle_cos,
                grad_angle_sin,
                grad_candidate,
                grad_output_gate,
                grad_initial_state,
                grad_transform_weight_contrib,
                grad_transform_bias_contrib,
                update_gate.stride(0),
                update_gate.stride(1),
                angle_cos.stride(0),
                angle_cos.stride(1),
                initial_state.stride(0),
                state_history.stride(0),
                state_history.stride(1),
                transform_weight.stride(0),
                transform_weight.stride(1),
                transform_weight.stride(2),
                grad_weight_batch_stride,
                grad_weight_block_stride,
                grad_weight_row_stride,
                grad_weight_col_stride,
                grad_bias_batch_stride,
                block_width,
                pair_width,
                SEQ_LEN=ctx.seq_len,
                BLOCK_PAIR_WIDTH=ctx.block_pair_width,
                COMPUTE_TRANSFORM_GRAD=compute_transform_grad,
                ATOMIC_TRANSFORM_GRAD=atomic_transform_grad,
                IDENTITY_TRANSFORM=ctx.identity_transform,
            )
        if compute_transform_grad:
            if atomic_transform_grad:
                grad_transform_weight = grad_transform_weight_contrib
                grad_transform_bias = grad_transform_bias_contrib
            else:
                with timed_region("path1.primitive.runtime.triton_sequence_scan_backward_reduce"):
                    grad_transform_weight = grad_transform_weight_contrib.sum(dim=0)
                    grad_transform_bias = grad_transform_bias_contrib.sum(dim=0)
        else:
            grad_transform_weight = None
            grad_transform_bias = None
        return (
            grad_update_gate,
            grad_retain_gate,
            grad_angle_cos,
            grad_angle_sin,
            grad_candidate,
            grad_output_gate,
            grad_initial_state,
            grad_transform_weight,
            grad_transform_bias,
            None,
        )


def _gdnp_matrix_multi_read_reference(
    queries: torch.Tensor,
    keys: torch.Tensor,
    value_bases: torch.Tensor,
    vector_states: torch.Tensor,
    alpha_gates: torch.Tensor,
    beta_gates: torch.Tensor,
    aux_query_state_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, head_count, head_dim = queries.shape
    matrix_state = torch.zeros(
        batch_size,
        head_count,
        head_dim,
        head_dim,
        device=queries.device,
        dtype=queries.dtype,
    )
    matrix_reads = []
    aux_matrix_reads = []
    aux_scale = aux_query_state_scale.view(1, head_count, head_dim)
    for position in range(seq_len):
        query = queries[:, position, :, :]
        key = keys[:, position, :, :]
        vector_heads = vector_states[:, position, :, :]
        value = value_bases[:, position, :, :] + vector_heads
        alpha = alpha_gates[:, position, :].view(batch_size, head_count, 1, 1)
        beta = beta_gates[:, position, :].view(batch_size, head_count, 1, 1)
        old_value = torch.einsum("bhvk,bhk->bhv", matrix_state, key)
        erase = torch.einsum("bhv,bhk->bhvk", old_value, key)
        write = torch.einsum("bhv,bhk->bhvk", value, key)
        matrix_state = alpha * (matrix_state - beta * erase) + beta * write
        matrix_reads.append(torch.einsum("bhvk,bhk->bhv", matrix_state, query))
        aux_query = F.normalize(query + vector_heads * aux_scale, p=2.0, dim=-1, eps=1.0e-6)
        aux_matrix_reads.append(torch.einsum("bhvk,bhk->bhv", matrix_state, aux_query))
    return torch.stack(matrix_reads, dim=1), torch.stack(aux_matrix_reads, dim=1)


if triton_runtime_available():  # pragma: no branch

    @triton.jit
    def _gdnp_matrix_multi_read_forward_kernel(
        query_ptr,
        key_ptr,
        value_base_ptr,
        vector_state_ptr,
        alpha_ptr,
        beta_ptr,
        aux_scale_ptr,
        matrix_read_ptr,
        aux_matrix_read_ptr,
        state_history_ptr,
        head_dim,
        SEQ_LEN: tl.constexpr,
        HEAD_COUNT: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        head_index = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_D)
        row_offsets = offsets[:, None]
        col_offsets = offsets[None, :]
        vector_mask = offsets < head_dim
        matrix_mask = (row_offsets < head_dim) & (col_offsets < head_dim)
        state = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
        aux_scale = tl.load(
            aux_scale_ptr + head_index * head_dim + offsets,
            mask=vector_mask,
            other=0.0,
        ).to(tl.float32)

        for position in range(SEQ_LEN):
            step_head_index = (batch_index * SEQ_LEN + position) * HEAD_COUNT + head_index
            vector_base = step_head_index * head_dim
            query = tl.load(query_ptr + vector_base + offsets, mask=vector_mask, other=0.0).to(tl.float32)
            key = tl.load(key_ptr + vector_base + offsets, mask=vector_mask, other=0.0).to(tl.float32)
            value_base = tl.load(value_base_ptr + vector_base + offsets, mask=vector_mask, other=0.0).to(tl.float32)
            vector_state = tl.load(vector_state_ptr + vector_base + offsets, mask=vector_mask, other=0.0).to(tl.float32)
            alpha = tl.load(alpha_ptr + step_head_index).to(tl.float32)
            beta = tl.load(beta_ptr + step_head_index).to(tl.float32)
            value = value_base + vector_state

            old_value = tl.sum(state * key[None, :], axis=1)
            state = alpha * (state - beta * old_value[:, None] * key[None, :]) + beta * value[:, None] * key[None, :]
            state = tl.where(matrix_mask, state, 0.0)
            state_base = step_head_index * head_dim * head_dim
            tl.store(
                state_history_ptr + state_base + row_offsets * head_dim + col_offsets,
                state,
                mask=matrix_mask,
            )

            matrix_read = tl.sum(state * query[None, :], axis=1)
            tl.store(matrix_read_ptr + vector_base + offsets, matrix_read, mask=vector_mask)

            aux_query_raw = query + vector_state * aux_scale
            aux_norm = tl.sqrt(tl.sum(aux_query_raw * aux_query_raw, axis=0))
            aux_query = aux_query_raw / tl.maximum(aux_norm, 1.0e-6)
            aux_matrix_read = tl.sum(state * aux_query[None, :], axis=1)
            tl.store(aux_matrix_read_ptr + vector_base + offsets, aux_matrix_read, mask=vector_mask)


    @triton.jit
    def _gdnp_matrix_multi_read_backward_kernel(
        query_ptr,
        key_ptr,
        value_base_ptr,
        vector_state_ptr,
        alpha_ptr,
        beta_ptr,
        aux_scale_ptr,
        state_history_ptr,
        grad_matrix_read_ptr,
        grad_aux_matrix_read_ptr,
        grad_query_ptr,
        grad_key_ptr,
        grad_value_base_ptr,
        grad_vector_state_ptr,
        grad_alpha_ptr,
        grad_beta_ptr,
        grad_aux_scale_contrib_ptr,
        head_dim,
        SEQ_LEN: tl.constexpr,
        HEAD_COUNT: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        head_index = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_D)
        row_offsets = offsets[:, None]
        col_offsets = offsets[None, :]
        vector_mask = offsets < head_dim
        matrix_mask = (row_offsets < head_dim) & (col_offsets < head_dim)
        grad_next_state = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
        aux_scale = tl.load(
            aux_scale_ptr + head_index * head_dim + offsets,
            mask=vector_mask,
            other=0.0,
        ).to(tl.float32)
        grad_aux_scale = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for reverse_index in range(SEQ_LEN):
            position = SEQ_LEN - 1 - reverse_index
            step_head_index = (batch_index * SEQ_LEN + position) * HEAD_COUNT + head_index
            vector_base = step_head_index * head_dim
            state_base = step_head_index * head_dim * head_dim

            query = tl.load(query_ptr + vector_base + offsets, mask=vector_mask, other=0.0).to(tl.float32)
            key = tl.load(key_ptr + vector_base + offsets, mask=vector_mask, other=0.0).to(tl.float32)
            value_base = tl.load(value_base_ptr + vector_base + offsets, mask=vector_mask, other=0.0).to(tl.float32)
            vector_state = tl.load(vector_state_ptr + vector_base + offsets, mask=vector_mask, other=0.0).to(tl.float32)
            alpha = tl.load(alpha_ptr + step_head_index).to(tl.float32)
            beta = tl.load(beta_ptr + step_head_index).to(tl.float32)
            value = value_base + vector_state
            state = tl.load(
                state_history_ptr + state_base + row_offsets * head_dim + col_offsets,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            if position > 0:
                previous_state_base = ((batch_index * SEQ_LEN + position - 1) * HEAD_COUNT + head_index) * head_dim * head_dim
                previous_state = tl.load(
                    state_history_ptr + previous_state_base + row_offsets * head_dim + col_offsets,
                    mask=matrix_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                previous_state = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)

            grad_matrix_read = tl.load(
                grad_matrix_read_ptr + vector_base + offsets,
                mask=vector_mask,
                other=0.0,
            ).to(tl.float32)
            grad_aux_read = tl.load(
                grad_aux_matrix_read_ptr + vector_base + offsets,
                mask=vector_mask,
                other=0.0,
            ).to(tl.float32)

            grad_state = grad_next_state
            grad_state += grad_matrix_read[:, None] * query[None, :]
            grad_query = tl.sum(state * grad_matrix_read[:, None], axis=0)

            aux_query_raw = query + vector_state * aux_scale
            aux_norm = tl.sqrt(tl.sum(aux_query_raw * aux_query_raw, axis=0))
            safe_aux_norm = tl.maximum(aux_norm, 1.0e-6)
            aux_query = aux_query_raw / safe_aux_norm
            grad_state += grad_aux_read[:, None] * aux_query[None, :]
            grad_aux_query = tl.sum(state * grad_aux_read[:, None], axis=0)
            aux_dot = tl.sum(grad_aux_query * aux_query, axis=0)
            grad_aux_raw = (grad_aux_query - aux_query * aux_dot) / safe_aux_norm
            grad_query += grad_aux_raw
            grad_vector_state = grad_aux_raw * aux_scale
            grad_aux_scale += grad_aux_raw * vector_state

            old_value = tl.sum(previous_state * key[None, :], axis=1)
            erase = old_value[:, None] * key[None, :]
            write = value[:, None] * key[None, :]
            base_state = previous_state - beta * erase

            grad_alpha = tl.sum(grad_state * base_state, axis=0)
            grad_alpha = tl.sum(grad_alpha, axis=0)

            grad_base_state = alpha * grad_state
            grad_beta = tl.sum(grad_state * write, axis=0)
            grad_beta = tl.sum(grad_beta, axis=0)
            grad_beta_erase = tl.sum(grad_base_state * erase, axis=0)
            grad_beta -= tl.sum(grad_beta_erase, axis=0)
            grad_erase = -beta * grad_base_state
            grad_write = beta * grad_state

            grad_value = tl.sum(grad_write * key[None, :], axis=1)
            grad_key = tl.sum(grad_write * value[:, None], axis=0)
            grad_old_value = tl.sum(grad_erase * key[None, :], axis=1)
            grad_key += tl.sum(grad_erase * old_value[:, None], axis=0)
            grad_previous_state = grad_base_state + grad_old_value[:, None] * key[None, :]
            grad_key += tl.sum(previous_state * grad_old_value[:, None], axis=0)
            grad_vector_state += grad_value

            tl.store(grad_query_ptr + vector_base + offsets, grad_query, mask=vector_mask)
            tl.store(grad_key_ptr + vector_base + offsets, grad_key, mask=vector_mask)
            tl.store(grad_value_base_ptr + vector_base + offsets, grad_value, mask=vector_mask)
            tl.store(grad_vector_state_ptr + vector_base + offsets, grad_vector_state, mask=vector_mask)
            tl.store(grad_alpha_ptr + step_head_index, grad_alpha)
            tl.store(grad_beta_ptr + step_head_index, grad_beta)
            grad_next_state = tl.where(matrix_mask, grad_previous_state, 0.0)

        aux_contrib_base = (batch_index * HEAD_COUNT + head_index) * head_dim
        tl.store(grad_aux_scale_contrib_ptr + aux_contrib_base + offsets, grad_aux_scale, mask=vector_mask)


class _GdnpMatrixMultiReadScan(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        queries: torch.Tensor,
        keys: torch.Tensor,
        value_bases: torch.Tensor,
        vector_states: torch.Tensor,
        alpha_gates: torch.Tensor,
        beta_gates: torch.Tensor,
        aux_query_state_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ensure_triton_runtime_available()
        tensors = (
            queries,
            keys,
            value_bases,
            vector_states,
            alpha_gates,
            beta_gates,
            aux_query_state_scale,
        )
        if any(tensor.device.type != "cuda" for tensor in tensors):
            raise RuntimeError("GDN/P20 Triton matrix scan requires CUDA tensors")
        if keys.shape != queries.shape or value_bases.shape != queries.shape or vector_states.shape != queries.shape:
            raise RuntimeError("GDN/P20 Triton matrix scan requires matching [batch, seq, heads, head_dim] tensors")
        batch_size, seq_len, head_count, head_dim = queries.shape
        if alpha_gates.shape != (batch_size, seq_len, head_count):
            raise RuntimeError("GDN/P20 Triton matrix scan requires alpha shape [batch, seq, heads]")
        if beta_gates.shape != (batch_size, seq_len, head_count):
            raise RuntimeError("GDN/P20 Triton matrix scan requires beta shape [batch, seq, heads]")
        if aux_query_state_scale.shape != (head_count, head_dim):
            raise RuntimeError("GDN/P20 Triton matrix scan requires aux scale shape [heads, head_dim]")
        if head_dim > 64:
            raise RuntimeError("GDN/P20 Triton matrix scan currently supports head_dim <= 64")

        queries = queries.contiguous()
        keys = keys.contiguous()
        value_bases = value_bases.contiguous()
        vector_states = vector_states.contiguous()
        alpha_gates = alpha_gates.contiguous()
        beta_gates = beta_gates.contiguous()
        aux_query_state_scale = aux_query_state_scale.contiguous()
        matrix_reads = torch.empty_like(queries)
        aux_matrix_reads = torch.empty_like(queries)
        state_history = torch.empty(
            batch_size,
            seq_len,
            head_count,
            head_dim,
            head_dim,
            device=queries.device,
            dtype=torch.float32,
        )
        block_d = _next_power_of_two(head_dim)
        grid = (batch_size, head_count)
        _gdnp_matrix_multi_read_forward_kernel[grid](
            queries,
            keys,
            value_bases,
            vector_states,
            alpha_gates,
            beta_gates,
            aux_query_state_scale,
            matrix_reads,
            aux_matrix_reads,
            state_history,
            head_dim,
            SEQ_LEN=seq_len,
            HEAD_COUNT=head_count,
            BLOCK_D=block_d,
            num_warps=4,
        )
        ctx.save_for_backward(
            queries,
            keys,
            value_bases,
            vector_states,
            alpha_gates,
            beta_gates,
            aux_query_state_scale,
            state_history,
        )
        return matrix_reads, aux_matrix_reads

    @staticmethod
    def backward(ctx, grad_matrix_reads: torch.Tensor | None, grad_aux_matrix_reads: torch.Tensor | None):  # type: ignore[override]
        (
            queries,
            keys,
            value_bases,
            vector_states,
            alpha_gates,
            beta_gates,
            aux_query_state_scale,
            state_history,
        ) = ctx.saved_tensors
        if grad_matrix_reads is None:
            grad_matrix_reads = torch.zeros_like(queries)
        if grad_aux_matrix_reads is None:
            grad_aux_matrix_reads = torch.zeros_like(queries)

        batch_size, seq_len, head_count, head_dim = queries.shape
        grad_queries = torch.empty_like(queries)
        grad_keys = torch.empty_like(keys)
        grad_value_bases = torch.empty_like(value_bases)
        grad_vector_states = torch.empty_like(vector_states)
        grad_alpha_gates = torch.empty_like(alpha_gates)
        grad_beta_gates = torch.empty_like(beta_gates)
        grad_aux_scale_contrib = torch.empty(
            batch_size,
            head_count,
            head_dim,
            device=queries.device,
            dtype=torch.float32,
        )
        block_d = _next_power_of_two(head_dim)
        grid = (batch_size, head_count)
        _gdnp_matrix_multi_read_backward_kernel[grid](
            queries.contiguous(),
            keys.contiguous(),
            value_bases.contiguous(),
            vector_states.contiguous(),
            alpha_gates.contiguous(),
            beta_gates.contiguous(),
            aux_query_state_scale.contiguous(),
            state_history.contiguous(),
            grad_matrix_reads.contiguous(),
            grad_aux_matrix_reads.contiguous(),
            grad_queries,
            grad_keys,
            grad_value_bases,
            grad_vector_states,
            grad_alpha_gates,
            grad_beta_gates,
            grad_aux_scale_contrib,
            head_dim,
            SEQ_LEN=seq_len,
            HEAD_COUNT=head_count,
            BLOCK_D=block_d,
            num_warps=4,
        )
        grad_aux_scale = grad_aux_scale_contrib.sum(dim=0).to(dtype=aux_query_state_scale.dtype)
        return (
            grad_queries,
            grad_keys,
            grad_value_bases,
            grad_vector_states,
            grad_alpha_gates,
            grad_beta_gates,
            grad_aux_scale,
        )


@dataclass(frozen=True)
class TritonPrimitiveBackend:
    name: str = "triton"

    def fused_p20_update_readout(
        self,
        update_gate: torch.Tensor,
        retain_gate: torch.Tensor,
        transformed_state: torch.Tensor,
        candidate: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _P20FusedUpdateReadout.apply(
            update_gate,
            retain_gate,
            transformed_state,
            candidate,
            output_gate,
        )

    def parcae_state_mix(
        self,
        state: torch.Tensor,
        decay: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        return _ParcaeTritonStateMix.apply(state, decay, injection)

    def parcae_residual_mix(
        self,
        mixed: torch.Tensor,
        block_out: torch.Tensor,
        nonlinear: torch.Tensor,
    ) -> torch.Tensor:
        return _ParcaeTritonResidualMix.apply(mixed, block_out, nonlinear)

    def parcae_loop_update(
        self,
        state: torch.Tensor,
        decay: torch.Tensor,
        injection: torch.Tensor,
        block_out: torch.Tensor,
        nonlinear: torch.Tensor,
    ) -> torch.Tensor:
        return _ParcaeTritonLoopUpdate.apply(state, decay, injection, block_out, nonlinear)

    def parcae_output_mix(
        self,
        anchor: torch.Tensor,
        delta: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        return _ParcaeTritonOutputMix.apply(anchor, delta, gate)

    def gelu(self, inputs: torch.Tensor) -> torch.Tensor:
        return _TritonGelu.apply(inputs)

    def scan_p20_dense_sequence(
        self,
        *,
        update_gate: torch.Tensor,
        retain_gate: torch.Tensor,
        angle_cos: torch.Tensor,
        angle_sin: torch.Tensor,
        candidate: torch.Tensor,
        output_gate: torch.Tensor,
        initial_state: torch.Tensor,
        transform_weight: torch.Tensor,
        transform_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _P20DenseSequenceScan.apply(
            update_gate,
            retain_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
        )

    def scan_p20_block_diagonal_sequence(
        self,
        *,
        update_gate: torch.Tensor,
        retain_gate: torch.Tensor,
        angle_cos: torch.Tensor,
        angle_sin: torch.Tensor,
        candidate: torch.Tensor,
        output_gate: torch.Tensor,
        initial_state: torch.Tensor,
        transform_weight: torch.Tensor,
        transform_bias: torch.Tensor,
        identity_transform: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _P20BlockDiagonalSequenceScan.apply(
            update_gate,
            retain_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
            identity_transform,
        )

    def scan_rotary_state_dense_sequence(
        self,
        *,
        update_gate: torch.Tensor,
        retain_gate: torch.Tensor,
        angle_cos: torch.Tensor,
        angle_sin: torch.Tensor,
        candidate: torch.Tensor,
        initial_state: torch.Tensor,
        transform_weight: torch.Tensor,
        transform_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_gate = torch.ones_like(update_gate)
        return _P20DenseSequenceScan.apply(
            update_gate,
            retain_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
        )

    def scan_rotary_state_block_diagonal_sequence(
        self,
        *,
        update_gate: torch.Tensor,
        retain_gate: torch.Tensor,
        angle_cos: torch.Tensor,
        angle_sin: torch.Tensor,
        candidate: torch.Tensor,
        initial_state: torch.Tensor,
        transform_weight: torch.Tensor,
        transform_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_gate = torch.ones_like(update_gate)
        return _P20BlockDiagonalSequenceScan.apply(
            update_gate,
            retain_gate,
            angle_cos,
            angle_sin,
            candidate,
            output_gate,
            initial_state,
            transform_weight,
            transform_bias,
        )

    def scan_gdnp_matrix_multi_read(
        self,
        *,
        queries: torch.Tensor,
        keys: torch.Tensor,
        value_bases: torch.Tensor,
        vector_states: torch.Tensor,
        alpha_gates: torch.Tensor,
        beta_gates: torch.Tensor,
        aux_query_state_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _GdnpMatrixMultiReadScan.apply(
            queries,
            keys,
            value_bases,
            vector_states,
            alpha_gates,
            beta_gates,
            aux_query_state_scale,
        )


def build_triton_primitive_backend() -> TritonPrimitiveBackend:
    ensure_triton_runtime_available()
    return TritonPrimitiveBackend()
