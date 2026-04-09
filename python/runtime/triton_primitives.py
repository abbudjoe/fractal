from __future__ import annotations

from dataclasses import dataclass

import torch

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


if triton_runtime_available():  # pragma: no branch

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


def build_triton_primitive_backend() -> TritonPrimitiveBackend:
    ensure_triton_runtime_available()
    return TritonPrimitiveBackend()
