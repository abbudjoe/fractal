from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from python.models.common import PositionWiseFeedForward


def _sum_except_last(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1, tensor.shape[-1]).sum(dim=0)


class _LayerNormGeluFfnResidual(torch.autograd.Function):
    """Manual-autograd standard FFN residual path.

    This deliberately keeps GEMMs on PyTorch/cuBLAS while owning the operation
    boundary and backward formula. It is not a fused CUDA kernel.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        norm_eps: float,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
    ) -> torch.Tensor:
        mean = residual.mean(dim=-1, keepdim=True)
        centered = residual - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + norm_eps)
        x_hat = centered * rstd
        normed = x_hat * norm_weight.to(dtype=residual.dtype).view(1, 1, -1)
        normed = normed + norm_bias.to(dtype=residual.dtype).view(1, 1, -1)
        preactivation = F.linear(
            normed,
            fc1_weight.to(dtype=normed.dtype),
            fc1_bias.to(dtype=normed.dtype),
        )
        activated = F.gelu(preactivation)
        ffn_output = F.linear(activated, fc2_weight, fc2_bias)
        output = residual + ffn_output
        ctx.save_for_backward(
            x_hat,
            rstd,
            norm_weight,
            normed,
            preactivation,
            activated,
            fc1_weight,
            fc2_weight,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (
            x_hat,
            rstd,
            norm_weight,
            normed,
            preactivation,
            activated,
            fc1_weight,
            fc2_weight,
        ) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        activated_flat = activated.reshape(-1, activated.shape[-1])

        grad_fc2_weight = grad_output_flat.float().T @ activated_flat.float()
        grad_fc2_bias = grad_output_flat.float().sum(dim=0)
        grad_activated = F.linear(grad_output, fc2_weight.to(dtype=grad_output.dtype).T)

        cdf = 0.5 * (1.0 + torch.erf(preactivation.float() / math.sqrt(2.0)))
        pdf = math.sqrt(0.5 / math.pi) * torch.exp(-0.5 * preactivation.float().square())
        grad_preactivation = grad_activated.float() * (cdf + preactivation.float() * pdf)
        grad_preactivation = grad_preactivation.to(dtype=grad_activated.dtype)
        grad_preactivation_flat = grad_preactivation.reshape(-1, grad_preactivation.shape[-1])
        normed_flat = normed.reshape(-1, normed.shape[-1])

        grad_fc1_weight = grad_preactivation_flat.float().T @ normed_flat.float()
        grad_fc1_bias = grad_preactivation_flat.float().sum(dim=0)
        grad_normed = F.linear(
            grad_preactivation,
            fc1_weight.to(dtype=grad_preactivation.dtype).T,
        )

        x_hat_float = x_hat.float()
        grad_normed_float = grad_normed.float()
        norm_weight_float = norm_weight.float().view(1, 1, -1)
        grad_norm_weight = _sum_except_last(grad_normed_float * x_hat_float)
        grad_norm_bias = _sum_except_last(grad_normed_float)

        grad_x_hat = grad_normed_float * norm_weight_float
        width = grad_x_hat.shape[-1]
        sum_grad = grad_x_hat.sum(dim=-1, keepdim=True)
        sum_grad_xhat = (grad_x_hat * x_hat_float).sum(dim=-1, keepdim=True)
        grad_residual_norm = (
            (grad_x_hat * width - sum_grad - x_hat_float * sum_grad_xhat)
            * (rstd.float() / width)
        )
        grad_residual = grad_output.float() + grad_residual_norm
        return (
            grad_residual.to(dtype=grad_output.dtype),
            grad_norm_weight.to(dtype=norm_weight.dtype),
            grad_norm_bias.to(dtype=norm_weight.dtype),
            None,
            grad_fc1_weight.to(dtype=fc1_weight.dtype),
            grad_fc1_bias.to(dtype=fc1_weight.dtype),
            grad_fc2_weight.to(dtype=fc2_weight.dtype),
            grad_fc2_bias.to(dtype=fc2_weight.dtype),
        )


class _LayerNormGeluFfnResidualRecompute(torch.autograd.Function):
    """Activation-lean FFN residual path.

    This owns the whole LayerNorm -> fc1 -> GELU -> fc2 -> residual boundary and
    saves only the residual plus parameters. Backward recomputes the FFN
    activations so the autograd tape does not retain the large intermediate
    preactivation/activation tensors from forward.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        norm_eps: float,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
    ) -> torch.Tensor:
        mean = residual.mean(dim=-1, keepdim=True)
        centered = residual - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + norm_eps)
        normed = centered * rstd
        normed = normed * norm_weight.to(dtype=residual.dtype).view(1, 1, -1)
        normed = normed + norm_bias.to(dtype=residual.dtype).view(1, 1, -1)
        preactivation = F.linear(
            normed,
            fc1_weight.to(dtype=normed.dtype),
            fc1_bias.to(dtype=normed.dtype),
        )
        activated = F.gelu(preactivation)
        ffn_output = F.linear(activated, fc2_weight, fc2_bias)
        output = residual + ffn_output
        ctx.norm_eps = norm_eps
        ctx.save_for_backward(
            residual,
            norm_weight,
            norm_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (
            residual,
            norm_weight,
            norm_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        ) = ctx.saved_tensors
        del fc2_bias

        mean = residual.mean(dim=-1, keepdim=True)
        centered = residual - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + ctx.norm_eps)
        x_hat = centered * rstd
        normed = x_hat * norm_weight.to(dtype=residual.dtype).view(1, 1, -1)
        normed = normed + norm_bias.to(dtype=residual.dtype).view(1, 1, -1)
        preactivation = F.linear(
            normed,
            fc1_weight.to(dtype=normed.dtype),
            fc1_bias.to(dtype=normed.dtype),
        )
        activated = F.gelu(preactivation)

        grad_output = grad_output.contiguous()
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        activated_flat = activated.reshape(-1, activated.shape[-1])

        grad_fc2_weight = grad_output_flat.float().T @ activated_flat.float()
        grad_fc2_bias = grad_output_flat.float().sum(dim=0)
        grad_activated = F.linear(
            grad_output,
            fc2_weight.to(dtype=grad_output.dtype).T,
        )

        cdf = 0.5 * (1.0 + torch.erf(preactivation.float() / math.sqrt(2.0)))
        pdf = math.sqrt(0.5 / math.pi) * torch.exp(-0.5 * preactivation.float().square())
        grad_preactivation = grad_activated.float() * (cdf + preactivation.float() * pdf)
        grad_preactivation = grad_preactivation.to(dtype=grad_activated.dtype)
        grad_preactivation_flat = grad_preactivation.reshape(-1, grad_preactivation.shape[-1])
        normed_flat = normed.reshape(-1, normed.shape[-1])

        grad_fc1_weight = grad_preactivation_flat.float().T @ normed_flat.float()
        grad_fc1_bias = grad_preactivation_flat.float().sum(dim=0)
        grad_normed = F.linear(
            grad_preactivation,
            fc1_weight.to(dtype=grad_preactivation.dtype).T,
        )

        x_hat_float = x_hat.float()
        grad_normed_float = grad_normed.float()
        norm_weight_float = norm_weight.float().view(1, 1, -1)
        grad_norm_weight = _sum_except_last(grad_normed_float * x_hat_float)
        grad_norm_bias = _sum_except_last(grad_normed_float)

        grad_x_hat = grad_normed_float * norm_weight_float
        width = grad_x_hat.shape[-1]
        sum_grad = grad_x_hat.sum(dim=-1, keepdim=True)
        sum_grad_xhat = (grad_x_hat * x_hat_float).sum(dim=-1, keepdim=True)
        grad_residual_norm = (
            (grad_x_hat * width - sum_grad - x_hat_float * sum_grad_xhat)
            * (rstd.float() / width)
        )
        grad_residual = grad_output.float() + grad_residual_norm
        return (
            grad_residual.to(dtype=grad_output.dtype),
            grad_norm_weight.to(dtype=norm_weight.dtype),
            grad_norm_bias.to(dtype=norm_weight.dtype),
            None,
            grad_fc1_weight.to(dtype=fc1_weight.dtype),
            grad_fc1_bias.to(dtype=fc1_weight.dtype),
            grad_fc2_weight.to(dtype=fc2_weight.dtype),
            grad_fc2_bias.to(dtype=fc2_weight.dtype),
        )


def manual_autograd_layernorm_gelu_ffn_residual(
    residual: torch.Tensor,
    output_norm: nn.Module,
    ffn: nn.Module,
) -> torch.Tensor:
    if not isinstance(output_norm, nn.LayerNorm):
        raise RuntimeError("ffn_backend=manual-autograd currently requires nn.LayerNorm output_norm")
    if not output_norm.elementwise_affine or output_norm.weight is None or output_norm.bias is None:
        raise RuntimeError("ffn_backend=manual-autograd requires affine nn.LayerNorm")
    if not isinstance(ffn, PositionWiseFeedForward):
        raise RuntimeError("ffn_backend=manual-autograd currently supports only PositionWiseFeedForward")
    if ffn.fc1.bias is None or ffn.fc2.bias is None:
        raise RuntimeError("ffn_backend=manual-autograd requires biased FFN linear layers")
    return _LayerNormGeluFfnResidual.apply(
        residual,
        output_norm.weight,
        output_norm.bias,
        float(output_norm.eps),
        ffn.fc1.weight,
        ffn.fc1.bias,
        ffn.fc2.weight,
        ffn.fc2.bias,
    )


def recompute_layernorm_gelu_ffn_residual(
    residual: torch.Tensor,
    output_norm: nn.Module,
    ffn: nn.Module,
) -> torch.Tensor:
    if not isinstance(output_norm, nn.LayerNorm):
        raise RuntimeError("ffn_backend=recompute currently requires nn.LayerNorm output_norm")
    if not output_norm.elementwise_affine or output_norm.weight is None or output_norm.bias is None:
        raise RuntimeError("ffn_backend=recompute requires affine nn.LayerNorm")
    if not isinstance(ffn, PositionWiseFeedForward):
        raise RuntimeError("ffn_backend=recompute currently supports only PositionWiseFeedForward")
    if ffn.fc1.bias is None or ffn.fc2.bias is None:
        raise RuntimeError("ffn_backend=recompute requires biased FFN linear layers")
    return _LayerNormGeluFfnResidualRecompute.apply(
        residual,
        output_norm.weight,
        output_norm.bias,
        float(output_norm.eps),
        ffn.fc1.weight,
        ffn.fc1.bias,
        ffn.fc2.weight,
        ffn.fc2.bias,
    )
