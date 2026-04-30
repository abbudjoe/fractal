from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from python.models.common import PositionWiseFeedForward, ReluSquaredFeedForward, SimpleRmsNorm, build_linear
from python.runtime.cuda_timing import timed_region
from python.runtime.manual_autograd_ffn import (
    manual_autograd_layernorm_gelu_ffn_residual,
    recompute_layernorm_gelu_ffn_residual,
)
from python.runtime.triton_primitives import (
    TritonPrimitiveBackend,
    build_triton_primitive_backend,
    ensure_triton_runtime_available,
)
from python.specs.common import FFN_BACKENDS
from python.specs.path1 import AttentionKernelProfile

try:  # pragma: no cover - availability depends on PyTorch version.
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:  # pragma: no cover - handled at runtime when the kernel is requested.
    create_block_mask = None
    flex_attention = None

try:  # pragma: no cover - optional CUDA package.
    from flash_attn import flash_attn_func
except Exception:  # pragma: no cover - handled at runtime when the kernel is requested.
    flash_attn_func = None


_COMPILED_FLEX_ATTENTION = None


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if flex_attention is None:
        raise RuntimeError("attention_kernel=flex-local requires torch.nn.attention.flex_attention")
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)
    return _COMPILED_FLEX_ATTENTION


def local_causal_attention_bias(
    seq_len: int,
    local_window: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    blocked = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for query in range(seq_len):
        earliest_visible = max(0, query - (local_window - 1))
        if earliest_visible > 0:
            blocked[query, :earliest_visible] = True
        if query + 1 < seq_len:
            blocked[query, query + 1 :] = True

    bias = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype, device=device)
    bias = bias.masked_fill(blocked.view(1, 1, seq_len, seq_len), torch.finfo(dtype).min)
    return bias


class LocalCausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_count: int,
        *,
        local_window: int | None = None,
        attention_kernel: AttentionKernelProfile = AttentionKernelProfile.SDPA,
    ) -> None:
        super().__init__()
        if d_model % head_count != 0:
            raise ValueError(f"local attention requires d_model divisible by head_count, got {d_model} and {head_count}")
        self.d_model = d_model
        self.head_count = head_count
        self.head_dim = d_model // head_count
        if attention_kernel is AttentionKernelProfile.FLEX_LOCAL and not _is_power_of_two(self.head_dim):
            raise ValueError(
                "attention_kernel=flex-local currently requires a power-of-two head_dim "
                f"for PyTorch FlexAttention, got d_model={d_model}, head_count={head_count}, "
                f"head_dim={self.head_dim}"
            )
        self.local_window = local_window
        self.attention_kernel = attention_kernel
        self._flex_block_masks: dict[tuple[str, int, int, int], object] = {}
        self.qkv_projection = build_linear(d_model, d_model * 3)
        self.output_projection = build_linear(d_model, d_model)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.head_count, self.head_dim).transpose(1, 2)

    def _flex_local_block_mask(self, *, seq_len: int, device: torch.device) -> object:
        if create_block_mask is None:
            raise RuntimeError("attention_kernel=flex-local requires torch.nn.attention.flex_attention")
        if self.local_window is None:
            raise RuntimeError("attention_kernel=flex-local requires LocalCausalSelfAttention.local_window")
        device_key = f"{device.type}:{device.index if device.index is not None else 0}"
        block_size = 128
        key = (device_key, seq_len, self.local_window, block_size)
        cached = self._flex_block_masks.get(key)
        if cached is not None:
            return cached

        local_window = self.local_window

        def local_causal_mask(batch, head, query_index, key_index):
            del batch, head
            return (key_index <= query_index) & (key_index >= query_index - (local_window - 1))

        block_mask = create_block_mask(
            local_causal_mask,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
            BLOCK_SIZE=block_size,
        )
        self._flex_block_masks[key] = block_mask
        return block_mask

    def _attention_mix(
        self,
        q_heads: torch.Tensor,
        k_heads: torch.Tensor,
        v_heads: torch.Tensor,
        attn_bias: torch.Tensor | None,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        if (
            self.attention_kernel is AttentionKernelProfile.FLASH_LOCAL
            and attn_bias is not None
        ):
            if flash_attn_func is None:
                raise RuntimeError("attention_kernel=flash-local requires the flash-attn package")
            if q_heads.device.type != "cuda":
                raise RuntimeError("attention_kernel=flash-local currently requires CUDA tensors for local windows")
            if self.local_window is None:
                raise RuntimeError("attention_kernel=flash-local requires LocalCausalSelfAttention.local_window")
            q = q_heads.transpose(1, 2).contiguous()
            k = k_heads.transpose(1, 2).contiguous()
            v = v_heads.transpose(1, 2).contiguous()
            with timed_region("path1.attention.flash_local"):
                mixed = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    causal=True,
                    window_size=(self.local_window - 1, 0),
                )
            return mixed.transpose(1, 2).contiguous()
        if (
            self.attention_kernel is AttentionKernelProfile.FLEX_LOCAL
            and attn_bias is not None
        ):
            if q_heads.device.type != "cuda":
                raise RuntimeError("attention_kernel=flex-local currently requires CUDA tensors for local windows")
            block_mask = self._flex_local_block_mask(seq_len=q_heads.shape[-2], device=q_heads.device)
            with timed_region("path1.attention.flex_local"):
                return _compiled_flex_attention()(
                    q_heads,
                    k_heads,
                    v_heads,
                    block_mask=block_mask,
                )
        with timed_region("path1.attention.sdpa"):
            return F.scaled_dot_product_attention(
                q_heads,
                k_heads,
                v_heads,
                attn_mask=attn_bias,
                dropout_p=0.0,
                is_causal=is_causal,
            )

    def _attention_mix_no_timing(
        self,
        q_heads: torch.Tensor,
        k_heads: torch.Tensor,
        v_heads: torch.Tensor,
        attn_bias: torch.Tensor | None,
        *,
        is_causal: bool,
        flex_block_mask: object | None = None,
        use_compiled_flex: bool = True,
    ) -> torch.Tensor:
        if (
            self.attention_kernel is AttentionKernelProfile.FLASH_LOCAL
            and attn_bias is not None
        ):
            if flash_attn_func is None:
                raise RuntimeError("attention_kernel=flash-local requires the flash-attn package")
            if q_heads.device.type != "cuda":
                raise RuntimeError("attention_kernel=flash-local currently requires CUDA tensors for local windows")
            if self.local_window is None:
                raise RuntimeError("attention_kernel=flash-local requires LocalCausalSelfAttention.local_window")
            q = q_heads.transpose(1, 2).contiguous()
            k = k_heads.transpose(1, 2).contiguous()
            v = v_heads.transpose(1, 2).contiguous()
            mixed = flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                causal=True,
                window_size=(self.local_window - 1, 0),
            )
            return mixed.transpose(1, 2).contiguous()
        if (
            self.attention_kernel is AttentionKernelProfile.FLEX_LOCAL
            and attn_bias is not None
        ):
            if q_heads.device.type != "cuda":
                raise RuntimeError("attention_kernel=flex-local currently requires CUDA tensors for local windows")
            block_mask = flex_block_mask
            if block_mask is None:
                block_mask = self._flex_local_block_mask(seq_len=q_heads.shape[-2], device=q_heads.device)
            flex_attention_fn = _compiled_flex_attention() if use_compiled_flex else flex_attention
            if flex_attention_fn is None:
                raise RuntimeError("attention_kernel=flex-local requires torch.nn.attention.flex_attention")
            return flex_attention_fn(
                q_heads,
                k_heads,
                v_heads,
                block_mask=block_mask,
            )
        return F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=is_causal,
        )

    def forward_no_timing(
        self,
        normed: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        *,
        position_features: torch.Tensor | None = None,
        flex_block_mask: object | None = None,
        use_compiled_flex: bool = True,
    ) -> torch.Tensor:
        if position_features is not None:
            if (
                position_features.ndim != normed.ndim
                or position_features.shape[1:] != normed.shape[1:]
                or position_features.shape[0] not in {1, normed.shape[0]}
            ):
                raise ValueError(
                    "attention position features must match the attention input shape "
                    "except for an optional broadcast batch dimension, "
                    f"got {tuple(position_features.shape)} and {tuple(normed.shape)}"
                )
            normed = normed + position_features.to(device=normed.device, dtype=normed.dtype)
        q, k, v = self.qkv_projection(normed).chunk(3, dim=-1)
        q_heads = self._reshape_heads(q)
        k_heads = self._reshape_heads(k)
        v_heads = self._reshape_heads(v)

        is_causal = attn_mask is None
        if attn_mask is None:
            attn_bias = None
        else:
            attn_bias = attn_mask.to(device=normed.device, dtype=normed.dtype)
            if attn_bias.ndim == 2:
                attn_bias = attn_bias.view(1, 1, attn_bias.shape[0], attn_bias.shape[1])

        mixed = self._attention_mix_no_timing(
            q_heads,
            k_heads,
            v_heads,
            attn_bias,
            is_causal=is_causal,
            flex_block_mask=flex_block_mask,
            use_compiled_flex=use_compiled_flex,
        )
        mixed = mixed.transpose(1, 2).contiguous().view(normed.shape[0], normed.shape[1], self.d_model)
        return self.output_projection(mixed)

    def forward(
        self,
        normed: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        *,
        position_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if position_features is not None:
            if (
                position_features.ndim != normed.ndim
                or position_features.shape[1:] != normed.shape[1:]
                or position_features.shape[0] not in {1, normed.shape[0]}
            ):
                raise ValueError(
                    "attention position features must match the attention input shape "
                    "except for an optional broadcast batch dimension, "
                    f"got {tuple(position_features.shape)} and {tuple(normed.shape)}"
                )
            normed = normed + position_features.to(device=normed.device, dtype=normed.dtype)
        with timed_region("path1.attention.qkv_projection"):
            q, k, v = self.qkv_projection(normed).chunk(3, dim=-1)
        q_heads = self._reshape_heads(q)
        k_heads = self._reshape_heads(k)
        v_heads = self._reshape_heads(v)

        is_causal = attn_mask is None
        if attn_mask is None:
            attn_bias = None
        else:
            attn_bias = attn_mask.to(device=normed.device, dtype=normed.dtype)
            if attn_bias.ndim == 2:
                attn_bias = attn_bias.view(1, 1, attn_bias.shape[0], attn_bias.shape[1])

        mixed = self._attention_mix(q_heads, k_heads, v_heads, attn_bias, is_causal=is_causal)
        mixed = mixed.transpose(1, 2).contiguous().view(normed.shape[0], normed.shape[1], self.d_model)
        with timed_region("path1.attention.output_projection"):
            return self.output_projection(mixed)


class LocalCausalTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_count: int,
        d_ff: int,
        *,
        attention_module: LocalCausalSelfAttention | None = None,
        ffn_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.attention = attention_module if attention_module is not None else LocalCausalSelfAttention(d_model, head_count)
        self.output_norm = nn.LayerNorm(d_model)
        self.ffn = ffn_module if ffn_module is not None else PositionWiseFeedForward(d_model, d_ff)
        self._ffn_backend = "dense"
        self._compiled_ffn_residual_impl = None
        self._compiled_full_block_impl = None
        self._triton_backend: TritonPrimitiveBackend | None = None

    def _ffn_residual_impl(self, residual: torch.Tensor) -> torch.Tensor:
        return residual + self.ffn(self.output_norm(residual))

    def _triton_gelu_ffn_residual_impl(self, residual: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.ffn, PositionWiseFeedForward):
            raise RuntimeError("ffn_backend=triton-gelu currently supports only PositionWiseFeedForward")
        if self._triton_backend is None:
            raise RuntimeError("ffn_backend=triton-gelu requires a configured Triton backend")
        normed = self.output_norm(residual)
        hidden = self.ffn.fc1(normed)
        hidden = self._triton_backend.gelu(hidden)
        return residual + self.ffn.fc2(hidden)

    def _full_block_impl_no_timing(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor | None,
        position_features: torch.Tensor | None,
        flex_block_mask: object | None,
    ) -> torch.Tensor:
        normed = self.input_norm(hidden)
        residual = hidden + self.attention.forward_no_timing(
            normed,
            attn_mask,
            position_features=position_features,
            flex_block_mask=flex_block_mask,
            use_compiled_flex=False,
        )
        return self._ffn_residual_impl(residual)

    def _full_block_flex_block_mask(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> object | None:
        if (
            self.attention.attention_kernel is AttentionKernelProfile.FLEX_LOCAL
            and attn_mask is not None
        ):
            return self.attention._flex_local_block_mask(seq_len=hidden.shape[1], device=hidden.device)
        return None

    def forward(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        *,
        position_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._compiled_full_block_impl is not None:
            flex_block_mask = self._full_block_flex_block_mask(hidden, attn_mask)
            with timed_region("path1.attention.full_block_compiled"):
                return self._compiled_full_block_impl(hidden, attn_mask, position_features, flex_block_mask)
        with timed_region("path1.attention.input_norm"):
            normed = self.input_norm(hidden)
        residual = hidden + self.attention(normed, attn_mask, position_features=position_features)
        with timed_region("path1.attention.feedforward"):
            if self._ffn_backend == "compiled":
                compiled_ffn = self._compiled_ffn_residual_impl
                if compiled_ffn is None:
                    if not hasattr(torch, "compile"):
                        raise RuntimeError("ffn_backend=compiled requires torch.compile")
                    compiled_ffn = torch.compile(self._ffn_residual_impl, mode="reduce-overhead")
                    self._compiled_ffn_residual_impl = compiled_ffn
                with timed_region("path1.attention.feedforward_compiled"):
                    return compiled_ffn(residual)
            if self._ffn_backend == "manual-autograd":
                with timed_region("path1.attention.feedforward_manual_autograd"):
                    return manual_autograd_layernorm_gelu_ffn_residual(residual, self.output_norm, self.ffn)
            if self._ffn_backend == "recompute":
                with timed_region("path1.attention.feedforward_recompute"):
                    return recompute_layernorm_gelu_ffn_residual(residual, self.output_norm, self.ffn)
            if self._ffn_backend == "triton-gelu":
                with timed_region("path1.attention.feedforward_triton_gelu"):
                    return self._triton_gelu_ffn_residual_impl(residual)
            return self._ffn_residual_impl(residual)

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        ffn_backend: str = "dense",
    ) -> None:
        del compile_mode
        if ffn_backend not in FFN_BACKENDS:
            raise ValueError(f"unsupported ffn_backend: {ffn_backend}")
        self._ffn_backend = ffn_backend
        self._compiled_ffn_residual_impl = None
        self._compiled_full_block_impl = None
        self._triton_backend = None
        if ffn_backend == "compiled":
            if not hasattr(torch, "compile"):
                raise RuntimeError("ffn_backend=compiled requires torch.compile")
            self._compiled_ffn_residual_impl = torch.compile(
                self._ffn_residual_impl,
                mode="reduce-overhead",
            )
        if ffn_backend == "triton-gelu":
            ensure_triton_runtime_available()
            self._triton_backend = build_triton_primitive_backend()

    def configure_full_block_compile(
        self,
        *,
        enabled: bool = True,
        compile_mode: str = "reduce-overhead",
    ) -> None:
        if not enabled:
            self._compiled_full_block_impl = None
            return
        if not hasattr(torch, "compile"):
            raise RuntimeError("full-block compile requires torch.compile")
        self._compiled_full_block_impl = torch.compile(
            self._full_block_impl_no_timing,
            mode=compile_mode,
        )


class Pr5LocalCausalTransformerBlock(nn.Module):
    """PR5-style exact-mixing seam with residual anchor access."""

    def __init__(
        self,
        d_model: int,
        head_count: int,
        d_ff: int,
        *,
        attention_module: LocalCausalSelfAttention | None = None,
    ) -> None:
        super().__init__()
        self.input_norm = SimpleRmsNorm(d_model)
        self.attention = attention_module if attention_module is not None else LocalCausalSelfAttention(d_model, head_count)
        self.output_norm = SimpleRmsNorm(d_model)
        self.ffn = ReluSquaredFeedForward(d_model, d_ff)
        self.attention_scale = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.ffn_scale = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.residual_mix = nn.Parameter(torch.stack((torch.ones(d_model), torch.zeros(d_model))).float())
        self._ffn_backend = "dense"
        self._compiled_ffn_residual_impl = None

    def _ffn_residual_impl(self, residual: torch.Tensor, hidden_dtype: torch.dtype) -> torch.Tensor:
        return residual + self.ffn_scale.to(dtype=hidden_dtype).view(1, 1, -1) * self.ffn(
            self.output_norm(residual)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        residual_anchor: torch.Tensor | None = None,
        *,
        position_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual_anchor is None:
            residual_anchor = hidden
        mix = self.residual_mix.to(dtype=hidden.dtype)
        mixed_hidden = mix[0].view(1, 1, -1) * hidden + mix[1].view(1, 1, -1) * residual_anchor
        with timed_region("path1.pr5_attention.input_norm"):
            normed = self.input_norm(mixed_hidden)
        attention_out = self.attention(normed, attn_mask, position_features=position_features)
        residual = mixed_hidden + self.attention_scale.to(dtype=hidden.dtype).view(1, 1, -1) * attention_out
        with timed_region("path1.pr5_attention.feedforward"):
            if self._ffn_backend == "compiled":
                compiled_ffn = self._compiled_ffn_residual_impl
                if compiled_ffn is None:
                    if not hasattr(torch, "compile"):
                        raise RuntimeError("ffn_backend=compiled requires torch.compile")
                    compiled_ffn = torch.compile(self._ffn_residual_impl, mode="reduce-overhead")
                    self._compiled_ffn_residual_impl = compiled_ffn
                with timed_region("path1.pr5_attention.feedforward_compiled"):
                    return compiled_ffn(residual, hidden.dtype)
            if self._ffn_backend == "manual-autograd":
                raise RuntimeError(
                    "ffn_backend=manual-autograd requested, but no manual-autograd FFN path is "
                    "registered for Pr5LocalCausalTransformerBlock"
                )
            return self._ffn_residual_impl(residual, hidden.dtype)

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        ffn_backend: str = "dense",
    ) -> None:
        del compile_mode
        if ffn_backend not in FFN_BACKENDS:
            raise ValueError(f"unsupported ffn_backend: {ffn_backend}")
        self._ffn_backend = ffn_backend
        self._compiled_ffn_residual_impl = None
        if ffn_backend == "compiled":
            if not hasattr(torch, "compile"):
                raise RuntimeError("ffn_backend=compiled requires torch.compile")
            self._compiled_ffn_residual_impl = torch.compile(
                self._ffn_residual_impl,
                mode="reduce-overhead",
            )
