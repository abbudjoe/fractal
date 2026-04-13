from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from python.models.common import PositionWiseFeedForward, build_linear


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
    ) -> None:
        super().__init__()
        if d_model % head_count != 0:
            raise ValueError(f"local attention requires d_model divisible by head_count, got {d_model} and {head_count}")
        self.d_model = d_model
        self.head_count = head_count
        self.head_dim = d_model // head_count
        self.qkv_projection = build_linear(d_model, d_model * 3)
        self.output_projection = build_linear(d_model, d_model)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.head_count, self.head_dim).transpose(1, 2)

    def forward(self, normed: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        with record_function("path1.attention.qkv_projection"):
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

        with record_function("path1.attention.sdpa"):
            mixed = F.scaled_dot_product_attention(
                q_heads,
                k_heads,
                v_heads,
                attn_mask=attn_bias,
                dropout_p=0.0,
                is_causal=is_causal,
            )
        mixed = mixed.transpose(1, 2).contiguous().view(normed.shape[0], normed.shape[1], self.d_model)
        with record_function("path1.attention.output_projection"):
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

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        with record_function("path1.attention.input_norm"):
            normed = self.input_norm(hidden)
        residual = hidden + self.attention(normed, attn_mask)
        with record_function("path1.attention.feedforward"):
            return residual + self.ffn(self.output_norm(residual))
