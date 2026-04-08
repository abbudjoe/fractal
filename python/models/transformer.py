from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class LocalCausalTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_count: int,
        d_ff: int,
        *,
        ffn_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if d_model % head_count != 0:
            raise ValueError(f"local attention requires d_model divisible by head_count, got {d_model} and {head_count}")
        self.d_model = d_model
        self.head_count = head_count
        self.head_dim = d_model // head_count
        self.input_norm = nn.LayerNorm(d_model)
        self.qkv_projection = build_linear(d_model, d_model * 3)
        self.output_projection = build_linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.ffn = ffn_module if ffn_module is not None else PositionWiseFeedForward(d_model, d_ff)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.head_count, self.head_dim).transpose(1, 2)

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.input_norm(hidden)
        q, k, v = self.qkv_projection(normed).chunk(3, dim=-1)
        q_heads = self._reshape_heads(q)
        k_heads = self._reshape_heads(k)
        v_heads = self._reshape_heads(v)

        if attn_mask is None:
            attn_bias = local_causal_attention_bias(
                seq_len=hidden.shape[1],
                local_window=hidden.shape[1],
                device=hidden.device,
                dtype=normed.dtype,
            )
        else:
            attn_bias = attn_mask.to(device=hidden.device, dtype=normed.dtype)
            if attn_bias.ndim == 2:
                attn_bias = attn_bias.view(1, 1, attn_bias.shape[0], attn_bias.shape[1])

        mixed = F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        mixed = mixed.transpose(1, 2).contiguous().view(hidden.shape[0], hidden.shape[1], self.d_model)
        residual = hidden + self.output_projection(mixed)
        return residual + self.ffn(self.output_norm(residual))
