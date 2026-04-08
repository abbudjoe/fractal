from __future__ import annotations

import torch
import torch.nn as nn

from python.models.common import PositionWiseFeedForward


def local_causal_mask(seq_len: int, local_window: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for query in range(seq_len):
        earliest_visible = max(0, query - (local_window - 1))
        if earliest_visible > 0:
            mask[query, :earliest_visible] = True
        if query + 1 < seq_len:
            mask[query, query + 1 :] = True
    return mask


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
        self.input_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=head_count,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.ffn = ffn_module if ffn_module is not None else PositionWiseFeedForward(d_model, d_ff)

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.input_norm(hidden)
        mixed, _ = self.attention(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        residual = hidden + mixed
        return residual + self.ffn(self.output_norm(residual))

