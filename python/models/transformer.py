from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from python.models.primitives import P20RotaryStateOutputRuntimeSequenceMixer
from python.models.common import (
    PositionWiseFeedForward,
    ReluSquaredFeedForward,
    SimpleRmsNorm,
    build_linear,
)
from python.specs.runtime import PrimitiveStateTransformMode


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
    bias = bias.masked_fill(
        blocked.view(1, 1, seq_len, seq_len), torch.finfo(dtype).min
    )
    return bias


def _explicit_attention_bias(
    attn_mask: torch.Tensor | None,
    *,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if attn_mask is None:
        return local_causal_attention_bias(seq_len, seq_len, device, dtype)
    attn_bias = attn_mask.to(device=device, dtype=dtype)
    if attn_bias.ndim == 2:
        return attn_bias.view(1, 1, attn_bias.shape[0], attn_bias.shape[1])
    return attn_bias


def causal_topk_token_positions(
    router_scores: torch.Tensor, route_fraction: float
) -> list[torch.Tensor]:
    """Select tokens with a prefix-top-k rule so routing stays autoregressive."""

    if router_scores.ndim != 2:
        raise ValueError(
            f"router_scores must have shape [batch, seq], got {tuple(router_scores.shape)}"
        )
    if not 0.0 < route_fraction <= 1.0:
        raise ValueError(f"route_fraction must be in (0, 1], got {route_fraction}")
    batch_size, seq_len = router_scores.shape
    selected_by_batch: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        batch_scores = router_scores[batch_index]
        positions: list[int] = []
        for position in range(seq_len):
            prefix_count = position + 1
            capacity = max(
                1, min(prefix_count, math.ceil(prefix_count * route_fraction))
            )
            current_score = batch_scores[position]
            prefix = batch_scores[: position + 1]
            earlier = prefix[:-1]
            rank = (earlier > current_score).sum() + (earlier == current_score).sum()
            if int(rank.detach().item()) < capacity:
                positions.append(position)
        if not positions:
            positions.append(0)
        selected_by_batch.append(
            torch.tensor(positions, dtype=torch.long, device=router_scores.device)
        )
    return selected_by_batch


def full_sequence_topc_token_positions(
    router_scores: torch.Tensor, route_fraction: float
) -> list[torch.Tensor]:
    """Select a fixed full-sequence top-C capacity per batch item.

    This is the training-time MoD primitive. It is intentionally non-causal:
    token selection may depend on future token router scores.
    """

    if router_scores.ndim != 2:
        raise ValueError(
            f"router_scores must have shape [batch, seq], got {tuple(router_scores.shape)}"
        )
    if not 0.0 < route_fraction <= 1.0:
        raise ValueError(f"route_fraction must be in (0, 1], got {route_fraction}")
    batch_size, seq_len = router_scores.shape
    capacity = max(1, min(seq_len, math.ceil(seq_len * route_fraction)))
    selected_by_batch: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        positions = router_scores[batch_index].topk(capacity, dim=0).indices
        selected_by_batch.append(positions.sort().values)
    return selected_by_batch


def selected_positions_to_mask(
    selected_positions: list[torch.Tensor], router_scores: torch.Tensor
) -> torch.Tensor:
    """Materialize selected token positions as a boolean [batch, seq] mask."""

    if router_scores.ndim != 2:
        raise ValueError(
            f"router_scores must have shape [batch, seq], got {tuple(router_scores.shape)}"
        )
    if len(selected_positions) != router_scores.shape[0]:
        raise ValueError(
            "selected_positions length must match router_scores batch dimension"
        )
    mask = torch.zeros_like(router_scores, dtype=torch.bool)
    for batch_index, positions in enumerate(selected_positions):
        mask[batch_index].index_fill_(0, positions, True)
    return mask


class LocalCausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_count: int,
    ) -> None:
        super().__init__()
        if d_model % head_count != 0:
            raise ValueError(
                f"local attention requires d_model divisible by head_count, got {d_model} and {head_count}"
            )
        self.d_model = d_model
        self.head_count = head_count
        self.head_dim = d_model // head_count
        self.qkv_projection = build_linear(d_model, d_model * 3)
        self.output_projection = build_linear(d_model, d_model)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(
            batch_size, seq_len, self.head_count, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        normed: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        depth_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if depth_memory is not None:
            raise ValueError("standard local attention does not consume depth memory")
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
        mixed = (
            mixed.transpose(1, 2)
            .contiguous()
            .view(normed.shape[0], normed.shape[1], self.d_model)
        )
        with record_function("path1.attention.output_projection"):
            return self.output_projection(mixed)

    def forward_selected(
        self,
        normed: torch.Tensor,
        selected_positions: list[torch.Tensor],
        attn_mask: torch.Tensor | None = None,
        depth_memory: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if depth_memory is not None:
            raise ValueError("selected local attention does not consume depth memory")
        batch_size, seq_len, _ = normed.shape
        if len(selected_positions) != batch_size:
            raise ValueError(
                "selected_positions must provide one index tensor per batch item"
            )
        with record_function("path1.attention.selected_qkv_projection"):
            q, k, v = self.qkv_projection(normed).chunk(3, dim=-1)
        q_heads = self._reshape_heads(q)
        k_heads = self._reshape_heads(k)
        v_heads = self._reshape_heads(v)
        attn_bias = _explicit_attention_bias(
            attn_mask,
            seq_len=seq_len,
            device=normed.device,
            dtype=normed.dtype,
        )
        outputs: list[torch.Tensor] = []
        for batch_index, positions in enumerate(selected_positions):
            if positions.ndim != 1:
                raise ValueError("selected position tensors must be one-dimensional")
            if positions.numel() == 0:
                raise ValueError("selected position tensors must not be empty")
            if torch.any(positions < 0) or torch.any(positions >= seq_len):
                raise ValueError(
                    f"selected positions must be in [0, {seq_len}), got {positions}"
                )
            q_selected = q_heads[batch_index : batch_index + 1].index_select(
                2, positions
            )
            k_batch = k_heads[batch_index : batch_index + 1]
            v_batch = v_heads[batch_index : batch_index + 1]
            bias_batch = (
                attn_bias
                if attn_bias.shape[0] == 1
                else attn_bias[batch_index : batch_index + 1]
            )
            bias_selected = bias_batch.index_select(2, positions)
            with record_function("path1.attention.selected_sdpa"):
                mixed = F.scaled_dot_product_attention(
                    q_selected,
                    k_batch,
                    v_batch,
                    attn_mask=bias_selected,
                    dropout_p=0.0,
                    is_causal=False,
                )
            mixed = (
                mixed.transpose(1, 2)
                .contiguous()
                .view(1, positions.numel(), self.d_model)
            )
            with record_function("path1.attention.selected_output_projection"):
                outputs.append(self.output_projection(mixed).squeeze(0))
        return outputs


class DepthAugmentedCausalSelfAttention(LocalCausalSelfAttention):
    """MoDA-style depth-memory attention approximation.

    This preserves the existing causal sequence attention path and, when
    provided, appends KV pairs projected from prior layer hidden states. It is a
    PyTorch approximation of MoDA's depth stream, not the paper's Flash-compatible
    depth-KV layout.
    """

    def __init__(
        self,
        d_model: int,
        head_count: int,
    ) -> None:
        super().__init__(d_model, head_count)
        self.depth_norm = nn.LayerNorm(d_model)
        self.depth_kv_projection = build_linear(d_model, d_model * 2)
        self.depth_logit_bias = nn.Parameter(
            torch.tensor(-2.1972246, dtype=torch.float32)
        )
        self._last_depth_layers: int = 0
        self._last_depth_tokens: int = 0
        self._last_sequence_attention_mass: float = 1.0
        self._last_depth_attention_mass: float = 0.0

    def forward(
        self,
        normed: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        depth_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if depth_memory is None:
            self._last_depth_layers = 0
            self._last_depth_tokens = 0
            self._last_sequence_attention_mass = 1.0
            self._last_depth_attention_mass = 0.0
            return super().forward(normed, attn_mask)
        if depth_memory.ndim != 4:
            raise ValueError(
                f"depth memory must have shape [batch, depth, seq, d_model], got {tuple(depth_memory.shape)}"
            )
        if depth_memory.shape[1] == 0:
            self._last_depth_layers = 0
            self._last_depth_tokens = 0
            self._last_sequence_attention_mass = 1.0
            self._last_depth_attention_mass = 0.0
            return super().forward(normed, attn_mask)
        batch_size, memory_layers, memory_seq_len, memory_width = depth_memory.shape
        if (
            batch_size != normed.shape[0]
            or memory_seq_len != normed.shape[1]
            or memory_width != self.d_model
        ):
            raise ValueError(
                "depth memory shape must match attention input batch/sequence/width: "
                f"memory={tuple(depth_memory.shape)}, input={tuple(normed.shape)}"
            )

        with record_function("path1.depth_attention.qkv_projection"):
            q, k, v = self.qkv_projection(normed).chunk(3, dim=-1)
        q_heads = self._reshape_heads(q)
        k_heads = self._reshape_heads(k)
        v_heads = self._reshape_heads(v)

        with record_function("path1.depth_attention.depth_kv_projection"):
            normed_depth = self.depth_norm(depth_memory)
            depth_k, depth_v = self.depth_kv_projection(normed_depth).chunk(2, dim=-1)
        depth_k_heads = (
            depth_k.view(
                batch_size,
                memory_layers,
                memory_seq_len,
                self.head_count,
                self.head_dim,
            )
            .permute(0, 3, 1, 2, 4)
            .reshape(
                batch_size,
                self.head_count,
                memory_layers * memory_seq_len,
                self.head_dim,
            )
        )
        depth_v_heads = (
            depth_v.view(
                batch_size,
                memory_layers,
                memory_seq_len,
                self.head_count,
                self.head_dim,
            )
            .permute(0, 3, 1, 2, 4)
            .reshape(
                batch_size,
                self.head_count,
                memory_layers * memory_seq_len,
                self.head_dim,
            )
        )

        current_bias = _explicit_attention_bias(
            attn_mask,
            seq_len=normed.shape[1],
            device=normed.device,
            dtype=normed.dtype,
        )
        depth_bias = current_bias.repeat(1, 1, 1, memory_layers)
        depth_bias = depth_bias + self.depth_logit_bias.to(dtype=normed.dtype).view(
            1, 1, 1, 1
        )
        combined_bias = torch.cat((current_bias, depth_bias), dim=-1)
        k_total = torch.cat((k_heads, depth_k_heads), dim=-2)
        v_total = torch.cat((v_heads, depth_v_heads), dim=-2)

        with torch.no_grad():
            logits = torch.matmul(
                q_heads.float(), k_total.float().transpose(-1, -2)
            ) * (self.head_dim**-0.5)
            logits = logits + combined_bias.float()
            probs = torch.softmax(logits, dim=-1)
            sequence_mass = probs[..., : normed.shape[1]].sum(dim=-1).mean()
            depth_mass = probs[..., normed.shape[1] :].sum(dim=-1).mean()
            self._last_sequence_attention_mass = float(sequence_mass.item())
            self._last_depth_attention_mass = float(depth_mass.item())

        with record_function("path1.depth_attention.sdpa"):
            mixed = F.scaled_dot_product_attention(
                q_heads,
                k_total,
                v_total,
                attn_mask=combined_bias,
                dropout_p=0.0,
                is_causal=False,
            )
        self._last_depth_layers = int(memory_layers)
        self._last_depth_tokens = int(memory_layers * memory_seq_len)
        mixed = (
            mixed.transpose(1, 2)
            .contiguous()
            .view(normed.shape[0], normed.shape[1], self.d_model)
        )
        with record_function("path1.depth_attention.output_projection"):
            return self.output_projection(mixed)

    def diagnostic_payload(self) -> dict[str, object]:
        return {
            "kind": "moda-depth-kv-approx",
            "last_depth_layers": self._last_depth_layers,
            "last_depth_tokens": self._last_depth_tokens,
            "depth_logit_bias": float(self.depth_logit_bias.detach().float().item()),
            "last_sequence_attention_mass": self._last_sequence_attention_mass,
            "last_depth_attention_mass": self._last_depth_attention_mass,
            "joint_softmax": True,
            "same_token_depth_memory": False,
            "depth_memory_scope": "prior hidden states flattened across prior layers and causal token positions",
            "label": "Approximate MoDA-style depth KV over prior hidden states; not Flash MoDA",
        }


class PaperMoDACausalSelfAttention(LocalCausalSelfAttention):
    """Slow paper-faithful MoDA reference attention.

    Each query attends over current sequence KV plus prior-depth KV at the same
    token position. Sequence and depth logits share one softmax.
    """

    def __init__(self, d_model: int, head_count: int) -> None:
        super().__init__(d_model, head_count)
        self._last_sequence_kv: tuple[torch.Tensor, torch.Tensor] | None = None
        self._last_depth_slots: int = 0
        self._last_sequence_attention_mass: float = 1.0
        self._last_depth_attention_mass: float = 0.0

    def _validate_depth_memory(
        self,
        depth_memory: Any,
        *,
        batch_size: int,
        seq_len: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if depth_memory is None:
            return []
        if not isinstance(depth_memory, (list, tuple)):
            raise ValueError(
                "paper MoDA depth memory must be a list of prior sequence KV tuples"
            )
        slots: list[tuple[torch.Tensor, torch.Tensor]] = []
        expected = (batch_size, self.head_count, seq_len, self.head_dim)
        for slot in depth_memory:
            if not isinstance(slot, tuple) or len(slot) != 2:
                raise ValueError("paper MoDA depth memory slots must be (k, v) tuples")
            key, value = slot
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError(
                    "paper MoDA depth KV must match current batch/head/token/head_dim: "
                    f"expected={expected}, key={tuple(key.shape)}, value={tuple(value.shape)}"
                )
            slots.append((key, value))
        return slots

    def forward(
        self,
        normed: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        depth_memory: Any = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = normed.shape
        with record_function("path1.paper_moda.qkv_projection"):
            q, k, v = self.qkv_projection(normed).chunk(3, dim=-1)
        q_heads = self._reshape_heads(q)
        k_heads = self._reshape_heads(k)
        v_heads = self._reshape_heads(v)
        self._last_sequence_kv = (k_heads, v_heads)

        current_bias = _explicit_attention_bias(
            attn_mask,
            seq_len=seq_len,
            device=normed.device,
            dtype=normed.dtype,
        )
        sequence_logits = (
            torch.matmul(q_heads, k_heads.transpose(-1, -2)) * (self.head_dim**-0.5)
        ) + current_bias

        slots = self._validate_depth_memory(
            depth_memory, batch_size=batch_size, seq_len=seq_len
        )
        if slots:
            depth_k = torch.stack([key for key, _ in slots], dim=1)
            depth_v = torch.stack([value for _, value in slots], dim=1)
            depth_logits = torch.einsum("bhtd,blhtd->bhtl", q_heads, depth_k) * (
                self.head_dim**-0.5
            )
            logits = torch.cat((sequence_logits, depth_logits), dim=-1)
            probs = torch.softmax(logits, dim=-1)
            sequence_probs = probs[..., :seq_len]
            depth_probs = probs[..., seq_len:]
            mixed = torch.matmul(sequence_probs, v_heads) + torch.einsum(
                "bhtl,blhtd->bhtd", depth_probs, depth_v
            )
            self._last_depth_slots = len(slots)
            self._last_sequence_attention_mass = float(
                sequence_probs.detach().float().sum(dim=-1).mean().item()
            )
            self._last_depth_attention_mass = float(
                depth_probs.detach().float().sum(dim=-1).mean().item()
            )
        else:
            probs = torch.softmax(sequence_logits, dim=-1)
            mixed = torch.matmul(probs, v_heads)
            self._last_depth_slots = 0
            self._last_sequence_attention_mass = 1.0
            self._last_depth_attention_mass = 0.0

        mixed = (
            mixed.transpose(1, 2)
            .contiguous()
            .view(normed.shape[0], normed.shape[1], self.d_model)
        )
        with record_function("path1.paper_moda.output_projection"):
            return self.output_projection(mixed)

    def diagnostic_payload(self) -> dict[str, object]:
        return {
            "kind": "paper-moda-depth-kv-reference",
            "last_depth_slots": self._last_depth_slots,
            "last_sequence_attention_mass": self._last_sequence_attention_mass,
            "last_depth_attention_mass": self._last_depth_attention_mass,
            "joint_softmax": True,
            "same_token_depth_memory": True,
            "depth_memory_scope": "same-token prior-depth sequence KV",
            "ffn_depth_slots": 0,
            "label": "Slow paper-faithful MoDA reference: current sequence KV plus same-token prior-depth KV",
        }


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
        self.attention = (
            attention_module
            if attention_module is not None
            else LocalCausalSelfAttention(d_model, head_count)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.ffn = (
            ffn_module
            if ffn_module is not None
            else PositionWiseFeedForward(d_model, d_ff)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        depth_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with record_function("path1.attention.input_norm"):
            normed = self.input_norm(hidden)
        residual = hidden + self.attention(normed, attn_mask, depth_memory)
        with record_function("path1.attention.feedforward"):
            return residual + self.ffn(self.output_norm(residual))

    def forward_selected(
        self,
        hidden: torch.Tensor,
        selected_positions: list[torch.Tensor],
        attn_mask: torch.Tensor | None = None,
        depth_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if depth_memory is not None:
            raise ValueError(
                "selected transformer block execution does not consume depth memory"
            )
        with record_function("path1.attention.selected_input_norm"):
            normed = self.input_norm(hidden)
        attention_outputs = self.attention.forward_selected(
            normed, selected_positions, attn_mask
        )
        batch_outputs: list[torch.Tensor] = []
        for batch_index, positions in enumerate(selected_positions):
            hidden_selected = hidden[batch_index].index_select(0, positions)
            residual = hidden_selected + attention_outputs[batch_index]
            with record_function("path1.attention.selected_feedforward"):
                selected_output = residual + self.ffn(
                    self.output_norm(residual.unsqueeze(0))
                ).squeeze(0)
            batch_outputs.append(
                hidden[batch_index].clone().index_copy(0, positions, selected_output)
            )
        return torch.stack(batch_outputs, dim=0)


class TokenRoutedLocalCausalTransformerBlock(LocalCausalTransformerBlock):
    """MoD-style causal token-routed block approximation.

    The selected tokens run attention/FFN updates while skipped tokens carry the
    incoming hidden state through unchanged. Routing uses a causal prefix-top-k
    rule rather than full-sequence top-k, preserving autoregressive behavior.
    """

    def __init__(
        self,
        d_model: int,
        head_count: int,
        d_ff: int,
        *,
        route_fraction: float,
        attention_module: LocalCausalSelfAttention | None = None,
        ffn_module: nn.Module | None = None,
    ) -> None:
        if not 0.0 < route_fraction <= 1.0:
            raise ValueError(f"route_fraction must be in (0, 1], got {route_fraction}")
        super().__init__(
            d_model,
            head_count,
            d_ff,
            attention_module=attention_module,
            ffn_module=ffn_module,
        )
        self.route_fraction = route_fraction
        self.router_norm = nn.LayerNorm(d_model)
        self.router = nn.Linear(d_model, 1)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router.bias)
        self._last_selected_fraction = 0.0
        self._last_selected_count = 0
        self._last_token_count = 0
        self._last_skipped_count = 0
        self._last_selected_attention_token_count = 0
        self._last_mean_router_score = 0.0
        self._last_mean_selected_router_score = 0.0
        self._last_mean_selected_gate = 0.0
        self._last_selected_first_half_fraction = 0.0
        self._last_selected_second_half_fraction = 0.0
        self._last_mean_selected_position = 0.0
        self._last_router_scores: torch.Tensor | None = None
        self._last_selected_mask: torch.Tensor | None = None

    def _route(self, hidden: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        router_scores = self.router(self.router_norm(hidden)).squeeze(-1)
        selected_positions = causal_topk_token_positions(
            router_scores, self.route_fraction
        )
        selected_mask = selected_positions_to_mask(selected_positions, router_scores)
        selected_scores = torch.cat(
            [
                router_scores[batch_index].index_select(0, positions)
                for batch_index, positions in enumerate(selected_positions)
            ]
        )
        self._last_selected_count = int(selected_scores.numel())
        self._last_token_count = int(router_scores.numel())
        self._last_skipped_count = self._last_token_count - self._last_selected_count
        self._last_selected_attention_token_count = self._last_selected_count
        self._last_selected_fraction = (
            self._last_selected_count / self._last_token_count
            if self._last_token_count
            else 0.0
        )
        self._last_mean_router_score = float(
            router_scores.detach().float().mean().item()
        )
        self._last_mean_selected_router_score = float(
            selected_scores.detach().float().mean().item()
        )
        self._last_mean_selected_gate = float(
            torch.sigmoid(selected_scores.detach().float()).mean().item()
        )
        seq_len = router_scores.shape[1]
        midpoint = max(1, seq_len // 2)
        selected_mask_float = selected_mask.detach().float()
        self._last_selected_first_half_fraction = float(
            selected_mask_float[:, :midpoint].mean().item()
        )
        self._last_selected_second_half_fraction = (
            float(selected_mask_float[:, midpoint:].mean().item())
            if midpoint < seq_len
            else 0.0
        )
        selected_position_values = torch.cat(
            [positions.detach().float() for positions in selected_positions]
        )
        self._last_mean_selected_position = float(
            (selected_position_values / max(1, seq_len - 1)).mean().item()
        )
        self._last_router_scores = router_scores
        self._last_selected_mask = selected_mask
        return selected_positions, router_scores

    def last_routing_tensors(self) -> dict[str, torch.Tensor]:
        if self._last_router_scores is None or self._last_selected_mask is None:
            raise RuntimeError("token routing tensors are unavailable before forward")
        return {
            "router_scores": self._last_router_scores,
            "selected_mask": self._last_selected_mask,
        }

    def forward(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        depth_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if depth_memory is not None:
            raise ValueError(
                "token-routed transformer blocks do not consume depth memory"
            )
        selected_positions, router_scores = self._route(hidden)
        selected_output = self.forward_selected(hidden, selected_positions, attn_mask)
        batch_outputs: list[torch.Tensor] = []
        for batch_index, positions in enumerate(selected_positions):
            selected_hidden = hidden[batch_index].index_select(0, positions)
            selected_updated = selected_output[batch_index].index_select(0, positions)
            gates = torch.sigmoid(
                router_scores[batch_index].index_select(0, positions)
            ).to(dtype=hidden.dtype)
            mixed = selected_hidden + gates.unsqueeze(-1) * (
                selected_updated - selected_hidden
            )
            batch_outputs.append(
                hidden[batch_index].clone().index_copy(0, positions, mixed)
            )
        output = torch.stack(batch_outputs, dim=0)
        if not torch.isfinite(output).all():
            raise FloatingPointError(
                "token-routed transformer block produced non-finite output"
            )
        return output

    def token_routing_payload(self) -> dict[str, object]:
        return {
            "kind": "causal-topk-block",
            "route_fraction": self.route_fraction,
            "last_selected_fraction": self._last_selected_fraction,
            "last_selected_count": self._last_selected_count,
            "last_skipped_count": self._last_skipped_count,
            "last_token_count": self._last_token_count,
            "last_mean_router_score": self._last_mean_router_score,
            "last_mean_selected_router_score": self._last_mean_selected_router_score,
            "last_mean_selected_gate": self._last_mean_selected_gate,
            "last_selected_first_half_fraction": self._last_selected_first_half_fraction,
            "last_selected_second_half_fraction": self._last_selected_second_half_fraction,
            "last_mean_selected_position": self._last_mean_selected_position,
            "selected_attention_scope": "selected-query/full-causal-key-value",
            "last_selected_attention_token_count": self._last_selected_attention_token_count,
            "causal_decode_safe": True,
            "label": "Causal prefix-top-k token block routing; approximates MoD and preserves autoregressive visibility",
        }


class PaperMoDTrainTopCTransformerBlock(TokenRoutedLocalCausalTransformerBlock):
    """Training-time MoD top-C block reference.

    This follows the paper primitive more closely than the causal decode
    approximation: full-sequence top-C routing chooses a fixed capacity, selected
    tokens run a selected-only transformer block, and skipped tokens are identity.
    """

    def __init__(
        self,
        d_model: int,
        head_count: int,
        d_ff: int,
        *,
        route_fraction: float,
        attention_module: LocalCausalSelfAttention | None = None,
        ffn_module: nn.Module | None = None,
    ) -> None:
        super().__init__(
            d_model,
            head_count,
            d_ff,
            route_fraction=route_fraction,
            attention_module=attention_module,
            ffn_module=ffn_module,
        )
        self._last_capacity = 0
        self._last_selected_attention_token_count = 0

    def _route(self, hidden: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        router_scores = self.router(self.router_norm(hidden)).squeeze(-1)
        selected_positions = full_sequence_topc_token_positions(
            router_scores, self.route_fraction
        )
        selected_mask = selected_positions_to_mask(selected_positions, router_scores)
        selected_scores = torch.cat(
            [
                router_scores[batch_index].index_select(0, positions)
                for batch_index, positions in enumerate(selected_positions)
            ]
        )
        self._last_capacity = int(selected_positions[0].numel())
        self._last_selected_attention_token_count = int(selected_scores.numel())
        self._last_selected_count = int(selected_scores.numel())
        self._last_token_count = int(router_scores.numel())
        self._last_skipped_count = self._last_token_count - self._last_selected_count
        self._last_selected_fraction = (
            self._last_selected_count / self._last_token_count
            if self._last_token_count
            else 0.0
        )
        self._last_mean_router_score = float(
            router_scores.detach().float().mean().item()
        )
        self._last_mean_selected_router_score = float(
            selected_scores.detach().float().mean().item()
        )
        self._last_mean_selected_gate = float(
            torch.sigmoid(selected_scores.detach().float()).mean().item()
        )
        seq_len = router_scores.shape[1]
        midpoint = max(1, seq_len // 2)
        selected_mask_float = selected_mask.detach().float()
        self._last_selected_first_half_fraction = float(
            selected_mask_float[:, :midpoint].mean().item()
        )
        self._last_selected_second_half_fraction = (
            float(selected_mask_float[:, midpoint:].mean().item())
            if midpoint < seq_len
            else 0.0
        )
        selected_position_values = torch.cat(
            [positions.detach().float() for positions in selected_positions]
        )
        self._last_mean_selected_position = float(
            (selected_position_values / max(1, seq_len - 1)).mean().item()
        )
        self._last_router_scores = router_scores
        self._last_selected_mask = selected_mask
        return selected_positions, router_scores

    def _selected_attention_bias(
        self,
        positions: torch.Tensor,
        *,
        seq_len: int,
        attn_mask: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if attn_mask is None:
            key_pos = positions.view(1, -1)
            query_pos = positions.view(-1, 1)
            blocked = key_pos > query_pos
            bias = torch.zeros(
                (positions.numel(), positions.numel()),
                device=positions.device,
                dtype=dtype,
            )
            return bias.masked_fill(blocked, torch.finfo(dtype).min)
        full_bias = _explicit_attention_bias(
            attn_mask,
            seq_len=seq_len,
            device=positions.device,
            dtype=dtype,
        )[0, 0]
        return full_bias.index_select(0, positions).index_select(1, positions)

    def _forward_selected_only(
        self,
        hidden: torch.Tensor,
        selected_positions: list[torch.Tensor],
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_outputs: list[torch.Tensor] = []
        seq_len = hidden.shape[1]
        for batch_index, positions in enumerate(selected_positions):
            hidden_selected = (
                hidden[batch_index].index_select(0, positions).unsqueeze(0)
            )
            selected_bias = self._selected_attention_bias(
                positions,
                seq_len=seq_len,
                attn_mask=attn_mask,
                dtype=hidden.dtype,
            )
            with record_function("path1.mod_train_topc.selected_input_norm"):
                normed = self.input_norm(hidden_selected)
            residual = hidden_selected + self.attention(normed, selected_bias)
            with record_function("path1.mod_train_topc.selected_feedforward"):
                selected_output = residual + self.ffn(self.output_norm(residual))
            batch_outputs.append(
                hidden[batch_index]
                .clone()
                .index_copy(0, positions, selected_output.squeeze(0))
            )
        return torch.stack(batch_outputs, dim=0)

    def forward(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        depth_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if depth_memory is not None:
            raise ValueError("paper MoD top-C blocks do not consume depth memory")
        selected_positions, router_scores = self._route(hidden)
        selected_output = self._forward_selected_only(
            hidden, selected_positions, attn_mask
        )
        batch_outputs: list[torch.Tensor] = []
        for batch_index, positions in enumerate(selected_positions):
            selected_hidden = hidden[batch_index].index_select(0, positions)
            selected_updated = selected_output[batch_index].index_select(0, positions)
            gates = torch.sigmoid(
                router_scores[batch_index].index_select(0, positions)
            ).to(dtype=hidden.dtype)
            mixed = selected_hidden + gates.unsqueeze(-1) * (
                selected_updated - selected_hidden
            )
            batch_outputs.append(
                hidden[batch_index].clone().index_copy(0, positions, mixed)
            )
        output = torch.stack(batch_outputs, dim=0)
        if not torch.isfinite(output).all():
            raise FloatingPointError(
                "paper MoD top-C transformer block produced non-finite output"
            )
        return output

    def token_routing_payload(self) -> dict[str, object]:
        return {
            "kind": "mod-train-topc-block",
            "route_fraction": self.route_fraction,
            "capacity": self._last_capacity,
            "last_selected_fraction": self._last_selected_fraction,
            "last_selected_count": self._last_selected_count,
            "last_skipped_count": self._last_skipped_count,
            "last_token_count": self._last_token_count,
            "last_mean_router_score": self._last_mean_router_score,
            "last_mean_selected_router_score": self._last_mean_selected_router_score,
            "last_mean_selected_gate": self._last_mean_selected_gate,
            "last_selected_first_half_fraction": self._last_selected_first_half_fraction,
            "last_selected_second_half_fraction": self._last_selected_second_half_fraction,
            "last_mean_selected_position": self._last_mean_selected_position,
            "selected_attention_scope": "selected-only",
            "last_selected_attention_token_count": self._last_selected_attention_token_count,
            "causal_decode_safe": False,
            "label": "Paper MoD training-time full-sequence top-C block routing; not safe for autoregressive decode without an auxiliary causal router",
        }


class SoftGatedLocalCausalTransformerBlock(LocalCausalTransformerBlock):
    """Decode-safe partial-update router with continuous token gates.

    This is a practical MoD-adjacent probe: every token runs the dense block in
    the reference harness, then a causal gate controls how much of the block
    delta is accepted. It tests whether routing should be a smooth update
    controller before we spend engineering effort on sparse execution.
    """

    def __init__(
        self,
        d_model: int,
        head_count: int,
        d_ff: int,
        *,
        gate_floor: float,
        attention_module: LocalCausalSelfAttention | None = None,
        ffn_module: nn.Module | None = None,
    ) -> None:
        if not 0.0 <= gate_floor < 1.0:
            raise ValueError(f"gate_floor must be in [0, 1), got {gate_floor}")
        super().__init__(
            d_model,
            head_count,
            d_ff,
            attention_module=attention_module,
            ffn_module=ffn_module,
        )
        self.gate_floor = gate_floor
        self.router_norm = nn.LayerNorm(d_model)
        self.gate_projection = nn.Linear(d_model, 1)
        nn.init.normal_(self.gate_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_projection.bias)
        self._last_gate_mean = 0.0
        self._last_gate_min = 0.0
        self._last_gate_max = 0.0
        self._last_gate_std = 0.0
        self._last_raw_gate_mean = 0.0
        self._last_delta_norm = 0.0
        self._last_accepted_delta_norm = 0.0
        self._last_accepted_delta_ratio = 0.0
        self._last_gate_first_half_mean = 0.0
        self._last_gate_second_half_mean = 0.0

    def _gate_source(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.router_norm(hidden)

    def _raw_gate_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.gate_projection(self._gate_source(hidden)).squeeze(-1)

    def _gate(self, hidden: torch.Tensor) -> torch.Tensor:
        raw_gate = torch.sigmoid(self._raw_gate_logits(hidden))
        gate = self.gate_floor + (1.0 - self.gate_floor) * raw_gate
        gate = gate.to(dtype=hidden.dtype)
        detached_gate = gate.detach().float()
        detached_raw = raw_gate.detach().float()
        seq_len = gate.shape[1]
        midpoint = max(1, seq_len // 2)
        self._last_gate_mean = float(detached_gate.mean().item())
        self._last_gate_min = float(detached_gate.min().item())
        self._last_gate_max = float(detached_gate.max().item())
        self._last_gate_std = float(detached_gate.std().item())
        self._last_raw_gate_mean = float(detached_raw.mean().item())
        self._last_gate_first_half_mean = float(
            detached_gate[:, :midpoint].mean().item()
        )
        self._last_gate_second_half_mean = (
            float(detached_gate[:, midpoint:].mean().item())
            if midpoint < seq_len
            else 0.0
        )
        return gate

    def forward(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        depth_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if depth_memory is not None:
            raise ValueError("soft-gated transformer blocks do not consume depth memory")
        dense_output = super().forward(hidden, attn_mask)
        delta = dense_output - hidden
        gate = self._gate(hidden)
        output = hidden + gate.unsqueeze(-1) * delta
        with torch.no_grad():
            delta_norm = delta.detach().float().norm(dim=-1).mean()
            accepted_norm = (output - hidden).detach().float().norm(dim=-1).mean()
            self._last_delta_norm = float(delta_norm.item())
            self._last_accepted_delta_norm = float(accepted_norm.item())
            self._last_accepted_delta_ratio = float(
                (accepted_norm / delta_norm.clamp_min(1.0e-6)).item()
            )
        if not torch.isfinite(output).all():
            raise FloatingPointError("soft-gated transformer block produced non-finite output")
        return output

    def token_routing_payload(self) -> dict[str, object]:
        return {
            "kind": "soft-gate-block",
            "gate_floor": self.gate_floor,
            "last_mean_gate": self._last_gate_mean,
            "last_min_gate": self._last_gate_min,
            "last_max_gate": self._last_gate_max,
            "last_gate_std": self._last_gate_std,
            "last_raw_gate_mean": self._last_raw_gate_mean,
            "last_delta_norm": self._last_delta_norm,
            "last_accepted_delta_norm": self._last_accepted_delta_norm,
            "last_accepted_delta_ratio": self._last_accepted_delta_ratio,
            "last_gate_first_half_mean": self._last_gate_first_half_mean,
            "last_gate_second_half_mean": self._last_gate_second_half_mean,
            "selected_attention_scope": "dense-reference/full-causal-key-value",
            "causal_decode_safe": True,
            "label": "Decode-safe soft partial-update block; dense reference execution with continuous token gates",
        }


class RotarySoftGatedLocalCausalTransformerBlock(SoftGatedLocalCausalTransformerBlock):
    """Soft partial-update block whose gate is controlled by a P20/RGRP scan."""

    def __init__(
        self,
        d_model: int,
        head_count: int,
        d_ff: int,
        *,
        gate_floor: float,
        attention_module: LocalCausalSelfAttention | None = None,
        ffn_module: nn.Module | None = None,
    ) -> None:
        super().__init__(
            d_model,
            head_count,
            d_ff,
            gate_floor=gate_floor,
            attention_module=attention_module,
            ffn_module=ffn_module,
        )
        self.rotary_controller = P20RotaryStateOutputRuntimeSequenceMixer(
            d_model,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
        )
        self.rotary_controller_output_norm = SimpleRmsNorm(d_model)
        nn.init.normal_(self.gate_projection.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.gate_projection.bias)
        self._last_raw_controller_norm = 0.0
        self._last_normalized_controller_norm = 0.0

    def _gate_source(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = self.router_norm(hidden)
        controller_output = self.rotary_controller.scan(normed).emitted_outputs
        normalized = self.rotary_controller_output_norm(controller_output)
        self._last_raw_controller_norm = float(
            controller_output.detach().float().norm(dim=-1).mean().item()
        )
        self._last_normalized_controller_norm = float(
            normalized.detach().float().norm(dim=-1).mean().item()
        )
        return normalized

    def token_routing_payload(self) -> dict[str, object]:
        payload = super().token_routing_payload()
        payload.update(
            {
                "kind": "rotary-soft-gate-block",
                "last_controller_norm": self._last_normalized_controller_norm,
                "last_raw_controller_norm": self._last_raw_controller_norm,
                "last_normalized_controller_norm": self._last_normalized_controller_norm,
                "controller": "P20RotaryStateOutputRuntimeSequenceMixer",
                "state_transform": PrimitiveStateTransformMode.BLOCK_DIAGONAL_4.value,
                "controller_output_norm": "SimpleRmsNorm",
                "label": "Decode-safe soft partial-update block controlled by a P20/RGRP-style rotary recurrent scan",
            }
        )
        return payload


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
        self.attention = (
            attention_module
            if attention_module is not None
            else LocalCausalSelfAttention(d_model, head_count)
        )
        self.output_norm = SimpleRmsNorm(d_model)
        self.ffn = ReluSquaredFeedForward(d_model, d_ff)
        self.attention_scale = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.ffn_scale = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.residual_mix = nn.Parameter(
            torch.stack((torch.ones(d_model), torch.zeros(d_model))).float()
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        residual_anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual_anchor is None:
            residual_anchor = hidden
        mix = self.residual_mix.to(dtype=hidden.dtype)
        mixed_hidden = (
            mix[0].view(1, 1, -1) * hidden + mix[1].view(1, 1, -1) * residual_anchor
        )
        with record_function("path1.pr5_attention.input_norm"):
            normed = self.input_norm(mixed_hidden)
        attention_out = self.attention(normed, attn_mask)
        residual = (
            mixed_hidden
            + self.attention_scale.to(dtype=hidden.dtype).view(1, 1, -1) * attention_out
        )
        with record_function("path1.pr5_attention.feedforward"):
            return residual + self.ffn_scale.to(dtype=hidden.dtype).view(
                1, 1, -1
            ) * self.ffn(self.output_norm(residual))
