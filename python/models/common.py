from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gated_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(tensor)


def one_minus(tensor: torch.Tensor) -> torch.Tensor:
    return 1.0 - tensor


def rotate_state_pairs(state: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    batch_size, width = state.shape
    pair_count = width // 2
    state_pairs = state.reshape(batch_size, pair_count, 2)
    first = state_pairs[..., 0]
    second = state_pairs[..., 1]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return torch.stack([rotated_first, rotated_second], dim=-1).reshape(batch_size, width)


def leading_state_slice(state: torch.Tensor, width: int) -> torch.Tensor:
    return state[..., :width]


class SimpleRmsNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        denom = torch.sqrt(torch.mean(tensor * tensor, dim=-1, keepdim=True) + self.eps)
        return tensor / denom * self.weight


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden)))


class ReluSquaredFeedForward(nn.Module):
    """PR5-style feed-forward block with a zero-start output projection."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        activated = F.relu(self.fc1(hidden))
        return self.fc2(activated.square())


class RealEmlNodeOperator(nn.Module):
    """Stable real-valued EML-inspired binary node.

    The EML paper uses a complex-valued Exp-Minus-Log operator. This node keeps
    the repeated binary-tree structure but uses a bounded real-valued operator
    so it is practical inside a transformer feed-forward path.
    """

    def __init__(self) -> None:
        super().__init__()
        self.left_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.right_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.product_weight = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.difference_weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        combined = (
            self.left_weight.to(dtype=left.dtype) * left
            + self.right_weight.to(dtype=left.dtype) * right
            + self.product_weight.to(dtype=left.dtype) * left * right
            + self.difference_weight.to(dtype=left.dtype) * (left - right)
            + self.bias.to(dtype=left.dtype)
        )
        return torch.tanh(combined)


class GenericTreeNodeOperator(nn.Module):
    """Minimal binary tree node used as a non-EML tree control."""

    def __init__(self) -> None:
        super().__init__()
        self.left_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.right_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        combined = (
            self.left_weight.to(dtype=left.dtype) * left
            + self.right_weight.to(dtype=left.dtype) * right
            + self.bias.to(dtype=left.dtype)
        )
        return torch.tanh(combined)


class EmlInspiredTreeFeedForward(nn.Module):
    """Feed-forward replacement built from repeated differentiable binary trees."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        slot_count: int = 8,
        tree_depth: int = 3,
    ) -> None:
        super().__init__()
        if slot_count <= 0:
            raise ValueError(f"EML-inspired slot_count must be positive, got {slot_count}")
        if tree_depth <= 0:
            raise ValueError(f"EML-inspired tree_depth must be positive, got {tree_depth}")
        self.d_model = d_model
        self.tree_width = d_ff
        self.slot_count = slot_count
        self.tree_depth = tree_depth
        self.leaf_count = 1 << tree_depth
        self.slot_projection = nn.Linear(d_model, slot_count)
        self.node = self.build_node_operator()
        self.register_buffer("constants", torch.tensor((-1.0, 0.0, 1.0), dtype=torch.float32), persistent=False)
        self.leaf_selector_logits = nn.Parameter(
            torch.empty(self.tree_width, self.leaf_count, slot_count + self.constants.numel())
        )
        self.output_projection = nn.Linear(self.tree_width, d_model)
        self.reset_parameters()

    def build_node_operator(self) -> nn.Module:
        return RealEmlNodeOperator()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.slot_projection.weight, gain=0.5)
        nn.init.zeros_(self.slot_projection.bias)
        nn.init.normal_(self.leaf_selector_logits, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.5)
        nn.init.zeros_(self.output_projection.bias)

    def _basis(self, hidden: torch.Tensor) -> torch.Tensor:
        slots = torch.tanh(self.slot_projection(hidden))
        constants = self.constants.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        constants = constants.expand(hidden.shape[0], hidden.shape[1], -1)
        return torch.cat((slots, constants), dim=-1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        basis = self._basis(hidden)
        selector = torch.softmax(self.leaf_selector_logits.to(dtype=hidden.dtype), dim=-1)
        values = torch.einsum("wlc,bsc->bswl", selector, basis)
        for _ in range(self.tree_depth):
            values = self.node(values[..., 0::2], values[..., 1::2])
        roots = values.squeeze(-1)
        output = self.output_projection(roots)
        if not torch.isfinite(output).all():
            raise FloatingPointError("EML-inspired feed-forward produced non-finite output")
        return output

    def selector_entropy(self) -> float:
        selector = torch.softmax(self.leaf_selector_logits.detach().float(), dim=-1)
        entropy = -(selector * torch.log(selector.clamp_min(1.0e-8))).sum(dim=-1)
        return float(entropy.mean().item())

    def diagnostic_payload(self) -> dict[str, object]:
        return {
            "kind": "eml-tree",
            "slot_count": self.slot_count,
            "tree_depth": self.tree_depth,
            "leaf_count": self.leaf_count,
            "selector_entropy": self.selector_entropy(),
        }


class GenericBinaryTreeFeedForward(EmlInspiredTreeFeedForward):
    """Tree-shaped FFN control without the EML product/difference node terms."""

    def build_node_operator(self) -> nn.Module:
        return GenericTreeNodeOperator()

    def diagnostic_payload(self) -> dict[str, object]:
        payload = super().diagnostic_payload()
        payload["kind"] = "generic-tree"
        return payload


class GatedEmlFeedForward(nn.Module):
    """Gated hybrid of the standard MLP and the EML-inspired tree FFN."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        slot_count: int = 8,
        tree_depth: int = 3,
        eml_mix_init: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < eml_mix_init < 1.0:
            raise ValueError(f"eml_mix_init must be in (0, 1), got {eml_mix_init}")
        self.mlp = PositionWiseFeedForward(d_model, d_ff)
        self.eml = EmlInspiredTreeFeedForward(
            d_model,
            d_ff,
            slot_count=slot_count,
            tree_depth=tree_depth,
        )
        self.mix_logit = nn.Parameter(torch.full((d_model,), torch.logit(torch.tensor(eml_mix_init)).item()))
        self._last_stats: dict[str, float] = {}

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        mix = torch.sigmoid(self.mix_logit.to(dtype=hidden.dtype)).view(1, 1, -1)
        mlp_output = self.mlp(hidden)
        eml_output = self.eml(hidden)
        output = (1.0 - mix) * mlp_output + mix * eml_output
        self._last_stats = _activation_stats(mlp_output, eml_output, output)
        return output

    def diagnostic_payload(self) -> dict[str, object]:
        mix = torch.sigmoid(self.mix_logit.detach().float())
        return {
            "kind": "mlp-eml-gated",
            "mean_channel_mix": float(mix.mean().item()),
            "min_channel_mix": float(mix.min().item()),
            "max_channel_mix": float(mix.max().item()),
            "tree": self.eml.diagnostic_payload(),
            "last_activation": dict(self._last_stats),
        }


class RoutedEmlFeedForward(nn.Module):
    """Standard MLP with a sparse token-routed EML expert side branch."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        slot_count: int = 8,
        tree_depth: int = 3,
        route_fraction: float = 0.25,
        eml_mix_init: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < route_fraction <= 1.0:
            raise ValueError(f"route_fraction must be in (0, 1], got {route_fraction}")
        if not 0.0 < eml_mix_init < 1.0:
            raise ValueError(f"eml_mix_init must be in (0, 1), got {eml_mix_init}")
        self.route_fraction = route_fraction
        self.mlp = PositionWiseFeedForward(d_model, d_ff)
        self.eml = EmlInspiredTreeFeedForward(
            d_model,
            d_ff,
            slot_count=slot_count,
            tree_depth=tree_depth,
        )
        self.router = nn.Linear(d_model, 1)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router.bias)
        self.mix_logit = nn.Parameter(torch.full((d_model,), torch.logit(torch.tensor(eml_mix_init)).item()))
        self._last_stats: dict[str, float] = {}

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"RoutedEmlFeedForward expects [batch, seq, d_model], got {tuple(hidden.shape)}")
        batch_size, seq_len, d_model = hidden.shape
        token_count = batch_size * seq_len
        selected_count = max(1, min(token_count, int(token_count * self.route_fraction + 0.999999)))

        base = self.mlp(hidden)
        router_logits = self.router(hidden).squeeze(-1)
        selected_indices = torch.topk(
            router_logits.reshape(-1),
            k=selected_count,
            sorted=False,
        ).indices
        selected_hidden = hidden.reshape(token_count, d_model).index_select(0, selected_indices)
        selected_eml = self.eml(selected_hidden.unsqueeze(0)).squeeze(0)
        selected_gate = torch.sigmoid(router_logits.reshape(-1).index_select(0, selected_indices))
        channel_mix = torch.sigmoid(self.mix_logit.to(dtype=hidden.dtype)).view(1, -1)
        selected_contribution = selected_eml * selected_gate.to(dtype=hidden.dtype).unsqueeze(-1) * channel_mix

        output = base.reshape(token_count, d_model).clone()
        output.index_add_(0, selected_indices, selected_contribution)
        output = output.view(batch_size, seq_len, d_model)
        self._last_stats = _activation_stats(base, selected_contribution, output)
        self._last_stats["selected_token_fraction"] = float(selected_count / token_count)
        self._last_stats["mean_selected_gate"] = float(selected_gate.detach().float().mean().item())
        if not torch.isfinite(output).all():
            raise FloatingPointError("routed EML feed-forward produced non-finite output")
        return output

    def diagnostic_payload(self) -> dict[str, object]:
        mix = torch.sigmoid(self.mix_logit.detach().float())
        return {
            "kind": "mlp-eml-routed",
            "route_fraction": self.route_fraction,
            "mean_channel_mix": float(mix.mean().item()),
            "min_channel_mix": float(mix.min().item()),
            "max_channel_mix": float(mix.max().item()),
            "tree": self.eml.diagnostic_payload(),
            "last_activation": dict(self._last_stats),
        }


class TinyMlpExpertFeedForward(nn.Module):
    """Standard MLP plus a tiny gated MLP expert side branch."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        expert_width: int | None = None,
        expert_mix_init: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < expert_mix_init < 1.0:
            raise ValueError(f"expert_mix_init must be in (0, 1), got {expert_mix_init}")
        width = expert_width or max(1, d_ff // 2)
        self.expert_width = width
        self.mlp = PositionWiseFeedForward(d_model, d_ff)
        self.expert = nn.Sequential(
            nn.Linear(d_model, width),
            nn.GELU(),
            nn.Linear(width, d_model),
        )
        self.mix_logit = nn.Parameter(torch.full((d_model,), torch.logit(torch.tensor(expert_mix_init)).item()))
        self._last_stats: dict[str, float] = {}

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        mix = torch.sigmoid(self.mix_logit.to(dtype=hidden.dtype)).view(1, 1, -1)
        mlp_output = self.mlp(hidden)
        expert_output = self.expert(hidden)
        output = (1.0 - mix) * mlp_output + mix * expert_output
        self._last_stats = _activation_stats(mlp_output, expert_output, output)
        return output

    def diagnostic_payload(self) -> dict[str, object]:
        mix = torch.sigmoid(self.mix_logit.detach().float())
        return {
            "kind": "tiny-mlp-gated",
            "expert_width": self.expert_width,
            "mean_channel_mix": float(mix.mean().item()),
            "min_channel_mix": float(mix.min().item()),
            "max_channel_mix": float(mix.max().item()),
            "last_activation": dict(self._last_stats),
        }


class TinyGluExpertFeedForward(nn.Module):
    """Standard MLP plus a tiny gated bilinear/GLU expert side branch."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        expert_width: int | None = None,
        expert_mix_init: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < expert_mix_init < 1.0:
            raise ValueError(f"expert_mix_init must be in (0, 1), got {expert_mix_init}")
        width = expert_width or max(1, d_ff // 2)
        self.expert_width = width
        self.mlp = PositionWiseFeedForward(d_model, d_ff)
        self.gate_value_projection = nn.Linear(d_model, width * 2)
        self.output_projection = nn.Linear(width, d_model)
        self.mix_logit = nn.Parameter(torch.full((d_model,), torch.logit(torch.tensor(expert_mix_init)).item()))
        self._last_stats: dict[str, float] = {}

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        mix = torch.sigmoid(self.mix_logit.to(dtype=hidden.dtype)).view(1, 1, -1)
        mlp_output = self.mlp(hidden)
        gate, value = self.gate_value_projection(hidden).chunk(2, dim=-1)
        expert_output = self.output_projection(F.silu(gate) * value)
        output = (1.0 - mix) * mlp_output + mix * expert_output
        self._last_stats = _activation_stats(mlp_output, expert_output, output)
        return output

    def diagnostic_payload(self) -> dict[str, object]:
        mix = torch.sigmoid(self.mix_logit.detach().float())
        return {
            "kind": "tiny-glu-gated",
            "expert_width": self.expert_width,
            "mean_channel_mix": float(mix.mean().item()),
            "min_channel_mix": float(mix.min().item()),
            "max_channel_mix": float(mix.max().item()),
            "last_activation": dict(self._last_stats),
        }


class GenericTreeExpertFeedForward(nn.Module):
    """Standard MLP plus a gated generic binary-tree expert control."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        slot_count: int = 8,
        tree_depth: int = 3,
        expert_mix_init: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < expert_mix_init < 1.0:
            raise ValueError(f"expert_mix_init must be in (0, 1), got {expert_mix_init}")
        self.mlp = PositionWiseFeedForward(d_model, d_ff)
        self.tree = GenericBinaryTreeFeedForward(
            d_model,
            d_ff,
            slot_count=slot_count,
            tree_depth=tree_depth,
        )
        self.mix_logit = nn.Parameter(torch.full((d_model,), torch.logit(torch.tensor(expert_mix_init)).item()))
        self._last_stats: dict[str, float] = {}

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        mix = torch.sigmoid(self.mix_logit.to(dtype=hidden.dtype)).view(1, 1, -1)
        mlp_output = self.mlp(hidden)
        tree_output = self.tree(hidden)
        output = (1.0 - mix) * mlp_output + mix * tree_output
        self._last_stats = _activation_stats(mlp_output, tree_output, output)
        return output

    def diagnostic_payload(self) -> dict[str, object]:
        mix = torch.sigmoid(self.mix_logit.detach().float())
        return {
            "kind": "generic-tree-gated",
            "mean_channel_mix": float(mix.mean().item()),
            "min_channel_mix": float(mix.min().item()),
            "max_channel_mix": float(mix.max().item()),
            "tree": self.tree.diagnostic_payload(),
            "last_activation": dict(self._last_stats),
        }


def _activation_stats(base: torch.Tensor, expert: torch.Tensor, output: torch.Tensor) -> dict[str, float]:
    base_norm = base.detach().float().norm(dim=-1).mean()
    expert_norm = expert.detach().float().norm(dim=-1).mean()
    output_norm = output.detach().float().norm(dim=-1).mean()
    return {
        "mean_base_norm": float(base_norm.item()),
        "mean_expert_norm": float(expert_norm.item()),
        "mean_output_norm": float(output_norm.item()),
    }


def build_linear(d_in: int, d_out: int, *, bias: bool = True) -> nn.Linear:
    return nn.Linear(d_in, d_out, bias=bias)
