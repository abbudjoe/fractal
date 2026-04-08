from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from python.models.common import PositionWiseFeedForward
from python.models.transformer import LocalCausalTransformerBlock, local_causal_mask
from python.specs.mini_moe import MiniMoeDispatchMode, MiniMoeSurfaceSpec, ResolvedDispatchContract


@dataclass(frozen=True)
class RoutePlan:
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    router_logits: torch.Tensor


@dataclass(frozen=True)
class DispatchObservation:
    active_expert_count: int
    route_entropy: float
    winner_margin: float


class MiniMoeObservabilitySink(Protocol):
    def record(self, layer_index: int, observation: DispatchObservation) -> None:
        ...


class NullMiniMoeObservabilitySink:
    def record(self, layer_index: int, observation: DispatchObservation) -> None:
        return None


class MiniMoeRouter(nn.Module):
    def plan(self, hidden: torch.Tensor) -> RoutePlan:  # pragma: no cover - abstract
        raise NotImplementedError


class OneShotTopKRouter(MiniMoeRouter):
    def __init__(self, hidden_dim: int, experts_per_block: int, active_experts_per_token: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, experts_per_block)
        self.active_experts_per_token = active_experts_per_token

    def plan(self, hidden: torch.Tensor) -> RoutePlan:
        logits = self.projection(hidden)
        top_values, top_indices = torch.topk(logits, k=self.active_experts_per_token, dim=-1)
        weights = torch.softmax(top_values, dim=-1)
        return RoutePlan(expert_indices=top_indices, expert_weights=weights, router_logits=logits)


class MiniMoeDispatcher(nn.Module):
    def dispatch(
        self,
        hidden: torch.Tensor,
        route_plan: RoutePlan,
        experts: nn.ModuleList,
    ) -> tuple[torch.Tensor, DispatchObservation]:  # pragma: no cover - abstract
        raise NotImplementedError


class SparseTopKDispatcher(MiniMoeDispatcher):
    def __init__(self, contract: ResolvedDispatchContract) -> None:
        super().__init__()
        self.contract = contract

    def dispatch(
        self,
        hidden: torch.Tensor,
        route_plan: RoutePlan,
        experts: nn.ModuleList,
    ) -> tuple[torch.Tensor, DispatchObservation]:
        expert_outputs = torch.stack([expert(hidden) for expert in experts], dim=-2)
        if self.contract.mode is MiniMoeDispatchMode.DENSE_DEBUG:
            dense_weights = torch.softmax(route_plan.router_logits, dim=-1)
            mixed = torch.sum(expert_outputs * dense_weights.unsqueeze(-1), dim=-2)
            top2 = torch.topk(dense_weights, k=min(2, dense_weights.shape[-1]), dim=-1).values
            observation = DispatchObservation(
                active_expert_count=dense_weights.shape[-1],
                route_entropy=float(
                    (-dense_weights * dense_weights.clamp_min(1.0e-9).log()).sum(dim=-1).mean().item()
                ),
                winner_margin=float((top2[..., 0] - top2[..., -1]).mean().item()),
            )
            return mixed, observation

        gather_index = route_plan.expert_indices.unsqueeze(-1).expand(
            *route_plan.expert_indices.shape,
            hidden.shape[-1],
        )
        selected_outputs = torch.gather(expert_outputs, -2, gather_index)
        mixed = torch.sum(selected_outputs * route_plan.expert_weights.unsqueeze(-1), dim=-2)
        route_probs = torch.softmax(route_plan.router_logits, dim=-1)
        top2 = torch.topk(route_probs, k=min(2, route_probs.shape[-1]), dim=-1).values
        observation = DispatchObservation(
            active_expert_count=int(route_plan.expert_indices.shape[-1]),
            route_entropy=float(
                (-route_probs * route_probs.clamp_min(1.0e-9).log()).sum(dim=-1).mean().item()
            ),
            winner_margin=float((top2[..., 0] - top2[..., -1]).mean().item()),
        )
        return mixed, observation


class SparseMoeFeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        expert_ffn_multiplier: int,
        experts_per_block: int,
        active_experts_per_token: int,
        *,
        router: MiniMoeRouter,
        dispatcher: MiniMoeDispatcher,
        observability_sink: MiniMoeObservabilitySink | None,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.router = router
        self.dispatcher = dispatcher
        self.observability_sink = observability_sink or NullMiniMoeObservabilitySink()
        self.layer_index = layer_index
        self.experts = nn.ModuleList(
            [
                PositionWiseFeedForward(hidden_dim, hidden_dim * expert_ffn_multiplier)
                for _ in range(experts_per_block)
            ]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        route_plan = self.router.plan(hidden)
        mixed, observation = self.dispatcher.dispatch(hidden, route_plan, self.experts)
        self.observability_sink.record(self.layer_index, observation)
        return mixed


class MiniMoeBackboneModel(nn.Module):
    def __init__(
        self,
        surface_spec: MiniMoeSurfaceSpec,
        *,
        observability_sink: MiniMoeObservabilitySink | None = None,
    ) -> None:
        super().__init__()
        surface_spec.validate()
        self.surface_spec = surface_spec
        self.embedding = nn.Embedding(
            surface_spec.architecture.backbone.vocab_size,
            surface_spec.architecture.backbone.hidden_dim,
        )
        self.output = nn.Linear(
            surface_spec.architecture.backbone.hidden_dim,
            surface_spec.architecture.backbone.vocab_size,
            bias=False,
        )
        self.blocks = nn.ModuleList()
        layout = surface_spec.architecture.resolved_layout()
        dispatch_contract = surface_spec.runtime.dispatch.resolve(
            surface_spec.architecture.moe.active_experts_per_token
        )
        sink = observability_sink or NullMiniMoeObservabilitySink()
        moe_layers = set(layout.moe_layers)
        for layer_index in range(surface_spec.architecture.backbone.total_layers):
            if layer_index in moe_layers:
                if surface_spec.architecture.router.kind != "one_shot":
                    raise NotImplementedError(
                        "recurrent mini-MoE routers are intentionally deferred; the substrate seam is present but no recurrent implementation is provided yet"
                    )
                ffn_module = SparseMoeFeedForward(
                    hidden_dim=surface_spec.architecture.backbone.hidden_dim,
                    expert_ffn_multiplier=surface_spec.architecture.moe.expert_ffn_multiplier,
                    experts_per_block=surface_spec.architecture.moe.experts_per_block,
                    active_experts_per_token=surface_spec.architecture.moe.active_experts_per_token,
                    router=OneShotTopKRouter(
                        hidden_dim=surface_spec.architecture.backbone.hidden_dim,
                        experts_per_block=surface_spec.architecture.moe.experts_per_block,
                        active_experts_per_token=surface_spec.architecture.moe.active_experts_per_token,
                    ),
                    dispatcher=SparseTopKDispatcher(dispatch_contract),
                    observability_sink=sink,
                    layer_index=layer_index,
                )
            else:
                ffn_module = PositionWiseFeedForward(
                    surface_spec.architecture.backbone.hidden_dim,
                    surface_spec.architecture.backbone.hidden_dim
                    * surface_spec.architecture.backbone.ffn_multiplier,
                )
            self.blocks.append(
                LocalCausalTransformerBlock(
                    surface_spec.architecture.backbone.hidden_dim,
                    surface_spec.architecture.backbone.head_count,
                    surface_spec.architecture.backbone.hidden_dim
                    * surface_spec.architecture.backbone.ffn_multiplier,
                    ffn_module=ffn_module,
                )
            )

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        mask = local_causal_mask(
            input_ids.shape[1],
            self.surface_spec.architecture.backbone.local_window,
            input_ids.device,
        )
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.output(hidden)

