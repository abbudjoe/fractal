from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import heapq
import math
from typing import Protocol

import torch
import torch.nn as nn

from python.models.common import AuxiliaryLossProvider, PositionWiseFeedForward
from python.models.transformer import LocalCausalTransformerBlock, local_causal_attention_bias
from python.reporting.schema import (
    ExpertUsageSummary,
    MiniMoeControllerRoundSummary,
    MiniMoeDispatchSummary,
    MiniMoeLayerSummary,
    MiniMoeReportSummary,
    MiniMoeRoutingSummary,
    MiniMoeTokenRoundTrace,
    MiniMoeTokenRouteTrace,
)
from python.specs.mini_moe import (
    LearnedGateTeacherKind,
    MiniMoeDispatchExecutionStrategy,
    MiniMoeDispatchMode,
    MiniMoeSurfaceSpec,
    RecurrentRoundExecutionStrategy,
    RecurrentRoundGateKind,
    RecurrentRoundGateSpec,
    ResolvedDispatchContract,
)

_ENTROPY_EPS = 1.0e-9


@dataclass(frozen=True)
class RouteRoundSummary:
    expert_logits: torch.Tensor
    expert_weights: torch.Tensor
    applied_token_fraction: float = 1.0
    mean_gate_probability: float | None = None


@dataclass(frozen=True)
class RoutePlan:
    expert_logits: torch.Tensor
    expert_weights: torch.Tensor
    round_summaries: tuple[RouteRoundSummary, ...]
    auxiliary_loss: torch.Tensor | None = None


@dataclass(frozen=True)
class DispatchPlan:
    layer_index: int
    mode: MiniMoeDispatchMode
    selected_expert_indices: torch.Tensor
    selected_expert_weights: torch.Tensor


class MiniMoeObservabilitySink(Protocol):
    def record_input_batch(self, input_ids: torch.Tensor) -> None:
        ...

    def record_route_plan(self, layer_index: int, route_plan: RoutePlan) -> None:
        ...

    def record_dispatch_plan(self, dispatch_plan: DispatchPlan) -> None:
        ...

    def reset(self) -> None:
        ...

    def finalize(self) -> MiniMoeReportSummary | None:
        ...


class NullMiniMoeObservabilitySink:
    def record_input_batch(self, input_ids: torch.Tensor) -> None:
        return None

    def record_route_plan(self, layer_index: int, route_plan: RoutePlan) -> None:
        return None

    def record_dispatch_plan(self, dispatch_plan: DispatchPlan) -> None:
        return None

    def reset(self) -> None:
        return None

    def finalize(self) -> MiniMoeReportSummary | None:
        return None


def _route_entropy_bits(route_weights: torch.Tensor) -> torch.Tensor:
    return (-route_weights * route_weights.clamp_min(_ENTROPY_EPS).log2()).sum(dim=-1)


def _normalized_route_entropy(route_weights: torch.Tensor) -> torch.Tensor:
    expert_count = route_weights.shape[-1]
    if expert_count <= 1:
        return torch.zeros(route_weights.shape[:-1], dtype=route_weights.dtype, device=route_weights.device)
    return _route_entropy_bits(route_weights) / math.log2(expert_count)


def _winner_margin(route_weights: torch.Tensor) -> torch.Tensor:
    top_values = torch.topk(route_weights, k=min(2, route_weights.shape[-1]), dim=-1).values
    if top_values.shape[-1] == 1:
        return top_values[..., 0]
    return top_values[..., 0] - top_values[..., 1]


def _lowest_margin_fraction_mask(
    route_weights: torch.Tensor,
    target_applied_fraction: float,
) -> torch.Tensor:
    margins = _winner_margin(route_weights)
    flat_margins = margins.reshape(-1)
    token_count = flat_margins.numel()
    if token_count == 0:
        return torch.zeros_like(margins, dtype=torch.bool)
    active_token_count = min(
        token_count,
        max(1, math.ceil(target_applied_fraction * token_count)),
    )
    sorted_token_indices = torch.argsort(flat_margins, stable=True)
    active_flat = torch.zeros(token_count, dtype=torch.bool, device=route_weights.device)
    active_flat[sorted_token_indices[:active_token_count]] = True
    return active_flat.reshape_as(margins)


def _top_fraction_mask(
    score_values: torch.Tensor,
    target_applied_fraction: float,
) -> torch.Tensor:
    flat_scores = score_values.reshape(-1)
    token_count = flat_scores.numel()
    if token_count == 0:
        return torch.zeros_like(score_values, dtype=torch.bool)
    active_token_count = min(
        token_count,
        max(1, math.ceil(target_applied_fraction * token_count)),
    )
    sorted_token_indices = torch.argsort(flat_scores, descending=True, stable=True)
    active_flat = torch.zeros(token_count, dtype=torch.bool, device=score_values.device)
    active_flat[sorted_token_indices[:active_token_count]] = True
    return active_flat.reshape_as(score_values)


def _teacher_ambiguity_score(route_weights: torch.Tensor) -> torch.Tensor:
    normalized_entropy = _normalized_route_entropy(route_weights)
    inverse_margin = 1.0 - _winner_margin(route_weights)
    return torch.clamp(0.5 * (normalized_entropy + inverse_margin), 0.0, 1.0)


def _scaled_winner_margin_threshold(
    *,
    base_threshold: float,
    expert_count: int,
    reference_experts_per_block: int,
) -> float:
    if expert_count <= 1:
        return base_threshold
    return base_threshold * (
        math.log2(reference_experts_per_block) / math.log2(expert_count)
    )


@dataclass
class _LayerAccumulator:
    sampled_tokens: int = 0
    route_entropy_sum: float = 0.0
    reroute_token_count: int = 0
    selection_counts: list[int] = field(default_factory=list)
    selected_weight_sums: list[float] = field(default_factory=list)
    dispatch_counts: list[int] = field(default_factory=list)
    dispatch_mode: MiniMoeDispatchMode | None = None


@dataclass
class _RoundAccumulator:
    sampled_tokens: int = 0
    entropy_sum: float = 0.0
    winner_margin_sum: float = 0.0
    adjustment_l1_sum: float = 0.0
    rerouted_token_count: int = 0
    applied_token_fraction_sum: float = 0.0
    gate_probability_sum: float = 0.0
    gate_probability_samples: int = 0
    has_adjustment: bool = False


@dataclass(frozen=True, order=True)
class _TokenTraceCandidate:
    score: tuple[float, float, float, float]
    trace: MiniMoeTokenRouteTrace = field(compare=False)


class CollectingMiniMoeObservabilitySink:
    def __init__(self, surface_spec: MiniMoeSurfaceSpec) -> None:
        self._expert_count = surface_spec.architecture.moe.experts_per_block
        self._max_token_route_traces_per_layer = (
            surface_spec.observability.max_token_route_traces_per_layer
        )
        self._collect_token_traces = self._max_token_route_traces_per_layer > 0
        self.reset()

    def reset(self) -> None:
        self._sampled_tokens = 0
        self._winner_counts = [0 for _ in range(self._expert_count)]
        self._mean_expert_weight_sums = [0.0 for _ in range(self._expert_count)]
        self._winner_margin_sum = 0.0
        self._route_entropy_sum = 0.0
        self._active_expert_count = 0
        self._round_count = 0
        self._round_adjustment_sums: dict[int, float] = defaultdict(float)
        self._round_adjustment_tokens: dict[int, int] = defaultdict(int)
        self._layers: dict[int, _LayerAccumulator] = {}
        self._rounds: dict[tuple[int, int], _RoundAccumulator] = {}
        self._current_input_ids: torch.Tensor | None = None
        self._current_forward_pass_index = 0
        self._forward_pass_index = 0
        self._trace_candidate_heaps: dict[int, list[_TokenTraceCandidate]] = defaultdict(list)
        self._trace_serial = 0

    def record_input_batch(self, input_ids: torch.Tensor) -> None:
        self._forward_pass_index += 1
        self._current_forward_pass_index = self._forward_pass_index
        if self._collect_token_traces:
            self._current_input_ids = input_ids.detach().cpu()
        else:
            self._current_input_ids = None

    def record_route_plan(self, layer_index: int, route_plan: RoutePlan) -> None:
        if not self._collect_token_traces:
            self._record_route_plan_reduced(layer_index, route_plan)
            return

        route_weights = route_plan.expert_weights.detach().float().cpu()
        token_count = int(route_weights.shape[0] * route_weights.shape[1])
        self._sampled_tokens += token_count
        self._round_count = max(self._round_count, len(route_plan.round_summaries))

        route_entropy = _route_entropy_bits(route_weights)
        winner_margin = _winner_margin(route_weights)
        winners = route_weights.argmax(dim=-1)
        self._route_entropy_sum += float(route_entropy.sum().item())
        self._winner_margin_sum += float(winner_margin.sum().item())
        for expert_index in range(self._expert_count):
            self._mean_expert_weight_sums[expert_index] += float(
                route_weights[..., expert_index].sum().item()
            )
            self._winner_counts[expert_index] += int((winners == expert_index).sum().item())

        layer = self._layers.setdefault(
            layer_index,
            _LayerAccumulator(
                selection_counts=[0 for _ in range(self._expert_count)],
                selected_weight_sums=[0.0 for _ in range(self._expert_count)],
                dispatch_counts=[0 for _ in range(self._expert_count)],
            ),
        )
        layer.sampled_tokens += token_count
        layer.route_entropy_sum += float(route_entropy.sum().item())

        first_round_winners: torch.Tensor | None = None
        previous_weights: torch.Tensor | None = None
        previous_winners: torch.Tensor | None = None
        for round_index, round_summary in enumerate(route_plan.round_summaries):
            round_weights = round_summary.expert_weights.detach().float().cpu()
            round_entropy = _route_entropy_bits(round_weights)
            round_margin = _winner_margin(round_weights)
            round_winners = round_weights.argmax(dim=-1)
            round_key = (layer_index, round_index)
            accumulator = self._rounds.setdefault(round_key, _RoundAccumulator())
            accumulator.sampled_tokens += token_count
            accumulator.entropy_sum += float(round_entropy.sum().item())
            accumulator.winner_margin_sum += float(round_margin.sum().item())
            accumulator.applied_token_fraction_sum += (
                round_summary.applied_token_fraction * token_count
            )
            if round_summary.mean_gate_probability is not None:
                accumulator.gate_probability_sum += (
                    round_summary.mean_gate_probability * token_count
                )
                accumulator.gate_probability_samples += token_count

            if round_index == 0:
                first_round_winners = round_winners
            if previous_weights is not None and previous_winners is not None:
                adjustment_l1 = (round_weights - previous_weights).abs().sum(dim=-1)
                rerouted = round_winners != previous_winners
                accumulator.adjustment_l1_sum += float(adjustment_l1.sum().item())
                accumulator.rerouted_token_count += int(rerouted.sum().item())
                accumulator.has_adjustment = True
                self._round_adjustment_sums[round_index - 1] += float(adjustment_l1.sum().item())
                self._round_adjustment_tokens[round_index - 1] += token_count

            previous_weights = round_weights
            previous_winners = round_winners

        if first_round_winners is not None:
            layer.reroute_token_count += int((winners != first_round_winners).sum().item())

        if self._current_input_ids is not None and self._max_token_route_traces_per_layer > 0:
            self._record_token_traces(layer_index, route_plan)

    def record_dispatch_plan(self, dispatch_plan: DispatchPlan) -> None:
        selected_indices = dispatch_plan.selected_expert_indices.detach().reshape(-1)
        selected_weights = dispatch_plan.selected_expert_weights.detach().float().reshape(-1)
        layer = self._layers.setdefault(
            dispatch_plan.layer_index,
            _LayerAccumulator(
                selection_counts=[0 for _ in range(self._expert_count)],
                selected_weight_sums=[0.0 for _ in range(self._expert_count)],
                dispatch_counts=[0 for _ in range(self._expert_count)],
            ),
        )
        layer.dispatch_mode = dispatch_plan.mode
        self._active_expert_count = max(
            self._active_expert_count,
            int(dispatch_plan.selected_expert_indices.shape[-1]),
        )
        selection_counts = torch.bincount(
            selected_indices,
            minlength=self._expert_count,
        ).detach().cpu().tolist()
        selected_weight_sums = torch.bincount(
            selected_indices,
            weights=selected_weights,
            minlength=self._expert_count,
        ).detach().cpu().tolist()
        for expert_index, selection_count in enumerate(selection_counts):
            layer.selection_counts[expert_index] += int(selection_count)
            layer.dispatch_counts[expert_index] += int(selection_count)
            layer.selected_weight_sums[expert_index] += float(selected_weight_sums[expert_index])

    def finalize(self) -> MiniMoeReportSummary | None:
        if self._sampled_tokens == 0:
            return None

        layers: list[MiniMoeLayerSummary] = []
        dispatch_summaries: list[MiniMoeDispatchSummary] = []
        for layer_index in sorted(self._layers):
            layer = self._layers[layer_index]
            expert_usage = [
                ExpertUsageSummary(
                    expert_id=expert_index,
                    selection_count=layer.selection_counts[expert_index],
                    mean_weight=(
                        layer.selected_weight_sums[expert_index]
                        / layer.selection_counts[expert_index]
                        if layer.selection_counts[expert_index] > 0
                        else 0.0
                    ),
                )
                for expert_index in range(self._expert_count)
            ]
            layers.append(
                MiniMoeLayerSummary(
                    layer_index=layer_index,
                    sampled_tokens=layer.sampled_tokens,
                    route_entropy_bits=layer.route_entropy_sum / max(layer.sampled_tokens, 1),
                    reroute_fraction=layer.reroute_token_count / max(layer.sampled_tokens, 1),
                    expert_usage=expert_usage,
                )
            )
            dispatch_summaries.append(
                MiniMoeDispatchSummary(
                    layer_index=layer_index,
                    mode=layer.dispatch_mode or MiniMoeDispatchMode.SPARSE_TOP_K,
                    selected_expert_counts=list(layer.dispatch_counts),
                    dropped_token_fraction=None,
                )
            )

        controller_rounds: list[MiniMoeControllerRoundSummary] = []
        for (layer_index, round_index) in sorted(self._rounds):
            round_stats = self._rounds[(layer_index, round_index)]
            controller_rounds.append(
                MiniMoeControllerRoundSummary(
                    layer_index=layer_index,
                    round_index=round_index + 1,
                    mean_route_entropy_bits=round_stats.entropy_sum / max(round_stats.sampled_tokens, 1),
                    mean_winner_margin=round_stats.winner_margin_sum / max(round_stats.sampled_tokens, 1),
                    mean_route_adjustment_l1=(
                        round_stats.adjustment_l1_sum / round_stats.sampled_tokens
                        if round_stats.has_adjustment
                        else None
                    ),
                    rerouted_token_fraction=(
                        round_stats.rerouted_token_count / max(round_stats.sampled_tokens, 1)
                        if round_stats.has_adjustment
                        else 0.0
                    ),
                    applied_token_fraction=(
                        round_stats.applied_token_fraction_sum / max(round_stats.sampled_tokens, 1)
                    ),
                    mean_gate_probability=(
                        round_stats.gate_probability_sum / max(round_stats.gate_probability_samples, 1)
                        if round_stats.gate_probability_samples > 0
                        else None
                    ),
                )
            )

        token_traces: list[MiniMoeTokenRouteTrace] = []
        for layer_index in sorted(self._trace_candidate_heaps):
            candidates = sorted(
                self._trace_candidate_heaps[layer_index],
                key=lambda candidate: candidate.score,
                reverse=True,
            )
            token_traces.extend(candidate.trace for candidate in candidates)

        routing = MiniMoeRoutingSummary(
            sampled_tokens=self._sampled_tokens,
            layer_count=len(self._layers),
            round_count=self._round_count,
            mean_route_entropy_bits=self._route_entropy_sum / max(self._sampled_tokens, 1),
            mean_winner_margin=self._winner_margin_sum / max(self._sampled_tokens, 1),
            mean_expert_weights=[
                weight_sum / max(self._sampled_tokens, 1) for weight_sum in self._mean_expert_weight_sums
            ],
            winner_counts=list(self._winner_counts),
            active_expert_count=self._active_expert_count,
            mean_round_adjustment_l1=[
                self._round_adjustment_sums[round_index]
                / max(self._round_adjustment_tokens[round_index], 1)
                for round_index in range(max(0, self._round_count - 1))
            ],
        )
        return MiniMoeReportSummary(
            routing=routing,
            layers=layers,
            dispatch=dispatch_summaries,
            controller_rounds=controller_rounds,
            token_traces=token_traces,
        )

    def _record_token_traces(self, layer_index: int, route_plan: RoutePlan) -> None:
        if self._current_input_ids is None:
            return
        token_ids = self._current_input_ids
        batch_size, seq_len = token_ids.shape
        if route_plan.round_summaries:
            round_weights = [summary.expert_weights.detach().float().cpu() for summary in route_plan.round_summaries]
        else:
            round_weights = [route_plan.expert_weights.detach().float().cpu()]

        round_winners = [weights.argmax(dim=-1) for weights in round_weights]
        round_entropies = [_route_entropy_bits(weights) for weights in round_weights]
        round_margins = [_winner_margin(weights) for weights in round_weights]
        round_winner_weights = [
            weights.gather(dim=-1, index=winners.unsqueeze(-1)).squeeze(-1)
            for weights, winners in zip(round_weights, round_winners)
        ]

        heap = self._trace_candidate_heaps[layer_index]
        for batch_index in range(batch_size):
            for position_index in range(seq_len):
                token_id = int(token_ids[batch_index, position_index].item())
                rounds: list[MiniMoeTokenRoundTrace] = []
                total_adjustment_l1 = 0.0
                for round_index, weights in enumerate(round_weights):
                    route_adjustment_l1 = None
                    if round_index > 0:
                        route_adjustment_l1 = float(
                            (weights[batch_index, position_index] - round_weights[round_index - 1][batch_index, position_index])
                            .abs()
                            .sum()
                            .item()
                        )
                        total_adjustment_l1 += route_adjustment_l1
                    rounds.append(
                        MiniMoeTokenRoundTrace(
                            round_index=round_index + 1,
                            winner_expert_id=int(round_winners[round_index][batch_index, position_index].item()),
                            winner_weight=float(round_winner_weights[round_index][batch_index, position_index].item()),
                            route_entropy_bits=float(round_entropies[round_index][batch_index, position_index].item()),
                            winner_margin=float(round_margins[round_index][batch_index, position_index].item()),
                            route_adjustment_l1=route_adjustment_l1,
                        )
                    )

                first_winner = rounds[0].winner_expert_id
                final_winner = rounds[-1].winner_expert_id
                rerouted = first_winner != final_winner
                margin_delta = abs(rounds[-1].winner_margin - rounds[0].winner_margin)
                trace = MiniMoeTokenRouteTrace(
                    layer_index=layer_index,
                    forward_pass_index=self._current_forward_pass_index,
                    batch_index=batch_index,
                    position_index=position_index,
                    token_id=token_id,
                    token_label=_byte_token_label(token_id),
                    rerouted=rerouted,
                    first_winner_expert_id=first_winner,
                    final_winner_expert_id=final_winner,
                    total_adjustment_l1=total_adjustment_l1,
                    rounds=rounds,
                )
                self._trace_serial += 1
                score = (
                    1.0 if rerouted else 0.0,
                    total_adjustment_l1,
                    margin_delta,
                    -float(self._trace_serial),
                )
                candidate = _TokenTraceCandidate(score=score, trace=trace)
                if len(heap) < self._max_token_route_traces_per_layer:
                    heapq.heappush(heap, candidate)
                elif candidate.score > heap[0].score:
                    heapq.heapreplace(heap, candidate)

    def _record_route_plan_reduced(self, layer_index: int, route_plan: RoutePlan) -> None:
        route_weights = route_plan.expert_weights.detach().float()
        token_count = int(route_weights.shape[0] * route_weights.shape[1])
        self._sampled_tokens += token_count
        self._round_count = max(self._round_count, len(route_plan.round_summaries))

        route_entropy = _route_entropy_bits(route_weights)
        winner_margin = _winner_margin(route_weights)
        winners = route_weights.argmax(dim=-1)
        self._route_entropy_sum += float(route_entropy.sum().item())
        self._winner_margin_sum += float(winner_margin.sum().item())

        mean_expert_weight_sums = route_weights.sum(dim=(0, 1)).detach().cpu().tolist()
        winner_counts = torch.bincount(
            winners.reshape(-1),
            minlength=self._expert_count,
        ).detach().cpu().tolist()
        for expert_index in range(self._expert_count):
            self._mean_expert_weight_sums[expert_index] += float(mean_expert_weight_sums[expert_index])
            self._winner_counts[expert_index] += int(winner_counts[expert_index])

        layer = self._layers.setdefault(
            layer_index,
            _LayerAccumulator(
                selection_counts=[0 for _ in range(self._expert_count)],
                selected_weight_sums=[0.0 for _ in range(self._expert_count)],
                dispatch_counts=[0 for _ in range(self._expert_count)],
            ),
        )
        layer.sampled_tokens += token_count
        layer.route_entropy_sum += float(route_entropy.sum().item())

        first_round_winners: torch.Tensor | None = None
        previous_weights: torch.Tensor | None = None
        previous_winners: torch.Tensor | None = None
        for round_index, round_summary in enumerate(route_plan.round_summaries):
            round_weights = round_summary.expert_weights.detach().float()
            round_entropy = _route_entropy_bits(round_weights)
            round_margin = _winner_margin(round_weights)
            round_winners = round_weights.argmax(dim=-1)
            round_key = (layer_index, round_index)
            accumulator = self._rounds.setdefault(round_key, _RoundAccumulator())
            accumulator.sampled_tokens += token_count
            accumulator.entropy_sum += float(round_entropy.sum().item())
            accumulator.winner_margin_sum += float(round_margin.sum().item())
            accumulator.applied_token_fraction_sum += (
                round_summary.applied_token_fraction * token_count
            )
            if round_summary.mean_gate_probability is not None:
                accumulator.gate_probability_sum += (
                    round_summary.mean_gate_probability * token_count
                )
                accumulator.gate_probability_samples += token_count

            if round_index == 0:
                first_round_winners = round_winners
            if previous_weights is not None and previous_winners is not None:
                adjustment_l1_sum = float((round_weights - previous_weights).abs().sum().item())
                rerouted_count = int((round_winners != previous_winners).sum().item())
                accumulator.adjustment_l1_sum += adjustment_l1_sum
                accumulator.rerouted_token_count += rerouted_count
                accumulator.has_adjustment = True
                self._round_adjustment_sums[round_index - 1] += adjustment_l1_sum
                self._round_adjustment_tokens[round_index - 1] += token_count

            previous_weights = round_weights
            previous_winners = round_winners

        if first_round_winners is not None:
            layer.reroute_token_count += int((winners != first_round_winners).sum().item())


def _byte_token_label(token_id: int) -> str:
    if token_id == 256:
        return "<PAD>"
    if token_id == 9:
        return "\\t"
    if token_id == 10:
        return "\\n"
    if token_id == 13:
        return "\\r"
    if 32 <= token_id <= 126:
        return chr(token_id)
    if 0 <= token_id <= 255:
        return f"0x{token_id:02x}"
    return f"<tok:{token_id}>"


class MiniMoeRouter(nn.Module):
    def plan(self, hidden: torch.Tensor) -> RoutePlan:  # pragma: no cover - abstract
        raise NotImplementedError


class OneShotRouter(MiniMoeRouter):
    def __init__(self, hidden_dim: int, experts_per_block: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, experts_per_block)
        self._compiled_mode: str | None = None
        self._plan_impl = self._plan_eager

    def _plan_eager(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expert_logits = self.projection(hidden)
        expert_weights = torch.softmax(expert_logits, dim=-1)
        return expert_logits, expert_weights

    def plan(self, hidden: torch.Tensor) -> RoutePlan:
        expert_logits, expert_weights = self._plan_impl(hidden)
        round_summary = RouteRoundSummary(
            expert_logits=expert_logits,
            expert_weights=expert_weights,
        )
        return RoutePlan(
            expert_logits=expert_logits,
            expert_weights=expert_weights,
            round_summaries=(round_summary,),
        )

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        del primitive_runtime_backend
        if compile_mode == self._compiled_mode:
            return
        if compile_mode is None:
            self._plan_impl = self._plan_eager
        else:
            self._plan_impl = torch.compile(self._plan_eager, mode=compile_mode)
        self._compiled_mode = compile_mode


class RecurrentPreExpertRouter(MiniMoeRouter):
    def __init__(
        self,
        hidden_dim: int,
        experts_per_block: int,
        *,
        state_dim: int,
        round_count: int,
        gate: RecurrentRoundGateSpec,
        execution_strategy: RecurrentRoundExecutionStrategy,
        round2_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.round_count = round_count
        self.gate = gate
        self.execution_strategy = execution_strategy
        self.round2_enabled = round2_enabled
        self.token_state_projection = nn.Linear(hidden_dim, state_dim)
        self.token_route_projection = nn.Linear(hidden_dim, experts_per_block)
        self.state_route_projection = nn.Linear(state_dim, experts_per_block, bias=False)
        self.route_feedback_projection = nn.Linear(experts_per_block, state_dim, bias=False)
        self.reset_gate_projection = nn.Linear(state_dim, state_dim)
        self.update_gate_projection = nn.Linear(state_dim, state_dim)
        self.candidate_input_projection = nn.Linear(state_dim, state_dim)
        self.candidate_state_projection = nn.Linear(state_dim, state_dim, bias=False)
        self.learned_gate_network: nn.Sequential | None = None
        if self.gate.kind in {
            RecurrentRoundGateKind.LEARNED_SCORE_ABOVE,
            RecurrentRoundGateKind.LEARNED_SCORE_TOP_FRACTION,
        }:
            assert self.gate.learned_hidden_dim is not None
            assert self.gate.learned_prior_probability is not None
            gate_input_dim = state_dim + experts_per_block + 3
            self.learned_gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, self.gate.learned_hidden_dim),
                nn.Tanh(),
                nn.Linear(self.gate.learned_hidden_dim, 1),
            )
            final_linear = self.learned_gate_network[-1]
            assert isinstance(final_linear, nn.Linear)
            with torch.no_grad():
                final_linear.weight.zero_()
                prior_logit = math.log(
                    self.gate.learned_prior_probability
                    / (1.0 - self.gate.learned_prior_probability)
                )
                final_linear.bias.fill_(prior_logit)
        self._compiled_mode: str | None = None
        self._plan_tensors_impl = self._plan_tensors_eager
        self._gated_round_update_impl = self._apply_gated_round_update_eager

    def plan(self, hidden: torch.Tensor) -> RoutePlan:
        (
            expert_logits,
            expert_weights,
            round_logits,
            round_weights,
            applied_token_fractions,
            mean_gate_probabilities,
            auxiliary_loss,
        ) = self._plan_tensors_impl(hidden)
        round_summaries = tuple(
            RouteRoundSummary(
                expert_logits=round_logits[round_index],
                expert_weights=round_weights[round_index],
                applied_token_fraction=float(applied_token_fractions[round_index].detach().item()),
                mean_gate_probability=(
                    None
                    if torch.isnan(mean_gate_probabilities[round_index]).item()
                    else float(mean_gate_probabilities[round_index].detach().item())
                ),
            )
            for round_index in range(len(round_logits))
        )
        return RoutePlan(
            expert_logits=expert_logits,
            expert_weights=expert_weights,
            round_summaries=round_summaries,
            auxiliary_loss=auxiliary_loss,
        )

    def _plan_tensors_eager(
        self,
        hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        effective_round_count = self.round_count if self.round2_enabled else 1
        batch_size, seq_len, _ = hidden.shape
        token_state = torch.tanh(self.token_state_projection(hidden))
        pooled_token_state = token_state.mean(dim=1, keepdim=True)
        base_expert_logits = self.token_route_projection(hidden)
        controller_state = torch.zeros(
            batch_size,
            1,
            pooled_token_state.shape[-1],
            dtype=hidden.dtype,
            device=hidden.device,
        )
        round_logits: list[torch.Tensor] = []
        round_weights: list[torch.Tensor] = []
        applied_token_fractions: list[torch.Tensor] = []
        mean_gate_probabilities: list[torch.Tensor] = []
        previous_logits: torch.Tensor | None = None
        previous_weights: torch.Tensor | None = None
        auxiliary_losses: list[torch.Tensor] = []
        nan_gate_probability = hidden.new_full((), float("nan"))

        for round_index in range(effective_round_count):
            if round_index == 0:
                expert_logits = base_expert_logits
                expert_weights = torch.softmax(expert_logits, dim=-1)
                applied_token_fraction = hidden.new_tensor(1.0)
                mean_gate_probability = nan_gate_probability
            else:
                assert previous_logits is not None
                assert previous_weights is not None
                active_mask, gate_values, mean_gate_probability, auxiliary_loss = self._resolve_gate(
                    previous_weights=previous_weights,
                    token_state=token_state,
                )
                if auxiliary_loss is not None:
                    auxiliary_losses.append(auxiliary_loss)
                applied_token_fraction = active_mask.float().mean()
                if active_mask.any() or self.gate.kind in {
                    RecurrentRoundGateKind.LEARNED_SCORE_ABOVE,
                    RecurrentRoundGateKind.LEARNED_SCORE_TOP_FRACTION,
                }:
                    state_route_bias = self.state_route_projection(controller_state)
                    candidate_logits = base_expert_logits + state_route_bias
                    if active_mask.all() and self.gate.kind not in {
                        RecurrentRoundGateKind.LEARNED_SCORE_ABOVE,
                        RecurrentRoundGateKind.LEARNED_SCORE_TOP_FRACTION,
                    }:
                        expert_logits = candidate_logits
                        expert_weights = torch.softmax(expert_logits, dim=-1)
                    else:
                        expert_logits, expert_weights = self._apply_gated_round_update(
                            previous_logits=previous_logits,
                            previous_weights=previous_weights,
                            candidate_logits=candidate_logits,
                            gate_values=gate_values,
                            active_mask=active_mask,
                        )
                else:
                    expert_logits = previous_logits
                    expert_weights = previous_weights
            round_logits.append(expert_logits)
            round_weights.append(expert_weights)
            applied_token_fractions.append(applied_token_fraction)
            mean_gate_probabilities.append(
                nan_gate_probability
                if mean_gate_probability is None
                else mean_gate_probability
            )
            pooled_feedback = self.route_feedback_projection(expert_weights.mean(dim=1, keepdim=True))
            state_input = pooled_token_state + pooled_feedback
            reset_gate = torch.sigmoid(self.reset_gate_projection(state_input))
            update_gate = torch.sigmoid(self.update_gate_projection(state_input))
            candidate_state = torch.tanh(
                self.candidate_input_projection(state_input)
                + self.candidate_state_projection(reset_gate * controller_state)
            )
            controller_state = (1.0 - update_gate) * controller_state + update_gate * candidate_state
            previous_logits = expert_logits
            previous_weights = expert_weights

        final_logits = round_logits[-1]
        final_weights = round_weights[-1]
        return (
            final_logits,
            final_weights,
            tuple(round_logits),
            tuple(round_weights),
            torch.stack(applied_token_fractions),
            torch.stack(mean_gate_probabilities),
            (
                torch.stack(auxiliary_losses).mean()
                if auxiliary_losses
                else None
            ),
        )

    def _resolve_gate(
        self,
        *,
        previous_weights: torch.Tensor,
        token_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.gate.kind is RecurrentRoundGateKind.ALWAYS_ON:
            active_mask = torch.ones(
                previous_weights.shape[:2],
                dtype=torch.bool,
                device=previous_weights.device,
            )
            return active_mask, active_mask.float(), None, None
        if self.gate.kind is RecurrentRoundGateKind.WINNER_MARGIN_BELOW:
            assert self.gate.threshold is not None
            active_mask = _winner_margin(previous_weights) < self.gate.threshold
            return active_mask, active_mask.float(), None, None
        if self.gate.kind is RecurrentRoundGateKind.SCALED_WINNER_MARGIN_BELOW:
            assert self.gate.threshold is not None
            assert self.gate.reference_experts_per_block is not None
            threshold = _scaled_winner_margin_threshold(
                base_threshold=self.gate.threshold,
                expert_count=previous_weights.shape[-1],
                reference_experts_per_block=self.gate.reference_experts_per_block,
            )
            active_mask = _winner_margin(previous_weights) < threshold
            return active_mask, active_mask.float(), None, None
        if self.gate.kind is RecurrentRoundGateKind.TARGET_APPLIED_FRACTION:
            assert self.gate.target_applied_fraction is not None
            active_mask = _lowest_margin_fraction_mask(
                previous_weights,
                self.gate.target_applied_fraction,
            )
            return active_mask, active_mask.float(), None, None
        if self.gate.kind is RecurrentRoundGateKind.NORMALIZED_ENTROPY_ABOVE:
            assert self.gate.normalized_entropy_threshold is not None
            active_mask = (
                _normalized_route_entropy(previous_weights) > self.gate.normalized_entropy_threshold
            )
            return active_mask, active_mask.float(), None, None
        if self.gate.kind in {
            RecurrentRoundGateKind.LEARNED_SCORE_ABOVE,
            RecurrentRoundGateKind.LEARNED_SCORE_TOP_FRACTION,
        }:
            assert self.learned_gate_network is not None
            normalized_entropy = _normalized_route_entropy(previous_weights).unsqueeze(-1)
            winner_margin = _winner_margin(previous_weights).unsqueeze(-1)
            teacher_score = self._teacher_score(previous_weights).unsqueeze(-1)
            gate_features = torch.cat(
                [token_state, previous_weights, normalized_entropy, winner_margin, teacher_score],
                dim=-1,
            )
            gate_logits = self.learned_gate_network(gate_features).squeeze(-1)
            gate_probabilities = torch.sigmoid(gate_logits)
            auxiliary_loss = self._teacher_supervision_loss(
                gate_logits=gate_logits,
                gate_probabilities=gate_probabilities,
                previous_weights=previous_weights,
            )
            if self.gate.kind is RecurrentRoundGateKind.LEARNED_SCORE_ABOVE:
                active_mask = gate_probabilities > 0.5
            else:
                assert self.gate.target_applied_fraction is not None
                active_mask = _top_fraction_mask(
                    gate_probabilities,
                    self.gate.target_applied_fraction,
                )
            gate_values = (
                active_mask.float()
                + gate_probabilities
                - gate_probabilities.detach()
            )
            return active_mask, gate_values, gate_probabilities.mean(), auxiliary_loss
        raise ValueError(f"unsupported recurrent round gate kind: {self.gate.kind}")

    def _teacher_score(self, previous_weights: torch.Tensor) -> torch.Tensor:
        if self.gate.teacher_kind is LearnedGateTeacherKind.BLENDED_UNCERTAINTY:
            return _teacher_ambiguity_score(previous_weights)
        raise ValueError(f"unsupported learned gate teacher kind: {self.gate.teacher_kind}")

    def _teacher_supervision_loss(
        self,
        *,
        gate_logits: torch.Tensor,
        gate_probabilities: torch.Tensor,
        previous_weights: torch.Tensor,
    ) -> torch.Tensor:
        assert self.gate.teacher_supervision_weight is not None
        teacher_score = self._teacher_score(previous_weights).detach()
        if self.gate.kind is RecurrentRoundGateKind.LEARNED_SCORE_TOP_FRACTION:
            assert self.gate.target_applied_fraction is not None
            teacher_target = _top_fraction_mask(
                teacher_score,
                self.gate.target_applied_fraction,
            ).float()
            teacher_loss = nn.functional.binary_cross_entropy_with_logits(
                gate_logits,
                teacher_target,
            )
        elif self.gate.kind is RecurrentRoundGateKind.LEARNED_SCORE_ABOVE:
            teacher_loss = nn.functional.binary_cross_entropy(
                gate_probabilities,
                teacher_score,
            )
        else:
            raise ValueError(f"unsupported learned gate kind for teacher supervision: {self.gate.kind}")
        return teacher_loss * self.gate.teacher_supervision_weight

    def _apply_gated_round_update(
        self,
        *,
        previous_logits: torch.Tensor,
        previous_weights: torch.Tensor,
        candidate_logits: torch.Tensor,
        gate_values: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._gated_round_update_impl(
            previous_logits=previous_logits,
            previous_weights=previous_weights,
            candidate_logits=candidate_logits,
            gate_values=gate_values,
            active_mask=active_mask,
        )

    def _apply_gated_round_update_eager(
        self,
        *,
        previous_logits: torch.Tensor,
        previous_weights: torch.Tensor,
        candidate_logits: torch.Tensor,
        gate_values: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.execution_strategy is RecurrentRoundExecutionStrategy.DENSE_BLEND:
            candidate_weights = torch.softmax(candidate_logits, dim=-1)
            gate_values_expanded = gate_values.unsqueeze(-1)
            expert_logits = (
                gate_values_expanded * candidate_logits
                + (1.0 - gate_values_expanded) * previous_logits
            )
            expert_weights = (
                gate_values_expanded * candidate_weights
                + (1.0 - gate_values_expanded) * previous_weights
            )
            return expert_logits, expert_weights
        if self.execution_strategy is RecurrentRoundExecutionStrategy.MASKED_TOKEN_UPDATE:
            expert_logits = previous_logits.clone()
            expert_weights = previous_weights.clone()
            updated_logits = candidate_logits[active_mask]
            updated_weights = torch.softmax(updated_logits, dim=-1)
            expert_logits[active_mask] = updated_logits
            expert_weights[active_mask] = updated_weights
            return expert_logits, expert_weights
        raise ValueError(
            f"unsupported recurrent round execution strategy: {self.execution_strategy}"
        )

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        del primitive_runtime_backend
        if compile_mode == self._compiled_mode:
            return
        if compile_mode is None:
            self._plan_tensors_impl = self._plan_tensors_eager
            self._gated_round_update_impl = self._apply_gated_round_update_eager
        else:
            self._plan_tensors_impl = torch.compile(self._plan_tensors_eager, mode=compile_mode)
            self._gated_round_update_impl = torch.compile(
                self._apply_gated_round_update_eager,
                mode=compile_mode,
            )
        self._compiled_mode = compile_mode


class MoeDispatcher(nn.Module):
    def __init__(self, contract: ResolvedDispatchContract) -> None:
        super().__init__()
        self.contract = contract
        self._compiled_mode: str | None = None
        self._compile_impl = self._compile_eager

    def compile(self, layer_index: int, route_plan: RoutePlan) -> DispatchPlan:
        selected_expert_indices, selected_expert_weights = self._compile_impl(route_plan.expert_weights)
        return DispatchPlan(
            layer_index=layer_index,
            mode=self.contract.mode,
            selected_expert_indices=selected_expert_indices,
            selected_expert_weights=selected_expert_weights,
        )

    def _compile_eager(self, route_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, expert_count = route_weights.shape
        if self.contract.mode is MiniMoeDispatchMode.DENSE_DEBUG:
            selected_expert_indices = torch.arange(
                expert_count,
                device=route_weights.device,
            ).reshape(1, 1, expert_count).expand(batch_size, seq_len, expert_count)
            return selected_expert_indices, route_weights

        top_weights, top_indices = torch.topk(
            route_weights,
            k=self.contract.active_experts_per_token,
            dim=-1,
        )
        selected_expert_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(_ENTROPY_EPS)
        return top_indices, selected_expert_weights

    def dispatch(
        self,
        hidden: torch.Tensor,
        experts: nn.ModuleList,
        dispatch_plan: DispatchPlan,
    ) -> torch.Tensor:
        if (
            dispatch_plan.mode is MiniMoeDispatchMode.SPARSE_TOP_K
            and self.contract.execution_strategy
            is MiniMoeDispatchExecutionStrategy.TOKEN_PACKED_SPARSE
        ):
            return self._dispatch_sparse_topk(hidden, experts, dispatch_plan)

        expert_outputs = torch.stack([expert(hidden) for expert in experts], dim=-2)
        gather_index = dispatch_plan.selected_expert_indices.unsqueeze(-1).expand(
            *dispatch_plan.selected_expert_indices.shape,
            hidden.shape[-1],
        )
        selected_outputs = torch.gather(expert_outputs, -2, gather_index)
        return torch.sum(selected_outputs * dispatch_plan.selected_expert_weights.unsqueeze(-1), dim=-2)

    def _dispatch_sparse_topk(
        self,
        hidden: torch.Tensor,
        experts: nn.ModuleList,
        dispatch_plan: DispatchPlan,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden.shape
        token_count = batch_size * seq_len
        hidden_flat = hidden.reshape(token_count, hidden_dim)
        selected_indices = dispatch_plan.selected_expert_indices.reshape(
            token_count,
            dispatch_plan.selected_expert_indices.shape[-1],
        )
        selected_weights = dispatch_plan.selected_expert_weights.reshape(
            token_count,
            dispatch_plan.selected_expert_weights.shape[-1],
        )
        mixed_flat = torch.zeros_like(hidden_flat)

        if selected_indices.shape[-1] == 1:
            top1_indices = selected_indices[:, 0]
            top1_weights = selected_weights[:, 0]
            for expert_index, expert in enumerate(experts):
                token_positions = torch.nonzero(top1_indices == expert_index, as_tuple=False).flatten()
                if token_positions.numel() == 0:
                    continue
                expert_hidden = hidden_flat.index_select(0, token_positions)
                expert_output = expert(expert_hidden)
                expert_weight = top1_weights.index_select(0, token_positions).unsqueeze(-1)
                mixed_flat.index_copy_(0, token_positions, expert_output * expert_weight)
            return mixed_flat.reshape(batch_size, seq_len, hidden_dim)

        for expert_index, expert in enumerate(experts):
            expert_mask = selected_indices == expert_index
            if not expert_mask.any():
                continue
            token_mask = expert_mask.any(dim=-1)
            token_positions = torch.nonzero(token_mask, as_tuple=False).flatten()
            expert_hidden = hidden_flat.index_select(0, token_positions)
            expert_output = expert(expert_hidden)
            expert_weight = selected_weights[expert_mask].unsqueeze(-1)
            mixed_flat.index_add_(0, token_positions, expert_output * expert_weight)

        return mixed_flat.reshape(batch_size, seq_len, hidden_dim)

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        del primitive_runtime_backend
        if compile_mode == self._compiled_mode:
            return
        if compile_mode is None:
            self._compile_impl = self._compile_eager
        else:
            self._compile_impl = torch.compile(self._compile_eager, mode=compile_mode)
        self._compiled_mode = compile_mode


SparseTopKDispatcher = MoeDispatcher
OneShotTopKRouter = OneShotRouter


class SparseMoeFeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        expert_ffn_multiplier: int,
        experts_per_block: int,
        *,
        router: MiniMoeRouter,
        dispatcher: MoeDispatcher,
        observability_sink: MiniMoeObservabilitySink | None,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.router = router
        self.dispatcher = dispatcher
        self.observability_sink = observability_sink or NullMiniMoeObservabilitySink()
        self.layer_index = layer_index
        self._last_auxiliary_loss: torch.Tensor | None = None
        self.experts = nn.ModuleList(
            [
                PositionWiseFeedForward(hidden_dim, hidden_dim * expert_ffn_multiplier)
                for _ in range(experts_per_block)
            ]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        route_plan = self.router.plan(hidden)
        dispatch_plan = self.dispatcher.compile(self.layer_index, route_plan)
        mixed = self.dispatcher.dispatch(hidden, self.experts, dispatch_plan)
        self._last_auxiliary_loss = route_plan.auxiliary_loss
        self.observability_sink.record_route_plan(self.layer_index, route_plan)
        self.observability_sink.record_dispatch_plan(dispatch_plan)
        return mixed

    def pop_auxiliary_loss(self) -> torch.Tensor | None:
        auxiliary_loss = self._last_auxiliary_loss
        self._last_auxiliary_loss = None
        return auxiliary_loss

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        for module in self.experts:
            configure = getattr(module, "configure_runtime_policy", None)
            if callable(configure):
                configure(
                    compile_mode=compile_mode,
                    primitive_runtime_backend=primitive_runtime_backend,
                )


def _build_router(surface_spec: MiniMoeSurfaceSpec, layer_index: int) -> MiniMoeRouter:
    hidden_dim = surface_spec.architecture.backbone.hidden_dim
    experts_per_block = surface_spec.architecture.moe.experts_per_block
    if surface_spec.architecture.router.kind == "one_shot":
        return OneShotRouter(hidden_dim=hidden_dim, experts_per_block=experts_per_block)
    recurrent_spec = surface_spec.architecture.router.recurrent_pre_expert
    if recurrent_spec is None:
        raise ValueError("recurrent_pre_expert router spec must be present for recurrent routing")
    round2_layers = set(surface_spec.architecture.resolved_round2_layer_indices())
    return RecurrentPreExpertRouter(
        hidden_dim=hidden_dim,
        experts_per_block=experts_per_block,
        state_dim=recurrent_spec.state_dim,
        round_count=recurrent_spec.round_count,
        gate=recurrent_spec.gate,
        execution_strategy=recurrent_spec.execution_strategy,
        round2_enabled=layer_index in round2_layers,
    )


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
        dispatch_contract = surface_spec.resolved_dispatch()
        sink = observability_sink or NullMiniMoeObservabilitySink()
        self.observability_sink = sink
        self._last_auxiliary_loss: torch.Tensor | None = None
        moe_layers = set(layout.moe_layers)
        for layer_index in range(surface_spec.architecture.backbone.total_layers):
            if layer_index in moe_layers:
                ffn_module = SparseMoeFeedForward(
                    hidden_dim=surface_spec.architecture.backbone.hidden_dim,
                    expert_ffn_multiplier=surface_spec.architecture.moe.expert_ffn_multiplier,
                    experts_per_block=surface_spec.architecture.moe.experts_per_block,
                    router=_build_router(surface_spec, layer_index),
                    dispatcher=MoeDispatcher(dispatch_contract),
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

    @property
    def model_label(self) -> str:
        return f"mini_moe_{self.surface_spec.architecture.label.replace('-', '_')}"

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.observability_sink.record_input_batch(input_ids)
        self._last_auxiliary_loss = None
        auxiliary_losses: list[torch.Tensor] = []
        hidden = self.embedding(input_ids)
        mask = local_causal_attention_bias(
            input_ids.shape[1],
            self.surface_spec.architecture.backbone.local_window,
            input_ids.device,
            hidden.dtype,
        )
        for block in self.blocks:
            hidden = block(hidden, mask)
            if isinstance(block, AuxiliaryLossProvider):
                block_auxiliary_loss = block.pop_auxiliary_loss()
                if block_auxiliary_loss is not None:
                    auxiliary_losses.append(block_auxiliary_loss)
        if auxiliary_losses:
            self._last_auxiliary_loss = torch.stack(auxiliary_losses).mean()
        return self.output(hidden)

    def pop_auxiliary_loss(self) -> torch.Tensor | None:
        auxiliary_loss = self._last_auxiliary_loss
        self._last_auxiliary_loss = None
        return auxiliary_loss

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        for block in self.blocks:
            configure = getattr(block, "configure_runtime_policy", None)
            if callable(configure):
                configure(
                    compile_mode=compile_mode,
                    primitive_runtime_backend=primitive_runtime_backend,
                )


def build_mini_moe_model(
    surface_spec: MiniMoeSurfaceSpec,
    *,
    observability_sink: MiniMoeObservabilitySink | None = None,
) -> MiniMoeBackboneModel:
    return MiniMoeBackboneModel(surface_spec, observability_sink=observability_sink)
