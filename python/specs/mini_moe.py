from __future__ import annotations

from dataclasses import dataclass

from python.specs.common import StringEnum, ValidationError, ensure_positive
from python.specs.runtime import RuntimeOptimizationFamily


def transfer_round2_layer_indices_by_depth_fraction(
    *,
    source_layer_indices: tuple[int, ...],
    source_total_layers: int,
    target_total_layers: int,
) -> tuple[int, ...]:
    ensure_positive(source_total_layers, "mini_moe.transfer.source_total_layers")
    ensure_positive(target_total_layers, "mini_moe.transfer.target_total_layers")
    transferred: list[int] = []
    for layer_index in source_layer_indices:
        if layer_index < 0 or layer_index >= source_total_layers:
            raise ValidationError(
                f"mini_moe.transfer.source_layer_indices contains out-of-range layer {layer_index}"
            )
        transferred_index = round((layer_index / source_total_layers) * target_total_layers)
        transferred.append(min(target_total_layers - 1, max(0, transferred_index)))
    return tuple(sorted(set(transferred)))


def contiguous_layer_bands(layer_indices: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if not layer_indices:
        return ()
    if tuple(sorted(set(layer_indices))) != layer_indices:
        raise ValidationError("mini_moe.transfer.layer_indices must be sorted, deduplicated, and ascending")
    bands: list[list[int]] = [[layer_indices[0]]]
    for layer_index in layer_indices[1:]:
        if layer_index == bands[-1][-1] + 1:
            bands[-1].append(layer_index)
        else:
            bands.append([layer_index])
    return tuple(tuple(band) for band in bands)


def transfer_round2_layer_bands_by_anchor_fill(
    *,
    source_layer_indices: tuple[int, ...],
    source_total_layers: int,
    target_total_layers: int,
) -> tuple[int, ...]:
    transferred_points = transfer_round2_layer_indices_by_depth_fraction(
        source_layer_indices=source_layer_indices,
        source_total_layers=source_total_layers,
        target_total_layers=target_total_layers,
    )
    source_to_target = {
        source_layer: target_layer
        for source_layer, target_layer in zip(source_layer_indices, transferred_points)
    }
    transferred_layers: set[int] = set()
    for band in contiguous_layer_bands(source_layer_indices):
        band_targets = [source_to_target[layer_index] for layer_index in band]
        transferred_layers.update(range(min(band_targets), max(band_targets) + 1))
    return tuple(sorted(transferred_layers))


def transfer_round2_layer_bands_by_scaled_span(
    *,
    source_layer_indices: tuple[int, ...],
    source_total_layers: int,
    target_total_layers: int,
) -> tuple[int, ...]:
    ensure_positive(source_total_layers, "mini_moe.transfer.source_total_layers")
    ensure_positive(target_total_layers, "mini_moe.transfer.target_total_layers")
    transferred_layers: set[int] = set()
    for band in contiguous_layer_bands(source_layer_indices):
        start_index = band[0]
        end_index = band[-1]
        if start_index < 0 or end_index >= source_total_layers:
            raise ValidationError(
                f"mini_moe.transfer.source_layer_indices contains out-of-range band ({start_index}, {end_index})"
            )
        start_target = round((start_index / source_total_layers) * target_total_layers)
        end_exclusive_target = round(((end_index + 1) / source_total_layers) * target_total_layers)
        start_target = min(target_total_layers - 1, max(0, start_target))
        end_exclusive_target = min(target_total_layers, max(start_target + 1, end_exclusive_target))
        transferred_layers.update(range(start_target, end_exclusive_target))
    return tuple(sorted(transferred_layers))


class MiniMoeDispatchMode(StringEnum):
    SPARSE_TOP_K = "sparse_top_k"
    DENSE_DEBUG = "dense_debug"


class MiniMoeDispatchExecutionStrategy(StringEnum):
    DENSE_GATHER = "dense_gather"
    TOKEN_PACKED_SPARSE = "token_packed_sparse"


class TieBreakPolicy(StringEnum):
    LOWEST_INDEX = "lowest_index"


class DispatchCapacityPolicy(StringEnum):
    UNLIMITED = "unlimited"


class MiniMoePreset(StringEnum):
    PHASE1_REFERENCE = "phase1_reference"
    PHASE1_RECURRENT = "phase1_recurrent"
    PHASE1_RECURRENT_GATED = "phase1_recurrent_gated"


class MiniMoeLayerSchedulePreset(StringEnum):
    ALL_LAYERS = "all_layers"
    ALTERNATING_FROM_ZERO = "alternating_from_zero"
    ALTERNATING_FROM_ONE = "alternating_from_one"


class MiniMoeLayerScheduleKind(StringEnum):
    ALL_LAYERS = "all_layers"
    EVERY_N = "every_n"
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class MiniMoeLayerSchedule:
    kind: MiniMoeLayerScheduleKind
    every_n: int | None = None
    explicit_layers: tuple[int, ...] = ()

    def validate(self, total_layers: int) -> None:
        ensure_positive(total_layers, "mini_moe.total_layers")
        if self.kind is MiniMoeLayerScheduleKind.ALL_LAYERS:
            return
        if self.kind is MiniMoeLayerScheduleKind.EVERY_N:
            if self.every_n is None:
                raise ValidationError("MiniMoeLayerSchedule(kind=every_n) must set every_n")
            ensure_positive(self.every_n, "mini_moe.layer_schedule.every_n")
            return
        if self.kind is MiniMoeLayerScheduleKind.EXPLICIT:
            if not self.explicit_layers:
                raise ValidationError(
                    "MiniMoeLayerSchedule(kind=explicit) must contain at least one layer"
                )
            for layer_index in self.explicit_layers:
                if layer_index < 0 or layer_index >= total_layers:
                    raise ValidationError(
                        f"mini_moe.layer_schedule explicit layer {layer_index} is out of range for total_layers={total_layers}"
                    )
            return
        raise ValidationError(f"unsupported MiniMoeLayerSchedule kind: {self.kind}")

    def resolve(self, total_layers: int) -> "ResolvedMiniMoeLayout":
        self.validate(total_layers)
        if self.kind is MiniMoeLayerScheduleKind.ALL_LAYERS:
            moe_layers = list(range(total_layers))
        elif self.kind is MiniMoeLayerScheduleKind.EVERY_N:
            moe_layers = list(range(0, total_layers, self.every_n))
        else:
            moe_layers = sorted(set(self.explicit_layers))
        dense_layers = [layer for layer in range(total_layers) if layer not in set(moe_layers)]
        return ResolvedMiniMoeLayout(moe_layers=tuple(moe_layers), dense_layers=tuple(dense_layers))

    @classmethod
    def from_preset(cls, preset: MiniMoeLayerSchedulePreset, total_layers: int) -> "MiniMoeLayerSchedule":
        if preset is MiniMoeLayerSchedulePreset.ALL_LAYERS:
            return cls(kind=MiniMoeLayerScheduleKind.ALL_LAYERS)
        if preset is MiniMoeLayerSchedulePreset.ALTERNATING_FROM_ZERO:
            return cls(
                kind=MiniMoeLayerScheduleKind.EXPLICIT,
                explicit_layers=tuple(range(0, total_layers, 2)),
            )
        return cls(
            kind=MiniMoeLayerScheduleKind.EXPLICIT,
            explicit_layers=tuple(range(1, total_layers, 2)),
        )


@dataclass(frozen=True)
class ResolvedMiniMoeLayout:
    moe_layers: tuple[int, ...]
    dense_layers: tuple[int, ...]

    def validate(self) -> None:
        if not self.moe_layers:
            raise ValidationError("resolved mini-moe layout must contain at least one moe layer")
        if tuple(sorted(set(self.moe_layers))) != self.moe_layers:
            raise ValidationError(
                "resolved mini-moe moe_layers must be sorted, deduplicated, and ascending"
            )


@dataclass(frozen=True)
class MiniMoeBackboneSpec:
    vocab_size: int
    hidden_dim: int
    head_count: int
    total_layers: int
    local_window: int
    ffn_multiplier: int

    def validate(self) -> None:
        ensure_positive(self.vocab_size, "mini_moe.backbone.vocab_size")
        ensure_positive(self.hidden_dim, "mini_moe.backbone.hidden_dim")
        ensure_positive(self.head_count, "mini_moe.backbone.head_count")
        ensure_positive(self.total_layers, "mini_moe.backbone.total_layers")
        ensure_positive(self.local_window, "mini_moe.backbone.local_window")
        ensure_positive(self.ffn_multiplier, "mini_moe.backbone.ffn_multiplier")
        if self.hidden_dim % self.head_count != 0:
            raise ValidationError(
                f"mini_moe.backbone.hidden_dim {self.hidden_dim} must be divisible by head_count {self.head_count}"
            )


@dataclass(frozen=True)
class MiniMoeStackSpec:
    experts_per_block: int
    active_experts_per_token: int
    moe_layer_schedule: MiniMoeLayerSchedule
    expert_ffn_multiplier: int
    load_balance_loss_weight: float

    def validate(self, total_layers: int) -> None:
        if self.experts_per_block < 2:
            raise ValidationError("mini_moe.stack.experts_per_block must be at least 2")
        if self.active_experts_per_token < 1:
            raise ValidationError("mini_moe.stack.active_experts_per_token must be at least 1")
        if self.active_experts_per_token > self.experts_per_block:
            raise ValidationError(
                "mini_moe.stack.active_experts_per_token must not exceed experts_per_block"
            )
        ensure_positive(self.expert_ffn_multiplier, "mini_moe.stack.expert_ffn_multiplier")
        if self.load_balance_loss_weight < 0:
            raise ValidationError("mini_moe.stack.load_balance_loss_weight must be non-negative")
        self.moe_layer_schedule.validate(total_layers)


@dataclass(frozen=True)
class OneShotRouterSpec:
    pass


class RecurrentRoundGateKind(StringEnum):
    ALWAYS_ON = "always_on"
    WINNER_MARGIN_BELOW = "winner_margin_below"
    SCALED_WINNER_MARGIN_BELOW = "scaled_winner_margin_below"
    TARGET_APPLIED_FRACTION = "target_applied_fraction"
    NORMALIZED_ENTROPY_ABOVE = "normalized_entropy_above"
    LEARNED_SCORE_ABOVE = "learned_score_above"
    LEARNED_SCORE_TOP_FRACTION = "learned_score_top_fraction"


class RecurrentRoundExecutionStrategy(StringEnum):
    DENSE_BLEND = "dense_blend"
    MASKED_TOKEN_UPDATE = "masked_token_update"


class LearnedGateTeacherKind(StringEnum):
    BLENDED_UNCERTAINTY = "blended_uncertainty"


@dataclass(frozen=True)
class RecurrentRoundGateSpec:
    kind: RecurrentRoundGateKind = RecurrentRoundGateKind.ALWAYS_ON
    threshold: float | None = None
    reference_experts_per_block: int | None = None
    target_applied_fraction: float | None = None
    normalized_entropy_threshold: float | None = None
    learned_hidden_dim: int | None = None
    learned_prior_probability: float | None = None
    teacher_kind: LearnedGateTeacherKind | None = None
    teacher_supervision_weight: float | None = None

    def validate(self) -> None:
        if self.kind is RecurrentRoundGateKind.ALWAYS_ON:
            if (
                self.threshold is not None
                or self.reference_experts_per_block is not None
                or self.target_applied_fraction is not None
                or self.normalized_entropy_threshold is not None
                or self.learned_hidden_dim is not None
                or self.learned_prior_probability is not None
                or self.teacher_kind is not None
                or self.teacher_supervision_weight is not None
            ):
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=always_on) must not set threshold, reference_experts_per_block, target_applied_fraction, normalized_entropy_threshold, learned_hidden_dim, learned_prior_probability, teacher_kind, or teacher_supervision_weight"
                )
            return
        if self.kind is RecurrentRoundGateKind.WINNER_MARGIN_BELOW:
            if self.threshold is None:
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=winner_margin_below) must set threshold"
                )
            if (
                self.reference_experts_per_block is not None
                or
                self.target_applied_fraction is not None
                or self.normalized_entropy_threshold is not None
                or self.learned_hidden_dim is not None
                or self.learned_prior_probability is not None
                or self.teacher_kind is not None
                or self.teacher_supervision_weight is not None
            ):
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=winner_margin_below) must not set reference_experts_per_block, target_applied_fraction, normalized_entropy_threshold, learned_hidden_dim, learned_prior_probability, teacher_kind, or teacher_supervision_weight"
                )
            if not 0.0 < self.threshold < 1.0:
                raise ValidationError(
                    "mini_moe.router.gate.threshold must be in the open interval (0, 1)"
                )
            return
        if self.kind is RecurrentRoundGateKind.SCALED_WINNER_MARGIN_BELOW:
            if self.threshold is None:
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=scaled_winner_margin_below) must set threshold"
                )
            if self.reference_experts_per_block is None:
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=scaled_winner_margin_below) must set reference_experts_per_block"
                )
            if (
                self.target_applied_fraction is not None
                or self.normalized_entropy_threshold is not None
                or self.learned_hidden_dim is not None
                or self.learned_prior_probability is not None
                or self.teacher_kind is not None
                or self.teacher_supervision_weight is not None
            ):
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=scaled_winner_margin_below) must not set target_applied_fraction, normalized_entropy_threshold, learned_hidden_dim, learned_prior_probability, teacher_kind, or teacher_supervision_weight"
                )
            if not 0.0 < self.threshold < 1.0:
                raise ValidationError(
                    "mini_moe.router.gate.threshold must be in the open interval (0, 1)"
                )
            if self.reference_experts_per_block < 2:
                raise ValidationError(
                    "mini_moe.router.gate.reference_experts_per_block must be at least 2"
                )
            return
        if self.kind is RecurrentRoundGateKind.TARGET_APPLIED_FRACTION:
            if self.target_applied_fraction is None:
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=target_applied_fraction) must set target_applied_fraction"
                )
            if (
                self.threshold is not None
                or self.reference_experts_per_block is not None
                or self.normalized_entropy_threshold is not None
                or self.learned_hidden_dim is not None
                or self.learned_prior_probability is not None
                or self.teacher_kind is not None
                or self.teacher_supervision_weight is not None
            ):
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=target_applied_fraction) must not set threshold, reference_experts_per_block, normalized_entropy_threshold, learned_hidden_dim, learned_prior_probability, teacher_kind, or teacher_supervision_weight"
                )
            if not 0.0 < self.target_applied_fraction <= 1.0:
                raise ValidationError(
                    "mini_moe.router.gate.target_applied_fraction must be in the interval (0, 1]"
                )
            return
        if self.kind is RecurrentRoundGateKind.NORMALIZED_ENTROPY_ABOVE:
            if self.normalized_entropy_threshold is None:
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=normalized_entropy_above) must set normalized_entropy_threshold"
                )
            if (
                self.threshold is not None
                or self.reference_experts_per_block is not None
                or self.target_applied_fraction is not None
                or self.learned_hidden_dim is not None
                or self.learned_prior_probability is not None
                or self.teacher_kind is not None
                or self.teacher_supervision_weight is not None
            ):
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=normalized_entropy_above) must not set threshold, reference_experts_per_block, target_applied_fraction, learned_hidden_dim, learned_prior_probability, teacher_kind, or teacher_supervision_weight"
                )
            if not 0.0 < self.normalized_entropy_threshold <= 1.0:
                raise ValidationError(
                    "mini_moe.router.gate.normalized_entropy_threshold must be in the interval (0, 1]"
                )
            return
        if self.kind is RecurrentRoundGateKind.LEARNED_SCORE_ABOVE:
            if self.learned_hidden_dim is None:
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=learned_score_above) must set learned_hidden_dim"
                )
            if self.learned_prior_probability is None:
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=learned_score_above) must set learned_prior_probability"
                )
            if (
                self.threshold is not None
                or self.reference_experts_per_block is not None
                or self.target_applied_fraction is not None
                or self.normalized_entropy_threshold is not None
                or self.teacher_kind is None
                or self.teacher_supervision_weight is None
            ):
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=learned_score_above) must set teacher_kind and teacher_supervision_weight, and must not set threshold, reference_experts_per_block, target_applied_fraction, or normalized_entropy_threshold"
                )
            ensure_positive(self.learned_hidden_dim, "mini_moe.router.gate.learned_hidden_dim")
            if not 0.0 < self.learned_prior_probability < 1.0:
                raise ValidationError(
                    "mini_moe.router.gate.learned_prior_probability must be in the open interval (0, 1)"
                )
            if self.teacher_supervision_weight <= 0:
                raise ValidationError(
                    "mini_moe.router.gate.teacher_supervision_weight must be greater than zero"
                )
            return
        if self.kind is RecurrentRoundGateKind.LEARNED_SCORE_TOP_FRACTION:
            if self.learned_hidden_dim is None:
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=learned_score_top_fraction) must set learned_hidden_dim"
                )
            if self.learned_prior_probability is None:
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=learned_score_top_fraction) must set learned_prior_probability"
                )
            if self.target_applied_fraction is None:
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=learned_score_top_fraction) must set target_applied_fraction"
                )
            if (
                self.threshold is not None
                or self.reference_experts_per_block is not None
                or self.normalized_entropy_threshold is not None
                or self.teacher_kind is None
                or self.teacher_supervision_weight is None
            ):
                raise ValidationError(
                    "RecurrentRoundGateSpec(kind=learned_score_top_fraction) must set teacher_kind and teacher_supervision_weight, and must not set threshold, reference_experts_per_block, or normalized_entropy_threshold"
                )
            ensure_positive(self.learned_hidden_dim, "mini_moe.router.gate.learned_hidden_dim")
            if not 0.0 < self.learned_prior_probability < 1.0:
                raise ValidationError(
                    "mini_moe.router.gate.learned_prior_probability must be in the open interval (0, 1)"
                )
            if not 0.0 < self.target_applied_fraction <= 1.0:
                raise ValidationError(
                    "mini_moe.router.gate.target_applied_fraction must be in the interval (0, 1]"
                )
            if self.teacher_supervision_weight <= 0:
                raise ValidationError(
                    "mini_moe.router.gate.teacher_supervision_weight must be greater than zero"
                )
            return
        raise ValidationError(f"unsupported recurrent round gate kind: {self.kind}")


@dataclass(frozen=True)
class RecurrentPreExpertRouterSpec:
    round_count: int
    state_dim: int
    gate: RecurrentRoundGateSpec = RecurrentRoundGateSpec()
    execution_strategy: RecurrentRoundExecutionStrategy = RecurrentRoundExecutionStrategy.DENSE_BLEND
    round2_layer_indices: tuple[int, ...] | None = None

    def validate(self) -> None:
        ensure_positive(self.round_count, "mini_moe.router.round_count")
        ensure_positive(self.state_dim, "mini_moe.router.state_dim")
        self.gate.validate()
        if self.round2_layer_indices is not None and tuple(sorted(set(self.round2_layer_indices))) != self.round2_layer_indices:
            raise ValidationError(
                "mini_moe.router.round2_layer_indices must be sorted, deduplicated, and ascending"
            )
        if (
            self.gate.kind in {
                RecurrentRoundGateKind.LEARNED_SCORE_ABOVE,
                RecurrentRoundGateKind.LEARNED_SCORE_TOP_FRACTION,
            }
            and self.execution_strategy is not RecurrentRoundExecutionStrategy.DENSE_BLEND
        ):
            raise ValidationError(
                "mini_moe.router.execution_strategy=masked_token_update is not supported for learned score gating"
            )


@dataclass(frozen=True)
class MiniMoeRouterSpec:
    kind: str
    one_shot: OneShotRouterSpec | None = None
    recurrent_pre_expert: RecurrentPreExpertRouterSpec | None = None

    def validate(self) -> None:
        if self.kind == "one_shot":
            if self.one_shot is None or self.recurrent_pre_expert is not None:
                raise ValidationError(
                    "MiniMoeRouterSpec(kind=one_shot) must set only one_shot"
                )
            return
        if self.kind == "recurrent_pre_expert":
            if self.recurrent_pre_expert is None or self.one_shot is not None:
                raise ValidationError(
                    "MiniMoeRouterSpec(kind=recurrent_pre_expert) must set only recurrent_pre_expert"
                )
            self.recurrent_pre_expert.validate()
            return
        raise ValidationError(f"unsupported mini_moe router kind: {self.kind}")


@dataclass(frozen=True)
class ResolvedDispatchContract:
    mode: MiniMoeDispatchMode
    active_experts_per_token: int
    execution_strategy: MiniMoeDispatchExecutionStrategy = MiniMoeDispatchExecutionStrategy.DENSE_GATHER
    tie_break: TieBreakPolicy = TieBreakPolicy.LOWEST_INDEX
    capacity: DispatchCapacityPolicy = DispatchCapacityPolicy.UNLIMITED


@dataclass(frozen=True)
class MiniMoeDispatchSpec:
    mode: MiniMoeDispatchMode = MiniMoeDispatchMode.SPARSE_TOP_K
    execution_strategy: MiniMoeDispatchExecutionStrategy = MiniMoeDispatchExecutionStrategy.DENSE_GATHER

    def resolve(self, active_experts_per_token: int) -> ResolvedDispatchContract:
        return ResolvedDispatchContract(
            mode=self.mode,
            active_experts_per_token=active_experts_per_token,
            execution_strategy=self.execution_strategy,
        )


@dataclass(frozen=True)
class MiniMoeRuntimeSpec:
    dispatch: MiniMoeDispatchSpec

    def validate(self) -> None:
        return None


@dataclass(frozen=True)
class MiniMoeObservabilitySpec:
    record_route_entropy: bool = True
    record_winner_margin: bool = True
    record_active_expert_count: bool = True
    max_token_route_traces_per_layer: int = 8

    def validate(self) -> None:
        if self.max_token_route_traces_per_layer < 0:
            raise ValidationError(
                "mini_moe.observability.max_token_route_traces_per_layer must be non-negative"
            )
        return None


@dataclass(frozen=True)
class MiniMoeArchitectureSpec:
    schema_version: int
    preset: MiniMoePreset | None
    label: str
    backbone: MiniMoeBackboneSpec
    moe: MiniMoeStackSpec
    router: MiniMoeRouterSpec

    def validate(self) -> None:
        ensure_positive(self.schema_version, "mini_moe.architecture.schema_version")
        if not self.label.strip():
            raise ValidationError("mini_moe.architecture.label must not be empty")
        self.backbone.validate()
        self.moe.validate(self.backbone.total_layers)
        self.router.validate()
        layout = self.resolved_layout()
        if self.router.kind == "recurrent_pre_expert":
            recurrent_spec = self.router.recurrent_pre_expert
            assert recurrent_spec is not None
            if recurrent_spec.round2_layer_indices is not None:
                moe_layers = set(layout.moe_layers)
                for layer_index in recurrent_spec.round2_layer_indices:
                    if layer_index < 0 or layer_index >= self.backbone.total_layers:
                        raise ValidationError(
                            f"mini_moe.router.round2_layer_indices contains out-of-range layer {layer_index}"
                        )
                    if layer_index not in moe_layers:
                        raise ValidationError(
                            f"mini_moe.router.round2_layer_indices layer {layer_index} is not a routed MoE layer"
                        )

    def resolved_layout(self) -> ResolvedMiniMoeLayout:
        layout = self.moe.moe_layer_schedule.resolve(self.backbone.total_layers)
        layout.validate()
        return layout

    def resolved_round2_layer_indices(self) -> tuple[int, ...]:
        if self.router.kind != "recurrent_pre_expert":
            return ()
        recurrent_spec = self.router.recurrent_pre_expert
        assert recurrent_spec is not None
        if recurrent_spec.round2_layer_indices is None:
            return self.resolved_layout().moe_layers
        return recurrent_spec.round2_layer_indices


@dataclass(frozen=True)
class MiniMoeSurfaceSpec:
    architecture: MiniMoeArchitectureSpec
    runtime: MiniMoeRuntimeSpec
    observability: MiniMoeObservabilitySpec

    @property
    def runtime_optimization_family(self) -> RuntimeOptimizationFamily:
        return RuntimeOptimizationFamily.TRANSFORMER_MOE_ROUTING

    def validate(self) -> None:
        self.architecture.validate()
        self.runtime.validate()
        self.observability.validate()
        self.resolved_dispatch()

    def resolved_dispatch(self) -> ResolvedDispatchContract:
        return self.runtime.dispatch.resolve(self.architecture.moe.active_experts_per_token)

    @staticmethod
    def _round2_layer_label_suffix(round2_layer_indices: tuple[int, ...] | None) -> str:
        if round2_layer_indices is None:
            return ""
        if not round2_layer_indices:
            return "-lnone"
        layer_token = "_".join(str(layer_index) for layer_index in round2_layer_indices)
        return f"-l{layer_token}"

    @classmethod
    def phase1_reference_default(cls) -> "MiniMoeSurfaceSpec":
        return cls(
            architecture=MiniMoeArchitectureSpec(
                schema_version=1,
                preset=MiniMoePreset.PHASE1_REFERENCE,
                label="phase1-mini-moe-reference",
                backbone=MiniMoeBackboneSpec(
                    vocab_size=257,
                    hidden_dim=128,
                    head_count=4,
                    total_layers=8,
                    local_window=256,
                    ffn_multiplier=4,
                ),
                moe=MiniMoeStackSpec(
                    experts_per_block=4,
                    active_experts_per_token=1,
                    moe_layer_schedule=MiniMoeLayerSchedule(
                        kind=MiniMoeLayerScheduleKind.ALL_LAYERS
                    ),
                    expert_ffn_multiplier=4,
                    load_balance_loss_weight=0.01,
                ),
                router=MiniMoeRouterSpec(
                    kind="one_shot",
                    one_shot=OneShotRouterSpec(),
                ),
            ),
            runtime=MiniMoeRuntimeSpec(dispatch=MiniMoeDispatchSpec()),
            observability=MiniMoeObservabilitySpec(),
        )

    @classmethod
    def phase1_recurrent_default(
        cls,
        *,
        round2_layer_indices: tuple[int, ...] | None = None,
    ) -> "MiniMoeSurfaceSpec":
        layer_suffix = cls._round2_layer_label_suffix(round2_layer_indices)
        return cls(
            architecture=MiniMoeArchitectureSpec(
                schema_version=1,
                preset=MiniMoePreset.PHASE1_RECURRENT,
                label=f"phase1-mini-moe-recurrent-r2-s64{layer_suffix}",
                backbone=MiniMoeBackboneSpec(
                    vocab_size=257,
                    hidden_dim=128,
                    head_count=4,
                    total_layers=8,
                    local_window=256,
                    ffn_multiplier=4,
                ),
                moe=MiniMoeStackSpec(
                    experts_per_block=4,
                    active_experts_per_token=1,
                    moe_layer_schedule=MiniMoeLayerSchedule(
                        kind=MiniMoeLayerScheduleKind.ALL_LAYERS
                    ),
                    expert_ffn_multiplier=4,
                    load_balance_loss_weight=0.01,
                ),
                router=MiniMoeRouterSpec(
                    kind="recurrent_pre_expert",
                    recurrent_pre_expert=RecurrentPreExpertRouterSpec(
                        round_count=2,
                        state_dim=64,
                        round2_layer_indices=round2_layer_indices,
                    ),
                ),
            ),
            runtime=MiniMoeRuntimeSpec(dispatch=MiniMoeDispatchSpec()),
            observability=MiniMoeObservabilitySpec(),
        )

    @classmethod
    def phase1_recurrent_gated_default(
        cls,
        *,
        winner_margin_threshold: float = 0.02,
        execution_strategy: RecurrentRoundExecutionStrategy = RecurrentRoundExecutionStrategy.DENSE_BLEND,
        round2_layer_indices: tuple[int, ...] | None = None,
    ) -> "MiniMoeSurfaceSpec":
        threshold_token = f"{winner_margin_threshold:.3f}".replace(".", "p")
        layer_suffix = cls._round2_layer_label_suffix(round2_layer_indices)
        return cls(
            architecture=MiniMoeArchitectureSpec(
                schema_version=1,
                preset=MiniMoePreset.PHASE1_RECURRENT_GATED,
                label=f"phase1-mini-moe-recurrent-gated-r2-s64-m{threshold_token}{layer_suffix}",
                backbone=MiniMoeBackboneSpec(
                    vocab_size=257,
                    hidden_dim=128,
                    head_count=4,
                    total_layers=8,
                    local_window=256,
                    ffn_multiplier=4,
                ),
                moe=MiniMoeStackSpec(
                    experts_per_block=4,
                    active_experts_per_token=1,
                    moe_layer_schedule=MiniMoeLayerSchedule(
                        kind=MiniMoeLayerScheduleKind.ALL_LAYERS
                    ),
                    expert_ffn_multiplier=4,
                    load_balance_loss_weight=0.01,
                ),
                router=MiniMoeRouterSpec(
                    kind="recurrent_pre_expert",
                    recurrent_pre_expert=RecurrentPreExpertRouterSpec(
                        round_count=2,
                        state_dim=64,
                        gate=RecurrentRoundGateSpec(
                            kind=RecurrentRoundGateKind.WINNER_MARGIN_BELOW,
                            threshold=winner_margin_threshold,
                        ),
                        execution_strategy=execution_strategy,
                        round2_layer_indices=round2_layer_indices,
                    ),
                ),
            ),
            runtime=MiniMoeRuntimeSpec(dispatch=MiniMoeDispatchSpec()),
            observability=MiniMoeObservabilitySpec(),
        )

    @classmethod
    def phase1_recurrent_scaled_margin_gated_default(
        cls,
        *,
        winner_margin_threshold: float = 0.02,
        reference_experts_per_block: int = 4,
        execution_strategy: RecurrentRoundExecutionStrategy = RecurrentRoundExecutionStrategy.DENSE_BLEND,
        round2_layer_indices: tuple[int, ...] | None = None,
    ) -> "MiniMoeSurfaceSpec":
        threshold_token = f"{winner_margin_threshold:.3f}".replace(".", "p")
        layer_suffix = cls._round2_layer_label_suffix(round2_layer_indices)
        return cls(
            architecture=MiniMoeArchitectureSpec(
                schema_version=1,
                preset=MiniMoePreset.PHASE1_RECURRENT_GATED,
                label=(
                    "phase1-mini-moe-recurrent-gated-r2-s64-"
                    f"sm{reference_experts_per_block}-m{threshold_token}{layer_suffix}"
                ),
                backbone=MiniMoeBackboneSpec(
                    vocab_size=257,
                    hidden_dim=128,
                    head_count=4,
                    total_layers=8,
                    local_window=256,
                    ffn_multiplier=4,
                ),
                moe=MiniMoeStackSpec(
                    experts_per_block=4,
                    active_experts_per_token=1,
                    moe_layer_schedule=MiniMoeLayerSchedule(
                        kind=MiniMoeLayerScheduleKind.ALL_LAYERS
                    ),
                    expert_ffn_multiplier=4,
                    load_balance_loss_weight=0.01,
                ),
                router=MiniMoeRouterSpec(
                    kind="recurrent_pre_expert",
                    recurrent_pre_expert=RecurrentPreExpertRouterSpec(
                        round_count=2,
                        state_dim=64,
                        gate=RecurrentRoundGateSpec(
                            kind=RecurrentRoundGateKind.SCALED_WINNER_MARGIN_BELOW,
                            threshold=winner_margin_threshold,
                            reference_experts_per_block=reference_experts_per_block,
                        ),
                        execution_strategy=execution_strategy,
                        round2_layer_indices=round2_layer_indices,
                    ),
                ),
            ),
            runtime=MiniMoeRuntimeSpec(dispatch=MiniMoeDispatchSpec()),
            observability=MiniMoeObservabilitySpec(),
        )

    @classmethod
    def phase1_recurrent_fraction_gated_default(
        cls,
        *,
        target_applied_fraction: float = 0.2,
        execution_strategy: RecurrentRoundExecutionStrategy = RecurrentRoundExecutionStrategy.DENSE_BLEND,
        round2_layer_indices: tuple[int, ...] | None = None,
    ) -> "MiniMoeSurfaceSpec":
        fraction_token = f"{target_applied_fraction:.3f}".replace(".", "p")
        layer_suffix = cls._round2_layer_label_suffix(round2_layer_indices)
        return cls(
            architecture=MiniMoeArchitectureSpec(
                schema_version=1,
                preset=MiniMoePreset.PHASE1_RECURRENT_GATED,
                label=f"phase1-mini-moe-recurrent-gated-r2-s64-f{fraction_token}{layer_suffix}",
                backbone=MiniMoeBackboneSpec(
                    vocab_size=257,
                    hidden_dim=128,
                    head_count=4,
                    total_layers=8,
                    local_window=256,
                    ffn_multiplier=4,
                ),
                moe=MiniMoeStackSpec(
                    experts_per_block=4,
                    active_experts_per_token=1,
                    moe_layer_schedule=MiniMoeLayerSchedule(
                        kind=MiniMoeLayerScheduleKind.ALL_LAYERS
                    ),
                    expert_ffn_multiplier=4,
                    load_balance_loss_weight=0.01,
                ),
                router=MiniMoeRouterSpec(
                    kind="recurrent_pre_expert",
                    recurrent_pre_expert=RecurrentPreExpertRouterSpec(
                        round_count=2,
                        state_dim=64,
                        gate=RecurrentRoundGateSpec(
                            kind=RecurrentRoundGateKind.TARGET_APPLIED_FRACTION,
                            target_applied_fraction=target_applied_fraction,
                        ),
                        execution_strategy=execution_strategy,
                        round2_layer_indices=round2_layer_indices,
                    ),
                ),
            ),
            runtime=MiniMoeRuntimeSpec(dispatch=MiniMoeDispatchSpec()),
            observability=MiniMoeObservabilitySpec(),
        )

    @classmethod
    def phase1_recurrent_entropy_gated_default(
        cls,
        *,
        normalized_entropy_threshold: float = 0.8,
        execution_strategy: RecurrentRoundExecutionStrategy = RecurrentRoundExecutionStrategy.DENSE_BLEND,
        round2_layer_indices: tuple[int, ...] | None = None,
    ) -> "MiniMoeSurfaceSpec":
        threshold_token = f"{normalized_entropy_threshold:.3f}".replace(".", "p")
        layer_suffix = cls._round2_layer_label_suffix(round2_layer_indices)
        return cls(
            architecture=MiniMoeArchitectureSpec(
                schema_version=1,
                preset=MiniMoePreset.PHASE1_RECURRENT_GATED,
                label=f"phase1-mini-moe-recurrent-gated-r2-s64-h{threshold_token}{layer_suffix}",
                backbone=MiniMoeBackboneSpec(
                    vocab_size=257,
                    hidden_dim=128,
                    head_count=4,
                    total_layers=8,
                    local_window=256,
                    ffn_multiplier=4,
                ),
                moe=MiniMoeStackSpec(
                    experts_per_block=4,
                    active_experts_per_token=1,
                    moe_layer_schedule=MiniMoeLayerSchedule(
                        kind=MiniMoeLayerScheduleKind.ALL_LAYERS
                    ),
                    expert_ffn_multiplier=4,
                    load_balance_loss_weight=0.01,
                ),
                router=MiniMoeRouterSpec(
                    kind="recurrent_pre_expert",
                    recurrent_pre_expert=RecurrentPreExpertRouterSpec(
                        round_count=2,
                        state_dim=64,
                        gate=RecurrentRoundGateSpec(
                            kind=RecurrentRoundGateKind.NORMALIZED_ENTROPY_ABOVE,
                            normalized_entropy_threshold=normalized_entropy_threshold,
                        ),
                        execution_strategy=execution_strategy,
                        round2_layer_indices=round2_layer_indices,
                    ),
                ),
            ),
            runtime=MiniMoeRuntimeSpec(dispatch=MiniMoeDispatchSpec()),
            observability=MiniMoeObservabilitySpec(),
        )

    @classmethod
    def phase1_recurrent_learned_gated_default(
        cls,
        *,
        learned_hidden_dim: int = 32,
        learned_prior_probability: float = 0.2,
        teacher_supervision_weight: float = 1.0,
        execution_strategy: RecurrentRoundExecutionStrategy = RecurrentRoundExecutionStrategy.DENSE_BLEND,
        round2_layer_indices: tuple[int, ...] | None = None,
    ) -> "MiniMoeSurfaceSpec":
        prior_token = f"{learned_prior_probability:.3f}".replace(".", "p")
        layer_suffix = cls._round2_layer_label_suffix(round2_layer_indices)
        return cls(
            architecture=MiniMoeArchitectureSpec(
                schema_version=1,
                preset=MiniMoePreset.PHASE1_RECURRENT_GATED,
                label=(
                    "phase1-mini-moe-recurrent-gated-r2-s64-"
                    f"lg{learned_hidden_dim}-p{prior_token}{layer_suffix}"
                ),
                backbone=MiniMoeBackboneSpec(
                    vocab_size=257,
                    hidden_dim=128,
                    head_count=4,
                    total_layers=8,
                    local_window=256,
                    ffn_multiplier=4,
                ),
                moe=MiniMoeStackSpec(
                    experts_per_block=4,
                    active_experts_per_token=1,
                    moe_layer_schedule=MiniMoeLayerSchedule(
                        kind=MiniMoeLayerScheduleKind.ALL_LAYERS
                    ),
                    expert_ffn_multiplier=4,
                    load_balance_loss_weight=0.01,
                ),
                router=MiniMoeRouterSpec(
                    kind="recurrent_pre_expert",
                    recurrent_pre_expert=RecurrentPreExpertRouterSpec(
                        round_count=2,
                        state_dim=64,
                        gate=RecurrentRoundGateSpec(
                            kind=RecurrentRoundGateKind.LEARNED_SCORE_ABOVE,
                            learned_hidden_dim=learned_hidden_dim,
                            learned_prior_probability=learned_prior_probability,
                            teacher_kind=LearnedGateTeacherKind.BLENDED_UNCERTAINTY,
                            teacher_supervision_weight=teacher_supervision_weight,
                        ),
                        execution_strategy=execution_strategy,
                        round2_layer_indices=round2_layer_indices,
                    ),
                ),
            ),
            runtime=MiniMoeRuntimeSpec(dispatch=MiniMoeDispatchSpec()),
            observability=MiniMoeObservabilitySpec(),
        )

    @classmethod
    def phase1_recurrent_learned_score_gated_default(
        cls,
        *,
        learned_hidden_dim: int = 32,
        learned_prior_probability: float = 0.2,
        target_applied_fraction: float = 0.2,
        teacher_supervision_weight: float = 1.0,
        execution_strategy: RecurrentRoundExecutionStrategy = RecurrentRoundExecutionStrategy.DENSE_BLEND,
        round2_layer_indices: tuple[int, ...] | None = None,
    ) -> "MiniMoeSurfaceSpec":
        prior_token = f"{learned_prior_probability:.3f}".replace(".", "p")
        fraction_token = f"{target_applied_fraction:.3f}".replace(".", "p")
        layer_suffix = cls._round2_layer_label_suffix(round2_layer_indices)
        return cls(
            architecture=MiniMoeArchitectureSpec(
                schema_version=1,
                preset=MiniMoePreset.PHASE1_RECURRENT_GATED,
                label=(
                    "phase1-mini-moe-recurrent-gated-r2-s64-"
                    f"lsg{learned_hidden_dim}-p{prior_token}-f{fraction_token}{layer_suffix}"
                ),
                backbone=MiniMoeBackboneSpec(
                    vocab_size=257,
                    hidden_dim=128,
                    head_count=4,
                    total_layers=8,
                    local_window=256,
                    ffn_multiplier=4,
                ),
                moe=MiniMoeStackSpec(
                    experts_per_block=4,
                    active_experts_per_token=1,
                    moe_layer_schedule=MiniMoeLayerSchedule(
                        kind=MiniMoeLayerScheduleKind.ALL_LAYERS
                    ),
                    expert_ffn_multiplier=4,
                    load_balance_loss_weight=0.01,
                ),
                router=MiniMoeRouterSpec(
                    kind="recurrent_pre_expert",
                    recurrent_pre_expert=RecurrentPreExpertRouterSpec(
                        round_count=2,
                        state_dim=64,
                        gate=RecurrentRoundGateSpec(
                            kind=RecurrentRoundGateKind.LEARNED_SCORE_TOP_FRACTION,
                            learned_hidden_dim=learned_hidden_dim,
                            learned_prior_probability=learned_prior_probability,
                            target_applied_fraction=target_applied_fraction,
                            teacher_kind=LearnedGateTeacherKind.BLENDED_UNCERTAINTY,
                            teacher_supervision_weight=teacher_supervision_weight,
                        ),
                        execution_strategy=execution_strategy,
                        round2_layer_indices=round2_layer_indices,
                    ),
                ),
            ),
            runtime=MiniMoeRuntimeSpec(dispatch=MiniMoeDispatchSpec()),
            observability=MiniMoeObservabilitySpec(),
        )
