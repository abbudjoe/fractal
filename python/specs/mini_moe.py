from __future__ import annotations

from dataclasses import dataclass

from python.specs.common import StringEnum, ValidationError, ensure_positive


class MiniMoeDispatchMode(StringEnum):
    SPARSE_TOP_K = "sparse_top_k"
    DENSE_DEBUG = "dense_debug"


class TieBreakPolicy(StringEnum):
    LOWEST_INDEX = "lowest_index"


class DispatchCapacityPolicy(StringEnum):
    UNLIMITED = "unlimited"


class MiniMoePreset(StringEnum):
    PHASE1_REFERENCE = "phase1_reference"
    PHASE1_RECURRENT = "phase1_recurrent"


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


@dataclass(frozen=True)
class RecurrentPreExpertRouterSpec:
    round_count: int
    state_dim: int

    def validate(self) -> None:
        ensure_positive(self.round_count, "mini_moe.router.round_count")
        ensure_positive(self.state_dim, "mini_moe.router.state_dim")


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
    tie_break: TieBreakPolicy = TieBreakPolicy.LOWEST_INDEX
    capacity: DispatchCapacityPolicy = DispatchCapacityPolicy.UNLIMITED


@dataclass(frozen=True)
class MiniMoeDispatchSpec:
    mode: MiniMoeDispatchMode = MiniMoeDispatchMode.SPARSE_TOP_K

    def resolve(self, active_experts_per_token: int) -> ResolvedDispatchContract:
        return ResolvedDispatchContract(
            mode=self.mode,
            active_experts_per_token=active_experts_per_token,
        )


@dataclass(frozen=True)
class MiniMoeRuntimeSpec:
    dispatch: MiniMoeDispatchSpec


@dataclass(frozen=True)
class MiniMoeObservabilitySpec:
    record_route_entropy: bool = True
    record_winner_margin: bool = True
    record_active_expert_count: bool = True


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

    def resolved_layout(self) -> ResolvedMiniMoeLayout:
        layout = self.moe.moe_layer_schedule.resolve(self.backbone.total_layers)
        layout.validate()
        return layout


@dataclass(frozen=True)
class MiniMoeSurfaceSpec:
    architecture: MiniMoeArchitectureSpec
    runtime: MiniMoeRuntimeSpec
    observability: MiniMoeObservabilitySpec

    def validate(self) -> None:
        self.architecture.validate()
