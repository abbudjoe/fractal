from __future__ import annotations

from dataclasses import dataclass

from .common import StringEnum, ValidationError, ensure_positive
from .runtime import PrimitiveStateTransformMode


BYTE_LEVEL_PAD_TOKEN = 0
BYTE_LEVEL_VOCAB_SIZE = 257
DEFAULT_D_MODEL = 128
DEFAULT_HEADS = 4
DEFAULT_LAYERS = 8
DEFAULT_LOCAL_WINDOW = 256


class Path1VariantKind(StringEnum):
    ATTENTION_ONLY = "attention-only"
    REFERENCE_SSM_HYBRID = "reference-ssm-hybrid"
    PRIMITIVE_HYBRID = "primitive-hybrid"


class HybridAttentionLayerRole(StringEnum):
    EXACT_ATTENTION = "exact-attention"
    SHARED_EXACT_ATTENTION = "shared-exact-attention"
    REFERENCE_SSM = "reference-ssm"
    PRIMITIVE = "primitive"


class ReferenceSsmProfile(StringEnum):
    GATED_DELTANET_FLA = "gated-deltanet-fla"
    GATED_DELTANET_FLA_P20_COMPAT = "gated-deltanet-fla-p20-compatible"
    GATED_DELTANET_FLA_P20_MULTI_READ = "gated-deltanet-fla-p20-multi-read"
    GATED_DELTANET_MAMBA3_TORCH = "gated-deltanet-mamba3-torch"
    GATED_DELTANET_P20_FUSED_ALL_TORCH = "gated-deltanet-p20-fused-all-torch"
    GATED_DELTANET_P20_FUSED_BETA_TORCH = "gated-deltanet-p20-fused-beta-torch"
    GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH = "gated-deltanet-p20-fused-multi-read-torch"
    GATED_DELTANET_P20_FUSED_QKV_TORCH = "gated-deltanet-p20-fused-qkv-torch"
    GATED_DELTANET_P20_FUSED_RESIDUAL_READOUT_TORCH = "gated-deltanet-p20-fused-residual-readout-torch"
    GATED_DELTANET_P20_FUSED_TORCH = "gated-deltanet-p20-fused-torch"
    GATED_DELTANET_P20_MAMBA3_TORCH = "gated-deltanet-p20-mamba3-torch"
    GATED_DELTANET_P20_THIN_TORCH = "gated-deltanet-p20-thin-torch"
    GATED_DELTANET_P20_TORCH = "gated-deltanet-p20-torch"
    GATED_DELTANET_TORCH = "gated-deltanet-torch"
    MAMBA3_MIMO_REFERENCE = "mamba3-mimo-reference"
    MAMBA3_SISO_REFERENCE = "mamba3-siso-reference"
    MAMBA3_SISO_RUNTIME = "mamba3-siso-runtime"
    P20_MAMBA3_TORCH = "p20-mamba3-torch"
    P20_THIN_TORCH = "p20-thin-torch"
    P20_TORCH = "p20-torch"

    @property
    def is_mimo(self) -> bool:
        return self is ReferenceSsmProfile.MAMBA3_MIMO_REFERENCE

    @property
    def is_gated_deltanet(self) -> bool:
        return self in {
            ReferenceSsmProfile.GATED_DELTANET_FLA,
            ReferenceSsmProfile.GATED_DELTANET_FLA_P20_COMPAT,
            ReferenceSsmProfile.GATED_DELTANET_FLA_P20_MULTI_READ,
            ReferenceSsmProfile.GATED_DELTANET_MAMBA3_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_ALL_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_BETA_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_QKV_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_RESIDUAL_READOUT_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_MAMBA3_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_THIN_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_TORCH,
        }

    @property
    def is_fla_gated_deltanet(self) -> bool:
        return self is ReferenceSsmProfile.GATED_DELTANET_FLA

    @property
    def is_fla_gdnp_compatible(self) -> bool:
        return self in {
            ReferenceSsmProfile.GATED_DELTANET_FLA_P20_COMPAT,
            ReferenceSsmProfile.GATED_DELTANET_FLA_P20_MULTI_READ,
        }

    @property
    def fla_gdnp_compatible_law(self) -> str:
        if self is ReferenceSsmProfile.GATED_DELTANET_FLA_P20_COMPAT:
            return "single-read"
        if self is ReferenceSsmProfile.GATED_DELTANET_FLA_P20_MULTI_READ:
            return "multi-read"
        raise ValueError(f"reference SSM profile {self.value} is not an FLA-compatible GDN/P20 profile")

    @property
    def is_gdnp_fused(self) -> bool:
        return self in {
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_ALL_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_BETA_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_QKV_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_RESIDUAL_READOUT_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_TORCH,
        }

    @property
    def gdnp_fused_law(self) -> str:
        if self is ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_ALL_TORCH:
            return "all"
        if self is ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_BETA_TORCH:
            return "beta"
        if self is ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH:
            return "multi-read"
        if self is ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_QKV_TORCH:
            return "qkv"
        if self is ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_RESIDUAL_READOUT_TORCH:
            return "residual-readout"
        if self is ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_TORCH:
            return "value"
        raise ValueError(f"reference SSM profile {self.value} is not a fused GDN/P20 profile")

    @property
    def composite_branches(self) -> tuple[str, ...]:
        if self is ReferenceSsmProfile.GATED_DELTANET_MAMBA3_TORCH:
            return ("gdn", "mamba3")
        if self is ReferenceSsmProfile.GATED_DELTANET_P20_MAMBA3_TORCH:
            return ("gdn", "p20", "mamba3")
        if self is ReferenceSsmProfile.GATED_DELTANET_P20_THIN_TORCH:
            return ("gdn", "p20_thin")
        if self is ReferenceSsmProfile.GATED_DELTANET_P20_TORCH:
            return ("gdn", "p20")
        if self is ReferenceSsmProfile.P20_MAMBA3_TORCH:
            return ("p20", "mamba3")
        return ()

    @property
    def is_composite(self) -> bool:
        return bool(self.composite_branches)

    @property
    def is_p20_scan(self) -> bool:
        return self in {
            ReferenceSsmProfile.P20_THIN_TORCH,
            ReferenceSsmProfile.P20_TORCH,
        }

    @property
    def p20_branch_width_factor(self) -> float:
        return 0.5 if self is ReferenceSsmProfile.P20_THIN_TORCH else 1.0

    @property
    def mimo_rank(self) -> int:
        return 4 if self.is_mimo else 1

    @property
    def runtime_oriented(self) -> bool:
        return self in {
            ReferenceSsmProfile.GATED_DELTANET_FLA,
            ReferenceSsmProfile.GATED_DELTANET_FLA_P20_COMPAT,
            ReferenceSsmProfile.GATED_DELTANET_FLA_P20_MULTI_READ,
            ReferenceSsmProfile.GATED_DELTANET_MAMBA3_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_ALL_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_BETA_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_QKV_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_RESIDUAL_READOUT_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_MAMBA3_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_THIN_TORCH,
            ReferenceSsmProfile.P20_MAMBA3_TORCH,
            ReferenceSsmProfile.MAMBA3_SISO_RUNTIME,
        }


class PrimitiveProfile(StringEnum):
    P1 = "p1"
    P1_FRACTAL_HYBRID = "p1-fractal-hybrid"
    P1_FRACTAL_HYBRID_COMPOSITE = "p1-fractal-hybrid-composite"
    P1_FRACTAL_HYBRID_DYN_GATE = "p1-fractal-hybrid-dyn-gate"
    P20 = "p2-0"
    P20_GDN_ROLE = "p2-0-gdn-role"
    P2 = "p2"
    P21 = "p2-1"
    P22 = "p2-2"
    P23 = "p2-3"
    P2_MANDELBROT = "p2-mandelbrot"
    P3_HIERARCHICAL = "p3-hierarchical"
    B1_FRACTAL_GATED = "b1-fractal-gated"
    B2_STABLE_HIERARCHICAL = "b2-stable-hierarchical"
    B3_FRACTAL_HIERARCHICAL = "b3-fractal-hierarchical"
    B4_UNIVERSAL = "b4-universal"
    IFS = "ifs"
    GENERALIZED_MOBIUS = "generalized-mobius"
    LOGISTIC_CHAOTIC_MAP = "logistic-chaotic-map"
    JULIA_RECURSIVE_ESCAPE = "julia-recursive-escape"
    MANDELBOX_RECURSIVE = "mandelbox-recursive"

    @property
    def state_width_factor(self) -> int:
        return 2 if self in {PrimitiveProfile.P21, PrimitiveProfile.P22} else 1

    @property
    def has_explicit_internal_readout(self) -> bool:
        return self in {
            PrimitiveProfile.P20_GDN_ROLE,
            PrimitiveProfile.P2,
            PrimitiveProfile.P22,
            PrimitiveProfile.P23,
        }


class PrimitiveResidualMode(StringEnum):
    PLAIN = "plain"
    SCALED = "scaled"
    GATED = "gated"


class PrimitiveReadoutMode(StringEnum):
    DIRECT = "direct"
    PROJECTED = "projected"
    PROJECTED_NORM = "projected-norm"


class PrimitiveNormMode(StringEnum):
    PRE_NORM_ONLY = "pre-norm-only"
    POST_READOUT_NORM = "post-readout-norm"
    RESIDUAL_RENORM = "residual-renorm"


class PrimitiveWrapperMode(StringEnum):
    STANDARD = "standard"
    MAMBA_RMS = "mamba-rms"


class PrimitiveExecutionProfile(StringEnum):
    REFERENCE = "reference"
    RUNTIME = "runtime"


@dataclass(frozen=True)
class Path1ModelShape:
    vocab_size: int = BYTE_LEVEL_VOCAB_SIZE
    d_model: int = DEFAULT_D_MODEL
    head_count: int = DEFAULT_HEADS
    total_layers: int = DEFAULT_LAYERS
    local_window: int = DEFAULT_LOCAL_WINDOW
    ffn_multiplier: int = 4

    @property
    def d_ff(self) -> int:
        return self.d_model * self.ffn_multiplier

    def validate(self) -> None:
        ensure_positive(self.vocab_size, "path1_shape.vocab_size")
        ensure_positive(self.d_model, "path1_shape.d_model")
        ensure_positive(self.head_count, "path1_shape.head_count")
        ensure_positive(self.total_layers, "path1_shape.total_layers")
        ensure_positive(self.local_window, "path1_shape.local_window")
        ensure_positive(self.ffn_multiplier, "path1_shape.ffn_multiplier")
        if self.d_model % self.head_count != 0:
            raise ValidationError(
                f"path1_shape.d_model {self.d_model} must be divisible by head_count {self.head_count}"
            )


DEFAULT_PATH1_MODEL_SHAPE = Path1ModelShape()

_LAYER_SCHEDULE_TOKEN_MAP = {
    "A": HybridAttentionLayerRole.EXACT_ATTENTION,
    "S": HybridAttentionLayerRole.SHARED_EXACT_ATTENTION,
    "R": HybridAttentionLayerRole.REFERENCE_SSM,
    "P": HybridAttentionLayerRole.PRIMITIVE,
}

_EXACT_ATTENTION_ROLES = {
    HybridAttentionLayerRole.EXACT_ATTENTION,
    HybridAttentionLayerRole.SHARED_EXACT_ATTENTION,
}


@dataclass(frozen=True)
class Path1VariantSpec:
    kind: Path1VariantKind
    label: str
    shape: Path1ModelShape
    layer_schedule: tuple[HybridAttentionLayerRole, ...]
    reference_ssm_profile: ReferenceSsmProfile | None = None
    primitive_profile: PrimitiveProfile | None = None
    primitive_residual_mode: PrimitiveResidualMode | None = None
    primitive_readout_mode: PrimitiveReadoutMode | None = None
    primitive_norm_mode: PrimitiveNormMode | None = None
    primitive_wrapper_mode: PrimitiveWrapperMode | None = None
    primitive_execution_profile: PrimitiveExecutionProfile | None = None
    primitive_state_transform_mode: PrimitiveStateTransformMode | None = None
    final_norm_kind: str = "identity"

    def validate(self) -> None:
        self.shape.validate()
        if not self.label.strip():
            raise ValidationError("path1_variant.label must not be empty")
        if len(self.layer_schedule) != self.shape.total_layers:
            raise ValidationError(
                "path1_variant.layer_schedule length must match shape.total_layers"
            )
        exact_attention_layers = sum(
            1 for role in self.layer_schedule if role in _EXACT_ATTENTION_ROLES
        )
        if exact_attention_layers == 0:
            raise ValidationError("path1_variant must retain at least one exact-attention layer")
        if self.kind is Path1VariantKind.ATTENTION_ONLY:
            if any(role not in _EXACT_ATTENTION_ROLES for role in self.layer_schedule):
                raise ValidationError("attention-only variant must contain only exact-attention layers")
            if any(
                value is not None
                for value in (
                    self.reference_ssm_profile,
                    self.primitive_profile,
                    self.primitive_residual_mode,
                    self.primitive_readout_mode,
                    self.primitive_norm_mode,
                    self.primitive_wrapper_mode,
                    self.primitive_execution_profile,
                    self.primitive_state_transform_mode,
                )
            ):
                raise ValidationError("attention-only variant must not set reference or primitive options")
        elif self.kind is Path1VariantKind.REFERENCE_SSM_HYBRID:
            if self.reference_ssm_profile is None:
                raise ValidationError("reference-ssm-hybrid variant must set reference_ssm_profile")
            if self.primitive_profile is not None:
                raise ValidationError("reference-ssm-hybrid variant must not set primitive_profile")
            if any(role not in {*_EXACT_ATTENTION_ROLES, HybridAttentionLayerRole.REFERENCE_SSM} for role in self.layer_schedule):
                raise ValidationError("reference-ssm-hybrid schedule may contain only exact-attention and reference-SSM roles")
        elif self.kind is Path1VariantKind.PRIMITIVE_HYBRID:
            if self.primitive_profile is None:
                raise ValidationError("primitive-hybrid variant must set primitive_profile")
            if self.reference_ssm_profile is not None:
                raise ValidationError("primitive-hybrid variant must not set reference_ssm_profile")
            if any(role not in {*_EXACT_ATTENTION_ROLES, HybridAttentionLayerRole.PRIMITIVE} for role in self.layer_schedule):
                raise ValidationError("primitive-hybrid schedule may contain only exact-attention and primitive roles")
            if any(
                value is None
                for value in (
                    self.primitive_residual_mode,
                    self.primitive_readout_mode,
                    self.primitive_norm_mode,
                    self.primitive_wrapper_mode,
                    self.primitive_execution_profile,
                    self.primitive_state_transform_mode,
                )
            ):
                raise ValidationError(
                    "primitive-hybrid variant must set primitive residual/readout/norm/wrapper/execution/state-transform modes"
                )
        else:
            raise ValidationError(f"unsupported path1 variant kind: {self.kind}")
        if self.final_norm_kind not in {"identity", "rmsnorm"}:
            raise ValidationError(
                f"path1_variant.final_norm_kind must be identity|rmsnorm, got {self.final_norm_kind}"
            )


@dataclass(frozen=True)
class Path1BaselineMatrix:
    attention_only: Path1VariantSpec
    reference_ssm_hybrid: Path1VariantSpec
    primitive_hybrid: Path1VariantSpec

    def validate(self) -> None:
        self.attention_only.validate()
        self.reference_ssm_hybrid.validate()
        self.primitive_hybrid.validate()
        reference_shape = self.attention_only.shape
        for variant in (self.reference_ssm_hybrid, self.primitive_hybrid):
            if variant.shape != reference_shape:
                raise ValidationError("Path1 baseline matrix variants must share the same model shape")


def _attention_schedule(total_layers: int) -> tuple[HybridAttentionLayerRole, ...]:
    return tuple(HybridAttentionLayerRole.EXACT_ATTENTION for _ in range(total_layers))


def _alternating_schedule(total_layers: int, odd_role: HybridAttentionLayerRole) -> tuple[HybridAttentionLayerRole, ...]:
    return tuple(
        HybridAttentionLayerRole.EXACT_ATTENTION if index % 2 == 0 else odd_role
        for index in range(total_layers)
    )


def _variant_label(*parts: str) -> str:
    return "-".join(part for part in parts if part)


def parse_layer_schedule_spec(schedule: str) -> tuple[HybridAttentionLayerRole, ...]:
    normalized = (
        schedule.strip().replace(",", "").replace("-", "").replace("_", "").replace(" ", "").upper()
    )
    if not normalized:
        raise ValidationError("path1_variant.layer_schedule override must not be empty")
    try:
        return tuple(_LAYER_SCHEDULE_TOKEN_MAP[token] for token in normalized)
    except KeyError as exc:
        raise ValidationError(
            "path1_variant.layer_schedule override may contain only A, S, R, or P tokens"
        ) from exc


def layer_schedule_signature(schedule: tuple[HybridAttentionLayerRole, ...]) -> str:
    return "".join(
        "a"
        if role is HybridAttentionLayerRole.EXACT_ATTENTION
        else "s"
        if role is HybridAttentionLayerRole.SHARED_EXACT_ATTENTION
        else "r"
        if role is HybridAttentionLayerRole.REFERENCE_SSM
        else "p"
        for role in schedule
    )


def phase1_attention_only_variant(
    shape: Path1ModelShape = DEFAULT_PATH1_MODEL_SHAPE,
    layer_schedule: tuple[HybridAttentionLayerRole, ...] | None = None,
) -> Path1VariantSpec:
    schedule = layer_schedule or _attention_schedule(shape.total_layers)
    default_schedule = _attention_schedule(shape.total_layers)
    schedule_suffix = (
        f"schedule-{layer_schedule_signature(schedule)}"
        if schedule != default_schedule
        else ""
    )
    return Path1VariantSpec(
        kind=Path1VariantKind.ATTENTION_ONLY,
        label=_variant_label("attention-only", schedule_suffix),
        shape=shape,
        layer_schedule=schedule,
    )


def phase1_reference_ssm_variant(
    shape: Path1ModelShape = DEFAULT_PATH1_MODEL_SHAPE,
    profile: ReferenceSsmProfile = ReferenceSsmProfile.MAMBA3_SISO_RUNTIME,
    layer_schedule: tuple[HybridAttentionLayerRole, ...] | None = None,
) -> Path1VariantSpec:
    final_norm = (
        "rmsnorm"
        if profile.is_gated_deltanet
        or profile
        in {
            ReferenceSsmProfile.MAMBA3_SISO_REFERENCE,
            ReferenceSsmProfile.MAMBA3_SISO_RUNTIME,
            ReferenceSsmProfile.P20_MAMBA3_TORCH,
            ReferenceSsmProfile.P20_THIN_TORCH,
            ReferenceSsmProfile.P20_TORCH,
        }
        else "identity"
    )
    default_schedule = _alternating_schedule(shape.total_layers, HybridAttentionLayerRole.REFERENCE_SSM)
    schedule = layer_schedule or default_schedule
    schedule_suffix = (
        f"schedule-{layer_schedule_signature(schedule)}"
        if schedule != default_schedule
        else ""
    )
    return Path1VariantSpec(
        kind=Path1VariantKind.REFERENCE_SSM_HYBRID,
        label=_variant_label("reference-ssm-hybrid", profile.value, schedule_suffix),
        shape=shape,
        layer_schedule=schedule,
        reference_ssm_profile=profile,
        final_norm_kind=final_norm,
    )


def phase1_primitive_variant(
    shape: Path1ModelShape = DEFAULT_PATH1_MODEL_SHAPE,
    primitive_profile: PrimitiveProfile = PrimitiveProfile.P1,
    execution_profile: PrimitiveExecutionProfile = PrimitiveExecutionProfile.REFERENCE,
    residual_mode: PrimitiveResidualMode = PrimitiveResidualMode.PLAIN,
    readout_mode: PrimitiveReadoutMode = PrimitiveReadoutMode.DIRECT,
    norm_mode: PrimitiveNormMode = PrimitiveNormMode.PRE_NORM_ONLY,
    wrapper_mode: PrimitiveWrapperMode = PrimitiveWrapperMode.STANDARD,
    state_transform_mode: PrimitiveStateTransformMode = PrimitiveStateTransformMode.DENSE,
    layer_schedule: tuple[HybridAttentionLayerRole, ...] | None = None,
) -> Path1VariantSpec:
    default_schedule = _alternating_schedule(shape.total_layers, HybridAttentionLayerRole.PRIMITIVE)
    schedule = layer_schedule or default_schedule
    schedule_suffix = (
        f"schedule-{layer_schedule_signature(schedule)}"
        if schedule != default_schedule
        else ""
    )
    return Path1VariantSpec(
        kind=Path1VariantKind.PRIMITIVE_HYBRID,
        label=_variant_label(
            "primitive-hybrid",
            primitive_profile.value,
            execution_profile.value,
            residual_mode.value,
            readout_mode.value,
            norm_mode.value,
            wrapper_mode.value,
            state_transform_mode.value,
            schedule_suffix,
        ),
        shape=shape,
        layer_schedule=schedule,
        primitive_profile=primitive_profile,
        primitive_residual_mode=residual_mode,
        primitive_readout_mode=readout_mode,
        primitive_norm_mode=norm_mode,
        primitive_wrapper_mode=wrapper_mode,
        primitive_execution_profile=execution_profile,
        primitive_state_transform_mode=state_transform_mode,
    )


def phase1_baseline_matrix(
    shape: Path1ModelShape = DEFAULT_PATH1_MODEL_SHAPE,
    reference_profile: ReferenceSsmProfile = ReferenceSsmProfile.MAMBA3_SISO_RUNTIME,
    primitive_profile: PrimitiveProfile = PrimitiveProfile.P1,
    primitive_execution_profile: PrimitiveExecutionProfile = PrimitiveExecutionProfile.REFERENCE,
    residual_mode: PrimitiveResidualMode = PrimitiveResidualMode.PLAIN,
    readout_mode: PrimitiveReadoutMode = PrimitiveReadoutMode.DIRECT,
    norm_mode: PrimitiveNormMode = PrimitiveNormMode.PRE_NORM_ONLY,
    wrapper_mode: PrimitiveWrapperMode = PrimitiveWrapperMode.STANDARD,
    state_transform_mode: PrimitiveStateTransformMode = PrimitiveStateTransformMode.DENSE,
) -> Path1BaselineMatrix:
    matrix = Path1BaselineMatrix(
        attention_only=phase1_attention_only_variant(shape),
        reference_ssm_hybrid=phase1_reference_ssm_variant(shape, reference_profile),
        primitive_hybrid=phase1_primitive_variant(
            shape,
            primitive_profile=primitive_profile,
            execution_profile=primitive_execution_profile,
            residual_mode=residual_mode,
            readout_mode=readout_mode,
            norm_mode=norm_mode,
            wrapper_mode=wrapper_mode,
            state_transform_mode=state_transform_mode,
        ),
    )
    matrix.validate()
    return matrix
