from __future__ import annotations

from dataclasses import dataclass

from .common import StringEnum, ValidationError, ensure_positive


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
    REFERENCE_SSM = "reference-ssm"
    PRIMITIVE = "primitive"


class ReferenceSsmProfile(StringEnum):
    MAMBA3_MIMO_REFERENCE = "mamba3-mimo-reference"
    MAMBA3_SISO_REFERENCE = "mamba3-siso-reference"
    MAMBA3_SISO_RUNTIME = "mamba3-siso-runtime"

    @property
    def is_mimo(self) -> bool:
        return self is ReferenceSsmProfile.MAMBA3_MIMO_REFERENCE

    @property
    def mimo_rank(self) -> int:
        return 4 if self.is_mimo else 1

    @property
    def runtime_oriented(self) -> bool:
        return self is ReferenceSsmProfile.MAMBA3_SISO_RUNTIME


class PrimitiveProfile(StringEnum):
    P1 = "p1"
    P20 = "p2-0"
    P2 = "p2"
    P21 = "p2-1"
    P22 = "p2-2"
    P23 = "p2-3"

    @property
    def state_width_factor(self) -> int:
        return 2 if self in {PrimitiveProfile.P21, PrimitiveProfile.P22} else 1

    @property
    def has_explicit_internal_readout(self) -> bool:
        return self in {PrimitiveProfile.P2, PrimitiveProfile.P22, PrimitiveProfile.P23}


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
            1 for role in self.layer_schedule if role is HybridAttentionLayerRole.EXACT_ATTENTION
        )
        if exact_attention_layers == 0:
            raise ValidationError("path1_variant must retain at least one exact-attention layer")
        if self.kind is Path1VariantKind.ATTENTION_ONLY:
            if any(role is not HybridAttentionLayerRole.EXACT_ATTENTION for role in self.layer_schedule):
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
                )
            ):
                raise ValidationError("attention-only variant must not set reference or primitive options")
        elif self.kind is Path1VariantKind.REFERENCE_SSM_HYBRID:
            if self.reference_ssm_profile is None:
                raise ValidationError("reference-ssm-hybrid variant must set reference_ssm_profile")
            if self.primitive_profile is not None:
                raise ValidationError("reference-ssm-hybrid variant must not set primitive_profile")
            if any(role not in {HybridAttentionLayerRole.EXACT_ATTENTION, HybridAttentionLayerRole.REFERENCE_SSM} for role in self.layer_schedule):
                raise ValidationError("reference-ssm-hybrid schedule may contain only exact-attention and reference-SSM roles")
        elif self.kind is Path1VariantKind.PRIMITIVE_HYBRID:
            if self.primitive_profile is None:
                raise ValidationError("primitive-hybrid variant must set primitive_profile")
            if self.reference_ssm_profile is not None:
                raise ValidationError("primitive-hybrid variant must not set reference_ssm_profile")
            if any(role not in {HybridAttentionLayerRole.EXACT_ATTENTION, HybridAttentionLayerRole.PRIMITIVE} for role in self.layer_schedule):
                raise ValidationError("primitive-hybrid schedule may contain only exact-attention and primitive roles")
            if any(
                value is None
                for value in (
                    self.primitive_residual_mode,
                    self.primitive_readout_mode,
                    self.primitive_norm_mode,
                    self.primitive_wrapper_mode,
                    self.primitive_execution_profile,
                )
            ):
                raise ValidationError(
                    "primitive-hybrid variant must set primitive residual/readout/norm/wrapper/execution modes"
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


def phase1_attention_only_variant(shape: Path1ModelShape = DEFAULT_PATH1_MODEL_SHAPE) -> Path1VariantSpec:
    return Path1VariantSpec(
        kind=Path1VariantKind.ATTENTION_ONLY,
        label="attention-only",
        shape=shape,
        layer_schedule=_attention_schedule(shape.total_layers),
    )


def phase1_reference_ssm_variant(
    shape: Path1ModelShape = DEFAULT_PATH1_MODEL_SHAPE,
    profile: ReferenceSsmProfile = ReferenceSsmProfile.MAMBA3_SISO_RUNTIME,
) -> Path1VariantSpec:
    final_norm = "rmsnorm" if profile in {ReferenceSsmProfile.MAMBA3_SISO_REFERENCE, ReferenceSsmProfile.MAMBA3_SISO_RUNTIME} else "identity"
    return Path1VariantSpec(
        kind=Path1VariantKind.REFERENCE_SSM_HYBRID,
        label=_variant_label("reference-ssm-hybrid", profile.value),
        shape=shape,
        layer_schedule=_alternating_schedule(shape.total_layers, HybridAttentionLayerRole.REFERENCE_SSM),
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
) -> Path1VariantSpec:
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
        ),
        shape=shape,
        layer_schedule=_alternating_schedule(shape.total_layers, HybridAttentionLayerRole.PRIMITIVE),
        primitive_profile=primitive_profile,
        primitive_residual_mode=residual_mode,
        primitive_readout_mode=readout_mode,
        primitive_norm_mode=norm_mode,
        primitive_wrapper_mode=wrapper_mode,
        primitive_execution_profile=execution_profile,
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
        ),
    )
    matrix.validate()
    return matrix
