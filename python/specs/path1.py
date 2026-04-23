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


class FeedForwardProfile(StringEnum):
    STANDARD = "standard"
    EML_TREE = "eml-tree"
    MLP_EML_GATED = "mlp-eml-gated"
    MLP_EML_ROUTED = "mlp-eml-routed"
    TINY_MLP_GATED = "tiny-mlp-gated"
    TINY_GLU_GATED = "tiny-glu-gated"
    GENERIC_TREE_GATED = "generic-tree-gated"


class AttentionProfile(StringEnum):
    STANDARD = "standard"
    MODA_DEPTH_KV = "moda-depth-kv"
    PAPER_MODA_DEPTH_KV = "paper-moda-depth-kv"


class RecurrentHaltingProfile(StringEnum):
    FIXED = "fixed"
    ACCELERATION = "acceleration"
    VECTOR_ACCELERATION = "vector-acceleration"
    NORMALIZED_STEP_NORM = "normalized-step-norm"


class TokenRoutingProfile(StringEnum):
    NONE = "none"
    CAUSAL_TOPK_BLOCK = "causal-topk-block"
    MOD_TRAIN_TOPC_BLOCK = "mod-train-topc-block"
    SOFT_GATE_BLOCK = "soft-gate-block"
    ROTARY_SOFT_GATE_BLOCK = "rotary-soft-gate-block"


class RecurrentTokenRoutingProfile(StringEnum):
    NONE = "none"
    CAUSAL_TOPK_STATE = "causal-topk-state"


class Path1ScaffoldProfile(StringEnum):
    STANDARD = "standard"
    PR5_HYBRID_GDN = "pr5-hybrid-gdn"
    PARCAE_LOOPED_ATTENTION = "parcae-looped-attention"
    PARCAE_BX_LOOPED_ATTENTION = "parcae-bx-looped-attention"
    PARCAE_P20_CONTROL_LOOPED_ATTENTION = "parcae-p20-control-looped-attention"
    PARCAE_P20_THIN_CONTROL_LOOPED_ATTENTION = (
        "parcae-p20-thin-control-looped-attention"
    )
    PARCAE_P20_THIN_GATE_CONTROL_LOOPED_ATTENTION = (
        "parcae-p20-thin-gate-control-looped-attention"
    )
    PARCAE_P20_QUARTER_CONTROL_LOOPED_ATTENTION = (
        "parcae-p20-quarter-control-looped-attention"
    )
    PARCAE_P20_THIN_VALUE_CONTROL_LOOPED_ATTENTION = (
        "parcae-p20-thin-value-control-looped-attention"
    )
    PARCAE_P20_THIN_GATE_BASEBLEND_LOOPED_ATTENTION = (
        "parcae-p20-thin-gate-baseblend-looped-attention"
    )
    PARCAE_P20_THIN_BASEBLEND_CONTROL_LOOPED_ATTENTION = (
        "parcae-p20-thin-baseblend-control-looped-attention"
    )
    PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION = (
        "parcae-p20-mod-gate-bias-looped-attention"
    )
    PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION = (
        "parcae-p20-mod-value-scale-looped-attention"
    )
    FIXED_LOOPED_LM = "fixed-looped-lm"
    LOOPED_ADDITIVE_INPUT = "looped-additive-input"
    HUGINN_ADAPTER_RECURRENCE = "huginn-adapter-recurrence"
    UNIVERSAL_TRANSFORMER = "universal-transformer"
    UNIVERSAL_TRANSFORMER_ACT = "universal-transformer-act"
    OURO_LEARNED_EXIT = "ouro-learned-exit"
    RRT_CYCLE = "rrt-cycle"
    MOR_EXPERT_CHOICE = "mor-expert-choice"


class HybridAttentionLayerRole(StringEnum):
    EXACT_ATTENTION = "exact-attention"
    SHARED_EXACT_ATTENTION = "shared-exact-attention"
    REFERENCE_SSM = "reference-ssm"
    PRIMITIVE = "primitive"


class ReferenceSsmProfile(StringEnum):
    GATED_DELTANET_FLA = "gated-deltanet-fla"
    GATED_DELTANET_FLA_CONTROL_SHELL = "gated-deltanet-fla-control-shell"
    GATED_DELTANET_FLA_P20_CONTROL_TINY = "gated-deltanet-fla-p20-control-tiny"
    GATED_DELTANET_FLA_P20_COMPAT = "gated-deltanet-fla-p20-compatible"
    GATED_DELTANET_FLA_P20_MULTI_READ = "gated-deltanet-fla-p20-multi-read"
    GATED_DELTANET_MAMBA3_TORCH = "gated-deltanet-mamba3-torch"
    GATED_DELTANET_P20_FUSED_ALL_TORCH = "gated-deltanet-p20-fused-all-torch"
    GATED_DELTANET_P20_FUSED_BETA_TORCH = "gated-deltanet-p20-fused-beta-torch"
    GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH = (
        "gated-deltanet-p20-fused-multi-read-torch"
    )
    GATED_DELTANET_P20_FUSED_QKV_TORCH = "gated-deltanet-p20-fused-qkv-torch"
    GATED_DELTANET_P20_FUSED_RESIDUAL_READOUT_TORCH = (
        "gated-deltanet-p20-fused-residual-readout-torch"
    )
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
            ReferenceSsmProfile.GATED_DELTANET_FLA_CONTROL_SHELL,
            ReferenceSsmProfile.GATED_DELTANET_FLA_P20_CONTROL_TINY,
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
    def is_fla_gdn_control_shell(self) -> bool:
        return self is ReferenceSsmProfile.GATED_DELTANET_FLA_CONTROL_SHELL

    @property
    def is_fla_gdnp_control_conditioned(self) -> bool:
        return self is ReferenceSsmProfile.GATED_DELTANET_FLA_P20_CONTROL_TINY

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
        raise ValueError(
            f"reference SSM profile {self.value} is not an FLA-compatible GDN/P20 profile"
        )

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
        raise ValueError(
            f"reference SSM profile {self.value} is not a fused GDN/P20 profile"
        )

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
            ReferenceSsmProfile.GATED_DELTANET_FLA_CONTROL_SHELL,
            ReferenceSsmProfile.GATED_DELTANET_FLA_P20_CONTROL_TINY,
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
    reference_ssm_profile_schedule: tuple[ReferenceSsmProfile, ...] | None = None
    reference_p20_ramp_init: float = 0.01
    primitive_profile: PrimitiveProfile | None = None
    primitive_residual_mode: PrimitiveResidualMode | None = None
    primitive_readout_mode: PrimitiveReadoutMode | None = None
    primitive_norm_mode: PrimitiveNormMode | None = None
    primitive_wrapper_mode: PrimitiveWrapperMode | None = None
    primitive_execution_profile: PrimitiveExecutionProfile | None = None
    primitive_state_transform_mode: PrimitiveStateTransformMode | None = None
    feed_forward_profile: FeedForwardProfile = FeedForwardProfile.STANDARD
    feed_forward_layer_indices: tuple[int, ...] | None = None
    eml_slot_count: int = 8
    eml_tree_depth: int = 3
    eml_route_fraction: float = 0.25
    parcae_loop_count: int = 2
    parcae_p20_value_scale: float = 1.0
    final_norm_kind: str = "identity"
    scaffold_profile: Path1ScaffoldProfile = Path1ScaffoldProfile.STANDARD
    attention_profile: AttentionProfile = AttentionProfile.STANDARD
    depth_memory_layers: int = 2
    recurrent_halting_profile: RecurrentHaltingProfile = RecurrentHaltingProfile.FIXED
    recurrent_min_steps: int = 1
    recurrent_halting_threshold: float = 0.01
    token_routing_profile: TokenRoutingProfile = TokenRoutingProfile.NONE
    token_route_fraction: float = 0.25
    token_routing_layer_indices: tuple[int, ...] | None = None
    recurrent_token_routing_profile: RecurrentTokenRoutingProfile = (
        RecurrentTokenRoutingProfile.NONE
    )
    recurrent_token_route_fraction: float = 0.25
    act_halting_threshold: float = 0.99
    act_ponder_loss_weight: float = 0.01
    ouro_entropy_weight: float = 0.05
    ouro_q_exit_threshold: float = 0.5
    mor_router_aux_loss_weight: float = 0.01
    mor_update_scale: float = 0.1

    def validate(self) -> None:
        self.shape.validate()
        ensure_positive(self.eml_slot_count, "path1_variant.eml_slot_count")
        ensure_positive(self.eml_tree_depth, "path1_variant.eml_tree_depth")
        ensure_positive(self.parcae_loop_count, "path1_variant.parcae_loop_count")
        if self.parcae_p20_value_scale <= 0.0:
            raise ValidationError(
                "path1_variant.parcae_p20_value_scale must be greater than zero, "
                f"got {self.parcae_p20_value_scale}"
            )
        ensure_positive(self.depth_memory_layers, "path1_variant.depth_memory_layers")
        ensure_positive(self.recurrent_min_steps, "path1_variant.recurrent_min_steps")
        if self.recurrent_min_steps > self.parcae_loop_count:
            raise ValidationError(
                "path1_variant.recurrent_min_steps must be <= parcae_loop_count, "
                f"got {self.recurrent_min_steps} and {self.parcae_loop_count}"
            )
        if self.recurrent_halting_threshold <= 0.0:
            raise ValidationError(
                "path1_variant.recurrent_halting_threshold must be greater than zero, "
                f"got {self.recurrent_halting_threshold}"
            )
        if not 0.0 < self.act_halting_threshold < 1.0:
            raise ValidationError(
                "path1_variant.act_halting_threshold must be in (0, 1), "
                f"got {self.act_halting_threshold}"
            )
        if self.act_ponder_loss_weight < 0.0:
            raise ValidationError(
                "path1_variant.act_ponder_loss_weight must be non-negative, "
                f"got {self.act_ponder_loss_weight}"
            )
        if self.ouro_entropy_weight < 0.0:
            raise ValidationError(
                "path1_variant.ouro_entropy_weight must be non-negative, "
                f"got {self.ouro_entropy_weight}"
            )
        if not 0.0 < self.ouro_q_exit_threshold <= 1.0:
            raise ValidationError(
                "path1_variant.ouro_q_exit_threshold must be in (0, 1], "
                f"got {self.ouro_q_exit_threshold}"
            )
        if self.mor_router_aux_loss_weight < 0.0:
            raise ValidationError(
                "path1_variant.mor_router_aux_loss_weight must be non-negative, "
                f"got {self.mor_router_aux_loss_weight}"
            )
        if not 0.0 < self.mor_update_scale <= 1.0:
            raise ValidationError(
                "path1_variant.mor_update_scale must be in (0, 1], "
                f"got {self.mor_update_scale}"
            )
        if not 0.0 < self.eml_route_fraction <= 1.0:
            raise ValidationError(
                "path1_variant.eml_route_fraction must be in (0, 1], "
                f"got {self.eml_route_fraction}"
            )
        if not 0.0 < self.token_route_fraction <= 1.0:
            raise ValidationError(
                "path1_variant.token_route_fraction must be in (0, 1], "
                f"got {self.token_route_fraction}"
            )
        if not 0.0 < self.recurrent_token_route_fraction <= 1.0:
            raise ValidationError(
                "path1_variant.recurrent_token_route_fraction must be in (0, 1], "
                f"got {self.recurrent_token_route_fraction}"
            )
        if (
            self.feed_forward_profile is FeedForwardProfile.STANDARD
            and self.feed_forward_layer_indices is not None
        ):
            raise ValidationError(
                "standard feed-forward profile must not set feed_forward_layer_indices"
            )
        if self.feed_forward_layer_indices is not None:
            if not self.feed_forward_layer_indices:
                raise ValidationError(
                    "feed_forward_layer_indices must not be empty when provided"
                )
            if len(set(self.feed_forward_layer_indices)) != len(
                self.feed_forward_layer_indices
            ):
                raise ValidationError(
                    "feed_forward_layer_indices must not contain duplicates"
                )
            for layer_index in self.feed_forward_layer_indices:
                if layer_index < 0 or layer_index >= self.shape.total_layers:
                    raise ValidationError(
                        "feed_forward_layer_indices entries must be valid layer indices, "
                        f"got {layer_index} for total_layers={self.shape.total_layers}"
                    )
        if (
            self.token_routing_profile is TokenRoutingProfile.NONE
            and self.token_routing_layer_indices is not None
        ):
            raise ValidationError(
                "token_routing_layer_indices requires a non-none token_routing_profile"
            )
        if self.token_routing_layer_indices is not None:
            if not self.token_routing_layer_indices:
                raise ValidationError(
                    "token_routing_layer_indices must not be empty when provided"
                )
            if len(set(self.token_routing_layer_indices)) != len(
                self.token_routing_layer_indices
            ):
                raise ValidationError(
                    "token_routing_layer_indices must not contain duplicates"
                )
            for layer_index in self.token_routing_layer_indices:
                if layer_index < 0 or layer_index >= self.shape.total_layers:
                    raise ValidationError(
                        "token_routing_layer_indices entries must be valid layer indices, "
                        f"got {layer_index} for total_layers={self.shape.total_layers}"
                    )
        if not self.label.strip():
            raise ValidationError("path1_variant.label must not be empty")
        if len(self.layer_schedule) != self.shape.total_layers:
            raise ValidationError(
                "path1_variant.layer_schedule length must match shape.total_layers"
            )
        looped_transformer_scaffolds = {
            Path1ScaffoldProfile.FIXED_LOOPED_LM,
            Path1ScaffoldProfile.LOOPED_ADDITIVE_INPUT,
            Path1ScaffoldProfile.HUGINN_ADAPTER_RECURRENCE,
        }
        universal_transformer_scaffolds = {
            Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER,
            Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER_ACT,
        }
        learned_exit_scaffolds = {
            Path1ScaffoldProfile.OURO_LEARNED_EXIT,
        }
        recursive_compression_scaffolds = {
            Path1ScaffoldProfile.RRT_CYCLE,
        }
        mixture_of_recursions_scaffolds = {
            Path1ScaffoldProfile.MOR_EXPERT_CHOICE,
        }
        tied_recurrent_attention_scaffolds = (
            looped_transformer_scaffolds
            | universal_transformer_scaffolds
            | learned_exit_scaffolds
            | recursive_compression_scaffolds
            | mixture_of_recursions_scaffolds
        )
        if (
            self.scaffold_profile is Path1ScaffoldProfile.RRT_CYCLE
            and self.shape.total_layers % self.parcae_loop_count != 0
        ):
            raise ValidationError(
                "rrt-cycle requires shape.total_layers divisible by parcae_loop_count, "
                f"got {self.shape.total_layers} and {self.parcae_loop_count}"
            )
        if (
            self.scaffold_profile is Path1ScaffoldProfile.MOR_EXPERT_CHOICE
            and self.shape.total_layers < 3
        ):
            raise ValidationError(
                "mor-expert-choice requires at least three stored layers "
                "(unique first, shared middle, unique last)"
            )
        exact_attention_layers = sum(
            1 for role in self.layer_schedule if role in _EXACT_ATTENTION_ROLES
        )
        if exact_attention_layers == 0:
            raise ValidationError(
                "path1_variant must retain at least one exact-attention layer"
            )
        if self.kind is Path1VariantKind.ATTENTION_ONLY:
            if any(role not in _EXACT_ATTENTION_ROLES for role in self.layer_schedule):
                raise ValidationError(
                    "attention-only variant must contain only exact-attention layers"
                )
            if self.feed_forward_layer_indices is not None:
                for layer_index in self.feed_forward_layer_indices:
                    if self.layer_schedule[layer_index] not in _EXACT_ATTENTION_ROLES:
                        raise ValidationError(
                            "feed_forward_layer_indices may only target exact-attention layers"
                        )
            if self.token_routing_layer_indices is not None:
                for layer_index in self.token_routing_layer_indices:
                    if self.layer_schedule[layer_index] not in _EXACT_ATTENTION_ROLES:
                        raise ValidationError(
                            "token_routing_layer_indices may only target exact-attention layers"
                        )
            if self.reference_ssm_profile_schedule is not None:
                raise ValidationError(
                    "attention-only variant must not set reference_ssm_profile_schedule"
                )
            if (
                self.token_routing_profile is not TokenRoutingProfile.NONE
                and self.attention_profile is not AttentionProfile.STANDARD
            ):
                raise ValidationError(
                    "token block routing currently supports standard attention only"
                )
            if (
                self.recurrent_token_routing_profile
                is RecurrentTokenRoutingProfile.CAUSAL_TOPK_STATE
            ):
                if self.attention_profile is not AttentionProfile.STANDARD:
                    raise ValidationError(
                        "causal top-k recurrent token routing currently supports standard attention only"
                    )
                if self.scaffold_profile not in {
                    Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_QUARTER_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_VALUE_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_BASEBLEND_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_BASEBLEND_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION,
                }:
                    raise ValidationError(
                        "recurrent token routing requires a Parcae-family looped scaffold"
                    )
            if (
                self.recurrent_halting_profile is not RecurrentHaltingProfile.FIXED
                and self.scaffold_profile
                not in {
                    Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_QUARTER_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_VALUE_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_BASEBLEND_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_BASEBLEND_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION,
                }
            ):
                raise ValidationError(
                    "adaptive recurrent halting requires a Parcae-family looped scaffold"
                )
            if self.scaffold_profile not in {
                Path1ScaffoldProfile.STANDARD,
                Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
                Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
                Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
                Path1ScaffoldProfile.PARCAE_P20_THIN_CONTROL_LOOPED_ATTENTION,
                Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_CONTROL_LOOPED_ATTENTION,
                Path1ScaffoldProfile.PARCAE_P20_QUARTER_CONTROL_LOOPED_ATTENTION,
                Path1ScaffoldProfile.PARCAE_P20_THIN_VALUE_CONTROL_LOOPED_ATTENTION,
                Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_BASEBLEND_LOOPED_ATTENTION,
                Path1ScaffoldProfile.PARCAE_P20_THIN_BASEBLEND_CONTROL_LOOPED_ATTENTION,
                Path1ScaffoldProfile.PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION,
                Path1ScaffoldProfile.PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION,
                *tied_recurrent_attention_scaffolds,
            }:
                raise ValidationError(
                    "attention-only variant may only use standard, Parcae-family, looped-transformer, Universal Transformer, learned-exit, recursive-compression, or mixture-of-recursions scaffold"
                )
            if self.scaffold_profile in tied_recurrent_attention_scaffolds:
                if any(
                    role is not HybridAttentionLayerRole.EXACT_ATTENTION
                    for role in self.layer_schedule
                ):
                    raise ValidationError(
                        "tied recurrent attention scaffold requires exact-attention layers only"
                    )
                if self.attention_profile is not AttentionProfile.STANDARD:
                    raise ValidationError(
                        "tied recurrent attention scaffold currently supports standard attention only"
                    )
                if self.feed_forward_profile is not FeedForwardProfile.STANDARD:
                    raise ValidationError(
                        "tied recurrent attention scaffold currently supports standard feed-forward blocks only"
                    )
                if self.token_routing_profile is not TokenRoutingProfile.NONE:
                    raise ValidationError(
                        "tied recurrent attention scaffold does not support token block routing"
                    )
            if (
                self.attention_profile is AttentionProfile.PAPER_MODA_DEPTH_KV
                and self.scaffold_profile is not Path1ScaffoldProfile.STANDARD
            ):
                raise ValidationError(
                    "paper-moda-depth-kv currently supports only the standard scaffold"
                )
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
                raise ValidationError(
                    "attention-only variant must not set reference or primitive options"
                )
        elif self.kind is Path1VariantKind.REFERENCE_SSM_HYBRID:
            if self.reference_ssm_profile is None:
                raise ValidationError(
                    "reference-ssm-hybrid variant must set reference_ssm_profile"
                )
            if self.feed_forward_layer_indices is not None:
                for layer_index in self.feed_forward_layer_indices:
                    if self.layer_schedule[layer_index] not in _EXACT_ATTENTION_ROLES:
                        raise ValidationError(
                            "feed_forward_layer_indices may only target exact-attention layers"
                        )
            if self.token_routing_layer_indices is not None:
                for layer_index in self.token_routing_layer_indices:
                    if self.layer_schedule[layer_index] not in _EXACT_ATTENTION_ROLES:
                        raise ValidationError(
                            "token_routing_layer_indices may only target exact-attention layers"
                        )
            if self.primitive_profile is not None:
                raise ValidationError(
                    "reference-ssm-hybrid variant must not set primitive_profile"
                )
            if (
                self.scaffold_profile is Path1ScaffoldProfile.PR5_HYBRID_GDN
                and self.attention_profile is not AttentionProfile.STANDARD
            ):
                raise ValidationError(
                    "PR5 scaffold does not support depth-augmented attention"
                )
            if any(
                role
                not in {*_EXACT_ATTENTION_ROLES, HybridAttentionLayerRole.REFERENCE_SSM}
                for role in self.layer_schedule
            ):
                raise ValidationError(
                    "reference-ssm-hybrid schedule may contain only exact-attention and reference-SSM roles"
                )
            if (
                self.recurrent_token_routing_profile
                is not RecurrentTokenRoutingProfile.NONE
            ):
                raise ValidationError(
                    "reference-ssm-hybrid variant does not support recurrent token routing"
                )
            if self.reference_ssm_profile_schedule is not None:
                reference_layer_count = sum(
                    1
                    for role in self.layer_schedule
                    if role is HybridAttentionLayerRole.REFERENCE_SSM
                )
                if len(self.reference_ssm_profile_schedule) != reference_layer_count:
                    raise ValidationError(
                        "reference_ssm_profile_schedule length must match the number of reference-SSM layers"
                    )
        elif self.kind is Path1VariantKind.PRIMITIVE_HYBRID:
            if self.primitive_profile is None:
                raise ValidationError(
                    "primitive-hybrid variant must set primitive_profile"
                )
            if self.reference_ssm_profile is not None:
                raise ValidationError(
                    "primitive-hybrid variant must not set reference_ssm_profile"
                )
            if self.reference_ssm_profile_schedule is not None:
                raise ValidationError(
                    "primitive-hybrid variant must not set reference_ssm_profile_schedule"
                )
            if self.scaffold_profile is not Path1ScaffoldProfile.STANDARD:
                raise ValidationError(
                    "primitive-hybrid variant must use the standard scaffold"
                )
            if self.recurrent_halting_profile is not RecurrentHaltingProfile.FIXED:
                raise ValidationError(
                    "primitive-hybrid variant does not support recurrent halting"
                )
            if self.feed_forward_layer_indices is not None:
                for layer_index in self.feed_forward_layer_indices:
                    if self.layer_schedule[layer_index] not in _EXACT_ATTENTION_ROLES:
                        raise ValidationError(
                            "feed_forward_layer_indices may only target exact-attention layers"
                        )
            if self.token_routing_layer_indices is not None:
                for layer_index in self.token_routing_layer_indices:
                    if self.layer_schedule[layer_index] not in _EXACT_ATTENTION_ROLES:
                        raise ValidationError(
                            "token_routing_layer_indices may only target exact-attention layers"
                        )
            if any(
                role
                not in {*_EXACT_ATTENTION_ROLES, HybridAttentionLayerRole.PRIMITIVE}
                for role in self.layer_schedule
            ):
                raise ValidationError(
                    "primitive-hybrid schedule may contain only exact-attention and primitive roles"
                )
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
            if (
                self.recurrent_token_routing_profile
                is not RecurrentTokenRoutingProfile.NONE
            ):
                raise ValidationError(
                    "primitive-hybrid variant does not support recurrent token routing"
                )
        else:
            raise ValidationError(f"unsupported path1 variant kind: {self.kind}")
        if (
            self.scaffold_profile is Path1ScaffoldProfile.PR5_HYBRID_GDN
            and self.token_routing_profile is not TokenRoutingProfile.NONE
        ):
            raise ValidationError("PR5 scaffold does not support token block routing")
        if (
            self.token_routing_profile is not TokenRoutingProfile.NONE
            and self.attention_profile is not AttentionProfile.STANDARD
        ):
            raise ValidationError(
                "token block routing currently supports standard attention only"
            )
        if self.final_norm_kind not in {"identity", "rmsnorm"}:
            raise ValidationError(
                f"path1_variant.final_norm_kind must be identity|rmsnorm, got {self.final_norm_kind}"
            )
        if not (0.0 < self.reference_p20_ramp_init < 1.0):
            raise ValidationError(
                "path1_variant.reference_p20_ramp_init must be in (0, 1), "
                f"got {self.reference_p20_ramp_init}"
            )

    def reference_profile_for_ordinal(
        self, reference_ordinal: int
    ) -> ReferenceSsmProfile:
        if self.reference_ssm_profile_schedule is None:
            if self.reference_ssm_profile is None:
                raise ValidationError(
                    "reference profile requested for non-reference variant"
                )
            return self.reference_ssm_profile
        return self.reference_ssm_profile_schedule[reference_ordinal]


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
                raise ValidationError(
                    "Path1 baseline matrix variants must share the same model shape"
                )


def _attention_schedule(total_layers: int) -> tuple[HybridAttentionLayerRole, ...]:
    return tuple(HybridAttentionLayerRole.EXACT_ATTENTION for _ in range(total_layers))


def _alternating_schedule(
    total_layers: int, odd_role: HybridAttentionLayerRole
) -> tuple[HybridAttentionLayerRole, ...]:
    return tuple(
        HybridAttentionLayerRole.EXACT_ATTENTION if index % 2 == 0 else odd_role
        for index in range(total_layers)
    )


def _variant_label(*parts: str) -> str:
    return "-".join(part for part in parts if part)


def parse_layer_schedule_spec(schedule: str) -> tuple[HybridAttentionLayerRole, ...]:
    normalized = (
        schedule.strip()
        .replace(",", "")
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .upper()
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
        (
            "a"
            if role is HybridAttentionLayerRole.EXACT_ATTENTION
            else (
                "s"
                if role is HybridAttentionLayerRole.SHARED_EXACT_ATTENTION
                else "r" if role is HybridAttentionLayerRole.REFERENCE_SSM else "p"
            )
        )
        for role in schedule
    )


def parse_reference_ssm_profile_schedule_spec(
    schedule: str,
) -> tuple[ReferenceSsmProfile, ...]:
    tokens = tuple(
        token.strip() for token in schedule.replace(",", " ").split() if token.strip()
    )
    if not tokens:
        raise ValidationError(
            "reference SSM profile schedule override must not be empty"
        )
    try:
        return tuple(ReferenceSsmProfile(token) for token in tokens)
    except ValueError as exc:
        raise ValidationError(
            "reference SSM profile schedule contains an unknown profile"
        ) from exc


def parse_layer_index_spec(indices: str) -> tuple[int, ...]:
    tokens = tuple(
        token.strip() for token in indices.replace(",", " ").split() if token.strip()
    )
    if not tokens:
        raise ValidationError("layer index override must not be empty")
    try:
        parsed = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise ValidationError(
            "layer index override must contain only integers"
        ) from exc
    if len(set(parsed)) != len(parsed):
        raise ValidationError("layer index override must not contain duplicates")
    if any(index < 0 for index in parsed):
        raise ValidationError("layer index override must not contain negative indices")
    return tuple(sorted(parsed))


def layer_index_signature(indices: tuple[int, ...] | None) -> str:
    if indices is None:
        return "all"
    return "-".join(str(index) for index in indices)


def fraction_signature(value: float) -> str:
    return f"{int(round(value * 100))}pct"


def threshold_signature(value: float) -> str:
    return f"{value:.3g}".replace(".", "p").replace("-", "m").replace("+", "")


_REFERENCE_PROFILE_LABEL_ALIASES = {
    ReferenceSsmProfile.GATED_DELTANET_FLA: "gdn-fla",
    ReferenceSsmProfile.GATED_DELTANET_FLA_CONTROL_SHELL: "gdn-fla-shell",
    ReferenceSsmProfile.GATED_DELTANET_FLA_P20_CONTROL_TINY: "gdn-fla-p20-ctrl",
    ReferenceSsmProfile.GATED_DELTANET_FLA_P20_COMPAT: "gdn-fla-p20",
    ReferenceSsmProfile.GATED_DELTANET_FLA_P20_MULTI_READ: "gdn-fla-p20-mr",
    ReferenceSsmProfile.GATED_DELTANET_MAMBA3_TORCH: "gdn-m3",
    ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_ALL_TORCH: "gdnp-all",
    ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_BETA_TORCH: "gdnp-beta",
    ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH: "gdnp-mr",
    ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_QKV_TORCH: "gdnp-qkv",
    ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_RESIDUAL_READOUT_TORCH: "gdnp-rr",
    ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_TORCH: "gdnp-value",
    ReferenceSsmProfile.GATED_DELTANET_P20_MAMBA3_TORCH: "gdn-p20-m3",
    ReferenceSsmProfile.GATED_DELTANET_P20_THIN_TORCH: "gdn-p20-thin",
    ReferenceSsmProfile.GATED_DELTANET_P20_TORCH: "gdn-p20",
    ReferenceSsmProfile.GATED_DELTANET_TORCH: "gdn",
    ReferenceSsmProfile.MAMBA3_MIMO_REFERENCE: "m3-mimo-ref",
    ReferenceSsmProfile.MAMBA3_SISO_REFERENCE: "m3-siso-ref",
    ReferenceSsmProfile.MAMBA3_SISO_RUNTIME: "m3-siso-rt",
    ReferenceSsmProfile.P20_MAMBA3_TORCH: "p20-m3",
    ReferenceSsmProfile.P20_THIN_TORCH: "p20-thin",
    ReferenceSsmProfile.P20_TORCH: "p20",
}


def reference_profile_schedule_signature(
    schedule: tuple[ReferenceSsmProfile, ...] | None,
) -> str:
    if schedule is None:
        return ""
    segments: list[str] = []
    current_token: str | None = None
    current_count = 0
    for profile in schedule:
        token = _REFERENCE_PROFILE_LABEL_ALIASES[profile]
        if token == current_token:
            current_count += 1
            continue
        if current_token is not None:
            segments.append(
                f"{current_token}x{current_count}"
                if current_count > 1
                else current_token
            )
        current_token = token
        current_count = 1
    if current_token is not None:
        segments.append(
            f"{current_token}x{current_count}" if current_count > 1 else current_token
        )
    return "profiles-" + "-".join(segments)


def phase1_attention_only_variant(
    shape: Path1ModelShape = DEFAULT_PATH1_MODEL_SHAPE,
    layer_schedule: tuple[HybridAttentionLayerRole, ...] | None = None,
    feed_forward_profile: FeedForwardProfile = FeedForwardProfile.STANDARD,
    feed_forward_layer_indices: tuple[int, ...] | None = None,
    eml_slot_count: int = 8,
    eml_tree_depth: int = 3,
    eml_route_fraction: float = 0.25,
    scaffold_profile: Path1ScaffoldProfile = Path1ScaffoldProfile.STANDARD,
    parcae_loop_count: int = 2,
    parcae_p20_value_scale: float = 1.0,
    attention_profile: AttentionProfile = AttentionProfile.STANDARD,
    depth_memory_layers: int = 2,
    recurrent_halting_profile: RecurrentHaltingProfile = RecurrentHaltingProfile.FIXED,
    recurrent_min_steps: int = 1,
    recurrent_halting_threshold: float = 0.01,
    token_routing_profile: TokenRoutingProfile = TokenRoutingProfile.NONE,
    token_route_fraction: float = 0.25,
    token_routing_layer_indices: tuple[int, ...] | None = None,
    recurrent_token_routing_profile: RecurrentTokenRoutingProfile = (
        RecurrentTokenRoutingProfile.NONE
    ),
    recurrent_token_route_fraction: float = 0.25,
    act_halting_threshold: float = 0.99,
    act_ponder_loss_weight: float = 0.01,
    ouro_entropy_weight: float = 0.05,
    ouro_q_exit_threshold: float = 0.5,
    mor_router_aux_loss_weight: float = 0.01,
    mor_update_scale: float = 0.1,
) -> Path1VariantSpec:
    schedule = layer_schedule or _attention_schedule(shape.total_layers)
    default_schedule = _attention_schedule(shape.total_layers)
    schedule_suffix = (
        f"schedule-{layer_schedule_signature(schedule)}"
        if schedule != default_schedule
        else ""
    )
    layer_suffix = (
        f"layers{layer_index_signature(feed_forward_layer_indices)}"
        if feed_forward_profile is not FeedForwardProfile.STANDARD
        and feed_forward_layer_indices is not None
        else ""
    )
    route_suffix = (
        f"route{fraction_signature(eml_route_fraction)}"
        if feed_forward_profile is FeedForwardProfile.MLP_EML_ROUTED
        else ""
    )
    ffn_suffix = (
        _variant_label(
            feed_forward_profile.value,
            f"slots{eml_slot_count}",
            f"depth{eml_tree_depth}",
            route_suffix,
            layer_suffix,
        )
        if feed_forward_profile is not FeedForwardProfile.STANDARD
        else ""
    )
    scaffold_suffix = (
        _variant_label(
            scaffold_profile.value,
            f"loops{parcae_loop_count}",
            f"route{fraction_signature(recurrent_token_route_fraction)}",
        )
        if scaffold_profile is Path1ScaffoldProfile.MOR_EXPERT_CHOICE
        else ""
    )
    scaffold_suffix = scaffold_suffix or (
        _variant_label(
            scaffold_profile.value,
            f"loops{parcae_loop_count}",
            (
                f"vscale{threshold_signature(parcae_p20_value_scale)}"
                if parcae_p20_value_scale != 1.0
                else ""
            ),
            (
                f"route{fraction_signature(token_route_fraction)}"
                if scaffold_profile
                in {
                    Path1ScaffoldProfile.PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION,
                }
                else ""
            ),
        )
        if scaffold_profile
        in {
            Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_QUARTER_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_VALUE_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_BASEBLEND_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_BASEBLEND_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION,
            Path1ScaffoldProfile.FIXED_LOOPED_LM,
            Path1ScaffoldProfile.LOOPED_ADDITIVE_INPUT,
            Path1ScaffoldProfile.HUGINN_ADAPTER_RECURRENCE,
            Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER,
            Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER_ACT,
            Path1ScaffoldProfile.OURO_LEARNED_EXIT,
            Path1ScaffoldProfile.RRT_CYCLE,
            Path1ScaffoldProfile.MOR_EXPERT_CHOICE,
        }
        else ""
    )
    attention_suffix = (
        _variant_label(attention_profile.value, f"depthmem{depth_memory_layers}")
        if attention_profile is not AttentionProfile.STANDARD
        else ""
    )
    halting_suffix = (
        _variant_label(
            f"halt-{recurrent_halting_profile.value}",
            f"min{recurrent_min_steps}",
            f"t{threshold_signature(recurrent_halting_threshold)}",
        )
        if recurrent_halting_profile is not RecurrentHaltingProfile.FIXED
        else ""
    )
    token_routing_suffix = (
        _variant_label(
            token_routing_profile.value,
            f"route{fraction_signature(token_route_fraction)}",
            f"layers{layer_index_signature(token_routing_layer_indices)}",
        )
        if token_routing_profile is not TokenRoutingProfile.NONE
        else ""
    )
    recurrent_token_suffix = (
        _variant_label(
            recurrent_token_routing_profile.value,
            f"route{fraction_signature(recurrent_token_route_fraction)}",
        )
        if recurrent_token_routing_profile is not RecurrentTokenRoutingProfile.NONE
        else ""
    )
    return Path1VariantSpec(
        kind=Path1VariantKind.ATTENTION_ONLY,
        label=_variant_label(
            "attention-only",
            attention_suffix,
            token_routing_suffix,
            scaffold_suffix,
            halting_suffix,
            recurrent_token_suffix,
            ffn_suffix,
            schedule_suffix,
        ),
        shape=shape,
        layer_schedule=schedule,
        feed_forward_profile=feed_forward_profile,
        feed_forward_layer_indices=feed_forward_layer_indices,
        eml_slot_count=eml_slot_count,
        eml_tree_depth=eml_tree_depth,
        eml_route_fraction=eml_route_fraction,
        scaffold_profile=scaffold_profile,
        parcae_loop_count=parcae_loop_count,
        parcae_p20_value_scale=parcae_p20_value_scale,
        attention_profile=attention_profile,
        depth_memory_layers=depth_memory_layers,
        recurrent_halting_profile=recurrent_halting_profile,
        recurrent_min_steps=recurrent_min_steps,
        recurrent_halting_threshold=recurrent_halting_threshold,
        token_routing_profile=token_routing_profile,
        token_route_fraction=token_route_fraction,
        token_routing_layer_indices=token_routing_layer_indices,
        recurrent_token_routing_profile=recurrent_token_routing_profile,
        recurrent_token_route_fraction=recurrent_token_route_fraction,
        act_halting_threshold=act_halting_threshold,
        act_ponder_loss_weight=act_ponder_loss_weight,
        ouro_entropy_weight=ouro_entropy_weight,
        ouro_q_exit_threshold=ouro_q_exit_threshold,
        mor_router_aux_loss_weight=mor_router_aux_loss_weight,
        mor_update_scale=mor_update_scale,
    )


def phase1_reference_ssm_variant(
    shape: Path1ModelShape = DEFAULT_PATH1_MODEL_SHAPE,
    profile: ReferenceSsmProfile = ReferenceSsmProfile.MAMBA3_SISO_RUNTIME,
    layer_schedule: tuple[HybridAttentionLayerRole, ...] | None = None,
    profile_schedule: tuple[ReferenceSsmProfile, ...] | None = None,
    scaffold_profile: Path1ScaffoldProfile = Path1ScaffoldProfile.STANDARD,
    reference_p20_ramp_init: float = 0.01,
    feed_forward_profile: FeedForwardProfile = FeedForwardProfile.STANDARD,
    feed_forward_layer_indices: tuple[int, ...] | None = None,
    eml_slot_count: int = 8,
    eml_tree_depth: int = 3,
    eml_route_fraction: float = 0.25,
    attention_profile: AttentionProfile = AttentionProfile.STANDARD,
    depth_memory_layers: int = 2,
    token_routing_profile: TokenRoutingProfile = TokenRoutingProfile.NONE,
    token_route_fraction: float = 0.25,
    token_routing_layer_indices: tuple[int, ...] | None = None,
) -> Path1VariantSpec:
    profiles_for_norm = profile_schedule or (profile,)
    final_norm = (
        "rmsnorm"
        if any(
            candidate.is_gated_deltanet
            or candidate
            in {
                ReferenceSsmProfile.MAMBA3_SISO_REFERENCE,
                ReferenceSsmProfile.MAMBA3_SISO_RUNTIME,
                ReferenceSsmProfile.P20_MAMBA3_TORCH,
                ReferenceSsmProfile.P20_THIN_TORCH,
                ReferenceSsmProfile.P20_TORCH,
            }
            for candidate in profiles_for_norm
        )
        else "identity"
    )
    default_schedule = _alternating_schedule(
        shape.total_layers, HybridAttentionLayerRole.REFERENCE_SSM
    )
    schedule = layer_schedule or default_schedule
    schedule_suffix = (
        f"schedule-{layer_schedule_signature(schedule)}"
        if schedule != default_schedule
        else ""
    )
    profile_schedule_suffix = (
        reference_profile_schedule_signature(profile_schedule)
        if profile_schedule is not None
        and any(candidate is not profile for candidate in profile_schedule)
        else ""
    )
    scaffold_suffix = (
        ""
        if scaffold_profile is Path1ScaffoldProfile.STANDARD
        else scaffold_profile.value
    )
    attention_suffix = (
        _variant_label(attention_profile.value, f"depthmem{depth_memory_layers}")
        if attention_profile is not AttentionProfile.STANDARD
        else ""
    )
    token_routing_suffix = (
        _variant_label(
            token_routing_profile.value,
            f"route{fraction_signature(token_route_fraction)}",
            f"layers{layer_index_signature(token_routing_layer_indices)}",
        )
        if token_routing_profile is not TokenRoutingProfile.NONE
        else ""
    )
    ffn_suffix = (
        _variant_label(
            feed_forward_profile.value,
            f"slots{eml_slot_count}",
            f"depth{eml_tree_depth}",
            (
                f"layers{layer_index_signature(feed_forward_layer_indices)}"
                if feed_forward_layer_indices is not None
                else ""
            ),
        )
        if feed_forward_profile is not FeedForwardProfile.STANDARD
        else ""
    )
    return Path1VariantSpec(
        kind=Path1VariantKind.REFERENCE_SSM_HYBRID,
        label=_variant_label(
            "reference-ssm-hybrid",
            profile.value,
            profile_schedule_suffix,
            attention_suffix,
            token_routing_suffix,
            scaffold_suffix,
            ffn_suffix,
            schedule_suffix,
        ),
        shape=shape,
        layer_schedule=schedule,
        reference_ssm_profile=profile,
        reference_ssm_profile_schedule=profile_schedule,
        reference_p20_ramp_init=reference_p20_ramp_init,
        feed_forward_profile=feed_forward_profile,
        feed_forward_layer_indices=feed_forward_layer_indices,
        eml_slot_count=eml_slot_count,
        eml_tree_depth=eml_tree_depth,
        eml_route_fraction=eml_route_fraction,
        final_norm_kind=final_norm,
        scaffold_profile=scaffold_profile,
        attention_profile=attention_profile,
        depth_memory_layers=depth_memory_layers,
        token_routing_profile=token_routing_profile,
        token_route_fraction=token_route_fraction,
        token_routing_layer_indices=token_routing_layer_indices,
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
    feed_forward_profile: FeedForwardProfile = FeedForwardProfile.STANDARD,
    feed_forward_layer_indices: tuple[int, ...] | None = None,
    eml_slot_count: int = 8,
    eml_tree_depth: int = 3,
    eml_route_fraction: float = 0.25,
    attention_profile: AttentionProfile = AttentionProfile.STANDARD,
    depth_memory_layers: int = 2,
    token_routing_profile: TokenRoutingProfile = TokenRoutingProfile.NONE,
    token_route_fraction: float = 0.25,
    token_routing_layer_indices: tuple[int, ...] | None = None,
) -> Path1VariantSpec:
    default_schedule = _alternating_schedule(
        shape.total_layers, HybridAttentionLayerRole.PRIMITIVE
    )
    schedule = layer_schedule or default_schedule
    schedule_suffix = (
        f"schedule-{layer_schedule_signature(schedule)}"
        if schedule != default_schedule
        else ""
    )
    ffn_suffix = (
        _variant_label(
            feed_forward_profile.value,
            f"slots{eml_slot_count}",
            f"depth{eml_tree_depth}",
            (
                f"layers{layer_index_signature(feed_forward_layer_indices)}"
                if feed_forward_layer_indices is not None
                else ""
            ),
        )
        if feed_forward_profile is not FeedForwardProfile.STANDARD
        else ""
    )
    attention_suffix = (
        _variant_label(attention_profile.value, f"depthmem{depth_memory_layers}")
        if attention_profile is not AttentionProfile.STANDARD
        else ""
    )
    token_routing_suffix = (
        _variant_label(
            token_routing_profile.value,
            f"route{fraction_signature(token_route_fraction)}",
            f"layers{layer_index_signature(token_routing_layer_indices)}",
        )
        if token_routing_profile is not TokenRoutingProfile.NONE
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
            attention_suffix,
            token_routing_suffix,
            ffn_suffix,
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
        feed_forward_profile=feed_forward_profile,
        feed_forward_layer_indices=feed_forward_layer_indices,
        eml_slot_count=eml_slot_count,
        eml_tree_depth=eml_tree_depth,
        eml_route_fraction=eml_route_fraction,
        attention_profile=attention_profile,
        depth_memory_layers=depth_memory_layers,
        token_routing_profile=token_routing_profile,
        token_route_fraction=token_route_fraction,
        token_routing_layer_indices=token_routing_layer_indices,
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
