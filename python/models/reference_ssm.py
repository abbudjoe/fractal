from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from python.models.common import (
    PositionWiseFeedForward,
    ReluSquaredFeedForward,
    SimpleRmsNorm,
    gated_sigmoid,
    one_minus,
)
from python.models.primitives import P20RotaryStateOutputRuntimeSequenceMixer
from python.runtime.recurrent import (
    BlockDiagonalLinear,
    PackedLinearProjection,
    build_state_transform_projection,
    rotate_state_pairs_with_trig,
    rotary_runtime_components,
)
from python.runtime.triton_primitives import (
    TritonPrimitiveBackend,
    build_triton_primitive_backend,
    ensure_triton_runtime_available,
)
from python.specs.runtime import PrimitiveStateTransformMode
from python.specs.path1 import ReferenceSsmProfile


@dataclass(frozen=True)
class ResolvedReferenceSsmConfig:
    d_model: int
    head_count: int
    d_state: int = 128
    expand: int = 2
    is_mimo: bool = False
    mimo_rank: int = 1
    chunk_size: int = 16
    is_outproj_norm: bool = False
    profile: ReferenceSsmProfile = ReferenceSsmProfile.MAMBA3_SISO_RUNTIME
    runtime_oriented: bool = False
    local_kernel_size: int = 4
    layer_index: int = 0
    p20_ramp_init: float = 0.01


def resolve_reference_ssm_config(
    d_model: int,
    head_count: int,
    profile: ReferenceSsmProfile,
    dtype_mode: str,
    layer_index: int = 0,
    p20_ramp_init: float = 0.01,
) -> ResolvedReferenceSsmConfig:
    if profile is ReferenceSsmProfile.MAMBA3_SISO_REFERENCE:
        chunk_size = 1
    else:
        chunk_size = 16 if dtype_mode == "bf16" else 8
    return ResolvedReferenceSsmConfig(
        d_model=d_model,
        head_count=head_count,
        is_mimo=profile.is_mimo,
        mimo_rank=profile.mimo_rank,
        chunk_size=chunk_size,
        profile=profile,
        runtime_oriented=profile.runtime_oriented,
        layer_index=layer_index,
        p20_ramp_init=p20_ramp_init,
    )


def _safe_logit(probability: float) -> float:
    clipped = min(max(float(probability), 1.0e-6), 1.0 - 1.0e-6)
    return torch.logit(torch.tensor(clipped, dtype=torch.float32)).item()


class OfficialMamba3SequenceMixer(nn.Module):
    def __init__(self, config: ResolvedReferenceSsmConfig) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba3
        except Exception as exc:  # pragma: no cover - exercised in runtime environments with mamba_ssm
            raise RuntimeError(
                "official PyTorch Mamba3 import failed. Install requirements from scripts/requirements-v3a-python-mamba3.txt"
            ) from exc

        self.mixer = Mamba3(
            d_model=config.d_model,
            d_state=config.d_state,
            headdim=config.d_model // config.head_count,
            is_mimo=config.is_mimo,
            mimo_rank=config.mimo_rank,
            chunk_size=config.chunk_size,
            is_outproj_norm=config.is_outproj_norm,
        )

    def forward(self, hidden: torch.Tensor, _attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        with record_function("path1.reference_ssm.native_mamba3"):
            return self.mixer(hidden)


class FlaGatedDeltaNetSequenceMixer(nn.Module):
    def __init__(self, config: ResolvedReferenceSsmConfig) -> None:
        super().__init__()
        try:
            from fla.layers import GatedDeltaNet
        except Exception as exc:  # pragma: no cover - exercised only in FLA runtime envs
            raise RuntimeError(
                "FLA GatedDeltaNet import failed. Install flash-linear-attention via "
                "scripts/requirements-v3a-python-gdn-fla.txt"
            ) from exc

        self.mixer = GatedDeltaNet(
            hidden_size=config.d_model,
            head_dim=config.d_model // config.head_count,
            num_heads=config.head_count,
            allow_neg_eigval=False,
            use_short_conv=True,
            expand_v=1,
            layer_idx=config.layer_index,
            mode="chunk",
        )

    def forward(self, hidden: torch.Tensor, _attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        with record_function("path1.reference_ssm.fla_gated_deltanet"):
            output = self.mixer(hidden)
        if isinstance(output, tuple):
            return output[0]
        return output

    def diagnostic_payload(self) -> dict[str, object]:
        return {"kind": "gated-deltanet-fla"}


class P20TinyControlConditioner(nn.Module):
    """Small vector-state conditioner that perturbs GDN controls without owning readout."""

    def __init__(
        self,
        d_model: int,
        head_count: int,
        *,
        width_factor: float = 0.125,
        ramp_init: float = 0.01,
    ) -> None:
        super().__init__()
        if width_factor <= 0.0 or width_factor > 1.0:
            raise ValueError(f"P20 control conditioner width_factor must be in (0, 1], got {width_factor}")
        bottleneck_width = max(2, int(round(d_model * width_factor)))
        if bottleneck_width % 2 != 0:
            bottleneck_width -= 1
        if bottleneck_width <= 0:
            raise ValueError(f"P20 control conditioner bottleneck width must be positive, got {bottleneck_width}")
        self.d_model = d_model
        self.head_count = head_count
        self.bottleneck_width = bottleneck_width
        self.width_factor = width_factor
        self.ramp_init = ramp_init
        self.input_projection = nn.Linear(d_model, bottleneck_width, bias=False)
        self.primitive = P20RotaryStateOutputRuntimeSequenceMixer(
            bottleneck_width,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
        )
        self.norm = SimpleRmsNorm(bottleneck_width)
        self.delta_projection = PackedLinearProjection(
            bottleneck_width,
            (
                d_model,
                d_model,
                d_model,
                head_count,
            ),
            bias=False,
        )
        self.qkv_ramp_logit = nn.Parameter(
            torch.full((3, d_model), _safe_logit(ramp_init), dtype=torch.float32)
        )
        self.beta_ramp_logit = nn.Parameter(
            torch.full((head_count,), _safe_logit(ramp_init), dtype=torch.float32)
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.1)
        with torch.no_grad():
            self.delta_projection.weight.zero_()

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with record_function("path1.reference_ssm.gdnp_control_tiny.p20_scan"):
            projected = self.input_projection(hidden)
            conditioner_state = self.primitive.scan(projected).emitted_outputs
            conditioner_state = self.norm(conditioner_state)
        with record_function("path1.reference_ssm.gdnp_control_tiny.delta_projection"):
            query_delta, key_delta, value_delta, beta_delta = self.delta_projection(conditioner_state)
        qkv_ramp = gated_sigmoid(self.qkv_ramp_logit).to(dtype=hidden.dtype)
        beta_ramp = gated_sigmoid(self.beta_ramp_logit).to(dtype=hidden.dtype)
        return (
            query_delta * qkv_ramp[0].view(1, 1, self.d_model),
            key_delta * qkv_ramp[1].view(1, 1, self.d_model),
            value_delta * qkv_ramp[2].view(1, 1, self.d_model),
            beta_delta * beta_ramp.view(1, 1, self.head_count),
        )

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        self.primitive.configure_runtime_policy(
            compile_mode=compile_mode,
            primitive_runtime_backend=primitive_runtime_backend,
        )

    def diagnostic_payload(self) -> dict[str, object]:
        return {
            "kind": "p20-tiny-control-conditioner",
            "d_model": self.d_model,
            "head_count": self.head_count,
            "bottleneck_width": self.bottleneck_width,
            "width_factor": self.width_factor,
            "ramp_init": self.ramp_init,
            "state_transform_mode": self.primitive.state_transform_mode.value,
            "delta_projection_zero_init": True,
        }


class FlaGdnpControlConditionedSequenceMixer(nn.Module):
    """FLA GDN core with a tiny P20/vector-state conditioner on q/k/v/beta controls."""

    def __init__(
        self,
        config: ResolvedReferenceSsmConfig,
        *,
        enable_p20_conditioner: bool = True,
    ) -> None:
        super().__init__()
        if config.d_model % config.head_count != 0:
            raise ValueError(
                "FLA GDN control-shell block requires d_model divisible by "
                f"head_count, got {config.d_model} and {config.head_count}"
            )
        self.d_model = config.d_model
        self.head_count = config.head_count
        self.head_dim = config.d_model // config.head_count
        self.head_v_dim = self.head_dim
        self.enable_p20_conditioner = enable_p20_conditioner
        self._native_control_shell = self._try_init_native_control_shell(config)
        if not self._native_control_shell:
            self._init_torch_fallback_control_shell(config)
        self.conditioner = None
        if enable_p20_conditioner:
            # Keep downstream layer initialization comparable to the no-conditioner shell.
            with torch.random.fork_rng(devices=[]):
                self.conditioner = P20TinyControlConditioner(
                    config.d_model,
                    config.head_count,
                    ramp_init=config.p20_ramp_init,
                )

    def _try_init_native_control_shell(self, config: ResolvedReferenceSsmConfig) -> bool:
        try:
            from fla.modules import FusedRMSNormGated, ShortConvolution
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except Exception:  # pragma: no cover - local CPU tests exercise the torch fallback
            self._chunk_gated_delta_rule = None
            return False

        self._chunk_gated_delta_rule = chunk_gated_delta_rule
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.a_proj = nn.Linear(config.d_model, config.head_count, bias=False)
        self.b_proj = nn.Linear(config.d_model, config.head_count, bias=False)
        a_init = torch.empty(config.head_count, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.A_log._no_weight_decay = True
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1.0e-4
        dt = torch.exp(
            torch.rand(config.head_count) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.q_conv1d = ShortConvolution(
            hidden_size=config.d_model,
            kernel_size=config.local_kernel_size,
            bias=False,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=config.d_model,
            kernel_size=config.local_kernel_size,
            bias=False,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=config.d_model,
            kernel_size=config.local_kernel_size,
            bias=False,
            activation="silu",
        )
        self.g_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=1.0e-5)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        return True

    def _init_torch_fallback_control_shell(self, config: ResolvedReferenceSsmConfig) -> None:
        self.control_projection = PackedLinearProjection(
            config.d_model,
            (
                config.d_model,
                config.d_model,
                config.d_model,
                config.head_count,
                config.head_count,
                config.d_model,
            ),
        )
        self.q_local = ReferenceCausalDepthwiseConv1d(config.d_model, kernel_size=config.local_kernel_size)
        self.k_local = ReferenceCausalDepthwiseConv1d(config.d_model, kernel_size=config.local_kernel_size)
        self.v_local = ReferenceCausalDepthwiseConv1d(config.d_model, kernel_size=config.local_kernel_size)
        self.output_norm = SimpleRmsNorm(self.head_dim)
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        self._init_parameters()

    def _init_parameters(self) -> None:
        if self._native_control_shell:
            return
        with torch.no_grad():
            if self.control_projection.bias is not None:
                split = self.control_projection.bias.split(self.control_projection.split_sizes, dim=0)
                decay_bias = split[3]
                beta_bias = split[4]
                output_gate_bias = split[5]
                decay_bias.fill_(1.5)
                beta_bias.fill_(-2.0)
                output_gate_bias.fill_(-1.5)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.1)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(*tensor.shape[:-1], self.head_count, self.head_dim)

    def _reshape_value_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(*tensor.shape[:-1], self.head_count, self.head_v_dim)

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        if self.conditioner is not None:
            self.conditioner.configure_runtime_policy(
                compile_mode=compile_mode,
                primitive_runtime_backend=primitive_runtime_backend,
            )

    def _control_deltas(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor | float, torch.Tensor | float, torch.Tensor | float, torch.Tensor | float]:
        if self.conditioner is None:
            return 0.0, 0.0, 0.0, 0.0
        with record_function("path1.reference_ssm.gdnp_control_tiny.conditioner"):
            return self.conditioner(hidden)

    def _scan_torch(
        self,
        *,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _head_count, _head_dim = queries.shape
        matrix_state = torch.zeros(
            batch_size,
            self.head_count,
            self.head_dim,
            self.head_dim,
            device=queries.device,
            dtype=queries.dtype,
        )
        matrix_reads = torch.empty_like(queries)
        alpha = torch.exp(decay)
        normed_queries = F.normalize(queries, p=2.0, dim=-1, eps=1.0e-6)
        normed_keys = F.normalize(keys, p=2.0, dim=-1, eps=1.0e-6)
        with record_function("path1.reference_ssm.gdnp_control_tiny.torch_delta_scan"):
            for position in range(seq_len):
                query = normed_queries[:, position, :, :]
                key = normed_keys[:, position, :, :]
                value = values[:, position, :, :]
                alpha_step = alpha[:, position, :].view(-1, self.head_count, 1, 1)
                beta_step = beta[:, position, :].view(-1, self.head_count, 1, 1)
                old_value = torch.einsum("bhvk,bhk->bhv", matrix_state, key)
                erase = torch.einsum("bhv,bhk->bhvk", old_value, key)
                write = torch.einsum("bhv,bhk->bhvk", value, key)
                matrix_state = alpha_step * (matrix_state - beta_step * erase) + beta_step * write
                matrix_reads[:, position, :, :] = torch.einsum("bhvk,bhk->bhv", matrix_state, query)
        return matrix_reads

    def _forward_native_shell(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        query_delta, key_delta, value_delta, beta_delta = self._control_deltas(hidden)
        with record_function("path1.reference_ssm.gdnp_control_tiny.native_short_conv"):
            query_inputs, _query_conv_state = self.q_conv1d(self.q_proj(hidden))
            key_inputs, _key_conv_state = self.k_conv1d(self.k_proj(hidden))
            value_inputs, _value_conv_state = self.v_conv1d(self.v_proj(hidden))
            query_inputs = query_inputs + query_delta
            key_inputs = key_inputs + key_delta
            value_inputs = value_inputs + value_delta
        queries = self._reshape_heads(query_inputs)
        keys = self._reshape_heads(key_inputs)
        values = self._reshape_value_heads(value_inputs)
        beta = torch.sigmoid(self.b_proj(hidden) + beta_delta)
        decay = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden).float() + self.dt_bias)
        assert self._chunk_gated_delta_rule is not None
        with record_function("path1.reference_ssm.gdnp_control_tiny.native_chunk_gated_delta_rule"):
            matrix_reads, _final_state = self._chunk_gated_delta_rule(
                q=queries,
                k=keys,
                v=values,
                g=decay,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
        with record_function("path1.reference_ssm.gdnp_control_tiny.native_sequence_readout"):
            output_gate = self._reshape_value_heads(self.g_proj(hidden))
            normed = self.o_norm(matrix_reads, output_gate)
            projected = self.o_proj(normed.reshape(batch_size, seq_len, self.d_model))
        return projected

    def _forward_torch_fallback_shell(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        query_delta, key_delta, value_delta, beta_delta = self._control_deltas(hidden)
        with record_function("path1.reference_ssm.gdnp_control_tiny.control_projection"):
            query_inputs, key_inputs, value_inputs, decay_inputs, beta_inputs, output_gate_inputs = (
                self.control_projection(hidden)
            )
        with record_function("path1.reference_ssm.gdnp_control_tiny.short_conv"):
            query_inputs = self.q_local(query_inputs) + query_delta
            key_inputs = self.k_local(key_inputs) + key_delta
            value_inputs = self.v_local(value_inputs) + value_delta

        queries = self._reshape_heads(query_inputs)
        keys = self._reshape_heads(key_inputs)
        values = self._reshape_heads(value_inputs)
        decay = F.logsigmoid(decay_inputs.float()).to(dtype=hidden.dtype)
        beta = torch.sigmoid(beta_inputs + beta_delta)
        if self._chunk_gated_delta_rule is not None:
            with record_function("path1.reference_ssm.gdnp_control_tiny.chunk_gated_delta_rule"):
                matrix_reads, _final_state = self._chunk_gated_delta_rule(
                    q=queries,
                    k=keys,
                    v=values,
                    g=decay,
                    beta=beta,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                )
        else:
            matrix_reads = self._scan_torch(
                queries=queries,
                keys=keys,
                values=values,
                decay=decay,
                beta=beta,
            )
        with record_function("path1.reference_ssm.gdnp_control_tiny.sequence_readout"):
            normed_matrix_reads = self.output_norm(matrix_reads).reshape(batch_size, seq_len, self.d_model)
            projected = self.output_projection(normed_matrix_reads)
        return torch.sigmoid(output_gate_inputs) * projected

    def forward(self, hidden: torch.Tensor, _attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self._native_control_shell:
            return self._forward_native_shell(hidden)
        return self._forward_torch_fallback_shell(hidden)

    def diagnostic_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": (
                "fla-gdnp-control-conditioned"
                if self.conditioner is not None
                else "fla-gdn-control-shell"
            ),
            "gdn_kernel": (
                "fla.chunk_gated_delta_rule"
                if self._chunk_gated_delta_rule is not None
                else "torch.loop_fallback"
            ),
            "shell_contract": (
                "fla-native"
                if self._native_control_shell
                else "torch-fallback"
            ),
            "conditioned_controls": (
                ("q", "k", "v", "beta")
                if self.conditioner is not None
                else ()
            ),
            "readout": "gdn-only",
            "matrix_heads": self.head_count,
            "matrix_head_dim": self.head_dim,
        }
        if self.conditioner is not None:
            payload["conditioner"] = self.conditioner.diagnostic_payload()
        return payload


class FlaGatedDeltaNetControlShellSequenceMixer(FlaGdnpControlConditionedSequenceMixer):
    """Matched custom FLA GDN shell with no P20 conditioner for wrapper diagnosis."""

    def __init__(self, config: ResolvedReferenceSsmConfig) -> None:
        super().__init__(config, enable_p20_conditioner=False)


class FlaGdnpCompatibleSequenceMixer(nn.Module):
    """FLA-kernel GDN core conditioned by a P20 vector-state side channel."""

    def __init__(self, config: ResolvedReferenceSsmConfig) -> None:
        super().__init__()
        if config.d_model % config.head_count != 0:
            raise ValueError(
                f"FLA GDN/P20 compatible block requires d_model divisible by head_count, got {config.d_model} and {config.head_count}"
            )
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except Exception as exc:  # pragma: no cover - exercised only in FLA runtime envs
            raise RuntimeError(
                "FLA gated-delta-rule import failed. Install flash-linear-attention via "
                "scripts/requirements-v3a-python-gdn-fla.txt in the primitive-triton environment"
            ) from exc

        self._chunk_gated_delta_rule = chunk_gated_delta_rule
        self.d_model = config.d_model
        self.head_count = config.head_count
        self.head_dim = config.d_model // config.head_count
        self.law_kind = config.profile.fla_gdnp_compatible_law
        self.uses_multi_read = self.law_kind == "multi-read"
        self.p20_ramp_init = config.p20_ramp_init
        self.p20 = P20SequencePrimitiveBranch(config.d_model)
        self.control_projection = PackedLinearProjection(
            config.d_model,
            (
                config.d_model,
                config.d_model,
                config.d_model,
                config.head_count,
                config.head_count,
                config.d_model,
            ),
        )
        self.p20_condition_projection = PackedLinearProjection(
            config.d_model,
            (
                config.d_model,
                config.d_model,
                config.d_model,
            ),
            bias=False,
        )
        if self.uses_multi_read:
            self.aux_query_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_local = ReferenceCausalDepthwiseConv1d(config.d_model, kernel_size=config.local_kernel_size)
        self.k_local = ReferenceCausalDepthwiseConv1d(config.d_model, kernel_size=config.local_kernel_size)
        self.v_local = ReferenceCausalDepthwiseConv1d(config.d_model, kernel_size=config.local_kernel_size)
        self.matrix_read_norm = SimpleRmsNorm(self.head_dim)
        self.vector_read_norm = SimpleRmsNorm(config.d_model)
        self.p20_to_gdn_ramp_logit = nn.Parameter(
            torch.full((config.d_model,), _safe_logit(config.p20_ramp_init))
        )
        readout_width = config.d_model * (3 if self.uses_multi_read else 2)
        self.output_projection = nn.Linear(readout_width, config.d_model)
        self._init_parameters()

    def _init_parameters(self) -> None:
        with torch.no_grad():
            if self.control_projection.bias is not None:
                split = self.control_projection.bias.split(self.control_projection.split_sizes, dim=0)
                decay_bias = split[3]
                beta_bias = split[4]
                output_gate_bias = split[5]
                decay_bias.fill_(1.5)
                beta_bias.fill_(-2.0)
                output_gate_bias.fill_(-1.5)
            self.p20_condition_projection.weight.mul_(0.05)
            if self.uses_multi_read:
                self.aux_query_projection.weight.mul_(0.05)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.1)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(*tensor.shape[:-1], self.head_count, self.head_dim)

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        self.p20.configure_runtime_policy(
            compile_mode=compile_mode,
            primitive_runtime_backend=primitive_runtime_backend,
        )

    def forward(self, hidden: torch.Tensor, _attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        with record_function("path1.reference_ssm.fla_gdnp.p20_condition"):
            p20_read = self.p20(hidden)
            normed_p20 = self.vector_read_norm(p20_read)
            p20_query, p20_key, p20_value = self.p20_condition_projection(normed_p20)
            p20_ramp = gated_sigmoid(self.p20_to_gdn_ramp_logit).view(1, 1, self.d_model)

        with record_function("path1.reference_ssm.fla_gdnp.control_projection"):
            query_inputs, key_inputs, value_inputs, decay_inputs, beta_inputs, output_gate_inputs = (
                self.control_projection(hidden)
            )

        with record_function("path1.reference_ssm.fla_gdnp.short_conv"):
            query_inputs = self.q_local(query_inputs)
            key_inputs = self.k_local(key_inputs)
            value_inputs = self.v_local(value_inputs)

        queries = self._reshape_heads(query_inputs + p20_ramp * p20_query)
        keys = self._reshape_heads(key_inputs + p20_ramp * p20_key)
        values = self._reshape_heads(value_inputs + p20_ramp * p20_value)
        decay = F.logsigmoid(decay_inputs.float()).to(dtype=hidden.dtype)
        beta = torch.sigmoid(beta_inputs)
        with record_function("path1.reference_ssm.fla_gdnp.chunk_gated_delta_rule"):
            matrix_reads, _final_state = self._chunk_gated_delta_rule(
                q=queries,
                k=keys,
                v=values,
                g=decay,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
        aux_matrix_reads = None
        if self.uses_multi_read:
            with record_function("path1.reference_ssm.fla_gdnp.aux_chunk_gated_delta_rule"):
                aux_queries = self._reshape_heads(query_inputs + p20_ramp * self.aux_query_projection(normed_p20))
                aux_matrix_reads, _aux_final_state = self._chunk_gated_delta_rule(
                    q=aux_queries,
                    k=keys,
                    v=values,
                    g=decay,
                    beta=beta,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                )
        with record_function("path1.reference_ssm.fla_gdnp.sequence_readout"):
            normed_matrix_reads = self.matrix_read_norm(matrix_reads).reshape(batch_size, seq_len, self.d_model)
            read_parts = [normed_matrix_reads]
            if aux_matrix_reads is not None:
                read_parts.append(
                    self.matrix_read_norm(aux_matrix_reads).reshape(batch_size, seq_len, self.d_model)
                )
            read_parts.append(normed_p20)
            fused_reads = torch.cat(read_parts, dim=-1)
            projected = self.output_projection(fused_reads)
        return torch.sigmoid(output_gate_inputs) * projected

    def diagnostic_payload(self) -> dict[str, object]:
        return {
            "kind": "fla-gdnp-compatible",
            "law": self.law_kind,
            "gdn_kernel": "fla.chunk_gated_delta_rule",
            "p20_conditioning": True,
            "p20_ramp_init": self.p20_ramp_init,
            "uses_multi_read": self.uses_multi_read,
            "p20_branch": self.p20.diagnostic_payload(),
            "matrix_heads": self.head_count,
            "matrix_head_dim": self.head_dim,
        }


class ReferenceCausalDepthwiseConv1d(nn.Module):
    def __init__(self, width: int, *, kernel_size: int = 4) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"causal depthwise conv kernel_size must be positive, got {kernel_size}")
        self.width = width
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(width, 1, kernel_size))
        with torch.no_grad():
            self.weight[:, 0, -1] = 1.0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        conv_inputs = F.pad(hidden.transpose(1, 2), (self.kernel_size - 1, 0))
        mixed = F.conv1d(conv_inputs, self.weight, groups=self.width).transpose(1, 2)
        return F.silu(mixed)


class TorchGatedDeltaNetSequenceMixer(nn.Module):
    """Fractal-native Gated DeltaNet block extracted from the PR #1564 contract."""

    def __init__(self, config: ResolvedReferenceSsmConfig) -> None:
        super().__init__()
        if config.d_model % config.head_count != 0:
            raise ValueError(
                f"Gated DeltaNet requires d_model divisible by head_count, got {config.d_model} and {config.head_count}"
            )
        self.d_model = config.d_model
        self.head_count = config.head_count
        self.head_dim = config.d_model // config.head_count
        self.qkv_projection = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.control_projection = nn.Linear(
            config.d_model,
            config.head_count * 2 + config.d_model,
            bias=True,
        )
        self.q_local = ReferenceCausalDepthwiseConv1d(config.d_model, kernel_size=config.local_kernel_size)
        self.k_local = ReferenceCausalDepthwiseConv1d(config.d_model, kernel_size=config.local_kernel_size)
        self.v_local = ReferenceCausalDepthwiseConv1d(config.d_model, kernel_size=config.local_kernel_size)
        self.output_norm = SimpleRmsNorm(self.head_dim)
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        self._init_parameters()

    def _init_parameters(self) -> None:
        with torch.no_grad():
            alpha_bias, beta_bias, output_gate_bias = self.control_projection.bias.split(
                (self.head_count, self.head_count, self.d_model),
                dim=0,
            )
            alpha_bias.fill_(1.5)
            beta_bias.fill_(-2.0)
            output_gate_bias.fill_(-1.5)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.1)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(*tensor.shape[:-1], self.head_count, self.head_dim)

    def forward(self, hidden: torch.Tensor, _attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        with record_function("path1.reference_ssm.gdn.qkv_projection"):
            q_inputs, k_inputs, v_inputs = self.qkv_projection(hidden).chunk(3, dim=-1)
        with record_function("path1.reference_ssm.gdn.control_projection"):
            alpha_inputs, beta_inputs, output_gate_inputs = self.control_projection(hidden).split(
                (self.head_count, self.head_count, self.d_model),
                dim=-1,
            )
        with record_function("path1.reference_ssm.gdn.short_conv"):
            queries = F.normalize(
                self._reshape_heads(self.q_local(q_inputs)),
                p=2.0,
                dim=-1,
                eps=1.0e-6,
            )
            keys = F.normalize(
                self._reshape_heads(self.k_local(k_inputs)),
                p=2.0,
                dim=-1,
                eps=1.0e-6,
            )
            values = self._reshape_heads(self.v_local(v_inputs))

        alpha_gates = torch.sigmoid(alpha_inputs).mul(0.98).add(0.01)
        beta_gates = torch.sigmoid(beta_inputs)
        output_gates = torch.sigmoid(output_gate_inputs)
        state = torch.zeros(
            batch_size,
            self.head_count,
            self.head_dim,
            self.head_dim,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        outputs = torch.empty(batch_size, seq_len, self.d_model, device=hidden.device, dtype=hidden.dtype)
        with record_function("path1.reference_ssm.gdn.delta_scan"):
            for position in range(seq_len):
                query = queries[:, position, :, :]
                key = keys[:, position, :, :]
                value = values[:, position, :, :]
                alpha = alpha_gates[:, position, :].view(-1, self.head_count, 1, 1)
                beta = beta_gates[:, position, :].view(-1, self.head_count, 1, 1)
                old_value = torch.einsum("bhvk,bhk->bhv", state, key)
                erase = torch.einsum("bhv,bhk->bhvk", old_value, key)
                write = torch.einsum("bhv,bhk->bhvk", value, key)
                state = alpha * (state - beta * erase) + beta * write
                read = torch.einsum("bhvk,bhk->bhv", state, query)
                read = self.output_norm(read).reshape(batch_size, self.d_model)
                projected = self.output_projection(read)
                outputs[:, position, :] = output_gates[:, position, :] * projected
        return outputs

    def diagnostic_payload(self) -> dict[str, object]:
        return {
            "kind": "gated-deltanet-torch",
            "matrix_heads": self.head_count,
            "matrix_head_dim": self.head_dim,
        }


class P20SequencePrimitiveBranch(nn.Module):
    def __init__(self, d_model: int, *, width_factor: float = 1.0) -> None:
        super().__init__()
        if width_factor <= 0.0 or width_factor > 1.0:
            raise ValueError(f"P20 branch width_factor must be in (0, 1], got {width_factor}")
        bottleneck_width = max(2, int(round(d_model * width_factor)))
        if bottleneck_width % 2 != 0:
            bottleneck_width -= 1
        if bottleneck_width <= 0:
            raise ValueError(f"P20 branch bottleneck width must be positive, got {bottleneck_width}")
        self.d_model = d_model
        self.bottleneck_width = bottleneck_width
        self.width_factor = width_factor
        self.input_projection = (
            nn.Identity()
            if bottleneck_width == d_model
            else nn.Linear(d_model, bottleneck_width, bias=False)
        )
        self.primitive = P20RotaryStateOutputRuntimeSequenceMixer(
            bottleneck_width,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
        )
        self.output_projection = (
            nn.Identity()
            if bottleneck_width == d_model
            else nn.Linear(bottleneck_width, d_model, bias=False)
        )

    def forward(self, hidden: torch.Tensor, _attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        projected = self.input_projection(hidden)
        mixed = self.primitive.scan(projected).emitted_outputs
        return self.output_projection(mixed)

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        self.primitive.configure_runtime_policy(
            compile_mode=compile_mode,
            primitive_runtime_backend=primitive_runtime_backend,
        )

    def diagnostic_payload(self) -> dict[str, object]:
        return {
            "branch_kind": "p20",
            "d_model": self.d_model,
            "bottleneck_width": self.bottleneck_width,
            "width_factor": self.width_factor,
            "state_transform_mode": self.primitive.state_transform_mode.value,
        }


class ParallelCompositeSequenceMixer(nn.Module):
    """Runs distinct recurrent memory mechanisms in parallel and learns a branch mix."""

    def __init__(self, config: ResolvedReferenceSsmConfig, branches: tuple[str, ...]) -> None:
        super().__init__()
        if not branches:
            raise ValueError("composite sequence mixer requires at least one branch")
        self.branch_names = branches
        branch_modules: dict[str, nn.Module] = {}
        for branch_name in branches:
            if branch_name == "gdn":
                branch_modules[branch_name] = TorchGatedDeltaNetSequenceMixer(config)
            elif branch_name == "p20":
                branch_modules[branch_name] = P20SequencePrimitiveBranch(config.d_model)
            elif branch_name == "p20_thin":
                branch_modules[branch_name] = P20SequencePrimitiveBranch(config.d_model, width_factor=0.5)
            elif branch_name == "mamba3":
                branch_modules[branch_name] = OfficialMamba3SequenceMixer(config)
            else:
                raise ValueError(f"unsupported composite reference-SSM branch: {branch_name}")
        self.branches = nn.ModuleDict(branch_modules)
        self.branch_norms = nn.ModuleDict(
            {branch_name: SimpleRmsNorm(config.d_model) for branch_name in branches}
        )
        self.branch_logits = nn.Parameter(torch.zeros(len(branches), config.d_model))

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        weights = torch.softmax(self.branch_logits.to(dtype=hidden.dtype), dim=0)
        output = torch.zeros_like(hidden)
        for branch_index, branch_name in enumerate(self.branch_names):
            with record_function(f"path1.reference_ssm.composite.{branch_name}"):
                branch_output = self.branches[branch_name](hidden, attn_mask)
            normalized = self.branch_norms[branch_name](branch_output)
            output = output + weights[branch_index].view(1, 1, -1) * normalized
        return output

    def diagnostic_payload(self) -> dict[str, object]:
        weights = torch.softmax(self.branch_logits.detach().float(), dim=0)
        branch_payloads: list[dict[str, object]] = []
        for branch_index, branch_name in enumerate(self.branch_names):
            branch_weights = weights[branch_index]
            module = self.branches[branch_name]
            module_diagnostic = getattr(module, "diagnostic_payload", None)
            branch_payloads.append(
                {
                    "branch": branch_name,
                    "mean_weight": float(branch_weights.mean().item()),
                    "min_weight": float(branch_weights.min().item()),
                    "max_weight": float(branch_weights.max().item()),
                    "std_weight": float(branch_weights.std(unbiased=False).item()),
                    "module": module_diagnostic() if callable(module_diagnostic) else {},
                }
            )
        entropy = -(weights * torch.log(weights.clamp_min(1.0e-12))).sum(dim=0)
        return {
            "kind": "parallel-composite",
            "branch_count": len(self.branch_names),
            "channel_count": int(self.branch_logits.shape[1]),
            "branches": branch_payloads,
            "mean_channel_entropy": float(entropy.mean().item()),
        }

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        for branch in self.branches.values():
            configure = getattr(branch, "configure_runtime_policy", None)
            if callable(configure):
                configure(
                    compile_mode=compile_mode,
                    primitive_runtime_backend=primitive_runtime_backend,
                )


class GdnpFusedSequenceMixer(nn.Module):
    """One-scan GDN/P20 fusion with typed vector and matrix recurrent state."""

    def __init__(self, config: ResolvedReferenceSsmConfig) -> None:
        super().__init__()
        if config.d_model % 2 != 0:
            raise ValueError(f"GDN/P20 fusion requires even d_model, got {config.d_model}")
        if config.d_model % config.head_count != 0:
            raise ValueError(
                f"GDN/P20 fusion requires d_model divisible by head_count, got {config.d_model} and {config.head_count}"
            )
        self.d_model = config.d_model
        self.head_count = config.head_count
        self.head_dim = config.d_model // config.head_count
        self.law_kind = config.profile.gdnp_fused_law
        self.modulates_beta = self.law_kind in {"all", "beta"}
        self.modulates_qkv = self.law_kind in {"all", "qkv"}
        self.uses_multi_read = self.law_kind in {"all", "multi-read"}
        self.uses_residual_readout = self.law_kind in {"all", "residual-readout"}
        self.state_transform_mode = PrimitiveStateTransformMode.BLOCK_DIAGONAL_2
        self._primitive_runtime_backend = "torch"
        self._triton_backend: TritonPrimitiveBackend | None = None
        self.control_projection = PackedLinearProjection(
            config.d_model,
            (
                config.d_model,  # P20 update gate
                config.d_model // 2,  # P20 rotary angle
                config.d_model,  # P20 candidate
                config.d_model,  # GDN query
                config.d_model,  # GDN key
                config.d_model,  # GDN value
                config.head_count,  # GDN alpha
                config.head_count,  # GDN beta
                config.d_model,  # output gate
            ),
        )
        self.state_transform_projection = build_state_transform_projection(
            config.d_model,
            self.state_transform_mode,
        )
        self.q_local = ReferenceCausalDepthwiseConv1d(config.d_model, kernel_size=config.local_kernel_size)
        self.k_local = ReferenceCausalDepthwiseConv1d(config.d_model, kernel_size=config.local_kernel_size)
        self.v_local = ReferenceCausalDepthwiseConv1d(config.d_model, kernel_size=config.local_kernel_size)
        self.matrix_read_norm = SimpleRmsNorm(self.head_dim)
        self.vector_read_norm = SimpleRmsNorm(config.d_model)
        if self.modulates_beta:
            self.beta_state_weight = nn.Parameter(torch.empty(self.head_count, self.head_dim))
        if self.modulates_qkv:
            self.query_state_scale = nn.Parameter(torch.empty(self.head_count, self.head_dim))
            self.key_state_scale = nn.Parameter(torch.empty(self.head_count, self.head_dim))
            self.value_state_scale = nn.Parameter(torch.empty(self.head_count, self.head_dim))
        if self.uses_multi_read:
            self.aux_query_state_scale = nn.Parameter(torch.empty(self.head_count, self.head_dim))
        if self.uses_residual_readout:
            self.matrix_output_projection = nn.Linear(config.d_model, config.d_model)
            if self.uses_multi_read:
                self.aux_matrix_output_projection = nn.Linear(config.d_model, config.d_model)
            self.vector_output_projection = nn.Linear(config.d_model, config.d_model)
            self.vector_readout_ramp_logit = nn.Parameter(torch.full((config.d_model,), -2.1972246))
        else:
            readout_width = config.d_model * (3 if self.uses_multi_read else 2)
            self.output_projection = nn.Linear(readout_width, config.d_model)
        self._init_parameters()

    def _init_parameters(self) -> None:
        with torch.no_grad():
            if self.control_projection.bias is None:
                return
            split = self.control_projection.bias.split(self.control_projection.split_sizes, dim=0)
            alpha_bias = split[6]
            beta_bias = split[7]
            output_gate_bias = split[8]
            alpha_bias.fill_(1.5)
            beta_bias.fill_(-2.0)
            output_gate_bias.fill_(-1.5)
        if self.modulates_beta:
            nn.init.zeros_(self.beta_state_weight)
        if self.modulates_qkv:
            nn.init.constant_(self.query_state_scale, 0.05)
            nn.init.constant_(self.key_state_scale, 0.05)
            nn.init.ones_(self.value_state_scale)
        if self.uses_multi_read:
            nn.init.constant_(self.aux_query_state_scale, 0.1)

        output_modules = []
        if self.uses_residual_readout:
            output_modules.append(self.matrix_output_projection)
            if self.uses_multi_read:
                output_modules.append(self.aux_matrix_output_projection)
            output_modules.append(self.vector_output_projection)
        else:
            output_modules.append(self.output_projection)
        for output_module in output_modules:
            nn.init.xavier_uniform_(output_module.weight, gain=0.1)
            if output_module.bias is not None:
                nn.init.zeros_(output_module.bias)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(*tensor.shape[:-1], self.head_count, self.head_dim)

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        if primitive_runtime_backend == "triton":
            ensure_triton_runtime_available()
            self._triton_backend = build_triton_primitive_backend()
        else:
            self._triton_backend = None
        self._primitive_runtime_backend = primitive_runtime_backend or "torch"
        del compile_mode

    def _scan_vector_states_torch(
        self,
        *,
        update_gates: torch.Tensor,
        retain_gates: torch.Tensor,
        angle_cos: torch.Tensor,
        angle_sin: torch.Tensor,
        candidates: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _width = update_gates.shape
        vector_state = initial_state
        vector_states = torch.empty_like(update_gates)
        for position in range(seq_len):
            projected_vector = self.state_transform_projection(vector_state)
            transformed_vector = rotate_state_pairs_with_trig(
                projected_vector,
                cos=angle_cos[:, position, :],
                sin=angle_sin[:, position, :],
            )
            vector_state = (
                update_gates[:, position, :] * transformed_vector
                + retain_gates[:, position, :] * candidates[:, position, :]
            )
            vector_states[:, position, :] = vector_state
        return vector_states

    def _scan_vector_states_triton(
        self,
        *,
        update_gates: torch.Tensor,
        retain_gates: torch.Tensor,
        angle_cos: torch.Tensor,
        angle_sin: torch.Tensor,
        candidates: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        if self._triton_backend is None:
            raise RuntimeError("GDN/P20 fused Triton vector scan requested without an initialized Triton backend")
        if not isinstance(self.state_transform_projection, BlockDiagonalLinear):
            raise RuntimeError("GDN/P20 fused Triton vector scan requires a block-diagonal state transform")
        with record_function("path1.reference_ssm.gdnp_fused.triton_vector_scan"):
            vector_states, _final_state = self._triton_backend.scan_rotary_state_block_diagonal_sequence(
                update_gate=update_gates,
                retain_gate=retain_gates,
                angle_cos=angle_cos,
                angle_sin=angle_sin,
                candidate=candidates,
                initial_state=initial_state,
                transform_weight=self.state_transform_projection.weight,
                transform_bias=self.state_transform_projection.bias,
            )
        return vector_states

    def _scan_vector_states(
        self,
        *,
        update_gates: torch.Tensor,
        retain_gates: torch.Tensor,
        angle_cos: torch.Tensor,
        angle_sin: torch.Tensor,
        candidates: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        initial_state = torch.zeros(
            batch_size,
            self.d_model,
            device=device,
            dtype=dtype,
        )
        if self._primitive_runtime_backend == "triton":
            return self._scan_vector_states_triton(
                update_gates=update_gates,
                retain_gates=retain_gates,
                angle_cos=angle_cos,
                angle_sin=angle_sin,
                candidates=candidates,
                initial_state=initial_state,
            )
        return self._scan_vector_states_torch(
            update_gates=update_gates,
            retain_gates=retain_gates,
            angle_cos=angle_cos,
            angle_sin=angle_sin,
            candidates=candidates,
            initial_state=initial_state,
        )

    def _can_use_triton_matrix_scan(self) -> bool:
        return (
            self._primitive_runtime_backend == "triton"
            and self.law_kind == "multi-read"
            and self._triton_backend is not None
        )

    def forward(self, hidden: torch.Tensor, _attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        with record_function("path1.reference_ssm.gdnp_fused.control_projection"):
            (
                update_gate_inputs,
                angle_inputs,
                candidate_inputs,
                query_inputs,
                key_inputs,
                value_inputs,
                alpha_inputs,
                beta_inputs,
                output_gate_inputs,
            ) = self.control_projection(hidden)

        with record_function("path1.reference_ssm.gdnp_fused.short_conv"):
            queries = F.normalize(
                self._reshape_heads(self.q_local(query_inputs)),
                p=2.0,
                dim=-1,
                eps=1.0e-6,
            )
            keys = F.normalize(
                self._reshape_heads(self.k_local(key_inputs)),
                p=2.0,
                dim=-1,
                eps=1.0e-6,
            )
            value_bases = self._reshape_heads(self.v_local(value_inputs))

        update_gates = gated_sigmoid(update_gate_inputs)
        retain_gates = one_minus(update_gates)
        angle_cos, angle_sin = rotary_runtime_components(angle_inputs)
        candidates = torch.tanh(candidate_inputs)
        alpha_gates = torch.sigmoid(alpha_inputs).mul(0.98).add(0.01)
        beta_gates = None if self.modulates_beta else torch.sigmoid(beta_inputs)
        output_gates = torch.sigmoid(output_gate_inputs)

        with record_function("path1.reference_ssm.gdnp_fused.vector_scan"):
            vector_states = self._scan_vector_states(
                update_gates=update_gates,
                retain_gates=retain_gates,
                angle_cos=angle_cos,
                angle_sin=angle_sin,
                candidates=candidates,
                batch_size=batch_size,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        matrix_state = torch.zeros(
            batch_size,
            self.head_count,
            self.head_dim,
            self.head_dim,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        matrix_reads = torch.empty(
            batch_size,
            seq_len,
            self.head_count,
            self.head_dim,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        aux_matrix_reads = (
            torch.empty(
                batch_size,
                seq_len,
                self.head_count,
                self.head_dim,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            if self.uses_multi_read
                else None
        )
        if self._can_use_triton_matrix_scan():
            assert self._triton_backend is not None
            assert beta_gates is not None
            with record_function("path1.reference_ssm.gdnp_fused.triton_matrix_scan"):
                matrix_reads, aux_matrix_reads = self._triton_backend.scan_gdnp_matrix_multi_read(
                    queries=queries,
                    keys=keys,
                    value_bases=value_bases,
                    vector_states=self._reshape_heads(vector_states),
                    alpha_gates=alpha_gates,
                    beta_gates=beta_gates,
                    aux_query_state_scale=self.aux_query_state_scale,
                )
        else:
            with record_function("path1.reference_ssm.gdnp_fused.matrix_scan"):
                for position in range(seq_len):
                    vector_state = vector_states[:, position, :]

                    vector_heads = self._reshape_heads(vector_state)
                    query_base = queries[:, position, :, :]
                    key_base = keys[:, position, :, :]
                    value_base = value_bases[:, position, :, :]
                    if self.modulates_qkv:
                        query = F.normalize(
                            query_base + vector_heads * self.query_state_scale.view(1, self.head_count, self.head_dim),
                            p=2.0,
                            dim=-1,
                            eps=1.0e-6,
                        )
                        key = F.normalize(
                            key_base + vector_heads * self.key_state_scale.view(1, self.head_count, self.head_dim),
                            p=2.0,
                            dim=-1,
                            eps=1.0e-6,
                        )
                        value = value_base + vector_heads * self.value_state_scale.view(1, self.head_count, self.head_dim)
                    else:
                        query = query_base
                        key = key_base
                        value = value_base + vector_heads
                    alpha = alpha_gates[:, position, :].view(-1, self.head_count, 1, 1)
                    if self.modulates_beta:
                        beta_inputs_for_step = beta_inputs[:, position, :] + (
                            vector_heads * self.beta_state_weight.view(1, self.head_count, self.head_dim)
                        ).mean(dim=-1)
                        beta_values = torch.sigmoid(beta_inputs_for_step)
                    else:
                        assert beta_gates is not None
                        beta_values = beta_gates[:, position, :]
                    beta = beta_values.view(-1, self.head_count, 1, 1)
                    old_value = torch.einsum("bhvk,bhk->bhv", matrix_state, key)
                    erase = torch.einsum("bhv,bhk->bhvk", old_value, key)
                    write = torch.einsum("bhv,bhk->bhvk", value, key)
                    matrix_state = alpha * (matrix_state - beta * erase) + beta * write
                    matrix_read = torch.einsum("bhvk,bhk->bhv", matrix_state, query)
                    matrix_reads[:, position, :, :] = matrix_read
                    if aux_matrix_reads is not None:
                        aux_query = F.normalize(
                            query_base + vector_heads * self.aux_query_state_scale.view(1, self.head_count, self.head_dim),
                            p=2.0,
                            dim=-1,
                            eps=1.0e-6,
                        )
                        aux_matrix_reads[:, position, :, :] = torch.einsum("bhvk,bhk->bhv", matrix_state, aux_query)
        with record_function("path1.reference_ssm.gdnp_fused.sequence_readout"):
            normed_matrix_reads = self.matrix_read_norm(matrix_reads).reshape(batch_size, seq_len, self.d_model)
            normed_vector_states = self.vector_read_norm(vector_states)
            if aux_matrix_reads is not None:
                normed_aux_matrix_reads = self.matrix_read_norm(aux_matrix_reads).reshape(
                    batch_size,
                    seq_len,
                    self.d_model,
                )
            else:
                normed_aux_matrix_reads = None

            if self.uses_residual_readout:
                projected = self.matrix_output_projection(normed_matrix_reads)
                if normed_aux_matrix_reads is not None:
                    projected = projected + self.aux_matrix_output_projection(normed_aux_matrix_reads)
                vector_ramp = gated_sigmoid(self.vector_readout_ramp_logit).view(1, 1, self.d_model)
                projected = projected + vector_ramp * self.vector_output_projection(normed_vector_states)
            else:
                read_parts = [normed_matrix_reads]
                if normed_aux_matrix_reads is not None:
                    read_parts.append(normed_aux_matrix_reads)
                read_parts.append(normed_vector_states)
                fused_reads = torch.cat(read_parts, dim=-1)
                projected = self.output_projection(fused_reads)
        return output_gates * projected

    def diagnostic_payload(self) -> dict[str, object]:
        return {
            "kind": "gdnp-fused",
            "law": self.law_kind,
            "modulates_beta": self.modulates_beta,
            "modulates_qkv": self.modulates_qkv,
            "uses_multi_read": self.uses_multi_read,
            "uses_residual_readout": self.uses_residual_readout,
            "primitive_runtime_backend": self._primitive_runtime_backend,
            "triton_matrix_scan": self._primitive_runtime_backend == "triton" and self.law_kind == "multi-read",
            "vector_state_width": self.d_model,
            "matrix_heads": self.head_count,
            "matrix_head_dim": self.head_dim,
            "state_transform_mode": self.state_transform_mode.value,
        }


class ReferenceSsmHybridBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_count: int,
        d_ff: int,
        *,
        profile: ReferenceSsmProfile,
        dtype_mode: str,
        layer_index: int = 0,
        use_pr5_scaffold: bool = False,
        p20_ramp_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_norm = SimpleRmsNorm(d_model)
        self.output_norm = SimpleRmsNorm(d_model)
        self.profile = profile
        self.layer_index = layer_index
        self.use_pr5_scaffold = use_pr5_scaffold
        config = resolve_reference_ssm_config(
            d_model=d_model,
            head_count=head_count,
            profile=profile,
            dtype_mode=dtype_mode,
            layer_index=layer_index,
            p20_ramp_init=p20_ramp_init,
        )
        if profile.is_composite:
            self.mixer = ParallelCompositeSequenceMixer(config, profile.composite_branches)
        elif profile.is_fla_gated_deltanet:
            self.mixer = FlaGatedDeltaNetSequenceMixer(config)
        elif profile.is_fla_gdn_control_shell:
            self.mixer = FlaGatedDeltaNetControlShellSequenceMixer(config)
        elif profile.is_fla_gdnp_control_conditioned:
            self.mixer = FlaGdnpControlConditionedSequenceMixer(config)
        elif profile.is_fla_gdnp_compatible:
            self.mixer = FlaGdnpCompatibleSequenceMixer(config)
        elif profile.is_gdnp_fused:
            self.mixer = GdnpFusedSequenceMixer(config)
        elif profile.is_gated_deltanet:
            self.mixer = TorchGatedDeltaNetSequenceMixer(config)
        elif profile.is_p20_scan:
            self.mixer = P20SequencePrimitiveBranch(
                d_model,
                width_factor=profile.p20_branch_width_factor,
            )
        else:
            self.mixer = OfficialMamba3SequenceMixer(config)
        self.feedforward = (
            ReluSquaredFeedForward(d_model, d_ff)
            if use_pr5_scaffold
            else PositionWiseFeedForward(d_model, d_ff)
        )
        if use_pr5_scaffold:
            self.mixer_scale = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
            self.feedforward_scale = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
            self.residual_mix = nn.Parameter(torch.stack((torch.ones(d_model), torch.zeros(d_model))).float())

    def forward(
        self,
        hidden: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        residual_anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_pr5_scaffold:
            if residual_anchor is None:
                residual_anchor = hidden
            mix = self.residual_mix.to(dtype=hidden.dtype)
            hidden = mix[0].view(1, 1, -1) * hidden + mix[1].view(1, 1, -1) * residual_anchor
        with record_function("path1.reference_ssm.input_norm"):
            normed = self.input_norm(hidden)
        mixed = self.mixer(normed, attn_mask)
        if self.use_pr5_scaffold:
            mixed = self.mixer_scale.to(dtype=hidden.dtype).view(1, 1, -1) * mixed
        residual = hidden + mixed
        with record_function("path1.reference_ssm.feedforward"):
            feedforward = self.feedforward(self.output_norm(residual))
            if self.use_pr5_scaffold:
                feedforward = self.feedforward_scale.to(dtype=hidden.dtype).view(1, 1, -1) * feedforward
            return residual + feedforward

    def diagnostic_payload(self) -> dict[str, object]:
        mixer_diagnostic = getattr(self.mixer, "diagnostic_payload", None)
        payload = mixer_diagnostic() if callable(mixer_diagnostic) else {}
        if not isinstance(payload, dict):
            payload = {}
        return {
            "layer_index": self.layer_index,
            "profile": self.profile.value,
            "pr5_scaffold": self.use_pr5_scaffold,
            "mixer": payload,
        }

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        configure = getattr(self.mixer, "configure_runtime_policy", None)
        if callable(configure):
            configure(
                compile_mode=compile_mode,
                primitive_runtime_backend=primitive_runtime_backend,
            )
