from __future__ import annotations

from dataclasses import dataclass, field

from python.specs.common import StringEnum, ValidationError, ensure_positive


class JaxTpuArchitectureFamily(StringEnum):
    ATTENTION_BASELINE = "attention-baseline"
    PARCAE_LOOPED_ATTENTION = "parcae-looped-attention"
    ROTARY_GATED_RECURRENT_STATE_UPDATE = "rotary-gated-recurrent-state-update"
    GATED_DELTANET = "gated-deltanet"
    MAMBA3 = "mamba3"
    CUSTOM = "custom"


class JaxTpuDatasetType(StringEnum):
    SYNTHETIC = "synthetic"
    HF = "hf"
    TFDS = "tfds"


@dataclass(frozen=True)
class JaxTpuKernelContract:
    architecture_family: JaxTpuArchitectureFamily
    lowering: str
    sequence_axis: str = "tokens"
    feature_axis: str = "channels"
    recurrence_axis: str | None = None
    carries_state_across_tokens: bool = False
    fusion_boundary: str = "maxtext-native-transformer-block"
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.lowering.strip():
            raise ValidationError("kernel_contract.lowering must not be empty")
        if not self.sequence_axis.strip():
            raise ValidationError("kernel_contract.sequence_axis must not be empty")
        if not self.feature_axis.strip():
            raise ValidationError("kernel_contract.feature_axis must not be empty")
        if self.carries_state_across_tokens and not self.recurrence_axis:
            raise ValidationError(
                "kernel_contract.recurrence_axis is required when carries_state_across_tokens=true"
            )
        if not self.fusion_boundary.strip():
            raise ValidationError("kernel_contract.fusion_boundary must not be empty")


@dataclass(frozen=True)
class JaxTpuCandidateSpec:
    slug: str
    label: str
    kernel_contract: JaxTpuKernelContract
    source_profile: str
    maxtext_model_name: str | None = None
    adapter_module: str | None = None
    adapter_overrides: dict[str, str | int | float | bool] = field(default_factory=dict)
    requires_patched_maxtext: bool = False

    def validate(self) -> None:
        if not self.slug.strip():
            raise ValidationError("candidate.slug must not be empty")
        if not self.label.strip():
            raise ValidationError("candidate.label must not be empty")
        if not self.source_profile.strip():
            raise ValidationError("candidate.source_profile must not be empty")
        self.kernel_contract.validate()
        if self.requires_patched_maxtext and not self.adapter_module:
            raise ValidationError(
                "candidate.adapter_module is required when requires_patched_maxtext=true"
            )


@dataclass(frozen=True)
class JaxTpuModelShape:
    vocab_size: int = 32_000
    sequence_length: int = 1_024
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ffn_multiplier: int = 4
    moe_experts: int = 0
    moe_top_k: int = 0

    @property
    def mlp_dim(self) -> int:
        return self.d_model * self.ffn_multiplier

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def validate(self) -> None:
        ensure_positive(self.vocab_size, "shape.vocab_size")
        ensure_positive(self.sequence_length, "shape.sequence_length")
        ensure_positive(self.d_model, "shape.d_model")
        ensure_positive(self.num_layers, "shape.num_layers")
        ensure_positive(self.num_heads, "shape.num_heads")
        ensure_positive(self.ffn_multiplier, "shape.ffn_multiplier")
        if self.d_model % self.num_heads != 0:
            raise ValidationError("shape.d_model must be divisible by shape.num_heads")
        if self.moe_experts < 0:
            raise ValidationError("shape.moe_experts must be non-negative")
        if self.moe_top_k < 0:
            raise ValidationError("shape.moe_top_k must be non-negative")
        if self.moe_top_k > 0 and self.moe_experts <= 0:
            raise ValidationError("shape.moe_experts must be positive when shape.moe_top_k is positive")
        if self.moe_experts > 0 and not 0 < self.moe_top_k <= self.moe_experts:
            raise ValidationError("shape.moe_top_k must be in 1..shape.moe_experts for MoE runs")


@dataclass(frozen=True)
class JaxTpuParallelismSpec:
    ici_data_parallelism: int = 1
    ici_fsdp_parallelism: int = 1
    ici_tensor_parallelism: int = 1
    ici_expert_parallelism: int = 1
    dcn_data_parallelism: int = 1
    dcn_fsdp_parallelism: int = 1
    dcn_tensor_parallelism: int = 1
    dcn_expert_parallelism: int = 1

    @property
    def product(self) -> int:
        value = 1
        for item in self.to_overrides().values():
            value *= int(item)
        return value

    def validate(self) -> None:
        for key, value in self.to_overrides().items():
            ensure_positive(value, f"parallelism.{key}")

    def to_overrides(self) -> dict[str, int]:
        return {
            "ici_data_parallelism": self.ici_data_parallelism,
            "ici_fsdp_parallelism": self.ici_fsdp_parallelism,
            "ici_tensor_parallelism": self.ici_tensor_parallelism,
            "ici_expert_parallelism": self.ici_expert_parallelism,
            "dcn_data_parallelism": self.dcn_data_parallelism,
            "dcn_fsdp_parallelism": self.dcn_fsdp_parallelism,
            "dcn_tensor_parallelism": self.dcn_tensor_parallelism,
            "dcn_expert_parallelism": self.dcn_expert_parallelism,
        }


@dataclass(frozen=True)
class JaxTpuDatasetSpec:
    dataset_type: JaxTpuDatasetType = JaxTpuDatasetType.SYNTHETIC
    dataset_path: str | None = None
    dataset_name: str | None = None
    train_split: str = "train"
    eval_split: str | None = None
    hf_path: str | None = None
    hf_name: str | None = None
    hf_data_dir: str | None = None
    hf_train_files: str | None = None
    hf_eval_split: str | None = None
    hf_eval_files: str | None = None

    def validate(self) -> None:
        if self.dataset_type is JaxTpuDatasetType.SYNTHETIC:
            return
        if self.dataset_type is JaxTpuDatasetType.HF:
            if not self.hf_path:
                raise ValidationError("dataset.hf_path is required for dataset_type=hf")
            if not self.train_split.strip():
                raise ValidationError("dataset.train_split must not be empty")
            return
        if self.dataset_type is JaxTpuDatasetType.TFDS:
            if not self.dataset_path:
                raise ValidationError("dataset.dataset_path is required for dataset_type=tfds")
            if not self.dataset_name:
                raise ValidationError("dataset.dataset_name is required for dataset_type=tfds")
            if not self.train_split.strip():
                raise ValidationError("dataset.train_split must not be empty")
            return
        raise ValidationError(f"unsupported JAX/TPU dataset type: {self.dataset_type}")

    def to_overrides(self) -> dict[str, str]:
        overrides = {"dataset_type": self.dataset_type.value}
        if self.dataset_path:
            overrides["dataset_path"] = self.dataset_path
        if self.dataset_name:
            overrides["dataset_name"] = self.dataset_name
        if self.train_split:
            overrides["train_split"] = self.train_split
        if self.eval_split:
            overrides["eval_split"] = self.eval_split
        if self.hf_path:
            overrides["hf_path"] = self.hf_path
        if self.hf_name:
            overrides["hf_name"] = self.hf_name
        if self.hf_data_dir:
            overrides["hf_data_dir"] = self.hf_data_dir
        if self.hf_train_files:
            overrides["hf_train_files"] = self.hf_train_files
        if self.hf_eval_split:
            overrides["hf_eval_split"] = self.hf_eval_split
        if self.hf_eval_files:
            overrides["hf_eval_files"] = self.hf_eval_files
        return overrides


@dataclass(frozen=True)
class JaxTpuTokenizerSpec:
    tokenizer_type: str | None = None
    tokenizer_path: str | None = None

    def validate(self) -> None:
        if self.tokenizer_type is not None and self.tokenizer_type not in {"huggingface", "sentencepiece", "tiktoken"}:
            raise ValidationError("tokenizer.tokenizer_type must be one of huggingface|sentencepiece|tiktoken")
        if self.tokenizer_type and not self.tokenizer_path:
            raise ValidationError("tokenizer.tokenizer_path is required when tokenizer.tokenizer_type is set")
        if self.tokenizer_path and not self.tokenizer_type:
            raise ValidationError("tokenizer.tokenizer_type is required when tokenizer.tokenizer_path is set")

    def to_overrides(self) -> dict[str, str]:
        overrides: dict[str, str] = {}
        if self.tokenizer_type:
            overrides["tokenizer_type"] = self.tokenizer_type
        if self.tokenizer_path:
            overrides["tokenizer_path"] = self.tokenizer_path
        return overrides


@dataclass(frozen=True)
class JaxTpuRunBudget:
    steps: int = 10
    per_device_batch_size: int = 1
    learning_rate: float = 1.0e-3
    enable_checkpointing: bool = False

    def validate(self) -> None:
        ensure_positive(self.steps, "budget.steps")
        ensure_positive(self.per_device_batch_size, "budget.per_device_batch_size")
        if self.learning_rate <= 0:
            raise ValidationError("budget.learning_rate must be greater than zero")

    def to_overrides(self) -> dict[str, str | int | float | bool]:
        return {
            "steps": self.steps,
            "per_device_batch_size": self.per_device_batch_size,
            "learning_rate": self.learning_rate,
            "enable_checkpointing": self.enable_checkpointing,
        }


@dataclass(frozen=True)
class JaxTpuBenchmarkSpec:
    run_name: str
    base_output_directory: str
    candidate: JaxTpuCandidateSpec
    shape: JaxTpuModelShape = field(default_factory=JaxTpuModelShape)
    parallelism: JaxTpuParallelismSpec = field(default_factory=JaxTpuParallelismSpec)
    dataset: JaxTpuDatasetSpec = field(default_factory=JaxTpuDatasetSpec)
    tokenizer: JaxTpuTokenizerSpec = field(default_factory=JaxTpuTokenizerSpec)
    budget: JaxTpuRunBudget = field(default_factory=JaxTpuRunBudget)
    dtype: str = "bfloat16"
    extra_overrides: dict[str, str | int | float | bool] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.run_name.strip():
            raise ValidationError("benchmark.run_name must not be empty")
        if not self.base_output_directory.strip():
            raise ValidationError("benchmark.base_output_directory must not be empty")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValidationError("benchmark.dtype must be one of bfloat16|float32")
        self.candidate.validate()
        self.shape.validate()
        self.parallelism.validate()
        self.dataset.validate()
        self.tokenizer.validate()
        self.budget.validate()
        if self.dataset.dataset_type is JaxTpuDatasetType.HF and not self.tokenizer.tokenizer_path:
            raise ValidationError("benchmark.tokenizer is required for dataset_type=hf")

    def to_maxtext_overrides(self, *, include_adapter_overrides: bool = False) -> dict[str, str | int | float | bool]:
        self.validate()
        overrides: dict[str, str | int | float | bool] = {
            "run_name": self.run_name,
            "base_output_directory": self.base_output_directory,
            "max_target_length": self.shape.sequence_length,
            "vocab_size": self.shape.vocab_size,
            "base_emb_dim": self.shape.d_model,
            "base_num_decoder_layers": self.shape.num_layers,
            "base_num_query_heads": self.shape.num_heads,
            "base_num_kv_heads": self.shape.num_heads,
            "head_dim": self.shape.head_dim,
            "base_mlp_dim": self.shape.mlp_dim,
            "dtype": self.dtype,
        }
        if self.candidate.maxtext_model_name:
            overrides["model_name"] = self.candidate.maxtext_model_name
        if self.shape.moe_experts:
            overrides["num_experts"] = self.shape.moe_experts
            overrides["num_experts_per_tok"] = self.shape.moe_top_k
        if include_adapter_overrides and self.candidate.adapter_module:
            overrides["fractal_candidate"] = self.candidate.slug
            overrides["fractal_adapter_module"] = self.candidate.adapter_module
            overrides.update(self.candidate.adapter_overrides)
        overrides.update(self.parallelism.to_overrides())
        overrides.update(self.dataset.to_overrides())
        overrides.update(self.tokenizer.to_overrides())
        overrides.update(self.budget.to_overrides())
        overrides.update(self.extra_overrides)
        return overrides


def candidate_registry() -> dict[str, JaxTpuCandidateSpec]:
    attention = JaxTpuCandidateSpec(
        slug="attention-baseline",
        label="MaxText dense attention baseline",
        source_profile="maxtext-native-transformer",
        kernel_contract=JaxTpuKernelContract(
            architecture_family=JaxTpuArchitectureFamily.ATTENTION_BASELINE,
            lowering="maxtext-native-jax",
            fusion_boundary="attention-and-mlp-native-to-maxtext",
            notes=("Use first to verify TPU setup, cost, data ingress, and log parsing.",),
        ),
    )
    p20 = JaxTpuCandidateSpec(
        slug="rotary-gated-recurrent-state-update",
        label="Fractal rotary gated recurrent state update",
        source_profile="python-path1-primitive-p2-0-runtime-family",
        adapter_module="python.jax_tpu.adapters.rotary_gated_recurrent_state_update",
        adapter_overrides={
            "decoder_block": "default",
            "fractal_rgrp_state_transform": "block-diagonal-4-masked-dense",
            "fractal_rgrp_scan_unroll": 3,
            "fractal_rgrp_projection_mode": "sequence",
            "fractal_rgrp_trig_mode": "precompute",
            "fractal_rgrp_residual_scale": 1.0,
        },
        requires_patched_maxtext=True,
        kernel_contract=JaxTpuKernelContract(
            architecture_family=JaxTpuArchitectureFamily.ROTARY_GATED_RECURRENT_STATE_UPDATE,
            lowering="jax-lax-scan-first-pallas-later",
            recurrence_axis="tokens",
            carries_state_across_tokens=True,
            fusion_boundary="ffn-side-recurrent-state-update",
            notes=(
                "First port should be a JAX lax.scan reference inside a MaxText FFN seam.",
                "Only promote to Pallas/custom lowering after baseline parity and loss signal.",
            ),
        ),
    )
    rgrp_mlp_sidecar = JaxTpuCandidateSpec(
        slug="rotary-gated-recurrent-state-update-mlp-sidecar",
        label="Fractal rotary gated recurrent state update MLP sidecar",
        source_profile="python-path1-primitive-p2-0-runtime-family-selected-layer-control",
        adapter_module="python.jax_tpu.adapters.rotary_gated_recurrent_state_update",
        adapter_overrides={
            "decoder_block": "default",
            "scan_layers": False,
            "fractal_candidate": "rotary-gated-recurrent-state-update",
            "fractal_rgrp_integration_mode": "mlp-sidecar",
            "fractal_rgrp_layers": "1",
            "fractal_rgrp_bottleneck_dim": 64,
            "fractal_rgrp_sidecar_type": "rgrp",
            "fractal_rgrp_state_transform": "block-diagonal-4-masked-dense",
            "fractal_rgrp_scan_unroll": 3,
            "fractal_rgrp_projection_mode": "sequence",
            "fractal_rgrp_trig_mode": "precompute",
            "fractal_rgrp_side_scale": 0.1,
            "fractal_rgrp_output_init": "xavier",
        },
        requires_patched_maxtext=True,
        kernel_contract=JaxTpuKernelContract(
            architecture_family=JaxTpuArchitectureFamily.ROTARY_GATED_RECURRENT_STATE_UPDATE,
            lowering="jax-lax-scan-sidecar",
            recurrence_axis="tokens",
            carries_state_across_tokens=True,
            fusion_boundary="selected-mlp-sidecar-recurrent-state-update",
            notes=(
                "Keeps MaxText attention and MLP intact and adds one bottlenecked RGRP side branch.",
                "scan_layers=false is intentional so selected-layer sidecar placement is explicit.",
                "This is the faithful TPU probe for the earlier sparse/mid-layer positive signal.",
            ),
        ),
    )
    control_candidates = []
    for sidecar_type in ("tiny-mlp", "tiny-glu", "binary-tree"):
        control_candidates.append(
            JaxTpuCandidateSpec(
                slug=f"{sidecar_type}-mlp-sidecar-control",
                label=f"Matched {sidecar_type} MLP sidecar control",
                source_profile="maxtext-selected-layer-small-sidecar-control",
                adapter_module="python.jax_tpu.adapters.rotary_gated_recurrent_state_update",
                adapter_overrides={
                    **rgrp_mlp_sidecar.adapter_overrides,
                    "fractal_rgrp_sidecar_type": sidecar_type,
                    "fractal_rgrp_output_init": "zero",
                    "fractal_rgrp_tree_depth": 2,
                    "fractal_rgrp_slot_count": 4,
                },
                requires_patched_maxtext=True,
                kernel_contract=JaxTpuKernelContract(
                    architecture_family=JaxTpuArchitectureFamily.CUSTOM,
                    lowering="jax-small-sidecar-control",
                    recurrence_axis=None,
                    carries_state_across_tokens=False,
                    fusion_boundary="selected-mlp-sidecar-control",
                    notes=(
                        "Keeps MaxText attention and MLP intact and swaps only the sidecar operator.",
                        "Used to isolate whether the RGRP quality signal is recurrent-state-specific.",
                    ),
                ),
            )
        )
    parcae_common_overrides = {
        "decoder_block": "default",
        "scan_layers": False,
        "fractal_parcae_loop_count": 2,
        "fractal_parcae_loop_policy": "fixed",
        "fractal_parcae_depth_distribution": "poisson",
        "fractal_parcae_mu_rec": 2.0,
        "fractal_parcae_mu_bwd": 2,
        "fractal_parcae_min_loop_count": 1,
        "fractal_parcae_max_loop_count": 0,
        "fractal_parcae_discretization": "stable-exp",
        "fractal_parcae_control_diagnostics": False,
    }
    parcae_looped = JaxTpuCandidateSpec(
        slug="parcae-looped-attention",
        label="Parcae looped-middle attention scaffold",
        source_profile="python-path1-parcae-looped-attention",
        adapter_module="python.jax_tpu.adapters.rotary_gated_recurrent_state_update",
        adapter_overrides={
            **parcae_common_overrides,
            "fractal_candidate": "parcae-looped-attention",
        },
        requires_patched_maxtext=True,
        kernel_contract=JaxTpuKernelContract(
            architecture_family=JaxTpuArchitectureFamily.PARCAE_LOOPED_ATTENTION,
            lowering="maxtext-unscanned-looped-middle-reference",
            recurrence_axis="decoder-layer-depth",
            carries_state_across_tokens=False,
            fusion_boundary="decoder-stack-looped-middle-scaffold",
            notes=(
                "Runs prelude layers once, reuses the middle layer band for recurrent depth, then runs coda layers.",
                "This is the non-RGRP Parcae scaffold control from the H100 proof ladder.",
            ),
        ),
    )
    parcae_bx = JaxTpuCandidateSpec(
        slug="parcae-bx-looped-attention",
        label="Parcae B(x) looped-middle attention scaffold",
        source_profile="python-path1-parcae-bx-looped-attention",
        adapter_module="python.jax_tpu.adapters.rotary_gated_recurrent_state_update",
        adapter_overrides={
            **parcae_common_overrides,
            "fractal_candidate": "parcae-bx-looped-attention",
        },
        requires_patched_maxtext=True,
        kernel_contract=JaxTpuKernelContract(
            architecture_family=JaxTpuArchitectureFamily.PARCAE_LOOPED_ATTENTION,
            lowering="maxtext-unscanned-looped-middle-bx-reference",
            recurrence_axis="decoder-layer-depth",
            carries_state_across_tokens=False,
            fusion_boundary="decoder-stack-looped-middle-bx-injection",
            notes=(
                "Adds learned B(x) value and gate projections at the Parcae loop injection seam.",
                "This is the closest current Parcae-reference lane; stochastic depth remains an explicit run override.",
            ),
        ),
    )
    parcae_rgrp_control = JaxTpuCandidateSpec(
        slug="parcae-rgrp-control-looped-attention",
        label="Parcae looped-middle scaffold with RGRP control",
        source_profile="python-path1-parcae-p20-control-looped-attention",
        adapter_module="python.jax_tpu.adapters.rotary_gated_recurrent_state_update",
        adapter_overrides={
            **parcae_common_overrides,
            "fractal_candidate": "parcae-rgrp-control-looped-attention",
            "fractal_rgrp_state_transform": "block-diagonal-4-masked-dense",
            "fractal_rgrp_scan_unroll": 3,
            "fractal_rgrp_projection_mode": "sequence",
            "fractal_rgrp_trig_mode": "precompute",
        },
        requires_patched_maxtext=True,
        kernel_contract=JaxTpuKernelContract(
            architecture_family=JaxTpuArchitectureFamily.PARCAE_LOOPED_ATTENTION,
            lowering="maxtext-unscanned-looped-middle-rgrp-control-reference",
            recurrence_axis="tokens-and-decoder-layer-depth",
            carries_state_across_tokens=True,
            fusion_boundary="decoder-stack-loop-injection-rgrp-controller",
            notes=(
                "Runs a full-width RGRP scan over the prelude stream and uses it to control the Parcae injection value/gate.",
                "Uses the same outer Parcae loop-depth policy as B(x); the RGRP token scan is not privately resampled.",
                "This is the JAX/TPU port target for the H100 P20-control winner.",
            ),
        ),
    )
    return {
        candidate.slug: candidate
        for candidate in (
            attention,
            p20,
            rgrp_mlp_sidecar,
            *control_candidates,
            parcae_looped,
            parcae_bx,
            parcae_rgrp_control,
        )
    }


def get_candidate(slug: str) -> JaxTpuCandidateSpec:
    try:
        return candidate_registry()[slug]
    except KeyError as exc:
        available = ", ".join(sorted(candidate_registry()))
        raise ValidationError(f"unknown JAX/TPU candidate {slug!r}; available: {available}") from exc
