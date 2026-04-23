from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from python.reporting.render import render_path1_table
from python.runners.path1 import Path1RunnerRequest, run_path1_variant
from python.specs.common import (
    BenchmarkBudgetSpec,
    BenchmarkRunManifest,
    DeviceRuntimeSpec,
    JsonlCorpusSpec,
    SeedSpec,
    TokenIdCorpusSpec,
    ValidationError,
)
from python.specs.path1 import (
    AttentionProfile,
    DEFAULT_PATH1_MODEL_SHAPE,
    FeedForwardProfile,
    HybridAttentionLayerRole,
    Path1VariantKind,
    Path1ScaffoldProfile,
    Path1ModelShape,
    PrimitiveExecutionProfile,
    PrimitiveNormMode,
    PrimitiveProfile,
    PrimitiveReadoutMode,
    PrimitiveResidualMode,
    PrimitiveStateTransformMode,
    PrimitiveWrapperMode,
    RecurrentHaltingProfile,
    RecurrentTokenRoutingProfile,
    ReferenceSsmProfile,
    TokenRoutingProfile,
    parse_layer_schedule_spec,
    parse_layer_index_spec,
    parse_reference_ssm_profile_schedule_spec,
    phase1_attention_only_variant,
    phase1_primitive_variant,
    phase1_reference_ssm_variant,
)

CUDA_FAITHFUL_SMALL_V1 = "cuda-faithful-small-v1"
CUDA_FAITHFUL_SMALL_CORPUS = "fineweb-stage0-local-bench-9row-v1"
CUDA_FAITHFUL_SMALL_RELATIVE_DIR = Path(
    "experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Python Path 1 hybrid benchmark on the shared research substrate."
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=[kind.value for kind in Path1VariantKind],
    )
    parser.add_argument(
        "--reference-ssm-profile",
        default=ReferenceSsmProfile.MAMBA3_SISO_RUNTIME.value,
        choices=[profile.value for profile in ReferenceSsmProfile],
    )
    parser.add_argument(
        "--reference-ssm-profile-schedule",
        help=(
            "Whitespace or comma separated ReferenceSsmProfile values, one per R layer. "
            "Use with --layer-schedule for sparse GDN/P20 insertion."
        ),
    )
    parser.add_argument(
        "--scaffold-profile",
        default=Path1ScaffoldProfile.STANDARD.value,
        choices=[profile.value for profile in Path1ScaffoldProfile],
    )
    parser.add_argument(
        "--parcae-loop-count",
        type=int,
        default=2,
        help="Number of recurrent passes through the middle attention block for parcae-looped-attention.",
    )
    parser.add_argument(
        "--parcae-p20-value-scale",
        type=float,
        default=1.0,
        help="Scale applied to the Parcae P20-control value projection before loop injection.",
    )
    parser.add_argument(
        "--attention-profile",
        default=AttentionProfile.STANDARD.value,
        choices=[profile.value for profile in AttentionProfile],
    )
    parser.add_argument(
        "--depth-memory-layers",
        type=int,
        default=2,
        help="Number of prior block outputs exposed to MoDA-style depth attention.",
    )
    parser.add_argument(
        "--recurrent-halting-profile",
        default=RecurrentHaltingProfile.FIXED.value,
        choices=[profile.value for profile in RecurrentHaltingProfile],
    )
    parser.add_argument(
        "--recurrent-min-steps",
        type=int,
        default=1,
        help="Minimum recurrent loop steps before adaptive halting may fire.",
    )
    parser.add_argument(
        "--recurrent-halting-threshold",
        type=float,
        default=0.01,
        help="Hidden-state halting threshold for adaptive recurrent exits.",
    )
    parser.add_argument(
        "--token-routing-profile",
        default=TokenRoutingProfile.NONE.value,
        choices=[profile.value for profile in TokenRoutingProfile],
        help="Optional causal token-routed block execution profile.",
    )
    parser.add_argument(
        "--token-route-fraction",
        type=float,
        default=0.25,
        help="Approximate fraction of tokens routed through token-routed blocks.",
    )
    parser.add_argument(
        "--token-routing-layer-indices",
        help="Comma or whitespace separated layer indices for token-routed blocks.",
    )
    parser.add_argument(
        "--recurrent-token-routing-profile",
        default=RecurrentTokenRoutingProfile.NONE.value,
        choices=[profile.value for profile in RecurrentTokenRoutingProfile],
        help="Optional token-selective routing profile inside recurrent Parcae loops.",
    )
    parser.add_argument(
        "--recurrent-token-route-fraction",
        type=float,
        default=0.25,
        help="Approximate fraction of tokens refined by token-selective recurrence.",
    )
    parser.add_argument(
        "--act-halting-threshold",
        type=float,
        default=0.99,
        help="Cumulative per-token halt threshold for universal-transformer-act.",
    )
    parser.add_argument(
        "--act-ponder-loss-weight",
        type=float,
        default=0.01,
        help="Auxiliary ponder-loss weight for universal-transformer-act.",
    )
    parser.add_argument(
        "--ouro-entropy-weight",
        type=float,
        default=0.05,
        help="Stage-1 exit-distribution entropy reward weight for ouro-learned-exit.",
    )
    parser.add_argument(
        "--ouro-q-exit-threshold",
        type=float,
        default=0.5,
        help="Deterministic CDF threshold used for ouro-learned-exit q-exit diagnostics.",
    )
    parser.add_argument(
        "--mor-router-aux-loss-weight",
        type=float,
        default=0.01,
        help="Auxiliary BCE router-loss weight for mor-expert-choice.",
    )
    parser.add_argument(
        "--mor-update-scale",
        type=float,
        default=0.1,
        help="Maximum sigmoid gate scale for mor-expert-choice selected-token updates.",
    )
    parser.add_argument(
        "--reference-p20-ramp-init",
        type=float,
        default=0.01,
        help="Initial P20 contribution ramp for FLA GDN/P20-compatible reference profiles.",
    )
    parser.add_argument(
        "--primitive-profile",
        default=PrimitiveProfile.P20.value,
        choices=[profile.value for profile in PrimitiveProfile],
    )
    parser.add_argument(
        "--primitive-execution-profile",
        default=PrimitiveExecutionProfile.REFERENCE.value,
        choices=[profile.value for profile in PrimitiveExecutionProfile],
    )
    parser.add_argument(
        "--primitive-residual-profile",
        default=PrimitiveResidualMode.PLAIN.value,
        choices=[mode.value for mode in PrimitiveResidualMode],
    )
    parser.add_argument(
        "--primitive-readout-profile",
        default=PrimitiveReadoutMode.DIRECT.value,
        choices=[mode.value for mode in PrimitiveReadoutMode],
    )
    parser.add_argument(
        "--primitive-norm-profile",
        default=PrimitiveNormMode.PRE_NORM_ONLY.value,
        choices=[mode.value for mode in PrimitiveNormMode],
    )
    parser.add_argument(
        "--primitive-wrapper-profile",
        default=PrimitiveWrapperMode.STANDARD.value,
        choices=[mode.value for mode in PrimitiveWrapperMode],
    )
    parser.add_argument(
        "--primitive-state-transform-profile",
        default=PrimitiveStateTransformMode.DENSE.value,
        choices=[mode.value for mode in PrimitiveStateTransformMode],
    )
    parser.add_argument("--backend", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument(
        "--env-kind",
        choices=[
            "requirements-only",
            "official-mamba3",
            "compile-safe",
            "primitive-triton",
        ],
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument(
        "--primitive-runtime-backend",
        default="torch",
        choices=["torch", "triton"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--head-count", type=int)
    parser.add_argument("--total-layers", type=int)
    parser.add_argument("--local-window", type=int)
    parser.add_argument("--ffn-multiplier", type=int)
    parser.add_argument(
        "--feed-forward-profile",
        default=FeedForwardProfile.STANDARD.value,
        choices=[profile.value for profile in FeedForwardProfile],
    )
    parser.add_argument("--eml-slot-count", type=int, default=8)
    parser.add_argument("--eml-tree-depth", type=int, default=3)
    parser.add_argument("--eml-route-fraction", type=float, default=0.25)
    parser.add_argument(
        "--feed-forward-layer-indices",
        help="Comma or whitespace separated zero-based layer indices for non-standard attention FFNs.",
    )
    parser.add_argument("--layer-schedule")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--full-train-pass", action="store_true")
    parser.add_argument("--full-eval-pass", action="store_true")
    parser.add_argument("--warmup-eval-batches", type=int, default=1)
    parser.add_argument("--warmup-train-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--benchmark-profile", choices=[CUDA_FAITHFUL_SMALL_V1])
    parser.add_argument("--benchmark-name")
    parser.add_argument(
        "--corpus-format", default="jsonl-text", choices=["jsonl-text", "token-ids"]
    )
    parser.add_argument("--jsonl-train-path", type=Path)
    parser.add_argument("--jsonl-eval-path", type=Path)
    parser.add_argument("--tokenized-manifest-path", type=Path)
    parser.add_argument("--corpus-name")
    parser.add_argument("--corpus-text-field", default="text")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", choices=["table", "json"], default="table")
    return parser


def _resolve_corpus_args(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    repo_root: Path,
) -> tuple[JsonlCorpusSpec | TokenIdCorpusSpec, BenchmarkBudgetSpec]:
    benchmark_name = args.benchmark_name
    if args.corpus_format == "token-ids":
        if args.benchmark_profile is not None:
            parser.error(
                "--corpus-format token-ids may not be combined with --benchmark-profile"
            )
        if args.jsonl_train_path is not None or args.jsonl_eval_path is not None:
            parser.error(
                "--corpus-format token-ids may not be combined with JSONL corpus paths"
            )
        if args.tokenized_manifest_path is None:
            parser.error(
                "--tokenized-manifest-path is required with --corpus-format token-ids"
            )
        corpus = TokenIdCorpusSpec(manifest_path=args.tokenized_manifest_path)
        try:
            manifest_payload = json.loads(
                args.tokenized_manifest_path.read_text(encoding="utf-8")
            )
            tokenizer_payload = manifest_payload.get("tokenizer", {})
            manifest_vocab_size = tokenizer_payload.get("vocab_size")
        except Exception as exc:
            parser.error(
                f"failed to read tokenized manifest for vocab-size inference: {exc}"
            )
        if not isinstance(manifest_vocab_size, int) or manifest_vocab_size <= 0:
            parser.error(
                "tokenized manifest tokenizer.vocab_size must be a positive integer"
            )
        if args.vocab_size is None:
            args.vocab_size = manifest_vocab_size
        elif args.vocab_size != manifest_vocab_size:
            parser.error(
                "--vocab-size must match tokenized manifest tokenizer.vocab_size "
                f"({manifest_vocab_size})"
            )
        full_train_pass = args.full_train_pass
        full_eval_pass = args.full_eval_pass
    elif args.benchmark_profile == CUDA_FAITHFUL_SMALL_V1:
        if args.jsonl_train_path is not None or args.jsonl_eval_path is not None:
            parser.error(
                f"--benchmark-profile {CUDA_FAITHFUL_SMALL_V1} may not be combined with explicit JSONL corpus paths"
            )
        corpus_dir = repo_root / CUDA_FAITHFUL_SMALL_RELATIVE_DIR
        train_path = corpus_dir / "train.jsonl"
        eval_path = corpus_dir / "eval.jsonl"
        corpus_name = args.corpus_name or CUDA_FAITHFUL_SMALL_CORPUS
        full_train_pass = True
        full_eval_pass = True
        if benchmark_name is None:
            benchmark_name = CUDA_FAITHFUL_SMALL_V1
        corpus = JsonlCorpusSpec(
            train_path=train_path,
            eval_path=eval_path,
            corpus_name=corpus_name,
            text_field=args.corpus_text_field,
        )
    else:
        if args.tokenized_manifest_path is not None:
            parser.error("--tokenized-manifest-path requires --corpus-format token-ids")
        if args.jsonl_train_path is None or args.jsonl_eval_path is None:
            parser.error(
                "either --benchmark-profile or both --jsonl-train-path and --jsonl-eval-path are required"
            )
        train_path = args.jsonl_train_path
        eval_path = args.jsonl_eval_path
        corpus_name = args.corpus_name or "jsonl-corpus"
        full_train_pass = args.full_train_pass
        full_eval_pass = args.full_eval_pass
        corpus = JsonlCorpusSpec(
            train_path=train_path,
            eval_path=eval_path,
            corpus_name=corpus_name,
            text_field=args.corpus_text_field,
        )

    budget = BenchmarkBudgetSpec(
        seq_len=args.seq_len,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        train_steps=args.steps,
        eval_batches=args.eval_batches,
        full_train_pass=full_train_pass,
        full_eval_pass=full_eval_pass,
        learning_rate=args.learning_rate,
        warmup_eval_batches=args.warmup_eval_batches,
        warmup_train_steps=args.warmup_train_steps,
    )
    args.benchmark_name = benchmark_name
    return corpus, budget


def _build_shape_and_schedule(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
) -> tuple[Path1ModelShape, tuple[HybridAttentionLayerRole, ...] | None]:
    layer_schedule = (
        parse_layer_schedule_spec(args.layer_schedule) if args.layer_schedule else None
    )
    if layer_schedule is None:
        total_layers = (
            args.total_layers
            if args.total_layers is not None
            else DEFAULT_PATH1_MODEL_SHAPE.total_layers
        )
    else:
        if args.total_layers is not None and args.total_layers != len(layer_schedule):
            parser.error(
                "--total-layers must match the explicit --layer-schedule length"
            )
        total_layers = len(layer_schedule)
    shape = Path1ModelShape(
        vocab_size=(
            args.vocab_size
            if args.vocab_size is not None
            else DEFAULT_PATH1_MODEL_SHAPE.vocab_size
        ),
        d_model=(
            args.d_model
            if args.d_model is not None
            else DEFAULT_PATH1_MODEL_SHAPE.d_model
        ),
        head_count=(
            args.head_count
            if args.head_count is not None
            else DEFAULT_PATH1_MODEL_SHAPE.head_count
        ),
        total_layers=total_layers,
        local_window=(
            args.local_window
            if args.local_window is not None
            else DEFAULT_PATH1_MODEL_SHAPE.local_window
        ),
        ffn_multiplier=(
            args.ffn_multiplier
            if args.ffn_multiplier is not None
            else DEFAULT_PATH1_MODEL_SHAPE.ffn_multiplier
        ),
    )
    try:
        shape.validate()
    except ValidationError as exc:
        parser.error(str(exc))
    return shape, layer_schedule


def _build_variant(args: argparse.Namespace, *, parser: argparse.ArgumentParser):
    shape, layer_schedule = _build_shape_and_schedule(args, parser=parser)
    kind = Path1VariantKind(args.variant)
    if kind is Path1VariantKind.ATTENTION_ONLY:
        return phase1_attention_only_variant(
            shape=shape,
            layer_schedule=layer_schedule,
            feed_forward_profile=FeedForwardProfile(args.feed_forward_profile),
            feed_forward_layer_indices=(
                parse_layer_index_spec(args.feed_forward_layer_indices)
                if args.feed_forward_layer_indices
                else None
            ),
            eml_slot_count=args.eml_slot_count,
            eml_tree_depth=args.eml_tree_depth,
            eml_route_fraction=args.eml_route_fraction,
            scaffold_profile=Path1ScaffoldProfile(args.scaffold_profile),
            parcae_loop_count=args.parcae_loop_count,
            parcae_p20_value_scale=args.parcae_p20_value_scale,
            attention_profile=AttentionProfile(args.attention_profile),
            depth_memory_layers=args.depth_memory_layers,
            recurrent_halting_profile=RecurrentHaltingProfile(
                args.recurrent_halting_profile
            ),
            recurrent_min_steps=args.recurrent_min_steps,
            recurrent_halting_threshold=args.recurrent_halting_threshold,
            token_routing_profile=TokenRoutingProfile(args.token_routing_profile),
            token_route_fraction=args.token_route_fraction,
            token_routing_layer_indices=(
                parse_layer_index_spec(args.token_routing_layer_indices)
                if args.token_routing_layer_indices
                else None
            ),
            recurrent_token_routing_profile=RecurrentTokenRoutingProfile(
                args.recurrent_token_routing_profile
            ),
            recurrent_token_route_fraction=args.recurrent_token_route_fraction,
            act_halting_threshold=args.act_halting_threshold,
            act_ponder_loss_weight=args.act_ponder_loss_weight,
            ouro_entropy_weight=args.ouro_entropy_weight,
            ouro_q_exit_threshold=args.ouro_q_exit_threshold,
            mor_router_aux_loss_weight=args.mor_router_aux_loss_weight,
            mor_update_scale=args.mor_update_scale,
        )
    if kind is Path1VariantKind.REFERENCE_SSM_HYBRID:
        feed_forward_layer_indices = (
            parse_layer_index_spec(args.feed_forward_layer_indices)
            if args.feed_forward_layer_indices
            else None
        )
        return phase1_reference_ssm_variant(
            shape=shape,
            profile=ReferenceSsmProfile(args.reference_ssm_profile),
            layer_schedule=layer_schedule,
            profile_schedule=(
                parse_reference_ssm_profile_schedule_spec(
                    args.reference_ssm_profile_schedule
                )
                if args.reference_ssm_profile_schedule
                else None
            ),
            scaffold_profile=Path1ScaffoldProfile(args.scaffold_profile),
            reference_p20_ramp_init=args.reference_p20_ramp_init,
            feed_forward_profile=FeedForwardProfile(args.feed_forward_profile),
            feed_forward_layer_indices=feed_forward_layer_indices,
            eml_slot_count=args.eml_slot_count,
            eml_tree_depth=args.eml_tree_depth,
            eml_route_fraction=args.eml_route_fraction,
            attention_profile=AttentionProfile(args.attention_profile),
            depth_memory_layers=args.depth_memory_layers,
            token_routing_profile=TokenRoutingProfile(args.token_routing_profile),
            token_route_fraction=args.token_route_fraction,
            token_routing_layer_indices=(
                parse_layer_index_spec(args.token_routing_layer_indices)
                if args.token_routing_layer_indices
                else None
            ),
        )
    feed_forward_layer_indices = (
        parse_layer_index_spec(args.feed_forward_layer_indices)
        if args.feed_forward_layer_indices
        else None
    )
    return phase1_primitive_variant(
        shape=shape,
        primitive_profile=PrimitiveProfile(args.primitive_profile),
        execution_profile=PrimitiveExecutionProfile(args.primitive_execution_profile),
        residual_mode=PrimitiveResidualMode(args.primitive_residual_profile),
        readout_mode=PrimitiveReadoutMode(args.primitive_readout_profile),
        norm_mode=PrimitiveNormMode(args.primitive_norm_profile),
        wrapper_mode=PrimitiveWrapperMode(args.primitive_wrapper_profile),
        state_transform_mode=PrimitiveStateTransformMode(
            args.primitive_state_transform_profile
        ),
        layer_schedule=layer_schedule,
        feed_forward_profile=FeedForwardProfile(args.feed_forward_profile),
        feed_forward_layer_indices=feed_forward_layer_indices,
        eml_slot_count=args.eml_slot_count,
        eml_tree_depth=args.eml_tree_depth,
        eml_route_fraction=args.eml_route_fraction,
        attention_profile=AttentionProfile(args.attention_profile),
        depth_memory_layers=args.depth_memory_layers,
        token_routing_profile=TokenRoutingProfile(args.token_routing_profile),
        token_route_fraction=args.token_route_fraction,
        token_routing_layer_indices=(
            parse_layer_index_spec(args.token_routing_layer_indices)
            if args.token_routing_layer_indices
            else None
        ),
    )


def _implementation_kind_for_variant(variant, *, primitive_runtime_backend: str) -> str:
    if variant.kind is Path1VariantKind.ATTENTION_ONLY:
        if variant.token_routing_profile is not TokenRoutingProfile.NONE:
            return f"python_attention_{variant.token_routing_profile.value.replace('-', '_')}"
        if (
            variant.recurrent_token_routing_profile
            is not RecurrentTokenRoutingProfile.NONE
        ):
            return f"python_attention_parcae_{variant.recurrent_token_routing_profile.value.replace('-', '_')}"
        if variant.scaffold_profile in {
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
        }:
            return (
                f"python_attention_{variant.scaffold_profile.value.replace('-', '_')}"
            )
        if variant.attention_profile is not AttentionProfile.STANDARD:
            return (
                f"python_attention_{variant.attention_profile.value.replace('-', '_')}"
            )
        if variant.feed_forward_profile is not FeedForwardProfile.STANDARD:
            return f"python_attention_{variant.feed_forward_profile.value.replace('-', '_')}"
        if variant.recurrent_halting_profile is not RecurrentHaltingProfile.FIXED:
            return f"python_attention_parcae_{variant.recurrent_halting_profile.value.replace('-', '_')}"
        return "python_attention_sdpa"
    if variant.kind is Path1VariantKind.REFERENCE_SSM_HYBRID:
        profiles = variant.reference_ssm_profile_schedule or (
            variant.reference_ssm_profile,
        )
        if len(set(profiles)) > 1:
            return "python_reference_ssm_profile_scheduled"
        if variant.reference_ssm_profile.is_composite:
            return f"python_reference_ssm_composite_{variant.reference_ssm_profile.value.replace('-', '_')}"
        if variant.reference_ssm_profile is ReferenceSsmProfile.GATED_DELTANET_FLA:
            return "python_reference_ssm_gated_deltanet_fla"
        if variant.reference_ssm_profile.is_fla_gdn_control_shell:
            return "python_reference_ssm_gated_deltanet_fla_control_shell"
        if variant.reference_ssm_profile.is_fla_gdnp_control_conditioned:
            return "python_reference_ssm_gdnp_fla_control_tiny"
        if variant.reference_ssm_profile.is_fla_gdnp_compatible:
            law = variant.reference_ssm_profile.fla_gdnp_compatible_law.replace(
                "-", "_"
            )
            return f"python_reference_ssm_gdnp_fla_compatible_{law}"
        if variant.reference_ssm_profile.is_gdnp_fused:
            law = variant.reference_ssm_profile.gdnp_fused_law.replace("-", "_")
            if primitive_runtime_backend == "triton":
                if law == "multi_read":
                    return f"python_reference_ssm_gdnp_fused_{law}_triton_vector_matrix"
                return f"python_reference_ssm_gdnp_fused_{law}_triton_vector"
            return f"python_reference_ssm_gdnp_fused_{law}_torch"
        if variant.reference_ssm_profile is ReferenceSsmProfile.GATED_DELTANET_TORCH:
            return "python_reference_ssm_gated_deltanet_torch"
        if variant.reference_ssm_profile.is_p20_scan:
            return f"python_reference_ssm_{variant.reference_ssm_profile.value.replace('-', '_')}"
        return (
            "python_reference_ssm_native_runtime"
            if variant.reference_ssm_profile.runtime_oriented
            else "python_reference_ssm_reference"
        )
    if primitive_runtime_backend == "triton":
        return "python_primitive_triton_runtime"
    return (
        "python_primitive_runtime"
        if variant.primitive_execution_profile is PrimitiveExecutionProfile.RUNTIME
        else "python_primitive_reference"
    )


def build_request_from_args(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    repo_root: Path,
) -> Path1RunnerRequest:
    corpus, budget = _resolve_corpus_args(args, parser=parser, repo_root=repo_root)
    variant = _build_variant(args, parser=parser)
    output_dir = args.output_dir or repo_root / "artifacts" / "v3a-python-path1"

    manifest = BenchmarkRunManifest(
        run_label=args.run_label,
        implementation_kind=_implementation_kind_for_variant(
            variant,
            primitive_runtime_backend=args.primitive_runtime_backend,
        ),
        benchmark_name=args.benchmark_name,
        seed_spec=SeedSpec(model_seed=args.seed, data_seed=args.data_seed),
        corpus=corpus,
        budget=budget,
        runtime=DeviceRuntimeSpec(
            backend=args.backend,
            cuda_device=args.cuda_device,
            dtype=args.dtype,
            env_kind=args.env_kind,
            compile_mode=args.compile_mode,
            primitive_runtime_backend=args.primitive_runtime_backend,
        ),
    )
    return Path1RunnerRequest(
        manifest=manifest,
        variant=variant,
        output_dir=output_dir,
        output_format=args.output,
        ledger_path=args.ledger_path,
    )


def cli_main(argv: Sequence[str] | None = None, *, repo_root: Path) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        request = build_request_from_args(args, parser=parser, repo_root=repo_root)
        report = run_path1_variant(request)
    except ValidationError as exc:
        parser.error(str(exc))
    if args.output == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_path1_table(report, request.variant.label))
        print(f"report_path={report.report_path}")
    return 0
