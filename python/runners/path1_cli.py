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
    AttentionKernelProfile,
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
    ReferenceSsmProfile,
    parse_int_tuple_spec,
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
        "--parcae-hourglass-pass-count",
        type=int,
        default=1,
        help="Number of shared wide->loop->wide hourglass scaffold passes to apply.",
    )
    parser.add_argument(
        "--parcae-backward-steps",
        type=int,
        help=(
            "Number of final Parcae recurrent passes that receive gradients. "
            "Omit for full BPTT through all loop passes."
        ),
    )
    parser.add_argument(
        "--parcae-prelude-norm-kind",
        default="layernorm",
        choices=["layernorm", "rmsnorm"],
        help="Normalization used at the Parcae loop input seam.",
    )
    parser.add_argument(
        "--parcae-discretization",
        default="stable-exp",
        choices=["stable-exp", "zoh"],
        help="Stable recurrent discretization used by the Parcae state injection.",
    )
    parser.add_argument(
        "--parcae-dt-raw-init",
        type=float,
        default=0.54132485,
        help="Raw step-size initialization for Parcae ZOH discretization.",
    )
    parser.add_argument(
        "--parcae-loop-d-model",
        type=int,
        help="Internal loop width for Parcae hourglass scaffolds.",
    )
    parser.add_argument(
        "--parcae-loop-head-count",
        type=int,
        help="Internal loop attention head count for Parcae hourglass scaffolds.",
    )
    parser.add_argument(
        "--parcae-loop-ffn-multiplier",
        type=int,
        help="Internal loop FFN multiplier for Parcae hourglass scaffolds.",
    )
    parser.add_argument(
        "--parcae-loop-layer-count",
        type=int,
        help=(
            "Number of centered transformer blocks inside the Parcae recurrent loop. "
            "Omit to use the default one-third middle band."
        ),
    )
    parser.add_argument(
        "--parcae-hourglass-band-schedule",
        help=(
            "Comma or whitespace separated wide/loop/wide segment lengths for multi-band "
            "hourglass scaffolds, e.g. 3,2,3,2,2."
        ),
    )
    parser.add_argument(
        "--parcae-control-position-kind",
        choices=["none", "learned"],
        default="none",
        help="Optional explicit position features added only to the Parcae P20/RGRP controller input.",
    )
    parser.add_argument(
        "--parcae-control-position-scale-init",
        type=float,
        default=0.01,
        help="Initial scalar scale for Parcae controller position features.",
    )
    parser.add_argument(
        "--parcae-control-stride",
        type=int,
        default=1,
        help=(
            "Causal left-anchor stride for the Parcae P20/RGRP controller scan. "
            "1 preserves per-token control."
        ),
    )
    parser.add_argument(
        "--parcae-control-state-transform",
        choices=["trainable", "trainable-block-diagonal-8", "frozen-identity"],
        default="trainable",
        help=(
            "State-transform contract for the Parcae P20/RGRP controller. "
            "trainable-block-diagonal-8 keeps wider controller tiles on the fast scan-backward path; "
            "frozen-identity removes transform-parameter gradients for a speed/quality ablation."
        ),
    )
    parser.add_argument(
        "--position-encoding-kind",
        choices=["none", "learned"],
        default="none",
        help="Token position encoding added after the token embedding.",
    )
    parser.add_argument(
        "--attention-position-contract",
        choices=["shared-input", "attention-only"],
        default="shared-input",
        help="Whether learned positions are added to the shared residual stream or only at attention call sites.",
    )
    parser.add_argument("--max-position-embeddings", type=int, default=1024)
    parser.add_argument(
        "--final-norm-kind",
        choices=["identity", "layernorm", "rmsnorm"],
        default="identity",
        help="Final hidden-state normalization before the LM head.",
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
        choices=["requirements-only", "official-mamba3", "compile-safe", "primitive-triton"],
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
    parser.add_argument(
        "--head-loss-backend",
        default="dense",
        choices=["dense", "compiled", "streaming-kernel"],
        help=(
            "Training/eval LM-head loss contract. dense preserves the explicit "
            "final_norm -> output projection -> cross-entropy path; compiled wraps "
            "that exact path in a narrow torch.compile seam; streaming-kernel is "
            "reserved for a registered native streaming LM-head CE kernel."
        ),
    )
    parser.add_argument(
        "--ffn-backend",
        default="dense",
        choices=["dense", "compiled", "manual-autograd", "triton-gelu", "recompute"],
        help=(
            "Transformer FFN runtime contract. dense preserves the explicit "
            "output_norm -> FFN -> residual-add path; compiled wraps that exact "
            "block-local path in a narrow torch.compile seam; manual-autograd "
            "uses a custom autograd.Function while still delegating GEMMs to "
            "PyTorch/cuBLAS, and is not a fused native kernel."
        ),
    )
    parser.add_argument(
        "--parcae-recurrent-compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help=(
            "Compile mode for Parcae recurrent full-block compile. This is separate "
            "from --compile-mode so primitive-triton runs can tune recurrent blocks "
            "without changing the global runtime contract."
        ),
    )
    parser.add_argument(
        "--parcae-loop-update-backend",
        default="eager",
        choices=[
            "eager",
            "lean-eager",
            "compiled",
            "manual-autograd",
            "triton-glue",
            "triton-loop-forward",
        ],
        help=(
            "Parcae loop-update runtime contract. eager keeps the promoted fast lane "
            "with separately compiled recurrent blocks; lean-eager keeps that block "
            "path but precomputes recurrent attention context and bypasses wrapper churn; "
            "compiled tries one larger torch.compile region for the recurrent loop "
            "iteration; manual-autograd keeps the recurrent block path intact while "
            "owning the scalar loop-glue backward; triton-glue uses native Triton "
            "forward kernels for the scalar loop glue; triton-loop-forward fuses the "
            "state-update and residual loop-glue forward around each recurrent block."
        ),
    )
    parser.add_argument(
        "--parcae-scaffold-backend",
        default="standard",
        choices=["standard", "compiled"],
        help="Runtime boundary for the whole Parcae scaffold path.",
    )
    parser.add_argument(
        "--parcae-band-block-contract",
        default="generic",
        choices=["generic", "compiled-direct"],
        help="Multi-band recurrent block dispatch contract: old generic forward path or direct compiled-block path.",
    )
    parser.add_argument(
        "--parcae-band-prepare-backend",
        default="standard",
        choices=["standard", "compiled"],
        help="Backend for multi-band loop projection plus control/injection preparation.",
    )
    parser.add_argument(
        "--parcae-output-mix-backend",
        default="standard",
        choices=["standard", "triton"],
        help="Backend for hourglass loop-output residual mix.",
    )
    parser.add_argument(
        "--parcae-fuse-first-state-mix",
        action="store_true",
        help="Skip the first Parcae state-mix launch when the recurrent band state is known zero.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--head-count", type=int)
    parser.add_argument("--total-layers", type=int)
    parser.add_argument("--local-window", type=int)
    parser.add_argument(
        "--attention-kernel",
        default=DEFAULT_PATH1_MODEL_SHAPE.attention_kernel.value,
        choices=[profile.value for profile in AttentionKernelProfile],
        help=(
            "Attention kernel contract. sdpa uses torch scaled_dot_product_attention; "
            "flex-local uses PyTorch FlexAttention for causal local windows when available; "
            "flash-local uses optional flash-attn local windows when installed."
        ),
    )
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
    parser.add_argument(
        "--train-loss-record-interval",
        type=int,
        default=1,
        help=(
            "Record and materialize train loss every N steps, while always recording "
            "the first and final step. Larger values avoid per-step CUDA scalar syncs "
            "and shrink long-run reports."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--optimizer-profile",
        default="adam",
        choices=["adam", "adam-fused", "adam-triton-2d", "muon-reference"],
        help=(
            "Training optimizer contract. adam preserves the historical baseline. "
            "adam-fused uses PyTorch's fused CUDA Adam kernel with the same Adam math. "
            "adam-triton-2d uses a prototype Triton Adam kernel for eligible large "
            "2D hidden matrices and fused Adam fallback for the rest. "
            "muon-reference applies torch.optim.Muon to eligible 2D hidden matrices "
            "and AdamW to embeddings, heads, biases, norms, and scalar controls."
        ),
    )
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument(
        "--muon-adjust-lr-fn",
        choices=["original", "match_rms_adamw"],
        help="Optional torch.optim.Muon learning-rate adjustment function.",
    )
    parser.add_argument("--benchmark-profile", choices=[CUDA_FAITHFUL_SMALL_V1])
    parser.add_argument("--benchmark-name")
    parser.add_argument("--corpus-format", default="jsonl-text", choices=["jsonl-text", "token-ids"])
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
            parser.error("--corpus-format token-ids may not be combined with --benchmark-profile")
        if args.jsonl_train_path is not None or args.jsonl_eval_path is not None:
            parser.error("--corpus-format token-ids may not be combined with JSONL corpus paths")
        if args.tokenized_manifest_path is None:
            parser.error("--tokenized-manifest-path is required with --corpus-format token-ids")
        corpus = TokenIdCorpusSpec(manifest_path=args.tokenized_manifest_path)
        try:
            manifest_payload = json.loads(args.tokenized_manifest_path.read_text(encoding="utf-8"))
            tokenizer_payload = manifest_payload.get("tokenizer", {})
            manifest_vocab_size = tokenizer_payload.get("vocab_size")
        except Exception as exc:
            parser.error(f"failed to read tokenized manifest for vocab-size inference: {exc}")
        if not isinstance(manifest_vocab_size, int) or manifest_vocab_size <= 0:
            parser.error("tokenized manifest tokenizer.vocab_size must be a positive integer")
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
        train_loss_record_interval=args.train_loss_record_interval,
        optimizer_profile=args.optimizer_profile,
        muon_weight_decay=args.muon_weight_decay,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
        muon_adjust_lr_fn=args.muon_adjust_lr_fn,
    )
    args.benchmark_name = benchmark_name
    return corpus, budget


def _build_shape_and_schedule(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
) -> tuple[Path1ModelShape, tuple[HybridAttentionLayerRole, ...] | None]:
    layer_schedule = parse_layer_schedule_spec(args.layer_schedule) if args.layer_schedule else None
    if layer_schedule is None:
        total_layers = (
            args.total_layers
            if args.total_layers is not None
            else DEFAULT_PATH1_MODEL_SHAPE.total_layers
        )
    else:
        if args.total_layers is not None and args.total_layers != len(layer_schedule):
            parser.error("--total-layers must match the explicit --layer-schedule length")
        total_layers = len(layer_schedule)
    shape = Path1ModelShape(
        vocab_size=(
            args.vocab_size
            if args.vocab_size is not None
            else DEFAULT_PATH1_MODEL_SHAPE.vocab_size
        ),
        d_model=args.d_model if args.d_model is not None else DEFAULT_PATH1_MODEL_SHAPE.d_model,
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
        attention_kernel=AttentionKernelProfile(args.attention_kernel),
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
            parcae_hourglass_pass_count=args.parcae_hourglass_pass_count,
            parcae_backward_steps=args.parcae_backward_steps,
            parcae_prelude_norm_kind=args.parcae_prelude_norm_kind,
            parcae_discretization=args.parcae_discretization,
            parcae_dt_raw_init=args.parcae_dt_raw_init,
            parcae_loop_d_model=args.parcae_loop_d_model,
            parcae_loop_head_count=args.parcae_loop_head_count,
            parcae_loop_ffn_multiplier=args.parcae_loop_ffn_multiplier,
            parcae_loop_layer_count=args.parcae_loop_layer_count,
            parcae_hourglass_band_schedule=(
                parse_int_tuple_spec(
                    args.parcae_hourglass_band_schedule,
                    name="parcae hourglass band schedule",
                )
                if args.parcae_hourglass_band_schedule
                else None
            ),
            parcae_control_position_kind=args.parcae_control_position_kind,
            parcae_control_position_scale_init=args.parcae_control_position_scale_init,
            parcae_control_stride=args.parcae_control_stride,
            parcae_control_state_transform=args.parcae_control_state_transform,
            parcae_recurrent_compile_mode=args.parcae_recurrent_compile_mode,
            parcae_loop_update_backend=args.parcae_loop_update_backend,
            parcae_scaffold_backend=args.parcae_scaffold_backend,
            parcae_band_block_contract=args.parcae_band_block_contract,
            parcae_band_prepare_backend=args.parcae_band_prepare_backend,
            parcae_output_mix_backend=args.parcae_output_mix_backend,
            parcae_fuse_first_state_mix=args.parcae_fuse_first_state_mix,
            attention_position_contract=args.attention_position_contract,
            position_encoding_kind=args.position_encoding_kind,
            max_position_embeddings=args.max_position_embeddings,
            final_norm_kind=args.final_norm_kind,
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
                parse_reference_ssm_profile_schedule_spec(args.reference_ssm_profile_schedule)
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
            attention_position_contract=args.attention_position_contract,
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
        state_transform_mode=PrimitiveStateTransformMode(args.primitive_state_transform_profile),
        layer_schedule=layer_schedule,
        feed_forward_profile=FeedForwardProfile(args.feed_forward_profile),
        feed_forward_layer_indices=feed_forward_layer_indices,
        eml_slot_count=args.eml_slot_count,
        eml_tree_depth=args.eml_tree_depth,
        eml_route_fraction=args.eml_route_fraction,
        attention_position_contract=args.attention_position_contract,
    )


def _implementation_kind_for_variant(variant, *, primitive_runtime_backend: str) -> str:
    if variant.kind is Path1VariantKind.ATTENTION_ONLY:
        attention_kernel = variant.shape.attention_kernel.value.replace("-", "_")
        scaffold = variant.scaffold_profile.value.replace("-", "_")
        scaffold_suffix = "" if variant.scaffold_profile is Path1ScaffoldProfile.STANDARD else f"_{scaffold}"
        if variant.feed_forward_profile is not FeedForwardProfile.STANDARD:
            ffn = variant.feed_forward_profile.value.replace("-", "_")
            return f"python_attention_{attention_kernel}{scaffold_suffix}_{ffn}"
        return f"python_attention_{attention_kernel}{scaffold_suffix}"
    if variant.kind is Path1VariantKind.REFERENCE_SSM_HYBRID:
        profiles = variant.reference_ssm_profile_schedule or (variant.reference_ssm_profile,)
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
            law = variant.reference_ssm_profile.fla_gdnp_compatible_law.replace("-", "_")
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
            head_loss_backend=args.head_loss_backend,
            ffn_backend=args.ffn_backend,
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
