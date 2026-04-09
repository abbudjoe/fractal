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
    ValidationError,
)
from python.specs.path1 import (
    Path1VariantKind,
    PrimitiveExecutionProfile,
    PrimitiveNormMode,
    PrimitiveProfile,
    PrimitiveReadoutMode,
    PrimitiveResidualMode,
    PrimitiveStateTransformMode,
    PrimitiveWrapperMode,
    ReferenceSsmProfile,
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
    parser.add_argument("--backend", default="cuda", choices=["cpu", "cuda"])
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int)
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
    parser.add_argument("--jsonl-train-path", type=Path)
    parser.add_argument("--jsonl-eval-path", type=Path)
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
) -> tuple[JsonlCorpusSpec, BenchmarkBudgetSpec]:
    benchmark_name = args.benchmark_name
    if args.benchmark_profile == CUDA_FAITHFUL_SMALL_V1:
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
    else:
        if args.jsonl_train_path is None or args.jsonl_eval_path is None:
            parser.error(
                "either --benchmark-profile or both --jsonl-train-path and --jsonl-eval-path are required"
            )
        train_path = args.jsonl_train_path
        eval_path = args.jsonl_eval_path
        corpus_name = args.corpus_name or "jsonl-corpus"
        full_train_pass = args.full_train_pass
        full_eval_pass = args.full_eval_pass

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
    corpus = JsonlCorpusSpec(
        train_path=train_path,
        eval_path=eval_path,
        corpus_name=corpus_name,
        text_field=args.corpus_text_field,
    )
    args.benchmark_name = benchmark_name
    return corpus, budget


def _build_variant(args: argparse.Namespace):
    kind = Path1VariantKind(args.variant)
    if kind is Path1VariantKind.ATTENTION_ONLY:
        return phase1_attention_only_variant()
    if kind is Path1VariantKind.REFERENCE_SSM_HYBRID:
        return phase1_reference_ssm_variant(
            profile=ReferenceSsmProfile(args.reference_ssm_profile)
        )
    return phase1_primitive_variant(
        primitive_profile=PrimitiveProfile(args.primitive_profile),
        execution_profile=PrimitiveExecutionProfile(args.primitive_execution_profile),
        residual_mode=PrimitiveResidualMode(args.primitive_residual_profile),
        readout_mode=PrimitiveReadoutMode(args.primitive_readout_profile),
        norm_mode=PrimitiveNormMode(args.primitive_norm_profile),
        wrapper_mode=PrimitiveWrapperMode(args.primitive_wrapper_profile),
        state_transform_mode=PrimitiveStateTransformMode(args.primitive_state_transform_profile),
    )


def _implementation_kind_for_variant(variant, *, primitive_runtime_backend: str) -> str:
    if variant.kind is Path1VariantKind.ATTENTION_ONLY:
        return "python_attention_sdpa"
    if variant.kind is Path1VariantKind.REFERENCE_SSM_HYBRID:
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
    variant = _build_variant(args)
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
