#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.runners.mini_moe_autoresearch import (  # noqa: E402
    bitmask_to_mask,
    bitmask_to_key,
    top_selective_mask_ids_from_state,
)
from python.runners.mini_moe_policy_search import (  # noqa: E402
    MiniMoePolicySearchCandidate,
    MiniMoePolicySearchRequest,
    run_mini_moe_policy_search,
)
from python.specs.common import (  # noqa: E402
    BenchmarkBudgetSpec,
    BenchmarkRunManifest,
    DeviceRuntimeSpec,
    JsonlCorpusSpec,
    SeedSpec,
)
from python.specs.mini_moe import MiniMoeSurfaceSpec  # noqa: E402
from python.specs.mini_moe import (  # noqa: E402
    MiniMoeDispatchExecutionStrategy,
    MiniMoeRuntimeSpec,
    RecurrentRoundExecutionStrategy,
)


DEFAULT_NOTE = (
    "Mini-MoE shortlist replay on the shared Python research substrate for CUDA/MPS "
    "translation of autoresearch winners"
)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(token.strip()) for token in value.split(",") if token.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay selected mini-MoE recurrent masks across seeds on a typed runner surface."
    )
    parser.add_argument("--backend", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
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
    parser.add_argument("--jsonl-train-path", type=Path, required=True)
    parser.add_argument("--jsonl-eval-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--benchmark-name", default="dreegmor-mini-moe-shortlist")
    parser.add_argument("--corpus-name", default="fineweb-stage0-canary")
    parser.add_argument("--corpus-text-field", default="text")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--full-train-pass", action="store_true")
    parser.add_argument("--full-eval-pass", action="store_true")
    parser.add_argument("--warmup-eval-batches", type=int, default=1)
    parser.add_argument("--warmup-train-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--total-layers", type=int, default=16)
    parser.add_argument("--experts-per-block", type=int, default=8)
    parser.add_argument("--entropy-threshold", type=float, default=0.95)
    parser.add_argument(
        "--dispatch-execution-strategy",
        choices=["dense_gather", "token_packed_sparse"],
        default="dense_gather",
    )
    parser.add_argument(
        "--round2-execution-strategy",
        choices=["dense_blend", "masked_token_update"],
        default="dense_blend",
    )
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--round2-mask",
        action="append",
        default=[],
        help="Repeatable comma-separated layer mask, e.g. 2,5,6,9,10,11,12",
    )
    parser.add_argument("--include-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-all-layers", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _manifest(args: argparse.Namespace) -> BenchmarkRunManifest:
    return BenchmarkRunManifest(
        run_label=args.run_label,
        implementation_kind="python_native",
        benchmark_name=args.benchmark_name,
        note=DEFAULT_NOTE,
        seed_spec=SeedSpec(model_seed=0),
        corpus=JsonlCorpusSpec(
            train_path=args.jsonl_train_path,
            eval_path=args.jsonl_eval_path,
            corpus_name=args.corpus_name,
            text_field=args.corpus_text_field,
        ),
        budget=BenchmarkBudgetSpec(
            seq_len=args.seq_len,
            window_stride=args.window_stride,
            batch_size=args.batch_size,
            train_steps=args.steps,
            eval_batches=args.eval_batches,
            full_train_pass=args.full_train_pass,
            full_eval_pass=args.full_eval_pass,
            learning_rate=args.learning_rate,
            warmup_eval_batches=args.warmup_eval_batches,
            warmup_train_steps=args.warmup_train_steps,
        ),
        runtime=DeviceRuntimeSpec(
            backend=args.backend,
            cuda_device=args.cuda_device,
            dtype=args.dtype,
            env_kind=args.env_kind,
            compile_mode=args.compile_mode,
            primitive_runtime_backend=args.primitive_runtime_backend,
        ),
    )


def _configured_surface(
    surface: MiniMoeSurfaceSpec,
    *,
    total_layers: int,
    experts_per_block: int,
    dispatch_execution_strategy: MiniMoeDispatchExecutionStrategy,
    round2_execution_strategy: RecurrentRoundExecutionStrategy,
) -> MiniMoeSurfaceSpec:
    architecture = replace(
        surface.architecture,
        label=f"{surface.architecture.label}-e{experts_per_block}-d{total_layers}",
        backbone=replace(surface.architecture.backbone, total_layers=total_layers),
        moe=replace(surface.architecture.moe, experts_per_block=experts_per_block),
    )
    if architecture.router.recurrent_pre_expert is not None:
        architecture = replace(
            architecture,
            router=replace(
                architecture.router,
                recurrent_pre_expert=replace(
                    architecture.router.recurrent_pre_expert,
                    execution_strategy=round2_execution_strategy,
                ),
            ),
        )
    runtime = MiniMoeRuntimeSpec(
        dispatch=replace(
            surface.runtime.dispatch,
            execution_strategy=dispatch_execution_strategy,
        )
    )
    observability = replace(surface.observability, max_token_route_traces_per_layer=0)
    return replace(surface, architecture=architecture, runtime=runtime, observability=observability)


def _selected_masks(args: argparse.Namespace) -> tuple[tuple[int, ...], ...]:
    explicit_masks = tuple(_parse_int_tuple(mask) for mask in args.round2_mask)
    if explicit_masks:
        return explicit_masks
    if args.state_path is None:
        raise ValueError("either --round2-mask or --state-path must be provided")
    top_mask_ids = top_selective_mask_ids_from_state(
        args.state_path,
        total_layers=args.total_layers,
        limit=args.top_k,
    )
    return tuple(bitmask_to_mask(mask_id, args.total_layers) for mask_id in top_mask_ids)


def _candidates(args: argparse.Namespace) -> tuple[MiniMoePolicySearchCandidate, ...]:
    candidates: list[MiniMoePolicySearchCandidate] = []
    dispatch_execution_strategy = MiniMoeDispatchExecutionStrategy(args.dispatch_execution_strategy)
    round2_execution_strategy = RecurrentRoundExecutionStrategy(args.round2_execution_strategy)
    if args.include_reference:
        candidates.append(
            MiniMoePolicySearchCandidate(
                name="reference",
                surface=_configured_surface(
                    MiniMoeSurfaceSpec.phase1_reference_default(),
                    total_layers=args.total_layers,
                    experts_per_block=args.experts_per_block,
                    dispatch_execution_strategy=dispatch_execution_strategy,
                    round2_execution_strategy=round2_execution_strategy,
                ),
                note="One-shot standard MoE CUDA replay baseline",
            )
        )
    if args.include_all_layers:
        candidates.append(
            MiniMoePolicySearchCandidate(
                name="entropy_all_layers",
                surface=_configured_surface(
                    MiniMoeSurfaceSpec.phase1_recurrent_entropy_gated_default(
                        normalized_entropy_threshold=args.entropy_threshold
                    ),
                    total_layers=args.total_layers,
                    experts_per_block=args.experts_per_block,
                    dispatch_execution_strategy=dispatch_execution_strategy,
                    round2_execution_strategy=round2_execution_strategy,
                ),
                note="All-layer recurrent entropy-gated CUDA replay baseline",
            )
        )
    for mask in _selected_masks(args):
        mask_key = bitmask_to_key(sum(1 << layer for layer in mask), args.total_layers)
        candidates.append(
            MiniMoePolicySearchCandidate(
                name=f"entropy_{mask_key}",
                surface=_configured_surface(
                    MiniMoeSurfaceSpec.phase1_recurrent_entropy_gated_default(
                        normalized_entropy_threshold=args.entropy_threshold,
                        round2_layer_indices=mask,
                    ),
                    total_layers=args.total_layers,
                    experts_per_block=args.experts_per_block,
                    dispatch_execution_strategy=dispatch_execution_strategy,
                    round2_execution_strategy=round2_execution_strategy,
                ),
                note=f"Selective recurrent entropy-gated CUDA replay mask {mask}",
            )
        )
    return tuple(candidates)


def main() -> int:
    args = parse_args()
    summary = run_mini_moe_policy_search(
        MiniMoePolicySearchRequest(
            benchmark_name=args.benchmark_name,
            manifest_template=_manifest(args),
            candidates=_candidates(args),
            seeds=_parse_int_tuple(args.seeds),
            output_dir=args.output_dir,
            output_format="json",
            ledger_path=args.ledger_path,
        )
    )
    payload = {
        "benchmark_name": summary.benchmark_name,
        "summary_path": summary.summary_path,
        "seeds": list(summary.seeds),
        "candidate_results": [
            {
                "candidate_name": result.candidate_name,
                "report_paths": list(result.report_paths),
                "avg_final_loss": result.avg_final_loss,
                "avg_train_toks_per_s": result.avg_train_toks_per_s,
                "avg_overall_toks_per_s": result.avg_overall_toks_per_s,
                "avg_peak_process_memory_delta_mb": result.avg_peak_process_memory_delta_mb,
                "avg_overall_round2_fraction": result.avg_overall_round2_fraction,
                "avg_mean_active_round2_fraction": result.avg_mean_active_round2_fraction,
            }
            for result in summary.candidate_results
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
