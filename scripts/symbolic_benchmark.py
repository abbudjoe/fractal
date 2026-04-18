#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.specs.symbolic import (
    SymbolicBenchmarkManifest,
    SymbolicDatasetSpec,
    SymbolicModelFamily,
    SymbolicPreset,
    SymbolicTrainSpec,
    SymbolicTreeOptimizer,
    preset_manifest,
)
from python.symbolic.runner import run_symbolic_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-aligned symbolic EML benchmark.")
    parser.add_argument("--preset", choices=[preset.value for preset in SymbolicPreset], default=SymbolicPreset.COMPACT.value)
    parser.add_argument("--run-label", default=f"symbolic-{int(time.time())}")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model",
        action="append",
        choices=["all"] + [family.value for family in SymbolicModelFamily],
        default=None,
        help="Model family to run. Repeat for multiple families; default is all.",
    )
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--backend", choices=["cpu", "cuda", "mps", "auto"], default=None)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--extrapolation-samples", type=int)
    parser.add_argument("--tasks-per-depth", type=int)
    parser.add_argument("--tree-learning-rate", type=float)
    parser.add_argument("--mlp-learning-rate", type=float)
    parser.add_argument("--paper-restarts", type=int)
    parser.add_argument(
        "--tree-optimizer",
        choices=[optimizer.value for optimizer in SymbolicTreeOptimizer],
        default=None,
    )
    parser.add_argument("--output", choices=["table", "json"], default="table")
    return parser.parse_args()


def model_families(args: argparse.Namespace) -> tuple[SymbolicModelFamily, ...]:
    requested = args.model or ["all"]
    if "all" in requested:
        return tuple(SymbolicModelFamily)
    return tuple(SymbolicModelFamily(value) for value in requested)


def build_manifest(args: argparse.Namespace) -> SymbolicBenchmarkManifest:
    preset = SymbolicPreset(args.preset)
    seeds = tuple(args.seed_start + index for index in range(args.seeds))
    manifest = preset_manifest(
        preset=preset,
        run_label=args.run_label,
        seeds=seeds,
        model_families=model_families(args),
    )
    dataset = manifest.dataset
    train = manifest.train
    if any(
        value is not None
        for value in (
            args.train_samples,
            args.validation_samples,
            args.extrapolation_samples,
            args.tasks_per_depth,
        )
    ):
        dataset = SymbolicDatasetSpec(
            train_samples=args.train_samples or dataset.train_samples,
            validation_samples=args.validation_samples or dataset.validation_samples,
            extrapolation_samples=args.extrapolation_samples or dataset.extrapolation_samples,
            tasks_per_depth=args.tasks_per_depth or dataset.tasks_per_depth,
        )
    if any(
        value is not None
        for value in (
            args.steps,
            args.tree_learning_rate,
            args.mlp_learning_rate,
            args.tree_optimizer,
            args.paper_restarts,
        )
    ):
        train = SymbolicTrainSpec(
            steps=args.steps or train.steps,
            tree_learning_rate=args.tree_learning_rate or train.tree_learning_rate,
            mlp_learning_rate=args.mlp_learning_rate or train.mlp_learning_rate,
            spsa_perturbation=train.spsa_perturbation,
            initial_temperature=train.initial_temperature,
            final_temperature=train.final_temperature,
            snap_penalty_weight=train.snap_penalty_weight,
            hardening_tolerance_multiplier=train.hardening_tolerance_multiplier,
            hidden_units=train.hidden_units,
            tree_optimizer=(
                SymbolicTreeOptimizer(args.tree_optimizer)
                if args.tree_optimizer is not None
                else train.tree_optimizer
            ),
            paper_restarts=args.paper_restarts or train.paper_restarts,
        )
    return SymbolicBenchmarkManifest(
        run_label=manifest.run_label,
        preset=manifest.preset,
        model_families=manifest.model_families,
        seeds=manifest.seeds,
        dataset=dataset,
        train=train,
        implementation_kind=manifest.implementation_kind,
        backend=args.backend or manifest.backend,
        note=manifest.note,
    )


def print_table(summary: dict[str, object]) -> None:
    print(
        "model\tsoft_val_rmse\thard_val_rmse\thard_extrap_rmse\texact\tnear_exact\tharden\texport\tcompile\tlatency_us"
    )
    model_rows = summary["model_families"]
    assert isinstance(model_rows, dict)
    for model_family, row in model_rows.items():
        assert isinstance(row, dict)
        print(
            f"{model_family}"
            f"\t{fmt(row['median_soft_validation_rmse'])}"
            f"\t{fmt(row['median_hardened_validation_rmse'])}"
            f"\t{fmt(row['median_hardened_extrapolation_rmse'])}"
            f"\t{row['exact_recovery_rate']:.2f}"
            f"\t{row['near_exact_recovery_rate']:.2f}"
            f"\t{row['hardening_success_rate']:.2f}"
            f"\t{row['export_success_rate']:.2f}"
            f"\t{row['compile_success_rate']:.2f}"
            f"\t{fmt(row['median_compiled_latency_us_per_sample'])}"
        )
    print(f"best_soft_fit={summary['best_soft_fit']}")
    print(f"approximation_leader={summary['approximation_leader']}")
    print(f"symbolic_recovery_leader={summary['symbolic_recovery_leader']}")
    print(f"best_hardening={summary['best_hardening']}")
    print(f"best_hardened_extrapolation={summary['best_hardened_extrapolation']}")


def fmt(value: object) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if abs(number) >= 1000.0 or (0.0 < abs(number) < 0.001):
        return f"{number:.3e}"
    return f"{number:.4f}"


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    output_dir = args.output_dir or (REPO_ROOT / "artifacts" / "symbolic-benchmark" / manifest.run_label)
    report = run_symbolic_benchmark(manifest, output_dir)
    if args.output == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print_table(report.summary)
        print(f"summary_path={output_dir / 'summary.json'}")
        print(f"markdown_path={output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
