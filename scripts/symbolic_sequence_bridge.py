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

from python.symbolic.bridge_sequence import run_sequence_bridge  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny sequence bridge over symbolic expert features.")
    parser.add_argument("--bridge-summary", type=Path, required=True)
    parser.add_argument("--run-label", default=f"symbolic-sequence-bridge-{int(time.time())}")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--output", choices=["table", "json"], default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or (REPO_ROOT / "artifacts" / "symbolic-sequence-bridge" / args.run_label)
    report = run_sequence_bridge(
        args.bridge_summary,
        output_dir,
        run_label=args.run_label,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        device=args.device,
    )
    if args.output == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            "run\tobjective\tfeatures\tval_token_acc\textrap_token_acc\t"
            "val_loss\textrap_loss\textrap_rmse\trouter_val_acc\trouter_extrap_acc"
        )
        for run in report.runs:
            print(
                f"{run.name}\t"
                f"{run.objective}\t"
                f"{run.feature_set}\t"
                f"{run.validation_token_accuracy:.3f}\t"
                f"{run.extrapolation_token_accuracy:.3f}\t"
                f"{run.validation_loss:.4g}\t"
                f"{run.extrapolation_loss:.4g}\t"
                f"{format_optional(run.extrapolation_rmse)}\t"
                f"{format_optional(run.validation_router_accuracy)}\t"
                f"{format_optional(run.extrapolation_router_accuracy)}"
            )
        safe = report.summary["safe_expert_coverage"]
        print(f"best_validation_token_accuracy={report.summary['best_validation_token_accuracy']}")
        print(f"best_extrapolation_token_accuracy={report.summary['best_extrapolation_token_accuracy']}")
        print(f"best_trained_extrapolation_token_accuracy={report.summary['best_trained_extrapolation_token_accuracy']}")
        print(f"best_trained_router_extrapolation_accuracy={report.summary['best_trained_router_extrapolation_accuracy']}")
        print(f"continuous_side_channel_delta={report.summary['continuous_side_channel_extrapolation_delta']:.3f}")
        print(f"router_expert_signal_delta={report.summary['router_expert_signal_extrapolation_delta']:.3f}")
        print(
            "safe_expert_coverage="
            f"train={safe['train']:.3f}"
            f"\tvalidation={safe['validation']:.3f}"
            f"\textrapolation={safe['extrapolation']:.3f}"
        )
        print(f"summary_path={output_dir / 'summary.json'}")
        print(f"markdown_path={output_dir / 'summary.md'}")
    return 0


def format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
