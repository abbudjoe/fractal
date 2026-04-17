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

from python.symbolic.bridge import run_symbolic_bridge  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tokenized symbolic-to-LM bridge probe.")
    parser.add_argument("--symbolic-summary", type=Path, required=True)
    parser.add_argument("--run-label", default=f"symbolic-bridge-{int(time.time())}")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--token-bins", type=int, default=32)
    parser.add_argument("--output", choices=["table", "json"], default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or (REPO_ROOT / "artifacts" / "symbolic-bridge" / args.run_label)
    report = run_symbolic_bridge(
        args.symbolic_summary,
        output_dir,
        run_label=args.run_label,
        token_bins=args.token_bins,
    )
    if args.output == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print("model\tval_token_acc\textrap_token_acc\tval_rmse\textrap_rmse\tsource_exact")
        for model, stats in report.summary["model_families"].items():
            print(
                f"{model}\t"
                f"{stats['median_validation_token_accuracy']:.3f}\t"
                f"{stats['median_extrapolation_token_accuracy']:.3f}\t"
                f"{stats['median_validation_continuous_rmse']:.4g}\t"
                f"{stats['median_extrapolation_continuous_rmse']:.4g}\t"
                f"{stats['source_exact_recovery_rate']:.2f}"
            )
        print(f"best_validation_token_accuracy={report.summary['best_validation_token_accuracy']}")
        print(f"best_extrapolation_token_accuracy={report.summary['best_extrapolation_token_accuracy']}")
        feature_table = report.summary["feature_table"]
        print(
            "feature_table="
            f"path={feature_table['path']}"
            f"\trows={feature_table['row_count']}"
            f"\tsafe_expert_coverage={feature_table['safe_expert_coverage']:.3f}"
        )
        print(f"feature_split_counts={feature_table['split_counts']}")
        print(f"feature_split_safe_expert_coverage={feature_table['split_safe_expert_coverage']}")
        side = report.summary["frozen_side_channel_probe"]
        print(f"best_side_channel_extrapolation={side['best_side_channel_extrapolation']}")
        router = report.summary["router_target_probe"]
        print(
            "task_router="
            f"val_acc={router['median_validation_token_accuracy']:.3f}"
            f"\textrap_acc={router['median_extrapolation_token_accuracy']:.3f}"
            f"\tmean_extrap_acc={router['mean_extrapolation_token_accuracy']:.3f}"
            f"\toracle_extrap_acc={router['oracle_extrapolation_token_accuracy']:.3f}"
            f"\toracle_mean_extrap_acc={router['oracle_mean_extrapolation_token_accuracy']:.3f}"
        )
        print(f"summary_path={output_dir / 'summary.json'}")
        print(f"markdown_path={output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
