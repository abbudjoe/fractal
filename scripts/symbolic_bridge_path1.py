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

from python.symbolic.bridge_lm import format_optional  # noqa: E402
from python.symbolic.bridge_path1 import run_symbolic_bridge_path1  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Path1 LM-side symbolic expert bridge contract.")
    parser.add_argument("--bridge-summary", type=Path, required=True)
    parser.add_argument("--run-label", default=f"symbolic-bridge-path1-{int(time.time())}")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--total-layers", type=int, default=4)
    parser.add_argument("--head-count", type=int, default=4)
    parser.add_argument("--ffn-multiplier", type=int, default=2)
    parser.add_argument("--router-loss-weight", type=float, default=10.0)
    parser.add_argument("--abstain-class-weight", type=float, default=1.0)
    parser.add_argument("--unsafe-call-loss-weight", type=float, default=0.0)
    parser.add_argument("--call-abstain-loss-weight", type=float, default=5.0)
    parser.add_argument("--unsafe-margin-loss-weight", type=float, default=0.0)
    parser.add_argument("--unsafe-margin", type=float, default=0.5)
    parser.add_argument("--router-call-threshold", type=float, default=0.99999)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--output", choices=["table", "json"], default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or (REPO_ROOT / "artifacts" / "symbolic-bridge-path1" / args.run_label)
    report = run_symbolic_bridge_path1(
        args.bridge_summary,
        output_dir,
        run_label=args.run_label,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        total_layers=args.total_layers,
        head_count=args.head_count,
        ffn_multiplier=args.ffn_multiplier,
        router_loss_weight=args.router_loss_weight,
        abstain_class_weight=args.abstain_class_weight,
        unsafe_call_loss_weight=args.unsafe_call_loss_weight,
        call_abstain_loss_weight=args.call_abstain_loss_weight,
        unsafe_margin_loss_weight=args.unsafe_margin_loss_weight,
        unsafe_margin=args.unsafe_margin,
        router_call_threshold=args.router_call_threshold,
        device=args.device,
    )
    if args.output == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            "run\tmode\tfeatures\tval_final_acc\textrap_final_acc\textrap_lm_acc\t"
            "router_extrap_acc\texpert_call_rate\tunsafe_call_rate\tabstain_recall"
        )
        for run in report.runs:
            print(
                f"{run.name}\t"
                f"{run.mode}\t"
                f"{run.feature_set}\t"
                f"{run.validation_final_token_accuracy:.3f}\t"
                f"{run.extrapolation_final_token_accuracy:.3f}\t"
                f"{run.extrapolation_lm_token_accuracy:.3f}\t"
                f"{format_optional(run.extrapolation_router_accuracy)}\t"
                f"{run.extrapolation_expert_call_rate:.3f}\t"
                f"{run.extrapolation_unsafe_call_rate:.3f}\t"
                f"{format_optional(run.extrapolation_abstain_recall)}"
            )
        print(f"path1_variant={report.summary['path1_variant']}")
        print(f"router_gain_vs_side_channel={report.summary['router_contract_extrapolation_gain_vs_side_channel']:.3f}")
        print(f"router_unsafe_call_rate={report.summary['router_contract_unsafe_call_rate']:.3f}")
        print(f"router_abstain_recall={format_optional(report.summary['router_contract_abstain_recall'])}")
        print(f"capability_gain_confirmed={report.summary['capability_gain_confirmed']}")
        print(f"safe_abstention_confirmed={report.summary['safe_abstention_confirmed']}")
        print(f"contract_confirmed={report.summary['contract_confirmed']}")
        print(f"failure_modes={report.summary['failure_modes']}")
        print(f"summary_path={output_dir / 'summary.json'}")
        print(f"markdown_path={output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
