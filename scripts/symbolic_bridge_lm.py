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

from python.symbolic.bridge_lm import run_symbolic_bridge_lm  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Confirm the symbolic router contract on a tiny LM-style model.")
    parser.add_argument("--bridge-summary", type=Path, required=True)
    parser.add_argument(
        "--extra-fit-bridge-summary",
        type=Path,
        action="append",
        default=[],
        help="Additional bridge corpus summary whose train/safety_calibration rows are used only for fitting.",
    )
    parser.add_argument("--run-label", default=f"symbolic-bridge-lm-{int(time.time())}")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--router-loss-weight", type=float, default=0.5)
    parser.add_argument("--abstain-class-weight", type=float, default=1.0)
    parser.add_argument("--unsafe-call-loss-weight", type=float, default=0.0)
    parser.add_argument("--call-abstain-loss-weight", type=float, default=0.0)
    parser.add_argument("--answer-call-abstain-loss-weight", type=float, default=0.0)
    parser.add_argument("--answer-unsafe-loss-weight", type=float, default=0.0)
    parser.add_argument("--non-answer-abstain-loss-weight", type=float, default=0.0)
    parser.add_argument("--unsafe-margin-loss-weight", type=float, default=0.0)
    parser.add_argument("--unsafe-margin", type=float, default=0.5)
    parser.add_argument("--router-call-threshold", type=float, default=0.0)
    parser.add_argument("--expert-logit-scale", type=float, default=6.0)
    parser.add_argument("--backbone", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn-multiplier", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output", choices=["table", "json"], default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or (REPO_ROOT / "artifacts" / "symbolic-bridge-lm" / args.run_label)
    report = run_symbolic_bridge_lm(
        args.bridge_summary,
        output_dir,
        run_label=args.run_label,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        router_loss_weight=args.router_loss_weight,
        abstain_class_weight=args.abstain_class_weight,
        unsafe_call_loss_weight=args.unsafe_call_loss_weight,
        call_abstain_loss_weight=args.call_abstain_loss_weight,
        answer_call_abstain_loss_weight=args.answer_call_abstain_loss_weight,
        answer_unsafe_loss_weight=args.answer_unsafe_loss_weight,
        non_answer_abstain_loss_weight=args.non_answer_abstain_loss_weight,
        unsafe_margin_loss_weight=args.unsafe_margin_loss_weight,
        unsafe_margin=args.unsafe_margin,
        router_call_threshold=args.router_call_threshold,
        expert_logit_scale=args.expert_logit_scale,
        backbone=args.backbone,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ffn_multiplier=args.transformer_ffn_multiplier,
        device=args.device,
        extra_fit_bridge_summary_paths=tuple(args.extra_fit_bridge_summary),
    )
    if args.output == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            f"backbone={report.backbone}\tconfig={report.backbone_config}"
        )
        print(
            "run\tmode\tfeatures\tval_final_acc\textrap_final_acc\tval_final_nll\textrap_final_nll\textrap_lm_acc\t"
            "router_extrap_acc\texpert_call_rate\tunsafe_call_rate\tabstain_recall"
        )
        for run in report.runs:
            print(
                f"{run.name}\t"
                f"{run.mode}\t"
                f"{run.feature_set}\t"
                f"{run.validation_final_token_accuracy:.3f}\t"
                f"{run.extrapolation_final_token_accuracy:.3f}\t"
                f"{run.validation_final_nll:.4g}\t"
                f"{run.extrapolation_final_nll:.4g}\t"
                f"{run.extrapolation_lm_token_accuracy:.3f}\t"
                f"{format_optional(run.extrapolation_router_accuracy)}\t"
                f"{run.extrapolation_expert_call_rate:.3f}\t"
                f"{run.extrapolation_unsafe_call_rate:.3f}\t"
                f"{format_optional(run.extrapolation_abstain_recall)}"
            )
        safe = report.summary["safe_expert_coverage"]
        print(f"extra_fit_bridge_summaries={report.extra_fit_bridge_summary_paths}")
        print(f"extra_fit_feature_tables={report.extra_fit_feature_table_paths}")
        print(f"fit_row_count={report.summary['fit_row_count']}")
        print(f"extra_fit_row_count={report.summary['extra_fit_row_count']}")
        print(f"best_validation_final_token_accuracy={report.summary['best_validation_final_token_accuracy']}")
        print(f"best_extrapolation_final_token_accuracy={report.summary['best_extrapolation_final_token_accuracy']}")
        print(f"best_trained_extrapolation_final_token_accuracy={report.summary['best_trained_extrapolation_final_token_accuracy']}")
        print(f"best_validation_final_nll={report.summary['best_validation_final_nll']}")
        print(f"best_extrapolation_final_nll={report.summary['best_extrapolation_final_nll']}")
        print(f"best_trained_extrapolation_final_nll={report.summary['best_trained_extrapolation_final_nll']}")
        print(f"logit_fusion_nll_gain_vs_side_channel={report.summary['logit_fusion_extrapolation_nll_delta_vs_side_channel']:.4g}")
        print(f"prob_mixture_nll_gain_vs_side_channel={report.summary['prob_mixture_extrapolation_nll_delta_vs_side_channel']:.4g}")
        print(f"logit_fusion_nll_gain_vs_token_only={report.summary['logit_fusion_extrapolation_nll_delta_vs_token_only']:.4g}")
        print(f"prob_mixture_nll_gain_vs_token_only={report.summary['prob_mixture_extrapolation_nll_delta_vs_token_only']:.4g}")
        print(f"logit_fusion_acc_gain_vs_side_channel={report.summary['logit_fusion_extrapolation_accuracy_delta_vs_side_channel']:.3f}")
        print(f"prob_mixture_acc_gain_vs_side_channel={report.summary['prob_mixture_extrapolation_accuracy_delta_vs_side_channel']:.3f}")
        print(f"logit_fusion_acc_gain_vs_token_only={report.summary['logit_fusion_extrapolation_accuracy_delta_vs_token_only']:.3f}")
        print(f"prob_mixture_acc_gain_vs_token_only={report.summary['prob_mixture_extrapolation_accuracy_delta_vs_token_only']:.3f}")
        print(f"router_gain_vs_token_only={report.summary['router_contract_extrapolation_gain_vs_token_only']:.3f}")
        print(f"router_gain_vs_x_task={report.summary['router_contract_extrapolation_gain_vs_x_task']:.3f}")
        print(f"router_gain_vs_side_channel={report.summary['router_contract_extrapolation_gain_vs_side_channel']:.3f}")
        print(f"router_gap_to_oracle={report.summary['router_contract_gap_to_oracle']:.3f}")
        print(f"router_unsafe_call_rate={report.summary['router_contract_unsafe_call_rate']:.3f}")
        print(f"router_abstain_recall={format_optional(report.summary['router_contract_abstain_recall'])}")
        print(f"capability_gain_confirmed={report.summary['capability_gain_confirmed']}")
        print(f"safe_abstention_confirmed={report.summary['safe_abstention_confirmed']}")
        print(f"contract_confirmed={report.summary['contract_confirmed']}")
        print(
            "prob_mixture_math_answer_accuracy_delta_vs_controls="
            f"{report.summary['prob_mixture_math_answer_accuracy_delta_vs_controls']:.3f}"
        )
        print(
            "prob_mixture_math_answer_unsafe_call_rate="
            f"{report.summary['prob_mixture_math_answer_unsafe_call_rate']:.3f}"
        )
        print(
            "prob_mixture_math_answer_contract_confirmed="
            f"{report.summary['prob_mixture_math_answer_contract_confirmed']}"
        )
        print(
            "safe_expert_coverage="
            f"train={safe['train']:.3f}"
            f"\tsafety_calibration={format_optional(safe.get('safety_calibration'))}"
            f"\tvalidation={safe['validation']:.3f}"
            f"\textrapolation={safe['extrapolation']:.3f}"
        )
        abstain_rate = report.summary["abstain_target_rate"]
        print(
            "abstain_target_rate="
            f"train={abstain_rate['train']:.3f}"
            f"\tsafety_calibration={format_optional(abstain_rate.get('safety_calibration'))}"
            f"\tvalidation={abstain_rate['validation']:.3f}"
            f"\textrapolation={abstain_rate['extrapolation']:.3f}"
        )
        print(f"fit_abstain_target_rate={report.summary['fit_abstain_target_rate']:.3f}")
        print(f"failure_modes={report.summary['failure_modes']}")
        print(f"summary_path={output_dir / 'summary.json'}")
        print(f"markdown_path={output_dir / 'summary.md'}")
    return 0


def format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
