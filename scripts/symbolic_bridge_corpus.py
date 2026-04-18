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

from python.symbolic.bridge_corpus import run_bridge_corpus  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build bridge-corpus-v1 feature tables.")
    parser.add_argument(
        "--corpus-kind",
        choices=[
            "pure-language",
            "language-math",
            "language-math-heldout-templates",
            "language-math-heldout-variance",
            "math-only",
            "expert-ablation",
            "expert-shuffle",
            "target-randomized",
            "wrong-expert",
        ],
        required=True,
    )
    parser.add_argument("--source-bridge-summary", type=Path)
    parser.add_argument("--source-corpus-summary", type=Path)
    parser.add_argument("--experts", default="", help="Comma-separated expert ids for ablation/shuffle corpora.")
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--run-label", default=f"bridge-corpus-{int(time.time())}")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pure-sequences-per-split", type=int, default=160)
    parser.add_argument("--language-train-per-group", type=int, default=12)
    parser.add_argument("--language-safety-per-group", type=int, default=16)
    parser.add_argument("--language-eval-per-group", type=int, default=20)
    parser.add_argument("--output", choices=["table", "json"], default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or (REPO_ROOT / "artifacts" / "bridge-corpus-v1" / args.run_label)
    expert_subset = tuple(expert.strip() for expert in args.experts.split(",") if expert.strip()) or None
    report = run_bridge_corpus(
        output_dir,
        run_label=args.run_label,
        corpus_kind=args.corpus_kind,
        source_bridge_summary_path=args.source_bridge_summary,
        source_corpus_summary_path=args.source_corpus_summary,
        expert_subset=expert_subset,
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        pure_sequences_per_split=args.pure_sequences_per_split,
        language_train_per_group=args.language_train_per_group,
        language_safety_per_group=args.language_safety_per_group,
        language_eval_per_group=args.language_eval_per_group,
    )
    if args.output == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        feature_table = report.summary["feature_table"]
        print(f"run_label={report.run_label}")
        print(f"corpus_kind={report.corpus_kind}")
        print(f"token_bins={report.token_bins}")
        print(f"feature_table={report.feature_table_path}")
        print(f"row_count={feature_table['row_count']}")
        print(f"split_counts={feature_table['split_counts']}")
        print(f"role_counts={feature_table['role_counts']}")
        print(f"split_safe_expert_coverage={feature_table['split_safe_expert_coverage']}")
        if "heldout_template" in report.summary:
            print(f"heldout_template={report.summary['heldout_template']}")
            print(f"template_counts={feature_table.get('template_counts', {})}")
            print(f"math_answer_index_counts={feature_table.get('math_answer_index_counts', {})}")
        if "expert_transform" in report.summary:
            print(f"expert_transform={report.summary['expert_transform']}")
        print(f"summary_path={output_dir / 'summary.json'}")
        print(f"markdown_path={output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
