#!/usr/bin/env python3
"""Materialize a frozen FineWeb JSONL train/eval slice for v3a local benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--config", default="CC-MAIN-2024-10")
    parser.add_argument("--split", default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--eval-rows", type=int, required=True)
    parser.add_argument("--text-field", default="text")
    return parser.parse_args()


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    base = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    url = f"{base}?{urlencode(params)}"
    with urlopen(url, timeout=120) as response:
        payload = json.load(response)
    return payload["rows"]


def write_jsonl(path: Path, rows: list[dict], text_field: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            text = row["row"][text_field]
            handle.write(json.dumps({text_field: text}, ensure_ascii=False))
            handle.write("\n")


def write_readme(
    path: Path,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    train_rows: int,
    eval_rows: int,
    text_field: str,
) -> None:
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(
        "\n".join(
            [
                "# v3A FineWeb Local Benchmark Slice",
                "",
                f"- dataset: `{dataset}`",
                f"- config: `{config}`",
                f"- split: `{split}`",
                f"- rows fetched: `offset={offset},length={length}` on {fetched_at}",
                f"- train rows: first {train_rows}",
                f"- eval rows: next {eval_rows}",
                f"- text field: `{text_field}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.length <= 0:
        raise SystemExit("--length must be greater than zero")
    if args.train_rows <= 0 or args.eval_rows <= 0:
        raise SystemExit("--train-rows and --eval-rows must be greater than zero")
    if args.train_rows + args.eval_rows > args.length:
        raise SystemExit("--train-rows + --eval-rows may not exceed --length")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = fetch_rows(args.dataset, args.config, args.split, args.offset, args.length)
    if len(rows) < args.train_rows + args.eval_rows:
        raise SystemExit(
            f"fetched {len(rows)} rows, but need at least {args.train_rows + args.eval_rows}"
        )

    train_rows = rows[: args.train_rows]
    eval_rows = rows[args.train_rows : args.train_rows + args.eval_rows]

    write_jsonl(output_dir / "train.jsonl", train_rows, args.text_field)
    write_jsonl(output_dir / "eval.jsonl", eval_rows, args.text_field)
    write_readme(
        output_dir / "README.md",
        args.dataset,
        args.config,
        args.split,
        args.offset,
        args.length,
        args.train_rows,
        args.eval_rows,
        args.text_field,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
