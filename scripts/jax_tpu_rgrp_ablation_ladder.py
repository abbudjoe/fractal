#!/usr/bin/env python3
"""Run a bounded JAX/TPU lowering ablation ladder for the RGRP FFN seam."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LM_SMOKE = REPO_ROOT / "scripts" / "jax_tpu_lm_smoke.py"


@dataclass(frozen=True)
class LadderCase:
    name: str
    variant: str
    state_transform: str = "block-diagonal-4-masked-dense"
    scan_unroll: int = 1
    projection_mode: str = "sequence"
    trig_mode: str = "precompute"


def default_cases() -> list[LadderCase]:
    """A small first ladder: one axis at a time plus the MLP control."""

    return [
        LadderCase(name="mlp-control", variant="mlp"),
        LadderCase(name="rgrp-default", variant="rgrp"),
        LadderCase(name="state-dense", variant="rgrp", state_transform="dense"),
        LadderCase(name="state-block4-grouped", variant="rgrp", state_transform="block-diagonal-4"),
        LadderCase(name="trig-inside-scan", variant="rgrp", trig_mode="scan"),
        LadderCase(name="projection-inside-scan", variant="rgrp", projection_mode="scan", trig_mode="scan"),
        LadderCase(name="unroll-2", variant="rgrp", scan_unroll=2),
        LadderCase(name="unroll-4", variant="rgrp", scan_unroll=4),
        LadderCase(name="unroll-8", variant="rgrp", scan_unroll=8),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded RGRP JAX/XLA lowering ablation ladder through the LM smoke CLI."
    )
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def command_for_case(args: argparse.Namespace, case: LadderCase) -> list[str]:
    command = [
        sys.executable,
        str(LM_SMOKE),
        "--variant",
        case.variant,
        "--vocab-size",
        str(args.vocab_size),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
        "--d-model",
        str(args.d_model),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--ffn-multiplier",
        str(args.ffn_multiplier),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
    ]
    if args.forward_only:
        command.append("--forward-only")
    if case.variant == "rgrp":
        command.extend(
            [
                "--rgrp-state-transform",
                case.state_transform,
                "--rgrp-scan-unroll",
                str(case.scan_unroll),
                "--rgrp-projection-mode",
                case.projection_mode,
                "--rgrp-trig-mode",
                case.trig_mode,
            ]
        )
    return command


def parse_json(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"subprocess did not emit a JSON object: {stdout}")
    return json.loads(stdout[start : end + 1])


def run_case(args: argparse.Namespace, case: LadderCase) -> dict[str, Any]:
    command = command_for_case(args, case)
    row: dict[str, Any] = {
        "case": case.name,
        "variant": case.variant,
        "rgrp_state_transform": case.state_transform if case.variant == "rgrp" else None,
        "rgrp_scan_unroll": case.scan_unroll if case.variant == "rgrp" else None,
        "rgrp_projection_mode": case.projection_mode if case.variant == "rgrp" else None,
        "rgrp_trig_mode": case.trig_mode if case.variant == "rgrp" else None,
        "command": " ".join(command),
    }
    if args.dry_run:
        row["status"] = "dry-run"
        return row
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    row["status"] = "ok" if completed.returncode == 0 else "failed"
    if completed.returncode == 0:
        row.update(parse_json(completed.stdout))
    else:
        row["returncode"] = completed.returncode
        row["stdout"] = completed.stdout
        row["stderr"] = completed.stderr
    return row


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Case | Variant | State | Projection | Trig | Unroll | Params | Compile s | Tok/s | Loss | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("case", ""),
                    row.get("variant", ""),
                    row.get("rgrp_state_transform") or "",
                    row.get("rgrp_projection_mode") or "",
                    row.get("rgrp_trig_mode") or "",
                    format_number(row.get("rgrp_scan_unroll"), 0),
                    format_number(row.get("parameter_count"), 0),
                    format_number(row.get("compile_seconds")),
                    format_number(row.get("steady_state_tokens_per_second")),
                    format_number(row.get("loss"), 4),
                    row.get("status", ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows = [run_case(args, case) for case in default_cases()]
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
    table = markdown_table(rows)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(table + "\n", encoding="utf-8")
    print(table)
    return 1 if any(row.get("status") == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
