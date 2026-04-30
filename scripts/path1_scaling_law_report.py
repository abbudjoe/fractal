#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_GLOB = "experiments/aws_sagemaker/path1_cuda_scout/*/extracted/path1-cuda-scout/summary.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "scaling_laws" / "path1_cuda_scaling"

LANE_LABELS = {
    "attention-only": "A",
    "parcae-bx-looped-attention": "B(x)",
    "parcae-p20-control-looped-attention": "RGRP-control",
}
LANE_COLORS = {
    "attention-only": "#263f8f",
    "parcae-bx-looped-attention": "#2f9d8f",
    "parcae-p20-control-looped-attention": "#d45d32",
}


@dataclass(frozen=True)
class ScalingPoint:
    run_label: str
    lane: str
    seed: int
    d_model: int
    total_layers: int
    parameters: int
    train_tokens: int
    final_loss: float
    train_tokens_per_second: float
    nominal_flops: float
    runtime_adjusted_flops: float


@dataclass(frozen=True)
class AggregatePoint:
    lane: str
    d_model: int
    total_layers: int
    parameters: int
    train_tokens: int
    count: int
    mean_loss: float
    loss_stdev: float
    mean_train_tokens_per_second: float
    nominal_flops: float
    runtime_adjusted_flops: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Path 1 CUDA scaling-law CSV/Markdown/SVG artifacts from SageMaker summary JSON files."
    )
    parser.add_argument(
        "--summary-glob",
        default=DEFAULT_SUMMARY_GLOB,
        help="Glob, relative to repo root unless absolute, for summary.json inputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-non-parity",
        action="store_true",
        help="Include older non-parity scout summaries. Defaults to parity-labeled runs only.",
    )
    return parser


def _glob_paths(pattern: str) -> list[Path]:
    if Path(pattern).is_absolute():
        paths = [Path(path) for path in sorted(glob.glob(pattern))]
    else:
        paths = sorted(REPO_ROOT.glob(pattern))
    return [path for path in paths if path.is_file()]


def _train_tokens(summary: dict) -> int:
    return int(summary["steps"]) * int(summary["batch_size"]) * int(summary["seq_len"])


def _nominal_flops(parameters: int, train_tokens: int) -> float:
    # Chinchilla-style dense LM training proxy. Architecture-specific runtime
    # factors are tracked separately via runtime_adjusted_flops.
    return float(6 * int(parameters) * int(train_tokens))


def load_points(paths: Iterable[Path], *, include_non_parity: bool) -> list[ScalingPoint]:
    points: list[ScalingPoint] = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        run_label = str(summary["run_label"])
        if not include_non_parity and "parity" not in run_label:
            continue
        train_tokens = _train_tokens(summary)
        rows = summary.get("rows", [])
        attention_row = next((row for row in rows if row.get("lane") == "attention-only"), None)
        attention_tps = (
            float(attention_row["train_tokens_per_second"])
            if attention_row is not None
            else None
        )
        for row in rows:
            lane = str(row["lane"])
            if lane not in LANE_LABELS:
                continue
            parameters = int(row["parameters"])
            nominal = _nominal_flops(parameters, train_tokens)
            lane_tps = float(row["train_tokens_per_second"])
            runtime_adjusted = nominal
            if attention_tps is not None and lane_tps > 0:
                runtime_adjusted = nominal * (attention_tps / lane_tps)
            points.append(
                ScalingPoint(
                    run_label=run_label,
                    lane=lane,
                    seed=int(summary["seed"]),
                    d_model=int(summary["d_model"]),
                    total_layers=int(summary["total_layers"]),
                    parameters=parameters,
                    train_tokens=train_tokens,
                    final_loss=float(row["final_loss"]),
                    train_tokens_per_second=lane_tps,
                    nominal_flops=nominal,
                    runtime_adjusted_flops=runtime_adjusted,
                )
            )
    return points


def aggregate_points(points: Iterable[ScalingPoint]) -> list[AggregatePoint]:
    buckets: dict[tuple[str, int, int, int, int], list[ScalingPoint]] = defaultdict(list)
    for point in points:
        buckets[
            (
                point.lane,
                point.d_model,
                point.total_layers,
                point.parameters,
                point.train_tokens,
            )
        ].append(point)

    aggregates: list[AggregatePoint] = []
    for (lane, d_model, total_layers, parameters, train_tokens), bucket in sorted(
        buckets.items(), key=lambda item: (item[0][4], item[0][0], item[0][1])
    ):
        losses = [point.final_loss for point in bucket]
        tps_values = [point.train_tokens_per_second for point in bucket]
        nominal_values = [point.nominal_flops for point in bucket]
        runtime_values = [point.runtime_adjusted_flops for point in bucket]
        aggregates.append(
            AggregatePoint(
                lane=lane,
                d_model=d_model,
                total_layers=total_layers,
                parameters=parameters,
                train_tokens=train_tokens,
                count=len(bucket),
                mean_loss=mean(losses),
                loss_stdev=stdev(losses) if len(losses) > 1 else 0.0,
                mean_train_tokens_per_second=mean(tps_values),
                nominal_flops=mean(nominal_values),
                runtime_adjusted_flops=mean(runtime_values),
            )
        )
    return aggregates


def write_csv(points: list[ScalingPoint], aggregates: list[AggregatePoint], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "path1_cuda_scaling_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_label",
                "seed",
                "lane",
                "d_model",
                "total_layers",
                "parameters",
                "train_tokens",
                "final_loss",
                "train_tokens_per_second",
                "nominal_flops",
                "runtime_adjusted_flops",
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(point.__dict__)

    with (output_dir / "path1_cuda_scaling_aggregates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lane",
                "d_model",
                "total_layers",
                "parameters",
                "train_tokens",
                "count",
                "mean_loss",
                "loss_stdev",
                "mean_train_tokens_per_second",
                "nominal_flops",
                "runtime_adjusted_flops",
            ],
        )
        writer.writeheader()
        for point in aggregates:
            writer.writerow(point.__dict__)


def _log_bounds(values: Iterable[float]) -> tuple[float, float]:
    positives = [value for value in values if value > 0]
    if not positives:
        return 1.0, 10.0
    lo = min(positives)
    hi = max(positives)
    if math.isclose(lo, hi):
        return lo / 1.4, hi * 1.4
    pad = math.sqrt(hi / lo)
    return lo / pad**0.20, hi * pad**0.20


def _linear_bounds(values: Iterable[float]) -> tuple[float, float]:
    vals = list(values)
    if not vals:
        return 0.0, 1.0
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        return lo - 0.5, hi + 0.5
    pad = (hi - lo) * 0.12
    return lo - pad, hi + pad


def _format_si(value: float) -> str:
    abs_value = abs(value)
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs_value >= scale:
            return f"{value / scale:.1f}{suffix}"
    return f"{value:.0f}"


def _format_pow10(value: float) -> str:
    if value <= 0:
        return "0"
    exponent = math.floor(math.log10(value))
    mantissa = value / (10**exponent)
    return f"{mantissa:.1f}e{exponent}"


class SvgCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = []

    def add(self, fragment: str) -> None:
        self.parts.append(fragment)

    def text(self, x: float, y: float, text: str, *, size: int = 12, fill: str = "#172033", anchor: str = "start") -> None:
        safe_text = html.escape(text)
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-family="Arial, sans-serif" text-anchor="{anchor}">{safe_text}</text>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, *, stroke: str = "#ccd3df", width: float = 1.0, dash: str = "") -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width:.1f}"{dash_attr}/>'
        )

    def circle(self, x: float, y: float, radius: float, *, fill: str, stroke: str = "#ffffff") -> None:
        self.add(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.4"/>'
        )

    def rect(self, x: float, y: float, width: float, height: float, *, fill: str, stroke: str = "none") -> None:
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
            f'fill="{fill}" stroke="{stroke}"/>'
        )

    def render(self) -> str:
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            '<rect width="100%" height="100%" fill="#f8fafc"/>\n'
            f"{body}\n</svg>\n"
        )


def _plot_panel(
    canvas: SvgCanvas,
    points: list[AggregatePoint],
    *,
    x_getter,
    y_getter,
    x_log: bool,
    y_log: bool,
    title: str,
    x_label: str,
    y_label: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    margin_left = 54
    margin_right = 14
    margin_top = 34
    margin_bottom = 44
    px0 = x + margin_left
    py0 = y + margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    px1 = px0 + plot_w
    py1 = py0 + plot_h

    xs = [float(x_getter(point)) for point in points]
    ys = [float(y_getter(point)) for point in points]
    x_min, x_max = _log_bounds(xs) if x_log else _linear_bounds(xs)
    y_min, y_max = _log_bounds(ys) if y_log else _linear_bounds(ys)

    def scale_x(value: float) -> float:
        if x_log:
            value = math.log10(value)
            lo = math.log10(x_min)
            hi = math.log10(x_max)
        else:
            lo = x_min
            hi = x_max
        return px0 + (value - lo) / (hi - lo) * plot_w

    def scale_y(value: float) -> float:
        if y_log:
            value = math.log10(value)
            lo = math.log10(y_min)
            hi = math.log10(y_max)
        else:
            lo = y_min
            hi = y_max
        return py1 - (value - lo) / (hi - lo) * plot_h

    canvas.rect(x, y, width, height, fill="#ffffff", stroke="#d7dde8")
    canvas.text(x + 14, y + 22, title, size=14, fill="#111827")

    for i in range(5):
        gx = px0 + plot_w * i / 4
        gy = py0 + plot_h * i / 4
        canvas.line(gx, py0, gx, py1, stroke="#e7ebf1")
        canvas.line(px0, gy, px1, gy, stroke="#e7ebf1")

    canvas.line(px0, py1, px1, py1, stroke="#9aa5b5", width=1.2)
    canvas.line(px0, py0, px0, py1, stroke="#9aa5b5", width=1.2)
    canvas.text((px0 + px1) / 2, y + height - 12, x_label, size=11, fill="#4b5563", anchor="middle")
    canvas.text(x + 12, (py0 + py1) / 2, y_label, size=11, fill="#4b5563", anchor="middle")

    for value, anchor_y in ((x_min, y + height - 28), (x_max, y + height - 28)):
        label = _format_pow10(value) if x_log else _format_si(value)
        canvas.text(scale_x(value), anchor_y, label, size=9, fill="#697386", anchor="middle")
    for value in (y_min, y_max):
        label = _format_pow10(value) if y_log else f"{value:.3g}"
        canvas.text(x + margin_left - 8, scale_y(value) + 4, label, size=9, fill="#697386", anchor="end")

    for point in points:
        color = LANE_COLORS[point.lane]
        cx = scale_x(float(x_getter(point)))
        cy = scale_y(float(y_getter(point)))
        radius = 4.0 + min(point.count, 5) * 0.7
        canvas.circle(cx, cy, radius, fill=color)
        label = LANE_LABELS[point.lane]
        canvas.text(cx + 7, cy - 5, f"{label}/{point.train_tokens // 1_000_000}M", size=8, fill=color)


def write_svg(aggregates: list[AggregatePoint], output_dir: Path) -> None:
    canvas = SvgCanvas(1320, 430)
    canvas.text(24, 30, "Path 1 CUDA Scaling-Law Scout", size=22, fill="#111827")
    canvas.text(
        24,
        50,
        "Aggregated SageMaker parity runs. FLOPs use 6ND nominal; runtime-adjusted FLOPs scale by attention tok/s divided by lane tok/s within each run.",
        size=11,
        fill="#4b5563",
    )
    panels = [
        (
            "Loss vs Parameters",
            "Parameters",
            "Eval loss",
            lambda p: p.parameters,
            lambda p: p.mean_loss,
            True,
            False,
        ),
        (
            "Parameters vs Runtime-Adjusted FLOPs",
            "Runtime-adjusted FLOPs",
            "Parameters",
            lambda p: p.runtime_adjusted_flops,
            lambda p: p.parameters,
            True,
            True,
        ),
        (
            "Tokens vs Runtime-Adjusted FLOPs",
            "Runtime-adjusted FLOPs",
            "Train tokens",
            lambda p: p.runtime_adjusted_flops,
            lambda p: p.train_tokens,
            True,
            True,
        ),
    ]
    for idx, (title, x_label, y_label, x_getter, y_getter, x_log, y_log) in enumerate(panels):
        _plot_panel(
            canvas,
            aggregates,
            x_getter=x_getter,
            y_getter=y_getter,
            x_log=x_log,
            y_log=y_log,
            title=title,
            x_label=x_label,
            y_label=y_label,
            x=24 + idx * 430,
            y=76,
            width=400,
            height=310,
        )

    legend_x = 24
    legend_y = 410
    for lane, label in LANE_LABELS.items():
        canvas.circle(legend_x, legend_y, 5, fill=LANE_COLORS[lane])
        canvas.text(legend_x + 10, legend_y + 4, label, size=11)
        legend_x += 150

    (output_dir / "path1_cuda_scaling.svg").write_text(canvas.render(), encoding="utf-8")


def write_markdown(points: list[ScalingPoint], aggregates: list[AggregatePoint], output_dir: Path) -> None:
    lines = [
        "# Path 1 CUDA Scaling-Law Scout",
        "",
        "This artifact aggregates completed SageMaker Path 1 CUDA parity summaries into scaling-law inputs.",
        "",
        "![Path 1 CUDA scaling scout](./path1_cuda_scaling.svg)",
        "",
        "## Aggregate Points",
        "",
        "| Lane | d_model | Tokens | Seeds | Mean Loss | Loss Stdev | Mean tok/s | Runtime-adjusted FLOPs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in aggregates:
        lines.append(
            f"| `{LANE_LABELS[point.lane]}` | {point.d_model} | {point.train_tokens:,} | "
            f"{point.count} | {point.mean_loss:.4f} | {point.loss_stdev:.4f} | "
            f"{point.mean_train_tokens_per_second:,.0f} | {point.runtime_adjusted_flops:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- `Loss vs Parameters` currently shows architecture offsets at nearly fixed parameter count; it is not yet a full parameter scaling law.",
            "- `Runtime-adjusted FLOPs` is a practical proxy until we add exact per-architecture FLOP accounting.",
            "- New model-size rungs should add points horizontally in the parameter plot; longer token rungs should add points upward/rightward in the tokens-vs-compute plot.",
            "",
            "## Source Runs",
            "",
            "| Run | Seed | Lane | Tokens | Loss | tok/s |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for point in sorted(points, key=lambda item: (item.train_tokens, item.seed, item.lane)):
        lines.append(
            f"| `{point.run_label}` | {point.seed} | `{LANE_LABELS[point.lane]}` | "
            f"{point.train_tokens:,} | {point.final_loss:.4f} | {point.train_tokens_per_second:,.0f} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    paths = _glob_paths(args.summary_glob)
    points = load_points(paths, include_non_parity=args.include_non_parity)
    if not points:
        raise SystemExit(f"no scaling points found for {args.summary_glob!r}")
    aggregates = aggregate_points(points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(points, aggregates, args.output_dir)
    write_svg(aggregates, args.output_dir)
    write_markdown(points, aggregates, args.output_dir)
    print(args.output_dir / "README.md")
    print(args.output_dir / "path1_cuda_scaling.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
