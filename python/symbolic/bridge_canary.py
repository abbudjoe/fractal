from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python.specs.common import repo_relative
from python.symbolic.bridge import deterministic_token_nll, mean, median


@dataclass(frozen=True)
class BridgeCanaryRun:
    name: str
    objective: str
    feature_set: str
    train_token_accuracy: float
    validation_token_accuracy: float
    extrapolation_token_accuracy: float
    train_nll: float
    validation_nll: float
    extrapolation_nll: float
    train_router_accuracy: float | None = None
    validation_router_accuracy: float | None = None
    extrapolation_router_accuracy: float | None = None
    selected_expert_counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "objective": self.objective,
            "feature_set": self.feature_set,
            "train_token_accuracy": self.train_token_accuracy,
            "validation_token_accuracy": self.validation_token_accuracy,
            "extrapolation_token_accuracy": self.extrapolation_token_accuracy,
            "train_nll": self.train_nll,
            "validation_nll": self.validation_nll,
            "extrapolation_nll": self.extrapolation_nll,
            "train_router_accuracy": self.train_router_accuracy,
            "validation_router_accuracy": self.validation_router_accuracy,
            "extrapolation_router_accuracy": self.extrapolation_router_accuracy,
            "selected_expert_counts": self.selected_expert_counts or {},
        }


@dataclass(frozen=True)
class BridgeCanaryReport:
    bridge_summary_path: str
    feature_table_path: str
    run_label: str
    token_bins: int
    seed: int
    epochs: int
    learning_rate: float
    hidden_units: int
    device: str
    runs: tuple[BridgeCanaryRun, ...]
    summary: dict[str, Any]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bridge_summary_path": self.bridge_summary_path,
            "feature_table_path": self.feature_table_path,
            "run_label": self.run_label,
            "token_bins": self.token_bins,
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden_units": self.hidden_units,
            "device": self.device,
            "runs": [run.to_dict() for run in self.runs],
            "summary": self.summary,
            "output_dir": self.output_dir,
        }


def run_bridge_canary(
    bridge_summary_path: Path,
    output_dir: Path,
    *,
    run_label: str,
    seed: int = 123,
    epochs: int = 400,
    learning_rate: float = 0.04,
    hidden_units: int = 0,
    device: str = "auto",
) -> BridgeCanaryReport:
    torch = import_torch()
    bridge_summary = json.loads(bridge_summary_path.read_text())
    feature_table_path = resolve_feature_table_path(bridge_summary, bridge_summary_path)
    rows = load_feature_rows(feature_table_path)
    if not rows:
        raise ValueError(f"bridge feature table has no rows: {feature_table_path}")

    random.seed(seed)
    torch.manual_seed(seed)
    selected_device = resolve_device(torch, device)
    token_bins = int(bridge_summary.get("token_bins") or rows[0]["token_bins"])
    task_ids = tuple(sorted({str(row["task_id"]) for row in rows}))
    expert_ids = tuple(sorted({str(expert) for row in rows for expert in row["experts"]}))

    runs = [
        evaluate_majority_token(rows, token_bins),
        evaluate_best_single_expert(rows, expert_ids, token_bins),
        evaluate_oracle_router(rows, expert_ids, token_bins),
    ]
    runs.append(
        train_token_classifier(
            torch,
            rows,
            task_ids,
            expert_ids,
            token_bins,
            feature_set="x-task",
            run_name=classifier_name("token-x-task", hidden_units),
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            device=selected_device,
        )
    )
    runs.append(
        train_token_classifier(
            torch,
            rows,
            task_ids,
            expert_ids,
            token_bins,
            feature_set="side-channel",
            run_name=classifier_name("token-frozen-side-channel", hidden_units),
            seed=seed + 1,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            device=selected_device,
        )
    )
    runs.append(
        train_router_classifier(
            torch,
            rows,
            task_ids,
            expert_ids,
            token_bins,
            feature_set="x-task",
            run_name=classifier_name("router-x-task", hidden_units),
            seed=seed + 2,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            device=selected_device,
        )
    )
    runs.append(
        train_router_classifier(
            torch,
            rows,
            task_ids,
            expert_ids,
            token_bins,
            feature_set="expert-signal",
            run_name=classifier_name("router-expert-signal", hidden_units),
            seed=seed + 3,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            device=selected_device,
        )
    )

    summary = summarize_canary(runs, rows, expert_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = BridgeCanaryReport(
        bridge_summary_path=repo_relative(bridge_summary_path),
        feature_table_path=repo_relative(feature_table_path),
        run_label=run_label,
        token_bins=token_bins,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        device=str(selected_device),
        runs=tuple(runs),
        summary=summary,
        output_dir=repo_relative(output_dir),
    )
    write_canary_report(report, output_dir)
    return report


def import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "symbolic bridge canary requires PyTorch. Run with "
            "`uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_canary.py ...`."
        ) from exc
    return torch


def resolve_feature_table_path(bridge_summary: dict[str, Any], bridge_summary_path: Path) -> Path:
    raw_path = bridge_summary["summary"]["feature_table"]["path"]
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = bridge_summary_path.parent / path
    if candidate.exists():
        return candidate
    return path


def load_feature_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def classifier_name(prefix: str, hidden_units: int) -> str:
    suffix = "linear" if hidden_units <= 0 else f"mlp{hidden_units}"
    return f"{prefix}-{suffix}"


def evaluate_majority_token(rows: list[dict[str, Any]], token_bins: int) -> BridgeCanaryRun:
    train_tokens = [int(row["target_token"]) for row in rows if row["split"] == "train"]
    majority = max(set(train_tokens), key=train_tokens.count)
    split_accuracy = {
        split: token_accuracy_for_constant(rows, split, majority)
        for split in ("train", "validation", "extrapolation")
    }
    return BridgeCanaryRun(
        name="token-majority",
        objective="token",
        feature_set="constant",
        train_token_accuracy=split_accuracy["train"],
        validation_token_accuracy=split_accuracy["validation"],
        extrapolation_token_accuracy=split_accuracy["extrapolation"],
        train_nll=deterministic_token_nll(split_accuracy["train"], token_bins),
        validation_nll=deterministic_token_nll(split_accuracy["validation"], token_bins),
        extrapolation_nll=deterministic_token_nll(split_accuracy["extrapolation"], token_bins),
    )


def evaluate_best_single_expert(
    rows: list[dict[str, Any]],
    expert_ids: tuple[str, ...],
    token_bins: int,
) -> BridgeCanaryRun:
    best_expert = max(
        expert_ids,
        key=lambda expert: expert_token_accuracy(rows, "train", expert),
    )
    split_accuracy = {
        split: expert_token_accuracy(rows, split, best_expert)
        for split in ("train", "validation", "extrapolation")
    }
    return BridgeCanaryRun(
        name="best-single-expert-by-train",
        objective="router",
        feature_set="global-expert",
        train_token_accuracy=split_accuracy["train"],
        validation_token_accuracy=split_accuracy["validation"],
        extrapolation_token_accuracy=split_accuracy["extrapolation"],
        train_nll=deterministic_token_nll(split_accuracy["train"], token_bins),
        validation_nll=deterministic_token_nll(split_accuracy["validation"], token_bins),
        extrapolation_nll=deterministic_token_nll(split_accuracy["extrapolation"], token_bins),
        selected_expert_counts={best_expert: len(rows)},
    )


def evaluate_oracle_router(
    rows: list[dict[str, Any]],
    expert_ids: tuple[str, ...],
    token_bins: int,
) -> BridgeCanaryRun:
    selected_counts: dict[str, int] = {}
    split_accuracy = {}
    for split in ("train", "validation", "extrapolation"):
        split_rows = rows_for_split(rows, split)
        correct = 0
        for row in split_rows:
            expert = str(row["best_expert_id"])
            selected_counts[expert] = selected_counts.get(expert, 0) + 1
            correct += int(expert_matches(row, expert))
        split_accuracy[split] = correct / len(split_rows) if split_rows else 0.0
    return BridgeCanaryRun(
        name="oracle-row-router",
        objective="router",
        feature_set="target-leaking-oracle",
        train_token_accuracy=split_accuracy["train"],
        validation_token_accuracy=split_accuracy["validation"],
        extrapolation_token_accuracy=split_accuracy["extrapolation"],
        train_nll=deterministic_token_nll(split_accuracy["train"], token_bins),
        validation_nll=deterministic_token_nll(split_accuracy["validation"], token_bins),
        extrapolation_nll=deterministic_token_nll(split_accuracy["extrapolation"], token_bins),
        train_router_accuracy=1.0,
        validation_router_accuracy=1.0,
        extrapolation_router_accuracy=1.0,
        selected_expert_counts={expert: selected_counts.get(expert, 0) for expert in expert_ids},
    )


def train_token_classifier(
    torch: Any,
    rows: list[dict[str, Any]],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    token_bins: int,
    *,
    feature_set: str,
    run_name: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    device: Any,
) -> BridgeCanaryRun:
    labels = [int(row["target_token"]) for row in rows]
    features = build_feature_matrix(rows, task_ids, expert_ids, feature_set)
    predictions, losses = train_classifier(
        torch,
        features,
        labels,
        rows,
        classes=token_bins,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        device=device,
    )
    split_accuracy = {
        split: indexed_accuracy(predictions[split], target_tokens(rows_for_split(rows, split)))
        for split in ("train", "validation", "extrapolation")
    }
    return BridgeCanaryRun(
        name=run_name,
        objective="token",
        feature_set=feature_set,
        train_token_accuracy=split_accuracy["train"],
        validation_token_accuracy=split_accuracy["validation"],
        extrapolation_token_accuracy=split_accuracy["extrapolation"],
        train_nll=losses["train"],
        validation_nll=losses["validation"],
        extrapolation_nll=losses["extrapolation"],
    )


def train_router_classifier(
    torch: Any,
    rows: list[dict[str, Any]],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    token_bins: int,
    *,
    feature_set: str,
    run_name: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    device: Any,
) -> BridgeCanaryRun:
    expert_to_index = {expert: index for index, expert in enumerate(expert_ids)}
    labels = [expert_to_index[str(row["best_expert_id"])] for row in rows]
    features = build_feature_matrix(rows, task_ids, expert_ids, feature_set)
    predictions, losses = train_classifier(
        torch,
        features,
        labels,
        rows,
        classes=len(expert_ids),
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        device=device,
    )
    split_router_accuracy = {
        split: indexed_accuracy(predictions[split], router_targets(rows_for_split(rows, split), expert_to_index))
        for split in ("train", "validation", "extrapolation")
    }
    split_token_accuracy = {
        split: routed_token_accuracy(rows_for_split(rows, split), predictions[split], expert_ids)
        for split in ("train", "validation", "extrapolation")
    }
    selected_counts = {expert: 0 for expert in expert_ids}
    for split in ("train", "validation", "extrapolation"):
        for prediction in predictions[split]:
            selected_counts[expert_ids[prediction]] += 1
    return BridgeCanaryRun(
        name=run_name,
        objective="router",
        feature_set=feature_set,
        train_token_accuracy=split_token_accuracy["train"],
        validation_token_accuracy=split_token_accuracy["validation"],
        extrapolation_token_accuracy=split_token_accuracy["extrapolation"],
        train_nll=losses["train"],
        validation_nll=losses["validation"],
        extrapolation_nll=losses["extrapolation"],
        train_router_accuracy=split_router_accuracy["train"],
        validation_router_accuracy=split_router_accuracy["validation"],
        extrapolation_router_accuracy=split_router_accuracy["extrapolation"],
        selected_expert_counts=selected_counts,
    )


def train_classifier(
    torch: Any,
    features: list[list[float]],
    labels: list[int],
    rows: list[dict[str, Any]],
    *,
    classes: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    device: Any,
) -> tuple[dict[str, list[int]], dict[str, float]]:
    torch.manual_seed(seed)
    indices = split_indices(rows)
    train_indices = indices["train"]
    normalized = normalize_features(features, train_indices)
    x = torch.tensor(normalized, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    model = build_classifier(torch, len(normalized[0]), classes, hidden_units).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x[train_indices])
        loss = torch.nn.functional.cross_entropy(logits, y[train_indices])
        loss.backward()
        optimizer.step()

    model.eval()
    predictions: dict[str, list[int]] = {}
    losses: dict[str, float] = {}
    with torch.no_grad():
        logits = model(x)
        for split, split_indices_value in indices.items():
            if not split_indices_value:
                predictions[split] = []
                losses[split] = math.inf
                continue
            split_logits = logits[split_indices_value]
            split_targets = y[split_indices_value]
            split_loss = torch.nn.functional.cross_entropy(split_logits, split_targets)
            predictions[split] = [int(value) for value in split_logits.argmax(dim=-1).detach().cpu().tolist()]
            losses[split] = float(split_loss.detach().cpu().item())
    return predictions, losses


def build_classifier(torch: Any, input_features: int, classes: int, hidden_units: int) -> Any:
    if hidden_units <= 0:
        return torch.nn.Linear(input_features, classes)
    return torch.nn.Sequential(
        torch.nn.Linear(input_features, hidden_units),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_units, classes),
    )


def build_feature_matrix(
    rows: list[dict[str, Any]],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    feature_set: str,
) -> list[list[float]]:
    task_to_index = {task: index for index, task in enumerate(task_ids)}
    features = []
    for row in rows:
        row_features = [float(row["x"])]
        if feature_set in {"x-task", "side-channel", "expert-signal"}:
            row_features.extend(one_hot(task_to_index[str(row["task_id"])], len(task_ids)))
        if feature_set in {"side-channel", "expert-signal"}:
            row_features.extend(expert_signal_features(row, expert_ids))
        features.append(row_features)
    return features


def expert_signal_features(row: dict[str, Any], expert_ids: tuple[str, ...]) -> list[float]:
    quantizer = row["quantizer"]
    minimum = float(quantizer["minimum"])
    maximum = float(quantizer["maximum"])
    width = max(maximum - minimum, 1.0e-12)
    token_bins = max(1, int(row["token_bins"]))
    features: list[float] = []
    for expert in expert_ids:
        payload = row["experts"].get(expert, {})
        raw_prediction = payload.get("prediction")
        if isinstance(raw_prediction, (int, float)) and math.isfinite(raw_prediction):
            normalized_prediction = (float(raw_prediction) - minimum) / width
            normalized_prediction = max(-4.0, min(5.0, normalized_prediction))
            valid = 1.0
        else:
            normalized_prediction = 0.0
            valid = 0.0
        token = int(payload.get("token", -1))
        normalized_token = token / max(1, token_bins - 1) if token >= 0 else -1.0
        features.extend([normalized_prediction, normalized_token, valid])
    return features


def normalize_features(features: list[list[float]], train_indices: list[int]) -> list[list[float]]:
    columns = len(features[0])
    means = []
    stds = []
    for column in range(columns):
        values = [features[index][column] for index in train_indices]
        column_mean = mean(values)
        variance = mean((value - column_mean) ** 2 for value in values)
        means.append(column_mean)
        stds.append(math.sqrt(variance) if variance > 1.0e-12 else 1.0)
    return [
        [(row[column] - means[column]) / stds[column] for column in range(columns)]
        for row in features
    ]


def resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("requested device=mps, but torch MPS is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of auto|cpu|mps")


def split_indices(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    indices = {"train": [], "validation": [], "extrapolation": []}
    for index, row in enumerate(rows):
        split = str(row["split"])
        if split in indices:
            indices[split].append(index)
    return indices


def rows_for_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split]


def one_hot(index: int, size: int) -> list[float]:
    return [1.0 if item == index else 0.0 for item in range(size)]


def target_tokens(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row["target_token"]) for row in rows]


def router_targets(rows: list[dict[str, Any]], expert_to_index: dict[str, int]) -> list[int]:
    return [expert_to_index[str(row["best_expert_id"])] for row in rows]


def indexed_accuracy(predictions: list[int], targets: list[int]) -> float:
    if not targets:
        return 0.0
    return sum(1 for prediction, target in zip(predictions, targets) if prediction == target) / len(targets)


def token_accuracy_for_constant(rows: list[dict[str, Any]], split: str, token: int) -> float:
    split_rows = rows_for_split(rows, split)
    if not split_rows:
        return 0.0
    return sum(1 for row in split_rows if int(row["target_token"]) == token) / len(split_rows)


def expert_token_accuracy(rows: list[dict[str, Any]], split: str, expert: str) -> float:
    split_rows = rows_for_split(rows, split)
    if not split_rows:
        return 0.0
    return sum(1 for row in split_rows if expert_matches(row, expert)) / len(split_rows)


def routed_token_accuracy(rows: list[dict[str, Any]], predictions: list[int], expert_ids: tuple[str, ...]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row, prediction in zip(rows, predictions):
        correct += int(expert_matches(row, expert_ids[prediction]))
    return correct / len(rows)


def expert_matches(row: dict[str, Any], expert: str) -> bool:
    payload = row["experts"].get(expert)
    return bool(payload and payload.get("token") == row["target_token"])


def summarize_canary(
    runs: list[BridgeCanaryRun],
    rows: list[dict[str, Any]],
    expert_ids: tuple[str, ...],
) -> dict[str, Any]:
    non_oracle_runs = [run for run in runs if not run.name.startswith("oracle")]
    trained_runs = [
        run
        for run in runs
        if run.name.startswith(("token-x-task", "token-frozen-side-channel", "router-x-task", "router-expert-signal"))
    ]
    side_run = next((run for run in runs if run.name.startswith("token-frozen-side-channel")), None)
    x_run = next((run for run in runs if run.name.startswith("token-x-task")), None)
    router_signal = next((run for run in runs if run.name.startswith("router-expert-signal")), None)
    single_expert = next((run for run in runs if run.name == "best-single-expert-by-train"), None)
    split_safe = {
        split: mean(1.0 if row["oracle_has_safe_expert"] else 0.0 for row in rows_for_split(rows, split))
        for split in ("train", "validation", "extrapolation")
    }
    return {
        "total_rows": len(rows),
        "expert_ids": list(expert_ids),
        "safe_expert_coverage": split_safe,
        "best_validation_token_accuracy": max(runs, key=lambda run: run.validation_token_accuracy).name,
        "best_extrapolation_token_accuracy": max(runs, key=lambda run: run.extrapolation_token_accuracy).name,
        "best_non_oracle_extrapolation_token_accuracy": max(non_oracle_runs, key=lambda run: run.extrapolation_token_accuracy).name,
        "best_trained_extrapolation_token_accuracy": max(trained_runs, key=lambda run: run.extrapolation_token_accuracy).name,
        "median_extrapolation_token_accuracy": median(run.extrapolation_token_accuracy for run in runs),
        "frozen_side_channel_extrapolation_delta": (
            side_run.extrapolation_token_accuracy - x_run.extrapolation_token_accuracy
            if side_run is not None and x_run is not None
            else 0.0
        ),
        "router_expert_signal_extrapolation_delta": (
            router_signal.extrapolation_token_accuracy - single_expert.extrapolation_token_accuracy
            if router_signal is not None and single_expert is not None
            else 0.0
        ),
    }


def write_canary_report(report: BridgeCanaryReport, output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    with (output_dir / "runs.jsonl").open("w") as handle:
        for run in report.runs:
            handle.write(json.dumps(run.to_dict(), sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(render_canary_markdown(report))


def render_canary_markdown(report: BridgeCanaryReport) -> str:
    lines = [
        f"# Symbolic Bridge Canary: {report.run_label}",
        "",
        f"Bridge summary: `{report.bridge_summary_path}`",
        f"Feature table: `{report.feature_table_path}`",
        f"Token bins: `{report.token_bins}`",
        f"Device: `{report.device}`",
        f"Epochs: `{report.epochs}`",
        f"Learning rate: `{report.learning_rate}`",
        f"Hidden units: `{report.hidden_units}`",
        "",
        "| run | objective | features | train token acc | val token acc | extrap token acc | val NLL | extrap NLL | router val acc | router extrap acc |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report.runs:
        lines.append(
            "| "
            f"`{run.name}` | "
            f"{run.objective} | "
            f"{run.feature_set} | "
            f"{run.train_token_accuracy:.3f} | "
            f"{run.validation_token_accuracy:.3f} | "
            f"{run.extrapolation_token_accuracy:.3f} | "
            f"{run.validation_nll:.4g} | "
            f"{run.extrapolation_nll:.4g} | "
            f"{format_optional(run.validation_router_accuracy)} | "
            f"{format_optional(run.extrapolation_router_accuracy)} |"
        )
    safe = report.summary["safe_expert_coverage"]
    lines.extend(
        [
            "",
            f"Best validation token accuracy: `{report.summary['best_validation_token_accuracy']}`",
            f"Best extrapolation token accuracy: `{report.summary['best_extrapolation_token_accuracy']}`",
            f"Best non-oracle extrapolation token accuracy: `{report.summary['best_non_oracle_extrapolation_token_accuracy']}`",
            f"Best trained extrapolation token accuracy: `{report.summary['best_trained_extrapolation_token_accuracy']}`",
            f"Frozen side-channel extrapolation delta vs x-task: `{report.summary['frozen_side_channel_extrapolation_delta']:.3f}`",
            f"Router expert-signal extrapolation delta vs best single expert: `{report.summary['router_expert_signal_extrapolation_delta']:.3f}`",
            f"Safe-expert coverage by split: `train={safe['train']:.3f}, validation={safe['validation']:.3f}, extrapolation={safe['extrapolation']:.3f}`",
            "",
        ]
    )
    return "\n".join(lines)


def format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"
