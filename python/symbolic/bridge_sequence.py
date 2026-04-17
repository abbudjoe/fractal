from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python.specs.common import repo_relative
from python.symbolic.bridge import deterministic_token_nll, mean, median
from python.symbolic.bridge_canary import (
    expert_matches,
    expert_signal_features,
    import_torch,
    load_feature_rows,
    one_hot,
    resolve_device,
    resolve_feature_table_path,
)


@dataclass(frozen=True)
class SequenceBridgeRun:
    name: str
    objective: str
    feature_set: str
    train_token_accuracy: float
    validation_token_accuracy: float
    extrapolation_token_accuracy: float
    train_loss: float
    validation_loss: float
    extrapolation_loss: float
    train_rmse: float | None = None
    validation_rmse: float | None = None
    extrapolation_rmse: float | None = None
    train_router_accuracy: float | None = None
    validation_router_accuracy: float | None = None
    extrapolation_router_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "objective": self.objective,
            "feature_set": self.feature_set,
            "train_token_accuracy": self.train_token_accuracy,
            "validation_token_accuracy": self.validation_token_accuracy,
            "extrapolation_token_accuracy": self.extrapolation_token_accuracy,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "extrapolation_loss": self.extrapolation_loss,
            "train_rmse": self.train_rmse,
            "validation_rmse": self.validation_rmse,
            "extrapolation_rmse": self.extrapolation_rmse,
            "train_router_accuracy": self.train_router_accuracy,
            "validation_router_accuracy": self.validation_router_accuracy,
            "extrapolation_router_accuracy": self.extrapolation_router_accuracy,
        }


@dataclass(frozen=True)
class SequenceBridgeReport:
    bridge_summary_path: str
    feature_table_path: str
    run_label: str
    token_bins: int
    seed: int
    epochs: int
    learning_rate: float
    hidden_units: int
    device: str
    runs: tuple[SequenceBridgeRun, ...]
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


@dataclass(frozen=True)
class SequenceSplit:
    rows: list[list[dict[str, Any]]]
    previous_tokens: list[list[int]]
    target_tokens: list[list[int]]
    target_values: list[list[float]]
    features: list[list[list[float]]]
    router_targets: list[list[int]] | None = None


def run_sequence_bridge(
    bridge_summary_path: Path,
    output_dir: Path,
    *,
    run_label: str,
    seed: int = 321,
    epochs: int = 600,
    learning_rate: float = 0.015,
    hidden_units: int = 32,
    device: str = "auto",
) -> SequenceBridgeReport:
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
        evaluate_majority_sequence(rows, token_bins),
        evaluate_best_single_expert_sequence(rows, expert_ids, token_bins),
        evaluate_oracle_router_sequence(rows, token_bins),
    ]
    for feature_set in ("x-task", "side-channel"):
        runs.append(
            train_sequence_token_model(
                torch,
                rows,
                task_ids,
                expert_ids,
                token_bins,
                feature_set=feature_set,
                run_name=f"seq-token-{feature_set}-gru{hidden_units}",
                seed=seed + len(runs),
                epochs=epochs,
                learning_rate=learning_rate,
                hidden_units=hidden_units,
                device=selected_device,
            )
        )
    for feature_set in ("x-task", "side-channel"):
        runs.append(
            train_sequence_regression_model(
                torch,
                rows,
                task_ids,
                expert_ids,
                token_bins,
                feature_set=feature_set,
                run_name=f"seq-continuous-{feature_set}-gru{hidden_units}",
                seed=seed + len(runs),
                epochs=epochs,
                learning_rate=learning_rate,
                hidden_units=hidden_units,
                device=selected_device,
            )
        )
    for feature_set in ("x-task", "expert-signal"):
        runs.append(
            train_sequence_router_model(
                torch,
                rows,
                task_ids,
                expert_ids,
                token_bins,
                feature_set=feature_set,
                run_name=f"seq-router-{feature_set}-gru{hidden_units}",
                seed=seed + len(runs),
                epochs=epochs,
                learning_rate=learning_rate,
                hidden_units=hidden_units,
                device=selected_device,
            )
        )

    summary = summarize_sequence_bridge(runs, rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = SequenceBridgeReport(
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
    write_sequence_bridge_report(report, output_dir)
    return report


def evaluate_majority_sequence(rows: list[dict[str, Any]], token_bins: int) -> SequenceBridgeRun:
    train_tokens = [int(row["target_token"]) for row in rows if row["split"] == "train"]
    majority = max(set(train_tokens), key=train_tokens.count)
    accuracies = {
        split: constant_token_accuracy(rows, split, majority)
        for split in ("train", "validation", "extrapolation")
    }
    return SequenceBridgeRun(
        name="seq-token-majority",
        objective="token",
        feature_set="constant",
        train_token_accuracy=accuracies["train"],
        validation_token_accuracy=accuracies["validation"],
        extrapolation_token_accuracy=accuracies["extrapolation"],
        train_loss=deterministic_token_nll(accuracies["train"], token_bins),
        validation_loss=deterministic_token_nll(accuracies["validation"], token_bins),
        extrapolation_loss=deterministic_token_nll(accuracies["extrapolation"], token_bins),
    )


def evaluate_best_single_expert_sequence(
    rows: list[dict[str, Any]],
    expert_ids: tuple[str, ...],
    token_bins: int,
) -> SequenceBridgeRun:
    best_expert = max(expert_ids, key=lambda expert: expert_accuracy(rows, "train", expert))
    accuracies = {
        split: expert_accuracy(rows, split, best_expert)
        for split in ("train", "validation", "extrapolation")
    }
    return SequenceBridgeRun(
        name="seq-best-single-expert-by-train",
        objective="router-call",
        feature_set=f"global:{best_expert}",
        train_token_accuracy=accuracies["train"],
        validation_token_accuracy=accuracies["validation"],
        extrapolation_token_accuracy=accuracies["extrapolation"],
        train_loss=deterministic_token_nll(accuracies["train"], token_bins),
        validation_loss=deterministic_token_nll(accuracies["validation"], token_bins),
        extrapolation_loss=deterministic_token_nll(accuracies["extrapolation"], token_bins),
    )


def evaluate_oracle_router_sequence(rows: list[dict[str, Any]], token_bins: int) -> SequenceBridgeRun:
    accuracies = {
        split: oracle_accuracy(rows, split)
        for split in ("train", "validation", "extrapolation")
    }
    return SequenceBridgeRun(
        name="seq-oracle-row-router",
        objective="router-call",
        feature_set="target-leaking-oracle",
        train_token_accuracy=accuracies["train"],
        validation_token_accuracy=accuracies["validation"],
        extrapolation_token_accuracy=accuracies["extrapolation"],
        train_loss=deterministic_token_nll(accuracies["train"], token_bins),
        validation_loss=deterministic_token_nll(accuracies["validation"], token_bins),
        extrapolation_loss=deterministic_token_nll(accuracies["extrapolation"], token_bins),
        train_router_accuracy=1.0,
        validation_router_accuracy=1.0,
        extrapolation_router_accuracy=1.0,
    )


def train_sequence_token_model(
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
) -> SequenceBridgeRun:
    splits = build_sequence_splits(rows, task_ids, expert_ids, feature_set)
    model = SequenceTokenHead(torch, token_bins + 1, len(splits["train"].features[0][0]), hidden_units, token_bins).to(device)
    losses, predictions = train_sequence_classifier(
        torch,
        model,
        splits,
        classes=token_bins,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    accuracies = {
        split: sequence_indexed_accuracy(predictions[split], splits[split].target_tokens)
        for split in ("train", "validation", "extrapolation")
    }
    return SequenceBridgeRun(
        name=run_name,
        objective="token",
        feature_set=feature_set,
        train_token_accuracy=accuracies["train"],
        validation_token_accuracy=accuracies["validation"],
        extrapolation_token_accuracy=accuracies["extrapolation"],
        train_loss=losses["train"],
        validation_loss=losses["validation"],
        extrapolation_loss=losses["extrapolation"],
    )


def train_sequence_regression_model(
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
) -> SequenceBridgeRun:
    splits = build_sequence_splits(rows, task_ids, expert_ids, feature_set)
    model = SequenceRegressionHead(torch, token_bins + 1, len(splits["train"].features[0][0]), hidden_units).to(device)
    losses, predicted_values = train_sequence_regressor(
        torch,
        model,
        splits,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    accuracies = {
        split: continuous_token_accuracy(predicted_values[split], splits[split].rows)
        for split in ("train", "validation", "extrapolation")
    }
    rmses = {
        split: continuous_rmse_for_sequences(predicted_values[split], splits[split].target_values)
        for split in ("train", "validation", "extrapolation")
    }
    return SequenceBridgeRun(
        name=run_name,
        objective="continuous",
        feature_set=feature_set,
        train_token_accuracy=accuracies["train"],
        validation_token_accuracy=accuracies["validation"],
        extrapolation_token_accuracy=accuracies["extrapolation"],
        train_loss=losses["train"],
        validation_loss=losses["validation"],
        extrapolation_loss=losses["extrapolation"],
        train_rmse=rmses["train"],
        validation_rmse=rmses["validation"],
        extrapolation_rmse=rmses["extrapolation"],
    )


def train_sequence_router_model(
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
) -> SequenceBridgeRun:
    splits = build_sequence_splits(rows, task_ids, expert_ids, feature_set)
    model = SequenceTokenHead(torch, token_bins + 1, len(splits["train"].features[0][0]), hidden_units, len(expert_ids)).to(device)
    losses, predictions = train_sequence_classifier(
        torch,
        model,
        splits,
        classes=len(expert_ids),
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        label_kind="router",
    )
    token_accuracies = {
        split: sequence_routed_token_accuracy(predictions[split], splits[split].rows, expert_ids)
        for split in ("train", "validation", "extrapolation")
    }
    router_accuracies = {
        split: sequence_indexed_accuracy(predictions[split], splits[split].router_targets or [])
        for split in ("train", "validation", "extrapolation")
    }
    return SequenceBridgeRun(
        name=run_name,
        objective="router-call",
        feature_set=feature_set,
        train_token_accuracy=token_accuracies["train"],
        validation_token_accuracy=token_accuracies["validation"],
        extrapolation_token_accuracy=token_accuracies["extrapolation"],
        train_loss=losses["train"],
        validation_loss=losses["validation"],
        extrapolation_loss=losses["extrapolation"],
        train_router_accuracy=router_accuracies["train"],
        validation_router_accuracy=router_accuracies["validation"],
        extrapolation_router_accuracy=router_accuracies["extrapolation"],
    )


def train_sequence_classifier(
    torch: Any,
    model: Any,
    splits: dict[str, SequenceSplit],
    *,
    classes: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    device: Any,
    label_kind: str = "token",
) -> tuple[dict[str, float], dict[str, list[list[int]]]]:
    del classes
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    train_batch = tensorize_split(torch, splits["train"], device)
    train_labels = train_batch["target_tokens"] if label_kind == "token" else train_batch["router_targets"]
    if train_labels is None:
        raise ValueError("router training requested without router targets")
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_batch["previous_tokens"], train_batch["features"])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), train_labels.reshape(-1))
        loss.backward()
        optimizer.step()
    losses: dict[str, float] = {}
    predictions: dict[str, list[list[int]]] = {}
    model.eval()
    with torch.no_grad():
        for split, split_value in splits.items():
            batch = tensorize_split(torch, split_value, device)
            labels = batch["target_tokens"] if label_kind == "token" else batch["router_targets"]
            if labels is None:
                raise ValueError("router evaluation requested without router targets")
            logits = model(batch["previous_tokens"], batch["features"])
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            losses[split] = float(loss.detach().cpu().item())
            predictions[split] = nested_int_list(logits.argmax(dim=-1).detach().cpu().tolist())
    return losses, predictions


def train_sequence_regressor(
    torch: Any,
    model: Any,
    splits: dict[str, SequenceSplit],
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    device: Any,
) -> tuple[dict[str, float], dict[str, list[list[float]]]]:
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    train_batch = tensorize_split(torch, splits["train"], device)
    train_targets = normalized_target_tensor(torch, splits["train"], device)
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(train_batch["previous_tokens"], train_batch["features"]).squeeze(-1)
        loss = torch.nn.functional.mse_loss(predictions, train_targets)
        loss.backward()
        optimizer.step()
    losses: dict[str, float] = {}
    predicted_values: dict[str, list[list[float]]] = {}
    model.eval()
    with torch.no_grad():
        for split, split_value in splits.items():
            batch = tensorize_split(torch, split_value, device)
            targets = normalized_target_tensor(torch, split_value, device)
            normalized_predictions = model(batch["previous_tokens"], batch["features"]).squeeze(-1)
            loss = torch.nn.functional.mse_loss(normalized_predictions, targets)
            losses[split] = float(loss.detach().cpu().item())
            predicted_values[split] = denormalize_predictions(nested_float_list(normalized_predictions.detach().cpu().tolist()), split_value.rows)
    return losses, predicted_values


class SequenceTokenHead:
    def __init__(self, torch: Any, token_vocab: int, feature_dim: int, hidden_units: int, classes: int) -> None:
        self.module = torch.nn.Sequential()
        self.embedding = torch.nn.Embedding(token_vocab, hidden_units)
        self.feature_projection = torch.nn.Linear(feature_dim, hidden_units)
        self.gru = torch.nn.GRU(input_size=hidden_units * 2, hidden_size=hidden_units, batch_first=True)
        self.output = torch.nn.Linear(hidden_units, classes)

    def to(self, device: Any) -> "SequenceTokenHead":
        self.embedding.to(device)
        self.feature_projection.to(device)
        self.gru.to(device)
        self.output.to(device)
        return self

    def parameters(self) -> Any:
        return list(self.embedding.parameters()) + list(self.feature_projection.parameters()) + list(self.gru.parameters()) + list(self.output.parameters())

    def train(self) -> None:
        self.embedding.train()
        self.feature_projection.train()
        self.gru.train()
        self.output.train()

    def eval(self) -> None:
        self.embedding.eval()
        self.feature_projection.eval()
        self.gru.eval()
        self.output.eval()

    def __call__(self, previous_tokens: Any, features: Any) -> Any:
        token_features = self.embedding(previous_tokens)
        projected_features = self.feature_projection(features).tanh()
        sequence_input = torch_cat_like(token_features, projected_features)
        hidden, _state = self.gru(sequence_input)
        return self.output(hidden)


class SequenceRegressionHead(SequenceTokenHead):
    def __init__(self, torch: Any, token_vocab: int, feature_dim: int, hidden_units: int) -> None:
        super().__init__(torch, token_vocab, feature_dim, hidden_units, classes=1)


def torch_cat_like(left: Any, right: Any) -> Any:
    torch = import_torch()
    return torch.cat((left, right), dim=-1)


def build_sequence_splits(
    rows: list[dict[str, Any]],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    feature_set: str,
    *,
    expert_ids_for_router: tuple[str, ...] | None = None,
) -> dict[str, SequenceSplit]:
    del expert_ids_for_router
    grouped = group_rows(rows)
    train_feature_rows = [
        row
        for key, group in grouped.items()
        if key[2] == "train"
        for row in group
    ]
    feature_stats = feature_normalization_stats(train_feature_rows, task_ids, expert_ids, feature_set)
    expert_to_index = {expert: index for index, expert in enumerate(expert_ids)}
    splits: dict[str, SequenceSplit] = {}
    for split in ("train", "validation", "extrapolation"):
        sequence_rows = [group for key, group in sorted(grouped.items()) if key[2] == split]
        features = [
            [normalized_row_features(row, task_ids, expert_ids, feature_set, feature_stats) for row in sequence]
            for sequence in sequence_rows
        ]
        target_tokens = [[int(row["target_token"]) for row in sequence] for sequence in sequence_rows]
        previous_tokens = [[int(sequence[0]["token_bins"])] + sequence_tokens[:-1] for sequence, sequence_tokens in zip(sequence_rows, target_tokens)]
        router_targets = [[expert_to_index[str(row["best_expert_id"])] for row in sequence] for sequence in sequence_rows]
        splits[split] = SequenceSplit(
            rows=sequence_rows,
            previous_tokens=previous_tokens,
            target_tokens=target_tokens,
            target_values=[[float(row["target_y"]) for row in sequence] for sequence in sequence_rows],
            features=features,
            router_targets=router_targets,
        )
    return splits


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["task_id"]), int(row["seed"]), str(row["split"]))
        grouped.setdefault(key, []).append(row)
    return {key: sorted(group, key=lambda row: int(row["index"])) for key, group in grouped.items()}


def raw_row_features(
    row: dict[str, Any],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    feature_set: str,
) -> list[float]:
    task_to_index = {task: index for index, task in enumerate(task_ids)}
    features = [float(row["x"])]
    if feature_set in {"x-task", "side-channel", "expert-signal"}:
        features.extend(one_hot(task_to_index[str(row["task_id"])], len(task_ids)))
    if feature_set in {"side-channel", "expert-signal"}:
        features.extend(expert_signal_features(row, expert_ids))
    return features


def feature_normalization_stats(
    rows: list[dict[str, Any]],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    feature_set: str,
) -> tuple[list[float], list[float]]:
    raw = [raw_row_features(row, task_ids, expert_ids, feature_set) for row in rows]
    column_count = len(raw[0])
    means = []
    stds = []
    for column in range(column_count):
        values = [row[column] for row in raw]
        column_mean = mean(values)
        variance = mean((value - column_mean) ** 2 for value in values)
        means.append(column_mean)
        stds.append(math.sqrt(variance) if variance > 1.0e-12 else 1.0)
    return means, stds


def normalized_row_features(
    row: dict[str, Any],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    feature_set: str,
    stats: tuple[list[float], list[float]],
) -> list[float]:
    raw = raw_row_features(row, task_ids, expert_ids, feature_set)
    means, stds = stats
    return [(value - means[index]) / stds[index] for index, value in enumerate(raw)]


def tensorize_split(torch: Any, split: SequenceSplit, device: Any) -> dict[str, Any]:
    router_targets = None
    if split.router_targets is not None:
        router_targets = torch.tensor(split.router_targets, dtype=torch.long, device=device)
    return {
        "previous_tokens": torch.tensor(split.previous_tokens, dtype=torch.long, device=device),
        "target_tokens": torch.tensor(split.target_tokens, dtype=torch.long, device=device),
        "features": torch.tensor(split.features, dtype=torch.float32, device=device),
        "router_targets": router_targets,
    }


def normalized_target_tensor(torch: Any, split: SequenceSplit, device: Any) -> Any:
    values: list[list[float]] = []
    for sequence, targets in zip(split.rows, split.target_values):
        normalized_sequence = []
        for row, target in zip(sequence, targets):
            minimum = float(row["quantizer"]["minimum"])
            maximum = float(row["quantizer"]["maximum"])
            normalized_sequence.append((target - minimum) / max(maximum - minimum, 1.0e-12))
        values.append(normalized_sequence)
    return torch.tensor(values, dtype=torch.float32, device=device)


def denormalize_predictions(predictions: list[list[float]], rows: list[list[dict[str, Any]]]) -> list[list[float]]:
    denormalized: list[list[float]] = []
    for sequence_predictions, sequence_rows in zip(predictions, rows):
        denormalized_sequence = []
        for prediction, row in zip(sequence_predictions, sequence_rows):
            minimum = float(row["quantizer"]["minimum"])
            maximum = float(row["quantizer"]["maximum"])
            denormalized_sequence.append(minimum + prediction * (maximum - minimum))
        denormalized.append(denormalized_sequence)
    return denormalized


def nested_int_list(values: Any) -> list[list[int]]:
    return [[int(item) for item in row] for row in values]


def nested_float_list(values: Any) -> list[list[float]]:
    return [[float(item) for item in row] for row in values]


def sequence_indexed_accuracy(predictions: list[list[int]], targets: list[list[int]]) -> float:
    total = 0
    correct = 0
    for prediction_sequence, target_sequence in zip(predictions, targets):
        for prediction, target in zip(prediction_sequence, target_sequence):
            total += 1
            correct += int(prediction == target)
    return correct / total if total else 0.0


def sequence_routed_token_accuracy(
    predictions: list[list[int]],
    rows: list[list[dict[str, Any]]],
    expert_ids: tuple[str, ...],
) -> float:
    total = 0
    correct = 0
    for prediction_sequence, row_sequence in zip(predictions, rows):
        for prediction, row in zip(prediction_sequence, row_sequence):
            total += 1
            correct += int(expert_matches(row, expert_ids[prediction]))
    return correct / total if total else 0.0


def continuous_token_accuracy(predictions: list[list[float]], rows: list[list[dict[str, Any]]]) -> float:
    total = 0
    correct = 0
    for prediction_sequence, row_sequence in zip(predictions, rows):
        for prediction, row in zip(prediction_sequence, row_sequence):
            total += 1
            correct += int(encode_from_row(prediction, row) == int(row["target_token"]))
    return correct / total if total else 0.0


def continuous_rmse_for_sequences(predictions: list[list[float]], targets: list[list[float]]) -> float:
    errors = []
    for prediction_sequence, target_sequence in zip(predictions, targets):
        for prediction, target in zip(prediction_sequence, target_sequence):
            if math.isfinite(prediction) and math.isfinite(target):
                errors.append((prediction - target) ** 2)
            else:
                return math.inf
    return math.sqrt(sum(errors) / len(errors)) if errors else math.inf


def encode_from_row(value: float, row: dict[str, Any]) -> int:
    if not math.isfinite(value):
        return -1
    minimum = float(row["quantizer"]["minimum"])
    maximum = float(row["quantizer"]["maximum"])
    bins = int(row["token_bins"])
    fraction = (value - minimum) / max(maximum - minimum, 1.0e-12)
    index = int(math.floor(fraction * bins))
    return max(0, min(bins - 1, index))


def constant_token_accuracy(rows: list[dict[str, Any]], split: str, token: int) -> float:
    split_rows = [row for row in rows if row["split"] == split]
    return mean(1.0 if int(row["target_token"]) == token else 0.0 for row in split_rows)


def expert_accuracy(rows: list[dict[str, Any]], split: str, expert: str) -> float:
    split_rows = [row for row in rows if row["split"] == split]
    return mean(1.0 if expert_matches(row, expert) else 0.0 for row in split_rows)


def oracle_accuracy(rows: list[dict[str, Any]], split: str) -> float:
    split_rows = [row for row in rows if row["split"] == split]
    return mean(1.0 if row["oracle_has_safe_expert"] else 0.0 for row in split_rows)


def summarize_sequence_bridge(runs: list[SequenceBridgeRun], rows: list[dict[str, Any]]) -> dict[str, Any]:
    trained = [
        run
        for run in runs
        if run.name.startswith(("seq-token-", "seq-continuous-", "seq-router-"))
        and not run.name.startswith(("seq-token-majority", "seq-oracle"))
    ]
    router_runs = [
        run
        for run in runs
        if run.objective == "router-call"
        and run.train_router_accuracy is not None
        and run.name.startswith("seq-router-")
    ]
    side_continuous = next((run for run in runs if run.name.startswith("seq-continuous-side-channel")), None)
    base_continuous = next((run for run in runs if run.name.startswith("seq-continuous-x-task")), None)
    router_signal = next((run for run in runs if run.name.startswith("seq-router-expert-signal")), None)
    best_single = next((run for run in runs if run.name == "seq-best-single-expert-by-train"), None)
    return {
        "total_rows": len(rows),
        "sequence_count": len(group_rows(rows)),
        "safe_expert_coverage": {
            split: oracle_accuracy(rows, split)
            for split in ("train", "validation", "extrapolation")
        },
        "best_validation_token_accuracy": max(runs, key=lambda run: run.validation_token_accuracy).name,
        "best_extrapolation_token_accuracy": max(runs, key=lambda run: run.extrapolation_token_accuracy).name,
        "best_trained_extrapolation_token_accuracy": max(trained, key=lambda run: run.extrapolation_token_accuracy).name,
        "best_trained_router_extrapolation_accuracy": (
            max(router_runs, key=lambda run: run.extrapolation_router_accuracy or 0.0).name
            if router_runs
            else ""
        ),
        "continuous_side_channel_extrapolation_delta": (
            side_continuous.extrapolation_token_accuracy - base_continuous.extrapolation_token_accuracy
            if side_continuous is not None and base_continuous is not None
            else 0.0
        ),
        "router_expert_signal_extrapolation_delta": (
            router_signal.extrapolation_token_accuracy - best_single.extrapolation_token_accuracy
            if router_signal is not None and best_single is not None
            else 0.0
        ),
        "median_extrapolation_token_accuracy": median(run.extrapolation_token_accuracy for run in runs),
    }


def write_sequence_bridge_report(report: SequenceBridgeReport, output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    with (output_dir / "runs.jsonl").open("w") as handle:
        for run in report.runs:
            handle.write(json.dumps(run.to_dict(), sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(render_sequence_bridge_markdown(report))


def render_sequence_bridge_markdown(report: SequenceBridgeReport) -> str:
    lines = [
        f"# Symbolic Sequence Bridge: {report.run_label}",
        "",
        f"Bridge summary: `{report.bridge_summary_path}`",
        f"Feature table: `{report.feature_table_path}`",
        f"Token bins: `{report.token_bins}`",
        f"Device: `{report.device}`",
        f"Epochs: `{report.epochs}`",
        f"Learning rate: `{report.learning_rate}`",
        f"Hidden units: `{report.hidden_units}`",
        "",
        "| run | objective | features | train token acc | val token acc | extrap token acc | val loss | extrap loss | extrap RMSE | router val acc | router extrap acc |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
            f"{run.validation_loss:.4g} | "
            f"{run.extrapolation_loss:.4g} | "
            f"{format_optional(run.extrapolation_rmse)} | "
            f"{format_optional(run.validation_router_accuracy)} | "
            f"{format_optional(run.extrapolation_router_accuracy)} |"
        )
    safe = report.summary["safe_expert_coverage"]
    lines.extend(
        [
            "",
            f"Best validation token accuracy: `{report.summary['best_validation_token_accuracy']}`",
            f"Best extrapolation token accuracy: `{report.summary['best_extrapolation_token_accuracy']}`",
            f"Best trained extrapolation token accuracy: `{report.summary['best_trained_extrapolation_token_accuracy']}`",
            f"Best trained router extrapolation accuracy: `{report.summary['best_trained_router_extrapolation_accuracy']}`",
            f"Continuous side-channel extrapolation delta vs x-task: `{report.summary['continuous_side_channel_extrapolation_delta']:.3f}`",
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
