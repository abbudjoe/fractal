from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python.specs.common import repo_relative
from python.symbolic.bridge import deterministic_token_nll, mean
from python.symbolic.bridge_canary import (
    expert_signal_features,
    import_torch,
    load_feature_rows,
    one_hot,
    resolve_device,
    resolve_feature_table_path,
)


@dataclass(frozen=True)
class SymbolicBridgeBatch:
    previous_tokens: Any
    target_ids: Any
    x_values: Any
    expert_tokens: Any
    expert_values: Any
    expert_valid_mask: Any
    expert_safe_mask: Any
    router_targets: Any
    symbolic_mask: Any
    features: Any


@dataclass(frozen=True)
class BridgeLmRun:
    name: str
    mode: str
    feature_set: str
    train_final_token_accuracy: float
    validation_final_token_accuracy: float
    extrapolation_final_token_accuracy: float
    train_lm_token_accuracy: float
    validation_lm_token_accuracy: float
    extrapolation_lm_token_accuracy: float
    train_router_accuracy: float | None
    validation_router_accuracy: float | None
    extrapolation_router_accuracy: float | None
    train_expert_call_rate: float
    validation_expert_call_rate: float
    extrapolation_expert_call_rate: float
    train_unsafe_call_rate: float
    validation_unsafe_call_rate: float
    extrapolation_unsafe_call_rate: float
    train_abstain_recall: float | None
    validation_abstain_recall: float | None
    extrapolation_abstain_recall: float | None
    train_loss: float
    validation_loss: float
    extrapolation_loss: float
    train_final_nll: float
    validation_final_nll: float
    extrapolation_final_nll: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "feature_set": self.feature_set,
            "train_final_token_accuracy": self.train_final_token_accuracy,
            "validation_final_token_accuracy": self.validation_final_token_accuracy,
            "extrapolation_final_token_accuracy": self.extrapolation_final_token_accuracy,
            "train_lm_token_accuracy": self.train_lm_token_accuracy,
            "validation_lm_token_accuracy": self.validation_lm_token_accuracy,
            "extrapolation_lm_token_accuracy": self.extrapolation_lm_token_accuracy,
            "train_router_accuracy": self.train_router_accuracy,
            "validation_router_accuracy": self.validation_router_accuracy,
            "extrapolation_router_accuracy": self.extrapolation_router_accuracy,
            "train_expert_call_rate": self.train_expert_call_rate,
            "validation_expert_call_rate": self.validation_expert_call_rate,
            "extrapolation_expert_call_rate": self.extrapolation_expert_call_rate,
            "train_unsafe_call_rate": self.train_unsafe_call_rate,
            "validation_unsafe_call_rate": self.validation_unsafe_call_rate,
            "extrapolation_unsafe_call_rate": self.extrapolation_unsafe_call_rate,
            "train_abstain_recall": self.train_abstain_recall,
            "validation_abstain_recall": self.validation_abstain_recall,
            "extrapolation_abstain_recall": self.extrapolation_abstain_recall,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "extrapolation_loss": self.extrapolation_loss,
            "train_final_nll": self.train_final_nll,
            "validation_final_nll": self.validation_final_nll,
            "extrapolation_final_nll": self.extrapolation_final_nll,
        }


@dataclass(frozen=True)
class BridgeLmReport:
    bridge_summary_path: str
    feature_table_path: str
    run_label: str
    token_bins: int
    seed: int
    epochs: int
    learning_rate: float
    hidden_units: int
    router_loss_weight: float
    abstain_class_weight: float
    unsafe_call_loss_weight: float
    router_call_threshold: float
    device: str
    expert_ids: tuple[str, ...]
    abstain_index: int
    runs: tuple[BridgeLmRun, ...]
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
            "router_loss_weight": self.router_loss_weight,
            "abstain_class_weight": self.abstain_class_weight,
            "unsafe_call_loss_weight": self.unsafe_call_loss_weight,
            "router_call_threshold": self.router_call_threshold,
            "device": self.device,
            "expert_ids": list(self.expert_ids),
            "abstain_index": self.abstain_index,
            "runs": [run.to_dict() for run in self.runs],
            "summary": self.summary,
            "output_dir": self.output_dir,
        }


def run_symbolic_bridge_lm(
    bridge_summary_path: Path,
    output_dir: Path,
    *,
    run_label: str,
    seed: int = 777,
    epochs: int = 700,
    learning_rate: float = 0.003,
    hidden_units: int = 64,
    router_loss_weight: float = 0.5,
    abstain_class_weight: float = 1.0,
    unsafe_call_loss_weight: float = 0.0,
    router_call_threshold: float = 0.0,
    device: str = "auto",
) -> BridgeLmReport:
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
    abstain_index = len(expert_ids)

    runs: list[BridgeLmRun] = [
        evaluate_majority_baseline(rows, token_bins),
        evaluate_best_single_expert(rows, expert_ids, token_bins),
        evaluate_oracle_abstain_router(rows, expert_ids, token_bins),
    ]
    model_specs = (
        ("lm-token-only", "token-only", False),
        ("lm-x-task", "x-task", False),
        ("lm-frozen-side-channel", "side-channel", False),
        ("lm-router-hard-call", "side-channel", True),
    )
    for offset, (name, feature_set, use_router) in enumerate(model_specs):
        runs.append(
            train_lm_contract_variant(
                torch,
                rows,
                task_ids,
                expert_ids,
                token_bins,
                feature_set=feature_set,
                name=name,
                use_router=use_router,
                seed=seed + 11 + offset,
                epochs=epochs,
                learning_rate=learning_rate,
                hidden_units=hidden_units,
                router_loss_weight=router_loss_weight,
                abstain_class_weight=abstain_class_weight,
                unsafe_call_loss_weight=unsafe_call_loss_weight,
                router_call_threshold=router_call_threshold,
                device=selected_device,
            )
        )

    summary = summarize_lm_contract(runs, rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = BridgeLmReport(
        bridge_summary_path=repo_relative(bridge_summary_path),
        feature_table_path=repo_relative(feature_table_path),
        run_label=run_label,
        token_bins=token_bins,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        router_loss_weight=router_loss_weight,
        abstain_class_weight=abstain_class_weight,
        unsafe_call_loss_weight=unsafe_call_loss_weight,
        router_call_threshold=router_call_threshold,
        device=str(selected_device),
        expert_ids=expert_ids,
        abstain_index=abstain_index,
        runs=tuple(runs),
        summary=summary,
        output_dir=repo_relative(output_dir),
    )
    write_bridge_lm_report(report, output_dir)
    return report


def train_lm_contract_variant(
    torch: Any,
    rows: list[dict[str, Any]],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    token_bins: int,
    *,
    feature_set: str,
    name: str,
    use_router: bool,
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    router_loss_weight: float,
    abstain_class_weight: float,
    unsafe_call_loss_weight: float,
    router_call_threshold: float,
    device: Any,
) -> BridgeLmRun:
    torch.manual_seed(seed)
    splits = build_contract_splits(torch, rows, task_ids, expert_ids, token_bins, feature_set, device)
    feature_dim = int(splits["fit"].features.shape[-1])
    model = build_symbolic_bridge_lm_model(
        torch,
        token_vocab_size=token_bins + 1,
        feature_dim=feature_dim,
        hidden_units=hidden_units,
        token_bins=token_bins,
        router_classes=len(expert_ids) + 1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    fit_batch = splits["fit"]
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        token_logits, router_logits = model(fit_batch.previous_tokens, fit_batch.features)
        token_loss = torch.nn.functional.cross_entropy(
            token_logits.reshape(-1, token_logits.shape[-1]),
            fit_batch.target_ids.reshape(-1),
        )
        if use_router:
            router_class_weights = router_loss_class_weights(
                torch,
                classes=len(expert_ids) + 1,
                abstain_class_weight=abstain_class_weight,
                device=device,
            )
            router_loss = torch.nn.functional.cross_entropy(
                router_logits.reshape(-1, router_logits.shape[-1]),
                fit_batch.router_targets.reshape(-1),
                weight=router_class_weights,
            )
            unsafe_loss = router_unsafe_probability_loss(torch, router_logits, fit_batch.expert_safe_mask)
            loss = token_loss + router_loss_weight * router_loss + unsafe_call_loss_weight * unsafe_loss
        else:
            loss = token_loss
        loss.backward()
        optimizer.step()

    split_metrics = {}
    model.eval()
    with torch.no_grad():
        for split, batch in splits.items():
            token_logits, router_logits = model(batch.previous_tokens, batch.features)
            token_loss = torch.nn.functional.cross_entropy(
                token_logits.reshape(-1, token_logits.shape[-1]),
                batch.target_ids.reshape(-1),
            )
            router_loss_value = 0.0
            unsafe_loss_value = 0.0
            if use_router:
                router_class_weights = router_loss_class_weights(
                    torch,
                    classes=len(expert_ids) + 1,
                    abstain_class_weight=abstain_class_weight,
                    device=device,
                )
                router_loss = torch.nn.functional.cross_entropy(
                    router_logits.reshape(-1, router_logits.shape[-1]),
                    batch.router_targets.reshape(-1),
                    weight=router_class_weights,
                )
                router_loss_value = float(router_loss.detach().cpu().item())
                unsafe_loss = router_unsafe_probability_loss(torch, router_logits, batch.expert_safe_mask)
                unsafe_loss_value = float(unsafe_loss.detach().cpu().item())
            split_metrics[split] = evaluate_contract_outputs(
                torch,
                batch,
                token_logits,
                router_logits,
                expert_ids,
                token_bins,
                use_router=use_router,
                router_call_threshold=router_call_threshold,
                loss=(
                    float(token_loss.detach().cpu().item())
                    + router_loss_weight * router_loss_value
                    + unsafe_call_loss_weight * unsafe_loss_value
                ),
            )
    return BridgeLmRun(
        name=name,
        mode="hard-call" if use_router else "lm",
        feature_set=feature_set,
        train_final_token_accuracy=split_metrics["train"]["final_token_accuracy"],
        validation_final_token_accuracy=split_metrics["validation"]["final_token_accuracy"],
        extrapolation_final_token_accuracy=split_metrics["extrapolation"]["final_token_accuracy"],
        train_lm_token_accuracy=split_metrics["train"]["lm_token_accuracy"],
        validation_lm_token_accuracy=split_metrics["validation"]["lm_token_accuracy"],
        extrapolation_lm_token_accuracy=split_metrics["extrapolation"]["lm_token_accuracy"],
        train_router_accuracy=split_metrics["train"]["router_accuracy"],
        validation_router_accuracy=split_metrics["validation"]["router_accuracy"],
        extrapolation_router_accuracy=split_metrics["extrapolation"]["router_accuracy"],
        train_expert_call_rate=split_metrics["train"]["expert_call_rate"],
        validation_expert_call_rate=split_metrics["validation"]["expert_call_rate"],
        extrapolation_expert_call_rate=split_metrics["extrapolation"]["expert_call_rate"],
        train_unsafe_call_rate=split_metrics["train"]["unsafe_call_rate"],
        validation_unsafe_call_rate=split_metrics["validation"]["unsafe_call_rate"],
        extrapolation_unsafe_call_rate=split_metrics["extrapolation"]["unsafe_call_rate"],
        train_abstain_recall=split_metrics["train"]["abstain_recall"],
        validation_abstain_recall=split_metrics["validation"]["abstain_recall"],
        extrapolation_abstain_recall=split_metrics["extrapolation"]["abstain_recall"],
        train_loss=split_metrics["train"]["loss"],
        validation_loss=split_metrics["validation"]["loss"],
        extrapolation_loss=split_metrics["extrapolation"]["loss"],
        train_final_nll=split_metrics["train"]["final_nll"],
        validation_final_nll=split_metrics["validation"]["final_nll"],
        extrapolation_final_nll=split_metrics["extrapolation"]["final_nll"],
    )


def build_symbolic_bridge_lm_model(
    torch: Any,
    *,
    token_vocab_size: int,
    feature_dim: int,
    hidden_units: int,
    token_bins: int,
    router_classes: int,
) -> Any:
    class TinySymbolicBridgeLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = torch.nn.Embedding(token_vocab_size, hidden_units)
            self.feature_projection = torch.nn.Linear(feature_dim, hidden_units)
            self.gru = torch.nn.GRU(input_size=hidden_units * 2, hidden_size=hidden_units, batch_first=True)
            self.token_head = torch.nn.Linear(hidden_units, token_bins)
            self.router_head = torch.nn.Linear(hidden_units, router_classes)

        def forward(self, previous_tokens: Any, features: Any) -> tuple[Any, Any]:
            token_features = self.token_embedding(previous_tokens)
            side_features = self.feature_projection(features).tanh()
            hidden, _state = self.gru(torch.cat((token_features, side_features), dim=-1))
            return self.token_head(hidden), self.router_head(hidden)

    return TinySymbolicBridgeLM()


def router_loss_class_weights(
    torch: Any,
    *,
    classes: int,
    abstain_class_weight: float,
    device: Any,
) -> Any:
    weights = torch.ones(classes, dtype=torch.float32, device=device)
    weights[-1] = float(abstain_class_weight)
    return weights


def router_unsafe_probability_loss(torch: Any, router_logits: Any, expert_safe_mask: Any) -> Any:
    expert_count = int(expert_safe_mask.shape[-1])
    probabilities = torch.nn.functional.softmax(router_logits, dim=-1)[..., :expert_count]
    unsafe_mask = ~expert_safe_mask
    unsafe_mass = (probabilities * unsafe_mask.float()).sum(dim=-1)
    unsafe_mass = unsafe_mass.clamp(min=0.0, max=1.0 - 1.0e-6)
    return -torch.log1p(-unsafe_mass).mean()


def build_contract_splits(
    torch: Any,
    rows: list[dict[str, Any]],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    token_bins: int,
    feature_set: str,
    device: Any,
) -> dict[str, SymbolicBridgeBatch]:
    grouped = group_rows(rows)
    fit_rows = [
        row
        for key, sequence in grouped.items()
        if key[2] in {"train", "safety_calibration"}
        for row in sequence
    ]
    if not fit_rows:
        fit_rows = [row for key, sequence in grouped.items() if key[2] == "train" for row in sequence]
    stats = feature_normalization_stats(fit_rows, task_ids, expert_ids, feature_set)
    splits: dict[str, SymbolicBridgeBatch] = {}
    for split in ("train", "safety_calibration", "validation", "extrapolation"):
        sequences = [sequence for key, sequence in sorted(grouped.items()) if key[2] == split]
        if sequences:
            splits[split] = batch_from_sequences(
                torch,
                sequences,
                task_ids,
                expert_ids,
                token_bins,
                feature_set,
                stats,
                device,
            )
    fit_sequences = [
        sequence
        for key, sequence in sorted(grouped.items())
        if key[2] in {"train", "safety_calibration"}
    ]
    if not fit_sequences:
        fit_sequences = [sequence for key, sequence in sorted(grouped.items()) if key[2] == "train"]
    splits["fit"] = batch_from_sequences(
        torch,
        fit_sequences,
        task_ids,
        expert_ids,
        token_bins,
        feature_set,
        stats,
        device,
    )
    return splits


def batch_from_sequences(
    torch: Any,
    sequences: list[list[dict[str, Any]]],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    token_bins: int,
    feature_set: str,
    stats: tuple[list[float], list[float]],
    device: Any,
) -> SymbolicBridgeBatch:
    target_ids = [[int(row["target_token"]) for row in sequence] for sequence in sequences]
    previous_tokens = [[token_bins] + sequence[:-1] for sequence in target_ids]
    features = [
        [normalized_features(row, task_ids, expert_ids, feature_set, stats) for row in sequence]
        for sequence in sequences
    ]
    expert_tokens = [
        [[expert_token(row, expert) for expert in expert_ids] for row in sequence]
        for sequence in sequences
    ]
    expert_values = [
        [[expert_value(row, expert) for expert in expert_ids] for row in sequence]
        for sequence in sequences
    ]
    expert_valid_mask = [
        [[expert_valid(row, expert) for expert in expert_ids] for row in sequence]
        for sequence in sequences
    ]
    expert_safe_mask = [
        [[expert_safe(row, expert) for expert in expert_ids] for row in sequence]
        for sequence in sequences
    ]
    router_targets = [
        [router_target(row, expert_ids) for row in sequence]
        for sequence in sequences
    ]
    x_values = [[[float(row["x"])] for row in sequence] for sequence in sequences]
    symbolic_mask = [[True for _row in sequence] for sequence in sequences]
    return SymbolicBridgeBatch(
        previous_tokens=torch.tensor(previous_tokens, dtype=torch.long, device=device),
        target_ids=torch.tensor(target_ids, dtype=torch.long, device=device),
        x_values=torch.tensor(x_values, dtype=torch.float32, device=device),
        expert_tokens=torch.tensor(expert_tokens, dtype=torch.long, device=device),
        expert_values=torch.tensor(expert_values, dtype=torch.float32, device=device),
        expert_valid_mask=torch.tensor(expert_valid_mask, dtype=torch.bool, device=device),
        expert_safe_mask=torch.tensor(expert_safe_mask, dtype=torch.bool, device=device),
        router_targets=torch.tensor(router_targets, dtype=torch.long, device=device),
        symbolic_mask=torch.tensor(symbolic_mask, dtype=torch.bool, device=device),
        features=torch.tensor(features, dtype=torch.float32, device=device),
    )


def evaluate_contract_outputs(
    torch: Any,
    batch: SymbolicBridgeBatch,
    token_logits: Any,
    router_logits: Any,
    expert_ids: tuple[str, ...],
    token_bins: int,
    *,
    use_router: bool,
    router_call_threshold: float,
    loss: float,
) -> dict[str, float | None]:
    del expert_ids
    lm_predictions = token_logits.argmax(dim=-1)
    abstain_index = batch.expert_tokens.shape[-1]
    router_predictions = router_logits.argmax(dim=-1) if use_router else torch.full_like(batch.target_ids, abstain_index)
    final_predictions = lm_predictions.clone()
    expert_count = int(abstain_index)
    expert_call_mask = router_predictions < expert_count
    if use_router:
        router_probabilities = torch.nn.functional.softmax(router_logits, dim=-1)
        router_confidence = router_probabilities.gather(-1, router_predictions.unsqueeze(-1)).squeeze(-1)
        confident_call_mask = expert_call_mask & (router_confidence >= router_call_threshold)
        effective_router_predictions = torch.where(
            confident_call_mask | (router_predictions == expert_count),
            router_predictions,
            torch.full_like(router_predictions, expert_count),
        )
        selected = router_predictions.clamp(max=max(0, expert_count - 1)).unsqueeze(-1)
        selected_tokens = batch.expert_tokens.gather(-1, selected).squeeze(-1)
        selected_valid = batch.expert_valid_mask.gather(-1, selected).squeeze(-1)
        usable_call_mask = confident_call_mask & selected_valid
        final_predictions = torch.where(usable_call_mask, selected_tokens, lm_predictions)
        selected_safe = batch.expert_safe_mask.gather(-1, selected).squeeze(-1)
        unsafe_call_mask = usable_call_mask & ~selected_safe
    else:
        unsafe_call_mask = torch.zeros_like(batch.target_ids, dtype=torch.bool)
        effective_router_predictions = router_predictions
    total = batch.target_ids.numel()
    final_accuracy = float((final_predictions == batch.target_ids).float().mean().detach().cpu().item())
    lm_accuracy = float((lm_predictions == batch.target_ids).float().mean().detach().cpu().item())
    router_accuracy = None
    abstain_recall = None
    if use_router:
        router_accuracy = float((effective_router_predictions == batch.router_targets).float().mean().detach().cpu().item())
        abstain_target = batch.router_targets == expert_count
        if bool(abstain_target.any().detach().cpu().item()):
            abstain_prediction = effective_router_predictions == expert_count
            abstain_recall = float((abstain_prediction & abstain_target).float().sum().detach().cpu().item() / abstain_target.float().sum().detach().cpu().item())
    expert_call_rate = float(usable_call_mask.float().mean().detach().cpu().item()) if use_router else 0.0
    unsafe_call_rate = float(unsafe_call_mask.float().sum().detach().cpu().item() / total) if use_router else 0.0
    return {
        "final_token_accuracy": final_accuracy,
        "lm_token_accuracy": lm_accuracy,
        "router_accuracy": router_accuracy,
        "expert_call_rate": expert_call_rate,
        "unsafe_call_rate": unsafe_call_rate,
        "abstain_recall": abstain_recall,
        "loss": loss,
        "final_nll": deterministic_token_nll(final_accuracy, token_bins),
    }


def evaluate_majority_baseline(rows: list[dict[str, Any]], token_bins: int) -> BridgeLmRun:
    train_tokens = [int(row["target_token"]) for row in rows if row["split"] == "train"]
    majority = max(set(train_tokens), key=train_tokens.count)
    metrics = {split: constant_accuracy(rows, split, majority) for split in ("train", "validation", "extrapolation")}
    return baseline_run("lm-majority", "constant", metrics, token_bins)


def evaluate_best_single_expert(
    rows: list[dict[str, Any]],
    expert_ids: tuple[str, ...],
    token_bins: int,
) -> BridgeLmRun:
    best_expert = max(expert_ids, key=lambda expert: expert_accuracy(rows, "train", expert))
    metrics = {split: expert_accuracy(rows, split, best_expert) for split in ("train", "validation", "extrapolation")}
    return baseline_run(
        "lm-best-single-expert-by-train",
        f"global:{best_expert}",
        metrics,
        token_bins,
        mode="global-expert",
        train_expert_call_rate=1.0,
        validation_expert_call_rate=1.0,
        extrapolation_expert_call_rate=1.0,
        train_unsafe_call_rate=1.0 - metrics["train"],
        validation_unsafe_call_rate=1.0 - metrics["validation"],
        extrapolation_unsafe_call_rate=1.0 - metrics["extrapolation"],
    )


def evaluate_oracle_abstain_router(
    rows: list[dict[str, Any]],
    expert_ids: tuple[str, ...],
    token_bins: int,
) -> BridgeLmRun:
    del expert_ids
    metrics = {split: safe_coverage(rows, split) for split in ("train", "validation", "extrapolation")}
    return BridgeLmRun(
        name="lm-oracle-abstain-router",
        mode="oracle-hard-call",
        feature_set="target-leaking-oracle",
        train_final_token_accuracy=metrics["train"],
        validation_final_token_accuracy=metrics["validation"],
        extrapolation_final_token_accuracy=metrics["extrapolation"],
        train_lm_token_accuracy=0.0,
        validation_lm_token_accuracy=0.0,
        extrapolation_lm_token_accuracy=0.0,
        train_router_accuracy=1.0,
        validation_router_accuracy=1.0,
        extrapolation_router_accuracy=1.0,
        train_expert_call_rate=metrics["train"],
        validation_expert_call_rate=metrics["validation"],
        extrapolation_expert_call_rate=metrics["extrapolation"],
        train_unsafe_call_rate=0.0,
        validation_unsafe_call_rate=0.0,
        extrapolation_unsafe_call_rate=0.0,
        train_abstain_recall=1.0,
        validation_abstain_recall=1.0,
        extrapolation_abstain_recall=1.0,
        train_loss=deterministic_token_nll(metrics["train"], token_bins),
        validation_loss=deterministic_token_nll(metrics["validation"], token_bins),
        extrapolation_loss=deterministic_token_nll(metrics["extrapolation"], token_bins),
        train_final_nll=deterministic_token_nll(metrics["train"], token_bins),
        validation_final_nll=deterministic_token_nll(metrics["validation"], token_bins),
        extrapolation_final_nll=deterministic_token_nll(metrics["extrapolation"], token_bins),
    )


def baseline_run(
    name: str,
    feature_set: str,
    metrics: dict[str, float],
    token_bins: int,
    *,
    mode: str = "baseline",
    train_expert_call_rate: float = 0.0,
    validation_expert_call_rate: float = 0.0,
    extrapolation_expert_call_rate: float = 0.0,
    train_unsafe_call_rate: float = 0.0,
    validation_unsafe_call_rate: float = 0.0,
    extrapolation_unsafe_call_rate: float = 0.0,
) -> BridgeLmRun:
    return BridgeLmRun(
        name=name,
        mode=mode,
        feature_set=feature_set,
        train_final_token_accuracy=metrics["train"],
        validation_final_token_accuracy=metrics["validation"],
        extrapolation_final_token_accuracy=metrics["extrapolation"],
        train_lm_token_accuracy=metrics["train"],
        validation_lm_token_accuracy=metrics["validation"],
        extrapolation_lm_token_accuracy=metrics["extrapolation"],
        train_router_accuracy=None,
        validation_router_accuracy=None,
        extrapolation_router_accuracy=None,
        train_expert_call_rate=train_expert_call_rate,
        validation_expert_call_rate=validation_expert_call_rate,
        extrapolation_expert_call_rate=extrapolation_expert_call_rate,
        train_unsafe_call_rate=train_unsafe_call_rate,
        validation_unsafe_call_rate=validation_unsafe_call_rate,
        extrapolation_unsafe_call_rate=extrapolation_unsafe_call_rate,
        train_abstain_recall=None,
        validation_abstain_recall=None,
        extrapolation_abstain_recall=None,
        train_loss=deterministic_token_nll(metrics["train"], token_bins),
        validation_loss=deterministic_token_nll(metrics["validation"], token_bins),
        extrapolation_loss=deterministic_token_nll(metrics["extrapolation"], token_bins),
        train_final_nll=deterministic_token_nll(metrics["train"], token_bins),
        validation_final_nll=deterministic_token_nll(metrics["validation"], token_bins),
        extrapolation_final_nll=deterministic_token_nll(metrics["extrapolation"], token_bins),
    )


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task_id"]), int(row["seed"]), str(row["split"])), []).append(row)
    return {key: sorted(sequence, key=lambda row: int(row["index"])) for key, sequence in grouped.items()}


def raw_features(
    row: dict[str, Any],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    feature_set: str,
) -> list[float]:
    task_index = {task: index for index, task in enumerate(task_ids)}
    if feature_set == "token-only":
        return [0.0]
    features = [float(row["x"])]
    if feature_set in {"x-task", "side-channel"}:
        features.extend(one_hot(task_index[str(row["task_id"])], len(task_ids)))
    if feature_set == "side-channel":
        features.extend(expert_signal_features(row, expert_ids))
    return features


def feature_normalization_stats(
    rows: list[dict[str, Any]],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    feature_set: str,
) -> tuple[list[float], list[float]]:
    raw = [raw_features(row, task_ids, expert_ids, feature_set) for row in rows]
    columns = len(raw[0])
    means = []
    stds = []
    for column in range(columns):
        values = [row[column] for row in raw]
        column_mean = mean(values)
        variance = mean((value - column_mean) ** 2 for value in values)
        means.append(column_mean)
        stds.append(math.sqrt(variance) if variance > 1.0e-12 else 1.0)
    return means, stds


def normalized_features(
    row: dict[str, Any],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    feature_set: str,
    stats: tuple[list[float], list[float]],
) -> list[float]:
    raw = raw_features(row, task_ids, expert_ids, feature_set)
    means, stds = stats
    return [(value - means[index]) / stds[index] for index, value in enumerate(raw)]


def expert_token(row: dict[str, Any], expert: str) -> int:
    payload = row["experts"].get(expert, {})
    token = payload.get("token", -1)
    return int(token) if isinstance(token, int) else -1


def expert_value(row: dict[str, Any], expert: str) -> float:
    payload = row["experts"].get(expert, {})
    value = payload.get("prediction")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else 0.0


def expert_valid(row: dict[str, Any], expert: str) -> bool:
    payload = row["experts"].get(expert, {})
    return payload.get("prediction") is not None and int(payload.get("token", -1)) >= 0


def expert_safe(row: dict[str, Any], expert: str) -> bool:
    payload = row["experts"].get(expert, {})
    return bool(payload.get("token_match", False))


def router_target(row: dict[str, Any], expert_ids: tuple[str, ...]) -> int:
    safe = [
        (index, float(row["experts"][expert].get("abs_residual", 1.0e300)))
        for index, expert in enumerate(expert_ids)
        if expert_safe(row, expert)
    ]
    if not safe:
        return len(expert_ids)
    return min(safe, key=lambda item: (item[1], item[0]))[0]


def constant_accuracy(rows: list[dict[str, Any]], split: str, token: int) -> float:
    split_rows = [row for row in rows if row["split"] == split]
    return mean(1.0 if int(row["target_token"]) == token else 0.0 for row in split_rows)


def expert_accuracy(rows: list[dict[str, Any]], split: str, expert: str) -> float:
    split_rows = [row for row in rows if row["split"] == split]
    return mean(1.0 if expert_safe(row, expert) else 0.0 for row in split_rows)


def safe_coverage(rows: list[dict[str, Any]], split: str) -> float:
    split_rows = [row for row in rows if row["split"] == split]
    return mean(1.0 if bool(row["oracle_has_safe_expert"]) else 0.0 for row in split_rows)


def summarize_lm_contract(runs: list[BridgeLmRun], rows: list[dict[str, Any]]) -> dict[str, Any]:
    trained = [run for run in runs if run.mode in {"lm", "hard-call"}]
    router = next((run for run in runs if run.name == "lm-router-hard-call"), None)
    side = next((run for run in runs if run.name == "lm-frozen-side-channel"), None)
    x_task = next((run for run in runs if run.name == "lm-x-task"), None)
    token_only = next((run for run in runs if run.name == "lm-token-only"), None)
    best_single = next((run for run in runs if run.name == "lm-best-single-expert-by-train"), None)
    oracle = next((run for run in runs if run.name == "lm-oracle-abstain-router"), None)
    reported_splits = tuple(
        split for split in ("train", "safety_calibration", "validation", "extrapolation")
        if any(row["split"] == split for row in rows)
    )
    safe = {split: safe_coverage(rows, split) for split in reported_splits}
    abstain_target_rate = {split: 1.0 - value for split, value in safe.items()}
    fit_rows = [row for row in rows if row["split"] in {"train", "safety_calibration"}]
    fit_abstain_target_rate = mean(
        0.0 if row["oracle_has_safe_expert"] else 1.0
        for row in fit_rows
    ) if fit_rows else abstain_target_rate.get("train", 0.0)
    router_gain_vs_side = (
        router.extrapolation_final_token_accuracy - side.extrapolation_final_token_accuracy
        if router is not None and side is not None
        else 0.0
    )
    router_unsafe = 0.0 if router is None else router.extrapolation_unsafe_call_rate
    router_abstain_recall = None if router is None else router.extrapolation_abstain_recall
    has_extrap_abstain_targets = abstain_target_rate["extrapolation"] > 0.0
    safe_abstention_confirmed = (
        router_unsafe <= 0.01
        and (
            not has_extrap_abstain_targets
            or (router_abstain_recall is not None and router_abstain_recall >= 0.8)
        )
    )
    failure_modes = []
    if fit_abstain_target_rate <= 0.0 and abstain_target_rate["extrapolation"] > 0.0:
        failure_modes.append("fit split contains no abstain targets while extrapolation does")
    if router_unsafe > 0.01:
        failure_modes.append("router makes unsafe expert calls")
    return {
        "total_rows": len(rows),
        "sequence_count": len(group_rows(rows)),
        "safe_expert_coverage": safe,
        "abstain_target_rate": abstain_target_rate,
        "fit_abstain_target_rate": fit_abstain_target_rate,
        "best_validation_final_token_accuracy": max(runs, key=lambda run: run.validation_final_token_accuracy).name,
        "best_extrapolation_final_token_accuracy": max(runs, key=lambda run: run.extrapolation_final_token_accuracy).name,
        "best_trained_extrapolation_final_token_accuracy": max(trained, key=lambda run: run.extrapolation_final_token_accuracy).name,
        "router_contract_extrapolation_gain_vs_token_only": (
            router.extrapolation_final_token_accuracy - token_only.extrapolation_final_token_accuracy
            if router is not None and token_only is not None
            else 0.0
        ),
        "router_contract_extrapolation_gain_vs_x_task": (
            router.extrapolation_final_token_accuracy - x_task.extrapolation_final_token_accuracy
            if router is not None and x_task is not None
            else 0.0
        ),
        "router_contract_extrapolation_gain_vs_side_channel": router_gain_vs_side,
        "router_contract_gap_to_oracle": (
            oracle.extrapolation_final_token_accuracy - router.extrapolation_final_token_accuracy
            if router is not None and oracle is not None
            else 0.0
        ),
        "router_contract_gap_to_best_single_expert": (
            best_single.extrapolation_final_token_accuracy - router.extrapolation_final_token_accuracy
            if router is not None and best_single is not None
            else 0.0
        ),
        "router_contract_unsafe_call_rate": router_unsafe,
        "router_contract_abstain_recall": router_abstain_recall,
        "capability_gain_confirmed": router_gain_vs_side > 0.05,
        "safe_abstention_confirmed": safe_abstention_confirmed,
        "contract_confirmed": router_gain_vs_side > 0.05 and safe_abstention_confirmed,
        "failure_modes": failure_modes,
    }


def write_bridge_lm_report(report: BridgeLmReport, output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    with (output_dir / "runs.jsonl").open("w") as handle:
        for run in report.runs:
            handle.write(json.dumps(run.to_dict(), sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(render_bridge_lm_markdown(report))


def render_bridge_lm_markdown(report: BridgeLmReport) -> str:
    lines = [
        f"# Symbolic Bridge LM Contract: {report.run_label}",
        "",
        f"Bridge summary: `{report.bridge_summary_path}`",
        f"Feature table: `{report.feature_table_path}`",
        f"Token bins: `{report.token_bins}`",
        f"Experts: `{list(report.expert_ids)}`",
        f"Abstain index: `{report.abstain_index}`",
        f"Device: `{report.device}`",
        f"Epochs: `{report.epochs}`",
        f"Learning rate: `{report.learning_rate}`",
        f"Hidden units: `{report.hidden_units}`",
        f"Router loss weight: `{report.router_loss_weight}`",
        f"Abstain class weight: `{report.abstain_class_weight}`",
        f"Unsafe call loss weight: `{report.unsafe_call_loss_weight}`",
        f"Router call threshold: `{report.router_call_threshold}`",
        "",
        "| run | mode | features | val final acc | extrap final acc | extrap LM acc | router extrap acc | expert call rate | unsafe call rate | abstain recall |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report.runs:
        lines.append(
            "| "
            f"`{run.name}` | "
            f"{run.mode} | "
            f"{run.feature_set} | "
            f"{run.validation_final_token_accuracy:.3f} | "
            f"{run.extrapolation_final_token_accuracy:.3f} | "
            f"{run.extrapolation_lm_token_accuracy:.3f} | "
            f"{format_optional(run.extrapolation_router_accuracy)} | "
            f"{run.extrapolation_expert_call_rate:.3f} | "
            f"{run.extrapolation_unsafe_call_rate:.3f} | "
            f"{format_optional(run.extrapolation_abstain_recall)} |"
        )
    safe = report.summary["safe_expert_coverage"]
    abstain_rate = report.summary["abstain_target_rate"]
    lines.extend(
        [
            "",
            f"Best validation final token accuracy: `{report.summary['best_validation_final_token_accuracy']}`",
            f"Best extrapolation final token accuracy: `{report.summary['best_extrapolation_final_token_accuracy']}`",
            f"Best trained extrapolation final token accuracy: `{report.summary['best_trained_extrapolation_final_token_accuracy']}`",
            f"Router contract gain vs token-only extrapolation: `{report.summary['router_contract_extrapolation_gain_vs_token_only']:.3f}`",
            f"Router contract gain vs x-task extrapolation: `{report.summary['router_contract_extrapolation_gain_vs_x_task']:.3f}`",
            f"Router contract gain vs frozen side-channel extrapolation: `{report.summary['router_contract_extrapolation_gain_vs_side_channel']:.3f}`",
            f"Router contract gap to oracle: `{report.summary['router_contract_gap_to_oracle']:.3f}`",
            f"Router contract gap to best single expert: `{report.summary['router_contract_gap_to_best_single_expert']:.3f}`",
            f"Router contract unsafe call rate: `{report.summary['router_contract_unsafe_call_rate']:.3f}`",
            f"Router contract abstain recall: `{format_optional(report.summary['router_contract_abstain_recall'])}`",
            f"Capability gain confirmed: `{report.summary['capability_gain_confirmed']}`",
            f"Safe abstention confirmed: `{report.summary['safe_abstention_confirmed']}`",
            f"Contract confirmed: `{report.summary['contract_confirmed']}`",
            f"Safe-expert coverage by split: `train={safe['train']:.3f}, validation={safe['validation']:.3f}, extrapolation={safe['extrapolation']:.3f}`",
            f"Safety-calibration safe-expert coverage: `{format_optional(safe.get('safety_calibration'))}`",
            f"Abstain target rate by split: `train={abstain_rate['train']:.3f}, safety_calibration={format_optional(abstain_rate.get('safety_calibration'))}, validation={abstain_rate['validation']:.3f}, extrapolation={abstain_rate['extrapolation']:.3f}`",
            f"Fit abstain target rate: `{report.summary['fit_abstain_target_rate']:.3f}`",
            f"Failure modes: `{report.summary['failure_modes']}`",
            "",
        ]
    )
    return "\n".join(lines)


def format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"
