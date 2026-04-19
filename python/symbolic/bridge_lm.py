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
    role_ids: Any
    role_names: tuple[str, ...]


@dataclass(frozen=True)
class FusionCalibrationPolicy:
    answer_roles: tuple[str, ...]
    temperature: float
    abstain_bias: float
    answer_call_threshold: float
    answer_fusion_cap: float
    non_answer_fusion_cap: float
    target_answer_unsafe: float
    min_answer_accuracy_gain: float
    selected_split: str
    selected_feasible: bool
    selected_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_roles": list(self.answer_roles),
            "temperature": self.temperature,
            "abstain_bias": self.abstain_bias,
            "answer_call_threshold": self.answer_call_threshold,
            "answer_fusion_cap": self.answer_fusion_cap,
            "non_answer_fusion_cap": self.non_answer_fusion_cap,
            "target_answer_unsafe": self.target_answer_unsafe,
            "min_answer_accuracy_gain": self.min_answer_accuracy_gain,
            "selected_split": self.selected_split,
            "selected_feasible": self.selected_feasible,
            "selected_metrics": self.selected_metrics,
        }


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
    role_metrics: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
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
        if self.role_metrics is not None:
            payload["role_metrics"] = self.role_metrics
        if self.calibration is not None:
            payload["calibration"] = self.calibration
        return payload


@dataclass(frozen=True)
class BridgeLmReport:
    bridge_summary_path: str
    feature_table_path: str
    extra_fit_bridge_summary_paths: tuple[str, ...]
    extra_fit_feature_table_paths: tuple[str, ...]
    run_label: str
    backbone: str
    backbone_config: dict[str, Any]
    token_bins: int
    seed: int
    epochs: int
    learning_rate: float
    hidden_units: int
    router_loss_weight: float
    abstain_class_weight: float
    unsafe_call_loss_weight: float
    call_abstain_loss_weight: float
    answer_call_abstain_loss_weight: float
    answer_unsafe_loss_weight: float
    non_answer_abstain_loss_weight: float
    non_answer_lm_retention_loss_weight: float
    non_answer_teacher_kl_loss_weight: float
    non_answer_teacher_kl_roles: tuple[str, ...]
    role_aware_calibration: bool
    calibration_target_answer_unsafe: float
    calibration_min_answer_accuracy_gain: float
    calibration_answer_roles: tuple[str, ...]
    unsafe_margin_loss_weight: float
    unsafe_margin: float
    router_call_threshold: float
    expert_logit_scale: float
    fusion_allowed_roles: tuple[str, ...]
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
            "extra_fit_bridge_summary_paths": list(self.extra_fit_bridge_summary_paths),
            "extra_fit_feature_table_paths": list(self.extra_fit_feature_table_paths),
            "run_label": self.run_label,
            "backbone": self.backbone,
            "backbone_config": self.backbone_config,
            "token_bins": self.token_bins,
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden_units": self.hidden_units,
            "router_loss_weight": self.router_loss_weight,
            "abstain_class_weight": self.abstain_class_weight,
            "unsafe_call_loss_weight": self.unsafe_call_loss_weight,
            "call_abstain_loss_weight": self.call_abstain_loss_weight,
            "answer_call_abstain_loss_weight": self.answer_call_abstain_loss_weight,
            "answer_unsafe_loss_weight": self.answer_unsafe_loss_weight,
            "non_answer_abstain_loss_weight": self.non_answer_abstain_loss_weight,
            "non_answer_lm_retention_loss_weight": self.non_answer_lm_retention_loss_weight,
            "non_answer_teacher_kl_loss_weight": self.non_answer_teacher_kl_loss_weight,
            "non_answer_teacher_kl_roles": list(self.non_answer_teacher_kl_roles),
            "role_aware_calibration": self.role_aware_calibration,
            "calibration_target_answer_unsafe": self.calibration_target_answer_unsafe,
            "calibration_min_answer_accuracy_gain": self.calibration_min_answer_accuracy_gain,
            "calibration_answer_roles": list(self.calibration_answer_roles),
            "unsafe_margin_loss_weight": self.unsafe_margin_loss_weight,
            "unsafe_margin": self.unsafe_margin,
            "router_call_threshold": self.router_call_threshold,
            "expert_logit_scale": self.expert_logit_scale,
            "fusion_allowed_roles": list(self.fusion_allowed_roles),
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
    call_abstain_loss_weight: float = 0.0,
    answer_call_abstain_loss_weight: float = 0.0,
    answer_unsafe_loss_weight: float = 0.0,
    non_answer_abstain_loss_weight: float = 0.0,
    non_answer_lm_retention_loss_weight: float = 0.0,
    non_answer_teacher_kl_loss_weight: float = 0.0,
    non_answer_teacher_kl_roles: tuple[str, ...] = ("prose", "math_context"),
    role_aware_calibration: bool = False,
    calibration_target_answer_unsafe: float = 0.05,
    calibration_min_answer_accuracy_gain: float = 0.01,
    calibration_answer_roles: tuple[str, ...] = ("math_answer", "math_only"),
    unsafe_margin_loss_weight: float = 0.0,
    unsafe_margin: float = 0.5,
    router_call_threshold: float = 0.0,
    expert_logit_scale: float = 6.0,
    fusion_allowed_roles: tuple[str, ...] = (),
    backbone: str = "gru",
    transformer_layers: int = 2,
    transformer_heads: int = 4,
    transformer_ffn_multiplier: int = 2,
    device: str = "auto",
    extra_fit_bridge_summary_paths: tuple[Path, ...] = (),
) -> BridgeLmReport:
    torch = import_torch()
    validate_bridge_lm_backbone(
        backbone,
        hidden_units=hidden_units,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_ffn_multiplier=transformer_ffn_multiplier,
    )
    bridge_summary = json.loads(bridge_summary_path.read_text())
    feature_table_path = resolve_feature_table_path(bridge_summary, bridge_summary_path)
    rows = load_feature_rows(feature_table_path)
    if not rows:
        raise ValueError(f"bridge feature table has no rows: {feature_table_path}")
    extra_fit_rows, extra_fit_feature_table_paths = load_extra_fit_rows(
        bridge_summary,
        bridge_summary_path,
        extra_fit_bridge_summary_paths,
    )
    fit_rows = rows + extra_fit_rows

    random.seed(seed)
    torch.manual_seed(seed)
    selected_device = resolve_device(torch, device)
    token_bins = int(bridge_summary.get("token_bins") or rows[0]["token_bins"])
    task_ids = tuple(sorted({str(row["task_id"]) for row in fit_rows}))
    expert_ids = tuple(sorted({str(expert) for row in fit_rows for expert in row["experts"]}))
    abstain_index = len(expert_ids)
    teacher_probabilities_by_split = (
        train_token_only_teacher_probabilities(
            torch,
            rows,
            task_ids,
            expert_ids,
            token_bins,
            fit_rows=fit_rows,
            seed=seed + 100_003,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            backbone=backbone,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ffn_multiplier=transformer_ffn_multiplier,
            device=selected_device,
        )
        if non_answer_teacher_kl_loss_weight > 0.0
        else None
    )

    runs: list[BridgeLmRun] = [
        evaluate_majority_baseline(rows, token_bins),
        evaluate_best_single_expert(rows, expert_ids, token_bins),
        evaluate_oracle_abstain_router(rows, expert_ids, token_bins),
    ]
    model_specs = (
        ("lm-token-only", "token-only", False, "none"),
        ("lm-x-task", "x-task", False, "none"),
        ("lm-frozen-side-channel", "side-channel", False, "none"),
        ("lm-router-hard-call", "side-channel", True, "hard-call"),
        ("lm-router-logit-fusion", "side-channel", True, "logit-fusion"),
        ("lm-router-prob-mixture", "side-channel", True, "prob-mixture"),
    )
    for offset, (name, feature_set, use_router, fusion_mode) in enumerate(model_specs):
        runs.append(
            train_lm_contract_variant(
                torch,
                rows,
                task_ids,
                expert_ids,
                token_bins,
                fit_rows=fit_rows,
                feature_set=feature_set,
                name=name,
                use_router=use_router,
                fusion_mode=fusion_mode,
                seed=seed + 11 + offset,
                epochs=epochs,
                learning_rate=learning_rate,
                hidden_units=hidden_units,
                router_loss_weight=router_loss_weight,
                abstain_class_weight=abstain_class_weight,
                unsafe_call_loss_weight=unsafe_call_loss_weight,
                call_abstain_loss_weight=call_abstain_loss_weight,
                answer_call_abstain_loss_weight=answer_call_abstain_loss_weight,
                answer_unsafe_loss_weight=answer_unsafe_loss_weight,
                non_answer_abstain_loss_weight=non_answer_abstain_loss_weight,
                non_answer_lm_retention_loss_weight=non_answer_lm_retention_loss_weight,
                non_answer_teacher_kl_loss_weight=non_answer_teacher_kl_loss_weight,
                non_answer_teacher_kl_roles=non_answer_teacher_kl_roles,
                role_aware_calibration=role_aware_calibration,
                calibration_target_answer_unsafe=calibration_target_answer_unsafe,
                calibration_min_answer_accuracy_gain=calibration_min_answer_accuracy_gain,
                calibration_answer_roles=calibration_answer_roles,
                unsafe_margin_loss_weight=unsafe_margin_loss_weight,
                unsafe_margin=unsafe_margin,
                router_call_threshold=router_call_threshold,
                expert_logit_scale=expert_logit_scale,
                fusion_allowed_roles=fusion_allowed_roles,
                teacher_probabilities_by_split=teacher_probabilities_by_split,
                backbone=backbone,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                transformer_ffn_multiplier=transformer_ffn_multiplier,
                device=selected_device,
            )
        )

    summary = summarize_lm_contract(runs, rows, fit_rows=fit_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = BridgeLmReport(
        bridge_summary_path=repo_relative(bridge_summary_path),
        feature_table_path=repo_relative(feature_table_path),
        extra_fit_bridge_summary_paths=tuple(repo_relative(path) for path in extra_fit_bridge_summary_paths),
        extra_fit_feature_table_paths=tuple(repo_relative(path) for path in extra_fit_feature_table_paths),
        run_label=run_label,
        backbone=backbone,
        backbone_config=bridge_lm_backbone_config(
            backbone,
            hidden_units=hidden_units,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ffn_multiplier=transformer_ffn_multiplier,
        ),
        token_bins=token_bins,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        router_loss_weight=router_loss_weight,
        abstain_class_weight=abstain_class_weight,
        unsafe_call_loss_weight=unsafe_call_loss_weight,
        call_abstain_loss_weight=call_abstain_loss_weight,
        answer_call_abstain_loss_weight=answer_call_abstain_loss_weight,
        answer_unsafe_loss_weight=answer_unsafe_loss_weight,
        non_answer_abstain_loss_weight=non_answer_abstain_loss_weight,
        non_answer_lm_retention_loss_weight=non_answer_lm_retention_loss_weight,
        non_answer_teacher_kl_loss_weight=non_answer_teacher_kl_loss_weight,
        non_answer_teacher_kl_roles=non_answer_teacher_kl_roles,
        role_aware_calibration=role_aware_calibration,
        calibration_target_answer_unsafe=calibration_target_answer_unsafe,
        calibration_min_answer_accuracy_gain=calibration_min_answer_accuracy_gain,
        calibration_answer_roles=calibration_answer_roles,
        unsafe_margin_loss_weight=unsafe_margin_loss_weight,
        unsafe_margin=unsafe_margin,
        router_call_threshold=router_call_threshold,
        expert_logit_scale=expert_logit_scale,
        fusion_allowed_roles=fusion_allowed_roles,
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
    fit_rows: list[dict[str, Any]] | None = None,
    feature_set: str,
    name: str,
    use_router: bool,
    fusion_mode: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    router_loss_weight: float,
    abstain_class_weight: float,
    unsafe_call_loss_weight: float,
    call_abstain_loss_weight: float,
    answer_call_abstain_loss_weight: float,
    answer_unsafe_loss_weight: float,
    non_answer_abstain_loss_weight: float,
    non_answer_lm_retention_loss_weight: float,
    non_answer_teacher_kl_loss_weight: float,
    non_answer_teacher_kl_roles: tuple[str, ...],
    role_aware_calibration: bool,
    calibration_target_answer_unsafe: float,
    calibration_min_answer_accuracy_gain: float,
    calibration_answer_roles: tuple[str, ...],
    unsafe_margin_loss_weight: float,
    unsafe_margin: float,
    router_call_threshold: float,
    expert_logit_scale: float,
    fusion_allowed_roles: tuple[str, ...],
    teacher_probabilities_by_split: dict[str, Any] | None,
    backbone: str,
    transformer_layers: int,
    transformer_heads: int,
    transformer_ffn_multiplier: int,
    device: Any,
) -> BridgeLmRun:
    torch.manual_seed(seed)
    splits = build_contract_splits(
        torch,
        rows,
        task_ids,
        expert_ids,
        token_bins,
        feature_set,
        device,
        fit_rows=fit_rows,
    )
    feature_dim = int(splits["fit"].features.shape[-1])
    max_sequence_length = max(int(batch.previous_tokens.shape[1]) for batch in splits.values())
    model = build_symbolic_bridge_lm_model(
        torch,
        backbone=backbone,
        token_vocab_size=token_bins + 1,
        feature_dim=feature_dim,
        hidden_units=hidden_units,
        token_bins=token_bins,
        router_classes=len(expert_ids) + 1,
        max_sequence_length=max_sequence_length,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_ffn_multiplier=transformer_ffn_multiplier,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    fit_batch = splits["fit"]
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        token_logits, router_logits = model(fit_batch.previous_tokens, fit_batch.features)
        final_logits, final_probabilities, _router_stats = fused_token_outputs(
            torch,
            fit_batch,
            token_logits,
            router_logits,
            token_bins,
            fusion_mode=fusion_mode,
            expert_logit_scale=expert_logit_scale,
            fusion_allowed_roles=fusion_allowed_roles,
        )
        token_loss = masked_token_nll(
            torch,
            final_logits,
            final_probabilities,
            fit_batch.target_ids,
            fit_batch.symbolic_mask,
        )
        if use_router:
            router_class_weights = router_loss_class_weights(
                torch,
                classes=len(expert_ids) + 1,
                abstain_class_weight=abstain_class_weight,
                device=device,
            )
            router_loss = masked_cross_entropy(
                torch,
                router_logits,
                fit_batch.router_targets,
                fit_batch.symbolic_mask,
                weight=router_class_weights,
            )
            unsafe_loss = router_unsafe_probability_loss(
                torch,
                router_logits,
                fit_batch.expert_safe_mask,
                mask=fit_batch.symbolic_mask,
            )
            call_abstain_loss = router_call_abstain_loss(
                torch,
                router_logits,
                fit_batch.expert_safe_mask,
                abstain_weight=abstain_class_weight,
                mask=fit_batch.symbolic_mask,
            )
            answer_call_abstain_loss = router_call_abstain_loss(
                torch,
                router_logits,
                fit_batch.expert_safe_mask,
                abstain_weight=abstain_class_weight,
                mask=role_mask(torch, fit_batch, ("math_answer", "math_only")),
            )
            unsafe_margin_loss = router_unsafe_margin_loss(
                torch,
                router_logits,
                fit_batch.expert_safe_mask,
                margin=unsafe_margin,
                mask=fit_batch.symbolic_mask,
            )
            answer_unsafe_loss = router_unsafe_probability_loss(
                torch,
                router_logits,
                fit_batch.expert_safe_mask,
                mask=role_mask(torch, fit_batch, ("math_answer", "math_only")),
            )
            non_answer_abstain_loss = router_abstain_probability_loss(
                torch,
                router_logits,
                mask=role_mask(torch, fit_batch, ("prose", "math_context")),
            )
            non_answer_lm_retention_loss = masked_token_nll(
                torch,
                token_logits,
                None,
                fit_batch.target_ids,
                role_mask(torch, fit_batch, ("prose", "math_context")),
            )
            non_answer_teacher_kl_loss = teacher_kl_loss_for_split(
                torch,
                fit_batch,
                final_logits,
                final_probabilities,
                teacher_probabilities_by_split,
                split="fit",
                roles=non_answer_teacher_kl_roles,
            )
            loss = (
                token_loss
                + router_loss_weight * router_loss
                + unsafe_call_loss_weight * unsafe_loss
                + call_abstain_loss_weight * call_abstain_loss
                + answer_call_abstain_loss_weight * answer_call_abstain_loss
                + answer_unsafe_loss_weight * answer_unsafe_loss
                + non_answer_abstain_loss_weight * non_answer_abstain_loss
                + non_answer_lm_retention_loss_weight * non_answer_lm_retention_loss
                + non_answer_teacher_kl_loss_weight * non_answer_teacher_kl_loss
                + unsafe_margin_loss_weight * unsafe_margin_loss
            )
        else:
            loss = token_loss
        loss.backward()
        optimizer.step()

    calibration_policy = None
    model.eval()
    with torch.no_grad():
        if role_aware_calibration and use_router and fusion_mode == "prob-mixture":
            if "fit_safety_calibration" in splits:
                calibration_split = "fit_safety_calibration"
            else:
                calibration_split = "safety_calibration" if "safety_calibration" in splits else "validation"
            calibration_batch = splits[calibration_split]
            calibration_token_logits, calibration_router_logits = model(
                calibration_batch.previous_tokens,
                calibration_batch.features,
            )
            capability_batch = splits.get("validation")
            capability_token_logits = None
            capability_router_logits = None
            if capability_batch is not None:
                capability_token_logits, capability_router_logits = model(
                    capability_batch.previous_tokens,
                    capability_batch.features,
                )
            calibration_policy = fit_role_aware_fusion_calibration(
                torch,
                calibration_batch,
                calibration_token_logits,
                calibration_router_logits,
                token_bins,
                capability_batch=capability_batch,
                capability_token_logits=capability_token_logits,
                capability_router_logits=capability_router_logits,
                expert_logit_scale=expert_logit_scale,
                fusion_allowed_roles=fusion_allowed_roles,
                answer_roles=calibration_answer_roles,
                target_answer_unsafe=calibration_target_answer_unsafe,
                min_answer_accuracy_gain=calibration_min_answer_accuracy_gain,
                selected_split=calibration_split,
            )
        split_metrics = {}
        for split, batch in splits.items():
            token_logits, router_logits = model(batch.previous_tokens, batch.features)
            final_logits, final_probabilities, router_stats = fused_token_outputs(
                torch,
                batch,
                token_logits,
                router_logits,
                token_bins,
                fusion_mode=fusion_mode,
                expert_logit_scale=expert_logit_scale,
                fusion_allowed_roles=fusion_allowed_roles,
                calibration_policy=calibration_policy,
            )
            token_loss = masked_token_nll(
                torch,
                final_logits,
                final_probabilities,
                batch.target_ids,
                batch.symbolic_mask,
            )
            router_loss_value = 0.0
            unsafe_loss_value = 0.0
            call_abstain_loss_value = 0.0
            answer_call_abstain_loss_value = 0.0
            answer_unsafe_loss_value = 0.0
            non_answer_abstain_loss_value = 0.0
            non_answer_lm_retention_loss_value = 0.0
            non_answer_teacher_kl_loss_value = 0.0
            unsafe_margin_loss_value = 0.0
            if use_router:
                router_class_weights = router_loss_class_weights(
                    torch,
                    classes=len(expert_ids) + 1,
                    abstain_class_weight=abstain_class_weight,
                    device=device,
                )
                router_loss = masked_cross_entropy(
                    torch,
                    router_logits,
                    batch.router_targets,
                    batch.symbolic_mask,
                    weight=router_class_weights,
                )
                router_loss_value = float(router_loss.detach().cpu().item())
                unsafe_loss = router_unsafe_probability_loss(
                    torch,
                    router_logits,
                    batch.expert_safe_mask,
                    mask=batch.symbolic_mask,
                )
                unsafe_loss_value = float(unsafe_loss.detach().cpu().item())
                call_abstain_loss = router_call_abstain_loss(
                    torch,
                    router_logits,
                    batch.expert_safe_mask,
                    abstain_weight=abstain_class_weight,
                    mask=batch.symbolic_mask,
                )
                call_abstain_loss_value = float(call_abstain_loss.detach().cpu().item())
                answer_call_abstain_loss = router_call_abstain_loss(
                    torch,
                    router_logits,
                    batch.expert_safe_mask,
                    abstain_weight=abstain_class_weight,
                    mask=role_mask(torch, batch, ("math_answer", "math_only")),
                )
                answer_call_abstain_loss_value = float(answer_call_abstain_loss.detach().cpu().item())
                unsafe_margin_loss = router_unsafe_margin_loss(
                    torch,
                    router_logits,
                    batch.expert_safe_mask,
                    margin=unsafe_margin,
                    mask=batch.symbolic_mask,
                )
                unsafe_margin_loss_value = float(unsafe_margin_loss.detach().cpu().item())
                answer_unsafe_loss = router_unsafe_probability_loss(
                    torch,
                    router_logits,
                    batch.expert_safe_mask,
                    mask=role_mask(torch, batch, ("math_answer", "math_only")),
                )
                answer_unsafe_loss_value = float(answer_unsafe_loss.detach().cpu().item())
                non_answer_abstain_loss = router_abstain_probability_loss(
                    torch,
                    router_logits,
                    mask=role_mask(torch, batch, ("prose", "math_context")),
                )
                non_answer_abstain_loss_value = float(non_answer_abstain_loss.detach().cpu().item())
                non_answer_lm_retention_loss = masked_token_nll(
                    torch,
                    token_logits,
                    None,
                    batch.target_ids,
                    role_mask(torch, batch, ("prose", "math_context")),
                )
                non_answer_lm_retention_loss_value = float(non_answer_lm_retention_loss.detach().cpu().item())
                non_answer_teacher_kl_loss = teacher_kl_loss_for_split(
                    torch,
                    batch,
                    final_logits,
                    final_probabilities,
                    teacher_probabilities_by_split,
                    split=split,
                    roles=non_answer_teacher_kl_roles,
                )
                non_answer_teacher_kl_loss_value = float(non_answer_teacher_kl_loss.detach().cpu().item())
            split_metrics[split] = evaluate_contract_outputs(
                torch,
                batch,
                token_logits,
                router_logits,
                expert_ids,
                token_bins,
                use_router=use_router,
                fusion_mode=fusion_mode,
                final_logits=final_logits,
                final_probabilities=final_probabilities,
                router_stats=router_stats,
                router_call_threshold=router_call_threshold,
                loss=(
                    float(token_loss.detach().cpu().item())
                    + router_loss_weight * router_loss_value
                    + unsafe_call_loss_weight * unsafe_loss_value
                    + call_abstain_loss_weight * call_abstain_loss_value
                    + answer_call_abstain_loss_weight * answer_call_abstain_loss_value
                    + answer_unsafe_loss_weight * answer_unsafe_loss_value
                    + non_answer_abstain_loss_weight * non_answer_abstain_loss_value
                    + non_answer_lm_retention_loss_weight * non_answer_lm_retention_loss_value
                    + non_answer_teacher_kl_loss_weight * non_answer_teacher_kl_loss_value
                    + unsafe_margin_loss_weight * unsafe_margin_loss_value
                ),
            )
    return BridgeLmRun(
        name=name,
        mode=fusion_mode if use_router else "lm",
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
        role_metrics={
            split: split_metrics[split].get("role_metrics", {})
            for split in ("train", "safety_calibration", "validation", "extrapolation")
            if split in split_metrics
        },
        calibration=None if calibration_policy is None else calibration_policy.to_dict(),
    )


def train_token_only_teacher_probabilities(
    torch: Any,
    rows: list[dict[str, Any]],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    token_bins: int,
    *,
    fit_rows: list[dict[str, Any]],
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    backbone: str,
    transformer_layers: int,
    transformer_heads: int,
    transformer_ffn_multiplier: int,
    device: Any,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    splits = build_contract_splits(
        torch,
        rows,
        task_ids,
        expert_ids,
        token_bins,
        "token-only",
        device,
        fit_rows=fit_rows,
    )
    feature_dim = int(splits["fit"].features.shape[-1])
    max_sequence_length = max(int(batch.previous_tokens.shape[1]) for batch in splits.values())
    model = build_symbolic_bridge_lm_model(
        torch,
        backbone=backbone,
        token_vocab_size=token_bins + 1,
        feature_dim=feature_dim,
        hidden_units=hidden_units,
        token_bins=token_bins,
        router_classes=len(expert_ids) + 1,
        max_sequence_length=max_sequence_length,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_ffn_multiplier=transformer_ffn_multiplier,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    fit_batch = splits["fit"]
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        token_logits, _router_logits = model(fit_batch.previous_tokens, fit_batch.features)
        loss = masked_token_nll(
            torch,
            token_logits,
            None,
            fit_batch.target_ids,
            fit_batch.symbolic_mask,
        )
        loss.backward()
        optimizer.step()

    model.eval()
    probabilities: dict[str, Any] = {}
    with torch.no_grad():
        for split, batch in splits.items():
            token_logits, _router_logits = model(batch.previous_tokens, batch.features)
            probabilities[split] = torch.nn.functional.softmax(token_logits, dim=-1).detach()
    return probabilities


def fused_token_outputs(
    torch: Any,
    batch: SymbolicBridgeBatch,
    token_logits: Any,
    router_logits: Any,
    token_bins: int,
    *,
    fusion_mode: str,
    expert_logit_scale: float,
    fusion_allowed_roles: tuple[str, ...] = (),
    calibration_policy: FusionCalibrationPolicy | None = None,
) -> tuple[Any | None, Any | None, dict[str, Any]]:
    if fusion_mode in {"none", "hard-call"}:
        return token_logits, None, {}
    if fusion_mode not in {"logit-fusion", "prob-mixture"}:
        raise ValueError("fusion_mode must be one of none|hard-call|logit-fusion|prob-mixture")
    role_gate = fusion_role_gate(torch, batch, fusion_allowed_roles)
    if calibration_policy is None:
        router_probabilities = torch.nn.functional.softmax(router_logits, dim=-1)
    else:
        calibrated_logits = router_logits / float(calibration_policy.temperature)
        if calibration_policy.abstain_bias:
            abstain_adjustment = torch.zeros_like(calibrated_logits)
            abstain_adjustment[..., -1] = float(calibration_policy.abstain_bias)
            calibrated_logits = calibrated_logits + abstain_adjustment
        router_probabilities = torch.nn.functional.softmax(calibrated_logits, dim=-1)
    expert_count = int(batch.expert_tokens.shape[-1])
    expert_probabilities = router_probabilities[..., :expert_count]
    valid_mask = batch.expert_valid_mask & (batch.expert_tokens >= 0)
    valid_probabilities = expert_probabilities * valid_mask.float() * role_gate.unsqueeze(-1).float()
    unsafe_probabilities = expert_probabilities * (~batch.expert_safe_mask).float() * role_gate.unsqueeze(-1).float()
    if calibration_policy is not None:
        answer_gate = fusion_role_gate(torch, batch, calibration_policy.answer_roles)
        raw_expert_mass = valid_probabilities.sum(dim=-1)
        answer_allowed = answer_gate & (raw_expert_mass >= float(calibration_policy.answer_call_threshold))
        non_answer_allowed = (~answer_gate) & (float(calibration_policy.non_answer_fusion_cap) > 0.0)
        allowed = (answer_allowed | non_answer_allowed) & batch.symbolic_mask
        cap = torch.where(
            answer_gate,
            torch.full_like(raw_expert_mass, float(calibration_policy.answer_fusion_cap)),
            torch.full_like(raw_expert_mass, float(calibration_policy.non_answer_fusion_cap)),
        )
        capped_mass = torch.minimum(raw_expert_mass, cap).clamp(min=0.0, max=1.0)
        scale = torch.where(
            raw_expert_mass > 0.0,
            capped_mass / raw_expert_mass.clamp(min=1.0e-8),
            torch.zeros_like(raw_expert_mass),
        )
        scale = scale * allowed.float()
        valid_probabilities = valid_probabilities * scale.unsqueeze(-1)
        unsafe_probabilities = unsafe_probabilities * scale.unsqueeze(-1)
    token_indices = batch.expert_tokens.clamp(min=0, max=max(0, token_bins - 1))
    expert_token_mass = torch.zeros(
        (*batch.target_ids.shape, token_bins),
        dtype=token_logits.dtype,
        device=token_logits.device,
    )
    expert_token_mass.scatter_add_(-1, token_indices, valid_probabilities.to(token_logits.dtype))
    expert_mass = expert_token_mass.sum(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)
    unsafe_mass = unsafe_probabilities.sum(dim=-1)
    stats = {
        "expert_mass": expert_mass.squeeze(-1),
        "unsafe_mass": unsafe_mass,
    }
    if fusion_mode == "logit-fusion":
        return token_logits + float(expert_logit_scale) * expert_token_mass, None, stats
    lm_probabilities = torch.nn.functional.softmax(token_logits, dim=-1)
    final_probabilities = (1.0 - expert_mass) * lm_probabilities + expert_token_mass
    final_probabilities = final_probabilities / final_probabilities.sum(dim=-1, keepdim=True).clamp(min=1.0e-8)
    return None, final_probabilities, stats


def fusion_role_gate(torch: Any, batch: SymbolicBridgeBatch, allowed_roles: tuple[str, ...]) -> Any:
    if not allowed_roles:
        return torch.ones_like(batch.symbolic_mask, dtype=torch.bool)
    selected = torch.zeros_like(batch.symbolic_mask, dtype=torch.bool)
    for role in allowed_roles:
        if role in batch.role_names:
            selected = selected | (batch.role_ids == batch.role_names.index(role))
    return selected & batch.symbolic_mask


def fit_role_aware_fusion_calibration(
    torch: Any,
    batch: SymbolicBridgeBatch,
    token_logits: Any,
    router_logits: Any,
    token_bins: int,
    *,
    capability_batch: SymbolicBridgeBatch | None = None,
    capability_token_logits: Any | None = None,
    capability_router_logits: Any | None = None,
    expert_logit_scale: float,
    fusion_allowed_roles: tuple[str, ...],
    answer_roles: tuple[str, ...],
    target_answer_unsafe: float,
    min_answer_accuracy_gain: float,
    selected_split: str,
) -> FusionCalibrationPolicy:
    candidates: list[tuple[bool, tuple[float, ...], FusionCalibrationPolicy]] = []
    temperatures = (0.75, 1.0, 1.25, 1.5, 2.0)
    abstain_biases = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)
    thresholds = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999)
    caps = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
    for temperature in temperatures:
        for abstain_bias in abstain_biases:
            for answer_call_threshold in thresholds:
                for answer_fusion_cap in caps:
                    policy = FusionCalibrationPolicy(
                        answer_roles=answer_roles,
                        temperature=temperature,
                        abstain_bias=abstain_bias,
                        answer_call_threshold=answer_call_threshold,
                        answer_fusion_cap=answer_fusion_cap,
                        non_answer_fusion_cap=0.0,
                        target_answer_unsafe=target_answer_unsafe,
                        min_answer_accuracy_gain=min_answer_accuracy_gain,
                        selected_split=selected_split,
                        selected_feasible=False,
                        selected_metrics={},
                    )
                    final_logits, final_probabilities, router_stats = fused_token_outputs(
                        torch,
                        batch,
                        token_logits,
                        router_logits,
                        token_bins,
                        fusion_mode="prob-mixture",
                        expert_logit_scale=expert_logit_scale,
                        fusion_allowed_roles=fusion_allowed_roles,
                        calibration_policy=policy,
                    )
                    metrics = evaluate_contract_outputs(
                        torch,
                        batch,
                        token_logits,
                        router_logits,
                        (),
                        token_bins,
                        use_router=True,
                        fusion_mode="prob-mixture",
                        final_logits=final_logits,
                        final_probabilities=final_probabilities,
                        router_stats=router_stats,
                        router_call_threshold=0.0,
                        loss=0.0,
                    )
                    answer_metrics = first_role_metrics(
                        metrics.get("role_metrics", {}),
                        answer_roles,
                    )
                    answer_accuracy = float(answer_metrics.get("final_token_accuracy", 0.0))
                    answer_nll = float(answer_metrics.get("final_nll", float("inf")))
                    answer_unsafe = float(answer_metrics.get("unsafe_call_rate", 1.0))
                    answer_lm_accuracy = float(answer_metrics.get("lm_token_accuracy", 0.0))
                    answer_gain = answer_accuracy - answer_lm_accuracy
                    capability_accuracy = answer_accuracy
                    capability_nll = answer_nll
                    capability_lm_accuracy = answer_lm_accuracy
                    capability_gain = answer_gain
                    if (
                        capability_batch is not None
                        and capability_token_logits is not None
                        and capability_router_logits is not None
                    ):
                        capability_final_logits, capability_final_probabilities, capability_router_stats = fused_token_outputs(
                            torch,
                            capability_batch,
                            capability_token_logits,
                            capability_router_logits,
                            token_bins,
                            fusion_mode="prob-mixture",
                            expert_logit_scale=expert_logit_scale,
                            fusion_allowed_roles=fusion_allowed_roles,
                            calibration_policy=policy,
                        )
                        capability_metrics = evaluate_contract_outputs(
                            torch,
                            capability_batch,
                            capability_token_logits,
                            capability_router_logits,
                            (),
                            token_bins,
                            use_router=True,
                            fusion_mode="prob-mixture",
                            final_logits=capability_final_logits,
                            final_probabilities=capability_final_probabilities,
                            router_stats=capability_router_stats,
                            router_call_threshold=0.0,
                            loss=0.0,
                        )
                        capability_answer_metrics = first_role_metrics(
                            capability_metrics.get("role_metrics", {}),
                            answer_roles,
                        )
                        capability_accuracy = float(capability_answer_metrics.get("final_token_accuracy", 0.0))
                        capability_nll = float(capability_answer_metrics.get("final_nll", float("inf")))
                        capability_lm_accuracy = float(capability_answer_metrics.get("lm_token_accuracy", 0.0))
                        capability_gain = capability_accuracy - capability_lm_accuracy
                    feasible = (
                        answer_unsafe <= target_answer_unsafe
                        and capability_gain >= min_answer_accuracy_gain
                    )
                    selected_metrics = {
                        "answer_accuracy": capability_accuracy,
                        "answer_lm_accuracy": capability_lm_accuracy,
                        "answer_accuracy_gain_vs_lm": capability_gain,
                        "answer_nll": capability_nll,
                        "answer_unsafe": answer_unsafe,
                        "calibration_answer_accuracy": answer_accuracy,
                        "calibration_answer_lm_accuracy": answer_lm_accuracy,
                        "calibration_answer_accuracy_gain_vs_lm": answer_gain,
                        "calibration_answer_nll": answer_nll,
                        "calibration_answer_unsafe": answer_unsafe,
                        "whole_accuracy": float(metrics["final_token_accuracy"]),
                        "whole_nll": float(metrics["final_nll"]),
                    }
                    final_policy = FusionCalibrationPolicy(
                        answer_roles=answer_roles,
                        temperature=temperature,
                        abstain_bias=abstain_bias,
                        answer_call_threshold=answer_call_threshold,
                        answer_fusion_cap=answer_fusion_cap,
                        non_answer_fusion_cap=0.0,
                        target_answer_unsafe=target_answer_unsafe,
                        min_answer_accuracy_gain=min_answer_accuracy_gain,
                        selected_split=selected_split,
                        selected_feasible=feasible,
                        selected_metrics=selected_metrics,
                    )
                    if feasible:
                        score = (answer_unsafe, -capability_accuracy, capability_nll, -capability_gain)
                    else:
                        score = (answer_unsafe, -capability_accuracy, capability_nll, -capability_gain)
                    candidates.append((feasible, score, final_policy))
    feasible_candidates = [candidate for candidate in candidates if candidate[0]]
    candidate_pool = feasible_candidates or candidates
    return min(candidate_pool, key=lambda candidate: candidate[1])[2]


def first_role_metrics(role_metrics: dict[str, Any], roles: tuple[str, ...]) -> dict[str, float]:
    for role in roles:
        metrics = role_metrics.get(role)
        if metrics is not None:
            return metrics
    return {}


def build_symbolic_bridge_lm_model(
    torch: Any,
    *,
    backbone: str,
    token_vocab_size: int,
    feature_dim: int,
    hidden_units: int,
    token_bins: int,
    router_classes: int,
    max_sequence_length: int,
    transformer_layers: int,
    transformer_heads: int,
    transformer_ffn_multiplier: int,
) -> Any:
    if backbone == "transformer":
        return build_symbolic_bridge_transformer_model(
            torch,
            token_vocab_size=token_vocab_size,
            feature_dim=feature_dim,
            hidden_units=hidden_units,
            token_bins=token_bins,
            router_classes=router_classes,
            max_sequence_length=max_sequence_length,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ffn_multiplier=transformer_ffn_multiplier,
        )
    if backbone != "gru":
        raise ValueError("backbone must be one of gru|transformer")

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


def build_symbolic_bridge_transformer_model(
    torch: Any,
    *,
    token_vocab_size: int,
    feature_dim: int,
    hidden_units: int,
    token_bins: int,
    router_classes: int,
    max_sequence_length: int,
    transformer_layers: int,
    transformer_heads: int,
    transformer_ffn_multiplier: int,
) -> Any:
    class TinyCausalTransformerBridgeLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = torch.nn.Embedding(token_vocab_size, hidden_units)
            self.position_embedding = torch.nn.Embedding(max_sequence_length, hidden_units)
            self.feature_projection = torch.nn.Linear(feature_dim, hidden_units, bias=False)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_units,
                nhead=transformer_heads,
                dim_feedforward=hidden_units * transformer_ffn_multiplier,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self.final_norm = torch.nn.LayerNorm(hidden_units)
            self.token_head = torch.nn.Linear(hidden_units, token_bins)
            self.router_head = torch.nn.Linear(hidden_units, router_classes)

        def forward(self, previous_tokens: Any, features: Any) -> tuple[Any, Any]:
            sequence_length = int(previous_tokens.shape[1])
            if sequence_length > max_sequence_length:
                raise ValueError(
                    f"sequence length {sequence_length} exceeds transformer limit {max_sequence_length}"
                )
            positions = torch.arange(sequence_length, device=previous_tokens.device).unsqueeze(0)
            hidden = (
                self.token_embedding(previous_tokens)
                + self.position_embedding(positions)
                + torch.tanh(self.feature_projection(features))
            )
            causal_mask = torch.triu(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    dtype=torch.bool,
                    device=previous_tokens.device,
                ),
                diagonal=1,
            )
            hidden = self.transformer(hidden, mask=causal_mask)
            hidden = self.final_norm(hidden)
            return self.token_head(hidden), self.router_head(hidden)

    return TinyCausalTransformerBridgeLM()


def validate_bridge_lm_backbone(
    backbone: str,
    *,
    hidden_units: int,
    transformer_layers: int,
    transformer_heads: int,
    transformer_ffn_multiplier: int,
) -> None:
    if backbone not in {"gru", "transformer"}:
        raise ValueError("backbone must be one of gru|transformer")
    if hidden_units <= 0:
        raise ValueError("hidden_units must be positive")
    if transformer_layers <= 0:
        raise ValueError("transformer_layers must be positive")
    if transformer_heads <= 0:
        raise ValueError("transformer_heads must be positive")
    if transformer_ffn_multiplier <= 0:
        raise ValueError("transformer_ffn_multiplier must be positive")
    if backbone == "transformer" and hidden_units % transformer_heads != 0:
        raise ValueError("hidden_units must be divisible by transformer_heads")


def bridge_lm_backbone_config(
    backbone: str,
    *,
    hidden_units: int,
    transformer_layers: int,
    transformer_heads: int,
    transformer_ffn_multiplier: int,
) -> dict[str, Any]:
    if backbone == "transformer":
        return {
            "type": "decoder-only-causal-transformer",
            "hidden_units": hidden_units,
            "layers": transformer_layers,
            "heads": transformer_heads,
            "ffn_multiplier": transformer_ffn_multiplier,
            "dropout": 0.0,
            "feature_projection_bias": False,
        }
    return {
        "type": "gru",
        "hidden_units": hidden_units,
        "feature_projection_bias": True,
    }


def load_extra_fit_rows(
    primary_summary: dict[str, Any],
    primary_summary_path: Path,
    extra_fit_bridge_summary_paths: tuple[Path, ...],
) -> tuple[list[dict[str, Any]], tuple[Path, ...]]:
    if not extra_fit_bridge_summary_paths:
        return [], ()
    primary_token_bins = int(primary_summary.get("token_bins") or 0)
    primary_vocabulary = primary_summary.get("summary", {}).get("vocabulary")
    extra_rows: list[dict[str, Any]] = []
    feature_table_paths: list[Path] = []
    for source_index, summary_path in enumerate(extra_fit_bridge_summary_paths):
        summary = json.loads(summary_path.read_text())
        token_bins = int(summary.get("token_bins") or 0)
        if token_bins != primary_token_bins:
            raise ValueError(
                f"extra fit bridge summary token_bins mismatch: {summary_path} has {token_bins}, "
                f"primary {primary_summary_path} has {primary_token_bins}"
            )
        vocabulary = summary.get("summary", {}).get("vocabulary")
        if isinstance(primary_vocabulary, list) and isinstance(vocabulary, list) and vocabulary != primary_vocabulary:
            raise ValueError(
                f"extra fit bridge summary vocabulary mismatch: {summary_path} does not match {primary_summary_path}"
            )
        feature_table_path = resolve_feature_table_path(summary, summary_path)
        feature_table_paths.append(feature_table_path)
        source_rows = load_feature_rows(feature_table_path)
        extra_rows.extend(
            namespace_extra_fit_row(row, source_index=source_index)
            for row in source_rows
            if str(row["split"]) in {"train", "safety_calibration"}
        )
    return extra_rows, tuple(feature_table_paths)


def namespace_extra_fit_row(row: dict[str, Any], *, source_index: int) -> dict[str, Any]:
    row_copy = dict(row)
    sequence_id = str(row_copy.get("sequence_id", "default"))
    row_copy["sequence_id"] = f"extra-fit-{source_index}:{sequence_id}"
    return row_copy


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


def masked_cross_entropy(
    torch: Any,
    logits: Any,
    targets: Any,
    mask: Any,
    *,
    weight: Any | None = None,
) -> Any:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    if bool(flat_mask.any().detach().cpu().item()):
        return torch.nn.functional.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask], weight=weight)
    return flat_logits.sum() * 0.0


def masked_token_nll(
    torch: Any,
    logits: Any | None,
    probabilities: Any | None,
    targets: Any,
    mask: Any,
) -> Any:
    if probabilities is None:
        if logits is None:
            raise ValueError("masked_token_nll requires logits or probabilities")
        return masked_cross_entropy(torch, logits, targets, mask)
    selected = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1).clamp(min=1.0e-8)
    losses = -torch.log(selected)
    return masked_mean(torch, losses, mask)


def masked_teacher_kl(
    torch: Any,
    teacher_probabilities: Any,
    logits: Any | None,
    probabilities: Any | None,
    mask: Any,
) -> Any:
    teacher = teacher_probabilities.detach().clamp(min=1.0e-8)
    if probabilities is None:
        if logits is None:
            raise ValueError("masked_teacher_kl requires logits or probabilities")
        student_log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    else:
        student_log_probabilities = torch.log(probabilities.clamp(min=1.0e-8))
    losses = (teacher * (torch.log(teacher) - student_log_probabilities)).sum(dim=-1)
    return masked_mean(torch, losses, mask)


def teacher_kl_loss_for_split(
    torch: Any,
    batch: SymbolicBridgeBatch,
    final_logits: Any | None,
    final_probabilities: Any | None,
    teacher_probabilities_by_split: dict[str, Any] | None,
    *,
    split: str,
    roles: tuple[str, ...],
) -> Any:
    if teacher_probabilities_by_split is None or split not in teacher_probabilities_by_split:
        source = final_logits if final_logits is not None else final_probabilities
        return source.sum() * 0.0
    teacher_probabilities = teacher_probabilities_by_split[split]
    if tuple(teacher_probabilities.shape[:2]) != tuple(batch.target_ids.shape):
        raise ValueError(
            f"teacher probability shape mismatch for {split}: "
            f"{tuple(teacher_probabilities.shape[:2])} vs {tuple(batch.target_ids.shape)}"
        )
    return masked_teacher_kl(
        torch,
        teacher_probabilities,
        final_logits,
        final_probabilities,
        role_mask(torch, batch, roles),
    )


def masked_mean(torch: Any, values: Any, mask: Any | None) -> Any:
    if mask is None:
        return values.mean()
    weights = mask.float()
    return (values * weights).sum() / weights.sum().clamp(min=1.0)


def router_unsafe_probability_loss(torch: Any, router_logits: Any, expert_safe_mask: Any, *, mask: Any | None = None) -> Any:
    expert_count = int(expert_safe_mask.shape[-1])
    probabilities = torch.nn.functional.softmax(router_logits, dim=-1)[..., :expert_count]
    unsafe_mask = ~expert_safe_mask
    unsafe_mass = (probabilities * unsafe_mask.float()).sum(dim=-1)
    unsafe_mass = unsafe_mass.clamp(min=0.0, max=1.0 - 1.0e-6)
    return masked_mean(torch, -torch.log1p(-unsafe_mass), mask)


def router_call_abstain_loss(
    torch: Any,
    router_logits: Any,
    expert_safe_mask: Any,
    *,
    abstain_weight: float,
    mask: Any | None = None,
) -> Any:
    expert_count = int(expert_safe_mask.shape[-1])
    probabilities = torch.nn.functional.softmax(router_logits, dim=-1)
    call_probability = probabilities[..., :expert_count].sum(dim=-1).clamp(min=1.0e-6, max=1.0 - 1.0e-6)
    call_target = expert_safe_mask.any(dim=-1).float()
    weights = torch.where(
        call_target > 0.5,
        torch.ones_like(call_target),
        torch.full_like(call_target, float(abstain_weight)),
    )
    loss = torch.nn.functional.binary_cross_entropy(call_probability, call_target, weight=weights, reduction="none")
    return masked_mean(torch, loss, mask)


def router_abstain_probability_loss(torch: Any, router_logits: Any, *, mask: Any | None = None) -> Any:
    abstain_probability = torch.nn.functional.softmax(router_logits, dim=-1)[..., -1].clamp(
        min=1.0e-6,
        max=1.0,
    )
    return masked_mean(torch, -torch.log(abstain_probability), mask)


def role_mask(torch: Any, batch: SymbolicBridgeBatch, roles: tuple[str, ...]) -> Any:
    selected = torch.zeros_like(batch.symbolic_mask, dtype=torch.bool)
    for role in roles:
        if role in batch.role_names:
            selected = selected | (batch.role_ids == batch.role_names.index(role))
    return selected & batch.symbolic_mask


def router_unsafe_margin_loss(
    torch: Any,
    router_logits: Any,
    expert_safe_mask: Any,
    *,
    margin: float,
    mask: Any | None = None,
) -> Any:
    expert_count = int(expert_safe_mask.shape[-1])
    expert_logits = router_logits[..., :expert_count]
    abstain_logits = router_logits[..., expert_count]
    safe_mask = expert_safe_mask
    unsafe_mask = ~safe_mask
    has_safe = safe_mask.any(dim=-1)
    has_unsafe = unsafe_mask.any(dim=-1)
    very_negative = torch.finfo(router_logits.dtype).min / 4.0
    safe_logits = expert_logits.masked_fill(~safe_mask, very_negative)
    unsafe_logits = expert_logits.masked_fill(~unsafe_mask, very_negative)
    safe_route_logits = torch.logsumexp(safe_logits, dim=-1)
    unsafe_route_logits = torch.logsumexp(unsafe_logits, dim=-1)
    allowed_route_logits = torch.where(has_safe, safe_route_logits, abstain_logits)
    margin_loss = torch.nn.functional.softplus(unsafe_route_logits - allowed_route_logits + float(margin))
    if mask is not None:
        has_unsafe = has_unsafe & mask
    if bool(has_unsafe.any().detach().cpu().item()):
        return margin_loss[has_unsafe].mean()
    return margin_loss.mean() * 0.0


def build_contract_splits(
    torch: Any,
    rows: list[dict[str, Any]],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    token_bins: int,
    feature_set: str,
    device: Any,
    *,
    fit_rows: list[dict[str, Any]] | None = None,
) -> dict[str, SymbolicBridgeBatch]:
    grouped = group_rows(rows)
    fit_source_rows = rows if fit_rows is None else fit_rows
    fit_grouped = group_rows(fit_source_rows)
    role_names = tuple(
        sorted({str(row.get("eval_role", "symbolic")) for row in rows + fit_source_rows})
    )
    fit_stat_rows = [
        row
        for key, sequence in fit_grouped.items()
        if key[2] in {"train", "safety_calibration"}
        for row in sequence
    ]
    if not fit_stat_rows:
        fit_stat_rows = [row for key, sequence in fit_grouped.items() if key[2] == "train" for row in sequence]
    stats = feature_normalization_stats(fit_stat_rows, task_ids, expert_ids, feature_set)
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
                role_names,
                device,
            )
    fit_safety_sequences = [
        sequence
        for key, sequence in sorted(fit_grouped.items())
        if key[2] == "safety_calibration"
    ]
    if fit_safety_sequences:
        splits["fit_safety_calibration"] = batch_from_sequences(
            torch,
            fit_safety_sequences,
            task_ids,
            expert_ids,
            token_bins,
            feature_set,
            stats,
            role_names,
            device,
        )
    fit_sequences = [
        sequence
        for key, sequence in sorted(fit_grouped.items())
        if key[2] in {"train", "safety_calibration"}
    ]
    if not fit_sequences:
        fit_sequences = [sequence for key, sequence in sorted(fit_grouped.items()) if key[2] == "train"]
    splits["fit"] = batch_from_sequences(
        torch,
        fit_sequences,
        task_ids,
        expert_ids,
        token_bins,
        feature_set,
        stats,
        role_names,
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
    role_names: tuple[str, ...],
    device: Any,
) -> SymbolicBridgeBatch:
    max_length = max(len(sequence) for sequence in sequences)
    feature_dim = len(stats[0])
    expert_count = len(expert_ids)
    role_index = {role: index for index, role in enumerate(role_names)}
    target_ids = []
    previous_tokens = []
    features = []
    expert_tokens = []
    expert_values = []
    expert_valid_mask = []
    expert_safe_mask = []
    router_targets = []
    x_values = []
    symbolic_mask = []
    role_ids = []
    for sequence in sequences:
        sequence_targets = [int(row["target_token"]) for row in sequence]
        sequence_previous = [token_bins] + sequence_targets[:-1]
        sequence_features = [normalized_features(row, task_ids, expert_ids, feature_set, stats) for row in sequence]
        sequence_expert_tokens = [[expert_token(row, expert) for expert in expert_ids] for row in sequence]
        sequence_expert_values = [[expert_value(row, expert) for expert in expert_ids] for row in sequence]
        sequence_expert_valid = [[expert_valid(row, expert) for expert in expert_ids] for row in sequence]
        sequence_expert_safe = [[expert_safe(row, expert) for expert in expert_ids] for row in sequence]
        sequence_router_targets = [router_target(row, expert_ids) for row in sequence]
        sequence_x_values = [[float(row["x"])] for row in sequence]
        sequence_mask = [True for _row in sequence]
        sequence_roles = [role_index[str(row.get("eval_role", "symbolic"))] for row in sequence]
        pad_count = max_length - len(sequence)
        if pad_count:
            sequence_targets.extend([0] * pad_count)
            sequence_previous.extend([token_bins] * pad_count)
            sequence_features.extend([[0.0] * feature_dim for _index in range(pad_count)])
            sequence_expert_tokens.extend([[0] * expert_count for _index in range(pad_count)])
            sequence_expert_values.extend([[0.0] * expert_count for _index in range(pad_count)])
            sequence_expert_valid.extend([[False] * expert_count for _index in range(pad_count)])
            sequence_expert_safe.extend([[False] * expert_count for _index in range(pad_count)])
            sequence_router_targets.extend([expert_count] * pad_count)
            sequence_x_values.extend([[0.0] for _index in range(pad_count)])
            sequence_mask.extend([False] * pad_count)
            sequence_roles.extend([0] * pad_count)
        target_ids.append(sequence_targets)
        previous_tokens.append(sequence_previous)
        features.append(sequence_features)
        expert_tokens.append(sequence_expert_tokens)
        expert_values.append(sequence_expert_values)
        expert_valid_mask.append(sequence_expert_valid)
        expert_safe_mask.append(sequence_expert_safe)
        router_targets.append(sequence_router_targets)
        x_values.append(sequence_x_values)
        symbolic_mask.append(sequence_mask)
        role_ids.append(sequence_roles)
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
        role_ids=torch.tensor(role_ids, dtype=torch.long, device=device),
        role_names=role_names,
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
    fusion_mode: str = "hard-call",
    final_logits: Any | None = None,
    final_probabilities: Any | None = None,
    router_stats: dict[str, Any] | None = None,
    router_call_threshold: float,
    loss: float,
) -> dict[str, float | None]:
    del expert_ids
    lm_predictions = token_logits.argmax(dim=-1)
    abstain_index = batch.expert_tokens.shape[-1]
    router_predictions = router_logits.argmax(dim=-1) if use_router else torch.full_like(batch.target_ids, abstain_index)
    if fusion_mode == "prob-mixture":
        if final_probabilities is None:
            raise ValueError("prob-mixture evaluation requires final probabilities")
        final_predictions = final_probabilities.argmax(dim=-1)
    elif fusion_mode == "logit-fusion":
        if final_logits is None:
            raise ValueError("logit-fusion evaluation requires final logits")
        final_predictions = final_logits.argmax(dim=-1)
    else:
        final_predictions = lm_predictions.clone()
    expert_count = int(abstain_index)
    expert_call_mask = router_predictions < expert_count
    if use_router and fusion_mode == "hard-call":
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
        expert_call_metric = usable_call_mask.float()
        unsafe_call_metric = unsafe_call_mask.float()
    elif use_router:
        router_probabilities = torch.nn.functional.softmax(router_logits, dim=-1)
        router_confidence = router_probabilities.gather(-1, router_predictions.unsqueeze(-1)).squeeze(-1)
        confident_call_mask = expert_call_mask & (router_confidence >= router_call_threshold)
        effective_router_predictions = torch.where(
            confident_call_mask | (router_predictions == expert_count),
            router_predictions,
            torch.full_like(router_predictions, expert_count),
        )
        if router_stats and "expert_mass" in router_stats and "unsafe_mass" in router_stats:
            expert_call_metric = router_stats["expert_mass"].clamp(min=0.0, max=1.0)
            unsafe_call_metric = router_stats["unsafe_mass"].clamp(min=0.0, max=1.0)
        else:
            expert_probabilities = router_probabilities[..., :expert_count]
            valid_mask = batch.expert_valid_mask & (batch.expert_tokens >= 0)
            expert_call_metric = (expert_probabilities * valid_mask.float()).sum(dim=-1).clamp(min=0.0, max=1.0)
            unsafe_call_metric = (expert_probabilities * (~batch.expert_safe_mask).float()).sum(dim=-1).clamp(min=0.0, max=1.0)
    else:
        effective_router_predictions = router_predictions
        expert_call_metric = torch.zeros_like(batch.target_ids, dtype=torch.float32)
        unsafe_call_metric = torch.zeros_like(batch.target_ids, dtype=torch.float32)
    active_mask = batch.symbolic_mask
    total = float(active_mask.float().sum().detach().cpu().item())
    if total <= 0.0:
        total = 1.0
    final_accuracy = float(((final_predictions == batch.target_ids) & active_mask).float().sum().detach().cpu().item() / total)
    lm_accuracy = float(((lm_predictions == batch.target_ids) & active_mask).float().sum().detach().cpu().item() / total)
    router_accuracy = None
    abstain_recall = None
    if use_router:
        router_accuracy = float(((effective_router_predictions == batch.router_targets) & active_mask).float().sum().detach().cpu().item() / total)
        abstain_target = (batch.router_targets == expert_count) & active_mask
        if bool(abstain_target.any().detach().cpu().item()):
            abstain_prediction = effective_router_predictions == expert_count
            abstain_recall = float((abstain_prediction & abstain_target).float().sum().detach().cpu().item() / abstain_target.float().sum().detach().cpu().item())
    expert_call_rate = float((expert_call_metric * active_mask.float()).sum().detach().cpu().item() / total) if use_router else 0.0
    unsafe_call_rate = float((unsafe_call_metric * active_mask.float()).sum().detach().cpu().item() / total) if use_router else 0.0
    if fusion_mode == "hard-call":
        final_nll = deterministic_token_nll(final_accuracy, token_bins)
    else:
        final_nll = float(
            masked_token_nll(
                torch,
                final_logits,
                final_probabilities,
                batch.target_ids,
                active_mask,
            ).detach().cpu().item()
        )
    role_metrics = evaluate_role_metrics(
        torch,
        batch,
        token_logits,
        final_logits,
        final_probabilities,
        final_predictions,
        lm_predictions,
        expert_call_metric,
        unsafe_call_metric,
        token_bins,
        use_router=use_router,
        fusion_mode=fusion_mode,
    )
    return {
        "final_token_accuracy": final_accuracy,
        "lm_token_accuracy": lm_accuracy,
        "router_accuracy": router_accuracy,
        "expert_call_rate": expert_call_rate,
        "unsafe_call_rate": unsafe_call_rate,
        "abstain_recall": abstain_recall,
        "loss": loss,
        "final_nll": final_nll,
        "role_metrics": role_metrics,
    }


def evaluate_role_metrics(
    torch: Any,
    batch: SymbolicBridgeBatch,
    token_logits: Any,
    final_logits: Any | None,
    final_probabilities: Any | None,
    final_predictions: Any,
    lm_predictions: Any,
    expert_call_metric: Any,
    unsafe_call_metric: Any,
    token_bins: int,
    *,
    use_router: bool,
    fusion_mode: str,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for index, role in enumerate(batch.role_names):
        role_mask = batch.symbolic_mask & (batch.role_ids == index)
        if not bool(role_mask.any().detach().cpu().item()):
            continue
        total = role_mask.float().sum().detach().cpu().item()
        final_accuracy = float(((final_predictions == batch.target_ids) & role_mask).float().sum().detach().cpu().item() / total)
        lm_accuracy = float(((lm_predictions == batch.target_ids) & role_mask).float().sum().detach().cpu().item() / total)
        if fusion_mode == "hard-call":
            final_nll = deterministic_token_nll(final_accuracy, token_bins)
        else:
            final_nll = float(
                masked_token_nll(
                    torch,
                    final_logits,
                    final_probabilities,
                    batch.target_ids,
                    role_mask,
                ).detach().cpu().item()
            )
        if use_router:
            expert_call_rate = float((expert_call_metric * role_mask.float()).sum().detach().cpu().item() / total)
            unsafe_call_rate = float((unsafe_call_metric * role_mask.float()).sum().detach().cpu().item() / total)
        else:
            expert_call_rate = 0.0
            unsafe_call_rate = 0.0
        metrics[role] = {
            "count": float(total),
            "final_token_accuracy": final_accuracy,
            "lm_token_accuracy": lm_accuracy,
            "final_nll": final_nll,
            "expert_call_rate": expert_call_rate,
            "unsafe_call_rate": unsafe_call_rate,
        }
    return metrics


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


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["task_id"]),
                int(row["seed"]),
                str(row["split"]),
                str(row.get("sequence_id", "default")),
            ),
            [],
        ).append(row)
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


def summarize_lm_contract(
    runs: list[BridgeLmRun],
    rows: list[dict[str, Any]],
    *,
    fit_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    trained = [run for run in runs if run.mode in {"lm", "hard-call", "logit-fusion", "prob-mixture"}]
    router = next((run for run in runs if run.name == "lm-router-hard-call"), None)
    logit_fusion = next((run for run in runs if run.name == "lm-router-logit-fusion"), None)
    prob_mixture = next((run for run in runs if run.name == "lm-router-prob-mixture"), None)
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
    effective_fit_rows = rows if fit_rows is None else fit_rows
    selected_fit_rows = [row for row in effective_fit_rows if row["split"] in {"train", "safety_calibration"}]
    fit_abstain_target_rate = mean(
        0.0 if row["oracle_has_safe_expert"] else 1.0
        for row in selected_fit_rows
    ) if selected_fit_rows else abstain_target_rate.get("train", 0.0)
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
    math_answer_metrics = extrapolation_role_metrics(runs, "math_answer")
    prob_math = math_answer_metrics.get("lm-router-prob-mixture", {})
    side_math = math_answer_metrics.get("lm-frozen-side-channel", {})
    token_math = math_answer_metrics.get("lm-token-only", {})
    prob_math_accuracy = float(prob_math.get("final_token_accuracy", 0.0))
    prob_math_unsafe = float(prob_math.get("unsafe_call_rate", 1.0))
    math_answer_gain = prob_math_accuracy - max(
        float(side_math.get("final_token_accuracy", 0.0)),
        float(token_math.get("final_token_accuracy", 0.0)),
    )
    math_answer_contract_confirmed = math_answer_gain > 0.05 and prob_math_unsafe <= 0.05
    return {
        "total_rows": len(rows),
        "sequence_count": len(group_rows(rows)),
        "fit_row_count": len(selected_fit_rows),
        "fit_sequence_count": len(
            group_rows(selected_fit_rows)
        ) if selected_fit_rows else 0,
        "extra_fit_row_count": max(0, len(selected_fit_rows) - sum(1 for row in rows if row["split"] in {"train", "safety_calibration"})),
        "safe_expert_coverage": safe,
        "abstain_target_rate": abstain_target_rate,
        "fit_abstain_target_rate": fit_abstain_target_rate,
        "best_validation_final_token_accuracy": max(runs, key=lambda run: run.validation_final_token_accuracy).name,
        "best_extrapolation_final_token_accuracy": max(runs, key=lambda run: run.extrapolation_final_token_accuracy).name,
        "best_trained_extrapolation_final_token_accuracy": max(trained, key=lambda run: run.extrapolation_final_token_accuracy).name,
        "best_validation_final_nll": min(runs, key=lambda run: run.validation_final_nll).name,
        "best_extrapolation_final_nll": min(runs, key=lambda run: run.extrapolation_final_nll).name,
        "best_trained_extrapolation_final_nll": min(trained, key=lambda run: run.extrapolation_final_nll).name,
        "logit_fusion_extrapolation_nll_delta_vs_side_channel": (
            side.extrapolation_final_nll - logit_fusion.extrapolation_final_nll
            if side is not None and logit_fusion is not None
            else 0.0
        ),
        "prob_mixture_extrapolation_nll_delta_vs_side_channel": (
            side.extrapolation_final_nll - prob_mixture.extrapolation_final_nll
            if side is not None and prob_mixture is not None
            else 0.0
        ),
        "logit_fusion_extrapolation_nll_delta_vs_token_only": (
            token_only.extrapolation_final_nll - logit_fusion.extrapolation_final_nll
            if token_only is not None and logit_fusion is not None
            else 0.0
        ),
        "prob_mixture_extrapolation_nll_delta_vs_token_only": (
            token_only.extrapolation_final_nll - prob_mixture.extrapolation_final_nll
            if token_only is not None and prob_mixture is not None
            else 0.0
        ),
        "logit_fusion_extrapolation_accuracy_delta_vs_side_channel": (
            logit_fusion.extrapolation_final_token_accuracy - side.extrapolation_final_token_accuracy
            if side is not None and logit_fusion is not None
            else 0.0
        ),
        "prob_mixture_extrapolation_accuracy_delta_vs_side_channel": (
            prob_mixture.extrapolation_final_token_accuracy - side.extrapolation_final_token_accuracy
            if side is not None and prob_mixture is not None
            else 0.0
        ),
        "logit_fusion_extrapolation_accuracy_delta_vs_token_only": (
            logit_fusion.extrapolation_final_token_accuracy - token_only.extrapolation_final_token_accuracy
            if token_only is not None and logit_fusion is not None
            else 0.0
        ),
        "prob_mixture_extrapolation_accuracy_delta_vs_token_only": (
            prob_mixture.extrapolation_final_token_accuracy - token_only.extrapolation_final_token_accuracy
            if token_only is not None and prob_mixture is not None
            else 0.0
        ),
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
        "math_answer_extrapolation_metrics": math_answer_metrics,
        "prob_mixture_math_answer_accuracy_delta_vs_controls": math_answer_gain,
        "prob_mixture_math_answer_unsafe_call_rate": prob_math_unsafe,
        "prob_mixture_math_answer_contract_confirmed": math_answer_contract_confirmed,
        "failure_modes": failure_modes,
    }


def extrapolation_role_metrics(
    runs: list[BridgeLmRun],
    role: str,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for run in runs:
        role_metrics = run.role_metrics or {}
        split_metrics = role_metrics.get("extrapolation", {})
        if role in split_metrics:
            metrics[run.name] = {
                key: float(value)
                for key, value in split_metrics[role].items()
            }
    return metrics


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
        f"Extra fit bridge summaries: `{list(report.extra_fit_bridge_summary_paths)}`",
        f"Extra fit feature tables: `{list(report.extra_fit_feature_table_paths)}`",
        f"Backbone: `{report.backbone}`",
        f"Backbone config: `{report.backbone_config}`",
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
        f"Call/abstain loss weight: `{report.call_abstain_loss_weight}`",
        f"Answer call/abstain loss weight: `{report.answer_call_abstain_loss_weight}`",
        f"Answer unsafe loss weight: `{report.answer_unsafe_loss_weight}`",
        f"Non-answer abstain loss weight: `{report.non_answer_abstain_loss_weight}`",
        f"Non-answer LM retention loss weight: `{report.non_answer_lm_retention_loss_weight}`",
        f"Non-answer teacher KL loss weight: `{report.non_answer_teacher_kl_loss_weight}`",
        f"Non-answer teacher KL roles: `{list(report.non_answer_teacher_kl_roles) or 'none'}`",
        f"Role-aware calibration: `{report.role_aware_calibration}`",
        f"Calibration answer roles: `{list(report.calibration_answer_roles) or 'none'}`",
        f"Calibration target answer unsafe: `{report.calibration_target_answer_unsafe}`",
        f"Calibration min answer accuracy gain: `{report.calibration_min_answer_accuracy_gain}`",
        f"Unsafe margin loss weight: `{report.unsafe_margin_loss_weight}`",
        f"Unsafe margin: `{report.unsafe_margin}`",
        f"Router call threshold: `{report.router_call_threshold}`",
        f"Expert logit scale: `{report.expert_logit_scale}`",
        f"Fusion allowed roles: `{list(report.fusion_allowed_roles) or 'all'}`",
        "",
        "| run | mode | features | val final acc | extrap final acc | val final NLL | extrap final NLL | extrap LM acc | router extrap acc | expert call rate | unsafe call rate | abstain recall |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report.runs:
        lines.append(
            "| "
            f"`{run.name}` | "
            f"{run.mode} | "
            f"{run.feature_set} | "
            f"{run.validation_final_token_accuracy:.3f} | "
            f"{run.extrapolation_final_token_accuracy:.3f} | "
            f"{run.validation_final_nll:.4g} | "
            f"{run.extrapolation_final_nll:.4g} | "
            f"{run.extrapolation_lm_token_accuracy:.3f} | "
            f"{format_optional(run.extrapolation_router_accuracy)} | "
            f"{run.extrapolation_expert_call_rate:.3f} | "
            f"{run.extrapolation_unsafe_call_rate:.3f} | "
            f"{format_optional(run.extrapolation_abstain_recall)} |"
        )
    calibration_rows = [run for run in report.runs if run.calibration is not None]
    if calibration_rows:
        lines.extend(
            [
                "",
                "## Calibration Policies",
                "",
                "| run | split | feasible | temperature | abstain bias | answer threshold | answer cap | answer unsafe | answer acc | answer NLL |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run in calibration_rows:
            calibration = run.calibration or {}
            selected = calibration.get("selected_metrics", {})
            lines.append(
                "| "
                f"`{run.name}` | "
                f"{calibration.get('selected_split')} | "
                f"{calibration.get('selected_feasible')} | "
                f"{float(calibration.get('temperature', 0.0)):.3f} | "
                f"{float(calibration.get('abstain_bias', 0.0)):.3f} | "
                f"{float(calibration.get('answer_call_threshold', 0.0)):.3f} | "
                f"{float(calibration.get('answer_fusion_cap', 0.0)):.3f} | "
                f"{float(selected.get('answer_unsafe', 0.0)):.3f} | "
                f"{float(selected.get('answer_accuracy', 0.0)):.3f} | "
                f"{float(selected.get('answer_nll', 0.0)):.4g} |"
            )
    role_rows = []
    for run in report.runs:
        role_metrics = run.role_metrics or {}
        for role, metrics in sorted(role_metrics.get("extrapolation", {}).items()):
            role_rows.append((run, role, metrics))
    if role_rows:
        lines.extend(
            [
                "",
                "## Extrapolation Role Metrics",
                "",
                "| run | role | count | final acc | final NLL | LM acc | expert call | unsafe call/mass |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run, role, metrics in role_rows:
            lines.append(
                "| "
                f"`{run.name}` | "
                f"{role} | "
                f"{metrics['count']:.0f} | "
                f"{metrics['final_token_accuracy']:.3f} | "
                f"{metrics['final_nll']:.4g} | "
                f"{metrics['lm_token_accuracy']:.3f} | "
                f"{metrics['expert_call_rate']:.3f} | "
                f"{metrics['unsafe_call_rate']:.3f} |"
            )
    safe = report.summary["safe_expert_coverage"]
    abstain_rate = report.summary["abstain_target_rate"]
    lines.extend(
        [
            "",
            f"Best validation final token accuracy: `{report.summary['best_validation_final_token_accuracy']}`",
            f"Best extrapolation final token accuracy: `{report.summary['best_extrapolation_final_token_accuracy']}`",
            f"Best trained extrapolation final token accuracy: `{report.summary['best_trained_extrapolation_final_token_accuracy']}`",
            f"Fit rows: `{report.summary['fit_row_count']}`",
            f"Extra fit rows: `{report.summary['extra_fit_row_count']}`",
            f"Best validation final NLL: `{report.summary['best_validation_final_nll']}`",
            f"Best extrapolation final NLL: `{report.summary['best_extrapolation_final_nll']}`",
            f"Best trained extrapolation final NLL: `{report.summary['best_trained_extrapolation_final_nll']}`",
            f"Logit-fusion extrapolation NLL delta vs frozen side-channel: `{report.summary['logit_fusion_extrapolation_nll_delta_vs_side_channel']:.4g}`",
            f"Prob-mixture extrapolation NLL delta vs frozen side-channel: `{report.summary['prob_mixture_extrapolation_nll_delta_vs_side_channel']:.4g}`",
            f"Logit-fusion extrapolation NLL delta vs token-only: `{report.summary['logit_fusion_extrapolation_nll_delta_vs_token_only']:.4g}`",
            f"Prob-mixture extrapolation NLL delta vs token-only: `{report.summary['prob_mixture_extrapolation_nll_delta_vs_token_only']:.4g}`",
            f"Logit-fusion extrapolation accuracy delta vs frozen side-channel: `{report.summary['logit_fusion_extrapolation_accuracy_delta_vs_side_channel']:.3f}`",
            f"Prob-mixture extrapolation accuracy delta vs frozen side-channel: `{report.summary['prob_mixture_extrapolation_accuracy_delta_vs_side_channel']:.3f}`",
            f"Logit-fusion extrapolation accuracy delta vs token-only: `{report.summary['logit_fusion_extrapolation_accuracy_delta_vs_token_only']:.3f}`",
            f"Prob-mixture extrapolation accuracy delta vs token-only: `{report.summary['prob_mixture_extrapolation_accuracy_delta_vs_token_only']:.3f}`",
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
            f"Prob-mixture math-answer accuracy delta vs controls: `{report.summary['prob_mixture_math_answer_accuracy_delta_vs_controls']:.3f}`",
            f"Prob-mixture math-answer unsafe call rate: `{report.summary['prob_mixture_math_answer_unsafe_call_rate']:.3f}`",
            f"Prob-mixture math-answer contract confirmed: `{report.summary['prob_mixture_math_answer_contract_confirmed']}`",
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
