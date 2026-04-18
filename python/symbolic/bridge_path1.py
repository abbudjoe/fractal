from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from python.models.path1 import build_path1_model
from python.specs.common import repo_relative
from python.specs.path1 import Path1ModelShape, phase1_attention_only_variant
from python.symbolic.bridge_canary import load_feature_rows, resolve_device, resolve_feature_table_path
from python.symbolic.bridge_lm import (
    BridgeLmReport,
    BridgeLmRun,
    build_contract_splits,
    evaluate_best_single_expert,
    evaluate_contract_outputs,
    evaluate_majority_baseline,
    evaluate_oracle_abstain_router,
    masked_cross_entropy,
    render_bridge_lm_markdown,
    router_call_abstain_loss,
    router_loss_class_weights,
    router_unsafe_margin_loss,
    router_unsafe_probability_loss,
    summarize_lm_contract,
)


def run_symbolic_bridge_path1(
    bridge_summary_path: Path,
    output_dir: Path,
    *,
    run_label: str,
    seed: int = 777,
    epochs: int = 900,
    learning_rate: float = 0.003,
    d_model: int = 96,
    total_layers: int = 4,
    head_count: int = 4,
    ffn_multiplier: int = 2,
    router_loss_weight: float = 10.0,
    abstain_class_weight: float = 1.0,
    unsafe_call_loss_weight: float = 0.0,
    call_abstain_loss_weight: float = 5.0,
    unsafe_margin_loss_weight: float = 0.0,
    unsafe_margin: float = 0.5,
    router_call_threshold: float = 0.99999,
    device: str = "auto",
) -> BridgeLmReport:
    bridge_summary = json.loads(bridge_summary_path.read_text())
    feature_table_path = resolve_feature_table_path(bridge_summary, bridge_summary_path)
    rows = load_feature_rows(feature_table_path)
    if not rows:
        raise ValueError(f"bridge feature table has no rows: {feature_table_path}")

    torch.manual_seed(seed)
    selected_device = resolve_device(torch, device)
    token_bins = int(bridge_summary.get("token_bins") or rows[0]["token_bins"])
    task_ids = tuple(sorted({str(row["task_id"]) for row in rows}))
    expert_ids = tuple(sorted({str(expert) for row in rows for expert in row["experts"]}))
    abstain_index = len(expert_ids)

    path1_shape = Path1ModelShape(
        vocab_size=token_bins + 1,
        d_model=d_model,
        head_count=head_count,
        total_layers=total_layers,
        local_window=512,
        ffn_multiplier=ffn_multiplier,
    )
    path1_variant = phase1_attention_only_variant(shape=path1_shape)
    path1_variant.validate()

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
            train_path1_contract_variant(
                rows,
                task_ids,
                expert_ids,
                token_bins,
                path1_variant=path1_variant,
                feature_set=feature_set,
                name=name,
                use_router=use_router,
                seed=seed + 31 + offset,
                epochs=epochs,
                learning_rate=learning_rate,
                router_loss_weight=router_loss_weight,
                abstain_class_weight=abstain_class_weight,
                unsafe_call_loss_weight=unsafe_call_loss_weight,
                call_abstain_loss_weight=call_abstain_loss_weight,
                unsafe_margin_loss_weight=unsafe_margin_loss_weight,
                unsafe_margin=unsafe_margin,
                router_call_threshold=router_call_threshold,
                device=selected_device,
            )
        )

    summary = summarize_lm_contract(runs, rows)
    summary["path1_variant"] = {
        "label": path1_variant.label,
        "d_model": d_model,
        "total_layers": total_layers,
        "head_count": head_count,
        "ffn_multiplier": ffn_multiplier,
        "integration": "Path1 hidden state + frozen compiled-expert features -> router over experts/ABSTAIN",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report = BridgeLmReport(
        bridge_summary_path=repo_relative(bridge_summary_path),
        feature_table_path=repo_relative(feature_table_path),
        run_label=run_label,
        backbone="path1",
        backbone_config={
            "type": "path1-attention-only",
            "d_model": d_model,
            "total_layers": total_layers,
            "head_count": head_count,
            "ffn_multiplier": ffn_multiplier,
        },
        token_bins=token_bins,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=d_model,
        router_loss_weight=router_loss_weight,
        abstain_class_weight=abstain_class_weight,
        unsafe_call_loss_weight=unsafe_call_loss_weight,
        call_abstain_loss_weight=call_abstain_loss_weight,
        answer_unsafe_loss_weight=0.0,
        non_answer_abstain_loss_weight=0.0,
        unsafe_margin_loss_weight=unsafe_margin_loss_weight,
        unsafe_margin=unsafe_margin,
        router_call_threshold=router_call_threshold,
        expert_logit_scale=0.0,
        device=str(selected_device),
        expert_ids=expert_ids,
        abstain_index=abstain_index,
        runs=tuple(runs),
        summary=summary,
        output_dir=repo_relative(output_dir),
    )
    write_path1_bridge_report(report, output_dir)
    return report


class Path1SymbolicBridgeAdapter(torch.nn.Module):
    def __init__(
        self,
        *,
        path1_variant: Any,
        feature_dim: int,
        router_classes: int,
        dtype_mode: str,
        use_features: bool,
    ) -> None:
        super().__init__()
        self.backbone = build_path1_model(path1_variant, dtype_mode=dtype_mode)
        d_model = path1_variant.shape.d_model
        self.use_features = use_features
        self.feature_projection = torch.nn.Linear(feature_dim, d_model) if use_features else None
        self.router_norm = torch.nn.LayerNorm(d_model)
        self.router_head = torch.nn.Linear(d_model, router_classes)

    def forward(self, previous_tokens: Any, features: Any) -> tuple[Any, Any]:
        hidden = self.backbone.forward_hidden(previous_tokens)
        if self.feature_projection is not None:
            hidden = hidden + torch.tanh(self.feature_projection(features))
        token_logits = self.backbone.output(hidden)
        router_logits = self.router_head(self.router_norm(hidden))
        return token_logits, router_logits


def train_path1_contract_variant(
    rows: list[dict[str, Any]],
    task_ids: tuple[str, ...],
    expert_ids: tuple[str, ...],
    token_bins: int,
    *,
    path1_variant: Any,
    feature_set: str,
    name: str,
    use_router: bool,
    seed: int,
    epochs: int,
    learning_rate: float,
    router_loss_weight: float,
    abstain_class_weight: float,
    unsafe_call_loss_weight: float,
    call_abstain_loss_weight: float,
    unsafe_margin_loss_weight: float,
    unsafe_margin: float,
    router_call_threshold: float,
    device: Any,
) -> BridgeLmRun:
    torch.manual_seed(seed)
    splits = build_contract_splits(torch, rows, task_ids, expert_ids, token_bins, feature_set, device)
    feature_dim = int(splits["fit"].features.shape[-1])
    model = Path1SymbolicBridgeAdapter(
        path1_variant=path1_variant,
        feature_dim=feature_dim,
        router_classes=len(expert_ids) + 1,
        dtype_mode="fp32",
        use_features=feature_set != "token-only",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    fit_batch = splits["fit"]
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        token_logits, router_logits = model(fit_batch.previous_tokens, fit_batch.features)
        token_loss = masked_cross_entropy(torch, token_logits, fit_batch.target_ids, fit_batch.symbolic_mask)
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
            unsafe_margin_loss = router_unsafe_margin_loss(
                torch,
                router_logits,
                fit_batch.expert_safe_mask,
                margin=unsafe_margin,
                mask=fit_batch.symbolic_mask,
            )
            loss = (
                token_loss
                + router_loss_weight * router_loss
                + unsafe_call_loss_weight * unsafe_loss
                + call_abstain_loss_weight * call_abstain_loss
                + unsafe_margin_loss_weight * unsafe_margin_loss
            )
        else:
            loss = token_loss
        loss.backward()
        optimizer.step()

    split_metrics = {}
    model.eval()
    with torch.no_grad():
        for split, batch in splits.items():
            token_logits, router_logits = model(batch.previous_tokens, batch.features)
            token_loss = masked_cross_entropy(torch, token_logits, batch.target_ids, batch.symbolic_mask)
            router_loss_value = 0.0
            unsafe_loss_value = 0.0
            call_abstain_loss_value = 0.0
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
                unsafe_margin_loss = router_unsafe_margin_loss(
                    torch,
                    router_logits,
                    batch.expert_safe_mask,
                    margin=unsafe_margin,
                    mask=batch.symbolic_mask,
                )
                unsafe_margin_loss_value = float(unsafe_margin_loss.detach().cpu().item())
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
                    + call_abstain_loss_weight * call_abstain_loss_value
                    + unsafe_margin_loss_weight * unsafe_margin_loss_value
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


def write_path1_bridge_report(report: BridgeLmReport, output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    with (output_dir / "runs.jsonl").open("w") as handle:
        for run in report.runs:
            handle.write(json.dumps(run.to_dict(), sort_keys=True) + "\n")
    markdown = render_bridge_lm_markdown(report).replace(
        "# Symbolic Bridge LM Contract:",
        "# Symbolic Bridge Path1 Contract:",
        1,
    )
    (output_dir / "summary.md").write_text(markdown)
