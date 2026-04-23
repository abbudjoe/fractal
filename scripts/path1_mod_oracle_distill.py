#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.data import load_byte_corpus
from python.data.byte_corpus import TokenBatch
from python.models.path1 import build_path1_model
from python.runtime import configure_reproducibility
from python.specs.common import DeviceRuntimeSpec, JsonlCorpusSpec, SeedSpec, repo_relative
from python.specs.path1 import (
    BYTE_LEVEL_PAD_TOKEN,
    Path1ModelShape,
    TokenRoutingProfile,
    phase1_attention_only_variant,
)

SUITE_NAME = "path1-mod-oracle-distill-v1"
DEFAULT_CORPUS = Path("experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1")


@dataclass(frozen=True)
class TrainResult:
    label: str
    seed: int
    initial_loss: float
    final_loss: float
    train_tokens_per_second: float
    parameter_count: int
    distill_loss_mean: float | None = None
    alignment: dict[str, float] | None = None


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(chunk) for chunk in value.replace(",", " ").split())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _corpus_spec(corpus_dir: Path) -> JsonlCorpusSpec:
    return JsonlCorpusSpec(
        train_path=corpus_dir / "train.jsonl",
        eval_path=corpus_dir / "eval.jsonl",
        corpus_name="fineweb-stage0-local-bench-9row-v1",
    )


def _shape() -> Path1ModelShape:
    return Path1ModelShape(
        d_model=32,
        head_count=4,
        total_layers=4,
        ffn_multiplier=2,
    )


def _teacher_variant() -> object:
    return phase1_attention_only_variant(
        shape=_shape(),
        token_routing_profile=TokenRoutingProfile.MOD_TRAIN_TOPC_BLOCK,
        token_route_fraction=0.5,
        token_routing_layer_indices=(1,),
    )


def _student_variant() -> object:
    return phase1_attention_only_variant(
        shape=_shape(),
        token_routing_profile=TokenRoutingProfile.CAUSAL_TOPK_BLOCK,
        token_route_fraction=0.5,
        token_routing_layer_indices=(1,),
    )


def _to_device(batch: TokenBatch, device: torch.device) -> TokenBatch:
    return batch.to_device(device)


def _lm_loss(model: Any, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    supervised_loss = getattr(model, "supervised_loss", None)
    if callable(supervised_loss):
        return supervised_loss(
            logits,
            target_ids,
            pad_token=BYTE_LEVEL_PAD_TOKEN,
            training=True,
        )
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=BYTE_LEVEL_PAD_TOKEN,
    )


def _routed_blocks(model: Any) -> list[Any]:
    blocks = []
    for block in getattr(model, "blocks", []):
        last_routing_tensors = getattr(block, "last_routing_tensors", None)
        if callable(last_routing_tensors):
            blocks.append(block)
    return blocks


def _routing_masks(model: Any) -> list[torch.Tensor]:
    masks: list[torch.Tensor] = []
    for block in _routed_blocks(model):
        masks.append(block.last_routing_tensors()["selected_mask"].detach().clone())
    if not masks:
        raise RuntimeError("model did not expose routed block masks")
    return masks


def _routing_scores(model: Any) -> list[torch.Tensor]:
    scores: list[torch.Tensor] = []
    for block in _routed_blocks(model):
        scores.append(block.last_routing_tensors()["router_scores"])
    if not scores:
        raise RuntimeError("model did not expose routed block scores")
    return scores


def _routing_distill_loss(student: Any, teacher_masks: list[torch.Tensor]) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for student_scores, teacher_mask in zip(_routing_scores(student), teacher_masks):
        losses.append(
            F.binary_cross_entropy_with_logits(
                student_scores,
                teacher_mask.to(device=student_scores.device, dtype=student_scores.dtype),
            )
        )
    if not losses:
        raise RuntimeError("no routed blocks available for oracle distillation")
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_loss(
    model: Any,
    batches: list[TokenBatch],
    *,
    eval_batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch in batches[:eval_batches]:
        batch = _to_device(batch, device)
        logits = model.forward_logits(batch.input_ids)
        loss = _lm_loss(model, logits, batch.target_ids)
        losses.append(float(loss.detach().float().item()))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate_alignment(
    *,
    teacher: Any,
    student: Any,
    batches: list[TokenBatch],
    eval_batches: int,
    device: torch.device,
) -> dict[str, float]:
    teacher.eval()
    student.eval()
    intersection = 0.0
    teacher_count = 0.0
    student_count = 0.0
    total = 0.0
    bce_values: list[float] = []
    for batch in batches[:eval_batches]:
        batch = _to_device(batch, device)
        teacher.forward_logits(batch.input_ids)
        teacher_masks = _routing_masks(teacher)
        student.forward_logits(batch.input_ids)
        student_scores = _routing_scores(student)
        student_masks = _routing_masks(student)
        for teacher_mask, student_score, student_mask in zip(
            teacher_masks, student_scores, student_masks
        ):
            teacher_mask = teacher_mask.to(device=student_mask.device)
            selected_both = torch.logical_and(teacher_mask, student_mask)
            intersection += float(selected_both.sum().item())
            teacher_count += float(teacher_mask.sum().item())
            student_count += float(student_mask.sum().item())
            total += float(teacher_mask.numel())
            bce = F.binary_cross_entropy_with_logits(
                student_score,
                teacher_mask.to(dtype=student_score.dtype),
            )
            bce_values.append(float(bce.detach().float().item()))
    precision = intersection / student_count if student_count else 0.0
    recall = intersection / teacher_count if teacher_count else 0.0
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom else 0.0
    union = teacher_count + student_count - intersection
    return {
        "oracle_bce": sum(bce_values) / max(len(bce_values), 1),
        "oracle_precision": precision,
        "oracle_recall": recall,
        "oracle_f1": f1,
        "oracle_jaccard": intersection / union if union else 0.0,
        "oracle_selected_fraction": teacher_count / total if total else 0.0,
        "student_selected_fraction": student_count / total if total else 0.0,
    }


def _optimizer(model: Any, learning_rate: float) -> torch.optim.Optimizer:
    parameter_groups = model.optimizer_parameter_groups(learning_rate)
    return torch.optim.Adam(parameter_groups, lr=learning_rate)


def train_lm_only(
    *,
    label: str,
    model: Any,
    train_batches: list[TokenBatch],
    eval_batches: list[TokenBatch],
    train_steps: int,
    eval_batch_count: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    teacher: Any | None = None,
) -> TrainResult:
    initial_loss = evaluate_loss(
        model, eval_batches, eval_batches=eval_batch_count, device=device
    )
    optimizer = _optimizer(model, learning_rate)
    model.train()
    seen_tokens = 0
    start = time.perf_counter()
    for step in range(train_steps):
        batch = _to_device(train_batches[step % len(train_batches)], device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_logits(batch.input_ids)
        loss = _lm_loss(model, logits, batch.target_ids)
        loss.backward()
        optimizer.step()
        seen_tokens += batch.token_count
    elapsed = time.perf_counter() - start
    final_loss = evaluate_loss(
        model, eval_batches, eval_batches=eval_batch_count, device=device
    )
    alignment = (
        evaluate_alignment(
            teacher=teacher,
            student=model,
            batches=eval_batches,
            eval_batches=eval_batch_count,
            device=device,
        )
        if teacher is not None
        else None
    )
    return TrainResult(
        label=label,
        seed=seed,
        initial_loss=initial_loss,
        final_loss=final_loss,
        train_tokens_per_second=seen_tokens / elapsed,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        alignment=alignment,
    )


def train_oracle_distilled(
    *,
    label: str,
    teacher: Any,
    student: Any,
    train_batches: list[TokenBatch],
    eval_batches: list[TokenBatch],
    train_steps: int,
    eval_batch_count: int,
    learning_rate: float,
    distill_weight: float,
    device: torch.device,
    seed: int,
) -> TrainResult:
    initial_loss = evaluate_loss(
        student, eval_batches, eval_batches=eval_batch_count, device=device
    )
    optimizer = _optimizer(student, learning_rate)
    teacher.eval()
    student.train()
    seen_tokens = 0
    distill_values: list[float] = []
    start = time.perf_counter()
    for step in range(train_steps):
        batch = _to_device(train_batches[step % len(train_batches)], device)
        with torch.no_grad():
            teacher.forward_logits(batch.input_ids)
            teacher_masks = _routing_masks(teacher)
        optimizer.zero_grad(set_to_none=True)
        logits = student.forward_logits(batch.input_ids)
        lm_loss = _lm_loss(student, logits, batch.target_ids)
        distill_loss = _routing_distill_loss(student, teacher_masks)
        total_loss = lm_loss + distill_weight * distill_loss
        total_loss.backward()
        optimizer.step()
        seen_tokens += batch.token_count
        distill_values.append(float(distill_loss.detach().float().item()))
    elapsed = time.perf_counter() - start
    final_loss = evaluate_loss(
        student, eval_batches, eval_batches=eval_batch_count, device=device
    )
    alignment = evaluate_alignment(
        teacher=teacher,
        student=student,
        batches=eval_batches,
        eval_batches=eval_batch_count,
        device=device,
    )
    return TrainResult(
        label=label,
        seed=seed,
        initial_loss=initial_loss,
        final_loss=final_loss,
        train_tokens_per_second=seen_tokens / elapsed,
        parameter_count=sum(parameter.numel() for parameter in student.parameters()),
        distill_loss_mean=sum(distill_values) / max(len(distill_values), 1),
        alignment=alignment,
    )


def _clone_teacher_into_student(teacher: Any, *, device: torch.device) -> Any:
    student = build_path1_model(_student_variant(), dtype_mode="fp32").to(device)
    student.load_state_dict(teacher.state_dict(), strict=True)
    return student


def _fresh_student(*, device: torch.device) -> Any:
    return build_path1_model(_student_variant(), dtype_mode="fp32").to(device)


def _aggregate(results: list[TrainResult]) -> list[dict[str, Any]]:
    labels = sorted({result.label for result in results})
    rows: list[dict[str, Any]] = []
    for label in labels:
        group = [result for result in results if result.label == label]
        losses = [result.final_loss for result in group]
        speeds = [result.train_tokens_per_second for result in group]
        row: dict[str, Any] = {
            "label": label,
            "seeds": [result.seed for result in group],
            "mean_final_loss": sum(losses) / len(losses),
            "std_final_loss": (
                math.sqrt(
                    sum((loss - (sum(losses) / len(losses))) ** 2 for loss in losses)
                    / len(losses)
                )
                if len(losses) > 1
                else 0.0
            ),
            "mean_train_tokens_per_second": sum(speeds) / len(speeds),
            "parameter_count": group[0].parameter_count,
        }
        distill_losses = [
            result.distill_loss_mean
            for result in group
            if result.distill_loss_mean is not None
        ]
        if distill_losses:
            row["mean_distill_loss"] = sum(distill_losses) / len(distill_losses)
        alignment_keys = sorted(
            {
                key
                for result in group
                if result.alignment is not None
                for key in result.alignment
            }
        )
        for key in alignment_keys:
            values = [
                result.alignment[key]
                for result in group
                if result.alignment is not None and key in result.alignment
            ]
            row[f"mean_{key}"] = sum(values) / len(values)
        rows.append(row)
    rows.sort(key=lambda row: float(row["mean_final_loss"]))
    return rows


def _write_summary(
    *,
    output_dir: Path,
    conditions: dict[str, Any],
    results: list[TrainResult],
) -> None:
    aggregate = _aggregate(results)
    payload = {
        "suite_name": SUITE_NAME,
        "conditions": conditions,
        "aggregate": aggregate,
        "runs": [
            {
                "label": result.label,
                "seed": result.seed,
                "initial_loss": result.initial_loss,
                "final_loss": result.final_loss,
                "train_tokens_per_second": result.train_tokens_per_second,
                "parameter_count": result.parameter_count,
                "distill_loss_mean": result.distill_loss_mean,
                "alignment": result.alignment,
            }
            for result in results
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        f"# {SUITE_NAME}",
        "",
        "Oracle-distillation smoke: train a full-sequence MoD top-C teacher, then test whether a causal prefix-top-k student can recover its routing mask.",
        "",
        "## Conditions",
        "",
    ]
    for key, value in conditions.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Lane | Loss | Loss Std | Tok/s | Oracle F1 | Oracle BCE | Params |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate:
        lines.append(
            "| {label} | {loss:.4f} | {std:.4f} | {speed:.1f} | {f1} | {bce} | {params} |".format(
                label=row["label"],
                loss=row["mean_final_loss"],
                std=row["std_final_loss"],
                speed=row["mean_train_tokens_per_second"],
                f1=(
                    f"{row['mean_oracle_f1']:.4f}"
                    if "mean_oracle_f1" in row
                    else ""
                ),
                bce=(
                    f"{row['mean_oracle_bce']:.4f}"
                    if "mean_oracle_bce" in row
                    else ""
                ),
                params=row["parameter_count"],
            )
        )
    lines.extend(["", "## Per-Seed Final Loss", ""])
    seed_values = sorted({result.seed for result in results})
    lines.append("| Lane | " + " | ".join(f"Seed {seed}" for seed in seed_values) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in seed_values) + " |")
    for label in sorted({result.label for result in results}):
        by_seed = {
            result.seed: result.final_loss
            for result in results
            if result.label == label
        }
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{by_seed[seed]:.4f}" for seed in seed_values)
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test oracle distillation from non-causal MoD top-C routing into a causal prefix-top-k router."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / SUITE_NAME)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--seeds", type=_parse_seeds, default=(42, 43, 44))
    parser.add_argument("--train-steps", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--distill-weight", type=float, default=0.2)
    parser.add_argument("--backend", choices=["cpu"], default="cpu")
    args = parser.parse_args()

    device = torch.device(args.backend)
    corpus = load_byte_corpus(
        _corpus_spec(args.corpus_dir),
        seq_len=args.seq_len,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        data_seed=None,
        shuffle_train=False,
        pin_memory=False,
    )
    conditions = {
        "backend": args.backend,
        "dtype": "fp32",
        "shape": "d_model=32, heads=4, layers=4, ffn_multiplier=2",
        "train_steps": args.train_steps,
        "eval_batches": args.eval_batches,
        "seq_len": args.seq_len,
        "window_stride": args.window_stride,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "distill_weight": args.distill_weight,
        "corpus": corpus.corpus_stats["corpus_name"],
    }

    all_results: list[TrainResult] = []
    runtime = DeviceRuntimeSpec(backend=args.backend, dtype="fp32")
    for seed in args.seeds:
        print(f"SEED {seed} teacher", flush=True)
        configure_reproducibility(SeedSpec(model_seed=seed), runtime)
        teacher = build_path1_model(_teacher_variant(), dtype_mode="fp32").to(device)
        teacher_result = train_lm_only(
            label="mod-train-topc-teacher",
            model=teacher,
            train_batches=corpus.train_batches,
            eval_batches=corpus.eval_batches,
            train_steps=args.train_steps,
            eval_batch_count=args.eval_batches,
            learning_rate=args.learning_rate,
            device=device,
            seed=seed,
        )
        all_results.append(teacher_result)

        print(f"SEED {seed} causal scratch", flush=True)
        configure_reproducibility(SeedSpec(model_seed=seed), runtime)
        scratch_student = _fresh_student(device=device)
        scratch_result = train_lm_only(
            label="causal-topk-scratch",
            model=scratch_student,
            train_batches=corpus.train_batches,
            eval_batches=corpus.eval_batches,
            train_steps=args.train_steps,
            eval_batch_count=args.eval_batches,
            learning_rate=args.learning_rate,
            device=device,
            seed=seed,
            teacher=teacher,
        )
        all_results.append(scratch_result)

        print(f"SEED {seed} causal teacher-init", flush=True)
        teacher_init_student = _clone_teacher_into_student(teacher, device=device)
        teacher_init_result = train_lm_only(
            label="causal-topk-teacher-init",
            model=teacher_init_student,
            train_batches=corpus.train_batches,
            eval_batches=corpus.eval_batches,
            train_steps=args.train_steps,
            eval_batch_count=args.eval_batches,
            learning_rate=args.learning_rate,
            device=device,
            seed=seed,
            teacher=teacher,
        )
        all_results.append(teacher_init_result)

        print(f"SEED {seed} causal oracle-distilled", flush=True)
        distilled_student = _clone_teacher_into_student(teacher, device=device)
        distilled_result = train_oracle_distilled(
            label="causal-topk-oracle-distilled",
            teacher=teacher,
            student=distilled_student,
            train_batches=corpus.train_batches,
            eval_batches=corpus.eval_batches,
            train_steps=args.train_steps,
            eval_batch_count=args.eval_batches,
            learning_rate=args.learning_rate,
            distill_weight=args.distill_weight,
            device=device,
            seed=seed,
        )
        all_results.append(distilled_result)

    _write_summary(
        output_dir=args.output_dir,
        conditions=conditions,
        results=all_results,
    )
    print(repo_relative(args.output_dir / "summary.md"), flush=True)


if __name__ == "__main__":
    main()
