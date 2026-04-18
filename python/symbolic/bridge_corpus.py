from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python.specs.common import repo_relative
from python.symbolic.bridge import mean
from python.symbolic.bridge_canary import load_feature_rows, resolve_feature_table_path


EXPERT_IDS = ("generic-tree", "paper-complex-eml", "small-mlp", "stable-real-eml")


@dataclass(frozen=True)
class BridgeCorpusReport:
    run_label: str
    corpus_kind: str
    token_bins: int
    feature_table_path: str
    summary: dict[str, Any]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_label": self.run_label,
            "corpus_kind": self.corpus_kind,
            "token_bins": self.token_bins,
            "feature_table_path": self.feature_table_path,
            "summary": self.summary,
            "output_dir": self.output_dir,
        }


class Vocabulary:
    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def token(self, label: str) -> int:
        if label not in self._ids:
            self._ids[label] = len(self._ids)
        return self._ids[label]

    def labels(self) -> list[str]:
        labels = [""] * len(self._ids)
        for label, index in self._ids.items():
            labels[index] = label
        return labels

    def __len__(self) -> int:
        return len(self._ids)


def run_bridge_corpus(
    output_dir: Path,
    *,
    run_label: str,
    corpus_kind: str,
    source_bridge_summary_path: Path | None = None,
    seed: int = 123,
    pure_sequences_per_split: int = 160,
    language_train_per_group: int = 12,
    language_safety_per_group: int = 16,
    language_eval_per_group: int = 20,
) -> BridgeCorpusReport:
    if corpus_kind == "pure-language":
        rows, token_bins, vocabulary, source_path = build_pure_language_rows(
            seed=seed,
            sequences_per_split=pure_sequences_per_split,
        )
    elif corpus_kind == "language-math":
        if source_bridge_summary_path is None:
            raise ValueError("language-math corpus requires source_bridge_summary_path")
        rows, token_bins, vocabulary, source_path = build_language_math_rows(
            source_bridge_summary_path,
            train_per_group=language_train_per_group,
            safety_per_group=language_safety_per_group,
            eval_per_group=language_eval_per_group,
        )
    elif corpus_kind == "math-only":
        if source_bridge_summary_path is None:
            raise ValueError("math-only corpus requires source_bridge_summary_path")
        rows, token_bins, vocabulary, source_path = build_math_only_rows(source_bridge_summary_path)
    else:
        raise ValueError("corpus_kind must be one of pure-language|language-math|math-only")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_table_path = output_dir / "feature_table.jsonl"
    with feature_table_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize_corpus_rows(
        rows,
        corpus_kind=corpus_kind,
        source_bridge_summary_path=source_path,
        vocabulary=vocabulary,
    )
    report = BridgeCorpusReport(
        run_label=run_label,
        corpus_kind=corpus_kind,
        token_bins=token_bins,
        feature_table_path=repo_relative(feature_table_path),
        summary={
            **summary,
            "feature_table": {
                **summary["feature_table"],
                "path": repo_relative(feature_table_path),
            },
        },
        output_dir=repo_relative(output_dir),
    )
    (output_dir / "summary.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(render_corpus_markdown(report))
    return report


def build_pure_language_rows(
    *,
    seed: int,
    sequences_per_split: int,
) -> tuple[list[dict[str, Any]], int, list[str], str | None]:
    rng = random.Random(seed)
    vocab = base_language_vocabulary()
    rows: list[dict[str, Any]] = []
    for split in ("train", "safety_calibration", "validation", "extrapolation"):
        for sequence_index in range(sequences_per_split):
            tokens = pure_language_sentence(vocab, rng, split, sequence_index)
            rows.extend(
                text_sequence_rows(
                    tokens,
                    vocab=vocab,
                    task_id="pure_language",
                    sequence_id=f"{split}-{sequence_index:04d}",
                    seed=seed,
                    split=split,
                    eval_roles=["prose"] * len(tokens),
                    x_values=[0.0] * len(tokens),
                    answer_source_row=None,
                    number_offset=None,
                )
            )
    return rows, len(vocab), vocab.labels(), None


def build_language_math_rows(
    source_bridge_summary_path: Path,
    *,
    train_per_group: int,
    safety_per_group: int,
    eval_per_group: int,
) -> tuple[list[dict[str, Any]], int, list[str], str]:
    source_summary = json.loads(source_bridge_summary_path.read_text())
    source_feature_path = resolve_feature_table_path(source_summary, source_bridge_summary_path)
    source_rows = load_feature_rows(source_feature_path)
    token_bins = int(source_summary.get("token_bins") or source_rows[0]["token_bins"])
    vocab = base_language_vocabulary()
    formula_tokens = {
        task_id: vocab.token(f"FORMULA_{task_id}")
        for task_id in sorted({str(row["task_id"]) for row in source_rows})
    }
    for index in range(token_bins):
        vocab.token(f"XBIN_{index:02d}")
    number_offset = len(vocab)
    for index in range(token_bins):
        vocab.token(f"NUM_{index:02d}")
    selected = select_source_rows(
        source_rows,
        train_per_group=train_per_group,
        safety_per_group=safety_per_group,
        eval_per_group=eval_per_group,
    )
    rows: list[dict[str, Any]] = []
    for source_row in selected:
        task_id = str(source_row["task_id"])
        source_token = int(source_row["target_token"])
        x_token = x_bin_token(vocab, source_row, token_bins)
        tokens = [
            vocab.token("for"),
            vocab.token("formula"),
            formula_tokens[task_id],
            vocab.token("at"),
            vocab.token("x"),
            x_token,
            vocab.token("the"),
            vocab.token("value"),
            vocab.token("is"),
            number_offset + source_token,
            vocab.token("."),
        ]
        roles = [
            "prose",
            "prose",
            "math_context",
            "prose",
            "math_context",
            "math_context",
            "prose",
            "prose",
            "prose",
            "math_answer",
            "prose",
        ]
        rows.extend(
            text_sequence_rows(
                tokens,
                vocab=vocab,
                task_id=task_id,
                sequence_id=f"{source_row['split']}-{source_row['seed']}-{source_row['index']}",
                seed=int(source_row["seed"]),
                split=str(source_row["split"]),
                eval_roles=roles,
                x_values=[float(source_row["x"])] * len(tokens),
                answer_source_row=source_row,
                number_offset=number_offset,
            )
        )
    return rows, len(vocab), vocab.labels(), repo_relative(source_bridge_summary_path)


def build_math_only_rows(source_bridge_summary_path: Path) -> tuple[list[dict[str, Any]], int, list[str], str]:
    source_summary = json.loads(source_bridge_summary_path.read_text())
    source_feature_path = resolve_feature_table_path(source_summary, source_bridge_summary_path)
    source_rows = load_feature_rows(source_feature_path)
    token_bins = int(source_summary.get("token_bins") or source_rows[0]["token_bins"])
    rows = []
    for row in source_rows:
        row_copy = dict(row)
        row_copy["sequence_id"] = f"{row_copy['task_id']}-{row_copy['seed']}-{row_copy['split']}"
        row_copy["eval_role"] = "math_only"
        rows.append(row_copy)
    vocabulary = [f"NUM_{index:02d}" for index in range(token_bins)]
    return rows, token_bins, vocabulary, repo_relative(source_bridge_summary_path)


def base_language_vocabulary() -> Vocabulary:
    vocab = Vocabulary()
    for label in (
        "the",
        "a",
        "quiet",
        "bright",
        "small",
        "careful",
        "red",
        "blue",
        "green",
        "silver",
        "bird",
        "coder",
        "artist",
        "robot",
        "garden",
        "signal",
        "writes",
        "sees",
        "keeps",
        "moves",
        "checks",
        "notes",
        "story",
        "pattern",
        "window",
        "river",
        "slowly",
        "quickly",
        "today",
        "again",
        "for",
        "formula",
        "at",
        "x",
        "value",
        "is",
        ".",
    ):
        vocab.token(label)
    return vocab


def pure_language_sentence(vocab: Vocabulary, rng: random.Random, split: str, index: int) -> list[int]:
    determiners = ("the", "a")
    adjectives = ("quiet", "bright", "small", "careful", "red", "blue", "green", "silver")
    subjects = ("bird", "coder", "artist", "robot", "garden", "signal")
    verbs = ("writes", "sees", "keeps", "moves", "checks")
    objects = ("notes", "story", "pattern", "window", "river")
    adverbs = ("slowly", "quickly", "today", "again")
    offset = {"train": 0, "safety_calibration": 3, "validation": 7, "extrapolation": 11}[split]
    return [
        vocab.token(determiners[(index + offset) % len(determiners)]),
        vocab.token(adjectives[(index + offset) % len(adjectives)]),
        vocab.token(subjects[(index * 3 + offset) % len(subjects)]),
        vocab.token(verbs[(index * 5 + offset) % len(verbs)]),
        vocab.token(determiners[(index + 1 + offset) % len(determiners)]),
        vocab.token(adjectives[(index * 2 + offset + rng.randrange(2)) % len(adjectives)]),
        vocab.token(objects[(index * 7 + offset) % len(objects)]),
        vocab.token(adverbs[(index * 11 + offset) % len(adverbs)]),
        vocab.token("."),
    ]


def text_sequence_rows(
    tokens: list[int],
    *,
    vocab: Vocabulary,
    task_id: str,
    sequence_id: str,
    seed: int,
    split: str,
    eval_roles: list[str],
    x_values: list[float],
    answer_source_row: dict[str, Any] | None,
    number_offset: int | None,
) -> list[dict[str, Any]]:
    rows = []
    for index, token in enumerate(tokens):
        is_answer = eval_roles[index] == "math_answer" and answer_source_row is not None and number_offset is not None
        experts = (
            answer_experts(answer_source_row, number_offset)
            if is_answer
            else invalid_experts()
        )
        safe_mask = {expert: bool(payload["token_match"]) for expert, payload in experts.items()}
        rows.append(
            {
                "task_id": task_id,
                "sequence_id": sequence_id,
                "seed": seed,
                "split": split,
                "index": index,
                "x": x_values[index],
                "target_y": float(token),
                "target_token": token,
                "token_bins": len(vocab),
                "quantizer": {
                    "minimum": 0.0,
                    "maximum": float(max(1, len(vocab) - 1)),
                },
                "in_training_range": split in {"train", "validation"},
                "best_expert_id": best_expert_id(experts),
                "oracle_has_safe_expert": any(safe_mask.values()),
                "safe_expert_mask": safe_mask,
                "experts": experts,
                "eval_role": eval_roles[index],
            }
        )
    return rows


def invalid_experts() -> dict[str, dict[str, Any]]:
    return {
        expert: {
            "prediction": None,
            "token": -1,
            "residual": None,
            "abs_residual": 1.0e300,
            "token_match": False,
        }
        for expert in EXPERT_IDS
    }


def answer_experts(source_row: dict[str, Any], number_offset: int) -> dict[str, dict[str, Any]]:
    experts = {}
    for expert in EXPERT_IDS:
        source_payload = source_row["experts"].get(expert, {})
        source_token = int(source_payload.get("token", -1))
        if source_token < 0:
            experts[expert] = invalid_experts()[expert]
            continue
        token = number_offset + source_token
        token_match = bool(source_payload.get("token_match", False))
        experts[expert] = {
            "prediction": float(token),
            "token": token,
            "residual": 0.0 if token_match else None,
            "abs_residual": float(source_payload.get("abs_residual", 1.0e300)),
            "token_match": token_match,
        }
    return experts


def best_expert_id(experts: dict[str, dict[str, Any]]) -> str:
    return min(
        experts,
        key=lambda expert: (
            not bool(experts[expert]["token_match"]),
            float(experts[expert]["abs_residual"]),
            expert,
        ),
    )


def select_source_rows(
    rows: list[dict[str, Any]],
    *,
    train_per_group: int,
    safety_per_group: int,
    eval_per_group: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task_id"]), int(row["seed"]), str(row["split"])), []).append(row)
    selected = []
    for (_task_id, _seed, split), group in sorted(grouped.items()):
        limit = {
            "train": train_per_group,
            "safety_calibration": safety_per_group,
            "validation": eval_per_group,
            "extrapolation": eval_per_group,
        }[split]
        selected.extend(sorted(group, key=lambda row: int(row["index"]))[:limit])
    return selected


def x_bin_token(vocab: Vocabulary, row: dict[str, Any], token_bins: int) -> int:
    quantizer = row["quantizer"]
    minimum = float(quantizer["minimum"])
    maximum = float(quantizer["maximum"])
    width = max(maximum - minimum, 1.0e-12)
    x_value = float(row["x"])
    fraction = (x_value - minimum) / width
    index = max(0, min(token_bins - 1, int(math.floor(fraction * token_bins))))
    return vocab.token(f"XBIN_{index:02d}")


def summarize_corpus_rows(
    rows: list[dict[str, Any]],
    *,
    corpus_kind: str,
    source_bridge_summary_path: str | None,
    vocabulary: list[str],
) -> dict[str, Any]:
    split_counts = count_by(rows, "split")
    role_counts = count_by(rows, "eval_role")
    split_role_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        split_role_counts.setdefault(str(row["split"]), {})
        role = str(row.get("eval_role", "symbolic"))
        split_role_counts[str(row["split"])][role] = split_role_counts[str(row["split"])].get(role, 0) + 1
    role_safe = {}
    for role in sorted(role_counts):
        role_rows = [row for row in rows if str(row.get("eval_role", "symbolic")) == role]
        role_safe[role] = mean(1.0 if row["oracle_has_safe_expert"] else 0.0 for row in role_rows)
    split_role_safe: dict[str, dict[str, float]] = {}
    for split in sorted(split_counts):
        split_role_safe[split] = {}
        for role in sorted(role_counts):
            role_rows = [
                row
                for row in rows
                if str(row["split"]) == split and str(row.get("eval_role", "symbolic")) == role
            ]
            if role_rows:
                split_role_safe[split][role] = mean(1.0 if row["oracle_has_safe_expert"] else 0.0 for row in role_rows)
    safe_by_split = {}
    for split in sorted(split_counts):
        split_rows = [row for row in rows if row["split"] == split]
        safe_by_split[split] = mean(1.0 if row["oracle_has_safe_expert"] else 0.0 for row in split_rows)
    return {
        "corpus_kind": corpus_kind,
        "source_bridge_summary_path": source_bridge_summary_path,
        "vocabulary": vocabulary,
        "feature_table": {
            "row_count": len(rows),
            "split_counts": split_counts,
            "role_counts": role_counts,
            "split_role_counts": split_role_counts,
            "role_safe_expert_coverage": role_safe,
            "split_role_safe_expert_coverage": split_role_safe,
            "split_safe_expert_coverage": safe_by_split,
            "safe_expert_coverage": mean(1.0 if row["oracle_has_safe_expert"] else 0.0 for row in rows),
        },
    }


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def render_corpus_markdown(report: BridgeCorpusReport) -> str:
    feature_table = report.summary["feature_table"]
    lines = [
        f"# Bridge Corpus: {report.run_label}",
        "",
        f"Corpus kind: `{report.corpus_kind}`",
        f"Token bins: `{report.token_bins}`",
        f"Feature table: `{report.feature_table_path}`",
        f"Rows: `{feature_table['row_count']}`",
        f"Split counts: `{feature_table['split_counts']}`",
        f"Role counts: `{feature_table['role_counts']}`",
        f"Safe expert coverage by split: `{feature_table['split_safe_expert_coverage']}`",
        f"Safe expert coverage by role: `{feature_table['role_safe_expert_coverage']}`",
        "",
    ]
    return "\n".join(lines)
