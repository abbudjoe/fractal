from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from python.specs.common import JsonlCorpusSpec, ValidationError, repo_relative


@dataclass(frozen=True)
class TokenBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    token_count: int

    def to_device(self, device: torch.device) -> "TokenBatch":
        if self.input_ids.device == device and self.target_ids.device == device:
            return self
        return TokenBatch(
            input_ids=self.input_ids.to(device=device, non_blocking=True),
            target_ids=self.target_ids.to(device=device, non_blocking=True),
            token_count=self.token_count,
        )


@dataclass(frozen=True)
class ByteCorpusBundle:
    corpus_stats: dict[str, object]
    train_batches: Sequence[TokenBatch]
    eval_batches: Sequence[TokenBatch]


def load_jsonl_text_documents(path: Path, text_field: str) -> list[bytes]:
    documents: list[bytes] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"failed to parse jsonl text corpus {path} line {line_index}: {exc}"
                ) from exc
            text = value.get(text_field)
            if not isinstance(text, str):
                raise ValidationError(
                    f"jsonl text corpus {path} line {line_index} is missing string field {text_field!r}"
                )
            if text:
                documents.append(text.encode("utf-8"))
    if not documents:
        raise ValidationError(f"jsonl text corpus {path} did not yield any non-empty documents")
    return documents


def byte_sequences_from_documents(
    documents: list[bytes],
    seq_len: int,
    window_stride: int,
) -> list[tuple[list[int], list[int]]]:
    required_len = seq_len + 1
    sequences: list[tuple[list[int], list[int]]] = []
    for raw in documents:
        if len(raw) < required_len:
            continue
        for start in range(0, len(raw) - required_len + 1, window_stride):
            window = raw[start : start + required_len]
            inputs = [int(window[index]) + 1 for index in range(seq_len)]
            targets = [int(window[index + 1]) + 1 for index in range(seq_len)]
            sequences.append((inputs, targets))
    if len(sequences) < 2:
        raise ValidationError(
            f"byte-level corpus must yield at least 2 sequences of length {seq_len}, got {len(sequences)}"
        )
    return sequences


def sequences_into_batches(
    sequences: list[tuple[list[int], list[int]]],
    batch_size: int,
    *,
    pin_memory: bool = False,
) -> list[TokenBatch]:
    batches: list[TokenBatch] = []
    seq_len = len(sequences[0][0])
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        input_flat = [token for inputs, _ in chunk for token in inputs]
        target_flat = [token for _, targets in chunk for token in targets]
        input_ids = torch.tensor(input_flat, dtype=torch.long).reshape(len(chunk), seq_len)
        target_ids = torch.tensor(target_flat, dtype=torch.long).reshape(len(chunk), seq_len)
        if pin_memory:
            input_ids = input_ids.pin_memory()
            target_ids = target_ids.pin_memory()
        batches.append(TokenBatch(input_ids=input_ids, target_ids=target_ids, token_count=len(chunk) * seq_len))
    return batches


def load_byte_corpus(
    corpus_spec: JsonlCorpusSpec,
    *,
    seq_len: int,
    window_stride: int,
    batch_size: int,
    data_seed: int | None,
    shuffle_train: bool = False,
    pin_memory: bool = False,
) -> ByteCorpusBundle:
    corpus_spec.validate()
    train_docs = load_jsonl_text_documents(corpus_spec.train_path, corpus_spec.text_field)
    eval_docs = load_jsonl_text_documents(corpus_spec.eval_path, corpus_spec.text_field)
    train_sequences = byte_sequences_from_documents(train_docs, seq_len, window_stride)
    eval_sequences = byte_sequences_from_documents(eval_docs, seq_len, window_stride)
    if shuffle_train and data_seed is not None:
        random.Random(data_seed).shuffle(train_sequences)
    train_batches = sequences_into_batches(train_sequences, batch_size, pin_memory=pin_memory)
    eval_batches = sequences_into_batches(eval_sequences, batch_size, pin_memory=pin_memory)
    total_bytes = sum(len(doc) for doc in train_docs) + sum(len(doc) for doc in eval_docs)
    corpus_stats = {
        "files": [repo_relative(corpus_spec.train_path), repo_relative(corpus_spec.eval_path)],
        "corpus_name": corpus_spec.corpus_name,
        "text_field": corpus_spec.text_field,
        "total_bytes": total_bytes,
        "total_sequences": sum(batch.input_ids.shape[0] for batch in train_batches)
        + sum(batch.input_ids.shape[0] for batch in eval_batches),
        "train_sequences": sum(batch.input_ids.shape[0] for batch in train_batches),
        "eval_sequences": sum(batch.input_ids.shape[0] for batch in eval_batches),
        "seq_len": seq_len,
        "window_stride": window_stride,
        "data_seed": data_seed,
        "shuffle_train": shuffle_train,
    }
    return ByteCorpusBundle(corpus_stats=corpus_stats, train_batches=train_batches, eval_batches=eval_batches)
