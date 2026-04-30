from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from python.data.byte_corpus import ByteCorpusBundle, TokenBatch
from python.specs.common import TokenIdCorpusSpec, ValidationError, repo_relative


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"failed to parse token-id corpus manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"token-id corpus manifest {path} must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValidationError(
            f"token-id corpus manifest {path} has unsupported schema_version={payload.get('schema_version')!r}"
        )
    return payload


def _split_shards(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValidationError("token-id corpus manifest must contain object field 'splits'")
    split_payload = splits.get(split)
    if not isinstance(split_payload, dict):
        raise ValidationError(f"token-id corpus manifest must contain split {split!r}")
    shards = split_payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValidationError(f"token-id corpus split {split!r} must contain non-empty shard list")
    for shard in shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("path"), str):
            raise ValidationError(f"token-id corpus split {split!r} contains invalid shard entry")
    return shards


def _resolve_shard_path(manifest_path: Path, shard: dict[str, Any]) -> Path:
    path = Path(shard["path"])
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _load_split_tokens(manifest_path: Path, manifest: dict[str, Any], split: str) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for shard in _split_shards(manifest, split):
        path = _resolve_shard_path(manifest_path, shard)
        if not path.exists():
            raise ValidationError(f"token-id corpus shard does not exist: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            tokens = payload.get("tokens")
        else:
            tokens = payload
        if not isinstance(tokens, torch.Tensor):
            raise ValidationError(f"token-id corpus shard {path} must contain a tensor or dict['tokens']")
        if tokens.ndim != 1:
            raise ValidationError(f"token-id corpus shard {path} must be a 1D token tensor")
        if not tokens.dtype.is_floating_point and tokens.dtype != torch.bool:
            tensors.append(tokens.to(dtype=torch.int32, device="cpu"))
        else:
            raise ValidationError(f"token-id corpus shard {path} must contain integer token ids")
    if not tensors:
        raise ValidationError(f"token-id corpus split {split!r} did not yield any tensors")
    return torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]


class LazyTokenBatchSequence(Sequence[TokenBatch]):
    """Sequence-like token batches that materialize only the requested batch."""

    def __init__(
        self,
        *,
        tokens: torch.Tensor,
        starts: torch.Tensor,
        seq_len: int,
        batch_size: int,
        pin_memory: bool,
    ) -> None:
        self._tokens = tokens
        self._starts = starts
        self._seq_len = seq_len
        self._batch_size = batch_size
        self._pin_memory = pin_memory
        self._offsets = torch.arange(seq_len, dtype=torch.long)

    @property
    def sequence_count(self) -> int:
        return int(self._starts.numel())

    def __len__(self) -> int:
        return (self.sequence_count + self._batch_size - 1) // self._batch_size

    def __getitem__(self, index: int | slice) -> TokenBatch | list[TokenBatch]:
        if isinstance(index, slice):
            return [self[batch_index] for batch_index in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start_index = index * self._batch_size
        end_index = min(start_index + self._batch_size, self.sequence_count)
        starts = self._starts[start_index:end_index].to(dtype=torch.long)
        indices = starts[:, None] + self._offsets[None, :]
        input_ids = self._tokens[indices].to(dtype=torch.long).contiguous()
        target_ids = self._tokens[indices + 1].to(dtype=torch.long).contiguous()
        if self._pin_memory:
            input_ids = input_ids.pin_memory()
            target_ids = target_ids.pin_memory()
        return TokenBatch(input_ids=input_ids, target_ids=target_ids, token_count=input_ids.numel())


def _batch_sequence_from_stream(
    tokens: torch.Tensor,
    seq_len: int,
    window_stride: int,
    batch_size: int,
    *,
    pin_memory: bool,
    data_seed: int | None,
    shuffle: bool,
) -> LazyTokenBatchSequence:
    required_len = seq_len + 1
    if tokens.numel() < required_len:
        raise ValidationError(
            f"token-id corpus split must contain at least {required_len} tokens, got {tokens.numel()}"
        )
    starts = torch.arange(0, tokens.numel() - required_len + 1, window_stride, dtype=torch.long)
    if starts.numel() < 2:
        raise ValidationError(
            f"token-id corpus split must yield at least 2 sequences of length {seq_len}, got {starts.numel()}"
        )
    if shuffle and data_seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(data_seed)
        starts = starts[torch.randperm(starts.numel(), generator=generator)]

    return LazyTokenBatchSequence(
        tokens=tokens,
        starts=starts,
        seq_len=seq_len,
        batch_size=batch_size,
        pin_memory=pin_memory,
    )


def _tokenizer_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValidationError("token-id corpus manifest must contain object field 'tokenizer'")
    vocab_size = tokenizer.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValidationError("token-id corpus tokenizer.vocab_size must be a positive integer")
    pad_token_id = tokenizer.get("pad_token_id", -100)
    if not isinstance(pad_token_id, int):
        raise ValidationError("token-id corpus tokenizer.pad_token_id must be an integer when present")
    return tokenizer


def load_tokenized_corpus(
    corpus_spec: TokenIdCorpusSpec,
    *,
    seq_len: int,
    window_stride: int,
    batch_size: int,
    data_seed: int | None,
    shuffle_train: bool = False,
    pin_memory: bool = False,
) -> ByteCorpusBundle:
    corpus_spec.validate()
    manifest = _load_manifest(corpus_spec.manifest_path)
    tokenizer = _tokenizer_payload(manifest)
    train_tokens = _load_split_tokens(corpus_spec.manifest_path, manifest, "train")
    eval_tokens = _load_split_tokens(corpus_spec.manifest_path, manifest, "eval")
    train_batches = _batch_sequence_from_stream(
        train_tokens,
        seq_len,
        window_stride,
        batch_size,
        pin_memory=pin_memory,
        data_seed=data_seed,
        shuffle=shuffle_train,
    )
    eval_batches = _batch_sequence_from_stream(
        eval_tokens,
        seq_len,
        window_stride,
        batch_size,
        pin_memory=pin_memory,
        data_seed=None,
        shuffle=False,
    )
    corpus_stats = {
        "files": [repo_relative(corpus_spec.manifest_path)],
        "corpus_name": manifest.get("corpus_name", corpus_spec.corpus_name),
        "corpus_format": "token-id-shards",
        "manifest_path": repo_relative(corpus_spec.manifest_path),
        "tokenizer": tokenizer,
        "vocab_size": int(tokenizer["vocab_size"]),
        "pad_token_id": int(tokenizer.get("pad_token_id", -100)),
        "total_tokens": int(train_tokens.numel() + eval_tokens.numel()),
        "train_tokens": int(train_tokens.numel()),
        "eval_tokens": int(eval_tokens.numel()),
        "total_sequences": train_batches.sequence_count + eval_batches.sequence_count,
        "train_sequences": train_batches.sequence_count,
        "eval_sequences": eval_batches.sequence_count,
        "seq_len": seq_len,
        "window_stride": window_stride,
        "data_seed": data_seed,
        "shuffle_train": shuffle_train,
    }
    return ByteCorpusBundle(corpus_stats=corpus_stats, train_batches=train_batches, eval_batches=eval_batches)
