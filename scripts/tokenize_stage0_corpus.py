#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokenize Stage 0 text corpora into reusable token-id shards.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--jsonl-train-path", type=Path)
    source.add_argument("--parquet-dir", type=Path)
    parser.add_argument("--jsonl-eval-path", type=Path)
    parser.add_argument("--parquet-glob", default="**/*.parquet")
    parser.add_argument("--tokenizer-model", type=Path, default=REPO_ROOT / "experiments/stage0/assets/open_llama_3b_v2/tokenizer.model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-name", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--parquet-batch-size", type=int, default=512)
    parser.add_argument("--shard-token-count", type=int, default=1_000_000)
    parser.add_argument("--max-train-tokens", type=int)
    parser.add_argument("--max-eval-tokens", type=int)
    parser.add_argument("--eval-doc-stride", type=int, default=100)
    parser.add_argument("--add-bos", action="store_true")
    parser.add_argument("--no-add-eos", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _load_sentencepiece(model_path: Path):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "sentencepiece is required for tokenizer preprocessing. "
            "Install it with: uv pip install --python .venv/bin/python sentencepiece"
        ) from exc
    processor = spm.SentencePieceProcessor()
    if not processor.Load(str(model_path)):
        raise SystemExit(f"failed to load SentencePiece model: {model_path}")
    return processor


def _jsonl_documents(path: Path, text_field: str) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"failed to parse {path} line {line_index}: {exc}") from exc
            text = row.get(text_field)
            if not isinstance(text, str):
                raise SystemExit(f"{path} line {line_index} missing string field {text_field!r}")
            if text:
                yield text


def _parquet_documents(root: Path, pattern: str, text_field: str, batch_size: int) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required for parquet tokenization. "
            "Install it with: uv pip install --python .venv/bin/python pyarrow"
        ) from exc
    files = sorted(root.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet files matched {root}/{pattern}")
    for path in files:
        parquet_file = pq.ParquetFile(path)
        if text_field not in parquet_file.schema_arrow.names:
            raise SystemExit(f"parquet file {path} does not contain text field {text_field!r}")
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=[text_field]):
            column = batch.column(0)
            for value in column.to_pylist():
                if isinstance(value, str) and value:
                    yield value


def _encode_document(processor: Any, text: str, *, add_bos: bool, add_eos: bool) -> list[int]:
    token_ids = processor.EncodeAsIds(text)
    if add_bos and processor.bos_id() >= 0:
        token_ids.insert(0, int(processor.bos_id()))
    if add_eos and processor.eos_id() >= 0:
        token_ids.append(int(processor.eos_id()))
    return [int(token_id) for token_id in token_ids]


class ShardWriter:
    def __init__(self, output_dir: Path, split: str, shard_token_count: int) -> None:
        self.output_dir = output_dir
        self.split = split
        self.shard_token_count = shard_token_count
        self.buffer: list[int] = []
        self.shards: list[dict[str, Any]] = []
        self.document_count = 0
        self.token_count = 0

    def add(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        self.document_count += 1
        self.buffer.extend(token_ids)
        self.token_count += len(token_ids)
        while len(self.buffer) >= self.shard_token_count:
            self.flush(self.shard_token_count)

    def flush(self, limit: int | None = None) -> None:
        if not self.buffer:
            return
        count = len(self.buffer) if limit is None else min(limit, len(self.buffer))
        token_ids = self.buffer[:count]
        del self.buffer[:count]
        shard_name = f"{self.split}-{len(self.shards):05d}.pt"
        path = self.output_dir / shard_name
        tensor = torch.tensor(token_ids, dtype=torch.int32)
        torch.save(
            {
                "schema_version": 1,
                "split": self.split,
                "tokens": tensor,
            },
            path,
        )
        self.shards.append({"path": shard_name, "token_count": int(tensor.numel())})

    def finish(self) -> None:
        self.flush()


def _tokenize_jsonl_splits(args: argparse.Namespace, processor: Any) -> tuple[ShardWriter, ShardWriter]:
    if args.jsonl_eval_path is None:
        raise SystemExit("--jsonl-eval-path is required with --jsonl-train-path")
    train_writer = ShardWriter(args.output_dir, "train", args.shard_token_count)
    eval_writer = ShardWriter(args.output_dir, "eval", args.shard_token_count)
    add_eos = not args.no_add_eos
    for text in _jsonl_documents(args.jsonl_train_path, args.text_field):
        if args.max_train_tokens is not None and train_writer.token_count >= args.max_train_tokens:
            break
        train_writer.add(_encode_document(processor, text, add_bos=args.add_bos, add_eos=add_eos))
    for text in _jsonl_documents(args.jsonl_eval_path, args.text_field):
        if args.max_eval_tokens is not None and eval_writer.token_count >= args.max_eval_tokens:
            break
        eval_writer.add(_encode_document(processor, text, add_bos=args.add_bos, add_eos=add_eos))
    return train_writer, eval_writer


def _tokenize_parquet(args: argparse.Namespace, processor: Any) -> tuple[ShardWriter, ShardWriter]:
    train_writer = ShardWriter(args.output_dir, "train", args.shard_token_count)
    eval_writer = ShardWriter(args.output_dir, "eval", args.shard_token_count)
    add_eos = not args.no_add_eos
    if args.eval_doc_stride <= 1:
        raise SystemExit("--eval-doc-stride must be greater than 1")
    for doc_index, text in enumerate(
        _parquet_documents(args.parquet_dir, args.parquet_glob, args.text_field, args.parquet_batch_size)
    ):
        train_done = args.max_train_tokens is not None and train_writer.token_count >= args.max_train_tokens
        eval_done = args.max_eval_tokens is not None and eval_writer.token_count >= args.max_eval_tokens
        if train_done and eval_done:
            break
        writer = eval_writer if doc_index % args.eval_doc_stride == 0 else train_writer
        if writer is eval_writer and eval_done:
            writer = train_writer
        if writer is train_writer and train_done:
            writer = eval_writer
        if writer is eval_writer and eval_done:
            continue
        if writer is train_writer and train_done:
            continue
        writer.add(_encode_document(processor, text, add_bos=args.add_bos, add_eos=add_eos))
    return train_writer, eval_writer


def _write_manifest(args: argparse.Namespace, processor: Any, train_writer: ShardWriter, eval_writer: ShardWriter) -> None:
    if not train_writer.shards or not eval_writer.shards:
        raise SystemExit("tokenization produced empty train or eval shards")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus_name": args.corpus_name,
        "source": {
            "kind": "parquet" if args.parquet_dir is not None else "jsonl-text",
            "parquet_dir": str(args.parquet_dir) if args.parquet_dir is not None else None,
            "parquet_glob": args.parquet_glob if args.parquet_dir is not None else None,
            "jsonl_train_path": str(args.jsonl_train_path) if args.jsonl_train_path is not None else None,
            "jsonl_eval_path": str(args.jsonl_eval_path) if args.jsonl_eval_path is not None else None,
            "text_field": args.text_field,
        },
        "tokenizer": {
            "kind": "sentencepiece",
            "model_path": str(args.tokenizer_model),
            "vocab_size": int(processor.GetPieceSize()),
            "unk_id": int(processor.unk_id()),
            "bos_id": int(processor.bos_id()),
            "eos_id": int(processor.eos_id()),
            "pad_token_id": -100,
            "add_bos": bool(args.add_bos),
            "add_eos": not bool(args.no_add_eos),
        },
        "splits": {
            "train": {
                "document_count": train_writer.document_count,
                "token_count": train_writer.token_count,
                "shards": train_writer.shards,
            },
            "eval": {
                "document_count": eval_writer.document_count,
                "token_count": eval_writer.token_count,
                "shards": eval_writer.shards,
            },
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(f"output dir already exists and is non-empty; pass --force to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("*.pt"):
        stale.unlink()
    processor = _load_sentencepiece(args.tokenizer_model)
    if args.parquet_dir is not None:
        train_writer, eval_writer = _tokenize_parquet(args, processor)
    else:
        train_writer, eval_writer = _tokenize_jsonl_splits(args, processor)
    train_writer.finish()
    eval_writer.finish()
    _write_manifest(args, processor, train_writer, eval_writer)
    print(args.output_dir / "manifest.json")
    print(
        f"train_tokens={train_writer.token_count} eval_tokens={eval_writer.token_count} "
        f"vocab_size={processor.GetPieceSize()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
