from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from python.data.byte_corpus import load_byte_corpus
from python.data.tokenized_corpus import load_tokenized_corpus
from python.specs.common import JsonlCorpusSpec, TokenIdCorpusSpec


class ByteCorpusTests(unittest.TestCase):
    def test_corpus_batches_stay_host_resident_and_data_seed_controls_shuffle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            train_rows = [
                {"text": "abcdefghijklmno"},
                {"text": "pqrstuvwxyzabcd"},
                {"text": "efghijklmnopqrs"},
            ]
            eval_rows = [
                {"text": "tuvwxyzabcdefghi"},
            ]
            train_path.write_text("".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8")
            eval_path.write_text("".join(json.dumps(row) + "\n" for row in eval_rows), encoding="utf-8")
            corpus = JsonlCorpusSpec(
                corpus_name="tmp-jsonl",
                train_path=train_path,
                eval_path=eval_path,
            )

            baseline = load_byte_corpus(
                corpus,
                seq_len=4,
                window_stride=2,
                batch_size=1,
                data_seed=None,
                shuffle_train=False,
            )
            shuffled_a = load_byte_corpus(
                corpus,
                seq_len=4,
                window_stride=2,
                batch_size=1,
                data_seed=11,
                shuffle_train=True,
            )
            shuffled_b = load_byte_corpus(
                corpus,
                seq_len=4,
                window_stride=2,
                batch_size=1,
                data_seed=17,
                shuffle_train=True,
            )

            self.assertEqual(baseline.train_batches[0].input_ids.device.type, "cpu")
            self.assertEqual(baseline.eval_batches[0].input_ids.device.type, "cpu")

            baseline_order = [tuple(batch.input_ids.reshape(-1).tolist()) for batch in baseline.train_batches]
            shuffled_a_order = [tuple(batch.input_ids.reshape(-1).tolist()) for batch in shuffled_a.train_batches]
            shuffled_b_order = [tuple(batch.input_ids.reshape(-1).tolist()) for batch in shuffled_b.train_batches]

            self.assertNotEqual(baseline_order, shuffled_a_order)
            self.assertNotEqual(shuffled_a_order, shuffled_b_order)
            self.assertTrue(shuffled_a.corpus_stats["shuffle_train"])

    def test_tokenized_corpus_manifest_batches_and_vocab_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            torch.save({"tokens": torch.arange(1, 33, dtype=torch.int32)}, root / "train-00000.pt")
            torch.save({"tokens": torch.arange(33, 49, dtype=torch.int32)}, root / "eval-00000.pt")
            manifest = {
                "schema_version": 1,
                "corpus_name": "tmp-tokenized",
                "tokenizer": {
                    "kind": "sentencepiece",
                    "vocab_size": 64,
                    "unk_id": 0,
                    "bos_id": 1,
                    "eos_id": 2,
                    "pad_token_id": -100,
                },
                "splits": {
                    "train": {"shards": [{"path": "train-00000.pt", "token_count": 32}]},
                    "eval": {"shards": [{"path": "eval-00000.pt", "token_count": 16}]},
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            corpus = load_tokenized_corpus(
                TokenIdCorpusSpec(manifest_path=manifest_path),
                seq_len=4,
                window_stride=4,
                batch_size=2,
                data_seed=7,
                shuffle_train=True,
            )

            self.assertEqual(corpus.corpus_stats["corpus_format"], "token-id-shards")
            self.assertEqual(corpus.corpus_stats["vocab_size"], 64)
            self.assertEqual(corpus.corpus_stats["pad_token_id"], -100)
            self.assertEqual(corpus.train_batches[0].input_ids.shape, (2, 4))
            self.assertEqual(corpus.train_batches[0].target_ids.shape, (2, 4))
            self.assertEqual(corpus.eval_batches[0].input_ids.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
