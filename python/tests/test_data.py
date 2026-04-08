from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from python.data.byte_corpus import load_byte_corpus
from python.specs.common import JsonlCorpusSpec


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


if __name__ == "__main__":
    unittest.main()
