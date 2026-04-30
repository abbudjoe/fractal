from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from python.data.byte_corpus import TokenBatch
from python.runtime.train_eval import language_model_cross_entropy, mark_compiler_step_boundary, run_training_benchmark


class _TinyLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(vocab_size))

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        return self.logits.view(1, 1, -1).expand(batch, seq_len, -1)


class _TinyForwardLossLanguageModel(_TinyLanguageModel):
    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__(vocab_size)
        self.loss_call_count = 0

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.float().unsqueeze(-1) * 0.0 + self.logits.view(1, 1, -1)

    def loss_from_hidden(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        pad_token: int,
    ) -> torch.Tensor:
        self.loss_call_count += 1
        return F.cross_entropy(
            hidden.reshape(-1, hidden.shape[-1]),
            target_ids.reshape(-1),
            ignore_index=pad_token,
        )


class TrainingBenchmarkTests(unittest.TestCase):
    def test_language_model_cross_entropy_matches_dense_cross_entropy_without_padding(self) -> None:
        model = _TinyLanguageModel(vocab_size=5)
        input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
        target_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        loss = language_model_cross_entropy(
            model,
            input_ids,
            target_ids,
            pad_token=-100,
        )
        logits = model.forward_logits(input_ids)
        expected = F.cross_entropy(logits.reshape(-1, 5), target_ids.reshape(-1), ignore_index=-100)

        self.assertTrue(torch.allclose(loss, expected))

    def test_language_model_cross_entropy_matches_dense_cross_entropy_with_ignore_index(self) -> None:
        model = _TinyLanguageModel(vocab_size=5)
        input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
        target_ids = torch.tensor([[1, -100, 3]], dtype=torch.long)

        loss = language_model_cross_entropy(
            model,
            input_ids,
            target_ids,
            pad_token=-100,
        )
        logits = model.forward_logits(input_ids)
        expected = F.cross_entropy(logits.reshape(-1, 5), target_ids.reshape(-1), ignore_index=-100)

        self.assertTrue(torch.allclose(loss, expected))

    def test_run_training_benchmark_uses_forward_loss_contract_when_available(self) -> None:
        model = _TinyForwardLossLanguageModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = TokenBatch(
            input_ids=torch.tensor([[0, 1]], dtype=torch.long),
            target_ids=torch.tensor([[1, 2]], dtype=torch.long),
            token_count=2,
        )

        run_training_benchmark(
            model=model,
            optimizer=optimizer,
            train_batches=[batch],
            eval_batches=[batch],
            train_steps=2,
            eval_batch_count=1,
            autocast_dtype=None,
            pad_token=-100,
            device=torch.device("cpu"),
            report_model_label="tiny-lm",
            implementation_kind="unit-test",
            note="",
            config_payload={},
            corpus_payload={},
        )

        self.assertEqual(model.loss_call_count, 4)

    def test_sparse_train_loss_recording_avoids_skipped_step_item_calls(self) -> None:
        model = _TinyLanguageModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = TokenBatch(
            input_ids=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            target_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            token_count=4,
        )
        original_item = torch.Tensor.item
        item_call_count = 0

        def counted_item(tensor):
            nonlocal item_call_count
            item_call_count += 1
            return original_item(tensor)

        with unittest.mock.patch.object(torch.Tensor, "item", counted_item):
            report = run_training_benchmark(
                model=model,
                optimizer=optimizer,
                train_batches=[batch],
                eval_batches=[batch],
                train_steps=5,
                eval_batch_count=1,
                autocast_dtype=None,
                pad_token=-100,
                device=torch.device("cpu"),
                report_model_label="tiny-lm",
                implementation_kind="unit-test",
                note="",
                config_payload={},
                corpus_payload={},
                train_loss_record_interval=3,
            )

        self.assertEqual([record.step for record in report.train_steps], [1, 3, 5])
        self.assertEqual(item_call_count, 5)

    def test_sparse_train_loss_recording_requires_positive_interval(self) -> None:
        model = _TinyLanguageModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = TokenBatch(
            input_ids=torch.tensor([[0, 1]], dtype=torch.long),
            target_ids=torch.tensor([[1, 2]], dtype=torch.long),
            token_count=2,
        )

        with self.assertRaises(ValueError):
            run_training_benchmark(
                model=model,
                optimizer=optimizer,
                train_batches=[batch],
                eval_batches=[batch],
                train_steps=1,
                eval_batch_count=1,
                autocast_dtype=None,
                pad_token=-100,
                device=torch.device("cpu"),
                report_model_label="tiny-lm",
                implementation_kind="unit-test",
                note="",
                config_payload={},
                corpus_payload={},
                train_loss_record_interval=0,
            )

    def test_cuda_compiler_step_boundary_marks_cudagraph_steps_when_available(self) -> None:
        compiler = getattr(torch, "compiler", None)
        if compiler is None or not hasattr(compiler, "cudagraph_mark_step_begin"):
            self.skipTest("torch.compiler.cudagraph_mark_step_begin is unavailable")

        with mock.patch.object(compiler, "cudagraph_mark_step_begin") as mark_step_begin:
            mark_compiler_step_boundary("cpu")
            mark_compiler_step_boundary("cuda")

        mark_step_begin.assert_called_once_with()
