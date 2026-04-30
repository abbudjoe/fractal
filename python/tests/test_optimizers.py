from __future__ import annotations

import pytest
import torch

from python.runtime.optimizers import (
    CompositeOptimizer,
    ReferenceMuon,
    build_optimizer,
    split_muon_parameters,
    split_triton_adam_2d_parameters,
)
from python.specs.common import BenchmarkBudgetSpec, ValidationError


class _TinyOptimizerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.position_embedding = torch.nn.Embedding(16, 4)
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.output = torch.nn.Linear(4, 8, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids) + self.position_embedding(torch.arange(input_ids.shape[1]))
        hidden = self.norm(self.linear(hidden))
        return self.output(hidden).mean()

    def optimizer_parameter_groups(self, base_lr: float) -> list[dict[str, object]]:
        return [
            {
                "name": "linear-fast",
                "lr": base_lr * 0.5,
                "params": [self.linear.weight, self.linear.bias],
            },
            {
                "name": "default",
                "lr": base_lr,
                "params": [
                    self.embedding.weight,
                    self.position_embedding.weight,
                    self.norm.weight,
                    self.norm.bias,
                    self.output.weight,
                ],
            },
        ]


def test_split_muon_parameters_keeps_embeddings_norms_biases_and_head_in_fallback() -> None:
    model = _TinyOptimizerModel()

    muon_groups, fallback_groups, split = split_muon_parameters(model, base_lr=1.0e-3)

    assert split.muon_tensor_count == 1
    assert split.muon_parameter_count == model.linear.weight.numel()
    assert split.fallback_tensor_count == 6
    assert split.fallback_parameter_count == (
        model.embedding.weight.numel()
        + model.position_embedding.weight.numel()
        + model.linear.bias.numel()
        + model.norm.weight.numel()
        + model.norm.bias.numel()
        + model.output.weight.numel()
    )
    assert muon_groups[0]["name"] == "muon:linear-fast"
    assert muon_groups[0]["lr"] == pytest.approx(5.0e-4)
    assert sum(len(group["params"]) for group in fallback_groups) == 6


def test_build_optimizer_preserves_adam_default_contract() -> None:
    model = _TinyOptimizerModel()

    optimizer = build_optimizer(model, BenchmarkBudgetSpec(optimizer_profile="adam"))

    assert isinstance(optimizer, torch.optim.Adam)
    assert [group.get("name") for group in optimizer.param_groups] == ["linear-fast", "default"]


def test_build_optimizer_rejects_adam_fused_on_cpu() -> None:
    model = _TinyOptimizerModel()

    with pytest.raises(ValidationError, match="requires CUDA parameters"):
        build_optimizer(model, BenchmarkBudgetSpec(optimizer_profile="adam-fused"))


def test_split_triton_adam_2d_parameters_keeps_cpu_model_in_fallback() -> None:
    model = _TinyOptimizerModel()

    native_groups, fallback_groups, split = split_triton_adam_2d_parameters(model, base_lr=1.0e-3)

    assert native_groups == []
    assert split.native_tensor_count == 0
    assert split.native_parameter_count == 0
    assert split.fallback_tensor_count == 7
    assert sum(len(group["params"]) for group in fallback_groups) == 7


def test_build_optimizer_rejects_triton_adam_2d_on_cpu() -> None:
    model = _TinyOptimizerModel()

    with pytest.raises(ValidationError, match="requires CUDA parameters"):
        build_optimizer(model, BenchmarkBudgetSpec(optimizer_profile="adam-triton-2d"))


def test_build_optimizer_creates_muon_reference_composite() -> None:
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon is not available in this PyTorch build")
    model = _TinyOptimizerModel()

    optimizer = build_optimizer(
        model,
        BenchmarkBudgetSpec(
            optimizer_profile="muon-reference",
            learning_rate=1.0e-3,
            muon_momentum=0.9,
            muon_ns_steps=2,
        ),
    )

    assert isinstance(optimizer, CompositeOptimizer)
    assert len(optimizer.optimizers) == 2
    assert type(optimizer.optimizers[0]).__name__ == "Muon"
    assert isinstance(optimizer.optimizers[1], torch.optim.AdamW)
    assert optimizer.muon_parameter_split.muon_parameter_count == model.linear.weight.numel()


def test_build_optimizer_uses_local_reference_muon_when_torch_muon_is_absent(monkeypatch) -> None:
    monkeypatch.delattr(torch.optim, "Muon", raising=False)
    model = _TinyOptimizerModel()

    optimizer = build_optimizer(model, BenchmarkBudgetSpec(optimizer_profile="muon-reference"))

    assert isinstance(optimizer, CompositeOptimizer)
    assert isinstance(optimizer.optimizers[0], ReferenceMuon)
    assert optimizer.muon_optimizer_kind == "ReferenceMuon"


def test_build_optimizer_rejects_muon_reference_with_cuda_graph_capture() -> None:
    model = _TinyOptimizerModel()

    with pytest.raises(ValidationError, match="not supported with cuda_graph_step"):
        build_optimizer(
            model,
            BenchmarkBudgetSpec(optimizer_profile="muon-reference"),
            capturable=True,
        )


def test_muon_reference_composite_can_step() -> None:
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon is not available in this PyTorch build")
    model = _TinyOptimizerModel()
    optimizer = build_optimizer(model, BenchmarkBudgetSpec(optimizer_profile="muon-reference"))
    before = model.linear.weight.detach().clone()

    loss = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
    loss.backward()
    optimizer.step()

    assert not torch.allclose(before, model.linear.weight.detach())
