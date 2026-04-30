from __future__ import annotations

from pathlib import Path

import pytest

import python.runners.path1_timing as path1_timing
from python.runners.path1 import Path1RunnerRequest
from python.runners.path1_timing import _derived_summary, time_path1_request
from python.specs.common import (
    BenchmarkBudgetSpec,
    BenchmarkRunManifest,
    DeviceRuntimeSpec,
    JsonlCorpusSpec,
    SeedSpec,
    ValidationError,
)
from python.specs.path1 import Path1ModelShape, phase1_attention_only_variant


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CORPUS_DIR = REPO_ROOT / "experiments" / "stage0" / "assets" / "fineweb" / "stage0-local-bench-9row-v1"


def test_derived_summary_includes_cuda_graph_replay_total() -> None:
    summary = _derived_summary(
        {
            "path1.train.step_total": {"total_ms": 100.0},
            "path1.cuda_graph.replay": {"total_ms": 80.0},
        },
        timing_kind="unit-test",
    )

    assert summary["cuda_graph_replay_total_ms"] == 80.0
    assert summary["cuda_graph_replay_step_share"] == 0.8


def test_derived_summary_includes_parcae_loop_region_breakdown() -> None:
    summary = _derived_summary(
        {
            "path1.train.step_total": {"total_ms": 200.0},
            "path1.parcae.injection_p20_scan": {"total_ms": 12.0},
            "path1.parcae.recurrent_blocks": {"total_ms": 40.0},
            "path1.parcae.loop_output_projection": {"total_ms": 8.0},
        },
        timing_kind="unit-test",
    )

    assert summary["parcae_region_totals_ms"]["injection_p20_scan"] == 12.0
    assert summary["parcae_region_totals_ms"]["recurrent_blocks"] == 40.0
    assert summary["parcae_region_totals_ms"]["loop_output_projection"] == 8.0
    assert summary["parcae_region_step_shares"]["injection_p20_scan"] == 0.06
    assert summary["parcae_region_step_shares"]["recurrent_blocks"] == 0.2
    assert "state_mix" not in summary["parcae_region_totals_ms"]


def test_derived_summary_aggregates_multiband_native_targets() -> None:
    summary = _derived_summary(
        {
            "path1.train.step_total": {"total_ms": 400.0},
            "path1.parcae.band0.prepare": {"total_ms": 10.0},
            "path1.parcae.band1.prepare": {"total_ms": 12.0},
            "path1.parcae.injection": {"total_ms": 8.0},
            "path1.parcae.injection_p20_scan": {"total_ms": 20.0},
            "path1.parcae.band0.recurrent_blocks": {"total_ms": 70.0},
            "path1.parcae.band1.recurrent_blocks": {"total_ms": 80.0},
            "path1.parcae.band0.recurrent_residual_mix": {"total_ms": 5.0},
            "path1.parcae.band1.recurrent_residual_mix": {"total_ms": 6.0},
            "path1.parcae.band0.triton_loop_update_forward": {"total_ms": 3.0},
            "path1.parcae.band1.triton_loop_update_forward": {"total_ms": 4.0},
            "path1.parcae.band0.loop_output_projection": {"total_ms": 9.0},
            "path1.parcae.band1.loop_output_projection": {"total_ms": 11.0},
            "path1.attention.qkv_projection": {"total_ms": 13.0},
            "path1.attention.flex_local": {"total_ms": 17.0},
            "path1.attention.output_projection": {"total_ms": 19.0},
            "path1.attention.feedforward_compiled": {"total_ms": 23.0},
        },
        timing_kind="unit-test",
    )

    assert summary["parcae_region_totals_ms"]["band_prepare"] == 22.0
    assert summary["parcae_region_totals_ms"]["recurrent_blocks"] == 150.0
    assert summary["parcae_region_totals_ms"]["recurrent_residual_mix"] == 11.0
    assert summary["parcae_region_totals_ms"]["triton_loop_update_forward"] == 7.0
    assert summary["parcae_region_totals_ms"]["loop_output_projection"] == 20.0
    assert summary["attention_region_totals_ms"]["feedforward_compiled"] == 23.0
    assert summary["native_target_totals_ms"]["parcae_prepare_and_control"] == 30.0
    assert summary["native_target_totals_ms"]["rgrp_scan_control"] == 20.0
    assert summary["native_target_totals_ms"]["loop_update_glue"] == 18.0
    assert summary["native_target_totals_ms"]["recurrent_block_region"] == 150.0
    assert summary["native_target_totals_ms"]["attention_kernel"] == 17.0
    assert summary["native_target_totals_ms"]["attention_ffn"] == 23.0
    assert summary["native_target_rank"][0]["target"] == "recurrent_block_region"
    assert summary["native_target_step_shares"]["recurrent_block_region"] == 0.375


def test_cuda_graph_step_requires_cuda_backend(tmp_path: Path) -> None:
    request = Path1RunnerRequest(
        manifest=BenchmarkRunManifest(
            run_label="unit-cuda-graph-reject",
            implementation_kind="unit-test",
            seed_spec=SeedSpec(model_seed=42),
            corpus=JsonlCorpusSpec(
                train_path=SMOKE_CORPUS_DIR / "train.jsonl",
                eval_path=SMOKE_CORPUS_DIR / "eval.jsonl",
                corpus_name="unit-smoke",
            ),
            budget=BenchmarkBudgetSpec(
                seq_len=8,
                window_stride=8,
                batch_size=1,
                train_steps=1,
                eval_batches=1,
                warmup_eval_batches=0,
                warmup_train_steps=0,
            ),
            runtime=DeviceRuntimeSpec(backend="cpu", dtype="fp32"),
        ),
        variant=phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=16, head_count=4, total_layers=1, local_window=8, ffn_multiplier=2)
        ),
        output_dir=tmp_path,
        output_format="json",
    )

    with pytest.raises(ValidationError, match="cuda_graph_step requires runtime.backend=cuda"):
        time_path1_request(request, output_dir=tmp_path, timing_steps=1, cuda_graph_step=True)


def test_cuda_graph_capture_uses_explicit_non_default_stream_contract() -> None:
    source = Path(path1_timing.__file__).read_text(encoding="utf-8")

    assert "capture_stream = torch.cuda.Stream" in source
    assert "torch.cuda.stream(capture_stream)" in source
    assert "torch.cuda.graph(graph, stream=capture_stream" in source
    assert 'capture_error_mode="thread_local"' in source
