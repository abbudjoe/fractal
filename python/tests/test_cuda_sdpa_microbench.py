from __future__ import annotations

import importlib.util
import sys
import tarfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MICROBENCH_PATH = REPO_ROOT / "scripts" / "v3a_cuda_sdpa_microbench.py"
SAGEMAKER_PATH = REPO_ROOT / "scripts" / "sagemaker_cuda_sdpa_microbench.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_shape_spec_parses_named_and_unnamed_specs():
    module = _load_module(MICROBENCH_PATH, "v3a_cuda_sdpa_microbench")

    named = module.parse_shape_spec("rgrp_d480_h10:480:10")
    unnamed = module.parse_shape_spec("448:8")

    assert named.name == "rgrp_d480_h10"
    assert named.d_model == 480
    assert named.head_count == 10
    assert named.head_dim == 48
    assert named.cuda_friendly_head_dim
    assert unnamed.name == "d448_h8"
    assert unnamed.head_dim == 56
    assert unnamed.cuda_friendly_head_dim


def test_local_attention_masks_match_expected_visibility():
    module = _load_module(MICROBENCH_PATH, "v3a_cuda_sdpa_microbench")

    keep = module.local_keep_mask(5, 3, device=torch.device("cpu"))

    expected = torch.tensor(
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, True, True, True, False],
            [False, False, True, True, True],
        ]
    )
    assert torch.equal(keep.cpu(), expected)


def test_run_case_cpu_smoke(tmp_path):
    module = _load_module(MICROBENCH_PATH, "v3a_cuda_sdpa_microbench")
    shape = module.parse_shape_spec("tiny:16:4")

    result = module.run_case(
        shape,
        batch_size=1,
        seq_len=8,
        local_window=4,
        dtype_name="fp32",
        device=torch.device("cpu"),
        mask_mode="local-additive",
        backend="auto",
        mode="forward",
        warmup=0,
        iters=1,
        profile_row_limit=5,
    )

    assert result.success
    assert result.mean_ms is not None
    assert result.mean_ms >= 0.0
    assert any("scaled_dot_product" in key for key in result.detected_profiler_keys)


def test_sagemaker_microbench_request_wires_args(monkeypatch):
    module = _load_module(SAGEMAKER_PATH, "sagemaker_cuda_sdpa_microbench")
    monkeypatch.setenv("FRACTAL_SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/test-sagemaker-role")
    args = module.build_parser().parse_args(
        [
            "--bucket",
            "example-bucket",
            "--job-name",
            "fractal-sdpa-test",
            "--region",
            "us-east-1",
            "--batch-size",
            "8",
            "--shape-specs",
            "tiny:16:4",
            "--mask-modes",
            "causal,local-additive",
            "--backends",
            "auto,math",
            "--iters",
            "3",
            "--install-flash-attn",
            "--flash-attn-version",
            "2.8.3",
        ]
    )

    request = module._training_request(
        args,
        source_s3_prefix="s3://example-bucket/fractal/test/source/",
        output_s3_path="s3://example-bucket/fractal/test/output",
    )

    assert request["ResourceConfig"]["InstanceType"] == "ml.g6.2xlarge"
    assert request["Environment"]["PYTHONUNBUFFERED"] == "1"
    assert request["Environment"]["FRACTAL_SDPA_INSTALL_FLASH_ATTN"] == "1"
    assert request["Environment"]["FRACTAL_SDPA_FLASH_ATTN_VERSION"] == "2.8.3"
    sdpa_args = request["Environment"]["FRACTAL_SDPA_ARGS"]
    assert "--batch-size 8" in sdpa_args
    assert "--shape-specs tiny:16:4" in sdpa_args
    assert "--mask-modes causal,local-additive" in sdpa_args
    assert "--backends auto,math" in sdpa_args
    assert "--iters 3" in sdpa_args


def test_sagemaker_microbench_source_bundle(tmp_path):
    module = _load_module(SAGEMAKER_PATH, "sagemaker_cuda_sdpa_microbench")
    bundle = tmp_path / "source.tar.gz"

    module._stage_source_bundle(REPO_ROOT, bundle)

    with tarfile.open(bundle, "r:gz") as tar:
        names = set(tar.getnames())
    assert "sagemaker_sdpa_entrypoint.py" in names
    assert "scripts/v3a_cuda_sdpa_microbench.py" in names
    assert not any(name.startswith("python/") for name in names)
