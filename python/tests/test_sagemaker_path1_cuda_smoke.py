from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sagemaker_path1_cuda_smoke.py"
PROMOTION_SCRIPT_PATH = REPO_ROOT / "scripts" / "v3a_python_path1_parcae_h100_promotion.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sagemaker_path1_cuda_smoke", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_promotion_module():
    spec = importlib.util.spec_from_file_location("v3a_python_path1_parcae_h100_promotion", PROMOTION_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_training_image_targets_region():
    module = _load_module()

    image = module._default_training_image("us-east-1")

    assert image.startswith("763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:")
    assert image.endswith("-sagemaker")


def test_lane_list_accepts_only_supported_scaffold_lanes():
    module = _load_module()

    assert module._lane_list("attention-only,parcae-rgrp-control-looped-attention") == [
        "attention-only",
        "parcae-p20-control-looped-attention",
    ]
    assert module._lane_list("gpt2-small,mamba-130m") == [
        "hf-gpt2-small",
        "hf-mamba-130m",
    ]
    assert module._lane_list("mamba-official") == ["official-mamba-130m"]
    assert module._lane_list("parcae-hourglass-rgrp-control-looped-attention") == [
        "parcae-hourglass-p20-control-looped-attention",
    ]
    with pytest.raises(SystemExit):
        module._lane_list("attention-only,not-a-lane")


def test_cuda_attention_shape_contract_rejects_slow_outer_head_dim():
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "--d-model",
            "472",
            "--head-count",
            "8",
        ]
    )

    with pytest.raises(SystemExit, match="not a CUDA SDPA-friendly multiple of 8"):
        module._validate_cuda_attention_shape_contract(args)

    args.allow_slow_attention_head_dim = True
    module._validate_cuda_attention_shape_contract(args)


def test_cuda_attention_shape_contract_skips_external_lm_lanes():
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "--lanes",
            "gpt2-small,mamba-official",
            "--d-model",
            "472",
            "--head-count",
            "8",
            "--attention-kernel",
            "flex-local",
        ]
    )

    module._validate_cuda_attention_shape_contract(args)


def test_cuda_attention_shape_contract_rejects_flex_local_non_power_of_two_head_dim():
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "--d-model",
            "480",
            "--head-count",
            "10",
            "--attention-kernel",
            "flex-local",
            "--allow-slow-attention-head-dim",
        ]
    )

    with pytest.raises(SystemExit, match="power-of-two head_dim"):
        module._validate_cuda_attention_shape_contract(args)


def test_cuda_attention_shape_contract_checks_hourglass_loop_head_dim():
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "--d-model",
            "448",
            "--head-count",
            "8",
            "--lanes",
            "parcae-hourglass-rgrp-control-looped-attention",
            "--parcae-loop-d-model",
            "312",
            "--parcae-loop-head-count",
            "8",
        ]
    )

    with pytest.raises(SystemExit, match="Parcae loop attention head_dim"):
        module._validate_cuda_attention_shape_contract(args)


def test_sagemaker_launcher_defaults_to_bf16():
    module = _load_module()

    args = module.build_parser().parse_args([])

    assert args.dtype == "bf16"


def test_training_request_wires_cuda_smoke_contract(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FRACTAL_SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/test-sagemaker-role")
    args = module.build_parser().parse_args(
        [
            "--bucket",
            "example-bucket",
            "--job-name",
            "fractal-test-job",
            "--region",
            "us-east-1",
            "--lanes",
            "attention-only,parcae-p20-control-looped-attention",
            "--steps",
            "7",
            "--parcae-prelude-norm-kind",
            "rmsnorm",
            "--parcae-backward-steps",
            "1",
            "--parcae-loop-d-model",
            "256",
            "--parcae-loop-head-count",
            "8",
            "--parcae-loop-layer-count",
            "1",
            "--attention-kernel",
            "flex-local",
            "--position-encoding-kind",
            "learned",
            "--attention-position-contract",
            "attention-only",
            "--final-norm-kind",
            "layernorm",
            "--parcae-control-position-kind",
            "learned",
            "--parcae-control-position-scale-init",
            "0.02",
            "--parcae-control-stride",
            "4",
            "--parcae-loop-update-backend",
            "manual-autograd",
            "--optimizer-profile",
            "muon-reference",
            "--muon-weight-decay",
            "0.01",
            "--muon-momentum",
            "0.9",
            "--muon-ns-steps",
            "3",
            "--muon-adjust-lr-fn",
            "match_rms_adamw",
        ]
    )

    request = module._training_request(
        args,
        source_s3_prefix="s3://example-bucket/fractal/test/source/",
        output_s3_path="s3://example-bucket/fractal/test/output",
    )

    assert request["ResourceConfig"]["InstanceType"] == "ml.g6.2xlarge"
    assert request["AlgorithmSpecification"]["TrainingInputMode"] == "File"
    assert request["AlgorithmSpecification"]["ContainerEntrypoint"] == ["bash", "-lc"]
    assert "sagemaker_path1_entrypoint.py" in request["AlgorithmSpecification"]["ContainerArguments"][0]
    assert request["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"].endswith("/source/")
    assert request["Environment"]["FRACTAL_SMOKE_STEPS"] == "7"
    assert request["Environment"]["FRACTAL_SMOKE_TRAIN_LOSS_RECORD_INTERVAL"] == "1"
    assert request["Environment"]["FRACTAL_SMOKE_PARCAE_PRELUDE_NORM_KIND"] == "rmsnorm"
    assert request["Environment"]["FRACTAL_SMOKE_PARCAE_BACKWARD_STEPS"] == "1"
    assert request["Environment"]["FRACTAL_SMOKE_PARCAE_DISCRETIZATION"] == "stable-exp"
    assert request["Environment"]["FRACTAL_SMOKE_PARCAE_LOOP_D_MODEL"] == "256"
    assert request["Environment"]["FRACTAL_SMOKE_PARCAE_LOOP_HEAD_COUNT"] == "8"
    assert request["Environment"]["FRACTAL_SMOKE_PARCAE_LOOP_LAYER_COUNT"] == "1"
    assert request["Environment"]["FRACTAL_SMOKE_ATTENTION_KERNEL"] == "flex-local"
    assert request["Environment"]["FRACTAL_SMOKE_PARCAE_CONTROL_POSITION_KIND"] == "learned"
    assert request["Environment"]["FRACTAL_SMOKE_PARCAE_CONTROL_POSITION_SCALE_INIT"] == "0.02"
    assert request["Environment"]["FRACTAL_SMOKE_PARCAE_CONTROL_STRIDE"] == "4"
    assert request["Environment"]["FRACTAL_SMOKE_PARCAE_LOOP_UPDATE_BACKEND"] == "manual-autograd"
    assert request["Environment"]["FRACTAL_SMOKE_POSITION_ENCODING_KIND"] == "learned"
    assert request["Environment"]["FRACTAL_SMOKE_ATTENTION_POSITION_CONTRACT"] == "attention-only"
    assert request["Environment"]["FRACTAL_SMOKE_MAX_POSITION_EMBEDDINGS"] == "1024"
    assert request["Environment"]["FRACTAL_SMOKE_FINAL_NORM_KIND"] == "layernorm"
    assert request["Environment"]["FRACTAL_SMOKE_OPTIMIZER_PROFILE"] == "muon-reference"
    assert request["Environment"]["FRACTAL_SMOKE_MUON_WEIGHT_DECAY"] == "0.01"
    assert request["Environment"]["FRACTAL_SMOKE_MUON_MOMENTUM"] == "0.9"
    assert request["Environment"]["FRACTAL_SMOKE_MUON_NS_STEPS"] == "3"
    assert request["Environment"]["FRACTAL_SMOKE_MUON_ADJUST_LR_FN"] == "match_rms_adamw"
    assert request["Environment"]["FRACTAL_SMOKE_PRIMITIVE_RUNTIME_BACKEND"] == "torch"
    assert request["Environment"]["FRACTAL_SMOKE_LANES"] == "attention-only,parcae-p20-control-looped-attention"
    assert "HF_TOKEN" not in request["Environment"]


def test_training_request_wires_token_cache_scout_contract(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FRACTAL_SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/test-sagemaker-role")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_for_test")
    args = module.build_parser().parse_args(
        [
            "--runner",
            "token-cache",
            "--bucket",
            "example-bucket",
            "--job-name",
            "fractal-scout-job",
            "--region",
            "us-east-1",
            "--lanes",
            "attention-only,parcae-bx-looped-attention,parcae-rgrp-control-looped-attention",
            "--steps",
            "11",
            "--train-loss-record-interval",
            "64",
            "--dtype",
            "bf16",
            "--parcae-prelude-norm-kind",
            "rmsnorm",
            "--parcae-backward-steps",
            "1",
            "--attention-kernel",
            "flex-local",
            "--position-encoding-kind",
            "learned",
            "--attention-position-contract",
            "attention-only",
            "--parcae-control-position-kind",
            "learned",
            "--parcae-control-stride",
            "4",
            "--parcae-recurrent-compile-mode",
            "max-autotune",
            "--parcae-loop-update-backend",
            "compiled",
            "--parcae-loop-layer-count",
            "1",
            "--profile-path1",
            "--profile-row-limit",
            "17",
            "--install-flash-attn",
            "--flash-attn-version",
            "2.8.3",
            "--optimizer-profile",
            "muon-reference",
            "--muon-ns-steps",
            "2",
        ]
    )

    request = module._training_request(
        args,
        source_s3_prefix="s3://example-bucket/fractal/test/source/",
        output_s3_path="s3://example-bucket/fractal/test/output",
        hf_env_s3_prefix="s3://example-bucket/fractal/test/hf_env/",
    )
    redacted = module._redact_request_for_display(request)

    assert "HF_TOKEN" not in request["Environment"]
    assert "HF_TOKEN" not in redacted["Environment"]
    assert request["InputDataConfig"][1]["ChannelName"] == "hf_env"
    assert request["InputDataConfig"][1]["DataSource"]["S3DataSource"]["S3Uri"] == (
        "s3://example-bucket/fractal/test/hf_env/"
    )
    assert request["Environment"]["FRACTAL_SCOUT_STEPS"] == "11"
    assert request["Environment"]["FRACTAL_SCOUT_TRAIN_LOSS_RECORD_INTERVAL"] == "64"
    assert request["Environment"]["FRACTAL_SCOUT_DTYPE"] == "bf16"
    assert request["Environment"]["FRACTAL_SCOUT_PARCAE_PRELUDE_NORM_KIND"] == "rmsnorm"
    assert request["Environment"]["FRACTAL_SCOUT_PARCAE_BACKWARD_STEPS"] == "1"
    assert request["Environment"]["FRACTAL_SCOUT_PARCAE_DISCRETIZATION"] == "stable-exp"
    assert request["Environment"]["FRACTAL_SCOUT_PARCAE_LOOP_D_MODEL"] == ""
    assert request["Environment"]["FRACTAL_SCOUT_PARCAE_LOOP_HEAD_COUNT"] == ""
    assert request["Environment"]["FRACTAL_SCOUT_PARCAE_LOOP_LAYER_COUNT"] == "1"
    assert request["Environment"]["FRACTAL_SCOUT_ATTENTION_KERNEL"] == "flex-local"
    assert request["Environment"]["FRACTAL_SCOUT_PARCAE_CONTROL_POSITION_KIND"] == "learned"
    assert request["Environment"]["FRACTAL_SCOUT_PARCAE_CONTROL_POSITION_SCALE_INIT"] == "0.01"
    assert request["Environment"]["FRACTAL_SCOUT_PARCAE_CONTROL_STRIDE"] == "4"
    assert request["Environment"]["FRACTAL_SCOUT_PARCAE_RECURRENT_COMPILE_MODE"] == "max-autotune"
    assert request["Environment"]["FRACTAL_SCOUT_PARCAE_LOOP_UPDATE_BACKEND"] == "compiled"
    assert request["Environment"]["FRACTAL_SCOUT_PROFILE"] == "true"
    assert request["Environment"]["FRACTAL_SCOUT_PROFILE_ROW_LIMIT"] == "17"
    assert request["Environment"]["FRACTAL_SCOUT_CUDA_TIMING"] == "false"
    assert request["Environment"]["FRACTAL_SCOUT_TIMING_STEPS"] == "20"
    assert request["Environment"]["FRACTAL_SCOUT_NSYS"] == "false"
    assert request["Environment"]["FRACTAL_SCOUT_NSYS_TRACE"] == "cuda,nvtx,cublas,cudnn"
    assert request["Environment"]["FRACTAL_SCOUT_NSYS_STATS"] == "true"
    assert request["Environment"]["FRACTAL_SCOUT_OPTIMIZER_PROFILE"] == "muon-reference"
    assert request["Environment"]["FRACTAL_SCOUT_MUON_NS_STEPS"] == "2"
    assert request["Environment"]["FRACTAL_SCOUT_INSTALL_FLASH_ATTN"] == "true"
    assert request["Environment"]["FRACTAL_SCOUT_FLASH_ATTN_VERSION"] == "2.8.3"
    assert request["Environment"]["FRACTAL_SCOUT_POSITION_ENCODING_KIND"] == "learned"
    assert request["Environment"]["FRACTAL_SCOUT_ATTENTION_POSITION_CONTRACT"] == "attention-only"
    assert request["Environment"]["FRACTAL_SCOUT_MAX_POSITION_EMBEDDINGS"] == "1024"
    assert request["Environment"]["FRACTAL_SCOUT_FINAL_NORM_KIND"] == "identity"
    assert request["Environment"]["FRACTAL_SCOUT_TOKEN_CACHE_ARTIFACT"].endswith(".tar.zst")
    assert request["Environment"]["FRACTAL_SCOUT_LANES"] == (
        "attention-only,parcae-bx-looped-attention,parcae-p20-control-looped-attention"
    )
    assert "FRACTAL_SMOKE_STEPS" not in request["Environment"]


def test_training_request_wires_external_lm_lanes(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FRACTAL_SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/test-sagemaker-role")
    args = module.build_parser().parse_args(
        [
            "--runner",
            "token-cache",
            "--bucket",
            "example-bucket",
            "--job-name",
            "fractal-hf-baseline-job",
            "--region",
            "us-east-1",
            "--token-cache-s3-uri",
            "s3://example-bucket/fractal/token-caches/fineweb-750m",
            "--lanes",
            "gpt2-small,mamba-official",
            "--steps",
            "1024",
        ]
    )

    request = module._training_request(
        args,
        source_s3_prefix="s3://example-bucket/fractal/test/source/",
        output_s3_path="s3://example-bucket/fractal/test/output",
    )

    assert request["Environment"]["FRACTAL_SCOUT_LANES"] == "hf-gpt2-small,official-mamba-130m"
    assert request["Environment"]["FRACTAL_SCOUT_STEPS"] == "1024"
    assert request["Environment"]["FRACTAL_SCOUT_DTYPE"] == "bf16"


def test_training_request_wires_token_cache_mamba_wheelhouse_channel(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FRACTAL_SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/test-sagemaker-role")
    wheelhouse_uri = (
        "s3://example-bucket/fractal/mamba-wheelhouse/job/output/job/output/model.tar.gz"
    )
    args = module.build_parser().parse_args(
        [
            "--runner",
            "token-cache",
            "--bucket",
            "example-bucket",
            "--job-name",
            "fractal-mamba-wheelhouse-consumer-job",
            "--region",
            "us-east-1",
            "--token-cache-s3-uri",
            "s3://example-bucket/fractal/token-caches/fineweb-750m",
            "--mamba-wheelhouse-s3-uri",
            wheelhouse_uri,
            "--lanes",
            "mamba-official",
        ]
    )

    request = module._training_request(
        args,
        source_s3_prefix="s3://example-bucket/fractal/test/source/",
        output_s3_path="s3://example-bucket/fractal/test/output",
    )

    assert request["Environment"]["FRACTAL_SCOUT_LANES"] == "official-mamba-130m"
    assert request["Environment"]["FRACTAL_SCOUT_MAMBA_WHEELHOUSE_DIR"] == (
        "/opt/ml/input/data/mamba_wheelhouse"
    )
    channels = {channel["ChannelName"]: channel for channel in request["InputDataConfig"]}
    assert channels["mamba_wheelhouse"]["InputMode"] == "File"
    assert channels["mamba_wheelhouse"]["DataSource"]["S3DataSource"]["S3Uri"] == wheelhouse_uri


def test_training_request_wires_mamba_wheelhouse_contract(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FRACTAL_SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/test-sagemaker-role")
    args = module.build_parser().parse_args(
        [
            "--runner",
            "mamba-wheelhouse",
            "--bucket",
            "example-bucket",
            "--job-name",
            "fractal-mamba-wheelhouse-job",
            "--region",
            "us-east-1",
            "--mamba-wheelhouse-torch-cuda-arch-list",
            "8.9;9.0",
            "--mamba-wheelhouse-max-jobs",
            "3",
        ]
    )

    request = module._training_request(
        args,
        source_s3_prefix="s3://example-bucket/fractal/test/source/",
        output_s3_path="s3://example-bucket/fractal/test/output",
    )

    assert request["Environment"]["FRACTAL_MAMBA_WHEELHOUSE_TORCH_CUDA_ARCH_LIST"] == "8.9;9.0"
    assert request["Environment"]["FRACTAL_MAMBA_WHEELHOUSE_MAX_JOBS"] == "3"
    assert request["Environment"]["FRACTAL_MAMBA_WHEELHOUSE_KERNELS_VERSION"] == "0.13.0"
    assert request["Environment"]["FRACTAL_MAMBA_WHEELHOUSE_CAUSAL_CONV1D_VERSION"] == "1.6.1"
    assert request["Environment"]["FRACTAL_MAMBA_WHEELHOUSE_MAMBA_SSM_VERSION"] == "2.3.1"
    assert request["Environment"]["FRACTAL_MAMBA_WHEELHOUSE_TRANSFORMERS_VERSION"] == "5.7.0"
    assert request["Environment"]["FRACTAL_MAMBA_WHEELHOUSE_SENTENCEPIECE_VERSION"] == "0.2.1"
    assert request["InputDataConfig"][0]["ChannelName"] == "source"
    assert len(request["InputDataConfig"]) == 1
    assert "FRACTAL_SCOUT_LANES" not in request["Environment"]
    assert "FRACTAL_SMOKE_LANES" not in request["Environment"]
    assert "HF_TOKEN" not in request["Environment"]


def test_training_request_wires_s3_fastfile_token_cache_without_hf_token(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FRACTAL_SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/test-sagemaker-role")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    args = module.build_parser().parse_args(
        [
            "--runner",
            "token-cache",
            "--bucket",
            "example-bucket",
            "--job-name",
            "fractal-fastfile-job",
            "--region",
            "us-east-1",
            "--token-cache-s3-uri",
            "s3://example-bucket/fractal/token-caches/fineweb-250m",
            "--token-cache-input-mode",
            "FastFile",
        ]
    )

    request = module._training_request(
        args,
        source_s3_prefix="s3://example-bucket/fractal/test/source/",
        output_s3_path="s3://example-bucket/fractal/test/output",
    )

    assert "HF_TOKEN" not in request["Environment"]
    assert request["Environment"]["FRACTAL_SCOUT_DATA_ROOT"] == "/opt/ml/input/data/token_cache"
    assert request["Environment"]["FRACTAL_SCOUT_DTYPE"] == "bf16"
    assert request["Environment"]["FRACTAL_SCOUT_TOKEN_CACHE_DIR"] == "."
    assert request["Environment"]["FRACTAL_SCOUT_FORCE_DOWNLOAD"] == "false"
    assert request["InputDataConfig"][0]["ChannelName"] == "source"
    assert request["InputDataConfig"][1]["ChannelName"] == "token_cache"
    assert request["InputDataConfig"][1]["InputMode"] == "FastFile"
    assert request["InputDataConfig"][1]["DataSource"]["S3DataSource"]["S3Uri"] == (
        "s3://example-bucket/fractal/token-caches/fineweb-250m/"
    )


def test_load_local_hf_token_reads_ignored_env_file(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=value\nHF_TOKEN='hf_from_env_file'\n", encoding="utf-8")

    assert module._load_local_hf_token(env_file) == "hf_from_env_file"


def test_write_hf_env_file_is_owner_only(tmp_path):
    module = _load_module()
    path = tmp_path / "hf.env"

    module._write_hf_env_file(path, token="hf_secret")

    assert path.read_text(encoding="utf-8") == "HF_TOKEN=hf_secret\n"
    assert path.stat().st_mode & 0o777 == 0o600


def test_training_request_wires_token_cache_cuda_timing_contract(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FRACTAL_SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/test-sagemaker-role")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    args = module.build_parser().parse_args(
        [
            "--runner",
            "token-cache",
            "--bucket",
            "example-bucket",
            "--job-name",
            "fractal-timing-job",
            "--region",
            "us-east-1",
            "--token-cache-s3-uri",
            "s3://example-bucket/fractal/token-caches/fineweb-250m",
            "--lanes",
            "parcae-hourglass-rgrp-control-looped-attention",
            "--cuda-timing-path1",
            "--cuda-graph-step",
            "--p20-triton-atomic-transform-grad",
            "--timing-steps",
            "12",
            "--warmup-train-steps",
            "19",
            "--warmup-eval-batches",
            "2",
        ]
    )

    request = module._training_request(
        args,
        source_s3_prefix="s3://example-bucket/fractal/test/source/",
        output_s3_path="s3://example-bucket/fractal/test/output",
    )

    assert request["Environment"]["FRACTAL_SCOUT_CUDA_TIMING"] == "true"
    assert request["Environment"]["FRACTAL_SCOUT_TIMING_STEPS"] == "12"
    assert request["Environment"]["FRACTAL_SCOUT_WARMUP_TRAIN_STEPS"] == "19"
    assert request["Environment"]["FRACTAL_SCOUT_TIMING_WARMUP_TRAIN_STEPS"] == "19"
    assert request["Environment"]["FRACTAL_SCOUT_TIMING_WARMUP_EVAL_BATCHES"] == "2"
    assert request["Environment"]["FRACTAL_SCOUT_CUDA_GRAPH_STEP"] == "true"
    assert request["Environment"]["FRACTAL_P20_TRITON_ATOMIC_TRANSFORM_GRAD"] == "true"
    assert request["Environment"]["FRACTAL_SCOUT_PROFILE"] == "false"
    assert request["Environment"]["FRACTAL_SCOUT_LANES"] == "parcae-hourglass-p20-control-looped-attention"


def test_token_cache_timing_entrypoint_preserves_parcae_loop_update_backend():
    module = _load_module()

    assert '"--parcae-loop-update-backend",' in module.TOKEN_CACHE_ENTRYPOINT
    assert '_env("FRACTAL_SCOUT_PARCAE_LOOP_UPDATE_BACKEND", "eager")' in module.TOKEN_CACHE_ENTRYPOINT


def test_token_cache_entrypoint_forces_attention_only_primitive_backend_to_torch():
    module = _load_module()

    assert 'def _primitive_backend_for_lane(lane: str) -> str:' in module.TOKEN_CACHE_ENTRYPOINT
    assert 'if lane == "attention-only":' in module.TOKEN_CACHE_ENTRYPOINT
    assert 'return "torch"' in module.TOKEN_CACHE_ENTRYPOINT


def test_token_cache_entrypoint_contains_external_lm_lane_contract():
    module = _load_module()

    assert "EXTERNAL_LM_LANES" in module.TOKEN_CACHE_ENTRYPOINT
    assert "v3a_external_lm_baseline.py" in module.TOKEN_CACHE_ENTRYPOINT
    assert "include_transformers=has_external_lanes" in module.TOKEN_CACHE_ENTRYPOINT
    assert "include_mamba_kernels=has_external_mamba_lane" in module.TOKEN_CACHE_ENTRYPOINT
    assert "FRACTAL_SCOUT_MAMBA_WHEELHOUSE_DIR" in module.TOKEN_CACHE_ENTRYPOINT
    assert "Mamba wheelhouse dependency leak" in module.TOKEN_CACHE_ENTRYPOINT
    assert '"--no-index"' in module.TOKEN_CACHE_ENTRYPOINT
    assert '"--no-deps"' in module.TOKEN_CACHE_ENTRYPOINT
    assert "mamba-ssm>=2.2.6.post3" in module.TOKEN_CACHE_ENTRYPOINT
    assert "_external_lm_lane_command" in module.TOKEN_CACHE_ENTRYPOINT


def test_promotion_runner_forces_attention_only_primitive_backend_to_torch():
    module = _load_promotion_module()
    args = module.build_parser().parse_args(
        [
            "--run-label",
            "test",
            "--primitive-runtime-backend",
            "triton",
        ]
    )

    assert module._primitive_runtime_backend_for_lane(args, "attention-only") == "torch"
    assert module._primitive_runtime_backend_for_lane(args, "parcae-p20-control-looped-attention") == "triton"
    assert module._env_kind_for_primitive_backend("torch") == "requirements-only"
    assert module._env_kind_for_primitive_backend("triton") == "primitive-triton"


def test_promotion_runner_classifies_parcae_scaffold_lanes():
    module = _load_promotion_module()

    assert not module._lane_uses_parcae_scaffold("attention-only")
    assert module._lane_uses_parcae_scaffold("parcae-hourglass-p20-control-looped-attention")


def test_training_request_wires_token_cache_nsys_contract(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("FRACTAL_SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/test-sagemaker-role")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    args = module.build_parser().parse_args(
        [
            "--runner",
            "token-cache",
            "--bucket",
            "example-bucket",
            "--job-name",
            "fractal-nsys-job",
            "--region",
            "us-east-1",
            "--token-cache-s3-uri",
            "s3://example-bucket/fractal/token-caches/fineweb-250m",
            "--lanes",
            "parcae-hourglass-rgrp-control-looped-attention",
            "--nsys-path1",
            "--nsys-trace",
            "cuda,nvtx,cublas",
            "--no-nsys-stats",
            "--timing-steps",
            "9",
        ]
    )

    request = module._training_request(
        args,
        source_s3_prefix="s3://example-bucket/fractal/test/source/",
        output_s3_path="s3://example-bucket/fractal/test/output",
    )

    assert request["Environment"]["FRACTAL_SCOUT_NSYS"] == "true"
    assert request["Environment"]["FRACTAL_SCOUT_NSYS_TRACE"] == "cuda,nvtx,cublas"
    assert request["Environment"]["FRACTAL_SCOUT_NSYS_STATS"] == "false"
    assert request["Environment"]["FRACTAL_SCOUT_TIMING_STEPS"] == "9"
    assert request["Environment"]["FRACTAL_SCOUT_PROFILE"] == "false"
    assert request["Environment"]["FRACTAL_SCOUT_CUDA_TIMING"] == "false"
    assert request["Environment"]["FRACTAL_SCOUT_LANES"] == "parcae-hourglass-p20-control-looped-attention"


def test_token_cache_entrypoint_wires_cuda_graph_step_to_timing_not_profile(tmp_path):
    module = _load_module()
    bundle = tmp_path / "source.tar.gz"

    module._stage_source_bundle(REPO_ROOT, bundle, runner="token-cache")

    import tarfile

    with tarfile.open(bundle, "r:gz") as tar:
        entrypoint = tar.extractfile("sagemaker_path1_entrypoint.py")
        assert entrypoint is not None
        source = entrypoint.read().decode("utf-8")
    profile_section = source.split("def _timing_lane_command", maxsplit=1)[0]
    timing_section = source.split("def _timing_lane_command", maxsplit=1)[1]
    assert "--cuda-graph-step" not in profile_section
    assert "--cuda-graph-step" in timing_section
    assert "FRACTAL_SCOUT_OPTIMIZER_PROFILE" in source
    assert "--optimizer-profile" in source


def test_token_cache_entrypoint_contains_nsys_mode_and_nvtx_contract(tmp_path):
    module = _load_module()
    bundle = tmp_path / "source.tar.gz"

    module._stage_source_bundle(REPO_ROOT, bundle, runner="token-cache")

    import tarfile

    with tarfile.open(bundle, "r:gz") as tar:
        entrypoint = tar.extractfile("sagemaker_path1_entrypoint.py")
        assert entrypoint is not None
        source = entrypoint.read().decode("utf-8")
    assert "FRACTAL_SCOUT_NSYS" in source
    assert "FRACTAL_ENABLE_NVTX_RANGES" in source
    assert "nsys not found; falling back to CUDA-event timing" in source
    assert "cuda,nvtx,cublas,cudnn" in source


def test_stage_source_bundle_is_minimal_and_runnable_shape(tmp_path):
    module = _load_module()
    bundle = tmp_path / "source.tar.gz"

    module._stage_source_bundle(REPO_ROOT, bundle)

    assert bundle.exists()
    import tarfile

    with tarfile.open(bundle, "r:gz") as tar:
        names = set(tar.getnames())
    assert "sagemaker_path1_entrypoint.py" in names
    assert "scripts/v3a_python_path1.py" in names
    assert "python/runners/path1_cli.py" in names
    assert "experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl" in names
    assert not any(name.startswith(".venv/") for name in names)
    assert not any(name.startswith("artifacts/") for name in names)


def test_stage_source_bundle_can_include_token_cache_runner(tmp_path):
    module = _load_module()
    bundle = tmp_path / "source.tar.gz"

    module._stage_source_bundle(REPO_ROOT, bundle, runner="token-cache")

    assert bundle.exists()
    import tarfile

    with tarfile.open(bundle, "r:gz") as tar:
        names = set(tar.getnames())
    assert "sagemaker_path1_entrypoint.py" in names
    assert "scripts/v3a_python_path1.py" in names
    assert "scripts/v3a_python_path1_parcae_h100_promotion.py" in names
    assert "scripts/v3a_external_lm_baseline.py" in names
    assert "scripts/v3a_python_path1_profile.py" in names
    assert "scripts/v3a_python_path1_timing.py" in names
    assert "python/runners/path1_cli.py" in names
    assert "experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl" not in names


def test_stage_source_bundle_can_include_mamba_wheelhouse_runner(tmp_path):
    module = _load_module()
    bundle = tmp_path / "source.tar.gz"

    module._stage_source_bundle(REPO_ROOT, bundle, runner="mamba-wheelhouse")

    assert bundle.exists()
    import tarfile

    with tarfile.open(bundle, "r:gz") as tar:
        names = set(tar.getnames())
        entrypoint = tar.extractfile("sagemaker_path1_entrypoint.py")
        assert entrypoint is not None
        source = entrypoint.read().decode("utf-8")
    assert "sagemaker_path1_entrypoint.py" in names
    assert "scripts/v3a_python_path1.py" not in names
    assert "python/runners/path1_cli.py" not in names
    assert "experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl" not in names
    assert "TORCH_CUDA_ARCH_LIST" in source
    assert "pip" in source
    assert "wheel" in source
    assert "--no-build-isolation" in source
    assert "--no-deps" in source
    assert "build-contract.json" in source
    assert "forbidden_wheel_prefixes" in source
    assert "wheelhouse dependency leak" in source
    assert "native_package_specs" in source
    assert "official Mamba fast path unavailable" in source
