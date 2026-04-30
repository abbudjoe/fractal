from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "v3a_external_lm_baseline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("v3a_external_lm_baseline", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_external_model_registry_names_open_baselines():
    module = _load_module()

    assert module.EXTERNAL_MODEL_SPECS["gpt2-small"].model_id == "openai-community/gpt2"
    assert module.EXTERNAL_MODEL_SPECS["mamba-130m"].model_id == "state-spaces/mamba-130m-hf"
    assert module.EXTERNAL_MODEL_SPECS["official-mamba-130m"].model_id == "state-spaces/mamba-130m"
    assert "not pretrained" in module.EXTERNAL_MODEL_SPECS["gpt2-small"].note
    assert "bypasses the HF Transformers Mamba wrapper" in module.EXTERNAL_MODEL_SPECS["official-mamba-130m"].note


def test_external_lm_parser_defaults_match_token_cache_contract(tmp_path):
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "--external-model",
            "gpt2-small",
            "--tokenized-manifest-path",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-label",
            "external-smoke",
        ]
    )

    assert args.dtype == "bf16"
    assert args.seq_len == 512
    assert args.window_stride == 513
    assert args.batch_size == 32
    assert args.optimizer_profile == "adam"


def test_external_mamba_preflight_refuses_missing_fast_path(monkeypatch):
    module = _load_module()

    fake_transformers = types.ModuleType("transformers")
    fake_models = types.ModuleType("transformers.models")
    fake_mamba = types.ModuleType("transformers.models.mamba")
    fake_modeling_mamba = types.ModuleType("transformers.models.mamba.modeling_mamba")
    fake_modeling_mamba.selective_state_update = None
    fake_modeling_mamba.selective_scan_fn = object()
    fake_modeling_mamba.causal_conv1d_fn = object()
    fake_modeling_mamba.causal_conv1d_update = object()
    fake_modeling_mamba.mamba_inner_fn = object()
    fake_mamba.modeling_mamba = fake_modeling_mamba
    fake_models.mamba = fake_mamba
    fake_transformers.models = fake_models
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.models", fake_models)
    monkeypatch.setitem(sys.modules, "transformers.models.mamba", fake_mamba)
    monkeypatch.setitem(sys.modules, "transformers.models.mamba.modeling_mamba", fake_modeling_mamba)

    try:
        module._assert_mamba_fast_path_available()
    except SystemExit as exc:
        assert "refusing to run invalid sequential fallback" in str(exc)
        assert "selective_state_update" in str(exc)
    else:
        raise AssertionError("expected missing Mamba fast path to abort")


def test_official_mamba_preflight_refuses_missing_native_kernels(monkeypatch):
    module = _load_module()

    fake_mamba_ssm = types.ModuleType("mamba_ssm")
    fake_ops = types.ModuleType("mamba_ssm.ops")
    fake_selective = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    fake_modules = types.ModuleType("mamba_ssm.modules")
    fake_simple = types.ModuleType("mamba_ssm.modules.mamba_simple")
    fake_selective.selective_scan_fn = object()
    fake_selective.mamba_inner_fn = None
    fake_simple.causal_conv1d_fn = object()
    fake_ops.selective_scan_interface = fake_selective
    fake_modules.mamba_simple = fake_simple
    fake_mamba_ssm.ops = fake_ops
    fake_mamba_ssm.modules = fake_modules
    monkeypatch.setitem(sys.modules, "mamba_ssm", fake_mamba_ssm)
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops.selective_scan_interface", fake_selective)
    monkeypatch.setitem(sys.modules, "mamba_ssm.modules", fake_modules)
    monkeypatch.setitem(sys.modules, "mamba_ssm.modules.mamba_simple", fake_simple)

    try:
        module._assert_official_mamba_fast_path_available()
    except SystemExit as exc:
        assert "official Mamba fast path is unavailable" in str(exc)
        assert "mamba_inner_fn" in str(exc)
    else:
        raise AssertionError("expected missing official Mamba native kernel to abort")
