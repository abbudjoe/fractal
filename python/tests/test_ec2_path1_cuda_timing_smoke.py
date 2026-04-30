from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ec2_path1_cuda_timing_smoke.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ec2_path1_cuda_timing_smoke", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_command_wires_current_kernel_contract() -> None:
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "--job-name",
            "fractal-ec2-contract-test",
            "--optimizer-profile",
            "adam-fused",
            "--head-loss-backend",
            "compiled",
            "--ffn-backend",
            "compiled",
            "--primitive-runtime-backend",
            "triton",
            "--cuda-graph-step",
            "--position-encoding-kind",
            "learned",
            "--attention-position-contract",
            "attention-only",
            "--parcae-control-position-kind",
            "learned",
            "--parcae-control-position-scale-init",
            "0.02",
            "--parcae-control-stride",
            "4",
            "--parcae-loop-layer-count",
            "1",
        ]
    )

    command = module._remote_command(args)

    assert command[0:2] == ["python", "scripts/v3a_python_path1_timing.py"]
    assert "--optimizer-profile" in command
    assert command[command.index("--optimizer-profile") + 1] == "adam-fused"
    assert "--head-loss-backend" in command
    assert command[command.index("--head-loss-backend") + 1] == "compiled"
    assert "--ffn-backend" in command
    assert command[command.index("--ffn-backend") + 1] == "compiled"
    assert "--primitive-runtime-backend" in command
    assert command[command.index("--primitive-runtime-backend") + 1] == "triton"
    assert "--cuda-graph-step" in command
    assert "--parcae-control-position-kind" in command
    assert command[command.index("--parcae-control-position-kind") + 1] == "learned"
    assert "--parcae-control-position-scale-init" in command
    assert command[command.index("--parcae-control-position-scale-init") + 1] == "0.02"
    assert "--parcae-control-stride" in command
    assert command[command.index("--parcae-control-stride") + 1] == "4"
    assert "--parcae-loop-layer-count" in command
    assert command[command.index("--parcae-loop-layer-count") + 1] == "1"


def test_remote_command_omits_p20_control_flags_for_attention_only_control() -> None:
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "--job-name",
            "fractal-ec2-attention-control-test",
            "--scaffold-profile",
            "standard",
            "--cuda-graph-step",
        ]
    )

    command = module._remote_command(args)

    assert "--scaffold-profile" in command
    assert command[command.index("--scaffold-profile") + 1] == "standard"
    assert "--cuda-graph-step" in command
    assert "--parcae-control-position-kind" not in command
    assert "--parcae-control-position-scale-init" not in command
    assert "--parcae-control-stride" not in command


def test_user_data_creates_timing_directory_before_redirect() -> None:
    module = _load_module()
    args = module.build_parser().parse_args(["--job-name", "fractal-ec2-user-data-test"])

    user_data = module._user_data(args, source_url="https://example.com/source.tar.gz")

    mkdir_index = user_data.index("mkdir -p /opt/fractal/source /opt/fractal/out/timing")
    redirect_index = user_data.index("> /opt/fractal/out/timing/full_timing.json")
    assert mkdir_index < redirect_index
    assert '"optimizer_profile": report.get("optimizer_profile")' in user_data
    assert '"cuda_graph_step": report.get("cuda_graph_step")' in user_data
