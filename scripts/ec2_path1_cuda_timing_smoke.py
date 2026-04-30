#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = "codex-eml"
DEFAULT_REGION = "us-east-1"
DEFAULT_BUCKET = "fractal-sagemaker-806880856465-us-east-1"
DEFAULT_PREFIX = "fractal/ec2-path1-cuda-smoke"
DEFAULT_AMI_ID = "ami-00613b158c7a09b63"
DEFAULT_INSTANCE_TYPE = "g6.2xlarge"
DEFAULT_SUBNET_ID = "subnet-0991ac275c3dd36b1"
DEFAULT_SECURITY_GROUP_ID = "sg-070d5d736cc3ef142"
SMOKE_CORPUS_REL = Path("experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1")


def _run(
    command: Sequence[str],
    *,
    capture: bool = False,
    check: bool = True,
) -> str:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(
        list(command),
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return completed.stdout or ""


def _aws(args: argparse.Namespace, aws_args: Sequence[str], *, capture: bool = False, check: bool = True) -> str:
    command = ["aws", *aws_args, "--region", args.region]
    if args.profile:
        command.extend(["--profile", args.profile])
    return _run(command, capture=capture, check=check)


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def _stage_source_bundle(bundle_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="fractal-ec2-src-") as tmp:
        stage = Path(tmp) / "source"
        stage.mkdir()
        _copy_tree(REPO_ROOT / "python", stage / "python")
        (stage / "scripts").mkdir()
        shutil.copy2(REPO_ROOT / "scripts" / "v3a_python_path1_timing.py", stage / "scripts")
        corpus_src = REPO_ROOT / SMOKE_CORPUS_REL
        corpus_dst = stage / SMOKE_CORPUS_REL
        corpus_dst.parent.mkdir(parents=True, exist_ok=True)
        _copy_tree(corpus_src, corpus_dst)
        with tarfile.open(bundle_path, "w:gz") as tar:
            for path in sorted(stage.rglob("*")):
                tar.add(path, arcname=path.relative_to(stage))


def _lane_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--variant",
        "attention-only",
        "--scaffold-profile",
        args.scaffold_profile,
        "--parcae-loop-count",
        str(args.parcae_loop_count),
        "--parcae-backward-steps",
        str(args.parcae_backward_steps),
        "--parcae-prelude-norm-kind",
        args.parcae_prelude_norm_kind,
        "--parcae-loop-d-model",
        str(args.parcae_loop_d_model),
        "--parcae-loop-head-count",
        str(args.parcae_loop_head_count),
        "--parcae-loop-ffn-multiplier",
        str(args.parcae_loop_ffn_multiplier),
        "--position-encoding-kind",
        args.position_encoding_kind,
        "--attention-position-contract",
        args.attention_position_contract,
        "--max-position-embeddings",
        str(args.max_position_embeddings),
    ]
    if args.parcae_loop_layer_count is not None:
        command.extend(["--parcae-loop-layer-count", str(args.parcae_loop_layer_count)])
    if "p20-control" in args.scaffold_profile:
        command.extend(
            [
                "--parcae-control-position-kind",
                args.parcae_control_position_kind,
                "--parcae-control-position-scale-init",
                str(args.parcae_control_position_scale_init),
                "--parcae-control-stride",
                str(args.parcae_control_stride),
                "--parcae-control-state-transform",
                args.parcae_control_state_transform,
            ]
        )
    return command


def _remote_command(args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        "scripts/v3a_python_path1_timing.py",
        "--backend",
        "cuda",
        "--cuda-device",
        "0",
        "--dtype",
        args.dtype,
        "--env-kind",
        "primitive-triton" if args.primitive_runtime_backend == "triton" else "requirements-only",
        "--primitive-runtime-backend",
        args.primitive_runtime_backend,
        "--head-loss-backend",
        args.head_loss_backend,
        "--ffn-backend",
        args.ffn_backend,
        "--jsonl-train-path",
        str(SMOKE_CORPUS_REL / "train.jsonl"),
        "--jsonl-eval-path",
        str(SMOKE_CORPUS_REL / "eval.jsonl"),
        "--seq-len",
        str(args.seq_len),
        "--window-stride",
        str(args.seq_len),
        "--batch-size",
        str(args.batch_size),
        "--steps",
        "1",
        "--eval-batches",
        "1",
        "--warmup-train-steps",
        str(args.warmup_train_steps),
        "--warmup-eval-batches",
        "0",
        "--learning-rate",
        str(args.learning_rate),
        "--optimizer-profile",
        args.optimizer_profile,
        "--seed",
        str(args.seed),
        "--data-seed",
        str(args.data_seed),
        "--d-model",
        str(args.d_model),
        "--head-count",
        str(args.head_count),
        "--total-layers",
        str(args.total_layers),
        "--local-window",
        str(args.local_window),
        "--attention-kernel",
        args.attention_kernel,
        "--ffn-multiplier",
        str(args.ffn_multiplier),
        "--timing-steps",
        str(args.timing_steps),
        "--timing-output-dir",
        "/opt/fractal/out/timing",
        "--run-label",
        args.job_name,
        "--output",
        "json",
        *_lane_args(args),
    ]
    if args.cuda_graph_step:
        command.append("--cuda-graph-step")
    return command


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _user_data(args: argparse.Namespace, *, source_url: str) -> str:
    remote_command = " ".join(_shell_quote(part) for part in _remote_command(args))
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        set -euxo pipefail
        exec > >(tee /dev/console) 2>&1
        echo "FRACTAL_EC2_SMOKE_START $(date -Is)"
        mkdir -p /opt/fractal/source /opt/fractal/out/timing
        cd /opt/fractal
        curl -fL {_shell_quote(source_url)} -o source.tar.gz
        tar -xzf source.tar.gz -C source
        cd source
        python - <<'PY'
        import torch
        print("torch_version", torch.__version__)
        print("cuda_available", torch.cuda.is_available())
        print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
        PY
        {remote_command} > /opt/fractal/out/timing/full_timing.json
        python - <<'PY'
        import json
        from pathlib import Path

        report = json.loads(Path("/opt/fractal/out/timing/full_timing.json").read_text())
        diagnostics = report.get("diagnostics", {{}})
        parcae = diagnostics.get("parcae_looped_attention", {{}})
        derived = report.get("derived_summary", {{}})
        wall = report.get("wall_derived_summary", {{}})
        timing = report.get("timing_summary", {{}})

        def mean_ms(name):
            payload = timing.get(name, {{}})
            return payload.get("mean_ms")

        compact = {{
            "run_label": report.get("run_label"),
            "device": report.get("device"),
            "variant_label": report.get("variant_label"),
            "timing_steps": report.get("timing_steps"),
            "step_total_ms": (derived.get("step_total_ms") or 0.0) / max(report.get("timing_steps") or 1, 1),
            "wall_step_total_ms": (wall.get("step_total_ms") or 0.0) / max(report.get("timing_steps") or 1, 1),
            "attention_call_mean_ms": mean_ms("path1.attention.flex_local") or mean_ms("path1.attention.flash_local") or mean_ms("path1.attention.sdpa"),
            "parcae_loop_step_mean_ms": mean_ms("path1.parcae.loop_step"),
            "parcae_recurrent_block_forward_mean_ms": mean_ms("path1.parcae.recurrent_block_forward"),
            "primitive_scan_mean_ms": mean_ms("path1.primitive.runtime.triton_sequence_scan"),
            "optimizer_profile": report.get("optimizer_profile"),
            "cuda_graph_step": report.get("cuda_graph_step"),
            "cuda_graph_replay_mean_ms": mean_ms("path1.cuda_graph.replay"),
            "head_loss_backend": diagnostics.get("head_loss_backend"),
            "ffn_backend": diagnostics.get("ffn_backend"),
            "prelude_layers": parcae.get("prelude_layers"),
            "recurrent_layers": parcae.get("recurrent_layers"),
            "coda_layers": parcae.get("coda_layers"),
            "configured_loop_layer_count": parcae.get("configured_loop_layer_count"),
            "loop_count": parcae.get("loop_count"),
        }}
        print("FRACTAL_EC2_TIMING_RESULT_START")
        print(json.dumps(compact, indent=2, sort_keys=True))
        print("FRACTAL_EC2_TIMING_RESULT_END")
        PY
        echo "FRACTAL_EC2_SMOKE_DONE $(date -Is)"
        shutdown -h now
        """
    )


def _extract_result(console_output: str) -> dict[str, Any] | None:
    start = "FRACTAL_EC2_TIMING_RESULT_START"
    end = "FRACTAL_EC2_TIMING_RESULT_END"
    if start not in console_output or end not in console_output:
        return None
    payload = console_output.split(start, 1)[1].split(end, 1)[0].strip()
    return json.loads(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a self-terminating EC2 CUDA timing smoke.")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE", DEFAULT_PROFILE))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", DEFAULT_REGION))
    parser.add_argument("--bucket", default=os.environ.get("FRACTAL_SAGEMAKER_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--job-name", default=f"fractal-ec2-path1-smoke-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--ami-id", default=DEFAULT_AMI_ID)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--subnet-id", default=DEFAULT_SUBNET_ID)
    parser.add_argument("--security-group-id", default=DEFAULT_SECURITY_GROUP_ID)
    parser.add_argument("--market", choices=["on-demand", "spot"], default="on-demand")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--primitive-runtime-backend", choices=["torch", "triton"], default="triton")
    parser.add_argument("--head-loss-backend", choices=["dense", "compiled", "streaming-kernel"], default="compiled")
    parser.add_argument("--ffn-backend", choices=["dense", "compiled", "manual-autograd", "triton-gelu", "recompute"], default="compiled")
    parser.add_argument("--optimizer-profile", choices=["adam", "adam-fused", "adam-triton-2d", "muon-reference"], default="adam-fused")
    parser.add_argument("--cuda-graph-step", action="store_true")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-train-steps", type=int, default=3)
    parser.add_argument("--timing-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=448)
    parser.add_argument("--head-count", type=int, default=7)
    parser.add_argument("--total-layers", type=int, default=8)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--attention-kernel", choices=["sdpa", "flex-local", "flash-local"], default="flex-local")
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--scaffold-profile", default="parcae-hourglass-p20-control-looped-attention")
    parser.add_argument("--parcae-loop-count", type=int, default=2)
    parser.add_argument("--parcae-backward-steps", type=int, default=1)
    parser.add_argument("--parcae-prelude-norm-kind", choices=["layernorm", "rmsnorm"], default="rmsnorm")
    parser.add_argument("--parcae-loop-d-model", type=int, default=256)
    parser.add_argument("--parcae-loop-head-count", type=int, default=4)
    parser.add_argument("--parcae-loop-ffn-multiplier", type=int, default=4)
    parser.add_argument("--parcae-loop-layer-count", type=int)
    parser.add_argument("--parcae-control-position-kind", choices=["none", "learned"], default="learned")
    parser.add_argument("--parcae-control-position-scale-init", type=float, default=0.01)
    parser.add_argument("--parcae-control-stride", type=int, default=1)
    parser.add_argument(
        "--parcae-control-state-transform",
        choices=["trainable", "trainable-block-diagonal-8", "frozen-identity"],
        default="trainable",
    )
    parser.add_argument("--position-encoding-kind", choices=["none", "learned"], default="none")
    parser.add_argument("--attention-position-contract", choices=["shared-input", "attention-only"], default="attention-only")
    parser.add_argument("--max-position-embeddings", type=int, default=1024)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with tempfile.TemporaryDirectory(prefix="fractal-ec2-bundle-") as tmp:
        bundle_path = Path(tmp) / "source.tar.gz"
        user_data_path = Path(tmp) / "user-data.sh"
        _stage_source_bundle(bundle_path)
        s3_key = f"{args.prefix.rstrip('/')}/{args.job_name}/source/source.tar.gz"
        source_s3 = f"s3://{args.bucket}/{s3_key}"
        _aws(args, ["s3", "cp", str(bundle_path), source_s3])
        source_url = _aws(args, ["s3", "presign", source_s3, "--expires-in", "3600"], capture=True).strip()
        user_data_path.write_text(_user_data(args, source_url=source_url), encoding="utf-8")

        run_args = [
            "ec2",
            "run-instances",
            "--image-id",
            args.ami_id,
            "--instance-type",
            args.instance_type,
            "--subnet-id",
            args.subnet_id,
            "--security-group-ids",
            args.security_group_id,
            "--instance-initiated-shutdown-behavior",
            "terminate",
            "--user-data",
            f"file://{user_data_path}",
            "--tag-specifications",
            f"ResourceType=instance,Tags=[{{Key=Name,Value={args.job_name}}},{{Key=Project,Value=fractal}}]",
            "--query",
            "Instances[0].InstanceId",
            "--output",
            "text",
        ]
        if args.market == "spot":
            run_args.extend(
                [
                    "--instance-market-options",
                    '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}',
                ]
            )

        try:
            instance_id = _aws(args, run_args, capture=True).strip()
        except subprocess.CalledProcessError as exc:
            print("ec2_launch_failed=true", flush=True)
            print(exc.stdout or "", flush=True)
            return exc.returncode
        print(f"instance_id={instance_id}", flush=True)

        deadline = time.time() + args.timeout_seconds
        last_state = ""
        while time.time() < deadline:
            state = _aws(
                args,
                [
                    "ec2",
                    "describe-instances",
                    "--instance-ids",
                    instance_id,
                    "--query",
                    "Reservations[0].Instances[0].State.Name",
                    "--output",
                    "text",
                ],
                capture=True,
                check=False,
            ).strip()
            if state != last_state:
                print(f"state={state}", flush=True)
                last_state = state
            if state in {"terminated", "shutting-down", "stopped"}:
                break
            time.sleep(args.poll_seconds)

        if last_state not in {"terminated", "shutting-down", "stopped"}:
            print("timeout_reached=true; terminating instance", flush=True)
            _aws(args, ["ec2", "terminate-instances", "--instance-ids", instance_id], check=False)

        console = _aws(
            args,
            ["ec2", "get-console-output", "--instance-id", instance_id, "--latest", "--output", "json"],
            capture=True,
            check=False,
        )
        try:
            payload = json.loads(console)
            output = payload.get("Output") or ""
        except json.JSONDecodeError:
            output = console
        result = _extract_result(output)
        if result is None:
            print("fractal_ec2_timing_result_found=false", flush=True)
            print(output[-8000:], flush=True)
            return 2
        print("fractal_ec2_timing_result_found=true", flush=True)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
