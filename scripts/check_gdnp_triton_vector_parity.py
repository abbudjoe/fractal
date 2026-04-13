#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.models.reference_ssm import GdnpFusedSequenceMixer, resolve_reference_ssm_config
from python.specs.path1 import ReferenceSsmProfile


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def _max_rel_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.detach().float() - right.detach().float()).abs()
    denominator = torch.maximum(left.detach().float().abs(), right.detach().float().abs()).clamp_min(1.0e-6)
    return float((numerator / denominator).max().item())


def _within_tolerance(left: torch.Tensor, right: torch.Tensor, *, atol: float, rtol: float) -> bool:
    return bool(torch.allclose(left.detach().float(), right.detach().float(), atol=atol, rtol=rtol))


def _grad_by_name(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            grads[name] = parameter.grad.detach().clone()
    return grads


def run_parity(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        message = "CUDA is not available; skipping GDN/P20 Triton parity check"
        if args.require_cuda:
            print(message, file=sys.stderr)
            return 2
        print(message)
        return 0

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda_device}")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    profile = ReferenceSsmProfile(args.profile)
    if not profile.is_gdnp_fused:
        raise ValueError(f"profile must be a fused GDN/P20 profile, got {profile.value}")

    config = resolve_reference_ssm_config(
        d_model=args.d_model,
        head_count=args.head_count,
        profile=profile,
        dtype_mode=args.dtype,
    )
    torch_mixer = GdnpFusedSequenceMixer(config).to(device=device, dtype=dtype)
    triton_mixer = GdnpFusedSequenceMixer(config).to(device=device, dtype=dtype)
    triton_mixer.load_state_dict(copy.deepcopy(torch_mixer.state_dict()))
    triton_mixer.configure_runtime_policy(
        compile_mode=None,
        primitive_runtime_backend="triton",
    )

    hidden = torch.randn(
        args.batch_size,
        args.seq_len,
        args.d_model,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    triton_hidden = hidden.detach().clone().requires_grad_(True)
    target = torch.randn(
        args.batch_size,
        args.seq_len,
        args.d_model,
        device=device,
        dtype=torch.float32,
    )

    torch_output = torch_mixer(hidden)
    triton_output = triton_mixer(triton_hidden)
    torch_loss = (torch_output.float() * target).mean()
    triton_loss = (triton_output.float() * target).mean()
    torch_loss.backward()
    triton_loss.backward()

    forward_abs = _max_abs_diff(torch_output, triton_output)
    forward_rel = _max_rel_diff(torch_output, triton_output)
    input_grad_abs = _max_abs_diff(hidden.grad, triton_hidden.grad)
    input_grad_rel = _max_rel_diff(hidden.grad, triton_hidden.grad)

    torch_grads = _grad_by_name(torch_mixer)
    triton_grads = _grad_by_name(triton_mixer)
    missing = sorted(set(torch_grads) ^ set(triton_grads))
    param_grad_abs = 0.0
    param_grad_rel = 0.0
    param_grad_failed = False
    worst_param = ""
    for name in sorted(set(torch_grads) & set(triton_grads)):
        abs_diff = _max_abs_diff(torch_grads[name], triton_grads[name])
        rel_diff = _max_rel_diff(torch_grads[name], triton_grads[name])
        if abs_diff > param_grad_abs:
            param_grad_abs = abs_diff
            worst_param = name
        param_grad_rel = max(param_grad_rel, rel_diff)
        if not _within_tolerance(torch_grads[name], triton_grads[name], atol=args.grad_atol, rtol=args.grad_rtol):
            param_grad_failed = True

    print(f"profile={profile.value}")
    print(f"dtype={args.dtype}")
    print(f"shape=batch{args.batch_size}_seq{args.seq_len}_d{args.d_model}_h{args.head_count}")
    print(f"forward_abs={forward_abs:.8g}")
    print(f"forward_rel={forward_rel:.8g}")
    print(f"input_grad_abs={input_grad_abs:.8g}")
    print(f"input_grad_rel={input_grad_rel:.8g}")
    print(f"param_grad_abs={param_grad_abs:.8g}")
    print(f"param_grad_rel={param_grad_rel:.8g}")
    print(f"worst_param={worst_param or 'none'}")
    if missing:
        print(f"missing_grad_names={missing}", file=sys.stderr)
        return 1

    failed = (
        not _within_tolerance(torch_output, triton_output, atol=args.forward_atol, rtol=args.forward_rtol)
        or not _within_tolerance(hidden.grad, triton_hidden.grad, atol=args.grad_atol, rtol=args.grad_rtol)
        or param_grad_failed
    )
    if failed:
        print("GDN/P20 Triton parity check FAILED", file=sys.stderr)
        return 1
    print("GDN/P20 Triton parity check passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Torch vs Triton parity for fused GDN/P20.")
    parser.add_argument("--profile", default=ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH.value)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--head-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--backend", choices=["cuda"], default="cuda")
    parser.add_argument("--forward-atol", type=float, default=1.0e-4)
    parser.add_argument("--forward-rtol", type=float, default=1.0e-3)
    parser.add_argument("--grad-atol", type=float, default=1.0e-4)
    parser.add_argument("--grad-rtol", type=float, default=1.0e-3)
    parser.add_argument("--require-cuda", action="store_true")
    return run_parity(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
