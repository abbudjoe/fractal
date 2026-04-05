#!/usr/bin/env python3
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


def load_tensor(bundle, key):
    spec = bundle[key]
    return torch.tensor(spec["values"], dtype=torch.float32).reshape(spec["shape"])


def dump_tensor(tensor):
    tensor = tensor.detach().cpu().to(torch.float32)
    return {
        "shape": list(tensor.shape),
        "values": tensor.reshape(-1).tolist(),
    }


def rms_norm_last_dim(tensor, weight, eps=1.0e-5):
    denom = torch.sqrt(torch.mean(tensor * tensor, dim=-1, keepdim=True) + eps)
    return tensor / denom * weight.reshape(*([1] * (tensor.dim() - 1)), -1)


def rotate_reference(q, angle_state, angle_proj, dt, rotate_pairwise):
    batch, rank, nheads, headdim = q.shape
    pair_count = angle_state.shape[-1]
    rotary_dim = pair_count * 2
    angle = angle_state + torch.tanh(angle_proj) * dt.unsqueeze(-1) * math.pi
    angle_expanded = angle.unsqueeze(1).expand(-1, rank, -1, -1)
    cos = torch.cos(angle_expanded)
    sin = torch.sin(angle_expanded)

    prefix = q[..., :rotary_dim]
    suffix = q[..., rotary_dim:]
    if rotate_pairwise:
        paired = prefix.reshape(batch, rank, nheads, pair_count, 2)
        first = paired[..., 0]
        second = paired[..., 1]
        rotated_first = first * cos - second * sin
        rotated_second = first * sin + second * cos
        rotated_prefix = torch.stack([rotated_first, rotated_second], dim=-1).reshape(
            batch, rank, nheads, rotary_dim
        )
    else:
        first = prefix[..., :pair_count]
        second = prefix[..., pair_count:rotary_dim]
        rotated_first = first * cos - second * sin
        rotated_second = first * sin + second * cos
        rotated_prefix = torch.cat([rotated_first, rotated_second], dim=-1)

    if suffix.numel() == 0:
        return rotated_prefix, angle
    return torch.cat([rotated_prefix, suffix], dim=-1), angle


def preprocess(bundle, input_tensor):
    config = bundle["config"]
    derived = bundle["derived"]

    in_proj_weight = load_tensor(bundle, "in_proj_weight")
    dt_bias = load_tensor(bundle, "dt_bias")
    b_bias = load_tensor(bundle, "b_bias")
    c_bias = load_tensor(bundle, "c_bias")
    b_norm_weight = load_tensor(bundle, "b_norm_weight")
    c_norm_weight = load_tensor(bundle, "c_norm_weight")

    projected = input_tensor @ in_proj_weight
    batch_size = input_tensor.shape[0]
    d_inner = derived["d_inner"]
    nheads = derived["nheads"]
    rank = derived["mimo_rank"]
    headdim = config["headdim"]
    d_state = config["d_state"]
    ngroups = config["ngroups"]
    bc_width = d_state * ngroups * rank

    z_end = d_inner
    x_end = z_end + d_inner
    b_end = x_end + bc_width
    c_end = b_end + bc_width
    dt_end = c_end + nheads
    a_end = dt_end + nheads
    trap_end = a_end + nheads

    z = projected[:, :z_end].reshape(batch_size, nheads, headdim)
    x = projected[:, z_end:x_end].reshape(batch_size, nheads, headdim)
    raw_b = projected[:, x_end:b_end].reshape(batch_size, rank, ngroups, d_state)
    raw_c = projected[:, b_end:c_end].reshape(batch_size, rank, ngroups, d_state)
    dd_dt = projected[:, c_end:dt_end].reshape(batch_size, nheads)
    dd_a = projected[:, dt_end:a_end].reshape(batch_size, nheads)
    trap_proj = projected[:, a_end:trap_end].reshape(batch_size, nheads)
    angle_proj = projected[:, trap_end:].reshape(batch_size, derived["num_rope_angles"])

    dt = torch.nn.functional.softplus(dd_dt + dt_bias.reshape(1, nheads))
    a = torch.clamp(
        -torch.nn.functional.softplus(dd_a), max=-float(config["a_floor"])
    )
    trap = torch.sigmoid(trap_proj)
    expanded_angles = angle_proj.reshape(batch_size, 1, derived["num_rope_angles"]).repeat(
        1, nheads, 1
    )

    b_normed = rms_norm_last_dim(raw_b, b_norm_weight)
    c_normed = rms_norm_last_dim(raw_c, c_norm_weight)
    heads_per_group = nheads // ngroups
    b = b_normed.reshape(batch_size, rank, ngroups, 1, d_state).repeat(
        1, 1, 1, heads_per_group, 1
    ).reshape(batch_size, rank, nheads, d_state)
    c = c_normed.reshape(batch_size, rank, ngroups, 1, d_state).repeat(
        1, 1, 1, heads_per_group, 1
    ).reshape(batch_size, rank, nheads, d_state)
    b = b + b_bias.reshape(1, rank, nheads, d_state)
    c = c + c_bias.reshape(1, rank, nheads, d_state)

    return {
        "config": config,
        "derived": derived,
        "z": z,
        "x": x,
        "dt": dt,
        "a": a,
        "trap": trap,
        "angles": expanded_angles,
        "b": b,
        "c": c,
    }


def selective_state_update_fused_ref_v2(
    state, A, B, C, xproj, x, zproj, z, dt, B_state, x_state, trap, D, outproj
):
    compute_dtype = torch.float32
    og_dtype = state.dtype
    A_f = A.to(compute_dtype)
    dt_f = dt.to(compute_dtype)
    trap_f = trap.to(compute_dtype)
    D_f = D.to(compute_dtype)
    x_f = x.to(compute_dtype)
    xst_f = x_state.to(compute_dtype)
    B_f = B.to(compute_dtype)
    C_f = C.to(compute_dtype)
    Bst_f = B_state.to(compute_dtype)
    Xp_f = xproj.to(compute_dtype)
    st_f = state.to(compute_dtype)

    alpha = torch.exp(A_f * dt_f)
    beta = (1.0 - trap_f) * dt_f * alpha
    gamma = trap_f * dt_f

    x_vals = x_f[:, None, :, :] * Xp_f[None, :, :, :]
    xs_vals = xst_f[:, None, :, :] * Xp_f[None, :, :, :]

    x_bt_state = torch.einsum(
        "brnh,brns->bnhs", x_vals * gamma.unsqueeze(-1).unsqueeze(1), B_f
    )
    x_bt_prev = torch.einsum(
        "brnh,brns->bnhs", xs_vals * beta.unsqueeze(-1).unsqueeze(1), Bst_f
    )
    new_state = st_f * alpha[:, :, None, None] + x_bt_state + x_bt_prev

    out_r = torch.einsum("bnhs,brns->brnh", new_state, C_f)
    out_r = out_r + (x_vals * D_f[None, :, None])

    if z is not None:
        z_f = z.to(compute_dtype)
        zproj_f = zproj.to(compute_dtype)
        z_vals = z_f[:, None, :, :] * zproj_f[None, :, :, :]
        out_r = out_r * z_vals * torch.sigmoid(z_vals)

    if outproj is not None:
        outproj_f = outproj.to(compute_dtype)
        out = torch.einsum("brnh,rnh->bnh", out_r, outproj_f)
    else:
        out = out_r

    return out.to(og_dtype), new_state.to(og_dtype)


def optional_tensor(bundle, key):
    spec = bundle.get(key)
    return None if spec is None else load_tensor(bundle, key)


def reference_step(bundle, input_tensor, angle_state, ssm_state, k_state, v_state):
    projections = preprocess(bundle, input_tensor)
    config = projections["config"]
    derived = projections["derived"]
    rotate_pairwise = not bool(config["is_mimo"])
    rank = derived["mimo_rank"]
    nheads = derived["nheads"]
    headdim = config["headdim"]

    rotated_b, next_angle_state = rotate_reference(
        projections["b"], angle_state, projections["angles"], projections["dt"], rotate_pairwise
    )
    rotated_c, _ = rotate_reference(
        projections["c"], angle_state, projections["angles"], projections["dt"], rotate_pairwise
    )

    mimo_x = optional_tensor(bundle, "mimo_x")
    mimo_z = optional_tensor(bundle, "mimo_z")
    mimo_o = optional_tensor(bundle, "mimo_o")
    d_skip = load_tensor(bundle, "d_skip")
    out_proj_weight = load_tensor(bundle, "out_proj_weight")

    if mimo_x is None:
        mimo_x = torch.ones(rank, nheads, headdim, dtype=torch.float32)
    if mimo_z is None:
        mimo_z = torch.ones(rank, nheads, headdim, dtype=torch.float32)
    if mimo_o is None:
        mimo_o = torch.ones(rank, nheads, headdim, dtype=torch.float32)

    if config["is_outproj_norm"]:
        raise SystemExit("is_outproj_norm=true is not yet supported by this parity script")

    out, next_ssm_state = selective_state_update_fused_ref_v2(
        state=ssm_state,
        A=projections["a"],
        B=rotated_b,
        C=rotated_c,
        xproj=mimo_x,
        x=projections["x"],
        zproj=mimo_z,
        z=projections["z"],
        dt=projections["dt"],
        B_state=k_state,
        x_state=v_state,
        trap=projections["trap"],
        D=d_skip,
        outproj=mimo_o,
    )
    output = out.reshape(input_tensor.shape[0], derived["d_inner"]) @ out_proj_weight

    return {
        "projections": projections,
        "output": output,
        "next_angle_state": next_angle_state,
        "next_ssm_state": next_ssm_state,
        "next_k_state": rotated_b,
        "next_v_state": projections["x"],
    }


def run_pre_kernel(bundle):
    input_tensor = load_tensor(bundle, "input")
    angle_state = load_tensor(bundle, "angle_state")
    projections = preprocess(bundle, input_tensor)
    rotate_pairwise = not bool(bundle["config"]["is_mimo"])
    rotated_b, next_angle_state = rotate_reference(
        projections["b"],
        angle_state,
        projections["angles"],
        projections["dt"],
        rotate_pairwise,
    )
    rotated_c, _ = rotate_reference(
        projections["c"],
        angle_state,
        projections["angles"],
        projections["dt"],
        rotate_pairwise,
    )

    return {
        "derived": bundle["derived"],
        "z": dump_tensor(projections["z"]),
        "x": dump_tensor(projections["x"]),
        "dt": dump_tensor(projections["dt"]),
        "a": dump_tensor(projections["a"]),
        "trap": dump_tensor(projections["trap"]),
        "angles": dump_tensor(projections["angles"]),
        "b": dump_tensor(projections["b"]),
        "c": dump_tensor(projections["c"]),
        "next_angle_state": dump_tensor(next_angle_state),
        "rotated_b": dump_tensor(rotated_b),
        "rotated_c": dump_tensor(rotated_c),
    }


def run_step(bundle):
    step = reference_step(
        bundle,
        load_tensor(bundle, "input"),
        load_tensor(bundle, "angle_state"),
        load_tensor(bundle, "ssm_state"),
        load_tensor(bundle, "k_state"),
        load_tensor(bundle, "v_state"),
    )
    return {
        "output": dump_tensor(step["output"]),
        "next_angle_state": dump_tensor(step["next_angle_state"]),
        "next_ssm_state": dump_tensor(step["next_ssm_state"]),
        "next_k_state": dump_tensor(step["next_k_state"]),
        "next_v_state": dump_tensor(step["next_v_state"]),
    }


def run_sequence(bundle):
    sequence_input = load_tensor(bundle, "sequence_input")
    batch, seq_len, _ = sequence_input.shape
    derived = bundle["derived"]
    config = bundle["config"]

    angle_state = load_tensor(bundle, "angle_state")
    ssm_state = load_tensor(bundle, "ssm_state")
    k_state = load_tensor(bundle, "k_state")
    v_state = load_tensor(bundle, "v_state")
    outputs = []
    for position in range(seq_len):
        step_input = sequence_input[:, position, :]
        step = reference_step(bundle, step_input, angle_state, ssm_state, k_state, v_state)
        outputs.append(step["output"].reshape(batch, 1, config["d_model"]))
        angle_state = step["next_angle_state"]
        ssm_state = step["next_ssm_state"]
        k_state = step["next_k_state"]
        v_state = step["next_v_state"]

    return {
        "outputs": dump_tensor(torch.cat(outputs, dim=1)),
        "final_angle_state": dump_tensor(angle_state),
        "final_ssm_state": dump_tensor(ssm_state),
        "final_k_state": dump_tensor(k_state),
        "final_v_state": dump_tensor(v_state),
    }


def run_model_smoke(bundle):
    sequence = run_sequence(bundle)
    mixed = load_tensor({"outputs": sequence["outputs"]}, "outputs")
    final_norm_weight = load_tensor(bundle, "final_norm_weight")
    lm_head_weight = load_tensor(bundle, "lm_head_weight")
    normalized = rms_norm_last_dim(mixed, final_norm_weight)
    logits = normalized @ lm_head_weight
    return {
        "logits": dump_tensor(logits),
        "final_angle_state": sequence["final_angle_state"],
        "final_ssm_state": sequence["final_ssm_state"],
        "final_k_state": sequence["final_k_state"],
        "final_v_state": sequence["final_v_state"],
    }


def make_stub_rms_norm_gated():
    class StubRMSNormGated(nn.Module):
        def __init__(
            self,
            dim,
            eps=1.0e-5,
            norm_before_gate=False,
            group_size=None,
            device=None,
            dtype=None,
            **kwargs,
        ):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype or torch.float32))
            self.eps = eps
            self.norm_before_gate = norm_before_gate
            self.group_size = group_size

        def forward(self, tensor, gate=None):
            normalized = rms_norm_last_dim(tensor, self.weight, self.eps)
            if gate is None:
                return normalized
            return normalized * torch.sigmoid(gate)

    return StubRMSNormGated


def official_apply_rotary_qk_inference_fwd(
    q, k, angle_state, angle_proj, dt, bias_q, bias_k, conjugate=False, inplace=False, rotate_pairwise=True
):
    q_biased = q + bias_q.unsqueeze(0)
    k_biased = k + bias_k.unsqueeze(0)
    rotated_q, next_angle_state = rotate_reference(q_biased, angle_state, angle_proj, dt, rotate_pairwise)
    rotated_k, _ = rotate_reference(k_biased, angle_state, angle_proj, dt, rotate_pairwise)
    return rotated_q, rotated_k, next_angle_state


def official_mamba3_step_fn(
    state,
    Bstate,
    Xstate,
    A,
    B,
    C,
    D,
    x,
    dt,
    trap,
    xproj,
    outproj,
    state_out,
    out,
    z,
    zproj,
    tile_D=64,
    num_warps=4,
):
    computed_out, next_state = selective_state_update_fused_ref_v2(
        state=state,
        A=A,
        B=B,
        C=C,
        xproj=xproj,
        x=x,
        zproj=zproj,
        z=z,
        dt=dt,
        B_state=Bstate,
        x_state=Xstate,
        trap=trap,
        D=D,
        outproj=outproj,
    )
    if state_out is None:
        state.copy_(next_state)
    else:
        state_out.copy_(next_state)
    out.copy_(computed_out)
    return out


def load_official_mamba3_class_with_reference_kernel_shims(official_repo):
    official_repo = Path(official_repo)
    module_path = official_repo / "mamba_ssm" / "modules" / "mamba3.py"
    if not module_path.exists():
        raise SystemExit(f"official Mamba3 module not found at {module_path}")

    fake_modules = {}

    def ensure_module(name):
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
            fake_modules[name] = module
        return module

    ensure_module("mamba_ssm")
    ensure_module("mamba_ssm.ops")
    ensure_module("mamba_ssm.ops.triton")
    ensure_module("mamba_ssm.ops.triton.mamba3")
    ensure_module("mamba_ssm.ops.tilelang")
    ensure_module("mamba_ssm.ops.tilelang.mamba3")
    ensure_module("mamba_ssm.ops.cute")
    ensure_module("mamba_ssm.ops.cute.mamba3")

    layernorm_mod = ensure_module("mamba_ssm.ops.triton.layernorm_gated")
    layernorm_mod.RMSNorm = make_stub_rms_norm_gated()

    angle_mod = ensure_module("mamba_ssm.ops.triton.angle_cumsum")
    angle_mod.angle_dt = lambda angles, dt: angles + torch.tanh(angles) * dt.unsqueeze(-1) * math.pi

    siso_mod = ensure_module("mamba_ssm.ops.triton.mamba3.mamba3_siso_combined")
    siso_mod.mamba3_siso_combined = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("siso combined kernel is not available in the CPU reference harness")
    )

    mimo_mod = ensure_module("mamba_ssm.ops.tilelang.mamba3.mamba3_mimo")
    mimo_mod.mamba3_mimo = object()

    # The official module's CPU-unavailable rotary and fused-step kernels are replaced with
    # explicit reference-kernel shims here. This validates wiring, projection layout, and
    # cache/state contracts through the official module surface, but it is not a kernel-level
    # differential against upstream Triton/CUTE implementations.
    rotary_mod = ensure_module("mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step")
    rotary_mod.apply_rotary_qk_inference_fwd = official_apply_rotary_qk_inference_fwd

    step_mod = ensure_module("mamba_ssm.ops.cute.mamba3.mamba3_step_fn")
    step_mod.mamba3_step_fn = official_mamba3_step_fn

    spec = importlib.util.spec_from_file_location("fractal_official_mamba3_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Mamba3


def set_official_module_weights(module, bundle):
    in_proj_weight = load_tensor(bundle, "in_proj_weight")
    dt_bias = load_tensor(bundle, "dt_bias")
    b_bias = load_tensor(bundle, "b_bias")
    c_bias = load_tensor(bundle, "c_bias")
    b_norm_weight = load_tensor(bundle, "b_norm_weight")
    c_norm_weight = load_tensor(bundle, "c_norm_weight")
    d_skip = load_tensor(bundle, "d_skip")
    out_proj_weight = load_tensor(bundle, "out_proj_weight")
    mimo_x = optional_tensor(bundle, "mimo_x")
    mimo_z = optional_tensor(bundle, "mimo_z")
    mimo_o = optional_tensor(bundle, "mimo_o")

    with torch.no_grad():
        module.in_proj.weight.copy_(in_proj_weight.transpose(0, 1))
        module.dt_bias.copy_(dt_bias)
        module.B_bias.copy_(b_bias.permute(1, 0, 2))
        module.C_bias.copy_(c_bias.permute(1, 0, 2))
        module.B_norm.weight.copy_(b_norm_weight)
        module.C_norm.weight.copy_(c_norm_weight)
        module.D.copy_(d_skip)
        module.out_proj.weight.copy_(out_proj_weight.transpose(0, 1))
        if mimo_x is not None:
            module.mimo_x.copy_(mimo_x.permute(1, 0, 2))
        if mimo_z is not None:
            module.mimo_z.copy_(mimo_z.permute(1, 0, 2))
        if mimo_o is not None:
            module.mimo_o.copy_(mimo_o.permute(1, 0, 2))


def instantiate_official_mamba3(bundle):
    official_repo = bundle.get("official_repo")
    if not official_repo:
        raise SystemExit("official_repo is required for official module wiring-shim modes")
    Mamba3 = load_official_mamba3_class_with_reference_kernel_shims(official_repo)
    config = bundle["config"]
    module = Mamba3(
        d_model=config["d_model"],
        d_state=config["d_state"],
        expand=config["expand"],
        headdim=config["headdim"],
        ngroups=config["ngroups"],
        rope_fraction=1.0 if bundle["derived"]["rotary_dim_divisor"] == 2 else 0.5,
        dt_min=config["dt_min"],
        dt_max=config["dt_max"],
        dt_init_floor=config["dt_init_floor"],
        A_floor=config["a_floor"],
        is_outproj_norm=config["is_outproj_norm"],
        is_mimo=config["is_mimo"],
        mimo_rank=config["mimo_rank"],
        chunk_size=config["chunk_size"],
        device="cpu",
        dtype=torch.float32,
    )
    set_official_module_weights(module, bundle)
    return module


def run_official_module_step(bundle):
    module = instantiate_official_mamba3(bundle)
    input_tensor = load_tensor(bundle, "input")
    angle_state = load_tensor(bundle, "angle_state")
    ssm_state = load_tensor(bundle, "ssm_state")
    k_state = load_tensor(bundle, "k_state")
    v_state = load_tensor(bundle, "v_state")
    with torch.no_grad():
        output, next_angle_state, next_ssm_state, next_k_state, next_v_state = module.step(
            input_tensor,
            angle_state.clone(),
            ssm_state.clone(),
            k_state.clone(),
            v_state.clone(),
        )
    return {
        "output": dump_tensor(output),
        "next_angle_state": dump_tensor(next_angle_state),
        "next_ssm_state": dump_tensor(next_ssm_state),
        "next_k_state": dump_tensor(next_k_state),
        "next_v_state": dump_tensor(next_v_state),
    }


def run_official_module_sequence(bundle):
    module = instantiate_official_mamba3(bundle)
    sequence_input = load_tensor(bundle, "sequence_input")
    batch, seq_len, _ = sequence_input.shape
    angle_state = load_tensor(bundle, "angle_state")
    ssm_state = load_tensor(bundle, "ssm_state")
    k_state = load_tensor(bundle, "k_state")
    v_state = load_tensor(bundle, "v_state")
    outputs = []
    with torch.no_grad():
        for position in range(seq_len):
            output, angle_state, ssm_state, k_state, v_state = module.step(
                sequence_input[:, position, :],
                angle_state,
                ssm_state,
                k_state,
                v_state,
            )
            outputs.append(output.reshape(batch, 1, module.d_model))
    return {
        "outputs": dump_tensor(torch.cat(outputs, dim=1)),
        "final_angle_state": dump_tensor(angle_state),
        "final_ssm_state": dump_tensor(ssm_state),
        "final_k_state": dump_tensor(k_state),
        "final_v_state": dump_tensor(v_state),
    }


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: mamba3_pytorch_reference.py <input.json> <output.json>")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    bundle = json.loads(input_path.read_text())
    mode = bundle.get("mode", "pre-kernel")
    if mode == "pre-kernel":
        output = run_pre_kernel(bundle)
    elif mode == "step":
        output = run_step(bundle)
    elif mode == "sequence":
        output = run_sequence(bundle)
    elif mode == "official-module-wiring-step":
        output = run_official_module_step(bundle)
    elif mode == "official-module-wiring-sequence":
        output = run_official_module_sequence(bundle)
    elif mode == "model-smoke":
        output = run_model_smoke(bundle)
    else:
        raise SystemExit(f"unsupported mode: {mode}")
    output_path.write_text(json.dumps(output))


if __name__ == "__main__":
    main()
