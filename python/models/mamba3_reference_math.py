from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import torch


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
    batch, rank, nheads, _headdim = q.shape
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
    a = torch.clamp(-torch.nn.functional.softplus(dd_a), max=-float(config["a_floor"]))
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
    print(
        json.dumps(
            {
                "rotated_b": dump_tensor(rotated_b),
                "rotated_c": dump_tensor(rotated_c),
                "next_angle_state": dump_tensor(next_angle_state),
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_step(bundle):
    result = reference_step(
        bundle,
        input_tensor=load_tensor(bundle, "input"),
        angle_state=load_tensor(bundle, "angle_state"),
        ssm_state=load_tensor(bundle, "ssm_state"),
        k_state=load_tensor(bundle, "k_state"),
        v_state=load_tensor(bundle, "v_state"),
    )
    print(
        json.dumps(
            {
                "output": dump_tensor(result["output"]),
                "next_angle_state": dump_tensor(result["next_angle_state"]),
                "next_ssm_state": dump_tensor(result["next_ssm_state"]),
                "next_k_state": dump_tensor(result["next_k_state"]),
                "next_v_state": dump_tensor(result["next_v_state"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


def import_upstream_reference_module(root: Path):
    upstream_path = root / "third_party" / "state-spaces-mamba" / "mamba_ssm" / "modules" / "mamba3.py"
    if not upstream_path.exists():
        raise SystemExit(f"failed to locate upstream reference module at {upstream_path}")
    module_name = "state_spaces_mamba_upstream_reference"
    spec = importlib.util.spec_from_file_location(module_name, upstream_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load upstream reference module from {upstream_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_upstream_step(bundle, repo_root: Path):
    module = import_upstream_reference_module(repo_root)
    input_tensor = load_tensor(bundle, "input")
    config = bundle["config"]
    headdim = config["headdim"]
    model = module.Mamba3(
        d_model=config["d_model"],
        d_state=config["d_state"],
        expand=config["expand"],
        headdim=headdim,
        ngroups=config["ngroups"],
        is_outproj_norm=config["is_outproj_norm"],
        is_mimo=config["is_mimo"],
        mimo_rank=config["mimo_rank"],
        chunk_size=config["chunk_size"],
    )
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.reshape(input_tensor.shape[0], 1, input_tensor.shape[1]))
    print(json.dumps({"output": dump_tensor(output[:, 0, :])}, indent=2, sort_keys=True))


def cli_main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        raise SystemExit(
            "usage: mamba3_pytorch_reference.py <pre-kernel|step|upstream-step> <bundle.json>"
        )

    command, bundle_path_arg = argv
    bundle_path = Path(bundle_path_arg).resolve()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    if command == "pre-kernel":
        run_pre_kernel(bundle)
    elif command == "step":
        run_step(bundle)
    elif command == "upstream-step":
        run_upstream_step(bundle, bundle_path.parents[1])
    else:
        raise SystemExit(f"unknown command: {command}")
    return 0
