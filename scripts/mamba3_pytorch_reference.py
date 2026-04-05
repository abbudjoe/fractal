#!/usr/bin/env python3
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


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: mamba3_pytorch_reference.py <input.json> <output.json>")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    bundle = json.loads(input_path.read_text())

    config = bundle["config"]
    derived = bundle["derived"]

    input_tensor = load_tensor(bundle, "input")
    angle_state = load_tensor(bundle, "angle_state")
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

    rotate_pairwise = not bool(config["is_mimo"])
    rotated_b, next_angle_state = rotate_reference(
        b, angle_state, expanded_angles, dt, rotate_pairwise
    )
    rotated_c, _ = rotate_reference(c, angle_state, expanded_angles, dt, rotate_pairwise)

    output = {
        "derived": derived,
        "z": dump_tensor(z),
        "x": dump_tensor(x),
        "dt": dump_tensor(dt),
        "a": dump_tensor(a),
        "trap": dump_tensor(trap),
        "angles": dump_tensor(expanded_angles),
        "b": dump_tensor(b),
        "c": dump_tensor(c),
        "next_angle_state": dump_tensor(next_angle_state),
        "rotated_b": dump_tensor(rotated_b),
        "rotated_c": dump_tensor(rotated_c),
    }
    output_path.write_text(json.dumps(output))


if __name__ == "__main__":
    main()
