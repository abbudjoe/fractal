#!/usr/bin/env python3
"""Patch a MaxText checkout with the Fractal RGRP FFN-seam adapter.

The upstream MaxText command/config surface is intentionally left untouched in
this repository. This script applies the minimal source changes needed inside a
temporary MaxText checkout so ``fractal_candidate=rotary-gated-recurrent-state-update``
has an actual runtime implementation instead of being a paper contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path


FRACTAL_RGRP_MODULE = '''# Copyright 2026
#
# Fractal local experiment patch for MaxText.

"""Rotary gated recurrent state update block for MaxText FFN seams.

This is an E2E-quality reference lane, not a fused kernel. It uses packed
sequence projections plus ``jax.lax.scan`` over the token axis.
"""

from __future__ import annotations

from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp


SUPPORTED_STATE_TRANSFORMS = (
    "dense",
    "block-diagonal-2-masked-dense",
    "block-diagonal-4-masked-dense",
)


def _block_count(state_transform: str) -> int:
  if state_transform == "dense":
    return 1
  if state_transform.startswith("block-diagonal-2"):
    return 2
  if state_transform.startswith("block-diagonal-4"):
    return 4
  raise ValueError(f"unsupported fractal_rgrp_state_transform={state_transform!r}")


def _block_mask(width: int, block_count: int) -> jax.Array:
  if block_count == 1:
    return jnp.ones((width, width), dtype=jnp.float32)
  if width % block_count != 0:
    raise ValueError(f"RGRP width {width} must be divisible by block_count {block_count}")
  block_width = width // block_count
  block_ids = jnp.arange(width) // block_width
  return (block_ids[:, None] == block_ids[None, :]).astype(jnp.float32)


class RotaryGatedRecurrentStateUpdate(nn.Module):
  """A MaxText-local Linen adapter for the Fractal recurrent FFN seam."""

  d_model: int
  state_transform: str = "block-diagonal-4-masked-dense"
  scan_unroll: int = 3
  projection_mode: str = "sequence"
  trig_mode: str = "precompute"
  residual_scale: float = 1.0
  dtype: Any = jnp.bfloat16
  weight_dtype: Any = jnp.float32
  matmul_precision: str = "default"

  def setup(self):
    if self.d_model <= 0:
      raise ValueError(f"d_model must be positive, got {self.d_model}")
    if self.d_model % 2 != 0:
      raise ValueError(f"d_model must be even for rotary pairs, got {self.d_model}")
    if self.state_transform not in SUPPORTED_STATE_TRANSFORMS:
      raise ValueError(
          f"fractal_rgrp_state_transform must be one of {SUPPORTED_STATE_TRANSFORMS}, got {self.state_transform!r}"
      )
    if self.scan_unroll <= 0:
      raise ValueError(f"fractal_rgrp_scan_unroll must be positive, got {self.scan_unroll}")
    if self.projection_mode != "sequence":
      raise ValueError("MaxText RGRP quality lane currently supports fractal_rgrp_projection_mode='sequence' only")
    if self.trig_mode != "precompute":
      raise ValueError("MaxText RGRP quality lane currently supports fractal_rgrp_trig_mode='precompute' only")

  @property
  def angle_width(self) -> int:
    return self.d_model // 2

  @property
  def packed_width(self) -> int:
    return self.d_model + self.angle_width + self.d_model + self.d_model

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    in_kernel = self.param(
        "in_projection_kernel",
        nn.initializers.xavier_uniform(),
        (self.d_model, self.packed_width),
        self.weight_dtype,
    )
    in_bias = self.param("in_projection_bias", nn.initializers.zeros, (self.packed_width,), self.weight_dtype)
    state_kernel = self.param(
        "state_transform_kernel",
        nn.initializers.xavier_uniform(),
        (self.d_model, self.d_model),
        self.weight_dtype,
    )
    state_bias = self.param("state_transform_bias", nn.initializers.zeros, (self.d_model,), self.weight_dtype)

    inputs_dtype = inputs.dtype
    run_inputs = inputs.astype(self.dtype)
    packed = jnp.einsum(
        "btd,dk->btk",
        run_inputs,
        in_kernel.astype(self.dtype),
        precision=self.matmul_precision,
    ) + in_bias.astype(self.dtype)
    update_inputs, angle_inputs, candidate_inputs, output_inputs = jnp.split(
        packed,
        (
            self.d_model,
            self.d_model + self.angle_width,
            self.d_model + self.angle_width + self.d_model,
        ),
        axis=-1,
    )
    update_gates = jax.nn.sigmoid(update_inputs)
    retain_gates = 1.0 - update_gates
    candidates = jnp.tanh(candidate_inputs)
    output_gates = jax.nn.sigmoid(output_inputs)
    angle_cos = jnp.cos(angle_inputs)
    angle_sin = jnp.sin(angle_inputs)

    block_mask = _block_mask(self.d_model, _block_count(self.state_transform)).astype(jnp.float32)
    masked_state_kernel = state_kernel.astype(jnp.float32) * block_mask
    state_bias_f32 = state_bias.astype(jnp.float32)

    streams = {
        "update_gates": jnp.swapaxes(update_gates, 0, 1),
        "retain_gates": jnp.swapaxes(retain_gates, 0, 1),
        "candidates": jnp.swapaxes(candidates, 0, 1),
        "output_gates": jnp.swapaxes(output_gates, 0, 1),
        "angle_cos": jnp.swapaxes(angle_cos, 0, 1),
        "angle_sin": jnp.swapaxes(angle_sin, 0, 1),
    }
    initial_state = jnp.zeros((inputs.shape[0], self.d_model), dtype=jnp.float32)

    def step(state: jax.Array, controls: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
      projected = jnp.matmul(state, masked_state_kernel, precision=self.matmul_precision) + state_bias_f32
      pairs = projected.reshape(projected.shape[0], self.angle_width, 2)
      first = pairs[..., 0]
      second = pairs[..., 1]
      cos = controls["angle_cos"].astype(jnp.float32)
      sin = controls["angle_sin"].astype(jnp.float32)
      rotated = jnp.stack((first * cos - second * sin, first * sin + second * cos), axis=-1)
      transformed = rotated.reshape(projected.shape)
      next_state = (
          controls["update_gates"].astype(jnp.float32) * transformed
          + controls["retain_gates"].astype(jnp.float32) * controls["candidates"].astype(jnp.float32)
      )
      emitted = controls["output_gates"].astype(jnp.float32) * next_state
      return next_state, emitted.astype(inputs_dtype)

    _, outputs_time_major = jax.lax.scan(step, initial_state, streams, unroll=self.scan_unroll)
    outputs = jnp.swapaxes(outputs_time_major, 0, 1)
    return (outputs * jnp.asarray(self.residual_scale, dtype=outputs.dtype)).astype(inputs_dtype)


def rgrp_block(
    *,
    d_model: int,
    state_transform: str,
    scan_unroll: int,
    projection_mode: str,
    trig_mode: str,
    residual_scale: float,
    dtype: Any,
    weight_dtype: Any,
    matmul_precision: str,
    name: str | None = None,
) -> RotaryGatedRecurrentStateUpdate:
  return RotaryGatedRecurrentStateUpdate(
      d_model=d_model,
      state_transform=state_transform,
      scan_unroll=scan_unroll,
      projection_mode=projection_mode,
      trig_mode=trig_mode,
      residual_scale=residual_scale,
      dtype=dtype,
      weight_dtype=weight_dtype,
      matmul_precision=matmul_precision,
      name=name,
  )
'''


def replace_once(text: str, old: str, new: str, *, path: Path) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match in {path} for patch fragment; found {count}")
    return text.replace(old, new, 1)


def resolve_package_dir(path: Path) -> Path:
    """Return the real MaxText import package directory for this checkout."""

    for package_dir in (path / "src" / "maxtext", path / "src" / "MaxText"):
        if (package_dir / "layers").is_dir() and (package_dir / "configs").is_dir():
            return package_dir
    raise RuntimeError(f"{path} does not look like a MaxText checkout; expected src/maxtext or src/MaxText")


def patch_types(package_dir: Path) -> None:
    types_path = package_dir / "configs" / "types.py"
    text = types_path.read_text()
    if "fractal_candidate" in text:
        return
    marker = '  mlp_activations: list[str] = Field(["silu", "linear"], description="Activation functions in the MLP layer.")\n'
    insert = (
        '  fractal_candidate: str = Field("", description="Fractal experimental adapter slug. Empty disables adapters.")\n'
        '  fractal_adapter_module: str = Field("", description="Fractal adapter module identifier used by external manifests.")\n'
        '  fractal_rgrp_state_transform: str = Field("block-diagonal-4-masked-dense", description="RGRP state transform contract.")\n'
        '  fractal_rgrp_scan_unroll: int = Field(3, description="RGRP lax.scan unroll factor.")\n'
        '  fractal_rgrp_projection_mode: str = Field("sequence", description="RGRP projection lowering mode.")\n'
        '  fractal_rgrp_trig_mode: str = Field("precompute", description="RGRP trigonometric lowering mode.")\n'
        '  fractal_rgrp_residual_scale: float = Field(1.0, description="Scale applied to the RGRP FFN-seam residual branch.")\n'
    )
    types_path.write_text(replace_once(text, marker, marker + insert, path=types_path))


def patch_decoders(package_dir: Path) -> None:
    decoders_path = package_dir / "layers" / "decoders.py"
    text = decoders_path.read_text()
    if "fractal_rgrp" not in text:
        text = replace_once(
            text,
            "from maxtext.layers import quantizations\n",
            "from maxtext.layers import quantizations\nfrom maxtext.layers import fractal_rgrp\n",
            path=decoders_path,
        )
    old = '''    # MLP block.
    mlp_lnx = linears.mlp_block(
        in_features=lnx.shape[-1],
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        model_mode=model_mode,
        config=cfg,
        quant=self.quant,
        mesh=self.mesh,
    )(lnx, deterministic=deterministic)
'''
    new = '''    # FFN block. The Fractal adapter intentionally plugs into the same
    # pre-normalized seam as the standard MLP so attention/output contracts stay
    # MaxText-native.
    if cfg.fractal_candidate == "rotary-gated-recurrent-state-update":
      mlp_lnx = fractal_rgrp.rgrp_block(
          d_model=lnx.shape[-1],
          state_transform=cfg.fractal_rgrp_state_transform,
          scan_unroll=cfg.fractal_rgrp_scan_unroll,
          projection_mode=cfg.fractal_rgrp_projection_mode,
          trig_mode=cfg.fractal_rgrp_trig_mode,
          residual_scale=cfg.fractal_rgrp_residual_scale,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          matmul_precision=cfg.matmul_precision,
          name="mlp",
      )(lnx)
    else:
      mlp_lnx = linears.mlp_block(
          in_features=lnx.shape[-1],
          intermediate_dim=cfg.mlp_dim,
          activations=cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          name="mlp",
          model_mode=model_mode,
          config=cfg,
          quant=self.quant,
          mesh=self.mesh,
      )(lnx, deterministic=deterministic)
'''
    if "cfg.fractal_candidate == \"rotary-gated-recurrent-state-update\"" not in text:
        text = replace_once(text, old, new, path=decoders_path)
    decoders_path.write_text(text)


def patch_fractal_module(package_dir: Path) -> None:
    module_path = package_dir / "layers" / "fractal_rgrp.py"
    module_path.write_text(FRACTAL_RGRP_MODULE)


def patch_maxtext(path: Path) -> None:
    package_dir = resolve_package_dir(path)
    patch_fractal_module(package_dir)
    patch_types(package_dir)
    patch_decoders(package_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Path to a MaxText checkout")
    args = parser.parse_args()
    patch_maxtext(args.checkout.resolve())
    print(f"patched MaxText checkout: {args.checkout.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
