from __future__ import annotations

import shlex
from typing import Iterable

from python.specs.common import ValidationError

from .contracts import JaxTpuBenchmarkSpec


MAXTEXT_PRETRAIN_MODULE = "maxtext.trainers.pre_train.train"


def _format_override_value(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_maxtext_command(
    spec: JaxTpuBenchmarkSpec,
    *,
    allow_patched_maxtext: bool = False,
) -> list[str]:
    spec.validate()
    if spec.candidate.requires_patched_maxtext and not allow_patched_maxtext:
        raise ValidationError(
            f"candidate {spec.candidate.slug!r} requires a patched MaxText adapter; "
            "rerun with allow_patched_maxtext=true only inside a branch that implements that adapter"
        )
    overrides = spec.to_maxtext_overrides(include_adapter_overrides=allow_patched_maxtext)
    command = ["python3", "-m", MAXTEXT_PRETRAIN_MODULE]
    command.extend(f"{key}={_format_override_value(value)}" for key, value in sorted(overrides.items()))
    return command


def render_shell_command(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
