from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MICROBENCH_PATH = REPO_ROOT / "scripts" / "v3a_eggroll_linear_microbench.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("v3a_eggroll_linear_microbench", MICROBENCH_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["v3a_eggroll_linear_microbench"] = module
    spec.loader.exec_module(module)
    return module


def test_parse_csv_ints_rejects_empty_and_non_positive_values():
    module = _load_module()

    assert module.parse_csv_ints("1, 2,3") == [1, 2, 3]
    for raw in ("", "0", "1,-2"):
        try:
            module.parse_csv_ints(raw)
        except Exception:
            pass
        else:
            raise AssertionError(f"expected parse failure for {raw!r}")


def test_run_case_cpu_smoke_virtual_and_materialized():
    module = _load_module()

    virtual = module.run_case(
        mode="virtual",
        population_size=2,
        batch_size=1,
        seq_len=4,
        d_in=8,
        d_out=8,
        rank=1,
        sigma=0.001,
        dtype_name="fp32",
        device=torch.device("cpu"),
        warmup=0,
        iters=1,
        seed=1,
    )
    materialized = module.run_case(
        mode="materialized",
        population_size=2,
        batch_size=1,
        seq_len=4,
        d_in=8,
        d_out=8,
        rank=1,
        sigma=0.001,
        dtype_name="fp32",
        device=torch.device("cpu"),
        warmup=0,
        iters=1,
        seed=1,
    )

    assert virtual.success
    assert materialized.success
    assert virtual.mean_ms is not None
    assert materialized.mean_ms is not None
