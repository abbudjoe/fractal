from __future__ import annotations

import argparse
import json

from python.specs.common import ValidationError, to_jsonable

from .contracts import (
    JaxTpuBenchmarkSpec,
    JaxTpuDatasetSpec,
    JaxTpuDatasetType,
    JaxTpuModelShape,
    JaxTpuParallelismSpec,
    JaxTpuRunBudget,
    get_candidate,
)
from .maxtext import build_maxtext_command, render_shell_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit a MaxText/JAX TPU command for Fractal architecture scout runs."
    )
    parser.add_argument("--candidate", default="attention-baseline")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-output-directory", required=True)
    parser.add_argument("--dataset-type", default=JaxTpuDatasetType.SYNTHETIC.value, choices=[kind.value for kind in JaxTpuDatasetType])
    parser.add_argument("--dataset-path")
    parser.add_argument("--dataset-name")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--moe-experts", type=int, default=0)
    parser.add_argument("--moe-top-k", type=int, default=0)
    parser.add_argument("--ici-data-parallelism", type=int, default=1)
    parser.add_argument("--ici-fsdp-parallelism", type=int, default=1)
    parser.add_argument("--ici-tensor-parallelism", type=int, default=1)
    parser.add_argument("--ici-expert-parallelism", type=int, default=1)
    parser.add_argument("--dcn-data-parallelism", type=int, default=1)
    parser.add_argument("--dcn-fsdp-parallelism", type=int, default=1)
    parser.add_argument("--dcn-tensor-parallelism", type=int, default=1)
    parser.add_argument("--dcn-expert-parallelism", type=int, default=1)
    parser.add_argument("--allow-patched-maxtext", action="store_true")
    parser.add_argument("--output", choices=["command", "json"], default="command")
    return parser


def spec_from_args(args: argparse.Namespace) -> JaxTpuBenchmarkSpec:
    return JaxTpuBenchmarkSpec(
        run_name=args.run_name,
        base_output_directory=args.base_output_directory,
        candidate=get_candidate(args.candidate),
        shape=JaxTpuModelShape(
            vocab_size=args.vocab_size,
            sequence_length=args.seq_len,
            d_model=args.d_model,
            num_layers=args.layers,
            num_heads=args.heads,
            ffn_multiplier=args.ffn_multiplier,
            moe_experts=args.moe_experts,
            moe_top_k=args.moe_top_k,
        ),
        parallelism=JaxTpuParallelismSpec(
            ici_data_parallelism=args.ici_data_parallelism,
            ici_fsdp_parallelism=args.ici_fsdp_parallelism,
            ici_tensor_parallelism=args.ici_tensor_parallelism,
            ici_expert_parallelism=args.ici_expert_parallelism,
            dcn_data_parallelism=args.dcn_data_parallelism,
            dcn_fsdp_parallelism=args.dcn_fsdp_parallelism,
            dcn_tensor_parallelism=args.dcn_tensor_parallelism,
            dcn_expert_parallelism=args.dcn_expert_parallelism,
        ),
        dataset=JaxTpuDatasetSpec(
            dataset_type=JaxTpuDatasetType(args.dataset_type),
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            train_split=args.train_split,
            eval_split=args.eval_split,
        ),
        budget=JaxTpuRunBudget(
            steps=args.steps,
            per_device_batch_size=args.per_device_batch_size,
            learning_rate=args.learning_rate,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        spec = spec_from_args(args)
        command = build_maxtext_command(spec, allow_patched_maxtext=args.allow_patched_maxtext)
    except ValidationError as exc:
        parser.error(str(exc))
    if args.output == "json":
        print(json.dumps({"spec": to_jsonable(spec), "command": command}, indent=2, sort_keys=True))
    else:
        print(render_shell_command(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
