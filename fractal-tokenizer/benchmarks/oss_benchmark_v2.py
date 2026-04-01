#!/usr/bin/env python3
"""Cloud GPU benchmark for p1_fractal_hybrid_dyn-state-norm_v2.

This script compares the standalone fractal tokenizer against native model
tokenizers for Llama 3.1 8B Instruct and Mistral 7B Instruct on two benchmark
inputs that already exist in the Rust tokenizer track.

Implementation notes:
- The fractal tokenizer path is a direct Python port of the v2 tokenizer logic
  from `fractal-tokenizer/src/tokenizer.rs` and
  `fractal-tokenizer/src/primitives/p1_fractal_hybrid.rs`.
- The native model forward pass always runs in the model's native vocabulary
  space. For the `v2` rows, we therefore report native forward latency as the
  forward-pass anchor and mark perplexity as `N/A`, because there is no learned
  adapter from fractal tokens into the model embedding table.
- Backend selection prefers `vllm` when available, falls back to
  `transformers` + `accelerate`, and uses `llama_cpp` as a GGUF-only fallback.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


TOKENIZER_NAME = "p1_fractal_hybrid_dyn-state-norm_v2"
DEFAULT_DIM = 64
DEFAULT_MAX_DEPTH = 6
DEFAULT_SEED = 42
DEFAULT_STRESS_REPEAT = 20
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
U64_MASK = (1 << 64) - 1


@dataclass
class TokenRecord:
    depth: int
    start: int
    end: int
    text: str
    token: str


@dataclass
class SeenMotif:
    depth: int
    digest: str
    norm: float
    signature: list[float]


@dataclass
class RepeatedMotif:
    digest: str
    total_reuse_count: int
    ranges: list[tuple[int, int]]
    example_start: int
    example_end: int


@dataclass
class ForwardMetrics:
    token_count: int
    tokenize_ms: float
    forward_ms: float
    wall_time_ms: float
    simple_perplexity: str


@dataclass
class BenchmarkInput:
    name: str
    text: str


class MotifRegistry:
    def __init__(self) -> None:
        self.assigned_by_depth: dict[int, set[str]] = defaultdict(set)
        self.seen: list[SeenMotif] = []

    def resolve(
        self,
        depth: int,
        digest: str,
        signature: Sequence[float],
        rolling_norm: float,
    ) -> str:
        if self.reuse_gate_open(rolling_norm):
            candidates = [
                (signature_distance(signature, entry.signature), entry)
                for entry in self.seen
                if entry.depth != depth
                and not self.digest_used_at_depth(depth, entry.digest)
            ]
            adaptive_threshold = dynamic_similarity_threshold_v2(
                signature,
                depth,
                rolling_norm,
                candidates,
            )
            nearest = min(candidates, key=lambda item: item[0], default=None)
            if nearest is not None and nearest[0] <= adaptive_threshold:
                resolved = nearest[1].digest
            else:
                resolved = digest
        else:
            resolved = digest

        self.assigned_by_depth[depth].add(resolved)
        self.seen.append(
            SeenMotif(
                depth=depth,
                digest=digest,
                norm=rolling_norm,
                signature=list(signature),
            )
        )
        return resolved

    def digest_used_at_depth(self, depth: int, digest: str) -> bool:
        return digest in self.assigned_by_depth.get(depth, set())

    def reuse_gate_open(self, rolling_norm: float) -> bool:
        mean_norm = self.mean_norm()
        return mean_norm is None or rolling_norm >= mean_norm

    def mean_norm(self) -> float | None:
        if not self.seen:
            return None
        return sum(entry.norm for entry in self.seen) / len(self.seen)


class P1HybridV2Rule:
    def __init__(self, hidden_dim: int, seed: int) -> None:
        self.hidden_dim = hidden_dim
        self.g_proj = seeded_linear(hidden_dim, hidden_dim, seed, 2)
        self.w_h = seeded_linear(hidden_dim, hidden_dim, seed, 3)
        self.u = seeded_linear(hidden_dim, hidden_dim, seed, 4)

    def apply(self, previous_state: Sequence[float], features: Sequence[float]) -> list[float]:
        g = sigmoid_vector(linear_forward(features, self.g_proj))
        clamp_scalar = sigmoid_scalar(row_l2_norm(previous_state)) * -0.225 + 0.75
        squared = clamp_symmetric_by_row(
            [value * value for value in previous_state],
            clamp_scalar,
        )
        main_update = add_vectors(
            add_vectors(linear_forward(previous_state, self.w_h), linear_forward(features, self.u)),
            squared,
        )
        return [
            gate * update + (1.0 - gate) * previous
            for gate, update, previous in zip(g, main_update, previous_state)
        ]


class NativeRuntime:
    backend_name = "unknown"

    def benchmark_native(self, text: str) -> ForwardMetrics:
        raise NotImplementedError


class TransformersRuntime(NativeRuntime):
    backend_name = "transformers"

    def __init__(self, model_path: str, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16
        if have_module("accelerate"):
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        if not have_module("accelerate"):
            self.model.to(device)

        self.model_device = next(self.model.parameters()).device

    def benchmark_native(self, text: str) -> ForwardMetrics:
        started = time.perf_counter()
        encoded = self.tokenizer(text, return_tensors="pt")
        tokenize_ms = (time.perf_counter() - started) * 1000.0
        encoded = {key: value.to(self.model_device) for key, value in encoded.items()}
        token_count = int(encoded["input_ids"].shape[-1])

        forward_started = time.perf_counter()
        perplexity = "N/A"
        with self.torch.no_grad():
            try:
                outputs = self.model(**encoded, labels=encoded["input_ids"])
                loss = float(outputs.loss.detach().float().cpu())
                perplexity = f"{math.exp(min(loss, 20.0)):.4f}"
            except Exception:
                self.model(**encoded)
        forward_ms = (time.perf_counter() - forward_started) * 1000.0

        return ForwardMetrics(
            token_count=token_count,
            tokenize_ms=tokenize_ms,
            forward_ms=forward_ms,
            wall_time_ms=tokenize_ms + forward_ms,
            simple_perplexity=perplexity,
        )


class VllmRuntime(NativeRuntime):
    backend_name = "vllm"

    def __init__(self, model_path: str) -> None:
        from vllm import LLM

        self.sampling_params = None
        self.llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=1)
        self.tokenizer = self.llm.get_tokenizer()

    def benchmark_native(self, text: str) -> ForwardMetrics:
        from vllm import SamplingParams

        tokenize_started = time.perf_counter()
        token_ids = self.tokenizer.encode(text)
        tokenize_ms = (time.perf_counter() - tokenize_started) * 1000.0

        forward_started = time.perf_counter()
        self.llm.generate(
            [text],
            SamplingParams(max_tokens=1, temperature=0.0, logprobs=1),
            use_tqdm=False,
        )
        forward_ms = (time.perf_counter() - forward_started) * 1000.0

        return ForwardMetrics(
            token_count=len(token_ids),
            tokenize_ms=tokenize_ms,
            forward_ms=forward_ms,
            wall_time_ms=tokenize_ms + forward_ms,
            simple_perplexity="N/A",
        )


class LlamaCppRuntime(NativeRuntime):
    backend_name = "llama_cpp"

    def __init__(self, model_path: str, device: str) -> None:
        from llama_cpp import Llama

        gpu_index = parse_cuda_index(device)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1 if gpu_index is not None else 0,
            main_gpu=gpu_index or 0,
            verbose=False,
        )

    def benchmark_native(self, text: str) -> ForwardMetrics:
        payload = text.encode("utf-8")
        tokenize_started = time.perf_counter()
        token_ids = self.llm.tokenize(payload, add_bos=False)
        tokenize_ms = (time.perf_counter() - tokenize_started) * 1000.0

        forward_started = time.perf_counter()
        self.llm.create_completion(
            prompt=text,
            max_tokens=1,
            temperature=0.0,
            echo=False,
        )
        forward_ms = (time.perf_counter() - forward_started) * 1000.0

        return ForwardMetrics(
            token_count=len(token_ids),
            tokenize_ms=tokenize_ms,
            forward_ms=forward_ms,
            wall_time_ms=tokenize_ms + forward_ms,
            simple_perplexity="N/A",
        )


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def parse_cuda_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    if ":" not in device:
        return 0
    suffix = device.split(":", 1)[1]
    return int(suffix) if suffix.isdigit() else 0


def maybe_set_cuda_visible_devices(device: str) -> None:
    cuda_index = parse_cuda_index(device)
    if cuda_index is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_index)


def is_gguf_model(model_path: str) -> bool:
    return model_path.lower().endswith(".gguf")


def build_runtime(model_path: str, device: str, preferred_backend: str) -> NativeRuntime:
    if is_gguf_model(model_path):
        if have_module("llama_cpp"):
            return LlamaCppRuntime(model_path, device)
        raise RuntimeError(
            "GGUF model provided but llama_cpp is not installed; install llama-cpp-python or use a Hugging Face model directory."
        )

    if preferred_backend in {"auto", "vllm"} and have_module("vllm"):
        return VllmRuntime(model_path)
    if preferred_backend in {"auto", "transformers"} and have_module("transformers"):
        return TransformersRuntime(model_path, device)
    if preferred_backend == "vllm":
        raise RuntimeError("vLLM requested but not installed.")
    if preferred_backend == "transformers":
        raise RuntimeError("transformers requested but not installed.")
    raise RuntimeError(
        "No supported inference backend found. Install vllm or transformers+accelerate."
    )


def mix_seed(seed: int, stream: int) -> int:
    state = (seed ^ ((stream * 0x9E37_79B9_7F4A_7C15) & U64_MASK)) & U64_MASK
    state ^= state >> 30
    state = (state * 0xBF58_476D_1CE4_E5B9) & U64_MASK
    state ^= state >> 27
    state = (state * 0x94D0_49BB_1331_11EB) & U64_MASK
    return state ^ (state >> 31)


def next_weight(state: int) -> tuple[int, float]:
    state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & U64_MASK
    unit = (state >> 11) / float(1 << 53)
    return state, ((unit * 2.0) - 1.0) * 0.125


def seeded_linear(d_input: int, d_output: int, seed: int, stream: int) -> tuple[list[list[float]], list[float]]:
    state = mix_seed(seed, stream)
    weights = []
    for _ in range(d_input):
        row = []
        for _ in range(d_output):
            state, weight = next_weight(state)
            row.append(weight)
        weights.append(row)
    bias = []
    for _ in range(d_output):
        state, weight = next_weight(state)
        bias.append(weight)
    return weights, bias


def linear_forward(vector: Sequence[float], linear: tuple[list[list[float]], list[float]]) -> list[float]:
    weights, bias = linear
    output = []
    for column, bias_value in enumerate(bias):
        total = bias_value
        for row, value in enumerate(vector):
            total += value * weights[row][column]
        output.append(total)
    return output


def sigmoid_scalar(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def sigmoid_vector(values: Sequence[float]) -> list[float]:
    return [sigmoid_scalar(value) for value in values]


def add_vectors(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [a + b for a, b in zip(left, right)]


def row_l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) + 1e-12)


def clamp_symmetric_by_row(values: Sequence[float], clamp: float) -> list[float]:
    lower = -clamp
    return [min(max(value, lower), clamp) for value in values]


def split_point(data: bytes) -> int | None:
    if len(data) <= 1:
        return None

    mid = len(data) // 2
    best_index = None
    best_distance = None
    for index in range(1, len(data)):
        if chr(data[index - 1]).isspace() or chr(data[index]).isspace():
            distance = abs(mid - index)
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
    return best_index if best_index is not None else max(mid, 1)


def segment_features(data: bytes, dim: int) -> list[float]:
    if not data:
        return [0.0] * dim

    features = [0.0] * dim
    whitespace = 0.0
    punctuation = 0.0
    uppercase = 0.0

    for index, byte in enumerate(data):
        centered = byte / 127.5 - 1.0
        bucket = index % dim
        mirror = dim - 1 - bucket
        features[bucket] += centered
        features[mirror] += centered * 0.5
        char = chr(byte)
        if char.isspace():
            whitespace += 1.0
        if char in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""":
            punctuation += 1.0
        if char.isupper():
            uppercase += 1.0

    length = float(len(data))
    features = [value / length for value in features]

    if dim >= 4:
        features[dim - 4] = length / max(dim, 1)
        features[dim - 3] = whitespace / length
        features[dim - 2] = punctuation / length
        features[dim - 1] = uppercase / length
    return features


def normalized_l2(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def mean_absolute_value(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def harmonic_mean(left: float, right: float) -> float:
    total = left + right
    if total == 0.0:
        return 0.0
    return (left * right / total) + (left * right / total)


def signature_distance(left: Sequence[float], right: Sequence[float]) -> float:
    width = min(len(left), len(right))
    if width == 0:
        return 0.0
    return sum(abs(a - b) for a, b in zip(left[:width], right[:width])) / width


def dynamic_similarity_threshold_v2(
    signature: Sequence[float],
    depth: int,
    rolling_norm: float,
    candidates: Sequence[tuple[float, SeenMotif]],
) -> float:
    if not candidates:
        return 0.0

    state_scale = rolling_norm * mean_absolute_value(signature)
    depth_scale = depth / (depth + 1.0)
    local_similarity = sum(distance for distance, _ in candidates) / len(candidates)
    nearest_distance = min(distance for distance, _ in candidates)
    local_selectivity = local_similarity / (local_similarity + nearest_distance)
    return harmonic_mean(state_scale, local_similarity) * depth_scale * local_selectivity


def summarize_readout(
    readout: Sequence[float],
    data: bytes,
    depth: int,
    motifs: MotifRegistry,
) -> str:
    digest = FNV_OFFSET
    for value in list(readout[:8]):
        quantized = int(round(value * 1000.0))
        digest ^= quantized & U64_MASK
        digest = (digest * FNV_PRIME) & U64_MASK
    digest_hex = f"{digest:016x}"
    rolling_norm = normalized_l2(readout)
    motif = motifs.resolve(depth, digest_hex, readout, rolling_norm)
    return f"d{depth}-n{len(data)}-{motif}"


def tokenize_v2(text: str, dim: int, max_depth: int, seed: int) -> list[TokenRecord]:
    rule = P1HybridV2Rule(dim, seed)
    tokens: list[TokenRecord] = []
    motifs = MotifRegistry()
    initial_state = [0.0] * dim

    def recurse(previous_state: Sequence[float], data: bytes, start: int, depth: int) -> None:
        if not data:
            return
        features = segment_features(data, dim)
        next_state = rule.apply(previous_state, features)
        token = summarize_readout(next_state, data, depth, motifs)
        tokens.append(
            TokenRecord(
                depth=depth,
                start=start,
                end=start + len(data),
                text=data.decode("utf-8", errors="replace"),
                token=token,
            )
        )
        if depth + 1 >= max_depth or len(data) <= 1:
            return
        split = split_point(data)
        if split is None:
            return
        recurse(next_state, data[:split], start, depth + 1)
        recurse(next_state, data[split:], start + split, depth + 1)

    recurse(initial_state, text.encode("utf-8"), 0, 0)
    return tokens


def token_digest(token: TokenRecord) -> str:
    return token.token.rsplit("-", 1)[-1]


def repeated_cross_depth_motifs(tokens: Sequence[TokenRecord]) -> list[RepeatedMotif]:
    motif_hits: dict[str, dict[str, object]] = {}
    for token in tokens:
        digest = token_digest(token)
        entry = motif_hits.setdefault(
            digest,
            {
                "depths": set(),
                "ranges": [],
                "example_start": token.start,
                "example_end": token.end,
            },
        )
        entry["depths"].add(token.depth)
        entry["ranges"].append((token.start, token.end))

    repeated = []
    for digest, entry in motif_hits.items():
        depths = entry["depths"]
        ranges = entry["ranges"]
        if len(depths) > 1 and len(ranges) >= 2:
            repeated.append(
                RepeatedMotif(
                    digest=digest,
                    total_reuse_count=len(ranges),
                    ranges=list(ranges),
                    example_start=int(entry["example_start"]),
                    example_end=int(entry["example_end"]),
                )
            )
    return sorted(repeated, key=lambda motif: motif.digest)


def truncated_span(text: str, start: int, end: int, limit: int = 80) -> str:
    span = text[start:end]
    if len(span) <= limit:
        return span
    return span[: limit - 3] + "..."


def stress_input(repeat_count: int) -> str:
    paragraph = (
        "The cat sat on the mat. The dog sat on the mat. "
        "The bird sat on the mat. The fox sat on the mat."
    )
    return " ".join([paragraph] * repeat_count) + " The cat sat on the mat once more."


def mixed_domain_input() -> str:
    return "\n".join(
        [
            "=== NEWS ===",
            "City officials said Tuesday that transit service resumed across the river corridor after overnight storms flooded two low-lying stations, while crews continued inspecting power lines and drainage pumps before the evening commute.",
            "=== CODE COMMENT ===",
            "This cache invalidation path keeps a rolling checksum for each segment so repeated blocks can be recognized without recomputing the full buffer; if a checksum disagrees, rebuild the branch and log the span that changed for debugging.",
            "=== LITERATURE ===",
            "By the time the lamps were lit, the street had gone quiet enough for the distant train to sound like weather, and the old bookseller stood in his doorway listening as if the night itself were turning a page.",
        ]
    )


def benchmark_inputs(stress_repeat: int) -> list[BenchmarkInput]:
    return [
        BenchmarkInput("stress-20x-repetition", stress_input(stress_repeat)),
        BenchmarkInput("mixed-domain", mixed_domain_input()),
    ]


def print_v2_result(model_name: str, benchmark_input: BenchmarkInput, tokens: Sequence[TokenRecord], wall_time_ms: float) -> None:
    motif_reuse = repeated_cross_depth_motifs(tokens)
    print(f"input_name={benchmark_input.name}")
    print(f"model_name={model_name}")
    print("tokenizer_type=v2")
    print(f"input_length={len(benchmark_input.text)}")
    print(f"token_count={len(tokens)}")
    print(f"avg_chars_per_token={len(benchmark_input.text) / max(len(tokens), 1):.2f}")
    print(f"motif_reuse_count={len(motif_reuse)}")
    print(f"wall_time_ms={wall_time_ms:.2f}")
    print("simple_perplexity=N/A")
    print(f"REUSED MOTIFS (cross-depth) [{TOKENIZER_NAME}]")
    if not motif_reuse:
        print("(none)")
    else:
        for motif in motif_reuse:
            ranges = ", ".join(f"{start}..{end}" for start, end in motif.ranges)
            text = truncated_span(
                benchmark_input.text,
                motif.example_start,
                motif.example_end,
            )
            print(
                f"digest={motif.digest} | total_reuse_count={motif.total_reuse_count} | "
                f"ranges={ranges} | text=\"{text}\""
            )


def print_native_result(
    model_name: str,
    benchmark_input: BenchmarkInput,
    metrics: ForwardMetrics,
    runtime: NativeRuntime,
) -> None:
    print(f"input_name={benchmark_input.name}")
    print(f"model_name={model_name}")
    print("tokenizer_type=native")
    print(f"input_length={len(benchmark_input.text)}")
    print(f"token_count={metrics.token_count}")
    print(f"avg_chars_per_token={len(benchmark_input.text) / max(metrics.token_count, 1):.2f}")
    print("motif_reuse_count=N/A")
    print(f"wall_time_ms={metrics.wall_time_ms:.2f}")
    print(f"simple_perplexity={metrics.simple_perplexity}")
    print(f"native_backend={runtime.backend_name}")


def model_specs_from_args(args: argparse.Namespace) -> list[tuple[str, str]]:
    specs = []
    if args.llama_model:
        specs.append(("Llama 3.1 8B Instruct", args.llama_model))
    if args.mistral_model:
        specs.append(("Mistral 7B Instruct", args.mistral_model))
    return specs


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark p1_fractal_hybrid_dyn-state-norm_v2 against native OSS model tokenizers.",
    )
    parser.add_argument("--llama-model", help="Path or model id for Llama 3.1 8B Instruct.")
    parser.add_argument("--mistral-model", help="Path or model id for Mistral 7B Instruct.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Target device string for transformers / GGUF backends. Default: cuda:0",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "vllm", "transformers"),
        default="auto",
        help="Preferred inference backend for Hugging Face models. Default: auto",
    )
    parser.add_argument(
        "--stress-repeat",
        type=int,
        default=DEFAULT_STRESS_REPEAT,
        help="Repetition count for the stress corpus. Default matches the Rust tests: 20",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIM,
        help="Hidden dimension for the fractal tokenizer. Default: 64",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Maximum recursive tokenizer depth. Default: 6",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic seed for the fractal tokenizer weights. Default: 42",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    maybe_set_cuda_visible_devices(args.device)

    specs = model_specs_from_args(args)
    if not specs:
        print(
            "No model paths were provided. Pass --llama-model and/or --mistral-model to run the native benchmark rows.",
            file=sys.stderr,
        )
        return 2

    inputs = benchmark_inputs(args.stress_repeat)

    for model_name, model_path in specs:
        try:
            runtime = build_runtime(model_path, args.device, args.backend)
        except Exception as error:
            print(f"model_name={model_name}")
            print("status=MODEL NOT FOUND — SKIPPED")
            print(f"reason={error}")
            print()
            continue

        for benchmark_input in inputs:
            native_metrics = runtime.benchmark_native(benchmark_input.text)

            v2_started = time.perf_counter()
            v2_tokens = tokenize_v2(
                benchmark_input.text,
                dim=args.dim,
                max_depth=args.max_depth,
                seed=args.seed,
            )
            v2_tokenize_ms = (time.perf_counter() - v2_started) * 1000.0
            v2_wall_ms = v2_tokenize_ms + native_metrics.forward_ms

            print_v2_result(model_name, benchmark_input, v2_tokens, v2_wall_ms)
            print()
            print_native_result(model_name, benchmark_input, native_metrics, runtime)
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
