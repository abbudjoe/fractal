from __future__ import annotations

from python.reporting.schema import BenchmarkReport


def render_path1_table(report: BenchmarkReport, variant_label: str) -> str:
    cuda_peak_bytes = 0
    if report.runtime.cuda_device_memory is not None:
        cuda_peak_bytes = int(report.runtime.cuda_device_memory.get("peak_used_bytes", 0))
    return (
        f"{variant_label}"
        f"\tinitial_loss={report.initial_eval.mean_loss:.4f}"
        f"\tfinal_loss={report.final_eval.mean_loss:.4f}"
        f"\ttrain_tok_s={report.runtime.train_tokens_per_second:.2f}"
        f"\tcuda_peak_mb={cuda_peak_bytes / (1024 * 1024):.2f}"
    )

