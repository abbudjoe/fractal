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


def render_mini_moe_table(report: BenchmarkReport, variant_label: str) -> str:
    cuda_peak_bytes = 0
    if report.runtime.cuda_device_memory is not None:
        cuda_peak_bytes = int(report.runtime.cuda_device_memory.get("peak_used_bytes", 0))
    routing = report.mini_moe_summary.routing if report.mini_moe_summary is not None else None
    route_entropy = 0.0 if routing is None else routing.mean_route_entropy_bits
    round_count = 0 if routing is None else routing.round_count
    return (
        f"{variant_label}"
        f"\tinitial_loss={report.initial_eval.mean_loss:.4f}"
        f"\tfinal_loss={report.final_eval.mean_loss:.4f}"
        f"\ttrain_tok_s={report.runtime.train_tokens_per_second:.2f}"
        f"\tcuda_peak_mb={cuda_peak_bytes / (1024 * 1024):.2f}"
        f"\troute_entropy_bits={route_entropy:.4f}"
        f"\trounds={round_count}"
    )


def render_mini_moe_round_transition_table(
    report: BenchmarkReport,
    *,
    round_index: int = 2,
) -> str:
    summary = report.mini_moe_summary
    if summary is None:
        return "no mini-moe summary available"

    rows = [
        round_summary
        for round_summary in summary.controller_rounds
        if round_summary.round_index == round_index
    ]
    if not rows:
        return f"no controller round {round_index} summaries available"

    lines = [
        f"layer={row.layer_index}"
        f"\tentropy_bits={row.mean_route_entropy_bits:.4f}"
        f"\tmargin={row.mean_winner_margin:.4f}"
        f"\tadjust_l1={0.0 if row.mean_route_adjustment_l1 is None else row.mean_route_adjustment_l1:.4f}"
        f"\trerouted={row.rerouted_token_fraction:.4f}"
        f"\tapplied={row.applied_token_fraction:.4f}"
        for row in rows
    ]
    return "\n".join(lines)


def render_mini_moe_token_trace_table(
    report: BenchmarkReport,
    *,
    limit_per_layer: int = 3,
) -> str:
    summary = report.mini_moe_summary
    if summary is None:
        return "no mini-moe summary available"
    if not summary.token_traces:
        return "no token route traces available"

    emitted_per_layer: dict[int, int] = {}
    lines: list[str] = []
    for trace in summary.token_traces:
        emitted_count = emitted_per_layer.get(trace.layer_index, 0)
        if emitted_count >= limit_per_layer:
            continue
        emitted_per_layer[trace.layer_index] = emitted_count + 1
        round_fragments = []
        for round_trace in trace.rounds:
            adjustment = (
                "-"
                if round_trace.route_adjustment_l1 is None
                else f"{round_trace.route_adjustment_l1:.4f}"
            )
            round_fragments.append(
                f"r{round_trace.round_index}:"
                f"e{round_trace.winner_expert_id}"
                f"@{round_trace.winner_weight:.3f}"
                f"/H={round_trace.route_entropy_bits:.3f}"
                f"/M={round_trace.winner_margin:.3f}"
                f"/d={adjustment}"
            )
        lines.append(
            f"layer={trace.layer_index}"
            f"\tpass={trace.forward_pass_index}"
            f"\tbatch={trace.batch_index}"
            f"\tpos={trace.position_index}"
            f"\ttoken={trace.token_label}"
            f"\ttoken_id={trace.token_id}"
            f"\trerouted={trace.rerouted}"
            f"\tfirst=e{trace.first_winner_expert_id}"
            f"\tfinal=e{trace.final_winner_expert_id}"
            f"\ttotal_adjust_l1={trace.total_adjustment_l1:.4f}"
            f"\t{' | '.join(round_fragments)}"
        )
    return "\n".join(lines) if lines else "no token route traces available"
