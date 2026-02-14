"""Result formatting utilities for RustSpice AI Agent."""

from typing import Any

from rustspice_agent.client import RunResult


def format_op_result(result: RunResult, precision: int = 6) -> str:
    """Format operating point results as a table."""
    lines = ["| Node | Voltage |", "|------|---------|"]
    for i, node in enumerate(result.node_names):
        if node == "0":  # Skip ground
            continue
        voltage = result.solution[i] if i < len(result.solution) else 0.0
        lines.append(f"| {node} | {format_value(voltage, precision)} V |")
    return "\n".join(lines)


def format_dc_sweep_result(result: RunResult, precision: int = 6) -> str:
    """Format DC sweep results as a table."""
    if not result.sweep_values or not result.sweep_solutions:
        return "No sweep data available."

    # Header
    sweep_var = result.sweep_var or "Sweep"
    header = f"| {sweep_var} |"
    separator = "|--------|"
    for node in result.node_names:
        if node == "0":
            continue
        header += f" V({node}) |"
        separator += "--------|"

    lines = [header, separator]

    # Data rows (limit to first 10 for readability)
    num_points = min(len(result.sweep_values), 10)
    for i in range(num_points):
        sweep_val = result.sweep_values[i]
        row = f"| {format_value(sweep_val, 4)} |"
        sol = result.sweep_solutions[i] if i < len(result.sweep_solutions) else []
        for j, node in enumerate(result.node_names):
            if node == "0":
                continue
            val = sol[j] if j < len(sol) else 0.0
            row += f" {format_value(val, precision)} |"
        lines.append(row)

    if len(result.sweep_values) > 10:
        lines.append(f"... ({len(result.sweep_values) - 10} more points)")

    return "\n".join(lines)


def format_tran_result(result: RunResult, precision: int = 6) -> str:
    """Format transient analysis results as a summary."""
    if not result.tran_times or not result.tran_solutions:
        return "No transient data available."

    lines = []
    lines.append(f"**Time Range:** {format_value(result.tran_times[0], 3)}s to {format_value(result.tran_times[-1], 3)}s")
    lines.append(f"**Points:** {len(result.tran_times)}")
    lines.append("")

    # Show final values
    lines.append("**Final Values:**")
    lines.append("| Node | Voltage |")
    lines.append("|------|---------|")
    final_sol = result.tran_solutions[-1] if result.tran_solutions else []
    for i, node in enumerate(result.node_names):
        if node == "0":
            continue
        val = final_sol[i] if i < len(final_sol) else 0.0
        lines.append(f"| {node} | {format_value(val, precision)} V |")

    return "\n".join(lines)


def format_ac_result(result: RunResult, precision: int = 4) -> str:
    """Format AC analysis results as a table."""
    if not result.ac_frequencies or not result.ac_magnitude_db:
        return "No AC data available."

    lines = []
    lines.append(f"**Frequency Range:** {format_value(result.ac_frequencies[0], 2)} Hz to {format_value(result.ac_frequencies[-1], 2)} Hz")
    lines.append(f"**Points:** {len(result.ac_frequencies)}")
    lines.append("")

    # Find output node (usually first non-ground node)
    output_idx = 0
    for i, node in enumerate(result.node_names):
        if node != "0" and node.lower() in ["out", "output", "vout"]:
            output_idx = i
            break
        elif node != "0" and output_idx == 0:
            output_idx = i

    output_node = result.node_names[output_idx] if output_idx < len(result.node_names) else "out"

    lines.append(f"**Output: V({output_node})**")
    lines.append("| Frequency | Magnitude | Phase |")
    lines.append("|-----------|-----------|-------|")

    # Show key frequency points
    num_points = min(len(result.ac_frequencies), 8)
    step = max(1, len(result.ac_frequencies) // num_points)
    for i in range(0, len(result.ac_frequencies), step):
        freq = result.ac_frequencies[i]
        mag = result.ac_magnitude_db[output_idx][i] if output_idx < len(result.ac_magnitude_db) and i < len(result.ac_magnitude_db[output_idx]) else 0.0
        phase = result.ac_phase_deg[output_idx][i] if output_idx < len(result.ac_phase_deg) and i < len(result.ac_phase_deg[output_idx]) else 0.0
        lines.append(f"| {format_freq(freq)} | {mag:.2f} dB | {phase:.1f} deg |")

    return "\n".join(lines)


def format_result(result: RunResult, precision: int = 6) -> str:
    """Format simulation result based on analysis type."""
    from rustspice_agent.client import AnalysisType

    header = f"**Analysis:** {result.analysis.value}\n**Status:** {result.status.value}\n\n"

    if result.analysis == AnalysisType.OP:
        return header + format_op_result(result, precision)
    elif result.analysis == AnalysisType.DC:
        return header + format_dc_sweep_result(result, precision)
    elif result.analysis == AnalysisType.TRAN:
        return header + format_tran_result(result, precision)
    elif result.analysis == AnalysisType.AC:
        return header + format_ac_result(result, precision)
    else:
        return header + "Unknown analysis type."


def format_value(value: float, precision: int = 6) -> str:
    """Format a numeric value with engineering notation."""
    if value == 0:
        return "0"

    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"{value/1e9:.{precision}g}G"
    elif abs_val >= 1e6:
        return f"{value/1e6:.{precision}g}M"
    elif abs_val >= 1e3:
        return f"{value/1e3:.{precision}g}k"
    elif abs_val >= 1:
        return f"{value:.{precision}g}"
    elif abs_val >= 1e-3:
        return f"{value*1e3:.{precision}g}m"
    elif abs_val >= 1e-6:
        return f"{value*1e6:.{precision}g}u"
    elif abs_val >= 1e-9:
        return f"{value*1e9:.{precision}g}n"
    elif abs_val >= 1e-12:
        return f"{value*1e12:.{precision}g}p"
    else:
        return f"{value:.{precision}e}"


def format_freq(freq: float) -> str:
    """Format frequency with appropriate unit."""
    if freq >= 1e9:
        return f"{freq/1e9:.2g} GHz"
    elif freq >= 1e6:
        return f"{freq/1e6:.2g} MHz"
    elif freq >= 1e3:
        return f"{freq/1e3:.2g} kHz"
    else:
        return f"{freq:.2g} Hz"


def format_circuit_summary(summary: dict[str, Any]) -> str:
    """Format circuit summary information."""
    return f"""**Circuit Summary:**
- Nodes: {summary.get('node_count', 0)}
- Devices: {summary.get('device_count', 0)}
- Models: {summary.get('model_count', 0)}
"""


def format_runs_list(runs: list[dict[str, Any]]) -> str:
    """Format list of simulation runs."""
    if not runs:
        return "No simulation runs yet."

    lines = ["| Run ID | Analysis | Status |", "|--------|----------|--------|"]
    for run in runs:
        lines.append(f"| {run.get('run_id', '?')} | {run.get('analysis', '?')} | {run.get('status', '?')} |")
    return "\n".join(lines)


def format_waveform_summary(waveform: dict[str, Any]) -> str:
    """Format waveform data summary."""
    x_vals = waveform.get("x_values", [])
    y_vals = waveform.get("y_values", [])

    if not x_vals or not y_vals:
        return "No waveform data available."

    lines = [
        f"**Signal:** {waveform.get('signal', 'unknown')}",
        f"**Analysis:** {waveform.get('analysis', 'unknown')}",
        f"**X-axis:** {waveform.get('x_label', 'x')} ({waveform.get('x_unit', '')})",
        f"**Y-axis:** {waveform.get('y_label', 'y')} ({waveform.get('y_unit', '')})",
        f"**Points:** {len(x_vals)}",
        f"**Range:** {format_value(min(y_vals))} to {format_value(max(y_vals))}",
    ]
    return "\n".join(lines)
