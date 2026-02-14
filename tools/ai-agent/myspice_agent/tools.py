"""Tool definitions for Claude LLM integration."""

from typing import Any

# Tool definitions in Anthropic format
TOOLS: list[dict[str, Any]] = [
    {
        "name": "run_operating_point",
        "description": "Run DC operating point analysis to find steady-state node voltages and branch currents. Use this for bias point analysis and DC circuit analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "Complete SPICE netlist text including .op command and .end"
                }
            },
            "required": ["netlist"]
        }
    },
    {
        "name": "run_dc_sweep",
        "description": "Run DC sweep analysis to plot DC transfer characteristics. Sweeps a voltage or current source and records node voltages at each point. Use for I-V curves, transfer functions, DC gain.",
        "input_schema": {
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "SPICE netlist text"
                },
                "source": {
                    "type": "string",
                    "description": "Name of voltage/current source to sweep (e.g., 'V1', 'I1')"
                },
                "start": {
                    "type": "number",
                    "description": "Start value of sweep"
                },
                "stop": {
                    "type": "number",
                    "description": "Stop value of sweep"
                },
                "step": {
                    "type": "number",
                    "description": "Step size for sweep"
                }
            },
            "required": ["netlist", "source", "start", "stop", "step"]
        }
    },
    {
        "name": "run_transient",
        "description": "Run time-domain transient analysis to simulate circuit behavior over time. Use for step response, pulse response, oscillations, switching behavior.",
        "input_schema": {
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "SPICE netlist text with time-varying sources (PULSE, SIN, etc.)"
                },
                "tstop": {
                    "type": "number",
                    "description": "Stop time in seconds"
                },
                "tstep": {
                    "type": "number",
                    "description": "Output time step in seconds (optional, defaults to tstop/100)"
                }
            },
            "required": ["netlist", "tstop"]
        }
    },
    {
        "name": "run_ac_analysis",
        "description": "Run AC small-signal frequency analysis to get frequency response (Bode plot). Use for filter analysis, amplifier bandwidth, stability analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "SPICE netlist with AC source (V1 ... DC 0 AC 1)"
                },
                "fstart": {
                    "type": "number",
                    "description": "Start frequency in Hz"
                },
                "fstop": {
                    "type": "number",
                    "description": "Stop frequency in Hz"
                },
                "points_per_decade": {
                    "type": "integer",
                    "description": "Number of frequency points per decade (default: 10)"
                }
            },
            "required": ["netlist", "fstart", "fstop"]
        }
    },
    {
        "name": "get_circuit_info",
        "description": "Get information about the currently loaded circuit including node count, device count, and model count.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_node_voltage",
        "description": "Get the voltage at a specific node from the most recent simulation result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "Simulation run ID (from previous simulation)"
                },
                "node": {
                    "type": "string",
                    "description": "Node name (e.g., 'out', 'in', 'vdd')"
                }
            },
            "required": ["run_id", "node"]
        }
    },
    {
        "name": "list_simulation_runs",
        "description": "List all simulation runs in the current session with their analysis types and status.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_waveform",
        "description": "Get detailed waveform data for a specific signal from a simulation run. Returns x and y values suitable for plotting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "Simulation run ID"
                },
                "signal": {
                    "type": "string",
                    "description": "Signal name (e.g., 'V(out)', 'out')"
                }
            },
            "required": ["run_id", "signal"]
        }
    },
    {
        "name": "export_results",
        "description": "Export simulation results to a file in various formats.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "Simulation run ID to export"
                },
                "path": {
                    "type": "string",
                    "description": "Output file path"
                },
                "format": {
                    "type": "string",
                    "enum": ["psf", "csv", "json"],
                    "description": "Output format (default: csv)"
                }
            },
            "required": ["run_id", "path"]
        }
    }
]


def get_tool_names() -> list[str]:
    """Get list of all tool names."""
    return [tool["name"] for tool in TOOLS]


def get_tool_by_name(name: str) -> dict[str, Any] | None:
    """Get tool definition by name."""
    for tool in TOOLS:
        if tool["name"] == name:
            return tool
    return None
