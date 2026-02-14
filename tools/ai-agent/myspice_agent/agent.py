"""Core AI agent for MySpice circuit simulation."""

import json
from typing import Any, Optional

from myspice_agent.client import SpiceClient, SpiceClientError, RunResult
from myspice_agent.config import Config
from myspice_agent.formatters import (
    format_result,
    format_circuit_summary,
    format_runs_list,
    format_waveform_summary,
)
from myspice_agent.prompts import SYSTEM_PROMPT
from myspice_agent.tools import TOOLS

# Try to import anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class SpiceAgent:
    """AI-powered circuit simulation assistant.

    This agent uses Claude to interpret user requests and execute
    appropriate simulations via the MySpice API.

    Example:
        agent = SpiceAgent()
        response = agent.chat("Analyze a voltage divider with 5V, 1k and 2k resistors")
        print(response)
    """

    def __init__(
        self,
        client: Optional[SpiceClient] = None,
        config: Optional[Config] = None,
    ):
        """Initialize the agent.

        Args:
            client: SpiceClient instance (created if not provided)
            config: Configuration (loaded from file/env if not provided)
        """
        self.config = config or Config.load()
        self.client = client or SpiceClient(
            base_url=self.config.api.url,
            timeout=self.config.api.timeout,
        )
        self._anthropic: Optional[Any] = None
        self._conversation: list[dict[str, Any]] = []
        self._last_run_id: Optional[int] = None

    @property
    def anthropic(self) -> Any:
        """Get or create Anthropic client."""
        if self._anthropic is None:
            if not ANTHROPIC_AVAILABLE:
                raise RuntimeError(
                    "Anthropic package not installed. "
                    "Install with: pip install myspice-agent[ai]"
                )
            if not self.config.ai.api_key:
                raise RuntimeError(
                    "Anthropic API key not set. "
                    "Set ANTHROPIC_API_KEY environment variable."
                )
            self._anthropic = anthropic.Anthropic(api_key=self.config.ai.api_key)
        return self._anthropic

    def chat(self, user_message: str) -> str:
        """Process a user message and return a response.

        Args:
            user_message: The user's input message

        Returns:
            The agent's response text
        """
        # Add user message to conversation
        self._conversation.append({"role": "user", "content": user_message})

        # Call Claude with tools
        response = self.anthropic.messages.create(
            model=self.config.ai.model,
            max_tokens=self.config.ai.max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=self._conversation,
        )

        # Process response
        return self._process_response(response)

    def _process_response(self, response: Any) -> str:
        """Process Claude's response, handling tool calls if needed."""
        assistant_content: list[dict[str, Any]] = []
        final_text = ""

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                final_text += block.text
            elif block.type == "tool_use":
                # Execute the tool
                tool_name = block.name
                tool_input = block.input
                tool_id = block.id

                assistant_content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                })

                # Execute tool and get result
                tool_result = self._execute_tool(tool_name, tool_input)

                # Add assistant message and tool result
                self._conversation.append({"role": "assistant", "content": assistant_content})
                self._conversation.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": tool_result,
                    }]
                })

                # Continue conversation to get final response
                continuation = self.anthropic.messages.create(
                    model=self.config.ai.model,
                    max_tokens=self.config.ai.max_tokens,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=self._conversation,
                )

                # Recursively process (handles multiple tool calls)
                return self._process_response(continuation)

        # No tool calls - add assistant response and return
        if assistant_content:
            self._conversation.append({"role": "assistant", "content": assistant_content})

        return final_text

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if tool_name == "run_operating_point":
                result = self.client.run_op(netlist=args["netlist"])
                self._last_run_id = result.run_id
                return format_result(result, self.config.output.precision)

            elif tool_name == "run_dc_sweep":
                result = self.client.run_dc(
                    source=args["source"],
                    start=args["start"],
                    stop=args["stop"],
                    step=args["step"],
                    netlist=args.get("netlist"),
                )
                self._last_run_id = result.run_id
                return format_result(result, self.config.output.precision)

            elif tool_name == "run_transient":
                result = self.client.run_tran(
                    tstop=args["tstop"],
                    tstep=args.get("tstep"),
                    netlist=args.get("netlist"),
                )
                self._last_run_id = result.run_id
                return format_result(result, self.config.output.precision)

            elif tool_name == "run_ac_analysis":
                result = self.client.run_ac(
                    fstart=args["fstart"],
                    fstop=args["fstop"],
                    points=args.get("points_per_decade", 10),
                    netlist=args.get("netlist"),
                )
                self._last_run_id = result.run_id
                return format_result(result, self.config.output.precision)

            elif tool_name == "get_circuit_info":
                summary = self.client.get_summary()
                return format_circuit_summary({
                    "node_count": summary.node_count,
                    "device_count": summary.device_count,
                    "model_count": summary.model_count,
                })

            elif tool_name == "get_node_voltage":
                run_id = args["run_id"]
                node = args["node"]
                result = self.client.get_run(run_id)
                voltage = self.client.get_node_voltage(result, node)
                if voltage is not None:
                    return f"V({node}) = {voltage:.6g} V"
                else:
                    return f"Node '{node}' not found in simulation results."

            elif tool_name == "list_simulation_runs":
                runs = self.client.get_runs()
                return format_runs_list(runs)

            elif tool_name == "get_waveform":
                run_id = args["run_id"]
                signal = args["signal"]
                waveform = self.client.get_waveform(run_id, signal)
                return format_waveform_summary(waveform)

            elif tool_name == "export_results":
                run_id = args["run_id"]
                path = args["path"]
                format_type = args.get("format", "csv")
                self.client.export_run(run_id, path, format_type)
                return f"Results exported to {path}"

            else:
                return f"Unknown tool: {tool_name}"

        except SpiceClientError as e:
            return f"Simulation error: {e}"
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation = []
        self._last_run_id = None

    def get_last_run_id(self) -> Optional[int]:
        """Get the ID of the most recent simulation run."""
        return self._last_run_id


class DirectAgent:
    """Direct (non-AI) interface for simulation.

    This class provides direct access to simulation functions
    without requiring an LLM, useful for scripting and testing.
    """

    def __init__(self, client: Optional[SpiceClient] = None):
        """Initialize direct agent."""
        self.client = client or SpiceClient()

    def run_op(self, netlist: str) -> RunResult:
        """Run operating point analysis."""
        return self.client.run_op(netlist=netlist)

    def run_dc(
        self,
        netlist: str,
        source: str,
        start: float,
        stop: float,
        step: float,
    ) -> RunResult:
        """Run DC sweep analysis."""
        return self.client.run_dc(
            source=source,
            start=start,
            stop=stop,
            step=step,
            netlist=netlist,
        )

    def run_tran(
        self,
        netlist: str,
        tstop: float,
        tstep: Optional[float] = None,
    ) -> RunResult:
        """Run transient analysis."""
        return self.client.run_tran(tstop=tstop, tstep=tstep, netlist=netlist)

    def run_ac(
        self,
        netlist: str,
        fstart: float,
        fstop: float,
        points: int = 10,
    ) -> RunResult:
        """Run AC analysis."""
        return self.client.run_ac(
            fstart=fstart,
            fstop=fstop,
            points=points,
            netlist=netlist,
        )
