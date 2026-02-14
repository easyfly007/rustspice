# AI Agent Integration Plan

This document outlines the design and implementation plan for integrating AI agents with RustSpice circuit simulator.

## 1. Overview

### 1.1 Goals

1. **Natural Language Interface**: Allow users to interact with the simulator using natural language
2. **Automated Analysis**: AI suggests appropriate analysis types based on circuit
3. **Result Interpretation**: AI explains simulation results in human-readable form
4. **Workflow Automation**: Support multi-step analysis sequences

### 1.2 Architecture Decision

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                    (Terminal / Chat / IDE)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AI Agent (Python)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ LLM Client  │  │ Tool Router │  │   Context   │             │
│  │  (Claude)   │◄─┤             │──┤   Manager   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │               │                                       │
│         ▼               ▼                                       │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Tool Implementations                    │       │
│  │  run_op │ run_dc │ run_tran │ run_ac │ query_*     │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    sim-api (Rust HTTP Server)                   │
│                       localhost:3000                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   sim-core (Simulation Engine)                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Design Principles

1. **Separation of Concerns**: AI logic in Python, simulation in Rust
2. **Stateless Communication**: HTTP REST API for all interactions
3. **Tool-based Architecture**: LLM uses function calling to invoke simulator
4. **Graceful Degradation**: Works without LLM for direct API access

---

## 2. Component Design

### 2.1 Python Package Structure

```
tools/ai-agent/
├── pyproject.toml          # Project configuration
├── requirements.txt        # Dependencies
├── README.md               # Usage documentation
│
├── rustspice_agent/          # Main package
│   ├── __init__.py
│   ├── cli.py              # CLI entry point
│   ├── agent.py            # Core AI agent logic
│   ├── client.py           # HTTP client wrapper
│   ├── tools.py            # Tool definitions for LLM
│   ├── prompts.py          # System prompts and templates
│   ├── formatters.py       # Result formatting utilities
│   └── config.py           # Configuration management
│
└── tests/
    ├── __init__.py
    ├── test_client.py      # HTTP client tests
    ├── test_tools.py       # Tool execution tests
    ├── test_agent.py       # Agent integration tests
    └── conftest.py         # Test fixtures
```

### 2.2 HTTP Client (`client.py`)

Responsibilities:
- Manage connection to sim-api server
- Handle request/response serialization
- Provide error handling and retries
- Support both sync and async modes

```python
class SpiceClient:
    def __init__(self, base_url: str = "http://localhost:3000")

    # Circuit operations
    def load_netlist(self, netlist: str) -> dict
    def load_file(self, path: str) -> dict
    def get_summary(self) -> dict
    def get_nodes(self) -> list[str]
    def get_devices(self) -> list[dict]  # Requires API addition

    # Simulation execution
    def run_op(self, netlist: str = None) -> RunResult
    def run_dc(self, source: str, start: float, stop: float, step: float) -> RunResult
    def run_tran(self, tstep: float, tstop: float, tstart: float = 0) -> RunResult
    def run_ac(self, sweep: str, points: int, fstart: float, fstop: float) -> RunResult

    # Result access
    def get_runs(self) -> list[RunSummary]
    def get_run(self, run_id: int) -> RunResult
    def export_run(self, run_id: int, path: str, format: str) -> None
```

### 2.3 Tool Definitions (`tools.py`)

Tools exposed to the LLM for function calling:

```python
TOOLS = [
    {
        "name": "run_operating_point",
        "description": "Run DC operating point analysis to find steady-state voltages and currents",
        "parameters": {
            "netlist": {"type": "string", "description": "SPICE netlist text"}
        }
    },
    {
        "name": "run_dc_sweep",
        "description": "Sweep a voltage or current source and plot DC transfer characteristics",
        "parameters": {
            "netlist": {"type": "string", "description": "SPICE netlist text"},
            "source": {"type": "string", "description": "Source name to sweep (e.g., 'V1')"},
            "start": {"type": "number", "description": "Start value"},
            "stop": {"type": "number", "description": "Stop value"},
            "step": {"type": "number", "description": "Step size"}
        }
    },
    {
        "name": "run_transient",
        "description": "Run time-domain transient analysis",
        "parameters": {
            "netlist": {"type": "string", "description": "SPICE netlist text"},
            "tstop": {"type": "number", "description": "Stop time in seconds"},
            "tstep": {"type": "number", "description": "Output time step"}
        }
    },
    {
        "name": "run_ac_analysis",
        "description": "Run small-signal AC frequency analysis (Bode plot)",
        "parameters": {
            "netlist": {"type": "string", "description": "SPICE netlist text"},
            "fstart": {"type": "number", "description": "Start frequency in Hz"},
            "fstop": {"type": "number", "description": "Stop frequency in Hz"},
            "points_per_decade": {"type": "integer", "description": "Points per decade", "default": 10}
        }
    },
    {
        "name": "get_circuit_info",
        "description": "Get information about the loaded circuit (nodes, devices, models)",
        "parameters": {}
    },
    {
        "name": "get_node_voltage",
        "description": "Get the voltage at a specific node from simulation results",
        "parameters": {
            "run_id": {"type": "integer", "description": "Simulation run ID"},
            "node": {"type": "string", "description": "Node name"}
        }
    },
    {
        "name": "export_results",
        "description": "Export simulation results to a file",
        "parameters": {
            "run_id": {"type": "integer", "description": "Simulation run ID"},
            "path": {"type": "string", "description": "Output file path"},
            "format": {"type": "string", "enum": ["psf", "csv", "json"], "default": "csv"}
        }
    }
]
```

### 2.4 AI Agent Core (`agent.py`)

The agent orchestrates LLM interaction and tool execution:

```python
class SpiceAgent:
    def __init__(self, client: SpiceClient, llm_client: Anthropic)

    def chat(self, user_message: str) -> str:
        """Process user message and return response"""

    def _build_messages(self, user_message: str) -> list[dict]:
        """Build message list with system prompt and context"""

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool and return formatted result"""

    def _format_for_user(self, result: dict, analysis_type: str) -> str:
        """Format simulation results for human reading"""
```

### 2.5 System Prompt (`prompts.py`)

```python
SYSTEM_PROMPT = """You are a helpful circuit simulation assistant for RustSpice,
a SPICE-compatible circuit simulator.

You can help users:
1. Create and modify SPICE netlists
2. Run circuit simulations (OP, DC sweep, Transient, AC analysis)
3. Interpret and explain simulation results
4. Suggest appropriate analysis types for their circuits

When the user describes a circuit or asks for analysis:
1. If they provide a netlist, use it directly
2. If they describe a circuit, help them create the netlist
3. Choose the appropriate analysis type based on their question
4. After simulation, explain the results in plain language

SPICE Netlist Basics:
- Resistor: R<name> <n+> <n-> <value>
- Capacitor: C<name> <n+> <n-> <value>
- Inductor: L<name> <n+> <n-> <value>
- Voltage source: V<name> <n+> <n-> DC <value> [AC <mag> [phase]]
- Current source: I<name> <n+> <n-> DC <value>
- Diode: D<name> <anode> <cathode> <model>
- MOSFET: M<name> <drain> <gate> <source> <bulk> <model> [W=<w>] [L=<l>]

Analysis Commands:
- .op - DC operating point
- .dc <source> <start> <stop> <step> - DC sweep
- .tran <tstep> <tstop> - Transient analysis
- .ac <type> <pts> <fstart> <fstop> - AC analysis (type: dec/oct/lin)

Always include .end at the end of netlists.
Node 0 or GND is always ground reference.
"""
```

---

## 3. API Enhancements Required

### 3.1 Missing Endpoints (Must Implement)

| Endpoint | Purpose | Priority |
|----------|---------|----------|
| `POST /v1/run/ac` | AC analysis | High |
| `GET /v1/devices` | List all devices | Medium |
| `GET /v1/devices/:name` | Device details | Medium |
| `GET /v1/models` | List all models | Medium |
| `GET /v1/runs/:id/waveform` | Get signal data | High |

### 3.2 AC Analysis Endpoint

```rust
// Request
POST /v1/run/ac
{
    "netlist": "...",
    "sweep": "dec",      // dec, oct, lin
    "points": 10,        // points per decade/octave or total
    "fstart": 1.0,       // start frequency Hz
    "fstop": 1e6         // stop frequency Hz
}

// Response
{
    "run_id": 1,
    "analysis": "Ac",
    "status": "Converged",
    "frequencies": [1.0, 10.0, 100.0, ...],
    "nodes": ["in", "out"],
    "magnitude_db": [[0.0, -3.01, ...], [0.0, -0.1, ...]],
    "phase_deg": [[0.0, -45.0, ...], [0.0, -1.0, ...]]
}
```

### 3.3 Waveform Endpoint

```rust
// Request
GET /v1/runs/1/waveform?signal=V(out)

// Response
{
    "signal": "V(out)",
    "analysis": "Tran",
    "x_label": "time",
    "x_unit": "s",
    "y_label": "voltage",
    "y_unit": "V",
    "x_values": [0.0, 1e-6, 2e-6, ...],
    "y_values": [0.0, 0.5, 0.9, ...]
}
```

---

## 4. Implementation Phases

### Phase 1: Foundation (Week 1)

**Objective**: Basic working agent without LLM

1. **Python project setup**
   - Create pyproject.toml with dependencies
   - Set up package structure
   - Configure testing framework (pytest)

2. **HTTP client implementation**
   - Implement SpiceClient class
   - Add request/response handling
   - Write unit tests with mocked responses

3. **Basic CLI**
   - Argument parsing (click/typer)
   - Direct command mode (no AI)
   - Example: `rustspice-agent run-op netlist.cir`

**Deliverables**:
- Working HTTP client
- Basic CLI for direct API access
- Unit tests passing

### Phase 2: API Completion (Week 1-2)

**Objective**: Complete sim-api endpoints needed for AI

1. **AC analysis endpoint**
   - Add route handler in http.rs
   - Wire up to Engine::run_ac_result()
   - Return frequency/magnitude/phase data

2. **Waveform query endpoint**
   - Parse signal name (V(node), I(device))
   - Extract data from RunResult
   - Format response with metadata

3. **Device/model query endpoints**
   - Extract from Circuit structure
   - Return structured JSON

**Deliverables**:
- All planned API endpoints working
- API tests passing

### Phase 3: LLM Integration (Week 2-3)

**Objective**: AI-powered interaction

1. **Tool definitions**
   - Define all tools in Anthropic format
   - Implement tool execution router
   - Handle tool results formatting

2. **Agent implementation**
   - Integrate Claude API (anthropic-sdk)
   - Implement conversation loop
   - Handle tool calling flow

3. **Result formatting**
   - Convert simulation results to readable text
   - Format tables for terminal display
   - Generate suggestions for follow-up

**Deliverables**:
- Working AI agent with tool calling
- Natural language circuit analysis

### Phase 4: Enhanced Features (Week 3-4)

**Objective**: Production-ready experience

1. **Context management**
   - Remember previous circuits/results
   - Support multi-turn conversations
   - Handle netlist modifications

2. **Error handling**
   - Parse simulation errors intelligently
   - Suggest fixes for common issues
   - Graceful degradation without LLM

3. **Output enhancements**
   - ASCII waveform plots in terminal
   - Markdown table formatting
   - Export suggestions

**Deliverables**:
- Robust error handling
- Enhanced output formatting
- Documentation and examples

---

## 5. Interaction Examples

### 5.1 Simple Operating Point

```
User: Analyze this voltage divider: 5V source, R1=1k, R2=2k

Agent: I'll create and analyze that voltage divider circuit.

[Calling run_operating_point with netlist:
V1 in 0 DC 5
R1 in out 1k
R2 out 0 2k
.op
.end]

Results:
┌─────────┬──────────┐
│ Node    │ Voltage  │
├─────────┼──────────┤
│ in      │ 5.000 V  │
│ out     │ 3.333 V  │
└─────────┴──────────┘

The output voltage is 3.333V, which matches the expected value
from the voltage divider formula: Vout = 5V × (2k/(1k+2k)) = 3.333V
```

### 5.2 DC Sweep Analysis

```
User: How does the output change as I vary V1 from 0 to 10V?

Agent: I'll run a DC sweep analysis to show you.

[Calling run_dc_sweep with source=V1, start=0, stop=10, step=0.5]

Results:
  V1    │  V(out)
────────┼──────────
  0.0V  │  0.000V
  2.0V  │  1.333V
  4.0V  │  2.667V
  6.0V  │  4.000V
  8.0V  │  5.333V
 10.0V  │  6.667V

The output varies linearly with input, maintaining the 2/3 ratio
from the voltage divider. This confirms the circuit is operating
in its linear region throughout the sweep range.
```

### 5.3 Transient Analysis

```
User: What happens if I add a 1uF capacitor across R2 and apply a step input?

Agent: Good question! Adding a capacitor will create an RC time constant.
Let me modify the circuit and run a transient analysis.

[Calling run_transient with netlist:
V1 in 0 PULSE(0 5 0 1n 1n 10m 20m)
R1 in out 1k
R2 out 0 2k
C1 out 0 1u
.tran 100u 5m
.end]

The time constant τ = (R1||R2) × C = 667Ω × 1µF = 667µs

Results show V(out) rising from 0V with exponential approach to 3.33V:
- At t=667µs (1τ): V(out) ≈ 2.1V (63% of final)
- At t=2ms (3τ): V(out) ≈ 3.2V (95% of final)
- At t=3.3ms (5τ): V(out) ≈ 3.31V (99% of final)

Would you like me to export this waveform data for plotting?
```

### 5.4 AC Analysis

```
User: What's the frequency response of this RC filter?

Agent: I'll run an AC analysis to generate a Bode plot.

[Calling run_ac_analysis with fstart=1, fstop=1e6, points_per_decade=10]

This is a first-order low-pass filter with:
- Cutoff frequency: fc = 1/(2π×R×C) = 1/(2π×667Ω×1µF) = 239 Hz

Frequency Response:
    Freq    │ Gain (dB) │ Phase
────────────┼───────────┼────────
     1 Hz   │   0.0 dB  │   0°
    10 Hz   │  -0.0 dB  │  -2°
   100 Hz   │  -1.3 dB  │ -23°
   239 Hz   │  -3.0 dB  │ -45°  ← cutoff
  1000 Hz   │ -12.5 dB  │ -76°
 10000 Hz   │ -32.5 dB  │ -88°

The -3dB point is at 239 Hz as expected. Above cutoff, gain rolls
off at -20dB/decade (first-order filter characteristic).
```

---

## 6. Configuration

### 6.1 Environment Variables

```bash
# Required for AI features
ANTHROPIC_API_KEY=sk-ant-...

# Optional
RUSTSPICE_API_URL=http://localhost:3000  # Default
RUSTSPICE_MODEL=claude-sonnet-4-20250514      # Default
RUSTSPICE_TIMEOUT=30                     # Request timeout seconds
```

### 6.2 Configuration File (~/.rustspice/config.toml)

```toml
[api]
url = "http://localhost:3000"
timeout = 30

[ai]
model = "claude-sonnet-4-20250514"
max_tokens = 4096
temperature = 0.3

[output]
precision = 6
format = "table"  # table, csv, json
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

- HTTP client with mocked responses
- Tool execution with mocked client
- Result formatting utilities

### 7.2 Integration Tests

- End-to-end with running sim-api server
- Compare results against known circuits
- Verify all analysis types

### 7.3 AI Tests (Optional)

- Verify tool calling works correctly
- Test error handling for invalid requests
- Check response formatting

---

## 8. Dependencies

### 8.1 Required

```
httpx>=0.27.0          # HTTP client (async support)
click>=8.1.0           # CLI framework
rich>=13.0.0           # Terminal formatting
pydantic>=2.0.0        # Data validation
```

### 8.2 For AI Features

```
anthropic>=0.40.0      # Claude API client
```

### 8.3 For Development

```
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-httpx>=0.30.0   # HTTP mocking
black>=24.0.0
ruff>=0.5.0
mypy>=1.10.0
```

---

## 9. Risk Assessment

### 9.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM hallucinations | Medium | Validate netlists before simulation |
| API latency | Low | Async operations, timeout handling |
| Large result sets | Medium | Pagination, streaming responses |
| Convergence failures | Medium | Clear error messages, suggestions |

### 9.2 Scope Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Feature creep | High | Strict phase boundaries |
| API changes | Medium | Version API, maintain compatibility |
| LLM API changes | Low | Abstract behind interface |

---

## 10. Success Criteria

### Phase 1
- [ ] HTTP client connects to sim-api
- [ ] All existing endpoints accessible
- [ ] Basic CLI working without AI

### Phase 2
- [ ] AC analysis endpoint working
- [ ] Waveform query endpoint working
- [ ] All tests passing

### Phase 3
- [ ] Natural language queries work
- [ ] Tool calling executes correctly
- [ ] Results formatted readably

### Phase 4
- [ ] Multi-turn conversations work
- [ ] Error recovery is graceful
- [ ] Documentation complete

---

## 11. Future Enhancements

After initial implementation:

1. **Visualization Integration**
   - Generate matplotlib/plotly code
   - Direct terminal plots (plotext)

2. **Circuit Synthesis**
   - AI suggests circuits for requirements
   - Component value optimization

3. **Educational Mode**
   - Explain circuit theory
   - Step-by-step analysis

4. **Batch Operations**
   - Monte Carlo analysis
   - Parameter sweeps
   - Corner analysis
