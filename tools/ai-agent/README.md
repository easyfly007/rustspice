# MySpice AI Agent

AI-powered command-line interface for MySpice circuit simulator. The agent provides both direct simulation commands and an interactive AI chat mode powered by Claude.

## Features

- **Direct CLI commands** for circuit simulation (no AI required)
- **Interactive AI chat mode** for natural language circuit analysis
- **Multiple analysis types**: Operating Point (OP), DC Sweep, Transient, AC
- **Export results** in PSF, CSV, or JSON formats
- **Rich terminal output** with formatted tables and markdown

## Installation

### Basic Installation (CLI only)

```bash
cd tools/ai-agent
pip install .
```

### With AI Features

```bash
pip install ".[ai]"
```

### Development Installation

```bash
pip install -e ".[all]"
```

## Prerequisites

1. **MySpice API Server**: The AI agent communicates with the MySpice simulation engine via HTTP API. Start the server first:

   ```bash
   # From project root
   cargo run -p sim-api -- --addr 127.0.0.1:3000
   ```

2. **Anthropic API Key** (for AI features): Set the `ANTHROPIC_API_KEY` environment variable:

   ```bash
   export ANTHROPIC_API_KEY=your-api-key-here
   ```

## Usage

### Direct CLI Commands

Run simulations directly without AI:

```bash
# Operating Point analysis
myspice-agent op circuit.cir

# DC Sweep
myspice-agent dc circuit.cir -s V1 --start 0 --stop 5 --step 0.5

# Transient analysis
myspice-agent tran circuit.cir --tstop 1e-3 --tstep 1e-6

# AC analysis
myspice-agent ac circuit.cir --fstart 1 --fstop 1e6 --points 10

# Check server status
myspice-agent status

# List simulation runs
myspice-agent runs
```

### Export Options

All simulation commands support export options:

```bash
# Export to CSV
myspice-agent op circuit.cir -o results.csv -f csv

# Export to JSON
myspice-agent dc circuit.cir -s V1 --start 0 --stop 5 --step 1 -o results.json -f json

# Export to PSF
myspice-agent tran circuit.cir --tstop 1e-3 -o results.psf -f psf
```

### Interactive AI Mode

Start the interactive AI chat mode:

```bash
myspice-agent
```

In interactive mode, you can:

- Describe circuits in natural language
- Paste SPICE netlists for analysis
- Ask questions about simulation results
- Request specific analysis types

Interactive commands:
- `/help` - Show help
- `/clear` - Clear conversation history
- `/quit` - Exit

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MYSPICE_API_URL` | API server URL | `http://localhost:3000` |
| `MYSPICE_TIMEOUT` | Request timeout (seconds) | `30.0` |
| `MYSPICE_MODEL` | Claude model name | `claude-sonnet-4-20250514` |
| `MYSPICE_PRECISION` | Output precision (digits) | `6` |
| `ANTHROPIC_API_KEY` | Anthropic API key | (required for AI mode) |

### Configuration File

Create `~/.myspice/config.toml` for persistent settings:

```toml
[api]
url = "http://localhost:3000"
timeout = 30.0

[ai]
model = "claude-sonnet-4-20250514"
max_tokens = 4096
temperature = 0.3

[output]
precision = 6
format = "table"
```

## Architecture

```
myspice-agent
    |
    +-- cli.py          # Click-based CLI entry point
    +-- client.py       # HTTP client for sim-api
    +-- agent.py        # AI agent with Claude integration
    +-- tools.py        # LLM tool definitions
    +-- formatters.py   # Result formatting utilities
    +-- config.py       # Configuration management
    +-- prompts.py      # System prompts for AI
```

### AI Tools

The agent provides these tools to Claude for circuit simulation:

| Tool | Description |
|------|-------------|
| `run_operating_point` | DC operating point analysis |
| `run_dc_sweep` | DC transfer characteristic sweep |
| `run_transient` | Time-domain transient analysis |
| `run_ac_analysis` | AC frequency response analysis |
| `get_circuit_info` | Query circuit structure |
| `get_node_voltage` | Get specific node voltage |
| `list_simulation_runs` | List all simulation runs |
| `get_waveform` | Get waveform data for plotting |
| `export_results` | Export results to file |

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=myspice_agent

# Run only unit tests (skip integration tests requiring server)
pytest -m "not integration"
```

### Code Quality

```bash
# Format code
black myspice_agent tests

# Lint
ruff check myspice_agent tests

# Type checking
mypy myspice_agent
```

## License

MIT License
