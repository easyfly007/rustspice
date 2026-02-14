# RustSpice GUI

Graphical user interface for RustSpice circuit simulator, built with PySide6 (Qt for Python).

## Features

- **Netlist Editor**: Full-featured editor with:
  - Syntax highlighting for SPICE keywords, devices, numbers
  - Line numbers with current line highlighting
  - Auto-completion for device types, commands, node names
  - Smart indentation (Tab/Shift+Tab)
  - Cursor position tracking
- **Waveform Viewer**: Interactive time-domain plotting with:
  - Multi-signal display with individual colors
  - Mouse wheel zoom and click-drag pan
  - Auto-scale and reset view
  - Grid toggle
  - Export to PNG/SVG
- **Bode Plot**: AC analysis visualization with:
  - Magnitude (dB) vs frequency
  - Phase (degrees) vs frequency
  - Logarithmic frequency axis
  - Linked X-axes for synchronized zooming
- **Measurement Cursors**:
  - Draggable vertical cursors
  - Delta time measurement
  - Frequency calculation (1/Delta)
  - Engineering notation display
- **Signal List Panel**:
  - Visibility checkboxes per signal
  - Color picker for each signal
  - Show all / Hide all buttons
- **Simulation Control**: Run OP, DC, TRAN, AC analyses
- **Results Panel**: Table and text views for results with engineering notation
- **Console Output**: Colored log messages with timestamps
- **Dockable Panels**: Flexible layout customization

## Installation

### Prerequisites

- Python 3.10 or later
- sim-api server running (from the main RustSpice project)

### Install from source

```bash
cd tools/gui
pip install -e .
```

### Install with development dependencies

```bash
pip install -e ".[dev]"
```

## Usage

### 1. Start the sim-api server

In a separate terminal:

```bash
# From the project root
cargo run -p sim-api -- --addr 127.0.0.1:3000
```

### 2. Launch the GUI

```bash
# Default server (localhost:3000)
rustspice-gui

# Connect to specific server
rustspice-gui --server http://192.168.1.100:3000

# Open a netlist file
rustspice-gui my_circuit.cir
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+N | New netlist |
| Ctrl+O | Open file |
| Ctrl+S | Save file |
| Ctrl+Shift+S | Save as |
| F5 | Run simulation |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+Q | Quit |

## Screenshot

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  RustSpice - rc_lowpass.cir                                              [─][□][×]
├─────────────────────────────────────────────────────────────────────────────────┤
│  File  Edit  Simulate  View  Help                                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  [New] [Open] [Save] │ [Run ▶]                                                  │
├────────────────────┬────────────────────────────────────┬───────────────────────┤
│                    │ [Waveform] [Bode]                  │ [Simulation][Signals] │
│   * RC circuit     │ ┌───────────────────────────────┐  │                       │
│   V1 in 0 5        │ │     ^                         │  │ Transient Analysis    │
│   R1 in out 1k     │ │   5 │  ___    ___    ___     │  │ tstep: [    1n  ]     │
│   C1 out 0 100n    │ │     │ /   \  /   \  /        │  │ tstop: [    1m  ]     │
│   .tran 10n 50u    │ │   0 │/     \/     \/         │  │                       │
│   .end             │ │     +------------------→ t   │  ├───────────────────────┤
│                    │ └───────────────────────────────┘  │ Signals               │
│                    ├────────────────────────────────────┤ [√] ■ V(in)     [×]  │
│                    │ Variable      │ Value              │ [√] ■ V(out)    [×]  │
│                    │───────────────┼────────────────────│ [All] [None]         │
│                    │ Time Points   │ 5001               ├───────────────────────┤
│                    │ Start Time    │ 0 s                │ Cursors               │
│                    │ Stop Time     │ 50 us              │ C1: 10.0 us           │
├────────────────────┴────────────────────────────────────┤ C2: 35.0 us           │
│ Console                                                 │ Δ:  25.0 us           │
│ [12:34:56] Connected to server                         │ 1/Δ: 40.0 kHz         │
│ [12:34:58] TRAN analysis completed successfully        │ [Add C1][Add C2][Clr] │
└─────────────────────────────────────────────────────────┴───────────────────────┘
```

## Architecture

```
rustspice_gui/
├── __init__.py       # Package exports
├── __main__.py       # Entry point
├── main_window.py    # Main window and panels
├── client.py         # HTTP client for sim-api
├── editor/           # Netlist editor components
│   ├── __init__.py
│   ├── editor.py     # Main editor with line numbers
│   ├── highlighter.py # Syntax highlighting
│   └── completer.py  # Auto-completion
├── viewer/           # Waveform viewer components
│   ├── __init__.py
│   ├── waveform.py   # Time-domain waveform viewer
│   ├── bode.py       # Bode plot for AC analysis
│   ├── signal_list.py # Signal list with visibility
│   └── cursors.py    # Measurement cursors
└── console/          # Console output widget
    ├── __init__.py
    └── console.py
```

## Development

### Running tests

```bash
pytest
```

### Code formatting

```bash
black rustspice_gui/
ruff check rustspice_gui/
```

## Roadmap

- [x] **Phase 1**: Core infrastructure, main window, HTTP client, console
- [x] **Phase 2**: Netlist editor with syntax highlighting
- [x] **Phase 3**: Simulation control panel with tabbed analysis types
- [x] **Phase 4**: Waveform viewer with pyqtgraph
- [x] **Phase 5**: Results table, Bode plot, signal list
- [x] **Phase 6**: Measurement cursors with delta display
- [ ] **Future**: Themes (dark/light), FFT analysis, waveform export

## Syntax Highlighting

The editor highlights:
- **Comments**: Gray italic (`* comment` or `; comment`)
- **Control commands**: Blue bold (`.op`, `.dc`, `.tran`, `.model`, etc.)
- **Device names**: Purple bold (`R1`, `C1`, `M1`, etc.)
- **Numbers**: Dark cyan (`1k`, `100n`, `1.5e-6`, etc.)
- **Waveforms**: Orange bold (`PULSE`, `PWL`, `SIN`, `EXP`)
- **Parameters**: Dark green (`W=`, `L=`, etc.)

## License

MIT License
