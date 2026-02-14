# MySpice GUI Implementation Plan

## Overview

This document outlines the plan for implementing a graphical user interface (GUI) for MySpice circuit simulator. The GUI will provide an intuitive interface for netlist editing, simulation control, and waveform visualization.

## Technology Selection

### Option Analysis

| Technology | Pros | Cons | Best For |
|------------|------|------|----------|
| **PySide6 (Qt)** | Mature, cross-platform, rich widgets, good plotting (pyqtgraph) | Requires Python, larger footprint | Full-featured desktop app |
| **Tauri (Rust + Web)** | Modern, small binary, web tech for UI | Young ecosystem, limited native widgets | Lightweight, modern app |
| **egui (Pure Rust)** | Pure Rust, fast, immediate mode | Limited widgets, basic plotting | Quick prototypes |
| **GTK4 (gtk-rs)** | Native look on Linux, Rust bindings | Platform-dependent appearance | Linux-focused app |

### Recommendation: **PySide6 (Qt)**

**Rationale:**
1. **Mature waveform plotting** - pyqtgraph/matplotlib integration is production-ready
2. **Existing Python tooling** - Leverages existing `tools/ai-agent/` Python ecosystem
3. **Cross-platform** - Consistent look on Windows/Linux/macOS
4. **Rich widgets** - Code editor, dockable panels, property editors
5. **HTTP client** - Easy integration with sim-api using httpx (already a dependency)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MySpice GUI                              │
│                        (PySide6 + Qt)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   Netlist    │  │  Simulation  │  │    Results Viewer      │ │
│  │    Editor    │  │   Control    │  │                        │ │
│  │              │  │    Panel     │  │  ┌──────────────────┐  │ │
│  │  - Syntax    │  │              │  │  │ Waveform Viewer  │  │ │
│  │    highlight │  │  - OP/DC/    │  │  │ (pyqtgraph)      │  │ │
│  │  - Auto-     │  │    TRAN/AC   │  │  └──────────────────┘  │ │
│  │    complete  │  │  - Params    │  │  ┌──────────────────┐  │ │
│  │  - Error     │  │  - Progress  │  │  │ Data Table       │  │ │
│  │    markers   │  │              │  │  │ (OP results)     │  │ │
│  └──────────────┘  └──────────────┘  │  └──────────────────┘  │ │
│                                       └────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                     Console / Log Output                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                      HTTP Client Layer                           │
│                        (httpx async)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        sim-api Server                            │
│                    (Rust, localhost:3000)                        │
├─────────────────────────────────────────────────────────────────┤
│  POST /v1/run/op     POST /v1/run/dc                            │
│  POST /v1/run/tran   POST /v1/run/ac                            │
│  GET  /v1/runs       GET  /v1/runs/:id/waveform                 │
│  GET  /v1/nodes      GET  /v1/devices                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. Main Window Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  File  Edit  Simulate  View  Help                              [_][□][X]
├─────────────────────────────────────────────────────────────────────┤
│ [New] [Open] [Save] │ [OP] [DC] [TRAN] [AC] │ [Run ▶] [Stop ■]     │
├────────────────┬────────────────────────────────────────────────────┤
│                │                                                    │
│   Project      │                 Waveform Viewer                    │
│   Explorer     │   ┌────────────────────────────────────────────┐  │
│                │   │     ^                                      │  │
│   📁 circuits  │   │  V  │    ╱╲    ╱╲    ╱╲                   │  │
│     📄 rc.cir  │   │     │   ╱  ╲  ╱  ╲  ╱  ╲                  │  │
│     📄 amp.cir │   │     │  ╱    ╲╱    ╲╱    ╲                 │  │
│                │   │     └────────────────────────► time        │  │
│                │   └────────────────────────────────────────────┘  │
│                │   Signal List: [✓] V(out) [✓] V(in) [ ] I(R1)     │
├────────────────┼────────────────────────────────────────────────────┤
│                │                                                    │
│   Netlist      │   Properties / Results                             │
│   Editor       │   ┌────────────────────────────────────────────┐  │
│                │   │  Node     │ Voltage (V)                    │  │
│   * RC circuit │   │───────────┼────────────────────────────────│  │
│   V1 in 0 5    │   │  in       │ 5.000000                       │  │
│   R1 in out 1k │   │  out      │ 3.333333                       │  │
│   C1 out 0 1n  │   │           │                                │  │
│   .tran 1n 10u │   └────────────────────────────────────────────┘  │
│   .end         │                                                    │
│                │                                                    │
├────────────────┴────────────────────────────────────────────────────┤
│ Console: Simulation completed. 45 time points, 3 rejected steps    │
│ > DC operating point: V(out) = 3.333V                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. Component Details

#### 2.1 Netlist Editor

**Features:**
- Syntax highlighting for SPICE netlists
- Line numbers
- Error/warning markers (from simulation)
- Auto-completion for:
  - Device types (R, C, L, V, I, D, M, E, G, F, H)
  - Control commands (.op, .dc, .tran, .ac, .model, .subckt)
  - Node names (from parsed netlist)
- Bracket matching
- Find/replace

**Implementation:**
```python
class NetlistEditor(QPlainTextEdit):
    def __init__(self):
        self.highlighter = SpiceHighlighter(self.document())
        self.line_number_area = LineNumberArea(self)
        self.completer = SpiceCompleter()
```

#### 2.2 Simulation Control Panel

**Features:**
- Analysis type selector (OP, DC, TRAN, AC)
- Parameter inputs for each analysis type:
  - **OP**: (no parameters)
  - **DC**: Source, Start, Stop, Step
  - **TRAN**: TStep, TStop, TStart, TMax
  - **AC**: Sweep type (DEC/OCT/LIN), Points, FStart, FStop
- Run/Stop buttons
- Progress indicator
- Server connection status

**Implementation:**
```python
class SimulationPanel(QWidget):
    def __init__(self, client: MySpiceClient):
        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.addTab(OpPanel(), "OP")
        self.analysis_tabs.addTab(DcPanel(), "DC")
        self.analysis_tabs.addTab(TranPanel(), "TRAN")
        self.analysis_tabs.addTab(AcPanel(), "AC")

        self.run_button = QPushButton("Run ▶")
        self.run_button.clicked.connect(self.run_simulation)
```

#### 2.3 Waveform Viewer

**Features:**
- Multi-signal plotting on same axes
- Zoom (mouse wheel, box zoom)
- Pan (drag)
- Cursors for measurement
- Multiple Y-axes (voltage, current)
- Signal list with visibility toggles
- Auto-scale / manual scale
- Export as PNG/SVG/CSV

**Implementation (pyqtgraph):**
```python
class WaveformViewer(pg.PlotWidget):
    def __init__(self):
        self.setBackground('w')
        self.showGrid(x=True, y=True)
        self.setLabel('bottom', 'Time', 's')
        self.setLabel('left', 'Voltage', 'V')
        self.legend = self.addLegend()

        self.signals = {}  # name -> PlotDataItem

    def add_waveform(self, name: str, x: list, y: list, color: str):
        pen = pg.mkPen(color, width=2)
        self.signals[name] = self.plot(x, y, pen=pen, name=name)

    def add_cursor(self, x: float):
        line = pg.InfiniteLine(x, pen='r')
        self.addItem(line)
```

#### 2.4 Results Table

**Features:**
- Operating point results display
- Sortable columns
- Copy to clipboard
- Filter by node/device
- Export to CSV

**Implementation:**
```python
class ResultsTable(QTableWidget):
    def __init__(self):
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Node", "Value"])
        self.setSortingEnabled(True)

    def load_op_results(self, nodes: list, solution: list):
        self.setRowCount(len(nodes))
        for i, (node, value) in enumerate(zip(nodes, solution)):
            self.setItem(i, 0, QTableWidgetItem(node))
            self.setItem(i, 1, QTableWidgetItem(f"{value:.6g}"))
```

#### 2.5 Console Output

**Features:**
- Scrollable log output
- Colored messages (info, warning, error)
- Timestamps
- Clear button
- Copy selection

---

## API Client

Reuse and extend the existing `tools/ai-agent/myspice_agent/client.py`:

```python
# myspice_gui/client.py

from dataclasses import dataclass
from typing import Optional, List
import httpx

@dataclass
class WaveformData:
    signal: str
    x_label: str
    y_label: str
    x_values: List[float]
    y_values: List[float]

class MySpiceClient:
    def __init__(self, base_url: str = "http://127.0.0.1:3000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)

    async def run_op(self, netlist: str) -> dict:
        resp = await self.client.post(
            f"{self.base_url}/v1/run/op",
            json={"netlist": netlist}
        )
        return resp.json()

    async def run_tran(self, netlist: str, tstep: float, tstop: float,
                       tstart: float = 0.0) -> dict:
        resp = await self.client.post(
            f"{self.base_url}/v1/run/tran",
            json={
                "netlist": netlist,
                "tstep": tstep,
                "tstop": tstop,
                "tstart": tstart
            }
        )
        return resp.json()

    async def get_waveform(self, run_id: int, signal: str) -> WaveformData:
        resp = await self.client.get(
            f"{self.base_url}/v1/runs/{run_id}/waveform",
            params={"signal": signal}
        )
        data = resp.json()
        return WaveformData(
            signal=data["signal"],
            x_label=data["x_label"],
            y_label=data["y_label"],
            x_values=data["x_values"],
            y_values=data["y_values"]
        )
```

---

## File Structure

```
tools/gui/
├── pyproject.toml           # Package configuration
├── README.md                # User documentation
├── myspice_gui/
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   ├── main_window.py       # Main application window
│   ├── client.py            # HTTP client (shared with ai-agent)
│   ├── editor/
│   │   ├── __init__.py
│   │   ├── editor.py        # Netlist editor widget
│   │   ├── highlighter.py   # Syntax highlighter
│   │   └── completer.py     # Auto-completion
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── panel.py         # Simulation control panel
│   │   └── worker.py        # Async simulation runner
│   ├── viewer/
│   │   ├── __init__.py
│   │   ├── waveform.py      # Waveform plot widget
│   │   ├── table.py         # Results table
│   │   └── cursors.py       # Measurement cursors
│   ├── console/
│   │   ├── __init__.py
│   │   └── console.py       # Console output widget
│   └── resources/
│       ├── icons/           # Application icons
│       ├── themes/          # Color themes
│       └── examples/        # Example netlists
└── tests/
    ├── test_client.py
    ├── test_highlighter.py
    └── test_viewer.py
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

**Goals:**
- Project setup with pyproject.toml
- Basic main window with dockable panels
- HTTP client integration
- Console output

**Deliverables:**
- [ ] Package structure and dependencies
- [ ] MainWindow class with menu bar and status bar
- [ ] DockWidget layout system
- [ ] Async HTTP client wrapper
- [ ] Console widget with logging

**Testing:**
- Manual: Launch app, verify window layout
- Unit: Client request/response parsing

### Phase 2: Netlist Editor (Week 2)

**Goals:**
- Full-featured text editor
- SPICE syntax highlighting
- File open/save dialogs

**Deliverables:**
- [ ] NetlistEditor widget
- [ ] SpiceHighlighter with token rules
- [ ] Line number display
- [ ] File I/O integration
- [ ] Recent files menu

**Syntax Highlighting Rules:**
```python
HIGHLIGHT_RULES = [
    # Comments
    (r'\*.*$', 'comment'),
    # Control commands
    (r'^\.(op|dc|tran|ac|model|subckt|ends|end|ic|param|include)\b', 'keyword'),
    # Device types
    (r'^[RCLVIDMEGFHXrclvidmegfhx]\w*', 'device'),
    # Numbers with suffixes
    (r'\b\d+\.?\d*[kKmMuUnNpPfF]?(eg)?\b', 'number'),
    # Node names (after device name)
    (r'\b[a-zA-Z_]\w*\b', 'identifier'),
]
```

### Phase 3: Simulation Control (Week 3) ✅

**Goals:**
- Analysis parameter forms
- Run/stop functionality
- Progress feedback

**Deliverables:**
- [x] SimulationPanel with tabs for each analysis
- [x] Parameter validation
- [x] Async simulation worker (QThread + asyncio)
- [x] Progress indicator
- [x] Error display

**Implementation Notes (2026-02-03):**
- Created `myspice_gui/simulation/` module with:
  - `panel.py`: SimulationPanel with OP/DC/TRAN/AC tabs, validation, progress bar
  - `worker.py`: SimulationWorker using QThread for non-blocking simulation
  - `__init__.py`: Module exports
- Added Run/Stop buttons with proper state management
- Added server connection status display
- Integrated with MainWindow via signals

**Analysis Forms:**

| Analysis | Fields |
|----------|--------|
| OP | (none) |
| DC | Source (combo), Start, Stop, Step |
| TRAN | TStep, TStop, TStart (optional), TMax (optional) |
| AC | Sweep (DEC/OCT/LIN), Points, FStart, FStop |

### Phase 4: Waveform Viewer (Week 4) ✅

**Goals:**
- Interactive waveform display
- Multi-signal support
- Basic measurements

**Deliverables:**
- [x] WaveformViewer widget (pyqtgraph-based)
- [x] Signal list with checkboxes
- [x] Zoom/pan controls
- [x] Auto-scale button
- [x] Export to PNG

**Implementation Notes (2026-02-04):**
- Created `myspice_gui/viewer/waveform.py`: WaveformViewer with:
  - Multi-signal plotting with automatic color assignment
  - pyqtgraph-based interactive plot
  - Auto-scale and reset view buttons
  - Grid toggle
  - Export to PNG/SVG
  - Context menu (right-click) with Reset Zoom, Auto Scale, Add Cursor, Toggle Grid, Export options
  - Double-click to add measurement cursor
  - CSV data export
- Created `myspice_gui/viewer/signal_list.py`: SignalListWidget with:
  - Visibility checkboxes for each signal
  - Color picker buttons
  - Show All / Hide All buttons
  - Remove signal buttons
- Created `myspice_gui/viewer/bode.py`: BodePlot with:
  - Magnitude (dB) and Phase (degrees) dual plots
  - Logarithmic frequency axis
  - Linked X-axes
  - Context menu with export options
  - CSV data export
- Created `myspice_gui/viewer/cursors.py`: CursorManager with:
  - Draggable vertical cursors
  - Delta measurement display
  - Frequency (1/Δ) calculation

**Interaction:**
- Mouse wheel: Zoom
- Left drag: Pan
- Right click: Context menu (reset zoom, export)
- Double click: Add cursor

### Phase 5: Results & Polish (Week 5) ✅

**Goals:**
- Operating point results table
- AC analysis (Bode plot)
- UI polish

**Deliverables:**
- [x] ResultsTable for OP
- [x] Bode plot (magnitude + phase)
- [x] DC sweep plot
- [x] Keyboard shortcuts
- [ ] Application icon (optional)
- [x] Dark/light theme toggle

**Implementation Notes (2026-02-03):**
- Created `myspice_gui/viewer/table.py`: Enhanced ResultsTable with:
  - Sortable columns
  - Copy to clipboard (Ctrl+C)
  - Export to CSV
  - Search/filter functionality
  - Engineering notation formatting
  - Context menu
- Created `myspice_gui/theme.py`: ThemeManager with:
  - Light and dark themes
  - Automatic stylesheet generation
  - QPalette generation
  - Settings persistence
  - Plot color adaptation
- Added keyboard shortcuts:
  - F5: Run simulation
  - Escape: Stop simulation
  - Ctrl+R: Re-run simulation
  - Ctrl+L: Clear console
  - Ctrl+1/2/3: Switch panels
  - Ctrl+Shift+T: Toggle theme
- Added View > Theme menu with Light/Dark options

### Phase 6: Advanced Features (Week 6+)

**Optional Enhancements:**
- [ ] Measurement cursors with delta display
- [x] FFT of transient waveforms
- [ ] Parameter sweep automation
- [ ] Netlist error underlining (from sim-api response)
- [ ] Session save/restore
- [ ] Multiple waveform panels

---

## Dependencies

```toml
# pyproject.toml
[project]
name = "myspice-gui"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "PySide6>=6.6.0",
    "pyqtgraph>=0.13.0",
    "httpx>=0.27.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-qt>=4.2.0",
]

[project.scripts]
myspice-gui = "myspice_gui.__main__:main"
```

---

## Usage

```bash
# Install
cd tools/gui
pip install -e .

# Start API server (in separate terminal)
cargo run -p sim-api -- --addr 127.0.0.1:3000

# Launch GUI
myspice-gui

# Or with specific server
myspice-gui --server http://localhost:3000
```

---

## Mockup Screenshots

### Main Window - Transient Analysis

```
┌─────────────────────────────────────────────────────────────────────┐
│  MySpice - rc_lowpass.cir                                      [─][□][×]
├─────────────────────────────────────────────────────────────────────┤
│  File  Edit  Simulate  View  Help                                   │
├─────────────────────────────────────────────────────────────────────┤
│  [📄 New] [📂 Open] [💾 Save] │ [▶ Run] [■ Stop] │ Server: ● Connected
├───────────────────┬─────────────────────────────────────────────────┤
│ * RC Low-pass     │ Transient Analysis                              │
│ * Filter          │ ┌─────────────────────────────────────────────┐ │
│ V1 in 0 PULSE     │ │ 5V ┤     ╭───────────────────────────────  │ │
│   (0 5 0 1n 1n    │ │    │    ╱                                   │ │
│    5u 10u)        │ │    │   ╱  ← V(out)                          │ │
│ R1 in out 1k      │ │    │  ╱                                     │ │
│ C1 out 0 100n     │ │    │ ╱                                      │ │
│ .tran 10n 50u     │ │ 0V ┼──────────────────────────────────────  │ │
│ .end              │ │    0        10u       20u       30u    time │ │
│                   │ └─────────────────────────────────────────────┘ │
│ [Line 5, Col 12]  │ Signals: [✓] V(in) [✓] V(out) [ ] I(R1)        │
├───────────────────┴─────────────────────────────────────────────────┤
│ Console                                                             │
│ > Simulation completed: 156 time points                            │
│ > Rise time (10%-90%): 220ns                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### AC Analysis - Bode Plot

```
┌─────────────────────────────────────────────────────────────────────┐
│ AC Analysis - Bode Plot                                             │
├─────────────────────────────────────────────────────────────────────┤
│ Magnitude (dB)                                                      │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │  0 ┤─────────────╲                                              │ │
│ │    │              ╲                                             │ │
│ │-20 ┤               ╲                                            │ │
│ │    │                ╲ -20dB/dec                                 │ │
│ │-40 ┤                 ╲───────────────────────                   │ │
│ │    1Hz     100Hz    10kHz    1MHz         freq (log)            │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│ Phase (degrees)                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │  0° ┤─────────╲                                                 │ │
│ │     │          ╲                                                │ │
│ │-45° ┤───────────●─────                                          │ │
│ │     │            ╲     fc = 1.59kHz                             │ │
│ │-90° ┤─────────────╲────────────────────────                     │ │
│ │     1Hz     100Hz    10kHz    1MHz         freq (log)           │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Future Enhancements

### Schematic Capture (Long-term)

A graphical schematic editor would be a major undertaking:

```
Phase A: Symbol Library
- Define component symbols (R, C, L, V, I, D, M)
- Symbol editor tool

Phase B: Schematic Canvas
- Place components
- Wire routing
- Property editing

Phase C: Netlist Generation
- Convert schematic to netlist
- Backannotation from simulation
```

This is a significant project and should be considered separately from the core GUI.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| pyqtgraph performance with large datasets | High | Downsampling, OpenGL backend |
| Qt event loop + asyncio compatibility | Medium | Use qasync or QThread workers |
| Cross-platform styling differences | Low | Use Qt style sheets for consistency |
| Server not running | Medium | Clear error message, auto-retry |

---

## Success Criteria

1. **Usability**: User can load netlist, run simulation, view waveforms in < 5 clicks
2. **Performance**: Render 10,000+ point waveforms at 60fps
3. **Reliability**: No crashes during normal operation
4. **Documentation**: README with installation and usage instructions

---

## References

- [PySide6 Documentation](https://doc.qt.io/qtforpython-6/)
- [pyqtgraph Documentation](https://pyqtgraph.readthedocs.io/)
- [Qt Style Sheets](https://doc.qt.io/qt-6/stylesheet-reference.html)
- [httpx Documentation](https://www.python-httpx.org/)
