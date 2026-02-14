"""
Simulation control panel widget.

Provides UI for configuring and running different types of circuit analysis:
- Operating Point (OP)
- DC Sweep
- Transient (TRAN)
- AC Frequency Analysis
"""

from typing import Optional, Dict, Any, List

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QTabWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
    QProgressBar,
    QFrame,
    QSizePolicy,
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, Signal, Slot

from ..client import AcSweepType
from .worker import AnalysisType


class ValidationError(Exception):
    """Exception raised when parameter validation fails."""
    pass


class AnalysisPanelBase(QWidget):
    """Base class for analysis parameter panels."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Override in subclass to set up UI."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get analysis parameters. Override in subclass."""
        return {}

    def validate(self) -> List[str]:
        """
        Validate parameters.

        Returns:
            List of validation error messages (empty if valid)
        """
        return []

    def set_enabled(self, enabled: bool):
        """Enable or disable all input controls."""
        for child in self.findChildren(QWidget):
            if isinstance(child, (QDoubleSpinBox, QSpinBox, QComboBox)):
                child.setEnabled(enabled)


class OpPanel(AnalysisPanelBase):
    """Operating Point analysis panel."""

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Description
        desc = QLabel(
            "<b>Operating Point Analysis</b><br>"
            "Calculates the DC operating point of the circuit.<br><br>"
            "No additional parameters required."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addStretch()


class DcPanel(AnalysisPanelBase):
    """DC Sweep analysis panel."""

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Description
        desc = QLabel(
            "<b>DC Sweep Analysis</b><br>"
            "Sweeps a voltage or current source and calculates "
            "the DC operating point at each step."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Parameters form
        form = QFormLayout()
        form.setSpacing(8)

        # Source selector
        self._source = QComboBox()
        self._source.setEditable(True)
        self._source.setPlaceholderText("Enter source name")
        self._source.addItems(["V1", "V2", "I1"])
        self._source.setToolTip("Name of the voltage or current source to sweep")
        form.addRow("Source:", self._source)

        # Start value
        self._start = QDoubleSpinBox()
        self._start.setRange(-1e12, 1e12)
        self._start.setDecimals(6)
        self._start.setValue(0.0)
        self._start.setSuffix(" V")
        self._start.setToolTip("Starting value for the sweep")
        form.addRow("Start:", self._start)

        # Stop value
        self._stop = QDoubleSpinBox()
        self._stop.setRange(-1e12, 1e12)
        self._stop.setDecimals(6)
        self._stop.setValue(5.0)
        self._stop.setSuffix(" V")
        self._stop.setToolTip("Ending value for the sweep")
        form.addRow("Stop:", self._stop)

        # Step value
        self._step = QDoubleSpinBox()
        self._step.setRange(1e-15, 1e12)
        self._step.setDecimals(6)
        self._step.setValue(0.1)
        self._step.setSuffix(" V")
        self._step.setToolTip("Step size for the sweep")
        form.addRow("Step:", self._step)

        layout.addLayout(form)
        layout.addStretch()

    def get_params(self) -> Dict[str, Any]:
        return {
            "source": self._source.currentText(),
            "start": self._start.value(),
            "stop": self._stop.value(),
            "step": self._step.value(),
        }

    def validate(self) -> List[str]:
        errors = []
        source = self._source.currentText().strip()
        if not source:
            errors.append("Source name is required")
        elif not source[0].upper() in ('V', 'I'):
            errors.append("Source must be a voltage (V) or current (I) source")

        start = self._start.value()
        stop = self._stop.value()
        step = self._step.value()

        if start == stop:
            errors.append("Start and stop values cannot be the same")
        if step <= 0:
            errors.append("Step must be positive")
        if (stop - start) / step > 10000:
            errors.append("Too many sweep points (max 10000)")

        return errors


class TranPanel(AnalysisPanelBase):
    """Transient analysis panel."""

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Description
        desc = QLabel(
            "<b>Transient Analysis</b><br>"
            "Performs time-domain simulation of the circuit."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Parameters form
        form = QFormLayout()
        form.setSpacing(8)

        # Time step
        self._tstep = QDoubleSpinBox()
        self._tstep.setRange(1e-18, 1e6)
        self._tstep.setDecimals(12)
        self._tstep.setValue(1e-9)
        self._tstep.setSuffix(" s")
        self._tstep.setToolTip("Suggested time step (actual step may be adjusted)")
        form.addRow("Time Step:", self._tstep)

        # Stop time
        self._tstop = QDoubleSpinBox()
        self._tstop.setRange(1e-18, 1e6)
        self._tstop.setDecimals(12)
        self._tstop.setValue(1e-3)
        self._tstop.setSuffix(" s")
        self._tstop.setToolTip("Simulation stop time")
        form.addRow("Stop Time:", self._tstop)

        # Start time (optional)
        self._tstart = QDoubleSpinBox()
        self._tstart.setRange(0, 1e6)
        self._tstart.setDecimals(12)
        self._tstart.setValue(0.0)
        self._tstart.setSuffix(" s")
        self._tstart.setToolTip("Time to start saving data (default: 0)")
        form.addRow("Start Time:", self._tstart)

        # Max step (optional)
        self._tmax = QDoubleSpinBox()
        self._tmax.setRange(0, 1e6)
        self._tmax.setDecimals(12)
        self._tmax.setValue(0.0)
        self._tmax.setSuffix(" s")
        self._tmax.setToolTip("Maximum time step (0 = automatic)")
        self._tmax.setSpecialValueText("Auto")
        form.addRow("Max Step:", self._tmax)

        layout.addLayout(form)
        layout.addStretch()

    def get_params(self) -> Dict[str, Any]:
        params = {
            "tstep": self._tstep.value(),
            "tstop": self._tstop.value(),
            "tstart": self._tstart.value(),
        }
        tmax = self._tmax.value()
        if tmax > 0:
            params["tmax"] = tmax
        return params

    def validate(self) -> List[str]:
        errors = []
        tstep = self._tstep.value()
        tstop = self._tstop.value()
        tstart = self._tstart.value()

        if tstep <= 0:
            errors.append("Time step must be positive")
        if tstop <= 0:
            errors.append("Stop time must be positive")
        if tstart >= tstop:
            errors.append("Start time must be less than stop time")
        if tstep > tstop:
            errors.append("Time step cannot be larger than stop time")
        if (tstop - tstart) / tstep > 1e7:
            errors.append("Too many time points (reduce stop time or increase step)")

        return errors


class AcPanel(AnalysisPanelBase):
    """AC frequency analysis panel."""

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Description
        desc = QLabel(
            "<b>AC Analysis</b><br>"
            "Performs small-signal frequency domain analysis."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Parameters form
        form = QFormLayout()
        form.setSpacing(8)

        # Sweep type
        self._sweep = QComboBox()
        self._sweep.addItems(["DEC", "OCT", "LIN"])
        self._sweep.setToolTip(
            "DEC: Points per decade\n"
            "OCT: Points per octave\n"
            "LIN: Linear sweep (total points)"
        )
        form.addRow("Sweep Type:", self._sweep)

        # Number of points
        self._points = QSpinBox()
        self._points.setRange(1, 10000)
        self._points.setValue(10)
        self._points.setToolTip("Number of points (per decade/octave for DEC/OCT)")
        form.addRow("Points:", self._points)

        # Start frequency
        self._fstart = QDoubleSpinBox()
        self._fstart.setRange(1e-6, 1e15)
        self._fstart.setDecimals(3)
        self._fstart.setValue(1.0)
        self._fstart.setSuffix(" Hz")
        self._fstart.setToolTip("Start frequency")
        form.addRow("Start Freq:", self._fstart)

        # Stop frequency
        self._fstop = QDoubleSpinBox()
        self._fstop.setRange(1e-6, 1e15)
        self._fstop.setDecimals(3)
        self._fstop.setValue(1e6)
        self._fstop.setSuffix(" Hz")
        self._fstop.setToolTip("Stop frequency")
        form.addRow("Stop Freq:", self._fstop)

        layout.addLayout(form)
        layout.addStretch()

    def get_params(self) -> Dict[str, Any]:
        sweep_map = {
            "DEC": AcSweepType.DEC,
            "OCT": AcSweepType.OCT,
            "LIN": AcSweepType.LIN
        }
        return {
            "sweep": sweep_map[self._sweep.currentText()],
            "points": self._points.value(),
            "fstart": self._fstart.value(),
            "fstop": self._fstop.value(),
        }

    def validate(self) -> List[str]:
        errors = []
        fstart = self._fstart.value()
        fstop = self._fstop.value()
        points = self._points.value()

        if fstart <= 0:
            errors.append("Start frequency must be positive")
        if fstop <= 0:
            errors.append("Stop frequency must be positive")
        if fstart >= fstop:
            errors.append("Start frequency must be less than stop frequency")
        if points <= 0:
            errors.append("Number of points must be positive")

        return errors


class SimulationPanel(QWidget):
    """
    Main simulation control panel.

    Provides tabbed interface for different analysis types with
    parameter configuration, run/stop controls, and progress indicator.

    Signals:
        run_requested: Emitted when user clicks Run (analysis_type, params)
        stop_requested: Emitted when user clicks Stop
    """

    run_requested = Signal(str, dict)  # analysis_type, params
    stop_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._is_running = False
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Analysis type tabs
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._op_panel = OpPanel()
        self._dc_panel = DcPanel()
        self._tran_panel = TranPanel()
        self._ac_panel = AcPanel()

        self._tabs.addTab(self._op_panel, "OP")
        self._tabs.addTab(self._dc_panel, "DC")
        self._tabs.addTab(self._tran_panel, "TRAN")
        self._tabs.addTab(self._ac_panel, "AC")

        layout.addWidget(self._tabs)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # Control buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self._run_btn = QPushButton("Run")
        self._run_btn.setMinimumHeight(32)
        self._run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 4px 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        btn_layout.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setMinimumHeight(32)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 4px 16px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #c41508;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        btn_layout.addWidget(self._stop_btn)

        layout.addLayout(btn_layout)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(8, 8, 8, 8)

        self._status_label = QLabel("Ready")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # Indeterminate
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(False)
        progress_layout.addWidget(self._progress_bar)

        layout.addWidget(progress_group)

        # Server status
        self._server_status = QLabel("Server: Unknown")
        self._server_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._server_status.setStyleSheet("color: gray;")
        layout.addWidget(self._server_status)

        layout.addStretch()

    def _connect_signals(self):
        """Connect internal signals."""
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._stop_btn.clicked.connect(self._on_stop_clicked)

    def _on_run_clicked(self):
        """Handle run button click."""
        # Validate parameters
        panel = self._get_current_panel()
        errors = panel.validate()

        if errors:
            self._status_label.setText(f"Error: {errors[0]}")
            self._status_label.setStyleSheet("color: red;")
            return

        # Get analysis type and params
        analysis_type = self.get_analysis_type()
        params = panel.get_params()

        self.run_requested.emit(analysis_type, params)

    def _on_stop_clicked(self):
        """Handle stop button click."""
        self.stop_requested.emit()

    def _get_current_panel(self) -> AnalysisPanelBase:
        """Get the currently selected analysis panel."""
        panels = [self._op_panel, self._dc_panel, self._tran_panel, self._ac_panel]
        return panels[self._tabs.currentIndex()]

    def get_analysis_type(self) -> str:
        """Get the currently selected analysis type."""
        types = ["op", "dc", "tran", "ac"]
        return types[self._tabs.currentIndex()]

    def get_analysis_enum(self) -> AnalysisType:
        """Get the currently selected analysis type as enum."""
        types = [AnalysisType.OP, AnalysisType.DC, AnalysisType.TRAN, AnalysisType.AC]
        return types[self._tabs.currentIndex()]

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for the current analysis type."""
        return self._get_current_panel().get_params()

    def validate(self) -> List[str]:
        """Validate current analysis parameters."""
        return self._get_current_panel().validate()

    # =========================================================================
    # State Management
    # =========================================================================

    def set_running(self, running: bool):
        """
        Set the running state of the panel.

        When running:
        - Run button is disabled
        - Stop button is enabled
        - Progress bar is visible
        - Parameter inputs are disabled
        """
        self._is_running = running

        self._run_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._progress_bar.setVisible(running)
        self._tabs.setEnabled(not running)

        # Disable inputs in all panels
        for panel in [self._op_panel, self._dc_panel, self._tran_panel, self._ac_panel]:
            panel.set_enabled(not running)

    def is_running(self) -> bool:
        """Check if a simulation is currently running."""
        return self._is_running

    def set_status(self, message: str, is_error: bool = False):
        """
        Update the status message.

        Args:
            message: Status message to display
            is_error: If True, display in red
        """
        self._status_label.setText(message)
        if is_error:
            self._status_label.setStyleSheet("color: red;")
        else:
            self._status_label.setStyleSheet("color: black;")

    def set_progress(self, message: str):
        """Update progress message during simulation."""
        self._status_label.setText(message)
        self._status_label.setStyleSheet("color: blue;")

    def set_success(self, message: str):
        """Display success message."""
        self._status_label.setText(message)
        self._status_label.setStyleSheet("color: green;")

    def set_server_status(self, connected: bool, url: str = ""):
        """
        Update server connection status.

        Args:
            connected: Whether server is connected
            url: Server URL (optional)
        """
        if connected:
            if url:
                self._server_status.setText(f"Server: Connected ({url})")
            else:
                self._server_status.setText("Server: Connected")
            self._server_status.setStyleSheet("color: green;")
        else:
            self._server_status.setText("Server: Disconnected")
            self._server_status.setStyleSheet("color: red;")

    def reset(self):
        """Reset the panel to initial state."""
        self.set_running(False)
        self.set_status("Ready")
