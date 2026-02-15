"""
FFT viewer widget for frequency-domain analysis of transient waveforms.

Provides:
- compute_fft(): Module-level function for FFT computation (testable without GUI)
- _make_window(): Window function generator (pure numpy, no scipy)
- FftViewer: Interactive FFT plot widget with window/NFFT selection
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QFileDialog,
    QMenu,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction

import pyqtgraph as pg


# Default colors
DEFAULT_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
]

# Available window functions
WINDOW_FUNCTIONS = ["Hann", "Hamming", "Blackman", "Rectangular", "Flat-top"]

# Available NFFT sizes
NFFT_OPTIONS = ["Auto", "1024", "2048", "4096", "8192", "16384", "32768", "65536"]


def _make_window(name: str, n: int) -> np.ndarray:
    """
    Create a window function array of length n.

    Implemented as pure numpy cosine-sum formulas (no scipy dependency).

    Args:
        name: Window function name (Rectangular, Hann, Hamming, Blackman, Flat-top)
        n: Window length

    Returns:
        Window array of shape (n,)
    """
    if n <= 0:
        return np.array([])

    if name == "Rectangular":
        return np.ones(n)

    # Common index array for cosine-sum windows
    k = np.arange(n)
    tau = 2.0 * np.pi * k / (n - 1) if n > 1 else np.zeros(n)

    if name == "Hann":
        return 0.5 - 0.5 * np.cos(tau)
    elif name == "Hamming":
        return 0.54 - 0.46 * np.cos(tau)
    elif name == "Blackman":
        return 0.42 - 0.5 * np.cos(tau) + 0.08 * np.cos(2.0 * tau)
    elif name == "Flat-top":
        return (
            0.21557895
            - 0.41663158 * np.cos(tau)
            + 0.277263158 * np.cos(2.0 * tau)
            - 0.083578947 * np.cos(3.0 * tau)
            + 0.006947368 * np.cos(4.0 * tau)
        )
    else:
        # Default to Hann
        return 0.5 - 0.5 * np.cos(tau)


def compute_fft(
    times: np.ndarray,
    values: np.ndarray,
    window: str = "Hann",
    nfft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT of (possibly non-uniform) time-domain data.

    Interpolates to a uniform grid, applies a window function,
    computes the FFT, and returns magnitude in dB.

    Args:
        times: Time values (may be non-uniform)
        values: Signal values corresponding to times
        window: Window function name
        nfft: Number of FFT points (None for auto)

    Returns:
        Tuple of (frequencies, magnitude_dB) arrays.
        DC bin (0 Hz) is excluded.
    """
    if len(times) < 2 or len(values) < 2:
        return np.array([]), np.array([])

    # Determine NFFT
    if nfft is None:
        # Next power of 2, capped at 65536
        raw = len(times)
        nfft = 1
        while nfft < raw:
            nfft <<= 1
        nfft = min(nfft, 65536)

    nfft = max(nfft, 4)  # Minimum sensible size

    # Create uniform time grid
    t_start = times[0]
    t_end = times[-1]
    dt = (t_end - t_start) / (nfft - 1)

    if dt <= 0:
        return np.array([]), np.array([])

    t_uniform = np.linspace(t_start, t_end, nfft)
    v_uniform = np.interp(t_uniform, times, values)

    # Apply window function
    win = _make_window(window, nfft)
    v_windowed = v_uniform * win

    # Compute FFT (real-valued input)
    spectrum = np.fft.rfft(v_windowed)
    frequencies = np.fft.rfftfreq(nfft, d=dt)

    # Compute magnitude in dB
    # Normalize: 2/N for single-sided spectrum, compensate for window energy loss
    win_sum = np.sum(win)
    if win_sum == 0:
        win_sum = 1.0
    magnitude = np.abs(spectrum) * 2.0 / win_sum

    # Convert to dB, avoiding log(0)
    with np.errstate(divide="ignore"):
        magnitude_db = 20.0 * np.log10(magnitude)
    magnitude_db = np.where(np.isfinite(magnitude_db), magnitude_db, -200.0)

    # Skip DC bin (index 0)
    return frequencies[1:], magnitude_db[1:]


@dataclass
class FftSignalData:
    """Data for an FFT signal."""
    name: str
    frequencies: np.ndarray
    magnitude_db: np.ndarray
    color: str
    visible: bool = True
    plot_item: Optional[pg.PlotDataItem] = None


class FftViewer(QWidget):
    """
    FFT viewer widget for frequency-domain analysis.

    Features:
    - FFT of transient simulation data
    - Window function selection (Hann, Hamming, Blackman, Rectangular, Flat-top)
    - NFFT size selection (Auto, 1024..65536)
    - Logarithmic frequency axis, dB magnitude axis
    - Multi-signal support
    - Auto-recompute on parameter change
    - Export to PNG/CSV
    - Context menu

    Signals:
        signal_added: Emitted when a signal is added (name)
        signal_removed: Emitted when a signal is removed (name)
    """

    signal_added = Signal(str)
    signal_removed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Computed FFT signals on the plot
        self._signals: Dict[str, FftSignalData] = {}
        # Raw time-domain data: name -> (times, values, color)
        self._time_data: Dict[str, Tuple[np.ndarray, np.ndarray, str]] = {}
        self._color_index = 0

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)

        # Window function selector
        toolbar.addWidget(QLabel("Window:"))
        self._window_combo = QComboBox()
        self._window_combo.addItems(WINDOW_FUNCTIONS)
        self._window_combo.setCurrentText("Hann")
        self._window_combo.currentTextChanged.connect(self._on_params_changed)
        toolbar.addWidget(self._window_combo)

        # NFFT selector
        toolbar.addWidget(QLabel("NFFT:"))
        self._nfft_combo = QComboBox()
        self._nfft_combo.addItems(NFFT_OPTIONS)
        self._nfft_combo.setCurrentText("Auto")
        self._nfft_combo.currentTextChanged.connect(self._on_params_changed)
        toolbar.addWidget(self._nfft_combo)

        # Compute button
        self._compute_btn = QPushButton("Compute FFT")
        self._compute_btn.clicked.connect(self.compute_all)
        toolbar.addWidget(self._compute_btn)

        self._auto_scale_btn = QPushButton("Auto Scale")
        self._auto_scale_btn.clicked.connect(self.auto_scale)
        toolbar.addWidget(self._auto_scale_btn)

        self._grid_btn = QPushButton("Grid")
        self._grid_btn.setCheckable(True)
        self._grid_btn.setChecked(True)
        self._grid_btn.clicked.connect(self._toggle_grid)
        toolbar.addWidget(self._grid_btn)

        self._export_btn = QPushButton("Export")
        self._export_btn.clicked.connect(self._export_image)
        toolbar.addWidget(self._export_btn)

        toolbar.addStretch()

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self.clear)
        toolbar.addWidget(self._clear_btn)

        layout.addLayout(toolbar)

        # Plot widget
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground('w')
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel('left', 'Magnitude', 'dB')
        self._plot_widget.setLabel('bottom', 'Frequency', 'Hz')
        self._plot_widget.setLogMode(x=True, y=False)
        self._plot_widget.setTitle("FFT")
        self._legend = self._plot_widget.addLegend()

        # Setup context menu
        self._setup_context_menu()

        layout.addWidget(self._plot_widget)

    def _setup_context_menu(self):
        """Set up the right-click context menu."""
        self._plot_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._plot_widget.customContextMenuRequested.connect(self._show_context_menu)

    def _show_context_menu(self, pos):
        """Show context menu at the given position."""
        menu = QMenu(self)

        # Reset Zoom action
        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.triggered.connect(self.auto_scale)
        menu.addAction(reset_zoom_action)

        menu.addSeparator()

        # Toggle Grid action
        toggle_grid_action = QAction("Toggle Grid", self)
        toggle_grid_action.setCheckable(True)
        toggle_grid_action.setChecked(self._grid_btn.isChecked())
        toggle_grid_action.triggered.connect(self._grid_btn.click)
        menu.addAction(toggle_grid_action)

        menu.addSeparator()

        # Export submenu
        export_menu = menu.addMenu("Export")

        export_png_action = QAction("Export as PNG...", self)
        export_png_action.triggered.connect(self._export_image)
        export_menu.addAction(export_png_action)

        export_csv_action = QAction("Export Data as CSV...", self)
        export_csv_action.triggered.connect(self._export_csv)
        export_menu.addAction(export_csv_action)

        menu.addSeparator()

        # Clear All action
        clear_action = QAction("Clear All Signals", self)
        clear_action.triggered.connect(self.clear)
        menu.addAction(clear_action)

        # Show the menu
        menu.exec_(self._plot_widget.mapToGlobal(pos))

    def set_time_data(
        self,
        name: str,
        times: np.ndarray,
        values: np.ndarray,
        color: str,
    ):
        """
        Store raw time-domain data for later FFT computation.

        Args:
            name: Signal name (e.g., "V(out)")
            times: Time values (may be non-uniform)
            values: Signal values
            color: Color for the FFT plot line
        """
        self._time_data[name] = (np.asarray(times), np.asarray(values), color)

    def compute_all(self):
        """Recompute FFT for all stored time-domain signals."""
        if not self._time_data:
            return

        # Get current parameters
        window = self._window_combo.currentText()
        nfft_text = self._nfft_combo.currentText()
        nfft = None if nfft_text == "Auto" else int(nfft_text)

        # Remove existing plot items
        for name in list(self._signals.keys()):
            self.remove_signal(name)

        # Recompute for each signal
        for name, (times, values, color) in self._time_data.items():
            freqs, mag_db = compute_fft(times, values, window=window, nfft=nfft)
            if len(freqs) > 0:
                self.add_signal(name, freqs, mag_db, color)

        self.auto_scale()

    def add_signal(
        self,
        name: str,
        frequencies: np.ndarray,
        magnitude_db: np.ndarray,
        color: Optional[str] = None,
    ):
        """
        Add a computed FFT signal to the plot.

        Args:
            name: Signal name
            frequencies: Frequency values in Hz
            magnitude_db: Magnitude values in dB
            color: Optional color (hex string)
        """
        # Remove existing signal with same name
        if name in self._signals:
            self.remove_signal(name)

        # Get color
        if color is None:
            color = DEFAULT_COLORS[self._color_index % len(DEFAULT_COLORS)]
            self._color_index += 1

        freq_arr = np.asarray(frequencies)
        mag_arr = np.asarray(magnitude_db)

        # Create plot item
        pen = pg.mkPen(color=color, width=2)
        plot_item = self._plot_widget.plot(
            freq_arr, mag_arr,
            pen=pen,
            name=name,
        )

        # Store signal data
        self._signals[name] = FftSignalData(
            name=name,
            frequencies=freq_arr,
            magnitude_db=mag_arr,
            color=color,
            visible=True,
            plot_item=plot_item,
        )

        self.signal_added.emit(name)

    def remove_signal(self, name: str):
        """Remove a signal from the FFT plot."""
        if name in self._signals:
            signal = self._signals[name]
            if signal.plot_item:
                self._plot_widget.removeItem(signal.plot_item)
            del self._signals[name]
            self.signal_removed.emit(name)

    def set_signal_visible(self, name: str, visible: bool):
        """Set signal visibility."""
        if name in self._signals:
            signal = self._signals[name]
            signal.visible = visible
            if signal.plot_item:
                signal.plot_item.setVisible(visible)

    def set_signal_color(self, name: str, color: str):
        """Set signal color and update both time data and plot."""
        if name in self._signals:
            signal = self._signals[name]
            signal.color = color
            if signal.plot_item:
                pen = pg.mkPen(color=color, width=2)
                signal.plot_item.setPen(pen)
        # Also update stored time data color
        if name in self._time_data:
            times, values, _ = self._time_data[name]
            self._time_data[name] = (times, values, color)

    def get_signal_names(self) -> List[str]:
        """Get list of signal names."""
        return list(self._signals.keys())

    def get_signal_color(self, name: str) -> Optional[str]:
        """Get the color of a signal, or None if not found."""
        if name in self._signals:
            return self._signals[name].color
        return None

    def clear(self):
        """Clear all signals and time data."""
        for name in list(self._signals.keys()):
            self.remove_signal(name)
        self._time_data.clear()
        self._color_index = 0

    def auto_scale(self):
        """Auto-scale the plot to fit all visible data."""
        self._plot_widget.enableAutoRange()
        self._plot_widget.autoRange()

    def get_plot_widget(self) -> pg.PlotWidget:
        """Get the underlying pyqtgraph PlotWidget."""
        return self._plot_widget

    def _on_params_changed(self):
        """Handle window or NFFT parameter change — recompute FFT."""
        if self._time_data:
            self.compute_all()

    def _toggle_grid(self, checked: bool):
        """Toggle grid visibility."""
        self._plot_widget.showGrid(x=checked, y=checked, alpha=0.3)

    def _export_image(self):
        """Export plot to image file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export FFT Plot",
            "fft.png",
            "PNG Files (*.png);;All Files (*)"
        )
        if path:
            exporter = pg.exporters.ImageExporter(self._plot_widget.plotItem)
            exporter.export(path)

    def _export_csv(self):
        """Export FFT data to CSV file."""
        if not self._signals:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export FFT Data as CSV",
            "fft_data.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            try:
                with open(path, 'w') as f:
                    # Write header
                    signal_names = list(self._signals.keys())
                    header_parts = ["Frequency (Hz)"]
                    for name in signal_names:
                        header_parts.append(f"{name} Magnitude (dB)")
                    f.write(",".join(header_parts) + "\n")

                    # Get frequency values from first signal
                    first_signal = list(self._signals.values())[0]
                    frequencies = first_signal.frequencies

                    # Write data rows
                    for i, freq in enumerate(frequencies):
                        row = [str(freq)]
                        for name in signal_names:
                            sig = self._signals[name]
                            if i < len(sig.magnitude_db):
                                row.append(str(sig.magnitude_db[i]))
                            else:
                                row.append("")
                        f.write(",".join(row) + "\n")
            except Exception as e:
                print(f"Error exporting CSV: {e}")
