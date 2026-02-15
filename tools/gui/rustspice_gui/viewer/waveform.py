"""
Waveform viewer widget for displaying time-domain simulation results.

Provides an interactive plot with:
- Multi-signal display
- Zoom and pan
- Auto-scaling
- Grid display
- Legend
- Export to PNG/SVG
- Context menu (right-click)
- Double-click to add cursor
"""

from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QMenu,
    QFileDialog,
    QColorDialog,
    QInputDialog,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QAction

import pyqtgraph as pg


# Default colors for signals (color-blind friendly)
DEFAULT_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]


@dataclass
class SignalData:
    """Data for a single signal."""
    name: str
    x_data: np.ndarray
    y_data: np.ndarray
    color: str
    visible: bool = True
    plot_item: Optional[pg.PlotDataItem] = None


class WaveformViewer(QWidget):
    """
    Interactive waveform viewer using pyqtgraph.

    Features:
    - Multi-signal plotting
    - Mouse wheel zoom
    - Click and drag pan
    - Auto-scale
    - Grid
    - Legend
    - Export to image
    - Context menu (right-click) for reset zoom, export
    - Double-click to add cursor

    Signals:
        signal_added: Emitted when a signal is added (name)
        signal_removed: Emitted when a signal is removed (name)
        cursor_moved: Emitted when cursor moves (x, y)
        cursor_add_requested: Emitted when user double-clicks to add cursor (x position)
    """

    signal_added = Signal(str)
    signal_removed = Signal(str)
    cursor_moved = Signal(float, float)
    cursor_add_requested = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._signals: Dict[str, SignalData] = {}
        self._color_index = 0
        self._x_label = "Time"
        self._x_unit = "s"
        self._y_label = "Voltage"
        self._y_unit = "V"
        self._last_mouse_pos: Optional[Tuple[float, float]] = None

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)

        self._auto_scale_btn = QPushButton("Auto Scale")
        self._auto_scale_btn.clicked.connect(self.auto_scale)
        toolbar.addWidget(self._auto_scale_btn)

        self._reset_btn = QPushButton("Reset View")
        self._reset_btn.clicked.connect(self.reset_view)
        toolbar.addWidget(self._reset_btn)

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
        self._plot_widget.setLabel('bottom', self._x_label, self._x_unit)
        self._plot_widget.setLabel('left', self._y_label, self._y_unit)

        # Enable mouse interaction
        self._plot_widget.setMouseEnabled(x=True, y=True)
        self._plot_widget.enableAutoRange()

        # Add legend
        self._legend = self._plot_widget.addLegend()

        # Connect mouse move for cursor tracking
        self._plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # Connect double-click for cursor placement
        self._plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # Setup context menu
        self._setup_context_menu()

        layout.addWidget(self._plot_widget)

    def add_signal(
        self,
        name: str,
        x_data: List[float],
        y_data: List[float],
        color: Optional[str] = None,
    ):
        """
        Add a signal to the plot.

        Args:
            name: Signal name (e.g., "V(out)")
            x_data: X-axis data (time values)
            y_data: Y-axis data (signal values)
            color: Optional color (hex string)
        """
        # Remove existing signal with same name
        if name in self._signals:
            self.remove_signal(name)

        # Get color
        if color is None:
            color = DEFAULT_COLORS[self._color_index % len(DEFAULT_COLORS)]
            self._color_index += 1

        # Convert to numpy arrays
        x_arr = np.array(x_data)
        y_arr = np.array(y_data)

        # Create plot item
        pen = pg.mkPen(color=color, width=2)
        plot_item = self._plot_widget.plot(
            x_arr, y_arr,
            pen=pen,
            name=name,
        )

        # Store signal data
        self._signals[name] = SignalData(
            name=name,
            x_data=x_arr,
            y_data=y_arr,
            color=color,
            visible=True,
            plot_item=plot_item,
        )

        self.signal_added.emit(name)

    def remove_signal(self, name: str):
        """Remove a signal from the plot."""
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
        """Set signal color."""
        if name in self._signals:
            signal = self._signals[name]
            signal.color = color
            if signal.plot_item:
                pen = pg.mkPen(color=color, width=2)
                signal.plot_item.setPen(pen)

    def get_signal_names(self) -> List[str]:
        """Get list of signal names."""
        return list(self._signals.keys())

    def get_signal_color(self, name: str) -> Optional[str]:
        """Get the color of a signal, or None if not found."""
        if name in self._signals:
            return self._signals[name].color
        return None

    def get_signal_data(self, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get signal data as (x, y) tuple."""
        if name in self._signals:
            signal = self._signals[name]
            return signal.x_data, signal.y_data
        return None

    def clear(self):
        """Clear all signals."""
        for name in list(self._signals.keys()):
            self.remove_signal(name)
        self._color_index = 0

    def auto_scale(self):
        """Auto-scale to fit all visible data."""
        self._plot_widget.enableAutoRange()
        self._plot_widget.autoRange()

    def reset_view(self):
        """Reset view to show all data."""
        self._plot_widget.getViewBox().autoRange()

    def set_labels(self, x_label: str, x_unit: str, y_label: str, y_unit: str):
        """Set axis labels."""
        self._x_label = x_label
        self._x_unit = x_unit
        self._y_label = y_label
        self._y_unit = y_unit
        self._plot_widget.setLabel('bottom', x_label, x_unit)
        self._plot_widget.setLabel('left', y_label, y_unit)

    def set_title(self, title: str):
        """Set plot title."""
        self._plot_widget.setTitle(title)

    def _toggle_grid(self, checked: bool):
        """Toggle grid visibility."""
        self._plot_widget.showGrid(x=checked, y=checked, alpha=0.3)

    def _export_image(self):
        """Export plot to image file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot",
            "waveform.png",
            "PNG Files (*.png);;SVG Files (*.svg);;All Files (*)"
        )
        if path:
            if path.lower().endswith('.svg'):
                exporter = pg.exporters.SVGExporter(self._plot_widget.plotItem)
            else:
                exporter = pg.exporters.ImageExporter(self._plot_widget.plotItem)
            exporter.export(path)

    def _on_mouse_moved(self, pos):
        """Handle mouse movement for cursor tracking."""
        if self._plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self._plot_widget.getViewBox().mapSceneToView(pos)
            self._last_mouse_pos = (mouse_point.x(), mouse_point.y())
            self.cursor_moved.emit(mouse_point.x(), mouse_point.y())

    def _on_mouse_clicked(self, event):
        """Handle mouse click events."""
        # Check for double-click
        if event.double():
            pos = event.scenePos()
            if self._plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self._plot_widget.getViewBox().mapSceneToView(pos)
                self.cursor_add_requested.emit(mouse_point.x())

    def _setup_context_menu(self):
        """Set up the right-click context menu."""
        self._plot_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._plot_widget.customContextMenuRequested.connect(self._show_context_menu)

    def _show_context_menu(self, pos):
        """Show context menu at the given position."""
        menu = QMenu(self)

        # Reset Zoom action
        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.triggered.connect(self.reset_view)
        menu.addAction(reset_zoom_action)

        # Auto Scale action
        auto_scale_action = QAction("Auto Scale", self)
        auto_scale_action.triggered.connect(self.auto_scale)
        menu.addAction(auto_scale_action)

        menu.addSeparator()

        # Add Cursor action
        add_cursor_action = QAction("Add Cursor Here", self)
        add_cursor_action.triggered.connect(self._add_cursor_at_last_pos)
        menu.addAction(add_cursor_action)

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
        export_png_action.triggered.connect(lambda: self._export_image_format("png"))
        export_menu.addAction(export_png_action)

        export_svg_action = QAction("Export as SVG...", self)
        export_svg_action.triggered.connect(lambda: self._export_image_format("svg"))
        export_menu.addAction(export_svg_action)

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

    def _add_cursor_at_last_pos(self):
        """Add cursor at the last known mouse position."""
        if self._last_mouse_pos:
            self.cursor_add_requested.emit(self._last_mouse_pos[0])

    def _export_image_format(self, fmt: str):
        """Export plot to specified image format."""
        if fmt == "svg":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Plot as SVG",
                "waveform.svg",
                "SVG Files (*.svg);;All Files (*)"
            )
            if path:
                exporter = pg.exporters.SVGExporter(self._plot_widget.plotItem)
                exporter.export(path)
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Plot as PNG",
                "waveform.png",
                "PNG Files (*.png);;All Files (*)"
            )
            if path:
                exporter = pg.exporters.ImageExporter(self._plot_widget.plotItem)
                exporter.export(path)

    def _export_csv(self):
        """Export signal data to CSV file."""
        if not self._signals:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data as CSV",
            "waveform_data.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            try:
                with open(path, 'w') as f:
                    # Write header
                    signal_names = list(self._signals.keys())
                    header = [self._x_label] + signal_names
                    f.write(",".join(header) + "\n")

                    # Get all x values (use first signal's x data as reference)
                    first_signal = list(self._signals.values())[0]
                    x_data = first_signal.x_data

                    # Write data rows
                    for i, x in enumerate(x_data):
                        row = [str(x)]
                        for name in signal_names:
                            sig = self._signals[name]
                            if i < len(sig.y_data):
                                row.append(str(sig.y_data[i]))
                            else:
                                row.append("")
                        f.write(",".join(row) + "\n")
            except Exception as e:
                print(f"Error exporting CSV: {e}")

    def get_plot_widget(self) -> pg.PlotWidget:
        """Get the underlying pyqtgraph PlotWidget."""
        return self._plot_widget
