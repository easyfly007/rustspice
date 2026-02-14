"""
Measurement cursors for waveform analysis.

Provides:
- Vertical cursors for time measurement
- Delta display between cursors
- Cursor position readout
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
)
from PySide6.QtCore import Qt, Signal

import pyqtgraph as pg


@dataclass
class Cursor:
    """Data for a measurement cursor."""
    name: str
    x_position: float
    color: str
    line: Optional[pg.InfiniteLine] = None
    label: Optional[pg.TextItem] = None


class CursorManager:
    """
    Manages measurement cursors on a plot.

    Features:
    - Add/remove cursors
    - Draggable cursor lines
    - Position readout
    - Delta measurement between cursors
    """

    def __init__(self, plot_widget: pg.PlotWidget):
        self._plot = plot_widget
        self._cursors: Dict[str, Cursor] = {}
        self._cursor_colors = ["#FF0000", "#00FF00"]  # Red, Green

    def add_cursor(
        self,
        name: str,
        x_position: float = 0.0,
        color: Optional[str] = None,
    ) -> Cursor:
        """
        Add a cursor to the plot.

        Args:
            name: Cursor name (e.g., "C1", "C2")
            x_position: Initial X position
            color: Cursor color (hex string)

        Returns:
            Cursor object
        """
        if name in self._cursors:
            self.remove_cursor(name)

        if color is None:
            idx = len(self._cursors) % len(self._cursor_colors)
            color = self._cursor_colors[idx]

        # Create vertical line
        line = pg.InfiniteLine(
            pos=x_position,
            angle=90,
            movable=True,
            pen=pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine),
            label=name,
            labelOpts={
                'position': 0.95,
                'color': color,
                'fill': (255, 255, 255, 150),
            }
        )

        self._plot.addItem(line)

        cursor = Cursor(
            name=name,
            x_position=x_position,
            color=color,
            line=line,
        )

        self._cursors[name] = cursor

        # Connect position change signal
        line.sigPositionChanged.connect(
            lambda: self._on_cursor_moved(name)
        )

        return cursor

    def remove_cursor(self, name: str):
        """Remove a cursor from the plot."""
        if name in self._cursors:
            cursor = self._cursors[name]
            if cursor.line:
                self._plot.removeItem(cursor.line)
            if cursor.label:
                self._plot.removeItem(cursor.label)
            del self._cursors[name]

    def clear(self):
        """Remove all cursors."""
        for name in list(self._cursors.keys()):
            self.remove_cursor(name)

    def get_cursor_position(self, name: str) -> Optional[float]:
        """Get cursor X position."""
        if name in self._cursors:
            cursor = self._cursors[name]
            if cursor.line:
                return cursor.line.value()
        return None

    def set_cursor_position(self, name: str, x_position: float):
        """Set cursor X position."""
        if name in self._cursors:
            cursor = self._cursors[name]
            cursor.x_position = x_position
            if cursor.line:
                cursor.line.setValue(x_position)

    def get_cursor_names(self) -> List[str]:
        """Get list of cursor names."""
        return list(self._cursors.keys())

    def get_delta(self) -> Optional[Tuple[float, float, float]]:
        """
        Get delta between first two cursors.

        Returns:
            Tuple of (x1, x2, delta) or None if less than 2 cursors
        """
        names = list(self._cursors.keys())
        if len(names) >= 2:
            x1 = self.get_cursor_position(names[0])
            x2 = self.get_cursor_position(names[1])
            if x1 is not None and x2 is not None:
                return (x1, x2, abs(x2 - x1))
        return None

    def _on_cursor_moved(self, name: str):
        """Handle cursor position change."""
        cursor = self._cursors.get(name)
        if cursor and cursor.line:
            cursor.x_position = cursor.line.value()


class CursorReadout(QWidget):
    """
    Widget for displaying cursor positions and delta.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Cursor 1
        c1_layout = QHBoxLayout()
        c1_layout.addWidget(QLabel("C1:"))
        self._c1_label = QLabel("---")
        self._c1_label.setStyleSheet("font-family: monospace;")
        c1_layout.addWidget(self._c1_label, stretch=1)
        layout.addLayout(c1_layout)

        # Cursor 2
        c2_layout = QHBoxLayout()
        c2_layout.addWidget(QLabel("C2:"))
        self._c2_label = QLabel("---")
        self._c2_label.setStyleSheet("font-family: monospace;")
        c2_layout.addWidget(self._c2_label, stretch=1)
        layout.addLayout(c2_layout)

        # Delta
        delta_layout = QHBoxLayout()
        delta_layout.addWidget(QLabel("Δ:"))
        self._delta_label = QLabel("---")
        self._delta_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        delta_layout.addWidget(self._delta_label, stretch=1)
        layout.addLayout(delta_layout)

        # Frequency (1/Δ)
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("1/Δ:"))
        self._freq_label = QLabel("---")
        self._freq_label.setStyleSheet("font-family: monospace;")
        freq_layout.addWidget(self._freq_label, stretch=1)
        layout.addLayout(freq_layout)

    def update_cursors(
        self,
        c1_pos: Optional[float],
        c2_pos: Optional[float],
        x_unit: str = "s",
    ):
        """Update cursor readout display."""
        if c1_pos is not None:
            self._c1_label.setText(self._format_value(c1_pos, x_unit))
        else:
            self._c1_label.setText("---")

        if c2_pos is not None:
            self._c2_label.setText(self._format_value(c2_pos, x_unit))
        else:
            self._c2_label.setText("---")

        if c1_pos is not None and c2_pos is not None:
            delta = abs(c2_pos - c1_pos)
            self._delta_label.setText(self._format_value(delta, x_unit))

            if delta > 0:
                freq = 1.0 / delta
                self._freq_label.setText(self._format_value(freq, "Hz"))
            else:
                self._freq_label.setText("---")
        else:
            self._delta_label.setText("---")
            self._freq_label.setText("---")

    def _format_value(self, value: float, unit: str) -> str:
        """Format a value with engineering notation."""
        if abs(value) == 0:
            return f"0 {unit}"

        # Engineering prefixes
        prefixes = [
            (1e15, "P"),
            (1e12, "T"),
            (1e9, "G"),
            (1e6, "M"),
            (1e3, "k"),
            (1, ""),
            (1e-3, "m"),
            (1e-6, "µ"),
            (1e-9, "n"),
            (1e-12, "p"),
            (1e-15, "f"),
        ]

        abs_value = abs(value)
        for scale, prefix in prefixes:
            if abs_value >= scale:
                return f"{value / scale:.4g} {prefix}{unit}"

        return f"{value:.4g} {unit}"


class CursorControlPanel(QWidget):
    """
    Panel for controlling cursors.
    """

    cursor_added = Signal(str)  # cursor name
    cursor_removed = Signal(str)  # cursor name

    def __init__(self, cursor_manager: CursorManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._manager = cursor_manager
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Cursor group
        group = QGroupBox("Cursors")
        group_layout = QVBoxLayout(group)

        # Buttons
        btn_layout = QHBoxLayout()

        self._add_c1_btn = QPushButton("Add C1")
        self._add_c1_btn.clicked.connect(lambda: self._add_cursor("C1"))
        btn_layout.addWidget(self._add_c1_btn)

        self._add_c2_btn = QPushButton("Add C2")
        self._add_c2_btn.clicked.connect(lambda: self._add_cursor("C2"))
        btn_layout.addWidget(self._add_c2_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_cursors)
        btn_layout.addWidget(self._clear_btn)

        group_layout.addLayout(btn_layout)

        # Readout
        self._readout = CursorReadout()
        group_layout.addWidget(self._readout)

        layout.addWidget(group)

    def _add_cursor(self, name: str):
        """Add a cursor."""
        # Get current view range to place cursor
        view_range = self._manager._plot.viewRange()
        x_min, x_max = view_range[0]
        x_pos = (x_min + x_max) / 2

        self._manager.add_cursor(name, x_pos)
        self._update_readout()
        self.cursor_added.emit(name)

    def _clear_cursors(self):
        """Clear all cursors."""
        for name in self._manager.get_cursor_names():
            self.cursor_removed.emit(name)
        self._manager.clear()
        self._update_readout()

    def _update_readout(self):
        """Update the cursor readout."""
        c1_pos = self._manager.get_cursor_position("C1")
        c2_pos = self._manager.get_cursor_position("C2")
        self._readout.update_cursors(c1_pos, c2_pos)

    def refresh(self):
        """Refresh the readout (call this when cursors move)."""
        self._update_readout()
