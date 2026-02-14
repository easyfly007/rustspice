"""
Signal list widget for selecting and managing displayed signals.

Provides a list of signals with:
- Visibility checkboxes
- Color indicators
- Color picker
- Remove button
"""

from typing import Optional, Dict, List, Callable

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QPushButton,
    QLabel,
    QColorDialog,
    QMenu,
    QAbstractItemView,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap, QPainter, QBrush


class ColorButton(QPushButton):
    """Button that displays and allows changing a color."""

    color_changed = Signal(str)  # hex color

    def __init__(self, color: str = "#1f77b4", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(20, 20)
        self.clicked.connect(self._pick_color)
        self._update_icon()

    def _update_icon(self):
        """Update the button icon with current color."""
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(self._color))
        self.setIcon(QIcon(pixmap))

    def _pick_color(self):
        """Open color picker dialog."""
        color = QColorDialog.getColor(QColor(self._color), self)
        if color.isValid():
            self._color = color.name()
            self._update_icon()
            self.color_changed.emit(self._color)

    def get_color(self) -> str:
        """Get current color."""
        return self._color

    def set_color(self, color: str):
        """Set current color."""
        self._color = color
        self._update_icon()


class SignalListItem(QWidget):
    """Widget for a single signal in the list."""

    visibility_changed = Signal(str, bool)  # name, visible
    color_changed = Signal(str, str)  # name, color
    remove_requested = Signal(str)  # name

    def __init__(
        self,
        name: str,
        color: str = "#1f77b4",
        visible: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._name = name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # Visibility checkbox
        self._checkbox = QCheckBox()
        self._checkbox.setChecked(visible)
        self._checkbox.stateChanged.connect(self._on_visibility_changed)
        layout.addWidget(self._checkbox)

        # Color button
        self._color_btn = ColorButton(color)
        self._color_btn.color_changed.connect(self._on_color_changed)
        layout.addWidget(self._color_btn)

        # Signal name
        self._label = QLabel(name)
        self._label.setMinimumWidth(100)
        layout.addWidget(self._label, stretch=1)

        # Remove button
        self._remove_btn = QPushButton("×")
        self._remove_btn.setFixedSize(20, 20)
        self._remove_btn.setToolTip("Remove signal")
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self._name))
        layout.addWidget(self._remove_btn)

    def _on_visibility_changed(self, state: int):
        """Handle visibility checkbox change."""
        visible = state == Qt.CheckState.Checked.value
        self.visibility_changed.emit(self._name, visible)

    def _on_color_changed(self, color: str):
        """Handle color change."""
        self.color_changed.emit(self._name, color)

    def get_name(self) -> str:
        """Get signal name."""
        return self._name

    def is_visible(self) -> bool:
        """Check if signal is visible."""
        return self._checkbox.isChecked()

    def set_visible(self, visible: bool):
        """Set signal visibility."""
        self._checkbox.setChecked(visible)

    def get_color(self) -> str:
        """Get signal color."""
        return self._color_btn.get_color()

    def set_color(self, color: str):
        """Set signal color."""
        self._color_btn.set_color(color)


class SignalListWidget(QWidget):
    """
    Widget for managing a list of signals.

    Features:
    - Add/remove signals
    - Toggle visibility
    - Change colors
    - Select all / deselect all

    Signals:
        signal_visibility_changed: (name, visible)
        signal_color_changed: (name, color)
        signal_removed: (name)
    """

    signal_visibility_changed = Signal(str, bool)
    signal_color_changed = Signal(str, str)
    signal_removed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._items: Dict[str, SignalListItem] = {}

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header with buttons
        header = QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 2)

        self._title = QLabel("Signals")
        self._title.setStyleSheet("font-weight: bold;")
        header.addWidget(self._title)

        header.addStretch()

        self._show_all_btn = QPushButton("All")
        self._show_all_btn.setFixedWidth(40)
        self._show_all_btn.setToolTip("Show all signals")
        self._show_all_btn.clicked.connect(self._show_all)
        header.addWidget(self._show_all_btn)

        self._hide_all_btn = QPushButton("None")
        self._hide_all_btn.setFixedWidth(40)
        self._hide_all_btn.setToolTip("Hide all signals")
        self._hide_all_btn.clicked.connect(self._hide_all)
        header.addWidget(self._hide_all_btn)

        layout.addLayout(header)

        # Signal list container
        self._list_layout = QVBoxLayout()
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(1)
        layout.addLayout(self._list_layout)

        # Placeholder when empty
        self._placeholder = QLabel("No signals")
        self._placeholder.setStyleSheet("color: #888; padding: 10px;")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._list_layout.addWidget(self._placeholder)

        layout.addStretch()

    def add_signal(self, name: str, color: str = "#1f77b4", visible: bool = True):
        """Add a signal to the list."""
        if name in self._items:
            return

        # Hide placeholder
        self._placeholder.hide()

        # Create item widget
        item = SignalListItem(name, color, visible)
        item.visibility_changed.connect(self._on_visibility_changed)
        item.color_changed.connect(self._on_color_changed)
        item.remove_requested.connect(self._on_remove_requested)

        self._items[name] = item
        self._list_layout.insertWidget(self._list_layout.count() - 1, item)

    def remove_signal(self, name: str):
        """Remove a signal from the list."""
        if name in self._items:
            item = self._items[name]
            self._list_layout.removeWidget(item)
            item.deleteLater()
            del self._items[name]

            # Show placeholder if empty
            if not self._items:
                self._placeholder.show()

    def clear(self):
        """Clear all signals."""
        for name in list(self._items.keys()):
            self.remove_signal(name)

    def get_signal_names(self) -> List[str]:
        """Get list of signal names."""
        return list(self._items.keys())

    def is_signal_visible(self, name: str) -> bool:
        """Check if a signal is visible."""
        if name in self._items:
            return self._items[name].is_visible()
        return False

    def set_signal_visible(self, name: str, visible: bool):
        """Set signal visibility."""
        if name in self._items:
            self._items[name].set_visible(visible)

    def get_signal_color(self, name: str) -> Optional[str]:
        """Get signal color."""
        if name in self._items:
            return self._items[name].get_color()
        return None

    def set_signal_color(self, name: str, color: str):
        """Set signal color."""
        if name in self._items:
            self._items[name].set_color(color)

    def _show_all(self):
        """Show all signals."""
        for name, item in self._items.items():
            item.set_visible(True)
            self.signal_visibility_changed.emit(name, True)

    def _hide_all(self):
        """Hide all signals."""
        for name, item in self._items.items():
            item.set_visible(False)
            self.signal_visibility_changed.emit(name, False)

    def _on_visibility_changed(self, name: str, visible: bool):
        """Forward visibility change signal."""
        self.signal_visibility_changed.emit(name, visible)

    def _on_color_changed(self, name: str, color: str):
        """Forward color change signal."""
        self.signal_color_changed.emit(name, color)

    def _on_remove_requested(self, name: str):
        """Handle remove request."""
        self.remove_signal(name)
        self.signal_removed.emit(name)
