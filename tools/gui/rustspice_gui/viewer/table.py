"""
Enhanced results table widget for displaying simulation results.

Provides a feature-rich table with:
- Sortable columns
- Copy to clipboard
- Export to CSV
- Search/filter
- Engineering notation formatting
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
    QLineEdit,
    QLabel,
    QMenu,
    QFileDialog,
    QApplication,
    QAbstractItemView,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence


class ResultType(Enum):
    """Type of result being displayed."""
    OP = "op"
    DC = "dc"
    TRAN = "tran"
    AC = "ac"


@dataclass
class ResultEntry:
    """A single result entry."""
    name: str
    value: float
    unit: str = ""
    formatted: str = ""


class ResultsTable(QWidget):
    """
    Enhanced results table widget.

    Features:
    - Sortable columns
    - Copy selected cells to clipboard
    - Export to CSV
    - Search/filter functionality
    - Engineering notation formatting
    - Context menu

    Signals:
        selection_changed: Emitted when selection changes
    """

    selection_changed = Signal(list)  # List of selected variable names

    # Engineering prefixes
    PREFIXES = [
        (1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"),
        (1, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f"),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._entries: List[ResultEntry] = []
        self._result_type: Optional[ResultType] = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 4, 4, 4)

        # Search box
        self._search_label = QLabel("Filter:")
        toolbar.addWidget(self._search_label)

        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Type to filter...")
        self._search_box.setClearButtonEnabled(True)
        self._search_box.setMaximumWidth(200)
        toolbar.addWidget(self._search_box)

        toolbar.addStretch()

        # Copy button
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setToolTip("Copy selected rows to clipboard (Ctrl+C)")
        toolbar.addWidget(self._copy_btn)

        # Export button
        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setToolTip("Export table to CSV file")
        toolbar.addWidget(self._export_btn)

        layout.addLayout(toolbar)

        # Table widget
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Variable", "Value", "Unit"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        # Style
        self._table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                font-family: Consolas, monospace;
            }
            QTableWidget::item:selected {
                background-color: #cce5ff;
                color: black;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #ddd;
                font-weight: bold;
            }
        """)

        layout.addWidget(self._table)

        # Status bar
        self._status = QLabel("No results")
        self._status.setStyleSheet("color: gray; padding: 2px;")
        layout.addWidget(self._status)

    def _connect_signals(self):
        """Connect internal signals."""
        self._search_box.textChanged.connect(self._filter_rows)
        self._copy_btn.clicked.connect(self.copy_selected)
        self._export_btn.clicked.connect(self.export_csv)
        self._table.customContextMenuRequested.connect(self._show_context_menu)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

    def _format_value(self, value: float, unit: str = "") -> tuple:
        """
        Format a value with engineering notation.

        Returns:
            Tuple of (formatted_value, unit_with_prefix)
        """
        if value == 0:
            return "0", unit

        abs_value = abs(value)
        for scale, prefix in self.PREFIXES:
            if abs_value >= scale * 0.9999:  # Small tolerance for floating point
                scaled = value / scale
                if abs(scaled) >= 100:
                    formatted = f"{scaled:.1f}"
                elif abs(scaled) >= 10:
                    formatted = f"{scaled:.2f}"
                else:
                    formatted = f"{scaled:.3f}"
                return formatted, f"{prefix}{unit}"

        # Very small values
        return f"{value:.3e}", unit

    def set_op_results(self, nodes: List[str], solution: List[float]):
        """
        Set operating point results.

        Args:
            nodes: List of node names
            solution: List of node voltages
        """
        self._result_type = ResultType.OP
        self._entries = []

        for node, value in zip(nodes, solution):
            if node != "0":  # Skip ground
                formatted, unit = self._format_value(value, "V")
                self._entries.append(ResultEntry(
                    name=f"V({node})",
                    value=value,
                    unit="V",
                    formatted=formatted + " " + unit
                ))

        self._populate_table()
        self._update_status()

    def set_dc_results(self, sweep_var: str, sweep_values: List[float],
                       nodes: List[str], solutions: List[List[float]]):
        """
        Set DC sweep results.

        Args:
            sweep_var: Name of sweep variable
            sweep_values: Sweep values
            nodes: List of node names
            solutions: List of solutions at each sweep point
        """
        self._result_type = ResultType.DC
        self._entries = []

        # Summary entries
        self._entries.append(ResultEntry(
            name="Sweep Variable",
            value=0,
            unit="",
            formatted=sweep_var
        ))
        self._entries.append(ResultEntry(
            name="Points",
            value=len(sweep_values),
            unit="",
            formatted=str(len(sweep_values))
        ))
        if sweep_values:
            self._entries.append(ResultEntry(
                name="Start",
                value=sweep_values[0],
                unit="V",
                formatted=f"{sweep_values[0]:.4g} V"
            ))
            self._entries.append(ResultEntry(
                name="Stop",
                value=sweep_values[-1],
                unit="V",
                formatted=f"{sweep_values[-1]:.4g} V"
            ))

        self._populate_table()
        self._update_status()

    def set_tran_results(self, times: List[float], nodes: List[str],
                         solutions: List[List[float]]):
        """
        Set transient results.

        Args:
            times: Time points
            nodes: List of node names
            solutions: List of solutions at each time point
        """
        self._result_type = ResultType.TRAN
        self._entries = []

        # Summary entries
        self._entries.append(ResultEntry(
            name="Time Points",
            value=len(times),
            unit="",
            formatted=str(len(times))
        ))
        if times:
            start_fmt, start_unit = self._format_value(times[0], "s")
            stop_fmt, stop_unit = self._format_value(times[-1], "s")
            self._entries.append(ResultEntry(
                name="Start Time",
                value=times[0],
                unit="s",
                formatted=f"{start_fmt} {start_unit}"
            ))
            self._entries.append(ResultEntry(
                name="Stop Time",
                value=times[-1],
                unit="s",
                formatted=f"{stop_fmt} {stop_unit}"
            ))

        self._populate_table()
        self._update_status()

    def set_ac_results(self, frequencies: List[float], nodes: List[str],
                       solutions: List[List[complex]]):
        """
        Set AC analysis results.

        Args:
            frequencies: Frequency points
            nodes: List of node names
            solutions: Complex solutions at each frequency
        """
        self._result_type = ResultType.AC
        self._entries = []

        # Summary entries
        self._entries.append(ResultEntry(
            name="Frequency Points",
            value=len(frequencies),
            unit="",
            formatted=str(len(frequencies))
        ))
        if frequencies:
            start_fmt, start_unit = self._format_value(frequencies[0], "Hz")
            stop_fmt, stop_unit = self._format_value(frequencies[-1], "Hz")
            self._entries.append(ResultEntry(
                name="Start Frequency",
                value=frequencies[0],
                unit="Hz",
                formatted=f"{start_fmt} {start_unit}"
            ))
            self._entries.append(ResultEntry(
                name="Stop Frequency",
                value=frequencies[-1],
                unit="Hz",
                formatted=f"{stop_fmt} {stop_unit}"
            ))

        self._populate_table()
        self._update_status()

    def _populate_table(self):
        """Populate the table with current entries."""
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._entries))

        for row, entry in enumerate(self._entries):
            # Variable name
            name_item = QTableWidgetItem(entry.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            # Value
            value_item = QTableWidgetItem(entry.formatted.split()[0] if entry.formatted else "")
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_item.setData(Qt.ItemDataRole.UserRole, entry.value)  # Store numeric value for sorting
            self._table.setItem(row, 1, value_item)

            # Unit
            parts = entry.formatted.split()
            unit_str = parts[1] if len(parts) > 1 else entry.unit
            unit_item = QTableWidgetItem(unit_str)
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 2, unit_item)

        self._table.setSortingEnabled(True)

    def _filter_rows(self, text: str):
        """Filter rows based on search text."""
        search = text.lower()
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 0)
            if name_item:
                visible = search in name_item.text().lower()
                self._table.setRowHidden(row, not visible)

    def _update_status(self):
        """Update status bar."""
        visible = sum(1 for row in range(self._table.rowCount())
                     if not self._table.isRowHidden(row))
        total = self._table.rowCount()
        if visible == total:
            self._status.setText(f"{total} entries")
        else:
            self._status.setText(f"Showing {visible} of {total} entries")

    def _on_selection_changed(self):
        """Handle selection change."""
        selected = []
        for item in self._table.selectedItems():
            if item.column() == 0:
                selected.append(item.text())
        self.selection_changed.emit(selected)

    def _show_context_menu(self, pos):
        """Show context menu."""
        menu = QMenu(self)

        copy_action = QAction("Copy", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self.copy_selected)
        menu.addAction(copy_action)

        copy_all_action = QAction("Copy All", self)
        copy_all_action.triggered.connect(self.copy_all)
        menu.addAction(copy_all_action)

        menu.addSeparator()

        export_action = QAction("Export to CSV...", self)
        export_action.triggered.connect(self.export_csv)
        menu.addAction(export_action)

        menu.exec(self._table.mapToGlobal(pos))

    def copy_selected(self):
        """Copy selected rows to clipboard."""
        rows = set()
        for item in self._table.selectedItems():
            rows.add(item.row())

        if not rows:
            return

        lines = []
        for row in sorted(rows):
            cols = []
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                cols.append(item.text() if item else "")
            lines.append("\t".join(cols))

        text = "\n".join(lines)
        QApplication.clipboard().setText(text)

    def copy_all(self):
        """Copy all visible rows to clipboard."""
        lines = []

        # Header
        headers = []
        for col in range(self._table.columnCount()):
            headers.append(self._table.horizontalHeaderItem(col).text())
        lines.append("\t".join(headers))

        # Data
        for row in range(self._table.rowCount()):
            if not self._table.isRowHidden(row):
                cols = []
                for col in range(self._table.columnCount()):
                    item = self._table.item(row, col)
                    cols.append(item.text() if item else "")
                lines.append("\t".join(cols))

        text = "\n".join(lines)
        QApplication.clipboard().setText(text)

    def export_csv(self):
        """Export table to CSV file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results",
            "results.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        with open(path, 'w', encoding='utf-8') as f:
            # Header
            headers = []
            for col in range(self._table.columnCount()):
                headers.append(self._table.horizontalHeaderItem(col).text())
            f.write(",".join(headers) + "\n")

            # Data
            for row in range(self._table.rowCount()):
                if not self._table.isRowHidden(row):
                    cols = []
                    for col in range(self._table.columnCount()):
                        item = self._table.item(row, col)
                        text = item.text() if item else ""
                        # Escape commas and quotes
                        if "," in text or '"' in text:
                            text = '"' + text.replace('"', '""') + '"'
                        cols.append(text)
                    f.write(",".join(cols) + "\n")

    def clear(self):
        """Clear all results."""
        self._table.setRowCount(0)
        self._entries = []
        self._result_type = None
        self._status.setText("No results")
        self._search_box.clear()

    def get_entry_count(self) -> int:
        """Get number of entries."""
        return len(self._entries)
