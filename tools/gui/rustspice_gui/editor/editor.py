"""
Main netlist editor widget with line numbers and syntax highlighting.

Provides a full-featured code editor for SPICE netlists with:
- Line number display
- Syntax highlighting
- Auto-completion
- Current line highlighting
- Tab/indent handling
"""

from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QPlainTextEdit,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QToolTip,
)
from PySide6.QtGui import (
    QColor,
    QPainter,
    QTextFormat,
    QFont,
    QFontMetrics,
    QKeyEvent,
    QTextCursor,
    QPaintEvent,
    QResizeEvent,
    QBrush,
    QPen,
)
from PySide6.QtCore import Qt, QRect, QSize, QEvent, QPoint, Signal, Slot

from .highlighter import SpiceHighlighter
from .completer import SpiceCompleter, get_completion_prefix


class LineNumberArea(QWidget):
    """
    Widget for displaying line numbers alongside the editor.

    This widget is positioned to the left of the text editor and
    displays line numbers that scroll with the text.
    """

    def __init__(self, editor: "NetlistEditor"):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QSize:
        return QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event: QPaintEvent):
        self._editor.line_number_area_paint_event(event)


class NetlistEditor(QPlainTextEdit):
    """
    Full-featured netlist editor with syntax highlighting and line numbers.

    Features:
    - Line number display
    - SPICE syntax highlighting
    - Auto-completion
    - Current line highlighting
    - Smart indentation
    - Bracket matching

    Signals:
        cursor_position_changed: Emitted when cursor moves (line, column)
    """

    cursor_position_changed = Signal(int, int)  # line, column

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Set up editor appearance
        self._setup_font()
        self._setup_colors()

        # Line number area
        self._line_number_area = LineNumberArea(self)

        # Syntax highlighter
        self._highlighter = SpiceHighlighter(self.document())

        # Auto-completer
        self._completer = SpiceCompleter(self)
        self._completer.setWidget(self)
        self._completer.activated.connect(self._insert_completion)

        # Error markers: {line_number: error_message}
        self._error_markers: dict[int, str] = {}

        # Connect signals
        self.blockCountChanged.connect(self._update_line_number_area_width)
        self.updateRequest.connect(self._update_line_number_area)
        self.cursorPositionChanged.connect(self._update_extra_selections)
        self.cursorPositionChanged.connect(self._emit_cursor_position)
        self.textChanged.connect(self._on_text_changed)
        self.textChanged.connect(self.clear_error_markers)

        # Initialize
        self._update_line_number_area_width(0)
        self._update_extra_selections()

        # Set placeholder
        self.setPlaceholderText("Enter SPICE netlist here...")

        # Tab settings
        self.setTabStopDistance(
            QFontMetrics(self.font()).horizontalAdvance(" ") * 4
        )

    def _setup_font(self):
        """Set up the editor font."""
        # Try to use a good monospace font
        font = QFont("Consolas", 11)
        if not font.exactMatch():
            font = QFont("Courier New", 11)
        if not font.exactMatch():
            font = QFont("Monospace", 11)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)

    def _setup_colors(self):
        """Set up editor colors."""
        # Background and text colors
        self.setStyleSheet("""
            QPlainTextEdit {
                background-color: #FAFAFA;
                color: #333333;
                border: 1px solid #DDD;
                border-radius: 2px;
                selection-background-color: #ADD8E6;
            }
        """)

        # Current line highlight color
        self._current_line_color = QColor("#FFFACD")  # Light yellow

    # =========================================================================
    # Line Number Area
    # =========================================================================

    def line_number_area_width(self) -> int:
        """Calculate the width needed for the line number area."""
        digits = 1
        max_num = max(1, self.blockCount())
        while max_num >= 10:
            max_num //= 10
            digits += 1

        # Minimum 3 digits, plus padding
        digits = max(3, digits)
        space = 10 + self.fontMetrics().horizontalAdvance("9") * digits
        return space

    def _update_line_number_area_width(self, _: int):
        """Update the viewport margins when line count changes."""
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_number_area(self, rect: QRect, dy: int):
        """Update the line number area when scrolling."""
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(
                0, rect.y(),
                self._line_number_area.width(), rect.height()
            )

        if rect.contains(self.viewport().rect()):
            self._update_line_number_area_width(0)

    def resizeEvent(self, event: QResizeEvent):
        """Handle resize events to update line number area."""
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._line_number_area.setGeometry(
            QRect(cr.left(), cr.top(),
                  self.line_number_area_width(), cr.height())
        )

    def line_number_area_paint_event(self, event: QPaintEvent):
        """Paint the line numbers and error indicators."""
        painter = QPainter(self._line_number_area)
        painter.fillRect(event.rect(), QColor("#F0F0F0"))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(
            self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                line_num = block_number + 1

                # Draw error indicator (red circle) for error lines
                if line_num in self._error_markers:
                    diameter = 8
                    x = 2
                    y = top + (self.fontMetrics().height() - diameter) // 2
                    painter.save()
                    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                    painter.setBrush(QBrush(QColor("#E03030")))
                    painter.setPen(QPen(Qt.PenStyle.NoPen))
                    painter.drawEllipse(x, y, diameter, diameter)
                    painter.restore()

                # Draw line number
                painter.setPen(QColor("#999999"))
                painter.drawText(
                    0, top,
                    self._line_number_area.width() - 5,
                    self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight,
                    str(line_num)
                )

            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

    # =========================================================================
    # Current Line & Error Highlighting
    # =========================================================================

    def _update_extra_selections(self):
        """Update all extra selections (current line highlight + error markers)."""
        extra_selections = []

        # Current line highlight
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            selection.format.setBackground(self._current_line_color)
            selection.format.setProperty(
                QTextFormat.Property.FullWidthSelection, True
            )
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extra_selections.append(selection)

        # Error line highlights
        error_color = QColor("#FFDEDE")
        for line_num in self._error_markers:
            block = self.document().findBlockByLineNumber(line_num - 1)
            if block.isValid():
                selection = QTextEdit.ExtraSelection()
                selection.format.setBackground(error_color)
                selection.format.setProperty(
                    QTextFormat.Property.FullWidthSelection, True
                )
                selection.cursor = QTextCursor(block)
                selection.cursor.clearSelection()
                extra_selections.append(selection)

        self.setExtraSelections(extra_selections)

    def set_error_markers(self, markers: dict[int, str]):
        """Set error markers on specific lines.

        Args:
            markers: Mapping of {line_number: error_message}.
        """
        self._error_markers = dict(markers)
        self._update_extra_selections()
        self._line_number_area.update()

    def clear_error_markers(self):
        """Clear all error markers."""
        if self._error_markers:
            self._error_markers.clear()
            self._update_extra_selections()
            self._line_number_area.update()

    # =========================================================================
    # Cursor Position
    # =========================================================================

    def _emit_cursor_position(self):
        """Emit cursor position signal."""
        cursor = self.textCursor()
        line = cursor.blockNumber() + 1
        column = cursor.columnNumber() + 1
        self.cursor_position_changed.emit(line, column)

    def get_cursor_position(self) -> tuple:
        """Get current cursor position as (line, column)."""
        cursor = self.textCursor()
        return cursor.blockNumber() + 1, cursor.columnNumber() + 1

    def go_to_line(self, line: int):
        """Move cursor to specified line."""
        block = self.document().findBlockByLineNumber(line - 1)
        cursor = QTextCursor(block)
        self.setTextCursor(cursor)
        self.centerCursor()

    def event(self, event: QEvent) -> bool:
        """Show tooltip for error markers on hover."""
        if event.type() == QEvent.Type.ToolTip:
            pos = event.pos()
            cursor = self.cursorForPosition(pos)
            line_num = cursor.blockNumber() + 1
            message = self._error_markers.get(line_num)
            if message:
                QToolTip.showText(event.globalPos(), f"Error: {message}")
            else:
                QToolTip.hideText()
            return True
        return super().event(event)

    # =========================================================================
    # Auto-completion
    # =========================================================================

    def _on_text_changed(self):
        """Handle text changes for auto-completion updates."""
        # Update node names for completion
        self._completer.update_node_names(self.toPlainText())

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events for completion and indentation."""
        # If completer popup is visible, let it handle navigation keys
        if self._completer.popup().isVisible():
            if event.key() in (
                Qt.Key.Key_Enter, Qt.Key.Key_Return,
                Qt.Key.Key_Escape, Qt.Key.Key_Tab, Qt.Key.Key_Backtab
            ):
                event.ignore()
                return

        # Handle special keys
        if event.key() == Qt.Key.Key_Tab:
            self._handle_tab(event)
            return
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            self._handle_return(event)
            return

        # Default handling
        super().keyPressEvent(event)

        # Show completion popup
        self._handle_completion(event)

    def _handle_tab(self, event: QKeyEvent):
        """Handle tab key for indentation."""
        cursor = self.textCursor()
        if cursor.hasSelection():
            # Indent selected lines
            self._indent_selection(cursor, event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        else:
            # Insert spaces
            cursor.insertText("    ")

    def _handle_return(self, event: QKeyEvent):
        """Handle return key with auto-indent."""
        cursor = self.textCursor()
        block = cursor.block()
        text = block.text()

        # Calculate current indentation
        indent = ""
        for char in text:
            if char in " \t":
                indent += char
            else:
                break

        # Insert newline with same indentation
        cursor.insertText("\n" + indent)

    def _indent_selection(self, cursor: QTextCursor, unindent: bool = False):
        """Indent or unindent selected lines."""
        start = cursor.selectionStart()
        end = cursor.selectionEnd()

        cursor.setPosition(start)
        start_block = cursor.blockNumber()

        cursor.setPosition(end)
        end_block = cursor.blockNumber()

        cursor.setPosition(start)
        cursor.beginEditBlock()

        for _ in range(end_block - start_block + 1):
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            if unindent:
                # Remove up to 4 spaces from start
                text = cursor.block().text()
                spaces = 0
                for char in text[:4]:
                    if char == " ":
                        spaces += 1
                    elif char == "\t":
                        spaces = 4
                        break
                    else:
                        break
                if spaces > 0:
                    cursor.movePosition(
                        QTextCursor.MoveOperation.Right,
                        QTextCursor.MoveMode.KeepAnchor,
                        spaces
                    )
                    cursor.removeSelectedText()
            else:
                cursor.insertText("    ")
            cursor.movePosition(QTextCursor.MoveOperation.NextBlock)

        cursor.endEditBlock()

    def _handle_completion(self, event: QKeyEvent):
        """Handle auto-completion popup."""
        # Only show completion for word characters
        if not event.text() or not event.text()[0].isalnum():
            if event.key() not in (Qt.Key.Key_Period, Qt.Key.Key_Underscore):
                self._completer.popup().hide()
                return

        # Get completion prefix
        cursor = self.textCursor()
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        prefix = cursor.selectedText()

        # Need at least 1 character to show completions
        if len(prefix) < 1:
            self._completer.popup().hide()
            return

        # Update completer
        if prefix != self._completer.completionPrefix():
            self._completer.setCompletionPrefix(prefix)
            self._completer.popup().setCurrentIndex(
                self._completer.completionModel().index(0, 0)
            )

        # Show popup
        cr = self.cursorRect()
        cr.setWidth(
            self._completer.popup().sizeHintForColumn(0)
            + self._completer.popup().verticalScrollBar().sizeHint().width()
        )
        self._completer.complete(cr)

    @Slot(str)
    def _insert_completion(self, completion: str):
        """Insert the selected completion."""
        cursor = self.textCursor()
        # Remove the prefix that was already typed
        extra = len(completion) - len(self._completer.completionPrefix())
        cursor.movePosition(QTextCursor.MoveOperation.Left)
        cursor.movePosition(QTextCursor.MoveOperation.EndOfWord)
        cursor.insertText(completion[-extra:])
        self.setTextCursor(cursor)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_highlighter(self) -> SpiceHighlighter:
        """Get the syntax highlighter."""
        return self._highlighter

    def get_completer(self) -> SpiceCompleter:
        """Get the auto-completer."""
        return self._completer

    def set_text(self, text: str):
        """Set the editor text."""
        self.setPlainText(text)

    def get_text(self) -> str:
        """Get the editor text."""
        return self.toPlainText()


class EditorStatusBar(QWidget):
    """Status bar for editor showing cursor position and file info."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(16)

        self._position_label = QLabel("Line 1, Col 1")
        layout.addWidget(self._position_label)

        layout.addStretch()

        self._encoding_label = QLabel("UTF-8")
        layout.addWidget(self._encoding_label)

        self._mode_label = QLabel("SPICE")
        layout.addWidget(self._mode_label)

    def update_position(self, line: int, column: int):
        """Update cursor position display."""
        self._position_label.setText(f"Line {line}, Col {column}")

    def set_encoding(self, encoding: str):
        """Set encoding display."""
        self._encoding_label.setText(encoding)

    def set_mode(self, mode: str):
        """Set mode display."""
        self._mode_label.setText(mode)
