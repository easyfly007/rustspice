"""
Console widget for displaying log messages.

Provides a scrollable text area with colored messages for
info, warning, error, and success states.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QLabel,
)
from PySide6.QtGui import QTextCharFormat, QColor, QFont, QTextCursor
from PySide6.QtCore import Qt, Signal


class MessageLevel(Enum):
    """Log message severity levels."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"


# Color scheme for message levels
MESSAGE_COLORS = {
    MessageLevel.INFO: "#000000",      # Black
    MessageLevel.SUCCESS: "#228B22",   # Forest Green
    MessageLevel.WARNING: "#FF8C00",   # Dark Orange
    MessageLevel.ERROR: "#DC143C",     # Crimson
    MessageLevel.DEBUG: "#708090",     # Slate Gray
}


class ConsoleWidget(QWidget):
    """
    Console widget for displaying log messages.

    Features:
    - Colored messages based on severity
    - Timestamps
    - Clear button
    - Auto-scroll to bottom
    - Copy selection to clipboard

    Signals:
        message_logged: Emitted when a message is logged (level, text)
    """

    message_logged = Signal(str, str)  # level, text

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._message_count = 0
        self._max_messages = 10000

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header with clear button
        header = QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 2)

        self._title_label = QLabel("Console")
        self._title_label.setStyleSheet("font-weight: bold;")
        header.addWidget(self._title_label)

        header.addStretch()

        self._count_label = QLabel("0 messages")
        self._count_label.setStyleSheet("color: #666;")
        header.addWidget(self._count_label)

        self._clear_button = QPushButton("Clear")
        self._clear_button.setFixedWidth(60)
        self._clear_button.clicked.connect(self.clear)
        header.addWidget(self._clear_button)

        layout.addLayout(header)

        # Text area
        self._text_edit = QPlainTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("Consolas", 9))
        self._text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._text_edit.setStyleSheet("""
            QPlainTextEdit {
                background-color: #FAFAFA;
                border: 1px solid #DDD;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self._text_edit)

    def log(
        self,
        message: str,
        level: MessageLevel = MessageLevel.INFO,
        timestamp: bool = True,
    ):
        """
        Log a message to the console.

        Args:
            message: The message text
            level: Message severity level
            timestamp: Whether to include timestamp
        """
        # Build the full message
        if timestamp:
            ts = datetime.now().strftime("%H:%M:%S")
            prefix = f"[{ts}] "
        else:
            prefix = "> "

        full_message = prefix + message

        # Get color for this level
        color = MESSAGE_COLORS.get(level, MESSAGE_COLORS[MessageLevel.INFO])

        # Create text format
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        if level == MessageLevel.ERROR:
            fmt.setFontWeight(QFont.Weight.Bold)

        # Append the message
        cursor = self._text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(full_message + "\n", fmt)

        # Auto-scroll to bottom
        self._text_edit.setTextCursor(cursor)
        self._text_edit.ensureCursorVisible()

        # Update count
        self._message_count += 1
        self._count_label.setText(f"{self._message_count} messages")

        # Trim if too many messages
        if self._message_count > self._max_messages:
            self._trim_messages()

        # Emit signal
        self.message_logged.emit(level.value, message)

    def info(self, message: str):
        """Log an info message."""
        self.log(message, MessageLevel.INFO)

    def success(self, message: str):
        """Log a success message."""
        self.log(message, MessageLevel.SUCCESS)

    def warning(self, message: str):
        """Log a warning message."""
        self.log(message, MessageLevel.WARNING)

    def error(self, message: str):
        """Log an error message."""
        self.log(message, MessageLevel.ERROR)

    def debug(self, message: str):
        """Log a debug message."""
        self.log(message, MessageLevel.DEBUG)

    def clear(self):
        """Clear all messages."""
        self._text_edit.clear()
        self._message_count = 0
        self._count_label.setText("0 messages")

    def _trim_messages(self):
        """Remove oldest messages to stay under limit."""
        # Remove first 10% of messages
        cursor = self._text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)

        lines_to_remove = self._max_messages // 10
        for _ in range(lines_to_remove):
            cursor.movePosition(
                QTextCursor.MoveOperation.Down,
                QTextCursor.MoveMode.KeepAnchor
            )

        cursor.removeSelectedText()
        self._message_count -= lines_to_remove

    def set_max_messages(self, max_messages: int):
        """Set the maximum number of messages to keep."""
        self._max_messages = max_messages

    def get_text(self) -> str:
        """Get all console text."""
        return self._text_edit.toPlainText()
