"""
SPICE syntax highlighter for the netlist editor.

Provides syntax highlighting for:
- Comments (* and ;)
- Control commands (.op, .dc, .tran, .ac, .model, etc.)
- Device types (R, C, L, V, I, D, M, E, G, F, H, X)
- Numbers with engineering suffixes (1k, 10u, 100n, etc.)
- Strings and identifiers
- Parameters and expressions
"""

import re
from typing import List, Tuple, Optional

from PySide6.QtGui import (
    QSyntaxHighlighter,
    QTextCharFormat,
    QColor,
    QFont,
    QTextDocument,
)


class SpiceHighlighter(QSyntaxHighlighter):
    """
    Syntax highlighter for SPICE netlists.

    Color scheme:
    - Comments: Gray italic
    - Control commands: Blue bold (.op, .dc, etc.)
    - Device types: Purple (R, C, L, V, etc.)
    - Numbers: Dark cyan
    - Strings: Brown
    - Parameters: Dark green
    - Model names: Magenta
    """

    def __init__(self, document: Optional[QTextDocument] = None):
        super().__init__(document)
        self._rules: List[Tuple[re.Pattern, QTextCharFormat]] = []
        self._setup_formats()
        self._setup_rules()

    def _setup_formats(self):
        """Set up text formats for different token types."""
        # Comment format: gray italic
        self._comment_format = QTextCharFormat()
        self._comment_format.setForeground(QColor("#6A737D"))
        self._comment_format.setFontItalic(True)

        # Control command format: blue bold (.op, .dc, etc.)
        self._command_format = QTextCharFormat()
        self._command_format.setForeground(QColor("#0000CC"))
        self._command_format.setFontWeight(QFont.Weight.Bold)

        # Device type format: purple (R, C, L, etc.)
        self._device_format = QTextCharFormat()
        self._device_format.setForeground(QColor("#8B008B"))
        self._device_format.setFontWeight(QFont.Weight.Bold)

        # Number format: dark cyan
        self._number_format = QTextCharFormat()
        self._number_format.setForeground(QColor("#008B8B"))

        # String format: brown
        self._string_format = QTextCharFormat()
        self._string_format.setForeground(QColor("#A52A2A"))

        # Parameter format: dark green
        self._param_format = QTextCharFormat()
        self._param_format.setForeground(QColor("#006400"))

        # Model name format: magenta
        self._model_format = QTextCharFormat()
        self._model_format.setForeground(QColor("#C71585"))

        # Waveform keywords: orange
        self._waveform_format = QTextCharFormat()
        self._waveform_format.setForeground(QColor("#D2691E"))
        self._waveform_format.setFontWeight(QFont.Weight.Bold)

        # Node name format: default with slight color
        self._node_format = QTextCharFormat()
        self._node_format.setForeground(QColor("#333333"))

        # Error/invalid format: red underline
        self._error_format = QTextCharFormat()
        self._error_format.setForeground(QColor("#DC143C"))
        self._error_format.setUnderlineStyle(
            QTextCharFormat.UnderlineStyle.WaveUnderline
        )
        self._error_format.setUnderlineColor(QColor("#DC143C"))

    def _setup_rules(self):
        """Set up highlighting rules."""
        # Order matters: later rules can override earlier ones

        # 1. Control commands (case insensitive)
        control_commands = [
            r"\.op\b", r"\.dc\b", r"\.ac\b", r"\.tran\b",
            r"\.model\b", r"\.subckt\b", r"\.ends\b", r"\.end\b",
            r"\.param\b", r"\.include\b", r"\.lib\b", r"\.ic\b",
            r"\.nodeset\b", r"\.options\b", r"\.print\b", r"\.plot\b",
            r"\.save\b", r"\.probe\b", r"\.measure\b", r"\.meas\b",
            r"\.four\b", r"\.noise\b", r"\.tf\b", r"\.sens\b",
            r"\.temp\b", r"\.width\b", r"\.global\b", r"\.func\b",
        ]
        for cmd in control_commands:
            pattern = re.compile(cmd, re.IGNORECASE)
            self._rules.append((pattern, self._command_format))

        # 2. Waveform keywords (PULSE, PWL, SIN, EXP, SFFM)
        waveform_keywords = [
            r"\bPULSE\b", r"\bPWL\b", r"\bSIN\b", r"\bEXP\b",
            r"\bSFFM\b", r"\bDC\b", r"\bAC\b",
        ]
        for kw in waveform_keywords:
            pattern = re.compile(kw, re.IGNORECASE)
            self._rules.append((pattern, self._waveform_format))

        # 3. Device types at start of line (R, C, L, V, I, D, M, Q, J, E, G, F, H, X)
        # Pattern: device name starting with device letter
        device_pattern = re.compile(
            r"^[RCLVIDMQJEGFHXrclvidmqjegfhx]\w*",
            re.MULTILINE
        )
        self._rules.append((device_pattern, self._device_format))

        # 4. Numbers with optional engineering suffix
        # Matches: 1, 1.0, 1e-3, 1.5e6, 1k, 10meg, 100u, 1.5n, etc.
        number_pattern = re.compile(
            r"\b[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?(?:meg|[kKmMuUnNpPfF])?\b"
        )
        self._rules.append((number_pattern, self._number_format))

        # 5. Parameter assignments (name=value)
        param_pattern = re.compile(r"\b\w+(?==)")
        self._rules.append((param_pattern, self._param_format))

        # 6. Model names after .model command
        model_name_pattern = re.compile(r"\.model\s+(\w+)", re.IGNORECASE)
        self._rules.append((model_name_pattern, self._model_format))

        # 7. Subcircuit names after .subckt command
        subckt_name_pattern = re.compile(r"\.subckt\s+(\w+)", re.IGNORECASE)
        self._rules.append((subckt_name_pattern, self._model_format))

    def highlightBlock(self, text: str):
        """Apply syntax highlighting to a block of text."""
        # First, check if the entire line is a comment
        stripped = text.lstrip()
        if stripped.startswith("*") or stripped.startswith(";"):
            self.setFormat(0, len(text), self._comment_format)
            return

        # Apply rules
        for pattern, fmt in self._rules:
            for match in pattern.finditer(text):
                # For patterns with groups, highlight the group
                if match.lastindex:
                    start = match.start(1)
                    length = match.end(1) - match.start(1)
                else:
                    start = match.start()
                    length = match.end() - match.start()
                self.setFormat(start, length, fmt)

        # Handle inline comments ($ or ; after content)
        for marker in ["$", ";"]:
            idx = text.find(marker)
            if idx > 0:  # Not at start (full-line comments handled above)
                self.setFormat(idx, len(text) - idx, self._comment_format)
                break


# Color scheme constants for reference
SPICE_COLORS = {
    "comment": "#6A737D",
    "command": "#0000CC",
    "device": "#8B008B",
    "number": "#008B8B",
    "string": "#A52A2A",
    "parameter": "#006400",
    "model": "#C71585",
    "waveform": "#D2691E",
    "node": "#333333",
    "error": "#DC143C",
}
