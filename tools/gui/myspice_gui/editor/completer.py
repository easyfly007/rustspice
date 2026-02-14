"""
Auto-completion for SPICE netlists.

Provides completion suggestions for:
- Device types (R, C, L, V, I, D, M, etc.)
- Control commands (.op, .dc, .tran, .ac, .model, etc.)
- Waveform types (PULSE, PWL, SIN, EXP)
- Common parameters
- Node names from the current netlist
"""

from typing import List, Optional, Set

from PySide6.QtWidgets import QCompleter, QPlainTextEdit
from PySide6.QtCore import Qt, QStringListModel
from PySide6.QtGui import QTextCursor


# SPICE keywords for auto-completion
DEVICE_TYPES = [
    "R",   # Resistor
    "C",   # Capacitor
    "L",   # Inductor
    "V",   # Voltage source
    "I",   # Current source
    "D",   # Diode
    "M",   # MOSFET
    "Q",   # BJT
    "J",   # JFET
    "E",   # VCVS
    "G",   # VCCS
    "F",   # CCCS
    "H",   # CCVS
    "X",   # Subcircuit instance
    "K",   # Mutual inductance
    "S",   # Voltage-controlled switch
    "W",   # Current-controlled switch
    "T",   # Transmission line
    "O",   # Lossy transmission line
    "U",   # Uniform RC line
    "B",   # Behavioral source
]

CONTROL_COMMANDS = [
    ".op",
    ".dc",
    ".ac",
    ".tran",
    ".model",
    ".subckt",
    ".ends",
    ".end",
    ".param",
    ".include",
    ".lib",
    ".ic",
    ".nodeset",
    ".options",
    ".print",
    ".plot",
    ".save",
    ".probe",
    ".measure",
    ".meas",
    ".four",
    ".noise",
    ".tf",
    ".sens",
    ".temp",
    ".width",
    ".global",
    ".func",
]

WAVEFORM_TYPES = [
    "DC",
    "AC",
    "PULSE",
    "PWL",
    "SIN",
    "EXP",
    "SFFM",
]

MODEL_TYPES = [
    "R",
    "C",
    "L",
    "D",
    "NPN",
    "PNP",
    "NMOS",
    "PMOS",
    "NJF",
    "PJF",
    "SW",
    "CSW",
]

AC_SWEEP_TYPES = [
    "DEC",
    "OCT",
    "LIN",
]

COMMON_PARAMETERS = [
    # Resistor
    "R=",
    "TC1=",
    "TC2=",
    # Capacitor
    "C=",
    "IC=",
    # Inductor
    "L=",
    # Diode
    "IS=",
    "N=",
    "RS=",
    "BV=",
    "IBV=",
    "TT=",
    "CJO=",
    "VJ=",
    "M=",
    # MOSFET common
    "W=",
    "L=",
    "AS=",
    "AD=",
    "PS=",
    "PD=",
    "NRS=",
    "NRD=",
    # Voltage/Current source
    "DC=",
    "AC=",
]


class SpiceCompleter(QCompleter):
    """
    Auto-completer for SPICE netlists.

    Features:
    - Case-insensitive matching
    - Completion for device types, commands, waveforms
    - Dynamic node name extraction from netlist
    """

    def __init__(self, parent: Optional[QPlainTextEdit] = None):
        super().__init__(parent)

        # Build completion list
        self._base_words = self._build_word_list()
        self._node_names: Set[str] = set()

        # Set up model
        self._model = QStringListModel(self._base_words, self)
        self.setModel(self._model)

        # Configure completer
        self.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.setFilterMode(Qt.MatchFlag.MatchStartsWith)

        # Track the editor
        self._editor = parent

    def _build_word_list(self) -> List[str]:
        """Build the base completion word list."""
        words = []

        # Add device types (both upper and lower case)
        for d in DEVICE_TYPES:
            words.append(d)
            words.append(d.lower())

        # Add control commands
        words.extend(CONTROL_COMMANDS)

        # Add waveform types
        words.extend(WAVEFORM_TYPES)
        words.extend([w.lower() for w in WAVEFORM_TYPES])

        # Add model types
        words.extend(MODEL_TYPES)

        # Add AC sweep types
        words.extend(AC_SWEEP_TYPES)
        words.extend([s.lower() for s in AC_SWEEP_TYPES])

        # Add common parameters
        words.extend(COMMON_PARAMETERS)

        return sorted(set(words))

    def update_node_names(self, netlist_text: str):
        """
        Extract node names from netlist and update completion list.

        Args:
            netlist_text: The current netlist text
        """
        self._node_names = self._extract_nodes(netlist_text)
        self._update_model()

    def _extract_nodes(self, text: str) -> Set[str]:
        """
        Extract node names from netlist text.

        Simple heuristic: words that appear after device names
        and are not numbers or keywords.
        """
        nodes = set()

        for line in text.split("\n"):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("*") or line.startswith(";"):
                continue

            # Skip control commands
            if line.startswith("."):
                continue

            # Split line into tokens
            tokens = line.split()
            if len(tokens) < 2:
                continue

            # First token is device name, following tokens may be nodes
            # (until we hit a number or parameter)
            for token in tokens[1:]:
                # Stop at parameters (contains =)
                if "=" in token:
                    break
                # Skip if it looks like a number
                if self._is_number(token):
                    break
                # Skip known keywords
                if token.upper() in WAVEFORM_TYPES:
                    continue
                # Skip parentheses content
                if token.startswith("(") or token.endswith(")"):
                    continue
                # Add as potential node name
                if token and token != "0":  # Skip ground
                    nodes.add(token)

        return nodes

    def _is_number(self, s: str) -> bool:
        """Check if string is a number (possibly with suffix)."""
        # Remove common suffixes
        s = s.rstrip("kKmMuUnNpPfF")
        if s.lower().endswith("meg"):
            s = s[:-3]
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _update_model(self):
        """Update the completion model with current words."""
        words = self._base_words + sorted(self._node_names)
        self._model.setStringList(words)

    def get_word_under_cursor(self, cursor: QTextCursor) -> str:
        """Get the word currently under the cursor."""
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        return cursor.selectedText()


def get_completion_prefix(text: str, cursor_pos: int) -> str:
    """
    Get the prefix to use for completion at cursor position.

    Args:
        text: Full text of the line
        cursor_pos: Cursor position in the line

    Returns:
        The prefix string to complete
    """
    if cursor_pos == 0:
        return ""

    # Find start of current word
    start = cursor_pos - 1
    while start >= 0 and (text[start].isalnum() or text[start] in "._"):
        start -= 1

    return text[start + 1:cursor_pos]
