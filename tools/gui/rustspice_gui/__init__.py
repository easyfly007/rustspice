"""
RustSpice GUI - Graphical user interface for RustSpice circuit simulator.

This package provides a PySide6-based GUI for:
- Editing SPICE netlists with syntax highlighting
- Running simulations (OP, DC, TRAN, AC)
- Viewing waveforms and results
"""

__version__ = "0.1.0"
__author__ = "RustSpice Team"

from .client import RustSpiceClient
from .main_window import MainWindow
from .editor import NetlistEditor, SpiceHighlighter, SpiceCompleter
from .console import ConsoleWidget

__all__ = [
    "RustSpiceClient",
    "MainWindow",
    "NetlistEditor",
    "SpiceHighlighter",
    "SpiceCompleter",
    "ConsoleWidget",
    "__version__",
]
