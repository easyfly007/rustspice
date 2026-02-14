"""
MySpice GUI - Graphical user interface for MySpice circuit simulator.

This package provides a PySide6-based GUI for:
- Editing SPICE netlists with syntax highlighting
- Running simulations (OP, DC, TRAN, AC)
- Viewing waveforms and results
"""

__version__ = "0.1.0"
__author__ = "MySpice Team"

from .client import MySpiceClient
from .main_window import MainWindow
from .editor import NetlistEditor, SpiceHighlighter, SpiceCompleter
from .console import ConsoleWidget

__all__ = [
    "MySpiceClient",
    "MainWindow",
    "NetlistEditor",
    "SpiceHighlighter",
    "SpiceCompleter",
    "ConsoleWidget",
    "__version__",
]
