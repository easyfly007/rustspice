"""
Netlist editor components with syntax highlighting.

This module provides a full-featured SPICE netlist editor with:
- Syntax highlighting for SPICE keywords, devices, numbers
- Line number display
- Auto-completion for device types and commands
- Bracket matching
"""

from .editor import NetlistEditor
from .highlighter import SpiceHighlighter
from .completer import SpiceCompleter

__all__ = ["NetlistEditor", "SpiceHighlighter", "SpiceCompleter"]
