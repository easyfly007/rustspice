"""Tests for the netlist editor components."""

import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QTextDocument

from rustspice_gui.editor import NetlistEditor, SpiceHighlighter, SpiceCompleter
from rustspice_gui.editor.completer import (
    DEVICE_TYPES,
    CONTROL_COMMANDS,
    WAVEFORM_TYPES,
    get_completion_prefix,
)


# Create QApplication for tests (required for Qt widgets)
@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestSpiceHighlighter:
    """Tests for SpiceHighlighter."""

    def test_creation(self, qapp):
        """Test highlighter creation."""
        doc = QTextDocument()
        highlighter = SpiceHighlighter(doc)
        assert highlighter is not None

    def test_creation_without_document(self, qapp):
        """Test highlighter creation without document."""
        highlighter = SpiceHighlighter()
        assert highlighter is not None

    def test_highlight_comment(self, qapp):
        """Test comment highlighting."""
        doc = QTextDocument()
        highlighter = SpiceHighlighter(doc)
        doc.setPlainText("* This is a comment")
        # Just verify no errors occur
        assert doc.toPlainText() == "* This is a comment"

    def test_highlight_control_command(self, qapp):
        """Test control command highlighting."""
        doc = QTextDocument()
        highlighter = SpiceHighlighter(doc)
        doc.setPlainText(".tran 1n 10u")
        assert doc.toPlainText() == ".tran 1n 10u"

    def test_highlight_device(self, qapp):
        """Test device highlighting."""
        doc = QTextDocument()
        highlighter = SpiceHighlighter(doc)
        doc.setPlainText("R1 in out 1k")
        assert doc.toPlainText() == "R1 in out 1k"

    def test_highlight_number(self, qapp):
        """Test number highlighting."""
        doc = QTextDocument()
        highlighter = SpiceHighlighter(doc)
        doc.setPlainText("C1 a b 100n")
        assert doc.toPlainText() == "C1 a b 100n"

    def test_highlight_waveform(self, qapp):
        """Test waveform keyword highlighting."""
        doc = QTextDocument()
        highlighter = SpiceHighlighter(doc)
        doc.setPlainText("V1 in 0 PULSE(0 5 0 1n 1n)")
        assert doc.toPlainText() == "V1 in 0 PULSE(0 5 0 1n 1n)"

    def test_highlight_full_netlist(self, qapp):
        """Test highlighting a full netlist."""
        doc = QTextDocument()
        highlighter = SpiceHighlighter(doc)
        netlist = """* RC Circuit
V1 in 0 DC 5
R1 in out 1k
C1 out 0 100n
.tran 1n 10u
.end
"""
        doc.setPlainText(netlist)
        assert ".tran" in doc.toPlainText()


class TestSpiceCompleter:
    """Tests for SpiceCompleter."""

    def test_creation(self, qapp):
        """Test completer creation."""
        completer = SpiceCompleter()
        assert completer is not None

    def test_device_types_in_completion(self, qapp):
        """Test that device types are in completion list."""
        completer = SpiceCompleter()
        model = completer.model()
        words = [model.data(model.index(i, 0)) for i in range(model.rowCount())]

        for device in DEVICE_TYPES:
            assert device in words or device.lower() in words

    def test_commands_in_completion(self, qapp):
        """Test that commands are in completion list."""
        completer = SpiceCompleter()
        model = completer.model()
        words = [model.data(model.index(i, 0)) for i in range(model.rowCount())]

        for cmd in CONTROL_COMMANDS:
            assert cmd in words

    def test_waveforms_in_completion(self, qapp):
        """Test that waveform types are in completion list."""
        completer = SpiceCompleter()
        model = completer.model()
        words = [model.data(model.index(i, 0)) for i in range(model.rowCount())]

        for wf in WAVEFORM_TYPES:
            assert wf in words or wf.lower() in words

    def test_update_node_names(self, qapp):
        """Test updating node names from netlist."""
        completer = SpiceCompleter()
        netlist = """V1 in 0 5
R1 in out 1k
C1 out gnd 100n
"""
        completer.update_node_names(netlist)

        model = completer.model()
        words = [model.data(model.index(i, 0)) for i in range(model.rowCount())]

        assert "in" in words
        assert "out" in words

    def test_extract_nodes_skips_comments(self, qapp):
        """Test that node extraction skips comments."""
        completer = SpiceCompleter()
        netlist = """* Comment line
R1 in out 1k
; Another comment
C1 out 0 100n
"""
        completer.update_node_names(netlist)

        # Should have extracted 'in' and 'out'
        assert "in" in completer._node_names
        assert "out" in completer._node_names

    def test_extract_nodes_skips_commands(self, qapp):
        """Test that node extraction skips control commands."""
        completer = SpiceCompleter()
        netlist = """R1 in out 1k
.tran 1n 10u
.end
"""
        completer.update_node_names(netlist)

        # Should not have 'tran' or 'end' as nodes
        assert "tran" not in completer._node_names
        assert "end" not in completer._node_names


class TestGetCompletionPrefix:
    """Tests for get_completion_prefix function."""

    def test_empty_line(self):
        """Test empty line."""
        assert get_completion_prefix("", 0) == ""

    def test_start_of_word(self):
        """Test at start of word."""
        assert get_completion_prefix("R1", 1) == "R"

    def test_middle_of_word(self):
        """Test in middle of word."""
        assert get_completion_prefix("R1 in out", 4) == "in"

    def test_after_dot(self):
        """Test after dot (control command)."""
        assert get_completion_prefix(".tra", 4) == ".tra"

    def test_with_underscore(self):
        """Test word with underscore."""
        assert get_completion_prefix("my_node", 7) == "my_node"


class TestNetlistEditor:
    """Tests for NetlistEditor."""

    def test_creation(self, qapp):
        """Test editor creation."""
        editor = NetlistEditor()
        assert editor is not None

    def test_has_highlighter(self, qapp):
        """Test that editor has syntax highlighter."""
        editor = NetlistEditor()
        highlighter = editor.get_highlighter()
        assert highlighter is not None
        assert isinstance(highlighter, SpiceHighlighter)

    def test_has_completer(self, qapp):
        """Test that editor has completer."""
        editor = NetlistEditor()
        completer = editor.get_completer()
        assert completer is not None
        assert isinstance(completer, SpiceCompleter)

    def test_set_and_get_text(self, qapp):
        """Test setting and getting text."""
        editor = NetlistEditor()
        text = "R1 in out 1k"
        editor.set_text(text)
        assert editor.get_text() == text

    def test_cursor_position(self, qapp):
        """Test cursor position tracking."""
        editor = NetlistEditor()
        editor.set_text("Line 1\nLine 2\nLine 3")

        line, col = editor.get_cursor_position()
        assert line == 1
        assert col == 1

    def test_go_to_line(self, qapp):
        """Test go to line functionality."""
        editor = NetlistEditor()
        editor.set_text("Line 1\nLine 2\nLine 3")
        editor.go_to_line(2)

        line, _ = editor.get_cursor_position()
        assert line == 2

    def test_line_number_area_width(self, qapp):
        """Test line number area width calculation."""
        editor = NetlistEditor()
        editor.set_text("Line 1\n" * 100)

        width = editor.line_number_area_width()
        assert width > 0

    def test_cursor_position_signal(self, qapp):
        """Test cursor position signal emission."""
        editor = NetlistEditor()
        editor.set_text("Line 1\nLine 2")

        received = []

        def on_position(line, col):
            received.append((line, col))

        editor.cursor_position_changed.connect(on_position)
        editor.go_to_line(2)

        # Should have received at least one signal
        assert len(received) > 0
        assert received[-1][0] == 2  # Line 2
