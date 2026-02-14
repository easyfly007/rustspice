"""Tests for viewer components."""

import pytest
import numpy as np


class TestWaveformViewer:
    """Tests for WaveformViewer widget."""

    def test_import(self):
        """Test that WaveformViewer can be imported."""
        from myspice_gui.viewer import WaveformViewer
        assert WaveformViewer is not None

    def test_add_signal(self, qtbot):
        """Test adding a signal to the viewer."""
        from myspice_gui.viewer import WaveformViewer

        viewer = WaveformViewer()
        qtbot.addWidget(viewer)

        x_data = [0, 1, 2, 3, 4]
        y_data = [0, 1, 0, -1, 0]
        viewer.add_signal("test_signal", x_data, y_data)

        assert "test_signal" in viewer.get_signal_names()

    def test_remove_signal(self, qtbot):
        """Test removing a signal from the viewer."""
        from myspice_gui.viewer import WaveformViewer

        viewer = WaveformViewer()
        qtbot.addWidget(viewer)

        viewer.add_signal("test", [0, 1], [0, 1])
        assert "test" in viewer.get_signal_names()

        viewer.remove_signal("test")
        assert "test" not in viewer.get_signal_names()

    def test_signal_visibility(self, qtbot):
        """Test signal visibility toggle."""
        from myspice_gui.viewer import WaveformViewer

        viewer = WaveformViewer()
        qtbot.addWidget(viewer)

        viewer.add_signal("test", [0, 1], [0, 1])
        viewer.set_signal_visible("test", False)

        # Check internal state
        signal = viewer._signals["test"]
        assert signal.visible is False

    def test_signal_color(self, qtbot):
        """Test changing signal color."""
        from myspice_gui.viewer import WaveformViewer

        viewer = WaveformViewer()
        qtbot.addWidget(viewer)

        viewer.add_signal("test", [0, 1], [0, 1])
        viewer.set_signal_color("test", "#FF0000")

        signal = viewer._signals["test"]
        assert signal.color == "#FF0000"

    def test_get_signal_data(self, qtbot):
        """Test retrieving signal data."""
        from myspice_gui.viewer import WaveformViewer

        viewer = WaveformViewer()
        qtbot.addWidget(viewer)

        x_data = [0, 1, 2]
        y_data = [0, 5, 10]
        viewer.add_signal("test", x_data, y_data)

        result = viewer.get_signal_data("test")
        assert result is not None
        x, y = result
        np.testing.assert_array_equal(x, x_data)
        np.testing.assert_array_equal(y, y_data)

    def test_clear(self, qtbot):
        """Test clearing all signals."""
        from myspice_gui.viewer import WaveformViewer

        viewer = WaveformViewer()
        qtbot.addWidget(viewer)

        viewer.add_signal("sig1", [0, 1], [0, 1])
        viewer.add_signal("sig2", [0, 1], [0, 2])
        assert len(viewer.get_signal_names()) == 2

        viewer.clear()
        assert len(viewer.get_signal_names()) == 0


class TestBodePlot:
    """Tests for BodePlot widget."""

    def test_import(self):
        """Test that BodePlot can be imported."""
        from myspice_gui.viewer import BodePlot
        assert BodePlot is not None

    def test_add_ac_signal(self, qtbot):
        """Test adding an AC signal to Bode plot."""
        from myspice_gui.viewer import BodePlot

        bode = BodePlot()
        qtbot.addWidget(bode)

        freqs = [1, 10, 100, 1000]
        mag_db = [0, -3, -6, -9]
        phase_deg = [0, -45, -90, -135]

        bode.add_signal("V(out)", freqs, mag_db, phase_deg)
        assert "V(out)" in bode.get_signal_names()

    def test_remove_ac_signal(self, qtbot):
        """Test removing a signal from Bode plot."""
        from myspice_gui.viewer import BodePlot

        bode = BodePlot()
        qtbot.addWidget(bode)

        bode.add_signal("test", [1, 10], [0, -3], [0, -45])
        assert "test" in bode.get_signal_names()

        bode.remove_signal("test")
        assert "test" not in bode.get_signal_names()

    def test_signal_visibility(self, qtbot):
        """Test AC signal visibility toggle."""
        from myspice_gui.viewer import BodePlot

        bode = BodePlot()
        qtbot.addWidget(bode)

        bode.add_signal("test", [1, 10], [0, -3], [0, -45])
        bode.set_signal_visible("test", False)

        signal = bode._signals["test"]
        assert signal.visible is False

    def test_clear(self, qtbot):
        """Test clearing all signals."""
        from myspice_gui.viewer import BodePlot

        bode = BodePlot()
        qtbot.addWidget(bode)

        bode.add_signal("sig1", [1, 10], [0, -3], [0, -45])
        bode.add_signal("sig2", [1, 10], [0, -6], [0, -90])
        assert len(bode.get_signal_names()) == 2

        bode.clear()
        assert len(bode.get_signal_names()) == 0


class TestSignalListWidget:
    """Tests for SignalListWidget."""

    def test_import(self):
        """Test that SignalListWidget can be imported."""
        from myspice_gui.viewer import SignalListWidget
        assert SignalListWidget is not None

    def test_add_signal(self, qtbot):
        """Test adding a signal to the list."""
        from myspice_gui.viewer import SignalListWidget

        widget = SignalListWidget()
        qtbot.addWidget(widget)

        widget.add_signal("V(out)", "#1f77b4")
        assert "V(out)" in widget.get_signal_names()

    def test_remove_signal(self, qtbot):
        """Test removing a signal from the list."""
        from myspice_gui.viewer import SignalListWidget

        widget = SignalListWidget()
        qtbot.addWidget(widget)

        widget.add_signal("test")
        assert "test" in widget.get_signal_names()

        widget.remove_signal("test")
        assert "test" not in widget.get_signal_names()

    def test_signal_visibility(self, qtbot):
        """Test signal visibility."""
        from myspice_gui.viewer import SignalListWidget

        widget = SignalListWidget()
        qtbot.addWidget(widget)

        widget.add_signal("test", visible=True)
        assert widget.is_signal_visible("test") is True

        widget.set_signal_visible("test", False)
        assert widget.is_signal_visible("test") is False

    def test_signal_color(self, qtbot):
        """Test signal color."""
        from myspice_gui.viewer import SignalListWidget

        widget = SignalListWidget()
        qtbot.addWidget(widget)

        widget.add_signal("test", color="#FF0000")
        assert widget.get_signal_color("test") == "#FF0000"

        widget.set_signal_color("test", "#00FF00")
        assert widget.get_signal_color("test") == "#00FF00"

    def test_clear(self, qtbot):
        """Test clearing all signals."""
        from myspice_gui.viewer import SignalListWidget

        widget = SignalListWidget()
        qtbot.addWidget(widget)

        widget.add_signal("sig1")
        widget.add_signal("sig2")
        assert len(widget.get_signal_names()) == 2

        widget.clear()
        assert len(widget.get_signal_names()) == 0


class TestCursorManager:
    """Tests for CursorManager."""

    def test_import(self):
        """Test that CursorManager can be imported."""
        from myspice_gui.viewer import CursorManager
        assert CursorManager is not None

    def test_add_cursor(self, qtbot):
        """Test adding a cursor."""
        import pyqtgraph as pg
        from myspice_gui.viewer import CursorManager

        plot = pg.PlotWidget()
        qtbot.addWidget(plot)

        manager = CursorManager(plot)
        cursor = manager.add_cursor("C1", x_position=1.0)

        assert cursor.name == "C1"
        assert "C1" in manager.get_cursor_names()

    def test_remove_cursor(self, qtbot):
        """Test removing a cursor."""
        import pyqtgraph as pg
        from myspice_gui.viewer import CursorManager

        plot = pg.PlotWidget()
        qtbot.addWidget(plot)

        manager = CursorManager(plot)
        manager.add_cursor("C1")
        assert "C1" in manager.get_cursor_names()

        manager.remove_cursor("C1")
        assert "C1" not in manager.get_cursor_names()

    def test_cursor_position(self, qtbot):
        """Test cursor position get/set."""
        import pyqtgraph as pg
        from myspice_gui.viewer import CursorManager

        plot = pg.PlotWidget()
        qtbot.addWidget(plot)

        manager = CursorManager(plot)
        manager.add_cursor("C1", x_position=1.0)

        assert manager.get_cursor_position("C1") == 1.0

        manager.set_cursor_position("C1", 2.5)
        assert manager.get_cursor_position("C1") == 2.5

    def test_delta_measurement(self, qtbot):
        """Test delta measurement between two cursors."""
        import pyqtgraph as pg
        from myspice_gui.viewer import CursorManager

        plot = pg.PlotWidget()
        qtbot.addWidget(plot)

        manager = CursorManager(plot)
        manager.add_cursor("C1", x_position=1.0)
        manager.add_cursor("C2", x_position=4.0)

        delta = manager.get_delta()
        assert delta is not None
        x1, x2, diff = delta
        assert x1 == 1.0
        assert x2 == 4.0
        assert diff == 3.0

    def test_clear_cursors(self, qtbot):
        """Test clearing all cursors."""
        import pyqtgraph as pg
        from myspice_gui.viewer import CursorManager

        plot = pg.PlotWidget()
        qtbot.addWidget(plot)

        manager = CursorManager(plot)
        manager.add_cursor("C1")
        manager.add_cursor("C2")
        assert len(manager.get_cursor_names()) == 2

        manager.clear()
        assert len(manager.get_cursor_names()) == 0


class TestCursorReadout:
    """Tests for CursorReadout widget."""

    def test_import(self):
        """Test that CursorReadout can be imported."""
        from myspice_gui.viewer import CursorReadout
        assert CursorReadout is not None

    def test_update_cursors(self, qtbot):
        """Test updating cursor readout."""
        from myspice_gui.viewer.cursors import CursorReadout

        readout = CursorReadout()
        qtbot.addWidget(readout)

        readout.update_cursors(1e-6, 5e-6, "s")

        # Check labels are updated
        assert readout._c1_label.text() != "---"
        assert readout._c2_label.text() != "---"
        assert readout._delta_label.text() != "---"
        assert readout._freq_label.text() != "---"

    def test_format_engineering(self, qtbot):
        """Test engineering notation formatting."""
        from myspice_gui.viewer.cursors import CursorReadout

        readout = CursorReadout()
        qtbot.addWidget(readout)

        # Test various scales
        assert "1" in readout._format_value(1.0, "V")
        assert "m" in readout._format_value(1e-3, "V")
        assert "u" in readout._format_value(1e-6, "V") or "µ" in readout._format_value(1e-6, "V")
        assert "n" in readout._format_value(1e-9, "V")
        assert "k" in readout._format_value(1e3, "Hz")
        assert "M" in readout._format_value(1e6, "Hz")
