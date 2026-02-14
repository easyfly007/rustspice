"""
Tests for the simulation control module.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from PySide6.QtWidgets import QApplication

from rustspice_gui.simulation import SimulationPanel, SimulationWorker, SimulationTask
from rustspice_gui.simulation.worker import AnalysisType, ConnectionChecker
from rustspice_gui.simulation.panel import OpPanel, DcPanel, TranPanel, AcPanel
from rustspice_gui.client import AcSweepType


# Ensure QApplication exists for widget tests
@pytest.fixture(scope="session")
def app():
    """Create QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestSimulationPanel:
    """Tests for SimulationPanel widget."""

    def test_panel_creation(self, app):
        """Test panel can be created."""
        panel = SimulationPanel()
        assert panel is not None

    def test_get_analysis_type(self, app):
        """Test analysis type getter."""
        panel = SimulationPanel()

        # Default is first tab (OP)
        assert panel.get_analysis_type() == "op"

        # Can get enum version
        assert panel.get_analysis_enum() == AnalysisType.OP

    def test_initial_state(self, app):
        """Test initial panel state."""
        panel = SimulationPanel()

        assert not panel.is_running()

    def test_set_running_state(self, app):
        """Test setting running state."""
        panel = SimulationPanel()

        panel.set_running(True)
        assert panel.is_running()

        panel.set_running(False)
        assert not panel.is_running()

    def test_set_status(self, app):
        """Test status message setting."""
        panel = SimulationPanel()

        panel.set_status("Test message")
        panel.set_status("Error message", is_error=True)
        panel.set_success("Success message")
        panel.set_progress("Progress message")

    def test_server_status(self, app):
        """Test server status setting."""
        panel = SimulationPanel()

        panel.set_server_status(True, "http://localhost:3000")
        panel.set_server_status(False)

    def test_reset(self, app):
        """Test panel reset."""
        panel = SimulationPanel()

        panel.set_running(True)
        panel.set_status("Test")
        panel.reset()

        assert not panel.is_running()


class TestOpPanel:
    """Tests for OP panel."""

    def test_creation(self, app):
        """Test OP panel creation."""
        panel = OpPanel()
        assert panel is not None

    def test_get_params(self, app):
        """Test OP panel returns empty params."""
        panel = OpPanel()
        params = panel.get_params()
        assert params == {}

    def test_validate(self, app):
        """Test OP panel validation always passes."""
        panel = OpPanel()
        errors = panel.validate()
        assert errors == []


class TestDcPanel:
    """Tests for DC panel."""

    def test_creation(self, app):
        """Test DC panel creation."""
        panel = DcPanel()
        assert panel is not None

    def test_get_params(self, app):
        """Test DC panel returns parameters."""
        panel = DcPanel()
        params = panel.get_params()

        assert "source" in params
        assert "start" in params
        assert "stop" in params
        assert "step" in params

    def test_validate_missing_source(self, app):
        """Test validation fails without source."""
        panel = DcPanel()
        panel._source.setCurrentText("")

        errors = panel.validate()
        assert len(errors) > 0
        assert "source" in errors[0].lower()

    def test_validate_invalid_source(self, app):
        """Test validation fails with invalid source name."""
        panel = DcPanel()
        panel._source.setCurrentText("R1")  # Not a V or I source

        errors = panel.validate()
        assert len(errors) > 0

    def test_validate_same_start_stop(self, app):
        """Test validation fails when start equals stop."""
        panel = DcPanel()
        panel._start.setValue(5.0)
        panel._stop.setValue(5.0)

        errors = panel.validate()
        assert len(errors) > 0


class TestTranPanel:
    """Tests for TRAN panel."""

    def test_creation(self, app):
        """Test TRAN panel creation."""
        panel = TranPanel()
        assert panel is not None

    def test_get_params(self, app):
        """Test TRAN panel returns parameters."""
        panel = TranPanel()
        params = panel.get_params()

        assert "tstep" in params
        assert "tstop" in params
        assert "tstart" in params

    def test_get_params_with_tmax(self, app):
        """Test TRAN panel includes tmax when set."""
        panel = TranPanel()
        panel._tmax.setValue(1e-6)

        params = panel.get_params()
        assert "tmax" in params
        assert params["tmax"] == 1e-6

    def test_validate_invalid_times(self, app):
        """Test validation fails with invalid times."""
        panel = TranPanel()
        panel._tstart.setValue(1.0)
        panel._tstop.setValue(0.5)

        errors = panel.validate()
        assert len(errors) > 0


class TestAcPanel:
    """Tests for AC panel."""

    def test_creation(self, app):
        """Test AC panel creation."""
        panel = AcPanel()
        assert panel is not None

    def test_get_params(self, app):
        """Test AC panel returns parameters."""
        panel = AcPanel()
        params = panel.get_params()

        assert "sweep" in params
        assert "points" in params
        assert "fstart" in params
        assert "fstop" in params
        assert isinstance(params["sweep"], AcSweepType)

    def test_validate_invalid_frequencies(self, app):
        """Test validation fails with invalid frequencies."""
        panel = AcPanel()
        panel._fstart.setValue(1e6)
        panel._fstop.setValue(1.0)

        errors = panel.validate()
        assert len(errors) > 0


class TestSimulationTask:
    """Tests for SimulationTask dataclass."""

    def test_creation(self):
        """Test task creation."""
        task = SimulationTask(
            analysis=AnalysisType.OP,
            netlist="V1 in 0 DC 5\n.op\n.end",
            params={}
        )

        assert task.analysis == AnalysisType.OP
        assert "V1" in task.netlist
        assert task.params == {}

    def test_dc_task(self):
        """Test DC task creation."""
        task = SimulationTask(
            analysis=AnalysisType.DC,
            netlist="V1 in 0 DC 0\n.dc V1 0 5 0.1\n.end",
            params={"source": "V1", "start": 0, "stop": 5, "step": 0.1}
        )

        assert task.analysis == AnalysisType.DC
        assert task.params["source"] == "V1"


class TestSimulationWorker:
    """Tests for SimulationWorker."""

    def test_creation(self, app):
        """Test worker creation."""
        client = MagicMock()
        worker = SimulationWorker(client)

        assert worker is not None

    def test_set_task(self, app):
        """Test setting a task."""
        client = MagicMock()
        worker = SimulationWorker(client)

        task = SimulationTask(
            analysis=AnalysisType.OP,
            netlist="V1 in 0 DC 5\n.op\n.end",
            params={}
        )
        worker.set_task(task)

        assert worker._task == task

    def test_request_stop(self, app):
        """Test stop request."""
        client = MagicMock()
        worker = SimulationWorker(client)

        worker.request_stop()
        assert worker._stop_requested


class TestConnectionChecker:
    """Tests for ConnectionChecker."""

    def test_creation(self, app):
        """Test checker creation."""
        client = MagicMock()
        checker = ConnectionChecker(client)

        assert checker is not None
