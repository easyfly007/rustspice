"""Tests for the RustSpice HTTP client."""

import pytest
from rustspice_gui.client import (
    RustSpiceClient,
    RunResult,
    WaveformData,
    AnalysisType,
    AcSweepType,
    CircuitSummary,
    DeviceInfo,
    ClientError,
)


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_from_dict_op(self):
        """Test creating RunResult from OP response."""
        data = {
            "run_id": 1,
            "analysis": "Op",
            "status": "Success",
            "iterations": 5,
            "nodes": ["in", "out"],
            "solution": [5.0, 3.333],
            "message": None,
        }
        result = RunResult.from_dict(data)

        assert result.run_id == 1
        assert result.analysis == AnalysisType.OP
        assert result.status == "Success"
        assert result.iterations == 5
        assert result.nodes == ["in", "out"]
        assert result.solution == [5.0, 3.333]

    def test_from_dict_tran(self):
        """Test creating RunResult from TRAN response."""
        data = {
            "run_id": 2,
            "analysis": "Tran",
            "status": "Success",
            "iterations": 10,
            "nodes": ["in", "out"],
            "solution": [],
            "tran_times": [0.0, 1e-9, 2e-9],
            "tran_solutions": [[5.0, 0.0], [5.0, 1.0], [5.0, 2.0]],
        }
        result = RunResult.from_dict(data)

        assert result.analysis == AnalysisType.TRAN
        assert result.tran_times == [0.0, 1e-9, 2e-9]
        assert len(result.tran_solutions) == 3

    def test_from_dict_dc(self):
        """Test creating RunResult from DC response."""
        data = {
            "run_id": 3,
            "analysis": "Dc",
            "status": "Success",
            "iterations": 0,
            "nodes": ["in", "out"],
            "solution": [],
            "sweep_var": "V1",
            "sweep_values": [0.0, 1.0, 2.0],
            "sweep_solutions": [[0.0, 0.0], [1.0, 0.67], [2.0, 1.33]],
        }
        result = RunResult.from_dict(data)

        assert result.analysis == AnalysisType.DC
        assert result.sweep_var == "V1"
        assert result.sweep_values == [0.0, 1.0, 2.0]

    def test_from_dict_ac(self):
        """Test creating RunResult from AC response."""
        data = {
            "run_id": 4,
            "analysis": "Ac",
            "status": "Success",
            "iterations": 0,
            "nodes": ["in", "out"],
            "solution": [],
            "ac_frequencies": [1.0, 10.0, 100.0],
            "ac_solutions": [[[0.0, 0.0], [-3.0, -45.0]], [[0.0, 0.0], [-10.0, -80.0]]],
        }
        result = RunResult.from_dict(data)

        assert result.analysis == AnalysisType.AC
        assert result.ac_frequencies == [1.0, 10.0, 100.0]

    def test_from_dict_unknown_analysis(self):
        """Test handling unknown analysis type."""
        data = {
            "run_id": 5,
            "analysis": "Unknown",
            "status": "Success",
            "iterations": 0,
            "nodes": [],
            "solution": [],
        }
        result = RunResult.from_dict(data)
        assert result.analysis == AnalysisType.OP  # Falls back to OP


class TestWaveformData:
    """Tests for WaveformData dataclass."""

    def test_creation(self):
        """Test creating WaveformData."""
        waveform = WaveformData(
            signal="V(out)",
            analysis="Tran",
            x_label="time",
            x_unit="s",
            y_label="voltage",
            y_unit="V",
            x_values=[0.0, 1e-9, 2e-9],
            y_values=[0.0, 1.0, 2.0],
        )

        assert waveform.signal == "V(out)"
        assert waveform.analysis == "Tran"
        assert len(waveform.x_values) == 3
        assert len(waveform.y_values) == 3


class TestCircuitSummary:
    """Tests for CircuitSummary dataclass."""

    def test_creation(self):
        """Test creating CircuitSummary."""
        summary = CircuitSummary(
            node_count=5,
            device_count=3,
            model_count=1,
        )

        assert summary.node_count == 5
        assert summary.device_count == 3
        assert summary.model_count == 1


class TestDeviceInfo:
    """Tests for DeviceInfo dataclass."""

    def test_creation(self):
        """Test creating DeviceInfo."""
        device = DeviceInfo(
            name="R1",
            device_type="resistor",
            nodes=["in", "out"],
            parameters={"r": "1k"},
        )

        assert device.name == "R1"
        assert device.device_type == "resistor"
        assert device.nodes == ["in", "out"]
        assert device.parameters["r"] == "1k"


class TestClientError:
    """Tests for ClientError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ClientError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.code is None
        assert error.details == []

    def test_error_with_details(self):
        """Test error with code and details."""
        error = ClientError(
            "Parse error",
            code="PARSE_ERROR",
            details=["Line 5: Unknown device", "Line 10: Missing .end"],
        )
        assert str(error) == "Parse error"
        assert error.code == "PARSE_ERROR"
        assert len(error.details) == 2


class TestAcSweepType:
    """Tests for AcSweepType enum."""

    def test_values(self):
        """Test sweep type values."""
        assert AcSweepType.DEC.value == "dec"
        assert AcSweepType.OCT.value == "oct"
        assert AcSweepType.LIN.value == "lin"


class TestRustSpiceClient:
    """Tests for RustSpiceClient (without actual server)."""

    def test_initialization(self):
        """Test client initialization."""
        client = RustSpiceClient("http://localhost:3000")
        assert client.base_url == "http://localhost:3000"
        assert client.timeout == 60.0

    def test_initialization_with_trailing_slash(self):
        """Test that trailing slash is removed."""
        client = RustSpiceClient("http://localhost:3000/")
        assert client.base_url == "http://localhost:3000"

    def test_initialization_custom_timeout(self):
        """Test custom timeout."""
        client = RustSpiceClient("http://localhost:3000", timeout=120.0)
        assert client.timeout == 120.0
