"""Tests for SpiceClient HTTP client."""

import pytest
from unittest.mock import MagicMock, patch

from rustspice_agent.client import (
    SpiceClient,
    SpiceClientError,
    RunResult,
    AnalysisType,
    RunStatus,
    CircuitSummary,
)


class TestRunResult:
    """Tests for RunResult data class."""

    def test_from_dict_op(self):
        data = {
            "run_id": 0,
            "analysis": "Op",
            "status": "Converged",
            "iterations": 5,
            "nodes": ["0", "in", "out"],
            "solution": [0.0, 5.0, 3.33],
        }
        result = RunResult.from_dict(data)
        assert result.run_id == 0
        assert result.analysis == AnalysisType.OP
        assert result.status == RunStatus.CONVERGED
        assert result.iterations == 5
        assert result.node_names == ["0", "in", "out"]
        assert len(result.solution) == 3

    def test_from_dict_dc_sweep(self):
        data = {
            "run_id": 1,
            "analysis": "Dc",
            "status": "Converged",
            "iterations": 10,
            "nodes": ["0", "out"],
            "solution": [],
            "sweep_var": "V1",
            "sweep_values": [0.0, 1.0, 2.0],
            "sweep_solutions": [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]],
        }
        result = RunResult.from_dict(data)
        assert result.analysis == AnalysisType.DC
        assert result.sweep_var == "V1"
        assert len(result.sweep_values) == 3
        assert len(result.sweep_solutions) == 3


class TestCircuitSummary:
    """Tests for CircuitSummary data class."""

    def test_from_dict(self):
        data = {"node_count": 5, "device_count": 10, "model_count": 2}
        summary = CircuitSummary.from_dict(data)
        assert summary.node_count == 5
        assert summary.device_count == 10
        assert summary.model_count == 2


class TestSpiceClientMocked:
    """Tests for SpiceClient with mocked HTTP."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP."""
        with patch("rustspice_agent.client.httpx.Client") as mock_httpx:
            client = SpiceClient(base_url="http://localhost:3000")
            yield client, mock_httpx

    def test_ping_success(self, mock_client):
        client, mock_httpx = mock_client
        mock_response = MagicMock()
        mock_response.json.return_value = {"node_count": 0, "device_count": 0, "model_count": 0}
        mock_response.raise_for_status = MagicMock()
        client._client.request = MagicMock(return_value=mock_response)

        assert client.ping() is True

    def test_ping_failure(self, mock_client):
        client, mock_httpx = mock_client
        import httpx
        client._client.request = MagicMock(
            side_effect=httpx.RequestError("Connection refused")
        )

        assert client.ping() is False


class TestSpiceClientIntegration:
    """Integration tests (require running sim-api server)."""

    @pytest.fixture
    def client(self):
        """Create a real client (skipped if server not running)."""
        client = SpiceClient(base_url="http://localhost:3000", timeout=5.0)
        if not client.ping():
            pytest.skip("sim-api server not running")
        return client

    @pytest.mark.integration
    def test_run_op(self, client):
        netlist = """
V1 in 0 DC 5
R1 in out 1k
R2 out 0 2k
.op
.end
"""
        result = client.run_op(netlist=netlist)
        assert result.status == RunStatus.CONVERGED
        assert "out" in result.node_names

        # Check voltage divider result
        out_voltage = client.get_node_voltage(result, "out")
        assert out_voltage is not None
        assert abs(out_voltage - 3.333) < 0.01

    @pytest.mark.integration
    def test_get_summary(self, client):
        # First load a circuit
        netlist = "V1 in 0 DC 1\nR1 in 0 1k\n.op\n.end"
        client.run_op(netlist=netlist)

        summary = client.get_summary()
        assert summary.node_count >= 2
        assert summary.device_count >= 2
