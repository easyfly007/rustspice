"""HTTP client for RustSpice sim-api server."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx

from rustspice_agent.config import Config


class AnalysisType(str, Enum):
    """Simulation analysis types."""

    OP = "Op"
    DC = "Dc"
    TRAN = "Tran"
    AC = "Ac"


class RunStatus(str, Enum):
    """Simulation run status."""

    CONVERGED = "Converged"
    MAX_ITERS = "MaxIters"
    FAILED = "Failed"


@dataclass
class RunResult:
    """Result from a simulation run."""

    run_id: int
    analysis: AnalysisType
    status: RunStatus
    iterations: int
    node_names: list[str]
    solution: list[float] = field(default_factory=list)
    # DC sweep data
    sweep_var: Optional[str] = None
    sweep_values: list[float] = field(default_factory=list)
    sweep_solutions: list[list[float]] = field(default_factory=list)
    # TRAN data
    tran_times: list[float] = field(default_factory=list)
    tran_solutions: list[list[float]] = field(default_factory=list)
    # AC data
    ac_frequencies: list[float] = field(default_factory=list)
    ac_magnitude_db: list[list[float]] = field(default_factory=list)
    ac_phase_deg: list[list[float]] = field(default_factory=list)
    # Error info
    message: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunResult":
        """Create RunResult from API response dictionary."""
        # Parse AC solutions if present (list of [mag, phase] pairs per node)
        ac_mag = []
        ac_phase = []
        if "ac_solutions" in data and data["ac_solutions"]:
            # ac_solutions is [[mag, phase], ...] for each frequency, for each node
            # We need to transpose this
            for node_data in data.get("ac_solutions", []):
                mags = [point[0] for point in node_data]
                phases = [point[1] for point in node_data]
                ac_mag.append(mags)
                ac_phase.append(phases)

        return cls(
            run_id=data.get("run_id", 0),
            analysis=AnalysisType(data.get("analysis", "Op")),
            status=RunStatus(data.get("status", "Converged")),
            iterations=data.get("iterations", 0),
            node_names=data.get("nodes", data.get("node_names", [])),
            solution=data.get("solution", []),
            sweep_var=data.get("sweep_var"),
            sweep_values=data.get("sweep_values", []),
            sweep_solutions=data.get("sweep_solutions", []),
            tran_times=data.get("tran_times", []),
            tran_solutions=data.get("tran_solutions", []),
            ac_frequencies=data.get("ac_frequencies", []),
            ac_magnitude_db=ac_mag or data.get("magnitude_db", []),
            ac_phase_deg=ac_phase or data.get("phase_deg", []),
            message=data.get("message"),
        )


@dataclass
class CircuitSummary:
    """Summary information about a loaded circuit."""

    node_count: int
    device_count: int
    model_count: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CircuitSummary":
        return cls(
            node_count=data.get("node_count", 0),
            device_count=data.get("device_count", 0),
            model_count=data.get("model_count", 0),
        )


@dataclass
class DeviceInfo:
    """Information about a circuit device."""

    name: str
    device_type: str
    nodes: list[str]
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeviceInfo":
        return cls(
            name=data.get("name", ""),
            device_type=data.get("type", data.get("device_type", "")),
            nodes=data.get("nodes", []),
            parameters=data.get("parameters", data.get("params", {})),
        )


class SpiceClientError(Exception):
    """Exception raised by SpiceClient operations."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class SpiceClient:
    """HTTP client for communicating with RustSpice sim-api server.

    Example usage:
        client = SpiceClient()
        result = client.run_op('''
            V1 in 0 DC 5
            R1 in out 1k
            R2 out 0 2k
            .op
            .end
        ''')
        print(f"V(out) = {result.solution[result.node_names.index('out')]}")
    """

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        """Initialize the client.

        Args:
            base_url: API server URL (default: http://localhost:3000)
            timeout: Request timeout in seconds (default: 30)
        """
        config = Config.load()
        self.base_url = (base_url or config.api.url).rstrip("/")
        self.timeout = timeout or config.api.timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "SpiceClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make an HTTP request and return JSON response."""
        try:
            response = self._client.request(method, path, json=json, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            # Try to extract error message from response
            try:
                error_data = e.response.json()
                message = error_data.get("error", str(e))
            except Exception:
                message = str(e)
            raise SpiceClientError(message, e.response.status_code) from e
        except httpx.RequestError as e:
            raise SpiceClientError(f"Connection error: {e}") from e

    # =========================================================================
    # Circuit Information
    # =========================================================================

    def get_summary(self) -> CircuitSummary:
        """Get summary information about the loaded circuit."""
        data = self._request("GET", "/v1/summary")
        return CircuitSummary.from_dict(data)

    def get_nodes(self) -> list[str]:
        """Get list of all nodes in the circuit."""
        data = self._request("GET", "/v1/nodes")
        return data.get("nodes", [])

    def get_devices(self) -> list[DeviceInfo]:
        """Get list of all devices in the circuit."""
        data = self._request("GET", "/v1/devices")
        return [DeviceInfo.from_dict(d) for d in data.get("devices", [])]

    # =========================================================================
    # Simulation Execution
    # =========================================================================

    def run_op(
        self,
        netlist: Optional[str] = None,
        path: Optional[str] = None,
    ) -> RunResult:
        """Run DC operating point analysis.

        Args:
            netlist: SPICE netlist text
            path: Path to netlist file (alternative to netlist)

        Returns:
            RunResult with node voltages at DC operating point
        """
        payload: dict[str, Any] = {}
        if netlist:
            payload["netlist"] = netlist
        if path:
            payload["path"] = path

        data = self._request("POST", "/v1/run/op", json=payload)
        return RunResult.from_dict(data)

    def run_dc(
        self,
        source: str,
        start: float,
        stop: float,
        step: float,
        netlist: Optional[str] = None,
        path: Optional[str] = None,
    ) -> RunResult:
        """Run DC sweep analysis.

        Args:
            source: Name of source to sweep (e.g., 'V1')
            start: Start value
            stop: Stop value
            step: Step size
            netlist: SPICE netlist text
            path: Path to netlist file

        Returns:
            RunResult with sweep data
        """
        payload: dict[str, Any] = {
            "source": source,
            "start": start,
            "stop": stop,
            "step": step,
        }
        if netlist:
            payload["netlist"] = netlist
        if path:
            payload["path"] = path

        data = self._request("POST", "/v1/run/dc", json=payload)
        return RunResult.from_dict(data)

    def run_tran(
        self,
        tstop: float,
        tstep: Optional[float] = None,
        tstart: float = 0.0,
        netlist: Optional[str] = None,
        path: Optional[str] = None,
    ) -> RunResult:
        """Run transient analysis.

        Args:
            tstop: Stop time in seconds
            tstep: Output time step (default: tstop/100)
            tstart: Start time (default: 0)
            netlist: SPICE netlist text
            path: Path to netlist file

        Returns:
            RunResult with time-domain waveforms
        """
        payload: dict[str, Any] = {
            "tstop": tstop,
            "tstart": tstart,
        }
        if tstep is not None:
            payload["tstep"] = tstep
        if netlist:
            payload["netlist"] = netlist
        if path:
            payload["path"] = path

        data = self._request("POST", "/v1/run/tran", json=payload)
        return RunResult.from_dict(data)

    def run_ac(
        self,
        fstart: float,
        fstop: float,
        points: int = 10,
        sweep: str = "dec",
        netlist: Optional[str] = None,
        path: Optional[str] = None,
    ) -> RunResult:
        """Run AC small-signal analysis.

        Args:
            fstart: Start frequency in Hz
            fstop: Stop frequency in Hz
            points: Points per decade/octave or total points
            sweep: Sweep type ('dec', 'oct', or 'lin')
            netlist: SPICE netlist text
            path: Path to netlist file

        Returns:
            RunResult with frequency response data
        """
        payload: dict[str, Any] = {
            "fstart": fstart,
            "fstop": fstop,
            "points": points,
            "sweep": sweep,
        }
        if netlist:
            payload["netlist"] = netlist
        if path:
            payload["path"] = path

        data = self._request("POST", "/v1/run/ac", json=payload)
        return RunResult.from_dict(data)

    # =========================================================================
    # Result Access
    # =========================================================================

    def get_runs(self) -> list[dict[str, Any]]:
        """Get list of all simulation runs."""
        data = self._request("GET", "/v1/runs")
        return data.get("runs", [])

    def get_run(self, run_id: int) -> RunResult:
        """Get a specific simulation run by ID."""
        data = self._request("GET", f"/v1/runs/{run_id}")
        return RunResult.from_dict(data)

    def get_waveform(
        self,
        run_id: int,
        signal: str,
    ) -> dict[str, Any]:
        """Get waveform data for a specific signal.

        Args:
            run_id: Simulation run ID
            signal: Signal name (e.g., 'V(out)', 'I(R1)')

        Returns:
            Dictionary with x_values, y_values, and metadata
        """
        data = self._request("GET", f"/v1/runs/{run_id}/waveform", params={"signal": signal})
        return data

    def export_run(
        self,
        run_id: int,
        path: str,
        format: str = "psf",
    ) -> None:
        """Export simulation results to file.

        Args:
            run_id: Simulation run ID
            path: Output file path
            format: Output format ('psf', 'csv', 'json', 'raw')
        """
        self._request(
            "POST",
            f"/v1/runs/{run_id}/export",
            json={"path": path, "format": format},
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def ping(self) -> bool:
        """Check if the server is reachable."""
        try:
            self._request("GET", "/v1/summary")
            return True
        except SpiceClientError:
            return False

    def get_node_voltage(self, result: RunResult, node: str) -> Optional[float]:
        """Get voltage at a specific node from simulation result.

        Args:
            result: Simulation result
            node: Node name

        Returns:
            Voltage value or None if node not found
        """
        try:
            idx = result.node_names.index(node)
            return result.solution[idx]
        except (ValueError, IndexError):
            return None
