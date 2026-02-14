"""
HTTP client for communicating with sim-api server.

This module provides an async HTTP client for running simulations
and retrieving results from the RustSpice sim-api server.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import httpx


class AnalysisType(Enum):
    """Simulation analysis types."""
    OP = "Op"
    DC = "Dc"
    TRAN = "Tran"
    AC = "Ac"


class AcSweepType(Enum):
    """AC sweep types."""
    DEC = "dec"
    OCT = "oct"
    LIN = "lin"


@dataclass
class WaveformData:
    """Waveform data from a simulation."""
    signal: str
    analysis: str
    x_label: str
    x_unit: str
    y_label: str
    y_unit: str
    x_values: List[float]
    y_values: List[float]


@dataclass
class RunResult:
    """Result from a simulation run."""
    run_id: int
    analysis: AnalysisType
    status: str
    iterations: int
    nodes: List[str]
    solution: List[float]
    message: Optional[str] = None

    # DC sweep data
    sweep_var: Optional[str] = None
    sweep_values: List[float] = field(default_factory=list)
    sweep_solutions: List[List[float]] = field(default_factory=list)

    # Transient data
    tran_times: List[float] = field(default_factory=list)
    tran_solutions: List[List[float]] = field(default_factory=list)

    # AC data
    ac_frequencies: List[float] = field(default_factory=list)
    ac_solutions: List[List[List[float]]] = field(default_factory=list)  # [freq][node][mag, phase]

    @property
    def dc_values(self) -> Dict[str, List[float]]:
        """DC sweep solutions as {node_name: [value_per_sweep_point]}."""
        if not self.sweep_solutions or not self.nodes:
            return {}
        result = {}
        for i, node in enumerate(self.nodes):
            result[node] = [sol[i] for sol in self.sweep_solutions if i < len(sol)]
        return result

    @property
    def tran_values(self) -> Dict[str, List[float]]:
        """Transient solutions as {node_name: [value_per_time_point]}."""
        if not self.tran_solutions or not self.nodes:
            return {}
        result = {}
        for i, node in enumerate(self.nodes):
            result[node] = [sol[i] for sol in self.tran_solutions if i < len(sol)]
        return result

    @property
    def ac_values(self) -> Dict[str, List]:
        """AC solutions as {node_name: [[mag, phase] per frequency]}."""
        if not self.ac_solutions or not self.nodes:
            return {}
        result = {}
        for i, node in enumerate(self.nodes):
            result[node] = [sol[i] for sol in self.ac_solutions if i < len(sol)]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "RunResult":
        """Create RunResult from API response dictionary."""
        analysis_str = data.get("analysis", "Op")
        try:
            analysis = AnalysisType(analysis_str)
        except ValueError:
            analysis = AnalysisType.OP

        return cls(
            run_id=data.get("run_id", 0),
            analysis=analysis,
            status=data.get("status", "Unknown"),
            iterations=data.get("iterations", 0),
            nodes=data.get("nodes", []),
            solution=data.get("solution", []),
            message=data.get("message"),
            sweep_var=data.get("sweep_var"),
            sweep_values=data.get("sweep_values", []),
            sweep_solutions=data.get("sweep_solutions", []),
            tran_times=data.get("tran_times", []),
            tran_solutions=data.get("tran_solutions", []),
            ac_frequencies=data.get("ac_frequencies", []),
            ac_solutions=data.get("ac_solutions", []),
        )


@dataclass
class CircuitSummary:
    """Circuit summary information."""
    node_count: int
    device_count: int
    model_count: int


@dataclass
class DeviceInfo:
    """Device information."""
    name: str
    device_type: str
    nodes: List[str]
    parameters: Dict[str, str]


class ClientError(Exception):
    """Exception raised for client errors."""
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[List[str]] = None):
        super().__init__(message)
        self.code = code
        self.details = details or []


class RustSpiceClient:
    """
    Async HTTP client for sim-api server.

    Usage:
        client = RustSpiceClient("http://127.0.0.1:3000")
        result = await client.run_op(netlist_text)
        waveform = await client.get_waveform(result.run_id, "V(out)")
    """

    def __init__(self, base_url: str = "http://127.0.0.1:3000", timeout: float = 60.0):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the sim-api server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _handle_error(self, response: httpx.Response):
        """Handle error responses from the server."""
        if response.status_code >= 400:
            try:
                data = response.json()
                error = data.get("error", {})
                raise ClientError(
                    message=error.get("message", f"HTTP {response.status_code}"),
                    code=error.get("code"),
                    details=error.get("details"),
                )
            except (ValueError, KeyError):
                raise ClientError(f"HTTP {response.status_code}: {response.text}")

    async def check_connection(self) -> bool:
        """Check if the server is reachable."""
        try:
            client = await self._get_client()
            response = await client.get("/v1/runs")
            return response.status_code == 200
        except Exception:
            return False

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    async def run_op(
        self,
        netlist: Optional[str] = None,
        path: Optional[str] = None,
    ) -> RunResult:
        """
        Run operating point analysis.

        Args:
            netlist: Netlist text content
            path: Path to netlist file (alternative to netlist)

        Returns:
            RunResult with DC solution
        """
        client = await self._get_client()
        payload = {}
        if netlist:
            payload["netlist"] = netlist
        if path:
            payload["path"] = path

        response = await client.post("/v1/run/op", json=payload)
        self._handle_error(response)
        return RunResult.from_dict(response.json())

    async def run_dc(
        self,
        source: str,
        start: float,
        stop: float,
        step: float,
        netlist: Optional[str] = None,
        path: Optional[str] = None,
    ) -> RunResult:
        """
        Run DC sweep analysis.

        Args:
            source: Name of the source to sweep (e.g., "V1")
            start: Start value
            stop: Stop value
            step: Step size
            netlist: Netlist text content
            path: Path to netlist file

        Returns:
            RunResult with sweep data
        """
        client = await self._get_client()
        payload = {
            "source": source,
            "start": start,
            "stop": stop,
            "step": step,
        }
        if netlist:
            payload["netlist"] = netlist
        if path:
            payload["path"] = path

        response = await client.post("/v1/run/dc", json=payload)
        self._handle_error(response)
        return RunResult.from_dict(response.json())

    async def run_tran(
        self,
        tstep: float,
        tstop: float,
        tstart: float = 0.0,
        tmax: Optional[float] = None,
        netlist: Optional[str] = None,
        path: Optional[str] = None,
    ) -> RunResult:
        """
        Run transient analysis.

        Args:
            tstep: Suggested time step
            tstop: Stop time
            tstart: Start time (default: 0)
            tmax: Maximum time step (optional)
            netlist: Netlist text content
            path: Path to netlist file

        Returns:
            RunResult with transient waveforms
        """
        client = await self._get_client()
        payload = {
            "tstep": tstep,
            "tstop": tstop,
            "tstart": tstart,
        }
        if tmax is not None:
            payload["tmax"] = tmax
        if netlist:
            payload["netlist"] = netlist
        if path:
            payload["path"] = path

        response = await client.post("/v1/run/tran", json=payload)
        self._handle_error(response)
        return RunResult.from_dict(response.json())

    async def run_ac(
        self,
        sweep: AcSweepType,
        points: int,
        fstart: float,
        fstop: float,
        netlist: Optional[str] = None,
        path: Optional[str] = None,
    ) -> RunResult:
        """
        Run AC frequency analysis.

        Args:
            sweep: Sweep type (DEC, OCT, or LIN)
            points: Number of points (per decade/octave for DEC/OCT, total for LIN)
            fstart: Start frequency in Hz
            fstop: Stop frequency in Hz
            netlist: Netlist text content
            path: Path to netlist file

        Returns:
            RunResult with AC data (magnitude in dB, phase in degrees)
        """
        client = await self._get_client()
        payload = {
            "sweep": sweep.value,
            "points": points,
            "fstart": fstart,
            "fstop": fstop,
        }
        if netlist:
            payload["netlist"] = netlist
        if path:
            payload["path"] = path

        response = await client.post("/v1/run/ac", json=payload)
        self._handle_error(response)
        return RunResult.from_dict(response.json())

    # =========================================================================
    # Results Methods
    # =========================================================================

    async def list_runs(self) -> List[Dict[str, Any]]:
        """
        List all simulation runs.

        Returns:
            List of run summaries
        """
        client = await self._get_client()
        response = await client.get("/v1/runs")
        self._handle_error(response)
        return response.json().get("runs", [])

    async def get_run(self, run_id: int) -> RunResult:
        """
        Get a specific run result.

        Args:
            run_id: ID of the run to retrieve

        Returns:
            RunResult with full data
        """
        client = await self._get_client()
        response = await client.get(f"/v1/runs/{run_id}")
        self._handle_error(response)
        return RunResult.from_dict(response.json())

    async def get_waveform(self, run_id: int, signal: str) -> WaveformData:
        """
        Get waveform data for a specific signal.

        Args:
            run_id: ID of the run
            signal: Signal name (e.g., "V(out)" or "out")

        Returns:
            WaveformData with x and y values
        """
        client = await self._get_client()
        response = await client.get(
            f"/v1/runs/{run_id}/waveform",
            params={"signal": signal}
        )
        self._handle_error(response)
        data = response.json()
        return WaveformData(
            signal=data["signal"],
            analysis=data["analysis"],
            x_label=data["x_label"],
            x_unit=data["x_unit"],
            y_label=data["y_label"],
            y_unit=data["y_unit"],
            x_values=data["x_values"],
            y_values=data["y_values"],
        )

    async def export_run(self, run_id: int, path: str, format: Optional[str] = None) -> bool:
        """
        Export a run to file.

        Args:
            run_id: ID of the run to export
            path: Output file path
            format: Output format (optional)

        Returns:
            True if successful
        """
        client = await self._get_client()
        payload = {"path": path}
        if format:
            payload["format"] = format

        response = await client.post(f"/v1/runs/{run_id}/export", json=payload)
        self._handle_error(response)
        return True

    # =========================================================================
    # Circuit Information Methods
    # =========================================================================

    async def get_summary(self) -> CircuitSummary:
        """
        Get circuit summary.

        Returns:
            CircuitSummary with counts
        """
        client = await self._get_client()
        response = await client.get("/v1/summary")
        self._handle_error(response)
        data = response.json()
        return CircuitSummary(
            node_count=data.get("node_count", 0),
            device_count=data.get("device_count", 0),
            model_count=data.get("model_count", 0),
        )

    async def get_nodes(self) -> List[str]:
        """
        Get list of circuit nodes.

        Returns:
            List of node names
        """
        client = await self._get_client()
        response = await client.get("/v1/nodes")
        self._handle_error(response)
        return response.json().get("nodes", [])

    async def get_devices(self) -> List[DeviceInfo]:
        """
        Get list of circuit devices.

        Returns:
            List of DeviceInfo objects
        """
        client = await self._get_client()
        response = await client.get("/v1/devices")
        self._handle_error(response)
        devices = []
        for d in response.json().get("devices", []):
            devices.append(DeviceInfo(
                name=d["name"],
                device_type=d["device_type"],
                nodes=d["nodes"],
                parameters=d.get("parameters", {}),
            ))
        return devices
