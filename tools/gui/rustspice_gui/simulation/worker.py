"""
Async simulation worker using QThread.

Provides non-blocking simulation execution with progress reporting
and cancellation support.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any, Dict

from PySide6.QtCore import QThread, Signal, QObject

from ..client import RustSpiceClient, RunResult, AcSweepType, ClientError


class AnalysisType(Enum):
    """Analysis types for simulation."""
    OP = "op"
    DC = "dc"
    TRAN = "tran"
    AC = "ac"


@dataclass
class SimulationTask:
    """
    Represents a simulation task to be executed.

    Attributes:
        analysis: Type of analysis (op, dc, tran, ac)
        netlist: The netlist text content
        params: Analysis-specific parameters
    """
    analysis: AnalysisType
    netlist: str
    params: Dict[str, Any]


class SimulationWorker(QThread):
    """
    Worker thread for running simulations asynchronously.

    This worker runs simulations in a separate thread to keep the UI responsive.
    It provides signals for progress updates, completion, and error handling.

    Signals:
        started: Emitted when simulation starts
        progress: Emitted with progress message
        finished: Emitted with RunResult when simulation completes successfully
        error: Emitted with error message when simulation fails
        stopped: Emitted when simulation is stopped by user
    """

    # Signals
    simulation_started = Signal(str)  # analysis type
    progress = Signal(str)  # progress message
    finished = Signal(object)  # RunResult
    error = Signal(str, list, str)  # error message, details, error code
    stopped = Signal()

    def __init__(self, client: RustSpiceClient, parent: Optional[QObject] = None):
        """
        Initialize the worker.

        Args:
            client: The RustSpiceClient for server communication
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self._client = client
        self._task: Optional[SimulationTask] = None
        self._stop_requested = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_task(self, task: SimulationTask):
        """
        Set the simulation task to execute.

        Args:
            task: The SimulationTask to run
        """
        self._task = task
        self._stop_requested = False

    def request_stop(self):
        """Request the simulation to stop."""
        self._stop_requested = True

    def run(self):
        """Execute the simulation in the worker thread."""
        if self._task is None:
            self.error.emit("No simulation task set", [], "")
            return

        # Create new event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self.simulation_started.emit(self._task.analysis.value.upper())
            self.progress.emit(f"Starting {self._task.analysis.value.upper()} analysis...")

            # Run the appropriate analysis
            result = self._loop.run_until_complete(self._run_analysis())

            if self._stop_requested:
                self.stopped.emit()
            else:
                self.finished.emit(result)

        except ClientError as e:
            self.error.emit(str(e), e.details or [], e.code or "")
        except Exception as e:
            self.error.emit(str(e), [], "")
        finally:
            self._loop.close()
            self._loop = None

    async def _run_analysis(self) -> RunResult:
        """Run the analysis based on task type."""
        task = self._task
        netlist = task.netlist
        params = task.params

        if task.analysis == AnalysisType.OP:
            self.progress.emit("Running operating point analysis...")
            return await self._client.run_op(netlist=netlist)

        elif task.analysis == AnalysisType.DC:
            self.progress.emit(f"Running DC sweep: {params['source']}...")
            return await self._client.run_dc(
                netlist=netlist,
                source=params["source"],
                start=params["start"],
                stop=params["stop"],
                step=params["step"]
            )

        elif task.analysis == AnalysisType.TRAN:
            self.progress.emit(f"Running transient analysis to {params['tstop']}s...")
            return await self._client.run_tran(
                netlist=netlist,
                tstep=params["tstep"],
                tstop=params["tstop"],
                tstart=params.get("tstart", 0.0),
                tmax=params.get("tmax")
            )

        elif task.analysis == AnalysisType.AC:
            self.progress.emit(f"Running AC analysis {params['fstart']}Hz to {params['fstop']}Hz...")
            return await self._client.run_ac(
                netlist=netlist,
                sweep=params["sweep"],
                points=params["points"],
                fstart=params["fstart"],
                fstop=params["fstop"]
            )

        else:
            raise ValueError(f"Unknown analysis type: {task.analysis}")


class ConnectionChecker(QThread):
    """
    Worker thread for checking server connection.

    Signals:
        result: Emitted with connection status (bool)
    """

    result = Signal(bool)

    def __init__(self, client: RustSpiceClient, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._client = client

    def run(self):
        """Check the connection in worker thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            connected = loop.run_until_complete(self._client.check_connection())
            self.result.emit(connected)
        except Exception:
            self.result.emit(False)
        finally:
            loop.close()
