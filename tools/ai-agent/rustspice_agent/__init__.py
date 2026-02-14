"""RustSpice AI Agent - AI-powered interface for circuit simulation."""

__version__ = "0.1.0"

from rustspice_agent.client import SpiceClient
from rustspice_agent.config import Config

__all__ = ["SpiceClient", "Config", "__version__"]
