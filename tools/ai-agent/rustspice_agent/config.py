"""Configuration management for RustSpice AI Agent."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Try to import tomllib (Python 3.11+) or tomli as fallback
try:
    import tomllib
except ImportError:
    tomllib = None  # type: ignore


@dataclass
class ApiConfig:
    """API connection configuration."""

    url: str = "http://localhost:3000"
    timeout: float = 30.0


@dataclass
class AiConfig:
    """AI/LLM configuration."""

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.3
    api_key: Optional[str] = None


@dataclass
class OutputConfig:
    """Output formatting configuration."""

    precision: int = 6
    format: str = "table"  # table, csv, json


@dataclass
class Config:
    """Main configuration container."""

    api: ApiConfig = field(default_factory=ApiConfig)
    ai: AiConfig = field(default_factory=AiConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from file and environment variables.

        Priority (highest to lowest):
        1. Environment variables
        2. Config file
        3. Default values
        """
        config = cls()

        # Try to load from config file
        if config_path is None:
            config_path = Path.home() / ".rustspice" / "config.toml"

        if config_path.exists() and tomllib is not None:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
                config._apply_dict(data)

        # Override with environment variables
        config._apply_env()

        return config

    def _apply_dict(self, data: dict) -> None:
        """Apply configuration from dictionary."""
        if "api" in data:
            if "url" in data["api"]:
                self.api.url = data["api"]["url"]
            if "timeout" in data["api"]:
                self.api.timeout = float(data["api"]["timeout"])

        if "ai" in data:
            if "model" in data["ai"]:
                self.ai.model = data["ai"]["model"]
            if "max_tokens" in data["ai"]:
                self.ai.max_tokens = int(data["ai"]["max_tokens"])
            if "temperature" in data["ai"]:
                self.ai.temperature = float(data["ai"]["temperature"])

        if "output" in data:
            if "precision" in data["output"]:
                self.output.precision = int(data["output"]["precision"])
            if "format" in data["output"]:
                self.output.format = data["output"]["format"]

    def _apply_env(self) -> None:
        """Apply configuration from environment variables."""
        # API settings
        if url := os.environ.get("RUSTSPICE_API_URL"):
            self.api.url = url
        if timeout := os.environ.get("RUSTSPICE_TIMEOUT"):
            self.api.timeout = float(timeout)

        # AI settings
        if model := os.environ.get("RUSTSPICE_MODEL"):
            self.ai.model = model
        if api_key := os.environ.get("ANTHROPIC_API_KEY"):
            self.ai.api_key = api_key

        # Output settings
        if precision := os.environ.get("RUSTSPICE_PRECISION"):
            self.output.precision = int(precision)
