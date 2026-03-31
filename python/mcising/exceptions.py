"""Custom exception hierarchy for mcising."""

from typing import Final

__all__: Final[list[str]] = [
    "MCIsingError",
    "ConfigurationError",
    "SimulationError",
]


class MCIsingError(Exception):
    """Base exception for all mcising errors."""


class ConfigurationError(MCIsingError):
    """Raised when simulation configuration is invalid."""


class SimulationError(MCIsingError):
    """Raised when a simulation encounters an error during execution."""