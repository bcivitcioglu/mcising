"""Custom exception hierarchy for mcising."""

from typing import Final

__all__: Final[list[str]] = [
    "MCIsingError",
    "ConfigurationError",
    "SimulationError",
]


class MCIsingError(Exception):
    """Base exception for all mcising errors."""


class ConfigurationError(MCIsingError, ValueError):
    """Raised when simulation configuration is invalid.

    Also a ``ValueError``: pre-1.0 config validation raised plain
    ``ValueError``, and the Rust core still surfaces its boundary
    errors as ``ValueError`` — so ``except ValueError`` catches every
    invalid-configuration error from both layers, while
    ``except ConfigurationError`` selects the Python-side validation
    specifically.
    """


class SimulationError(MCIsingError):
    """Raised when a simulation encounters an error during execution."""
