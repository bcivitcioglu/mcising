"""Frozen dataclass configurations for Ising model simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from mcising.constants import (
    DEFAULT_H,
    DEFAULT_J1,
    DEFAULT_J2,
    DEFAULT_LATTICE_SIZE,
    DEFAULT_MEASUREMENT_INTERVAL,
    DEFAULT_N_SWEEPS,
    DEFAULT_N_THERMALIZATION,
    DEFAULT_SEED,
)

__all__: Final[list[str]] = [
    "LatticeType",
    "Algorithm",
    "LatticeConfig",
    "SimulationConfig",
]


class LatticeType(str, Enum):
    """Available lattice geometries."""

    SQUARE = "square"
    # Future lattice types:
    # TRIANGULAR = "triangular"
    # HONEYCOMB = "honeycomb"
    # KAGOME = "kagome"
    # CUBIC = "cubic"
    # CHAIN = "chain"


class Algorithm(str, Enum):
    """Available Monte Carlo update algorithms."""

    METROPOLIS = "metropolis"
    # Future algorithms:
    # WOLFF = "wolff"
    # SWENDSEN_WANG = "swendsen_wang"
    # WANG_LANDAU = "wang_landau"
    # PARALLEL_TEMPERING = "parallel_tempering"


@dataclass(frozen=True)
class LatticeConfig:
    """Configuration for lattice geometry.

    Parameters
    ----------
    lattice_type : LatticeType
        Type of lattice geometry.
    size : int
        Linear size L of the lattice (creates L x L for 2D).
    j1 : float
        Nearest-neighbor coupling strength.
    j2 : float
        Next-nearest-neighbor coupling strength.
    h : float
        External magnetic field.
    """

    lattice_type: LatticeType = LatticeType.SQUARE
    size: int = DEFAULT_LATTICE_SIZE
    j1: float = DEFAULT_J1
    j2: float = DEFAULT_J2
    h: float = DEFAULT_H

    def __post_init__(self) -> None:
        if self.size < 2:
            msg = f"Lattice size must be >= 2, got {self.size}"
            raise ValueError(msg)
        if not isinstance(self.j1, (int, float)) or not _is_finite(self.j1):
            msg = f"j1 must be a finite number, got {self.j1}"
            raise ValueError(msg)
        if not isinstance(self.j2, (int, float)) or not _is_finite(self.j2):
            msg = f"j2 must be a finite number, got {self.j2}"
            raise ValueError(msg)
        if not isinstance(self.h, (int, float)) or not _is_finite(self.h):
            msg = f"h must be a finite number, got {self.h}"
            raise ValueError(msg)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a Monte Carlo simulation run.

    Parameters
    ----------
    lattice : LatticeConfig
        Lattice geometry and coupling parameters.
    algorithm : Algorithm
        Monte Carlo update algorithm to use.
    seed : int
        Random seed for reproducibility.
    temperatures : tuple[float, ...]
        Temperatures to simulate at (in descending order for cool-down).
    n_sweeps : int
        Number of MC sweeps per temperature point.
    n_thermalization : int
        Number of thermalization sweeps before measurement.
    measurement_interval : int
        Collect a measurement every this many sweeps.
    compute_correlation : bool
        Whether to compute the correlation function.
    """

    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    algorithm: Algorithm = Algorithm.METROPOLIS
    seed: int = DEFAULT_SEED
    temperatures: tuple[float, ...] = (2.269,)
    n_sweeps: int = DEFAULT_N_SWEEPS
    n_thermalization: int = DEFAULT_N_THERMALIZATION
    measurement_interval: int = DEFAULT_MEASUREMENT_INTERVAL
    compute_correlation: bool = False

    def __post_init__(self) -> None:
        if self.n_sweeps < 1:
            msg = f"n_sweeps must be >= 1, got {self.n_sweeps}"
            raise ValueError(msg)
        if self.n_thermalization < 0:
            msg = f"n_thermalization must be >= 0, got {self.n_thermalization}"
            raise ValueError(msg)
        if self.measurement_interval < 1:
            msg = f"measurement_interval must be >= 1, got {self.measurement_interval}"
            raise ValueError(msg)
        for temp in self.temperatures:
            if temp <= 0 or not _is_finite(temp):
                msg = f"All temperatures must be positive and finite, got {temp}"
                raise ValueError(msg)
        if len(self.temperatures) == 0:
            msg = "At least one temperature must be specified"
            raise ValueError(msg)


def _is_finite(value: float) -> bool:
    """Check if a float is finite (not inf, -inf, or nan)."""
    import math

    return math.isfinite(value)
