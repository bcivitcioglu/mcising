"""Frozen dataclass configurations for Ising model simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from mcising.constants import (
    DEFAULT_ADAPTIVE_C_WINDOW,
    DEFAULT_ADAPTIVE_MAX_THERMALIZATION,
    DEFAULT_ADAPTIVE_MAX_TOTAL_SWEEPS,
    DEFAULT_ADAPTIVE_MIN_INDEPENDENT_SAMPLES,
    DEFAULT_ADAPTIVE_MIN_THERMALIZATION,
    DEFAULT_ADAPTIVE_TAU_MULTIPLIER,
    DEFAULT_H,
    DEFAULT_J1,
    DEFAULT_J2,
    DEFAULT_J3,
    DEFAULT_LATTICE_SIZE,
    DEFAULT_MEASUREMENT_INTERVAL,
    DEFAULT_N_SWEEPS,
    DEFAULT_N_THERMALIZATION,
    DEFAULT_SEED,
)
from mcising.exceptions import ConfigurationError

__all__: Final[list[str]] = [
    "LatticeType",
    "Algorithm",
    "LatticeConfig",
    "AdaptiveConfig",
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
    WOLFF = "wolff"
    SWENDSEN_WANG = "swendsen_wang"
    # Future algorithms:
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
    j3 : float
        Third-nearest-neighbor coupling strength.
    h : float
        External magnetic field.
    """

    lattice_type: LatticeType = LatticeType.SQUARE
    size: int = DEFAULT_LATTICE_SIZE
    j1: float = DEFAULT_J1
    j2: float = DEFAULT_J2
    j3: float = DEFAULT_J3
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
        if not isinstance(self.j3, (int, float)) or not _is_finite(self.j3):
            msg = f"j3 must be a finite number, got {self.j3}"
            raise ValueError(msg)
        if not isinstance(self.h, (int, float)) or not _is_finite(self.h):
            msg = f"h must be a finite number, got {self.h}"
            raise ValueError(msg)


@dataclass(frozen=True)
class AdaptiveConfig:
    """Configuration for adaptive thermalization and measurement spacing.

    When enabled, the simulation records energy during thermalization sweeps,
    uses MSER to verify equilibration, estimates the integrated autocorrelation
    time (tau_int) via Sokal's windowing method, and sets the measurement
    interval to ``tau_multiplier * tau_int`` for independent samples.

    Parameters
    ----------
    enabled : bool
        Whether to use adaptive mode. When False (default), the simulation
        uses fixed n_sweeps / measurement_interval / n_thermalization.
    min_thermalization_sweeps : int
        Minimum thermalization sweeps per temperature (including cool-down).
    max_thermalization_sweeps : int
        Maximum thermalization sweeps (cap to prevent runaway near T_c).
    c_window : float
        Sokal windowing constant for tau_int estimation.
    min_independent_samples : int
        Target number of effectively independent samples per temperature.
    max_total_sweeps : int
        Hard cap on total sweeps per temperature (thermalization + production).
    tau_multiplier : float
        Measurement interval = tau_multiplier * tau_int.
        Using 2*tau gives ~86% independence between consecutive samples.
    """

    enabled: bool = False
    min_thermalization_sweeps: int = DEFAULT_ADAPTIVE_MIN_THERMALIZATION
    max_thermalization_sweeps: int = DEFAULT_ADAPTIVE_MAX_THERMALIZATION
    c_window: float = DEFAULT_ADAPTIVE_C_WINDOW
    min_independent_samples: int = DEFAULT_ADAPTIVE_MIN_INDEPENDENT_SAMPLES
    max_total_sweeps: int = DEFAULT_ADAPTIVE_MAX_TOTAL_SWEEPS
    tau_multiplier: float = DEFAULT_ADAPTIVE_TAU_MULTIPLIER

    def __post_init__(self) -> None:
        if self.min_thermalization_sweeps < 1:
            msg = (
                "min_thermalization_sweeps must be >= 1, "
                f"got {self.min_thermalization_sweeps}"
            )
            raise ValueError(msg)
        if self.max_thermalization_sweeps < self.min_thermalization_sweeps:
            msg = (
                f"max_thermalization_sweeps "
                f"({self.max_thermalization_sweeps}) must be >= "
                f"min ({self.min_thermalization_sweeps})"
            )
            raise ValueError(msg)
        if self.c_window <= 0:
            msg = f"c_window must be > 0, got {self.c_window}"
            raise ValueError(msg)
        if self.min_independent_samples < 1:
            msg = (
                "min_independent_samples must be >= 1, "
                f"got {self.min_independent_samples}"
            )
            raise ValueError(msg)
        if self.tau_multiplier <= 0:
            msg = f"tau_multiplier must be > 0, got {self.tau_multiplier}"
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
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)

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
        if self.algorithm in (Algorithm.WOLFF, Algorithm.SWENDSEN_WANG):
            has_frustration = (
                self.lattice.j2 != 0.0
                or self.lattice.j3 != 0.0
                or self.lattice.h != 0.0
            )
            if has_frustration:
                raise ConfigurationError(
                    "Cluster algorithms require J2=0, J3=0, and h=0. "
                    "Use algorithm='metropolis' for J1-J2, J1-J3, or "
                    "external field simulations."
                )


def _is_finite(value: float) -> bool:
    """Check if a float is finite (not inf, -inf, or nan)."""
    import math

    return math.isfinite(value)
