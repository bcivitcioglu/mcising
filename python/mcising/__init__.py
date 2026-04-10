"""
mcising: High-performance Ising model Monte Carlo simulation.

A Rust-accelerated Python library for Monte Carlo simulation of Ising spin
systems on various lattice geometries with multiple update algorithms.
"""

from typing import Final

from mcising._core import IsingSimulation
from mcising.config import (
    AdaptiveConfig,
    Algorithm,
    ExecutionMode,
    LatticeConfig,
    LatticeType,
    SimulationConfig,
)
from mcising.exceptions import ConfigurationError, MCIsingError, SimulationError
from mcising.io import checkpoint_run, load_hdf5, save_hdf5, save_json_summary
from mcising.plotting import (
    export_lattices,
    plot_correlation,
    plot_energy,
    plot_energy_timeseries,
    plot_lattice,
    plot_magnetization,
    plot_magnetization_histogram,
    plot_observables,
    plot_specific_heat,
    plot_susceptibility,
)
from mcising.simulation import Simulation, SimulationResults

__version__: Final[str] = "0.2.0"

__all__: Final[list[str]] = [
    # Core Rust binding
    "IsingSimulation",
    # High-level API
    "Simulation",
    "SimulationResults",
    # Configuration
    "SimulationConfig",
    "LatticeConfig",
    "LatticeType",
    "Algorithm",
    "ExecutionMode",
    "AdaptiveConfig",
    # I/O
    "save_hdf5",
    "load_hdf5",
    "save_json_summary",
    "checkpoint_run",
    # Plotting
    "plot_energy",
    "plot_magnetization",
    "plot_specific_heat",
    "plot_susceptibility",
    "plot_lattice",
    "plot_correlation",
    "plot_energy_timeseries",
    "plot_magnetization_histogram",
    "export_lattices",
    "plot_observables",
    # Exceptions
    "MCIsingError",
    "ConfigurationError",
    "SimulationError",
]
