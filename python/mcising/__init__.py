"""
mcising: High-performance Ising model Monte Carlo simulation.

A Rust-accelerated Python library for Monte Carlo simulation of Ising spin
systems on various lattice geometries with multiple update algorithms.
"""

from typing import TYPE_CHECKING, Any, Final

from mcising._core import IsingSimulation
from mcising._provenance import package_version
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
from mcising.simulation import Simulation, SimulationResults
from mcising.statistics import Estimate, ObservableStatistics

if TYPE_CHECKING:
    from mcising.plotting import (
        export_lattices,
        plot_correlation,
        plot_energy,
        plot_energy_timeseries,
        plot_lattice,
        plot_magnetization,
        plot_magnetization_histogram,
        plot_specific_heat,
        plot_susceptibility,
    )

__version__: Final[str] = package_version()

#: Names served lazily from mcising.plotting (PEP 562) so that
#: ``import mcising`` never imports matplotlib — matplotlib is the
#: optional ``plot`` extra since 0.26.0.
_PLOTTING_EXPORTS: Final[frozenset[str]] = frozenset(
    {
        "plot_energy",
        "plot_magnetization",
        "plot_specific_heat",
        "plot_susceptibility",
        "plot_lattice",
        "plot_correlation",
        "plot_energy_timeseries",
        "plot_magnetization_histogram",
        "export_lattices",
    }
)


def __getattr__(name: str) -> Any:
    if name in _PLOTTING_EXPORTS:
        from mcising import plotting

        return getattr(plotting, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


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
    # Statistics
    "Estimate",
    "ObservableStatistics",
    # I/O
    "save_hdf5",
    "load_hdf5",
    "save_json_summary",
    "checkpoint_run",
    # Plotting (lazy; requires the `plot` extra)
    "plot_energy",
    "plot_magnetization",
    "plot_specific_heat",
    "plot_susceptibility",
    "plot_lattice",
    "plot_correlation",
    "plot_energy_timeseries",
    "plot_magnetization_histogram",
    "export_lattices",
    # Exceptions
    "MCIsingError",
    "ConfigurationError",
    "SimulationError",
]
