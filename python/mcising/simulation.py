"""High-level simulation interface wrapping the Rust core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from mcising._core import IsingSimulation as _RustSim
from mcising.config import SimulationConfig
from mcising.constants import INF_TEMP
from mcising.exceptions import SimulationError

__all__: Final[list[str]] = [
    "Simulation",
    "SimulationResults",
]


@dataclass
class SimulationResults:
    """Container for simulation results across temperatures.

    Attributes
    ----------
    temperatures : list[float]
        Temperature values that were simulated.
    energy : dict[float, NDArray[np.float64]]
        Energy per site measurements at each temperature.
    magnetization : dict[float, NDArray[np.float64]]
        Magnetization per site measurements at each temperature.
    configurations : dict[float, NDArray[np.int8]]
        Spin configurations at each temperature. Shape: (n_samples, L, L).
    correlation_function : dict[float, tuple[NDArray, NDArray]] | None
        (distances, correlations) at each temperature, or None if not computed.
    correlation_length : dict[float, NDArray[np.float64]] | None
        Correlation length measurements at each temperature, or None.
    metadata : dict[str, object]
        Simulation metadata (config, timing, seed, etc.).
    """

    temperatures: list[float] = field(default_factory=list)
    energy: dict[float, NDArray[np.float64]] = field(default_factory=dict)
    magnetization: dict[float, NDArray[np.float64]] = field(default_factory=dict)
    configurations: dict[float, NDArray[np.int8]] = field(default_factory=dict)
    correlation_function: (
        dict[float, tuple[NDArray[np.float64], NDArray[np.float64]]] | None
    ) = None
    correlation_length: dict[float, NDArray[np.float64]] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class Simulation:
    """High-level interface to Ising model Monte Carlo simulation.

    Parameters
    ----------
    config : SimulationConfig
        Complete simulation configuration.

    Examples
    --------
    >>> from mcising import Simulation, SimulationConfig, LatticeConfig
    >>> config = SimulationConfig(
    ...     lattice=LatticeConfig(size=16, j1=1.0),
    ...     temperatures=(3.0, 2.269, 1.5),
    ...     n_sweeps=500,
    ... )
    >>> sim = Simulation(config)
    >>> results = sim.run()
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config: Final[SimulationConfig] = config
        self._core = _RustSim(
            lattice_size=config.lattice.size,
            j1=config.lattice.j1,
            j2=config.lattice.j2,
            h=config.lattice.h,
            seed=config.seed,
        )

    def run(self, *, show_progress: bool = True) -> SimulationResults:
        """Execute the full simulation across all temperatures.

        Temperatures are processed in descending order using a cool-down
        approach. The system is initialized at high temperature and gradually
        cooled to avoid metastable states.

        Parameters
        ----------
        show_progress : bool
            Whether to display a Rich progress bar.

        Returns
        -------
        SimulationResults
            Collected measurements across all temperatures.
        """
        import time

        start_time = time.monotonic()

        # Sort temperatures descending for cool-down
        sorted_temps = sorted(self.config.temperatures, reverse=True)

        results = SimulationResults(
            temperatures=sorted_temps,
            metadata={
                "config": self.config,
                "version": "0.2.0",
            },
        )
        if self.config.compute_correlation:
            results.correlation_function = {}
            results.correlation_length = {}

        # Prepend high temperature for initial thermalization
        temp_schedule = [INF_TEMP, *sorted_temps]

        progress_columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ]

        with Progress(*progress_columns, disable=not show_progress) as progress:
            task = progress.add_task("Temperature scan", total=len(sorted_temps))

            for i in range(len(sorted_temps)):
                from_temp = temp_schedule[i]
                to_temp = temp_schedule[i + 1]

                progress.update(
                    task,
                    description=f"T={to_temp:.3f} (thermalizing)",
                )

                # Thermalize
                self._thermalize(from_temp, to_temp, self.config.n_thermalization)

                progress.update(
                    task,
                    description=f"T={to_temp:.3f} (measuring)",
                )

                # Collect measurements
                self._collect_at_temperature(to_temp, results)

                progress.advance(task)

        elapsed = time.monotonic() - start_time
        results.metadata["elapsed_seconds"] = elapsed

        return results

    def sweep(self, temperature: float, n_sweeps: int = 1) -> dict[str, float]:
        """Perform sweeps at a given temperature and return observables.

        Parameters
        ----------
        temperature : float
            Simulation temperature (must be > 0).
        n_sweeps : int
            Number of MC sweeps to perform.

        Returns
        -------
        dict[str, float]
            Dictionary with keys 'energy', 'magnetization', 'acceptance_rate'.
        """
        if temperature <= 0:
            msg = f"Temperature must be positive, got {temperature}"
            raise SimulationError(msg)

        beta = 1.0 / temperature
        accepted, attempted = self._core.metropolis_sweep(n_sweeps, beta)

        return {
            "energy": self._core.energy(),
            "magnetization": self._core.magnetization(),
            "acceptance_rate": accepted / attempted if attempted > 0 else 0.0,
        }

    @property
    def spins(self) -> NDArray[np.int8]:
        """Current spin configuration as a 2D NumPy array."""
        return np.asarray(self._core.get_spins())

    @spins.setter
    def spins(self, value: NDArray[np.int8]) -> None:
        self._core.set_spins(value)

    @property
    def energy(self) -> float:
        """Current energy per site."""
        return float(self._core.energy())

    @property
    def magnetization(self) -> float:
        """Current magnetization per site."""
        return float(self._core.magnetization())

    def _thermalize(self, from_temp: float, to_temp: float, n_steps: int) -> None:
        """Gradually cool the system from from_temp to to_temp."""
        if n_steps <= 0:
            return

        temp_schedule = np.linspace(from_temp, to_temp, num=n_steps)
        for temp in temp_schedule:
            if temp <= 0:
                continue
            beta = 1.0 / float(temp)
            self._core.metropolis_sweep(1, beta)

    def _collect_at_temperature(
        self, temperature: float, results: SimulationResults
    ) -> None:
        """Run sweeps and collect measurements at a single temperature."""
        beta = 1.0 / temperature
        n_measurements = self.config.n_sweeps // self.config.measurement_interval

        energies = np.empty(n_measurements, dtype=np.float64)
        magnetizations = np.empty(n_measurements, dtype=np.float64)
        configs = np.empty(
            (n_measurements, self.config.lattice.size, self.config.lattice.size),
            dtype=np.int8,
        )

        corr_lengths: list[float] = []

        # Store one representative correlation function per temperature
        last_distances: NDArray[np.float64] | None = None
        last_correlations: NDArray[np.float64] | None = None

        for m in range(n_measurements):
            self._core.metropolis_sweep(self.config.measurement_interval, beta)

            energies[m] = self._core.energy()
            magnetizations[m] = self._core.magnetization()
            configs[m] = self._core.get_spins()

            if self.config.compute_correlation:
                distances, correlations = self._core.correlation_function()
                last_distances = np.asarray(distances)
                last_correlations = np.asarray(correlations)
                corr_lengths.append(self._core.correlation_length())

        results.energy[temperature] = energies
        results.magnetization[temperature] = magnetizations
        results.configurations[temperature] = configs

        if (
            self.config.compute_correlation
            and results.correlation_function is not None
            and results.correlation_length is not None
            and last_distances is not None
            and last_correlations is not None
        ):
            results.correlation_function[temperature] = (
                last_distances,
                last_correlations,
            )
            results.correlation_length[temperature] = np.array(
                corr_lengths, dtype=np.float64
            )
