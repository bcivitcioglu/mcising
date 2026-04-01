"""High-level simulation interface wrapping the Rust core."""

from __future__ import annotations

from collections.abc import Callable
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
    "AdaptiveDiagnostics",
]


@dataclass
class AdaptiveDiagnostics:
    """Per-temperature diagnostics from the adaptive measurement scheme.

    Attributes
    ----------
    thermalization_sweeps : int
        Total thermalization sweeps used (cool-down + extension).
    truncation_point : int
        MSER truncation point in the thermalization energy series.
    is_thermalized : bool
        Whether the series was detected as stationary.
    tau_int : float
        Estimated integrated autocorrelation time.
    measurement_interval : int
        Measurement interval used for production (tau_multiplier * tau_int).
    production_sweeps : int
        Total production sweeps used.
    n_samples : int
        Number of measurement samples collected.
    """

    thermalization_sweeps: int = 0
    truncation_point: int = 0
    is_thermalized: bool = True
    tau_int: float = 0.5
    measurement_interval: int = 1
    production_sweeps: int = 0
    n_samples: int = 0


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
    adaptive_diagnostics: dict[float, AdaptiveDiagnostics] | None = None
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

    def run(
        self,
        *,
        show_progress: bool = True,
        on_temperature_complete: (
            Callable[[float, SimulationResults], None] | None
        ) = None,
        skip_temperatures: frozenset[float] | None = None,
    ) -> SimulationResults:
        """Execute the full simulation across all temperatures.

        Temperatures are processed in descending order using a cool-down
        approach. The system is initialized at high temperature and gradually
        cooled to avoid metastable states.

        Parameters
        ----------
        show_progress : bool
            Whether to display a Rich progress bar.
        on_temperature_complete : callable, optional
            Called with ``(temperature, results)`` after each temperature
            finishes data collection. Useful for checkpointing.
        skip_temperatures : frozenset[float], optional
            Temperatures to skip (e.g. already checkpointed). Skipped
            temperatures are excluded from the thermalization schedule.

        Returns
        -------
        SimulationResults
            Collected measurements across all temperatures.
        """
        import time

        start_time = time.monotonic()

        # Sort temperatures descending for cool-down
        sorted_temps = sorted(self.config.temperatures, reverse=True)

        # Build effective schedule excluding skipped temperatures
        effective_temps = [
            t for t in sorted_temps if t not in (skip_temperatures or frozenset())
        ]

        results = SimulationResults(
            temperatures=list(effective_temps),
            metadata={
                "config": self.config,
                "version": "0.2.0",
            },
        )
        if self.config.compute_correlation:
            results.correlation_function = {}
            results.correlation_length = {}

        adaptive = self.config.adaptive.enabled
        if adaptive:
            results.adaptive_diagnostics = {}

        # Prepend high temperature for initial thermalization
        temp_schedule = [INF_TEMP, *effective_temps]

        progress_columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ]

        n_skipped = len(sorted_temps) - len(effective_temps)

        with Progress(*progress_columns, disable=not show_progress) as progress:
            task = progress.add_task("Temperature scan", total=len(sorted_temps))

            # Pre-advance for skipped temperatures
            if n_skipped > 0:
                progress.advance(task, advance=n_skipped)

            for i in range(len(effective_temps)):
                from_temp = temp_schedule[i]
                to_temp = temp_schedule[i + 1]

                progress.update(
                    task,
                    description=f"T={to_temp:.3f} (thermalizing)",
                )

                if adaptive:
                    # Adaptive: thermalize with diagnostics, then adaptive collection
                    self._thermalize_adaptive(from_temp, to_temp, results)

                    progress.update(
                        task,
                        description=f"T={to_temp:.3f} (measuring)",
                    )

                    self._collect_at_temperature_adaptive(to_temp, results)
                else:
                    # Fixed: original behavior
                    self._thermalize(from_temp, to_temp, self.config.n_thermalization)

                    progress.update(
                        task,
                        description=f"T={to_temp:.3f} (measuring)",
                    )

                    self._collect_at_temperature(to_temp, results)

                if on_temperature_complete is not None:
                    on_temperature_complete(to_temp, results)

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

    def _thermalize_adaptive(
        self,
        from_temp: float,
        to_temp: float,
        results: SimulationResults,
    ) -> None:
        """Adaptive thermalization: cool-down with energy recording + MSER check."""
        ac = self.config.adaptive
        n_therm = max(self.config.n_thermalization, ac.min_thermalization_sweeps)

        # Cool-down phase: linspace from from_temp to to_temp, record energy
        temp_schedule = np.linspace(from_temp, to_temp, num=n_therm).tolist()
        energy_series = np.asarray(
            self._core.thermalize_with_diagnostics(temp_schedule)
        )
        total_therm = len(energy_series)

        # Analyze: MSER + Sokal on the cool-down energy series
        analysis = _RustSim.analyze_thermalization_series(
            energy_series, ac.c_window, ac.tau_multiplier
        )

        # If not thermalized, extend with sweeps at target temperature
        beta = 1.0 / to_temp
        while (
            not analysis["is_thermalized"]
            and total_therm < ac.max_thermalization_sweeps
        ):
            extra_n = min(
                n_therm,
                ac.max_thermalization_sweeps - total_therm,
            )
            extra_energies = np.asarray(
                self._core.extend_thermalization(extra_n, beta)
            )
            energy_series = np.concatenate([energy_series, extra_energies])
            total_therm += extra_n

            analysis = _RustSim.analyze_thermalization_series(
                energy_series, ac.c_window, ac.tau_multiplier
            )

        # Store diagnostics (production filled by _collect_adaptive)
        if results.adaptive_diagnostics is not None:
            results.adaptive_diagnostics[to_temp] = AdaptiveDiagnostics(
                thermalization_sweeps=total_therm,
                truncation_point=int(analysis["truncation_point"]),
                is_thermalized=bool(analysis["is_thermalized"]),
                tau_int=float(analysis["tau_int"]),
                measurement_interval=int(analysis["recommended_interval"]),
            )

    def _collect_at_temperature_adaptive(
        self, temperature: float, results: SimulationResults
    ) -> None:
        """Adaptive production: use tau_int to set measurement spacing."""
        ac = self.config.adaptive
        beta = 1.0 / temperature

        # Get the interval from diagnostics
        diag = (
            results.adaptive_diagnostics.get(temperature)
            if results.adaptive_diagnostics is not None
            else None
        )
        interval = diag.measurement_interval if diag else 1
        interval = max(1, interval)

        # Calculate number of production measurements
        n_measurements = ac.min_independent_samples

        # Enforce total sweep budget
        therm_used = diag.thermalization_sweeps if diag else 0
        remaining_budget = ac.max_total_sweeps - therm_used
        max_measurements = max(1, remaining_budget // interval)
        n_measurements = min(n_measurements, max_measurements)

        # Single Rust call for all production measurements
        energies, magnetizations, configs = self._core.production_sweeps(
            n_measurements, interval, beta, True
        )

        results.energy[temperature] = np.asarray(energies)
        results.magnetization[temperature] = np.asarray(magnetizations)
        results.configurations[temperature] = np.asarray(configs)

        # Update diagnostics with production info
        if diag is not None:
            diag.production_sweeps = n_measurements * interval
            diag.n_samples = n_measurements

        # Correlation function (computed after production, using final config)
        if self.config.compute_correlation:
            distances, correlations = self._core.correlation_function()
            if results.correlation_function is not None:
                results.correlation_function[temperature] = (
                    np.asarray(distances),
                    np.asarray(correlations),
                )
            if results.correlation_length is not None:
                results.correlation_length[temperature] = np.array(
                    [self._core.correlation_length()], dtype=np.float64
                )
