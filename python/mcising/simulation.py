"""High-level simulation interface wrapping the Rust core."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

import mcising.statistics as _mcstats
from mcising._core import IsingSimulation as _RustSim
from mcising._core import run_independent_temperatures as _run_independent
from mcising._core import run_parallel_tempering as _run_pt
from mcising._provenance import HDF5_SCHEMA_VERSION, git_commit, package_version
from mcising.config import ExecutionMode, SimulationConfig
from mcising.constants import INF_TEMP
from mcising.exceptions import ConfigurationError, SimulationError
from mcising.statistics import ObservableStatistics

__all__: Final[list[str]] = [
    "Simulation",
    "SimulationResults",
    "AdaptiveDiagnostics",
]


def _fill_results_from_raw(
    raw: list[dict[str, Any]], results: SimulationResults
) -> None:
    """Copy per-temperature arrays from Rust runner dicts into results."""
    for entry in raw:
        temp = float(entry["temperature"])
        results.energy[temp] = np.asarray(entry["energies"])
        results.magnetization[temp] = np.asarray(entry["magnetizations"])
        if "configurations" in entry:
            results.configurations[temp] = np.asarray(entry["configurations"])
        if (
            "correlation_function" in entry
            and results.correlation_function is not None
            and results.correlation_length is not None
        ):
            results.correlation_function[temp] = (
                np.asarray(entry["correlation_distances"]),
                np.asarray(entry["correlation_function"]),
            )
            results.correlation_length[temp] = np.asarray(entry["correlation_length"])


def _run_metadata(config: SimulationConfig) -> dict[str, object]:
    """Provenance metadata every run path stamps into its results (B12)."""
    metadata: dict[str, object] = {
        "config": config,
        "version": package_version(),
        "schema_version": HDF5_SCHEMA_VERSION,
        "seed": config.seed,
        "mode": config.mode.value,
        "algorithm": config.algorithm.value,
    }
    commit = git_commit()
    if commit is not None:
        metadata["git_commit"] = commit
    return metadata


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
        Provenance and timing: the ``SimulationConfig`` object under
        ``"config"``, plus ``version``, ``schema_version``, ``seed``,
        ``mode``, ``algorithm``, optional ``git_commit``, and
        ``elapsed_seconds``. Loaded legacy files (mcising <= 0.23.0)
        carry whatever subset their file recorded.
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
    _statistics_cache: dict[float, ObservableStatistics] = field(
        default_factory=dict, repr=False, compare=False
    )

    @cached_property
    def num_sites(self) -> int:
        """Total number of spins N, resolved from provenance.

        Prefers the restored :class:`~mcising.config.SimulationConfig`
        (present for every run and every loaded file with provenance);
        falls back to the shape of stored spin configurations for
        config-less legacy files. Never guesses: a wrong N silently
        mis-scales specific heat and susceptibility (B11).

        Raises
        ------
        ConfigurationError
            If neither the config metadata nor stored configurations
            can supply the site count.
        """
        n = self._num_sites_or_none()
        if n is None:
            msg = (
                "Cannot determine the number of lattice sites: results "
                "carry no config metadata and no stored spin "
                "configurations. Re-save the file with mcising >= 0.24.0 "
                "or attach a config to results.metadata['config']."
            )
            raise ConfigurationError(msg)
        return n

    def _num_sites_or_none(self) -> int | None:
        """Non-raising twin of :attr:`num_sites` for display paths."""
        config = self.metadata.get("config")
        lattice = getattr(config, "lattice", None)
        if lattice is not None:
            return int(lattice.num_sites)
        for cfg in self.configurations.values():
            if cfg.ndim >= 2:
                return int(np.prod(cfg.shape[1:]))
        return None

    def statistics(self, temperature: float) -> ObservableStatistics:
        """Observable estimates with standard errors at one temperature.

        Computed lazily from the stored measurement series and memoized.
        Means (E, M, |M|) carry blocking standard errors; specific heat,
        susceptibility, and the Binder cumulant carry delete-one-block
        jackknife errors (see :mod:`mcising.statistics`). Total: a
        temperature with no or degenerate data yields ``nan`` estimates
        rather than an exception.

        Parameters
        ----------
        temperature : float
            Temperature to compute statistics at.

        Returns
        -------
        ObservableStatistics
            Per-temperature estimates; errors are ``nan`` where the
            series is too short to quote a principled uncertainty.
        """
        cached = self._statistics_cache.get(temperature)
        if cached is None:
            cached = _mcstats.observable_statistics(
                temperature,
                self.energy.get(temperature, ()),
                self.magnetization.get(temperature, ()),
                self._num_sites_or_none(),
            )
            self._statistics_cache[temperature] = cached
        return cached

    def specific_heat(self, temperature: float) -> float:
        """Specific heat per site: Cv = N * Var(E) / T^2.

        Parameters
        ----------
        temperature : float
            Temperature to compute Cv at.

        Returns
        -------
        float
            Specific heat per site. For the standard error use
            ``statistics(temperature).specific_heat.error``.
        """
        return _mcstats.specific_heat(
            self.energy[temperature],
            temperature=temperature,
            num_sites=self.num_sites,
        )

    def susceptibility(self, temperature: float) -> float:
        """Magnetic susceptibility per site: chi = N * Var(M) / T.

        Uses the signed magnetization series (see
        :func:`mcising.statistics.susceptibility` for the convention
        and its ordered-phase caveat).

        Parameters
        ----------
        temperature : float
            Temperature to compute chi at.

        Returns
        -------
        float
            Susceptibility per site. For the standard error use
            ``statistics(temperature).susceptibility.error``.
        """
        return _mcstats.susceptibility(
            self.magnetization[temperature],
            temperature=temperature,
            num_sites=self.num_sites,
        )

    def binder_cumulant(self, temperature: float) -> float:
        """Binder cumulant U4 = 1 - <m^4> / (3 <m^2>^2).

        Parameters
        ----------
        temperature : float
            Temperature to compute U4 at.

        Returns
        -------
        float
            Binder cumulant (dimensionless; no site count needed). For
            the standard error use
            ``statistics(temperature).binder_cumulant.error``.
        """
        return _mcstats.binder_cumulant(self.magnetization[temperature])

    def summary(self) -> None:
        """Print a Rich table summarizing results per temperature.

        Shows mean energy, magnetization, specific heat,
        susceptibility, and Binder cumulant with standard errors
        (``value ± error``; ``n/a`` where the series is too short to
        quote one).
        """
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Simulation Results", border_style="blue")
        table.add_column("T", justify="right", style="bold")
        table.add_column("<E>/N", justify="right")
        table.add_column("<|M|>/N", justify="right")
        table.add_column("Cv/N", justify="right")
        table.add_column("chi/N", justify="right")
        table.add_column("U4", justify="right")
        table.add_column("samples", justify="right", style="dim")

        for t in sorted(self.temperatures):
            if t not in self.energy:
                continue
            stats = self.statistics(t)
            table.add_row(
                f"{t:.4f}",
                str(stats.energy),
                str(stats.abs_magnetization),
                str(stats.specific_heat),
                str(stats.susceptibility),
                str(stats.binder_cumulant),
                str(stats.n_samples),
            )

        Console().print(table)

    def to_dataframe(self) -> object:
        """Convert results to a pandas DataFrame.

        Returns a DataFrame with columns: T, E_mean, E_err, E_std,
        M_mean, M_err, M_std, Cv, Cv_err, chi, chi_err, U4, U4_err,
        tau_int, samples. The ``*_err`` columns are standard errors of
        the mean/estimator (blocking for means, jackknife for derived
        quantities); ``E_std``/``M_std`` remain the sample spread of
        the series (not an uncertainty). Errors are ``nan`` where the
        series is too short to quote one.

        Returns
        -------
        pandas.DataFrame
            Summary statistics per temperature.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        import pandas as pd  # type: ignore[import-untyped]

        rows = []
        for t in sorted(self.temperatures):
            if t not in self.energy:
                continue
            stats = self.statistics(t)
            rows.append(
                {
                    "T": t,
                    "E_mean": stats.energy.value,
                    "E_err": stats.energy.error,
                    "E_std": float(np.std(self.energy[t])),
                    "M_mean": stats.abs_magnetization.value,
                    "M_err": stats.abs_magnetization.error,
                    "M_std": float(np.std(self.magnetization[t])),
                    "Cv": stats.specific_heat.value,
                    "Cv_err": stats.specific_heat.error,
                    "chi": stats.susceptibility.value,
                    "chi_err": stats.susceptibility.error,
                    "U4": stats.binder_cumulant.value,
                    "U4_err": stats.binder_cumulant.error,
                    "tau_int": stats.tau_int,
                    "samples": stats.n_samples,
                }
            )
        return pd.DataFrame(rows)


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
            j3=config.lattice.j3,
            h=config.lattice.h,
            seed=config.seed,
            algorithm=config.algorithm.value,
            lattice_type=config.lattice.lattice_type.value,
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

        Behavior depends on ``config.mode``:

        - **COOLDOWN** (default): Temperatures processed sequentially in
          descending order. Spins carried from high T to low T.
        - **INDEPENDENT**: Each temperature runs in parallel from random
          initialization. Uses all CPU cores via Rayon.
        - **PARALLEL_TEMPERING**: All temperatures run as one coupled
          replica-exchange ensemble with periodic swap attempts.

        Parameters
        ----------
        show_progress : bool
            Whether to display a Rich progress bar.
        on_temperature_complete : callable, optional
            Called once per completed temperature. In cooldown mode it
            fires as each temperature finishes; in the parallel modes the
            batch computes every temperature first and the callback then
            fires once per temperature.
        skip_temperatures : frozenset[float], optional
            Temperatures to leave out of this run (e.g. already completed
            in a checkpoint). In independent mode the remaining
            temperatures keep the RNG streams they would have had in a
            full run. In parallel tempering only skipping every
            temperature (or none) is allowed — the replicas form one
            coupled ensemble.

        Returns
        -------
        SimulationResults
            Collected measurements across all temperatures.

        Raises
        ------
        ConfigurationError
            In parallel tempering, when ``skip_temperatures`` covers a
            non-empty proper subset of the configured temperatures.
        """
        if self.config.mode == ExecutionMode.INDEPENDENT:
            return self._run_independent(
                show_progress=show_progress,
                on_temperature_complete=on_temperature_complete,
                skip_temperatures=skip_temperatures,
            )

        if self.config.mode == ExecutionMode.PARALLEL_TEMPERING:
            return self._run_parallel_tempering(
                show_progress=show_progress,
                on_temperature_complete=on_temperature_complete,
                skip_temperatures=skip_temperatures,
            )

        return self._run_cooldown(
            show_progress=show_progress,
            on_temperature_complete=on_temperature_complete,
            skip_temperatures=skip_temperatures,
        )

    def _run_independent(
        self,
        *,
        show_progress: bool = True,
        on_temperature_complete: (
            Callable[[float, SimulationResults], None] | None
        ) = None,
        skip_temperatures: frozenset[float] | None = None,
    ) -> SimulationResults:
        """Run all temperatures in parallel via Rayon.

        ``skip_temperatures`` filters the batch; each surviving temperature
        keeps the seed offset it has in the full configured scan, so a
        resumed run reproduces the uninterrupted run's RNG streams. The
        callback fires once per temperature after the batch returns.
        """
        import time

        start_time = time.monotonic()
        skip = skip_temperatures or frozenset()
        temps: list[float] = []
        seed_offsets: list[int] = []
        for i, temp in enumerate(self.config.temperatures):
            if temp not in skip:
                temps.append(temp)
                seed_offsets.append(i)

        results = SimulationResults(
            temperatures=temps,
            metadata=_run_metadata(self.config),
        )
        if self.config.compute_correlation:
            results.correlation_function = {}
            results.correlation_length = {}

        if not temps:
            results.metadata["elapsed_seconds"] = time.monotonic() - start_time
            return results

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            disable=not show_progress,
        ) as progress:
            progress.add_task(
                f"Running {len(temps)} temperatures in parallel...",
                total=None,
            )

            raw = _run_independent(
                lattice_size=self.config.lattice.size,
                j1=self.config.lattice.j1,
                j2=self.config.lattice.j2,
                j3=self.config.lattice.j3,
                h=self.config.lattice.h,
                base_seed=self.config.seed,
                algorithm=self.config.algorithm.value,
                lattice_type=self.config.lattice.lattice_type.value,
                temperatures=temps,
                n_thermalization=self.config.n_thermalization,
                n_sweeps=self.config.n_sweeps,
                measurement_interval=self.config.measurement_interval,
                store_configs=self.config.store_configs,
                compute_correlation=self.config.compute_correlation,
                seed_offsets=seed_offsets,
            )

        _fill_results_from_raw(raw, results)

        if on_temperature_complete is not None:
            for temp in temps:
                on_temperature_complete(temp, results)

        elapsed = time.monotonic() - start_time
        results.metadata["elapsed_seconds"] = elapsed

        return results

    def _run_parallel_tempering(
        self,
        *,
        show_progress: bool = True,
        on_temperature_complete: (
            Callable[[float, SimulationResults], None] | None
        ) = None,
        skip_temperatures: frozenset[float] | None = None,
    ) -> SimulationResults:
        """Run Parallel Tempering via Rayon.

        The replicas form one coupled ensemble, so ``skip_temperatures``
        is all-or-nothing: skipping every configured temperature returns
        empty results (nothing left to do), skipping a proper subset
        raises. The callback fires once per temperature after the run.
        """
        import time

        start_time = time.monotonic()
        skip = skip_temperatures or frozenset()
        all_temps = list(self.config.temperatures)
        temps = [t for t in all_temps if t not in skip]

        results = SimulationResults(
            temperatures=sorted(temps),
            metadata=_run_metadata(self.config),
        )
        if self.config.compute_correlation:
            results.correlation_function = {}
            results.correlation_length = {}

        if not temps:
            results.metadata["elapsed_seconds"] = time.monotonic() - start_time
            return results

        if len(temps) != len(all_temps):
            # Replica exchange couples every temperature into one ensemble:
            # removing completed rungs would change the swap ladder and its
            # dynamics for every remaining replica.
            raise ConfigurationError(
                "Parallel tempering cannot resume a partially completed "
                "temperature ladder: the replicas form one coupled ensemble. "
                "Delete the checkpoint to rerun the full ladder, or use "
                "mode='independent' for per-temperature resume. Got "
                f"{len(all_temps) - len(temps)} completed of "
                f"{len(all_temps)} temperatures."
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            disable=not show_progress,
        ) as progress:
            progress.add_task(
                f"Parallel Tempering: {len(temps)} replicas...",
                total=None,
            )

            raw = _run_pt(
                lattice_size=self.config.lattice.size,
                j1=self.config.lattice.j1,
                j2=self.config.lattice.j2,
                j3=self.config.lattice.j3,
                h=self.config.lattice.h,
                base_seed=self.config.seed,
                algorithm=self.config.algorithm.value,
                lattice_type=self.config.lattice.lattice_type.value,
                temperatures=temps,
                n_thermalization=self.config.n_thermalization,
                n_sweeps=self.config.n_sweeps,
                measurement_interval=self.config.measurement_interval,
                swap_interval=self.config.swap_interval,
                store_configs=self.config.store_configs,
                compute_correlation=self.config.compute_correlation,
            )

        _fill_results_from_raw(raw, results)

        if on_temperature_complete is not None:
            for temp in sorted(temps):
                on_temperature_complete(temp, results)

        elapsed = time.monotonic() - start_time
        results.metadata["elapsed_seconds"] = elapsed

        return results

    def _run_cooldown(
        self,
        *,
        show_progress: bool = True,
        on_temperature_complete: (
            Callable[[float, SimulationResults], None] | None
        ) = None,
        skip_temperatures: frozenset[float] | None = None,
    ) -> SimulationResults:
        """Run temperatures sequentially via cool-down (original behavior)."""
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
            metadata=_run_metadata(self.config),
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
                    self._thermalize_adaptive(from_temp, to_temp, results)
                    progress.update(
                        task,
                        description=f"T={to_temp:.3f} (measuring)",
                    )
                    self._collect_at_temperature_adaptive(to_temp, results)
                else:
                    self._thermalize(
                        from_temp,
                        to_temp,
                        self.config.n_thermalization,
                    )
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
        accepted, attempted = self._core.sweep(n_sweeps, beta)

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
            self._core.sweep(1, beta)

    def _collect_at_temperature(
        self, temperature: float, results: SimulationResults
    ) -> None:
        """Run sweeps and collect measurements at a single temperature."""
        beta = 1.0 / temperature
        n_measurements = self.config.n_sweeps // self.config.measurement_interval

        energies = np.empty(n_measurements, dtype=np.float64)
        magnetizations = np.empty(n_measurements, dtype=np.float64)
        configs: NDArray[np.int8] | None = None
        if self.config.store_configs:
            spin_shape = self._core.get_spins().shape
            configs = np.empty(
                (n_measurements, *spin_shape),
                dtype=np.int8,
            )

        corr_lengths: list[float] = []

        # Store one representative correlation function per temperature
        last_distances: NDArray[np.float64] | None = None
        last_correlations: NDArray[np.float64] | None = None

        for m in range(n_measurements):
            self._core.sweep(self.config.measurement_interval, beta)

            energies[m] = self._core.energy()
            magnetizations[m] = self._core.magnetization()
            if configs is not None:
                configs[m] = self._core.get_spins()

            if self.config.compute_correlation:
                distances, correlations = self._core.correlation_function()
                last_distances = np.asarray(distances)
                last_correlations = np.asarray(correlations)
                corr_lengths.append(self._core.correlation_length())

        results.energy[temperature] = energies
        results.magnetization[temperature] = magnetizations
        if configs is not None:
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
            extra_energies = np.asarray(self._core.extend_thermalization(extra_n, beta))
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
            n_measurements, interval, beta, self.config.store_configs
        )

        results.energy[temperature] = np.asarray(energies)
        results.magnetization[temperature] = np.asarray(magnetizations)
        if configs is not None:
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
