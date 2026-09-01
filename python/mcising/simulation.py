"""High-level simulation interface wrapping the Rust core."""

from __future__ import annotations

import warnings
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
from mcising.config import AdaptiveConfig, ExecutionMode, SimulationConfig
from mcising.constants import (
    DEFAULT_MEASUREMENT_INTERVAL,
    DEFAULT_N_SWEEPS,
    INF_TEMP,
    MIN_DIAGNOSTIC_SWEEPS,
)
from mcising.exceptions import ConfigurationError, SimulationError
from mcising.statistics import ObservableStatistics

__all__: Final[list[str]] = [
    "Simulation",
    "SimulationResults",
    "AdaptiveDiagnostics",
]


def _fill_results_entry(
    temp: float, entry: dict[str, Any], results: SimulationResults
) -> None:
    """Copy one per-temperature Rust measurement dict into ``results``.

    ``temp`` is the caller's key object (the configured temperature), so the
    dict's float round-trip never changes key identity. Correlation data is
    recorded only when at least one evaluation happened — a run shorter than
    one ``correlation_interval`` leaves the entry absent, as a run shorter
    than one measurement always has.
    """
    results.energy[temp] = np.asarray(entry["energies"])
    results.magnetization[temp] = np.asarray(entry["magnetizations"])
    if "n_cluster_flips" in entry:
        results.n_cluster_flips[temp] = int(entry["n_cluster_flips"])
    if "configurations" in entry:
        results.configurations[temp] = np.asarray(entry["configurations"])
    if (
        "correlation_function" in entry
        and results.correlation_function is not None
        and results.correlation_length is not None
        and len(entry["correlation_length"]) > 0
    ):
        results.correlation_function[temp] = (
            np.asarray(entry["correlation_distances"]),
            np.asarray(entry["correlation_function"]),
        )
        results.correlation_length[temp] = np.asarray(entry["correlation_length"])


def _fill_results_from_raw(
    raw: list[dict[str, Any]], results: SimulationResults
) -> None:
    """Copy per-temperature arrays from Rust runner dicts into results."""
    for entry in raw:
        _fill_results_entry(float(entry["temperature"]), entry, results)


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
        Total thermalization sweeps used (annealing ramp + all
        fixed-temperature diagnostic sweeps).
    stationary_sweeps : int
        Fixed-temperature sweeps analyzed for stationarity and tau_int.
        Never includes the cool-down ramp: the ramp's energy trace is
        non-stationary by construction and is excluded from all
        estimation (B9, #20).
    truncation_point : int
        MSER truncation point within the fixed-temperature series.
    is_thermalized : bool
        Whether the fixed-temperature series was detected as stationary.
    tau_int : float
        Integrated autocorrelation time, estimated on the stationary
        tail of the fixed-temperature series only.
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
    stationary_sweeps: int = 0


def _analyze_thermalization(
    series: NDArray[np.float64], config: AdaptiveConfig
) -> dict[str, Any]:
    """Run MSER + Sokal analysis on a fixed-temperature energy series.

    Module-level seam over the static Rust method: PyO3 classes reject
    attribute assignment, so tests spy on the analyzed series by
    monkeypatching this function rather than ``_RustSim``.
    """
    return _RustSim.analyze_thermalization_series(
        series, config.c_window, config.tau_multiplier
    )


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
        Spin configurations at each temperature, shape ``(n_samples,
        *lattice_shape)``: ``(L, L)`` for the square and triangular
        lattices, ``(L, L, 2)`` for the honeycomb, ``(L, L, L)`` for the
        cubic lattice and ``(L,)`` for the chain.
    correlation_function : dict[float, tuple[NDArray, NDArray]] | None
        (distances, correlations) at each temperature, or None if not computed.
    correlation_length : dict[float, NDArray[np.float64]] | None
        Correlation length measurements at each temperature, or None.
    n_cluster_flips : dict[float, int]
        Cluster flips during the measurement sweeps at each temperature
        (thermalization excluded). 0 for Metropolis; for Wolff this is
        the number of cluster updates (one per sweep), the honest work
        record behind the ``n_sweeps`` accounting.
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
    n_cluster_flips: dict[float, int] = field(default_factory=dict)
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

    def susceptibility(
        self,
        temperature: float,
        *,
        kind: _mcstats.SusceptibilityKind = "connected",
    ) -> float:
        """Magnetic susceptibility per site.

        The default connected convention is
        ``chi = N * (<m**2> - <|m|>**2) / T`` (standard for finite-size
        scaling; breaking default since P10, #39). ``kind="signed"``
        selects the pre-1.0 ``N * Var(m) / T`` form — see
        :func:`mcising.statistics.susceptibility`.

        Parameters
        ----------
        temperature : float
            Temperature to compute chi at.
        kind : {"connected", "signed"}
            Susceptibility convention.

        Returns
        -------
        float
            Susceptibility per site. For the standard error (connected
            convention) use ``statistics(temperature).susceptibility.error``.
        """
        return _mcstats.susceptibility(
            self.magnetization[temperature],
            temperature=temperature,
            num_sites=self.num_sites,
            kind=kind,
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
    >>> results = sim.run(show_progress=False)
    >>> sorted(results.energy) == [1.5, 2.269, 3.0]
    True
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config: Final[SimulationConfig] = config
        self._core = self._build_core()

    def _build_core(self) -> _RustSim:
        """Construct a fresh Rust core from the configuration.

        The single construction site: ``__init__`` and ``reset()`` both
        go through here, so a reset core is bit-identical to a freshly
        constructed one (same seed, same initial spins).
        """
        config = self.config
        return _RustSim(
            lattice_size=config.lattice.size,
            j1=config.lattice.j1,
            j2=config.lattice.j2,
            j3=config.lattice.j3,
            h=config.lattice.h,
            seed=config.seed,
            algorithm=config.algorithm.value,
            lattice_type=config.lattice.lattice_type.value,
        )

    def reset(self) -> None:
        """Discard all evolved state and return to the initial condition.

        The spin configuration and RNG stream are restored to exactly
        what a fresh ``Simulation(config)`` starts with; manual
        ``sweep()`` calls and ``spins`` assignments are forgotten.
        """
        self._core = self._build_core()

    def run(
        self,
        *,
        reset: bool = True,
        show_progress: bool = True,
        on_temperature_complete: (
            Callable[[float, SimulationResults], None] | None
        ) = None,
        skip_temperatures: frozenset[float] | None = None,
    ) -> SimulationResults:
        """Execute the full simulation across all temperatures.

        ``run()`` first resets the simulation to its deterministic
        initial condition (see ``reset()``), so repeated calls on the
        same object return identical results, and any prior manual
        ``sweep()`` or ``spins`` assignment has no effect on the run.
        Pass ``reset=False`` to continue from the current core state
        instead (checkpoint resume uses this).

        Behavior depends on ``config.mode``:

        - **COOLDOWN** (default): Temperatures processed sequentially in
          descending order. Spins carried from high T to low T.
        - **INDEPENDENT**: Each temperature runs in parallel from random
          initialization. Uses all CPU cores via Rayon.
        - **PARALLEL_TEMPERING**: All temperatures run as one coupled
          replica-exchange ensemble with periodic swap attempts.

        Parameters
        ----------
        reset : bool
            When True (default), rebuild the core from the configuration
            before running. When False, keep the current spins and RNG
            state (only meaningful in cooldown mode — the parallel modes
            build fresh Rust replicas per call regardless).
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
        if reset:
            self.reset()

        if self.config.adaptive.enabled:
            self._warn_adaptive_overrides()

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

    def _warn_adaptive_overrides(self) -> None:
        """Warn once per ``run()`` about silently ignored settings (P10).

        A frozen dataclass cannot record whether a field was passed
        explicitly, so "explicit" is approximated as "differs from the
        default" — an explicit value equal to the default is ignored
        without a warning, which loses nothing.
        """
        if self.config.mode != ExecutionMode.COOLDOWN:
            warnings.warn(
                "adaptive mode is only honored in cooldown; this "
                f"{self.config.mode.value} run uses the configured fixed "
                "sweep schedule.",
                UserWarning,
                stacklevel=3,
            )
            return
        ignored = []
        if self.config.n_sweeps != DEFAULT_N_SWEEPS:
            ignored.append(f"n_sweeps={self.config.n_sweeps}")
        if self.config.measurement_interval != DEFAULT_MEASUREMENT_INTERVAL:
            ignored.append(
                f"measurement_interval={self.config.measurement_interval}"
            )
        if ignored:
            warnings.warn(
                "adaptive mode ignores " + ", ".join(ignored) + "; the "
                "production sample count and spacing derive from "
                "min_independent_samples and the measured tau_int instead.",
                UserWarning,
                stacklevel=3,
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
                correlation_interval=self.config.correlation_interval,
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
                correlation_interval=self.config.correlation_interval,
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

    def sweep(self, n_sweeps: int = 1, *, temperature: float) -> dict[str, float]:
        """Perform sweeps at a given temperature and return observables.

        One sweep is ``num_sites`` attempted-flip equivalents for
        Metropolis and Swendsen-Wang, but ONE cluster update for Wolff
        — measuring at a per-sweep flip-budget boundary is size-biased
        (P10 exact-enumeration rejection), so Wolff callers scale
        ``n_sweeps`` by roughly ``num_sites`` over the expected cluster
        size instead. The returned counters report the work honestly.

        Parameters
        ----------
        n_sweeps : int
            Number of MC sweeps to perform.
        temperature : float
            Simulation temperature (must be > 0). Keyword-only, so a
            pre-1.0 positional ``beta`` can never be silently
            reinterpreted as a temperature.

        Returns
        -------
        dict[str, float]
            Dictionary with keys 'energy', 'magnetization',
            'acceptance_rate', and 'n_cluster_flips' (0.0 for
            Metropolis). 'acceptance_rate' is the classic acceptance
            fraction for Metropolis, the flipped-spin fraction for
            Swendsen-Wang, and identically 1.0 for Wolff
            (rejection-free).
        """
        if temperature <= 0:
            msg = f"Temperature must be positive, got {temperature}"
            raise SimulationError(msg)

        accepted, attempted, cluster_flips = self._core.sweep(
            n_sweeps, temperature=temperature
        )

        return {
            "energy": self._core.energy(),
            "magnetization": self._core.magnetization(),
            "acceptance_rate": accepted / attempted if attempted > 0 else 0.0,
            "n_cluster_flips": float(cluster_flips),
        }

    @property
    def num_sites(self) -> int:
        """Total number of spins N in the simulated lattice."""
        return int(self._core.num_sites)

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
        """Anneal from from_temp to to_temp in one Rust call.

        The linear schedule is the one the former per-sweep loop walked
        (``num=1`` yields ``[from_temp]``); ``anneal`` skips non-positive
        entries exactly as that loop did and consumes the RNG identically.
        """
        if n_steps <= 0:
            return

        temp_schedule = np.linspace(from_temp, to_temp, num=n_steps)
        self._core.anneal(temp_schedule.tolist())

    def _collect_at_temperature(
        self, temperature: float, results: SimulationResults
    ) -> None:
        """Run the production block at one temperature in a single Rust call.

        ``production_sweeps`` sweeps, measures and snapshots inside Rust
        with the GIL released. The RNG is consumed exactly as the former
        per-measurement ``sweep`` / ``energy`` / ``magnetization`` /
        ``get_spins`` loop consumed it, so fixed-seed results are
        bit-identical (``tests/test_golden.py``); the correlation bins are
        computed once per evaluation instead of twice.
        """
        n_measurements = self.config.n_sweeps // self.config.measurement_interval
        entry = self._core.production_sweeps(
            n_measurements,
            self.config.measurement_interval,
            temperature=temperature,
            store_configs=self.config.store_configs,
            compute_correlation=self.config.compute_correlation,
            correlation_interval=self.config.correlation_interval,
        )
        _fill_results_entry(temperature, entry, results)

    def _thermalize_adaptive(
        self,
        from_temp: float,
        to_temp: float,
        results: SimulationResults,
    ) -> None:
        """Adaptive thermalization: annealing ramp + fixed-T diagnostics.

        The cool-down ramp is pure annealing — its energy trace is
        non-stationary by construction (the temperature changes every
        sweep), so it is never analyzed (B9, #20). Stationarity (MSER)
        and tau_int (Sokal) are estimated exclusively on a
        fixed-temperature diagnostic series collected at ``to_temp``;
        the production measurement interval derives from the stationary
        tail of that series only.
        """
        ac = self.config.adaptive
        n_ramp = max(self.config.n_thermalization, ac.min_thermalization_sweeps)

        # Annealing ramp (single FFI call). Ramp energies are never
        # recorded or analyzed (B9).
        temp_schedule = np.linspace(from_temp, to_temp, num=n_ramp).tolist()
        self._core.anneal(temp_schedule)

        # Fixed-temperature diagnostic block: the only series the
        # analyzer ever sees. Always collected, even when the ramp
        # already used the thermalization budget — without it there is
        # nothing to base the measurement interval on.
        block = max(ac.min_thermalization_sweeps, MIN_DIAGNOSTIC_SWEEPS)
        energy_series = np.asarray(
            self._core.extend_thermalization(block, temperature=to_temp)
        )
        stationary = len(energy_series)

        analysis = _analyze_thermalization(energy_series, ac)

        # Extend at fixed temperature while not thermalized; MSER's
        # truncation discards the stale prefix on each re-analysis. The
        # cap bounds the total (ramp + fixed-T), as before.
        while (
            not analysis["is_thermalized"]
            and n_ramp + stationary < ac.max_thermalization_sweeps
        ):
            extra_n = min(
                block,
                ac.max_thermalization_sweeps - n_ramp - stationary,
            )
            extra_energies = np.asarray(
                self._core.extend_thermalization(extra_n, temperature=to_temp)
            )
            energy_series = np.concatenate([energy_series, extra_energies])
            stationary += extra_n

            analysis = _analyze_thermalization(energy_series, ac)

        if not analysis["is_thermalized"]:
            warnings.warn(
                f"Thermalization not detected at T={to_temp:g} after "
                f"{stationary} fixed-temperature sweeps; the measurement "
                "interval derives from a non-stationary tail.",
                UserWarning,
                stacklevel=2,
            )

        # Store diagnostics (production filled by _collect_adaptive)
        if results.adaptive_diagnostics is not None:
            results.adaptive_diagnostics[to_temp] = AdaptiveDiagnostics(
                thermalization_sweeps=n_ramp + stationary,
                stationary_sweeps=stationary,
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

        # Never silently deliver fewer samples than asked for; the
        # interval itself is not capped (it is the tau-derived spacing).
        if n_measurements < ac.min_independent_samples:
            warnings.warn(
                f"Sweep budget at T={temperature:g} allows only "
                f"{n_measurements} of the requested "
                f"{ac.min_independent_samples} samples at measurement "
                f"interval {interval}.",
                UserWarning,
                stacklevel=2,
            )

        # Single Rust call for all production measurements. Correlation
        # data is deliberately not requested here: adaptive mode takes one
        # end-of-production snapshot below (unchanged behaviour).
        entry = self._core.production_sweeps(
            n_measurements,
            interval,
            temperature=temperature,
            store_configs=self.config.store_configs,
        )
        _fill_results_entry(temperature, entry, results)

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
