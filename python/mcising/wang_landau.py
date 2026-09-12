"""Wang-Landau sampling with a multicanonical production run.

Parallel tempering fails at a strongly first-order transition: the
replicas never cross the free-energy barrier between the coexisting phases
and return smooth, wrong averages. A flat-histogram method walks through
the barrier instead. :class:`WangLandauSimulation` runs two stages in the
Rust core:

1. **Wang-Landau** (Wang & Landau, Phys. Rev. Lett. 86, 2050 (2001))
   estimates the density of states ``g(E)`` on the exact energy grid of
   the couplings, halving the modification factor at every flat
   histogram and finishing with the ``1/t`` schedule of Belardinelli &
   Pereyra (Phys. Rev. E 75, 046701 (2007)).
2. **Multicanonical production** (Berg & Neuhaus, Phys. Rev. Lett. 68, 9
   (1992)) freezes the weights ``W(E) = 1/g(E)`` and runs independent
   walkers whose flat energy histogram covers ordered and disordered
   phases alike, recording the energy, magnetization and staggered
   magnetizations of every measurement.

:class:`WangLandauResults` then reweights the production series to any
temperature (energy, specific heat, energy cumulant, magnetization and
the order parameter of a symmetry-broken phase with their susceptibilities
and Binder cumulants), builds the canonical energy histogram, and measures
the free-energy barrier, the interface tension and the pseudo-transition
temperatures of a first-order transition (Lee & Kosterlitz, Phys. Rev.
Lett. 65, 137 (1990)). Every estimate carries a delete-one-block jackknife
error whose blocks never straddle walkers, plus the effective sample size
that says how much of the production run actually supports it.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from mcising import reweighting
from mcising._core import run_wang_landau as _run_wang_landau
from mcising._provenance import WANG_LANDAU_SCHEMA_VERSION, git_commit, package_version
from mcising.config import LatticeConfig, LatticeType, _construct, _known_fields
from mcising.constants import (
    DEFAULT_SEED,
    DEFAULT_WL_CHECK_INTERVAL,
    DEFAULT_WL_DRIVE_BETA,
    DEFAULT_WL_DRIVE_MAX_SWEEPS,
    DEFAULT_WL_EXCHANGE_INTERVAL,
    DEFAULT_WL_FLATNESS,
    DEFAULT_WL_LOG_F_FINAL,
    DEFAULT_WL_LOG_F_INITIAL,
    DEFAULT_WL_PRODUCTION_SWEEPS,
    DEFAULT_WL_WINDOW_OVERLAP,
)
from mcising.exceptions import ConfigurationError
from mcising.statistics import Estimate

__all__: Final[list[str]] = [
    "BarrierEstimate",
    "CanonicalEstimates",
    "MulticanonicalDiagnostics",
    "WalkerSeries",
    "WangLandauConfig",
    "WangLandauDiagnostics",
    "WangLandauResults",
    "WangLandauSimulation",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: Spatial dimension per lattice, for the interface tension of a periodic
#: box (two interfaces of area ``L^(d-1)``).
_DIMENSION: Final[dict[LatticeType, int]] = {
    LatticeType.CHAIN: 1,
    LatticeType.SQUARE: 2,
    LatticeType.TRIANGULAR: 2,
    LatticeType.HONEYCOMB: 2,
    LatticeType.CUBIC: 3,
}


def _is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


@dataclass(frozen=True)
class WangLandauConfig:
    """Configuration of a Wang-Landau run with multicanonical production.

    Parameters
    ----------
    lattice : LatticeConfig
        Lattice geometry and couplings. The energy grid is exact when
        every coupling is exactly representable in binary at a usable
        resolution (``1``, ``-0.5``, ``0.25``, ...); otherwise
        ``bin_width`` is required.
    seed : int
        Seed of the Wang-Landau walker; production walker ``k`` uses
        ``seed + 1000 + k``.
    energy_window : tuple[float, float] | None
        Per-site energy window ``(lo, hi)`` to sample, or ``None`` for
        the whole spectrum. A window that covers the two coexisting
        phases of a first-order transition (from a short canonical pilot
        on either side of it) cuts the cost by orders of magnitude at
        large sizes. The walker is driven into the window with
        Metropolis sweeps at ``±drive_beta`` first.
    bin_width : float | None
        Energy-bin width in **total-energy** units. ``None`` selects the
        exact grid (the greatest common divisor of every single-flip
        energy change, e.g. ``4`` for the nearest-neighbour square
        lattice at ``j1 = 1``); a value coarsens the grid, and is
        required for couplings that are not exactly representable in
        binary.
    flatness : float
        Flatness criterion ``min H >= flatness * mean H`` over the
        visited bins, in ``(0, 1]``.
    log_f_initial, log_f_final : float
        Modification-factor schedule: ``ln f`` starts at
        ``log_f_initial``, halves at every flat histogram, follows the
        ``1/t`` law once it drops below ``1/t`` (``t`` = mean visits per
        bin of the window), and stops below ``log_f_final``. With a
        production stage ``log_f_final`` is a cost knob, not an accuracy
        knob: the 1/t stage costs about ``n_bins / log_f_final`` flip
        attempts.
    check_interval : int
        Sweeps between flatness checks.
    max_wl_sweeps : int | None
        Cap on the Wang-Landau stage; a capped run reports
        ``converged=False`` and still produces. ``0`` keeps an
        ``initial_log_g`` passed to :meth:`WangLandauSimulation.run` as
        the frozen weights.
    production_sweeps : int
        Sweeps per production walker (``0`` skips production).
    production_thermalization : int
        Sweeps each production walker discards first.
    measurement_interval : int
        Sweeps between measurements in production.
    n_walkers : int
        Independent production walkers, run in parallel; each is an
        independent chain, so eight or more make the jackknife blocks
        coincide with walkers.
    store_configs : bool
        Store a spin configuration at every production measurement.
    drive_beta : float
        Inverse temperature of the Metropolis drive into the window.
    drive_max_sweeps : int
        Cap on the drive-in; an unreachable window raises.
    n_windows : int
        Energy windows of the replica-exchange Wang-Landau stage (Vogel,
        Li, Wüst & Landau 2013). ``1`` runs the serial walker; more
        split the range into overlapping windows sampled in parallel and
        joined at the end.
    walkers_per_window : int
        Walkers per window, run in parallel; their estimates are averaged
        at every check. ``n_windows * walkers_per_window`` is the
        parallelism of the first stage.
    window_overlap : float
        Fraction of a window's length shared with its neighbour, in
        ``[0, 1)``.
    exchange_interval : int
        Sweeps between replica-exchange attempts; ``check_interval`` must
        be a multiple of it when the parallel stage runs.
    """

    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    seed: int = DEFAULT_SEED
    energy_window: tuple[float, float] | None = None
    bin_width: float | None = None
    flatness: float = DEFAULT_WL_FLATNESS
    log_f_initial: float = DEFAULT_WL_LOG_F_INITIAL
    log_f_final: float = DEFAULT_WL_LOG_F_FINAL
    check_interval: int = DEFAULT_WL_CHECK_INTERVAL
    max_wl_sweeps: int | None = None
    production_sweeps: int = DEFAULT_WL_PRODUCTION_SWEEPS
    production_thermalization: int = 0
    measurement_interval: int = 1
    n_walkers: int = 1
    store_configs: bool = False
    drive_beta: float = DEFAULT_WL_DRIVE_BETA
    drive_max_sweeps: int = DEFAULT_WL_DRIVE_MAX_SWEEPS
    n_windows: int = 1
    walkers_per_window: int = 1
    window_overlap: float = DEFAULT_WL_WINDOW_OVERLAP
    exchange_interval: int = DEFAULT_WL_EXCHANGE_INTERVAL

    def __post_init__(self) -> None:
        if self.energy_window is not None:
            window = self.energy_window
            if (
                len(window) != 2
                or not _is_finite(window[0])
                or not _is_finite(window[1])
                or window[0] >= window[1]
            ):
                raise ConfigurationError(
                    "energy_window must be a finite (lo, hi) pair per site with "
                    f"lo < hi, got {window!r}"
                )
            object.__setattr__(
                self, "energy_window", (float(window[0]), float(window[1]))
            )
        if self.bin_width is not None and (
            not _is_finite(self.bin_width) or self.bin_width <= 0
        ):
            msg = f"bin_width must be positive and finite, got {self.bin_width}"
            raise ConfigurationError(msg)
        if not _is_finite(self.flatness) or not 0.0 < self.flatness <= 1.0:
            msg = f"flatness must be in (0, 1], got {self.flatness}"
            raise ConfigurationError(msg)
        if not _is_finite(self.log_f_initial) or self.log_f_initial <= 0:
            msg = f"log_f_initial must be positive and finite, got {self.log_f_initial}"
            raise ConfigurationError(msg)
        if (
            not _is_finite(self.log_f_final)
            or self.log_f_final <= 0
            or self.log_f_final >= self.log_f_initial
        ):
            raise ConfigurationError(
                "log_f_final must satisfy 0 < log_f_final < log_f_initial, got "
                f"log_f_final={self.log_f_final}, log_f_initial={self.log_f_initial}"
            )
        if self.check_interval < 1:
            msg = f"check_interval must be >= 1, got {self.check_interval}"
            raise ConfigurationError(msg)
        if self.max_wl_sweeps is not None and self.max_wl_sweeps < 0:
            msg = f"max_wl_sweeps must be >= 0 or None, got {self.max_wl_sweeps}"
            raise ConfigurationError(msg)
        if self.production_sweeps < 0:
            msg = f"production_sweeps must be >= 0, got {self.production_sweeps}"
            raise ConfigurationError(msg)
        if self.production_thermalization < 0:
            raise ConfigurationError(
                "production_thermalization must be >= 0, got "
                f"{self.production_thermalization}"
            )
        if self.measurement_interval < 1:
            msg = f"measurement_interval must be >= 1, got {self.measurement_interval}"
            raise ConfigurationError(msg)
        if self.n_walkers < 1:
            msg = f"n_walkers must be >= 1, got {self.n_walkers}"
            raise ConfigurationError(msg)
        if not _is_finite(self.drive_beta) or self.drive_beta <= 0:
            msg = f"drive_beta must be positive and finite, got {self.drive_beta}"
            raise ConfigurationError(msg)
        if self.drive_max_sweeps < 1:
            msg = f"drive_max_sweeps must be >= 1, got {self.drive_max_sweeps}"
            raise ConfigurationError(msg)
        if self.n_windows < 1:
            msg = f"n_windows must be >= 1, got {self.n_windows}"
            raise ConfigurationError(msg)
        if self.walkers_per_window < 1:
            msg = f"walkers_per_window must be >= 1, got {self.walkers_per_window}"
            raise ConfigurationError(msg)
        if not _is_finite(self.window_overlap) or not 0.0 <= self.window_overlap < 1.0:
            msg = f"window_overlap must be in [0, 1), got {self.window_overlap}"
            raise ConfigurationError(msg)
        if self.exchange_interval < 1:
            msg = f"exchange_interval must be >= 1, got {self.exchange_interval}"
            raise ConfigurationError(msg)
        if self.parallel_stage and self.check_interval % self.exchange_interval != 0:
            raise ConfigurationError(
                "The parallel Wang-Landau stage checks flatness on "
                "exchange_interval boundaries, so check_interval must be a "
                f"multiple of exchange_interval; got check_interval="
                f"{self.check_interval}, exchange_interval={self.exchange_interval}"
            )

    @property
    def parallel_stage(self) -> bool:
        """Whether the replica-exchange (parallel) first stage runs."""
        return self.n_windows > 1 or self.walkers_per_window > 1

    @property
    def n_measurements(self) -> int:
        """Measurements per production walker."""
        return self.production_sweeps // self.measurement_interval

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WangLandauConfig:
        """Build a WangLandauConfig from a mapping.

        The inverse of ``dataclasses.asdict``: the nested ``lattice``
        mapping becomes a :class:`~mcising.config.LatticeConfig` and a
        list-valued ``energy_window`` becomes a tuple. Unknown keys are
        ignored and missing keys take their defaults; validation runs as
        usual.

        Parameters
        ----------
        data : Mapping[str, Any]
            Field values, e.g. a decoded ``config_json`` record.

        Returns
        -------
        WangLandauConfig
            A validated configuration.

        Raises
        ------
        ConfigurationError
            If ``data`` is not a mapping or any value is invalid.
        """
        kwargs = _known_fields(cls, data, "Wang-Landau config")
        if "lattice" in kwargs:
            kwargs["lattice"] = LatticeConfig.from_dict(kwargs["lattice"])
        if isinstance(kwargs.get("energy_window"), list):
            kwargs["energy_window"] = tuple(kwargs["energy_window"])
        return _construct(cls, kwargs, "Wang-Landau config")


@dataclass(frozen=True)
class WangLandauDiagnostics:
    """Record of the Wang-Landau stage, one entry per modification factor.

    Attributes
    ----------
    iteration_log_f, iteration_sweeps, iteration_flatness, iteration_visited_bins
        Per iteration: the modification factor ``ln f`` it started with,
        its length in sweeps, the flatness ratio ``min H / mean H`` at
        its end, and the number of visited bins. The last entry is the
        ``1/t`` stage when it was entered.
    total_sweeps : int
        Wang-Landau sweeps in total.
    one_over_t_switch_sweep : int | None
        Sweep at which the ``1/t`` schedule took over, or ``None``.
    final_log_f : float
        ``ln f`` at the end of the stage.
    converged : bool
        Whether ``ln f`` reached ``log_f_final`` (``False`` when
        ``max_wl_sweeps`` cut the stage short: the weights are then only
        as good as the histogram flatness of the production run says).
    accepted, attempted : int
        Flip acceptance counters of the stage.
    drive_in_sweeps : int
        Metropolis sweeps spent reaching the energy window (summed over
        walkers).
    visited_bins : int
        Energy bins entered at least once.
    n_windows, walkers_per_window : int
        Layout of the replica-exchange stage (``1`` and ``1`` for the
        serial walker).
    window_bins : tuple[tuple[int, int], ...]
        Inclusive bin range of every window.
    exchange_attempted, exchange_accepted : tuple[int, ...]
        Replica-exchange attempts and acceptances per adjacent window
        pair (empty for a single window).
    merge_bins : tuple[int, ...]
        Bin at which each window's estimate was joined to the previous
        one (empty for a single window).
    """

    iteration_log_f: tuple[float, ...]
    iteration_sweeps: tuple[int, ...]
    iteration_flatness: tuple[float, ...]
    iteration_visited_bins: tuple[int, ...]
    total_sweeps: int
    one_over_t_switch_sweep: int | None
    final_log_f: float
    converged: bool
    accepted: int
    attempted: int
    drive_in_sweeps: int
    visited_bins: int
    n_windows: int = 1
    walkers_per_window: int = 1
    window_bins: tuple[tuple[int, int], ...] = ()
    exchange_attempted: tuple[int, ...] = ()
    exchange_accepted: tuple[int, ...] = ()
    merge_bins: tuple[int, ...] = ()

    @property
    def n_iterations(self) -> int:
        """Number of modification-factor values used."""
        return len(self.iteration_log_f)

    @property
    def acceptance(self) -> float:
        """Flip acceptance rate of the stage (``nan`` without attempts)."""
        return self.accepted / self.attempted if self.attempted else math.nan

    @property
    def exchange_acceptance(self) -> FloatArray:
        """Replica-exchange acceptance per adjacent window pair (``nan`` where none)."""
        attempted = np.asarray(self.exchange_attempted, dtype=np.float64)
        accepted = np.asarray(self.exchange_accepted, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(attempted > 0, accepted / attempted, np.nan)

    @classmethod
    def _from_raw(cls, raw: Mapping[str, Any]) -> WangLandauDiagnostics:
        switch = raw["one_over_t_switch_sweep"]
        return cls(
            iteration_log_f=tuple(float(x) for x in raw["iteration_log_f"]),
            iteration_sweeps=tuple(int(x) for x in raw["iteration_sweeps"]),
            iteration_flatness=tuple(float(x) for x in raw["iteration_flatness"]),
            iteration_visited_bins=tuple(int(x) for x in raw["iteration_visited_bins"]),
            total_sweeps=int(raw["total_sweeps"]),
            one_over_t_switch_sweep=None if switch is None else int(switch),
            final_log_f=float(raw["final_log_f"]),
            converged=bool(raw["converged"]),
            accepted=int(raw["accepted"]),
            attempted=int(raw["attempted"]),
            drive_in_sweeps=int(raw["drive_in_sweeps"]),
            visited_bins=int(raw["visited_bins"]),
            n_windows=int(raw.get("n_windows", 1)),
            walkers_per_window=int(raw.get("walkers_per_window", 1)),
            window_bins=tuple(
                (int(lo), int(hi)) for lo, hi in raw.get("window_bins", ())
            ),
            exchange_attempted=tuple(int(x) for x in raw.get("exchange_attempted", ())),
            exchange_accepted=tuple(int(x) for x in raw.get("exchange_accepted", ())),
            merge_bins=tuple(int(x) for x in raw.get("merge_bins", ())),
        )


@dataclass(frozen=True)
class MulticanonicalDiagnostics:
    """Mixing evidence of the production stage.

    Whether a flat-histogram run actually crossed the barrier cannot be
    read off the reweighted averages; these counters can. ``round_trips``
    counts, per walker, the completed excursions from the lowest visited
    energy bin to the highest and back (the analogue of a replica's
    round trip in parallel tempering); a walker with none has not
    demonstrably connected the two ends of its window.

    Attributes
    ----------
    n_walkers : int
        Production walkers.
    sweeps_per_walker, thermalization_sweeps : int
        Recorded and discarded sweeps per walker.
    accepted, attempted, rejected_unvisited : tuple[int, ...]
        Per walker: accepted flips, attempted flips, and proposals into
        a bin the Wang-Landau stage never entered (rejected, as no weight
        exists there).
    round_trips : tuple[int, ...]
        Per-walker round trips between the edge bins.
    histogram_flatness : float
        ``min H / mean H`` of the pooled production histogram over the
        visited bins: how good the frozen weights were (1 is perfect).
    edge_bins : tuple[int, int] | None
        The bins between which round trips are counted.
    """

    n_walkers: int
    sweeps_per_walker: int
    thermalization_sweeps: int
    accepted: tuple[int, ...]
    attempted: tuple[int, ...]
    rejected_unvisited: tuple[int, ...]
    round_trips: tuple[int, ...]
    histogram_flatness: float
    edge_bins: tuple[int, int] | None

    @property
    def acceptance(self) -> FloatArray:
        """Flip acceptance rate per walker (``nan`` where nothing was attempted)."""
        attempted = np.asarray(self.attempted, dtype=np.float64)
        accepted = np.asarray(self.accepted, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(attempted > 0, accepted / attempted, np.nan)

    @property
    def total_round_trips(self) -> int:
        """Round trips summed over every walker."""
        return int(sum(self.round_trips))

    @classmethod
    def _from_raw(cls, raw: Mapping[str, Any]) -> MulticanonicalDiagnostics:
        edges = raw["edge_bins"]
        return cls(
            n_walkers=int(raw["n_walkers"]),
            sweeps_per_walker=int(raw["sweeps_per_walker"]),
            thermalization_sweeps=int(raw["thermalization_sweeps"]),
            accepted=tuple(int(x) for x in raw["accepted"]),
            attempted=tuple(int(x) for x in raw["attempted"]),
            rejected_unvisited=tuple(int(x) for x in raw["rejected_unvisited"]),
            round_trips=tuple(int(x) for x in raw["round_trips"]),
            histogram_flatness=float(raw["histogram_flatness"]),
            edge_bins=None if edges is None else (int(edges[0]), int(edges[1])),
        )


@dataclass
class WalkerSeries:
    """The production series of one walker.

    Attributes
    ----------
    energy : NDArray[np.float64]
        Energy per site at every measurement.
    magnetization : NDArray[np.float64]
        Magnetization per site.
    staggered_magnetization : NDArray[np.float64]
        Staggered magnetizations, shape ``(n_samples, 2 ** n_axes)`` with
        the bitmask column convention of
        :class:`~mcising.SimulationResults`.
    bin_index : NDArray[np.int64]
        Energy bin of every measurement (index into the grid arrays of
        :class:`WangLandauResults`).
    configurations : NDArray[np.int8] | None
        Spin configurations when stored, shape ``(n_samples, *shape)``.
    accepted, attempted, round_trips : int
        This walker's counters.
    """

    energy: FloatArray
    magnetization: FloatArray
    staggered_magnetization: FloatArray
    bin_index: NDArray[np.int64]
    configurations: NDArray[np.int8] | None
    accepted: int
    attempted: int
    round_trips: int

    @property
    def n_samples(self) -> int:
        """Measurements recorded by this walker."""
        return int(self.energy.size)

    @classmethod
    def _from_raw(cls, raw: Mapping[str, Any]) -> WalkerSeries:
        configurations = raw.get("configurations")
        return cls(
            energy=np.asarray(raw["energies"], dtype=np.float64),
            magnetization=np.asarray(raw["magnetizations"], dtype=np.float64),
            staggered_magnetization=np.asarray(
                raw["staggered_magnetizations"], dtype=np.float64
            ),
            bin_index=np.asarray(raw["bin_index"]).astype(np.int64),
            configurations=(
                None
                if configurations is None
                else np.asarray(configurations, dtype=np.int8)
            ),
            accepted=int(raw["accepted"]),
            attempted=int(raw["attempted"]),
            round_trips=int(raw["round_trips"]),
        )


@dataclass(frozen=True)
class CanonicalEstimates:
    """Canonical averages at one temperature, reweighted from a production run.

    Every value is an :class:`~mcising.statistics.Estimate` with a
    delete-one-block jackknife error. Conventions match
    :mod:`mcising.statistics`: energies per site, ``specific_heat = N
    beta^2 (<e^2> - <e>^2)``, connected susceptibilities ``N beta (<x^2> -
    <|x|>^2)``, Binder cumulants ``1 - <x^4> / (3 <x^2>^2)``.

    Attributes
    ----------
    temperature : float
        Temperature of the estimates.
    energy, specific_heat, energy_cumulant : Estimate
        Energy per site, specific heat per site, and the energy cumulant
        ``V = 1 - <e^4> / (3 <e^2>^2)`` whose minimum locates a
        first-order transition (Challa, Landau & Binder 1986).
    abs_magnetization, susceptibility, binder_cumulant : Estimate
        Uniform magnetization observables.
    order_parameter, order_susceptibility, order_binder : Estimate
        The same for the order parameter ``psi = max_k |m_k|`` over the
        chosen staggered components (the phase whose uniform
        magnetization vanishes).
    effective_samples : float
        Kish effective sample size of the reweighting at this
        temperature; a small value means the production run barely
        covers this temperature.
    edge_weight : float
        Fraction of the canonical weight sitting in the energy bins at
        which the energy window cuts the spectrum (a window edge the
        walker reached; an edge it never reached, or the end of the whole
        spectrum, is the physical end of the density of states and does
        not count). A value that is not tiny means the window cuts into
        the canonical distribution and the estimate is not trustworthy;
        ``0`` for a run over the whole spectrum.
    """

    temperature: float
    energy: Estimate
    specific_heat: Estimate
    energy_cumulant: Estimate
    abs_magnetization: Estimate
    susceptibility: Estimate
    binder_cumulant: Estimate
    order_parameter: Estimate
    order_susceptibility: Estimate
    order_binder: Estimate
    effective_samples: float
    edge_weight: float


@dataclass(frozen=True)
class BarrierEstimate:
    """Free-energy barrier of a bimodal canonical energy distribution.

    Attributes
    ----------
    temperature : float
        Temperature of the canonical distribution.
    barrier : Estimate
        ``ΔF / T = ln(P_peak / P_bottom)`` in units of the temperature
        (Lee & Kosterlitz 1990), measured from the lower peak; ``nan``
        when the distribution is not bimodal.
    energy_low, energy_high, energy_bottom : float
        Per-site energies of the lower-energy peak, the higher-energy
        peak and the bottom between them (``nan`` when not bimodal).
    interface_tension : Estimate
        ``σ = T ΔF/T / (2 L^(d-1))`` in energy units: a periodic box at
        coexistence holds two interfaces of area ``L^(d-1)``. Meaningful
        only once the bottom is a flat plateau (sizes where the peaks
        are well separated).
    """

    temperature: float
    barrier: Estimate
    energy_low: float
    energy_high: float
    energy_bottom: float
    interface_tension: Estimate


def _estimates(values: FloatArray, errors: FloatArray) -> list[Estimate]:
    return [Estimate(float(v), float(e)) for v, e in zip(values, errors, strict=True)]


def _cumulant(second: float, fourth: float) -> float:
    """``1 - <x^4> / (3 <x^2>^2)``; ``nan`` when the second moment vanishes."""
    if not second > 0.0:
        return math.nan
    return 1.0 - fourth / (3.0 * second * second)


@dataclass
class WangLandauResults:
    """Density of states and production series of a Wang-Landau run.

    Attributes
    ----------
    energy_bins : NDArray[np.float64]
        Per-site energy at the centre of every bin of the grid.
    log_g : NDArray[np.float64]
        ``ln g(E)`` per bin up to an additive constant, ``nan`` where the
        Wang-Landau stage never entered (unreachable energies, or bins
        outside the window).
    wl_histogram, production_histogram : NDArray[np.int64]
        Visits per bin: the Wang-Landau histogram since its last reset
        and the pooled production histogram.
    bin_width : float
        Bin width in total-energy units.
    window_bins : tuple[int, int]
        Inclusive bin range the run sampled.
    walkers : list[WalkerSeries]
        Production series, one per walker.
    wang_landau : WangLandauDiagnostics
        Stage-1 record.
    production : MulticanonicalDiagnostics
        Stage-2 mixing evidence.
    final_spins : NDArray[np.int8]
        Flat configuration of the Wang-Landau walker at the end.
    final_rng_state : bytes
        Serialized generator state of the Wang-Landau walker.
    metadata : dict[str, object]
        Provenance: the :class:`WangLandauConfig` under ``"config"``,
        plus ``kind`` (``"wang_landau"``), ``version``,
        ``schema_version``, ``seed``, optional ``git_commit`` and
        ``elapsed_seconds``.
    """

    energy_bins: FloatArray
    log_g: FloatArray
    wl_histogram: NDArray[np.int64]
    production_histogram: NDArray[np.int64]
    bin_width: float
    window_bins: tuple[int, int]
    walkers: list[WalkerSeries]
    wang_landau: WangLandauDiagnostics
    production: MulticanonicalDiagnostics
    final_spins: NDArray[np.int8]
    final_rng_state: bytes
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def _from_raw(
        cls, raw: Mapping[str, Any], metadata: dict[str, object]
    ) -> WangLandauResults:
        """Build from the dict the Rust runner returns."""
        window = raw["window_bins"]
        return cls(
            energy_bins=np.asarray(raw["energy_bins"], dtype=np.float64),
            log_g=np.asarray(raw["log_g"], dtype=np.float64),
            wl_histogram=np.asarray(raw["wl_histogram"]).astype(np.int64),
            production_histogram=np.asarray(raw["production_histogram"]).astype(
                np.int64
            ),
            bin_width=float(raw["bin_width"]),
            window_bins=(int(window[0]), int(window[1])),
            walkers=[WalkerSeries._from_raw(entry) for entry in raw["walkers"]],
            wang_landau=WangLandauDiagnostics._from_raw(raw["wl_diagnostics"]),
            production=MulticanonicalDiagnostics._from_raw(
                raw["production_diagnostics"]
            ),
            final_spins=np.asarray(raw["final_spins"], dtype=np.int8),
            final_rng_state=bytes(raw["final_rng_state"]),
            metadata=metadata,
        )

    # -- geometry -------------------------------------------------------

    @property
    def config(self) -> WangLandauConfig:
        """The configuration the run was made with."""
        config = self.metadata.get("config")
        if not isinstance(config, WangLandauConfig):
            msg = "results carry no WangLandauConfig in metadata['config']"
            raise ConfigurationError(msg)
        return config

    @property
    def num_sites(self) -> int:
        """Total number of spins N."""
        return self.config.lattice.num_sites

    @property
    def n_bins(self) -> int:
        """Bins of the energy grid."""
        return int(self.energy_bins.size)

    @property
    def visited(self) -> BoolArray:
        """Mask of the bins the Wang-Landau stage entered."""
        return np.isfinite(self.log_g)

    # -- concatenated series --------------------------------------------

    @property
    def walker_lengths(self) -> tuple[int, ...]:
        """Samples per walker, in concatenation order."""
        return tuple(w.n_samples for w in self.walkers)

    @property
    def n_samples(self) -> int:
        """Production measurements over every walker."""
        return sum(self.walker_lengths)

    @property
    def energy(self) -> FloatArray:
        """Per-site energies of every measurement, walkers concatenated."""
        return self._concat([w.energy for w in self.walkers])

    @property
    def magnetization(self) -> FloatArray:
        """Per-site magnetizations of every measurement, walkers concatenated."""
        return self._concat([w.magnetization for w in self.walkers])

    @property
    def staggered_magnetization(self) -> FloatArray:
        """Staggered magnetizations, shape ``(n_samples, 2 ** n_axes)``."""
        if not self.walkers:
            return np.empty((0, 0), dtype=np.float64)
        return np.concatenate([w.staggered_magnetization for w in self.walkers], axis=0)

    @property
    def bin_index(self) -> NDArray[np.int64]:
        """Energy bin of every measurement, walkers concatenated."""
        if not self.walkers:
            return np.empty(0, dtype=np.int64)
        return np.concatenate([w.bin_index for w in self.walkers])

    @property
    def walker_id(self) -> NDArray[np.int64]:
        """Walker of every measurement in the concatenated series."""
        return np.repeat(
            np.arange(len(self.walkers), dtype=np.int64), self.walker_lengths
        )

    @staticmethod
    def _concat(parts: Sequence[FloatArray]) -> FloatArray:
        if not parts:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(parts)

    # -- reweighting ----------------------------------------------------

    def _log_weights(self, temperature: float) -> FloatArray:
        if not _is_finite(temperature) or temperature <= 0:
            msg = f"temperature must be positive and finite, got {temperature}"
            raise ConfigurationError(msg)
        return reweighting.log_weights(
            self.energy * self.num_sites, self.log_g[self.bin_index], 1.0 / temperature
        )

    def _blocks(self, n_blocks: int | None) -> list[slice]:
        lengths = self.walker_lengths
        if n_blocks is None:
            n_walkers = len(lengths)
            n_blocks = (
                n_walkers
                if n_walkers >= reweighting.BLOCK_PER_WALKER_FROM
                else max(n_walkers, reweighting.DEFAULT_N_BLOCKS)
            )
        return reweighting.block_slices(lengths, n_blocks)

    def order_columns(
        self, order_parameter: Sequence[int] | None = None
    ) -> tuple[int, ...]:
        """Staggered-magnetization columns the order parameter ``psi`` runs over.

        ``None`` selects the components that alternate along a single
        lattice axis (``1, 2`` on the square lattice, ``1, 2, 4`` on the
        cubic lattice: stripe and layered order); pass the columns of the
        phase under study explicitly otherwise (``3, 5, 6`` for the
        columnar phase, ``7`` for the Néel phase of the cubic lattice).
        """
        n_axes = len(self.config.lattice.shape)
        if order_parameter is None:
            return tuple(1 << axis for axis in range(n_axes))
        columns = tuple(int(c) for c in order_parameter)
        if not columns or any(c < 0 or c >= (1 << n_axes) for c in columns):
            raise ConfigurationError(
                f"order_parameter columns must lie in [0, {1 << n_axes}), got {columns}"
            )
        return columns

    def effective_sample_size(self, temperature: float) -> float:
        """Kish effective sample size of the reweighting at ``temperature``."""
        return reweighting.effective_sample_size(self._log_weights(temperature))

    def reweight(
        self,
        temperature: float,
        *,
        order_parameter: Sequence[int] | None = None,
        n_blocks: int | None = None,
    ) -> CanonicalEstimates:
        """Canonical averages at ``temperature`` with jackknife errors.

        Parameters
        ----------
        temperature : float
            Target temperature.
        order_parameter : Sequence[int], optional
            Staggered-magnetization columns of the order parameter
            ``psi = max_k |m_k|`` (see :meth:`order_columns`).
        n_blocks : int, optional
            Jackknife blocks over all walkers; the default is one block
            per walker from eight walkers on, otherwise about twenty
            contiguous blocks that never straddle walkers. Blocks shorter
            than a few round-trip times underestimate the errors.

        Returns
        -------
        CanonicalEstimates
            Total: degenerate input yields ``nan`` fields, never an
            exception.
        """
        log_w = self._log_weights(temperature)
        beta = 1.0 / temperature
        n = float(self.num_sites)
        energy = self.energy
        magnetization = np.abs(self.magnetization)
        columns = list(self.order_columns(order_parameter))
        staggered = self.staggered_magnetization
        psi = (
            np.max(np.abs(staggered[:, columns]), axis=1)
            if staggered.size
            else np.empty(0, dtype=np.float64)
        )

        def estimator(keep: BoolArray) -> FloatArray:
            lw = log_w[keep]
            e = energy[keep]
            m = magnetization[keep]
            p = psi[keep]
            mean = reweighting.weighted_mean
            e1, e2, e4 = mean(e, lw), mean(e * e, lw), mean(e**4, lw)
            m1, m2, m4 = mean(m, lw), mean(m * m, lw), mean(m**4, lw)
            p1, p2, p4 = mean(p, lw), mean(p * p, lw), mean(p**4, lw)
            return np.array(
                [
                    e1,
                    n * beta * beta * (e2 - e1 * e1),
                    _cumulant(e2, e4),
                    m1,
                    n * beta * (m2 - m1 * m1),
                    _cumulant(m2, m4),
                    p1,
                    n * beta * (p2 - p1 * p1),
                    _cumulant(p2, p4),
                ]
            )

        values, errors = reweighting.block_jackknife(
            estimator, self.n_samples, self._blocks(n_blocks)
        )
        est = _estimates(values, errors)
        return CanonicalEstimates(
            temperature=float(temperature),
            energy=est[0],
            specific_heat=est[1],
            energy_cumulant=est[2],
            abs_magnetization=est[3],
            susceptibility=est[4],
            binder_cumulant=est[5],
            order_parameter=est[6],
            order_susceptibility=est[7],
            order_binder=est[8],
            effective_samples=reweighting.effective_sample_size(log_w),
            edge_weight=self._edge_weight(log_w),
        )

    def _edge_weight(self, log_w: FloatArray) -> float:
        """Canonical weight in the visited edge bins that coincide with a
        window edge (an edge the walker never reached, or the end of the
        whole spectrum, is the physical end of the density of states)."""
        edges = self.production.edge_bins
        if edges is None or log_w.size == 0:
            return math.nan
        window_lo, window_hi = self.window_bins
        binding = []
        if edges[0] == window_lo and window_lo > 0:
            binding.append(edges[0])
        if edges[1] == window_hi and window_hi < self.n_bins - 1:
            binding.append(edges[1])
        if not binding:
            return 0.0
        w = np.exp(log_w)
        total = float(w.sum())
        if total <= 0.0:
            return math.nan
        on_edge = np.isin(self.bin_index, binding)
        return float(w[on_edge].sum() / total)

    def reweight_curve(
        self,
        temperatures: Sequence[float],
        *,
        order_parameter: Sequence[int] | None = None,
        n_blocks: int | None = None,
    ) -> list[CanonicalEstimates]:
        """:meth:`reweight` at every temperature, in the order given."""
        return [
            self.reweight(float(t), order_parameter=order_parameter, n_blocks=n_blocks)
            for t in temperatures
        ]

    def to_dataframe(
        self,
        temperatures: Sequence[float],
        *,
        order_parameter: Sequence[int] | None = None,
    ) -> object:
        """Reweighted estimates at ``temperatures`` as a pandas DataFrame.

        Columns: ``T``, then value/``_err`` pairs for ``E``, ``Cv``, ``V``,
        ``M``, ``chi``, ``U4``, ``psi``, ``chi_psi``, ``U4_psi``, plus
        ``n_eff`` and ``edge_weight``.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        import pandas as pd  # type: ignore[import-untyped]

        rows = []
        for est in self.reweight_curve(temperatures, order_parameter=order_parameter):
            rows.append(
                {
                    "T": est.temperature,
                    "E": est.energy.value,
                    "E_err": est.energy.error,
                    "Cv": est.specific_heat.value,
                    "Cv_err": est.specific_heat.error,
                    "V": est.energy_cumulant.value,
                    "V_err": est.energy_cumulant.error,
                    "M": est.abs_magnetization.value,
                    "M_err": est.abs_magnetization.error,
                    "chi": est.susceptibility.value,
                    "chi_err": est.susceptibility.error,
                    "U4": est.binder_cumulant.value,
                    "U4_err": est.binder_cumulant.error,
                    "psi": est.order_parameter.value,
                    "psi_err": est.order_parameter.error,
                    "chi_psi": est.order_susceptibility.value,
                    "chi_psi_err": est.order_susceptibility.error,
                    "U4_psi": est.order_binder.value,
                    "U4_psi_err": est.order_binder.error,
                    "n_eff": est.effective_samples,
                    "edge_weight": est.edge_weight,
                }
            )
        return pd.DataFrame(rows)

    # -- histograms and barriers ----------------------------------------

    def canonical_energy_histogram(
        self, temperature: float
    ) -> tuple[FloatArray, FloatArray]:
        """Canonical probability of every energy bin at ``temperature``.

        Returns ``(energy_bins, probability)``; bins the production run
        never visited have probability zero.
        """
        probability = reweighting.canonical_histogram(
            self.bin_index, self._log_weights(temperature), self.n_bins
        )
        return self.energy_bins, probability

    def _histogram_from(self, keep: BoolArray, log_w: FloatArray) -> FloatArray:
        return reweighting.canonical_histogram(
            self.bin_index[keep], log_w[keep], self.n_bins
        )

    @property
    def peak_smoothing(self) -> int:
        """Half-width, in bins, of the moving average used to locate the
        peaks of a canonical energy histogram: one percent of the visited
        bins, so a grid of thousands of bins is read at the scale of its
        phases (hundreds of bins wide) rather than of its per-bin noise,
        while a small lattice with a few dozen levels is not smoothed."""
        return int(self.visited.sum()) // 100

    def free_energy_barrier(
        self,
        temperature: float,
        *,
        estimator: reweighting.BarrierEstimator = "plateau",
        n_blocks: int | None = None,
    ) -> BarrierEstimate:
        """Barrier ``ΔF/T`` between the two peaks of the canonical energy histogram.

        See :func:`mcising.reweighting.two_peak_barrier` for the peak and
        bottom definitions (the peaks are located on the histogram
        smoothed over :attr:`peak_smoothing` bins); the jackknife error
        re-detects the peaks in every block. ``nan`` values mean the
        distribution at this temperature is not bimodal.
        """
        log_w = self._log_weights(temperature)
        smooth = self.peak_smoothing
        probability = self._histogram_from(np.ones(self.n_samples, dtype=bool), log_w)
        barrier, low, high, bottom = reweighting.two_peak_barrier(
            probability, estimator=estimator, smooth=smooth
        )

        def block_estimator(keep: BoolArray) -> float:
            value, *_ = reweighting.two_peak_barrier(
                self._histogram_from(keep, log_w), estimator=estimator, smooth=smooth
            )
            return value

        values, errors = reweighting.block_jackknife(
            block_estimator, self.n_samples, self._blocks(n_blocks)
        )
        barrier_estimate = Estimate(barrier, float(errors[0]))
        lattice = self.config.lattice
        area = 2.0 * float(lattice.size) ** (_DIMENSION[lattice.lattice_type] - 1)
        tension = Estimate(
            temperature * barrier / area, temperature * float(errors[0]) / area
        )
        energies = self.energy_bins
        return BarrierEstimate(
            temperature=float(temperature),
            barrier=barrier_estimate,
            energy_low=float(energies[low]) if low >= 0 else math.nan,
            energy_high=float(energies[high]) if high >= 0 else math.nan,
            energy_bottom=float(energies[bottom]) if bottom >= 0 else math.nan,
            interface_tension=tension,
        )

    def _phase_cut(self, bracket: tuple[float, float], n_grid: int = 25) -> int | None:
        """Energy bin separating the two phases: the bottom of the
        best-separated bimodal canonical histogram over a temperature
        grid in ``bracket``; ``None`` when no temperature is bimodal."""
        lo, hi = float(bracket[0]), float(bracket[1])
        if not (_is_finite(lo) and _is_finite(hi)) or lo >= hi or lo <= 0.0:
            return None
        full = np.ones(self.n_samples, dtype=bool)
        smooth = self.peak_smoothing
        best: tuple[float, int] | None = None
        for temperature in np.linspace(lo, hi, n_grid):
            probability = self._histogram_from(
                full, self._log_weights(float(temperature))
            )
            barrier, _, _, bottom = reweighting.two_peak_barrier(
                probability, estimator="minimum", smooth=smooth
            )
            if math.isfinite(barrier) and (best is None or barrier > best[0]):
                best = (barrier, bottom)
        return None if best is None else best[1]

    def _peak_height_difference(
        self, temperature: float, keep: BoolArray, cut: int
    ) -> float:
        probability = self._histogram_from(keep, self._log_weights(temperature))
        low = float(probability[: cut + 1].max())
        high = float(probability[cut + 1 :].max())
        if not (low > 0.0 and high > 0.0):
            return math.nan
        return math.log(low) - math.log(high)

    def _phase_weight_ratio(
        self, temperature: float, keep: BoolArray, cut: int, q: float
    ) -> float:
        probability = self._histogram_from(keep, self._log_weights(temperature))
        ordered = float(probability[: cut + 1].sum())
        disordered = float(probability[cut + 1 :].sum())
        if not (ordered > 0.0 and disordered > 0.0):
            return math.nan
        return math.log(ordered / disordered) - math.log(q)

    def _solve_temperature(
        self,
        bracket: tuple[float, float],
        objective: Any,
        n_blocks: int | None,
    ) -> Estimate:
        """Root of ``objective(T, keep, cut)`` in ``bracket`` with a
        jackknife error; the phase cut is fixed from the full sample."""
        cut = self._phase_cut(bracket)
        if cut is None:
            return Estimate(math.nan, math.nan)
        full = np.ones(self.n_samples, dtype=bool)
        value = reweighting.bisect_temperature(
            lambda t: objective(t, full, cut), bracket
        )

        def block_estimator(keep: BoolArray) -> float:
            return reweighting.bisect_temperature(
                lambda t: objective(t, keep, cut), bracket
            )

        _, errors = reweighting.block_jackknife(
            block_estimator, self.n_samples, self._blocks(n_blocks)
        )
        return Estimate(value, float(errors[0]))

    def equal_height_temperature(
        self, bracket: tuple[float, float], *, n_blocks: int | None = None
    ) -> Estimate:
        """Temperature in ``bracket`` where the two energy peaks are equally high.

        The two phases are separated by a fixed energy cut, the bottom of
        the best-separated bimodal histogram found on a temperature grid
        over the bracket; the peak heights on either side of the cut are
        then equalised by bisection. This is the conventional temperature
        at which the barrier is quoted; it shifts towards the transition
        as ``L^-d``. ``nan`` when no temperature in the bracket is bimodal
        or the height difference does not change sign in it.
        """
        return self._solve_temperature(bracket, self._peak_height_difference, n_blocks)

    def equal_weight_temperature(
        self,
        bracket: tuple[float, float],
        *,
        q: float = 1.0,
        n_blocks: int | None = None,
    ) -> Estimate:
        """Temperature in ``bracket`` where the phase weights have ratio ``q``.

        The weight of the low-energy phase (bins up to the fixed cut, see
        :meth:`equal_height_temperature`) over the high-energy phase
        equals ``q``, the number of ordered states per disordered one
        (``q = 6`` for the layered phase of the cubic lattice, ``q = 1``
        for equal weights); this estimator of the transition temperature
        has exponentially small finite-size corrections (Borgs & Kotecký
        1990). ``nan`` when no root lies in the bracket.
        """
        if not _is_finite(q) or q <= 0:
            msg = f"q must be positive and finite, got {q}"
            raise ConfigurationError(msg)
        return self._solve_temperature(
            bracket,
            lambda t, keep, cut: self._phase_weight_ratio(t, keep, cut, q),
            n_blocks,
        )

    def log_density_of_states(self) -> FloatArray:
        """``ln g`` refined by the production histogram, up to a constant.

        ``ln g_prod(E) = ln H_prod(E) + ln g_WL(E)``: the frozen-weight
        chain visits bin ``E`` in proportion to ``g(E) W(E)``, so its
        histogram corrects the Wang-Landau estimate. The difference to
        :attr:`log_g` is the measured error of the weights; ``nan`` where
        production never visited.
        """
        with np.errstate(divide="ignore"):
            refined = self.log_g + np.log(self.production_histogram.astype(np.float64))
        refined = np.where(self.production_histogram > 0, refined, np.nan)
        finite = refined[np.isfinite(refined)]
        if finite.size:
            refined = (
                refined - finite.max() + np.nanmax(self.log_g[np.isfinite(refined)])
            )
        return np.asarray(refined, dtype=np.float64)

    def summary(self, temperatures: Sequence[float] | None = None) -> None:
        """Print the run diagnostics and, when given, a table of reweighted estimates.

        The diagnostics line says loudly when the Wang-Landau stage did
        not converge; the table shows energy, specific heat, order
        parameter, its Binder cumulant, the effective sample size and the
        edge weight at every temperature.
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()
        wl = self.wang_landau
        prod = self.production
        status = (
            "converged" if wl.converged else "NOT CONVERGED (capped by max_wl_sweeps)"
        )
        layout = (
            f"; {wl.n_windows} window(s) x {wl.walkers_per_window} walker(s)"
            if wl.n_windows > 1 or wl.walkers_per_window > 1
            else ""
        )
        console.print(
            f"Wang-Landau: {status}; {wl.n_iterations} modification factors, "
            f"final ln f = {wl.final_log_f:.2e}, {wl.total_sweeps} sweeps, "
            f"{wl.visited_bins} visited bins{layout}"
        )
        console.print(
            f"Production: {prod.n_walkers} walker(s) x {prod.sweeps_per_walker} "
            f"sweeps; histogram flatness {prod.histogram_flatness:.3f}; round "
            f"trips {list(prod.round_trips)}"
        )
        if not temperatures:
            return
        table = Table(title="Reweighted canonical estimates", border_style="blue")
        for column in ("T", "<E>/N", "Cv/N", "psi", "U4(psi)", "n_eff", "edge w"):
            table.add_column(column, justify="right")
        for est in self.reweight_curve(temperatures):
            table.add_row(
                f"{est.temperature:.4f}",
                str(est.energy),
                str(est.specific_heat),
                str(est.order_parameter),
                str(est.order_binder),
                f"{est.effective_samples:.0f}",
                f"{est.edge_weight:.1e}",
            )
        console.print(table)


class WangLandauSimulation:
    """Run a Wang-Landau estimate of the density of states and a production stage.

    Parameters
    ----------
    config : WangLandauConfig
        Complete configuration.

    Examples
    --------
    >>> from mcising import LatticeConfig, WangLandauConfig, WangLandauSimulation
    >>> config = WangLandauConfig(
    ...     lattice=LatticeConfig(size=4, j1=1.0),
    ...     log_f_final=1e-4,
    ...     production_sweeps=2000,
    ... )
    >>> results = WangLandauSimulation(config).run(show_progress=False)
    >>> results.wang_landau.converged
    True
    >>> -1.7 < results.reweight(2.269).energy.value < -1.4   # exact: -1.566
    True
    """

    def __init__(self, config: WangLandauConfig) -> None:
        self.config: Final[WangLandauConfig] = config

    def run(
        self,
        *,
        show_progress: bool = True,
        initial_log_g: FloatArray | Sequence[float] | None = None,
    ) -> WangLandauResults:
        """Execute both stages and return the results.

        Parameters
        ----------
        show_progress : bool
            Show a spinner with the elapsed time (the run is one call
            into the Rust core).
        initial_log_g : array_like, optional
            Starting estimate of ``ln g`` with one entry per energy bin
            (``nan`` = unknown), e.g. the :attr:`WangLandauResults.log_g`
            or :meth:`WangLandauResults.log_density_of_states` of an
            earlier run. With ``max_wl_sweeps=0`` it is used as the
            frozen production weights unchanged.

        Returns
        -------
        WangLandauResults
            Density of states, production series, diagnostics and
            provenance.
        """
        config = self.config
        start = time.monotonic()
        metadata: dict[str, object] = {
            "config": config,
            "kind": "wang_landau",
            "version": package_version(),
            "schema_version": WANG_LANDAU_SCHEMA_VERSION,
            "seed": config.seed,
        }
        commit = git_commit()
        if commit is not None:
            metadata["git_commit"] = commit
        initial = (
            None
            if initial_log_g is None
            else np.ascontiguousarray(initial_log_g, dtype=np.float64)
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            disable=not show_progress,
        ) as progress:
            progress.add_task(
                f"Wang-Landau on {config.lattice.num_sites} spins, then "
                f"{config.n_walkers} production walker(s)...",
                total=None,
            )
            raw = _run_wang_landau(
                lattice_size=config.lattice.size,
                j1=config.lattice.j1,
                j2=config.lattice.j2,
                j3=config.lattice.j3,
                h=config.lattice.h,
                base_seed=config.seed,
                lattice_type=config.lattice.lattice_type.value,
                production_sweeps=config.production_sweeps,
                measurement_interval=config.measurement_interval,
                n_walkers=config.n_walkers,
                production_thermalization=config.production_thermalization,
                store_configs=config.store_configs,
                bin_width=config.bin_width,
                energy_window=config.energy_window,
                initial_log_g=initial,
                flatness=config.flatness,
                log_f_initial=config.log_f_initial,
                log_f_final=config.log_f_final,
                check_interval=config.check_interval,
                max_wl_sweeps=config.max_wl_sweeps,
                drive_beta=config.drive_beta,
                drive_max_sweeps=config.drive_max_sweeps,
                n_windows=config.n_windows,
                walkers_per_window=config.walkers_per_window,
                window_overlap=config.window_overlap,
                exchange_interval=config.exchange_interval,
            )
        results = WangLandauResults._from_raw(raw, metadata)
        results.metadata["elapsed_seconds"] = time.monotonic() - start
        return results
