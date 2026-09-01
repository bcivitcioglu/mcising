"""Frozen dataclass configurations for Ising model simulations."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Final, TypeVar

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
    "ExecutionMode",
    "LatticeConfig",
    "AdaptiveConfig",
    "SimulationConfig",
]


class LatticeType(str, Enum):
    """Available lattice geometries."""

    SQUARE = "square"
    TRIANGULAR = "triangular"
    CHAIN = "chain"
    HONEYCOMB = "honeycomb"
    CUBIC = "cubic"


class Algorithm(str, Enum):
    """Available Monte Carlo update algorithms."""

    METROPOLIS = "metropolis"
    WOLFF = "wolff"
    SWENDSEN_WANG = "swendsen_wang"


class ExecutionMode(str, Enum):
    """Execution strategy for temperature scans.

    COOLDOWN: Sequential cool-down — carry spins from high T to low T.
        Best for avoiding metastable states. Single-threaded.
    INDEPENDENT: Each temperature runs independently from random init.
        Fully parallelized via Rayon. Uses all CPU cores.
    PARALLEL_TEMPERING: All temperatures run as one coupled
        replica-exchange ensemble with periodic swap attempts between
        adjacent temperatures. Not resumable per temperature — the
        replicas advance together.
    """

    COOLDOWN = "cooldown"
    INDEPENDENT = "independent"
    PARALLEL_TEMPERING = "parallel_tempering"


@dataclass(frozen=True)
class LatticeConfig:
    """Configuration for lattice geometry.

    Parameters
    ----------
    lattice_type : LatticeType
        Type of lattice geometry.
    size : int
        Linear size L of the lattice (creates L x L for 2D). Must be even
        for triangular and honeycomb lattices (their periodic wrap is only
        consistent for even L).
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
            raise ConfigurationError(msg)
        if (
            self.lattice_type in (LatticeType.TRIANGULAR, LatticeType.HONEYCOMB)
            and self.size % 2 != 0
        ):
            # Row-parity offset coordinates make rows 0 and L-1 share a
            # parity when L is odd, so bonds across the vertical wrap seam
            # are not reciprocal and the Hamiltonian is invalid (B2, #13).
            raise ConfigurationError(
                f"The {self.lattice_type.value} lattice requires even size L "
                "under periodic boundary conditions (odd L breaks "
                "neighbor-table symmetry across the wrap seam; odd-L support "
                f"is future work). Got size={self.size}."
            )
        if not isinstance(self.j1, (int, float)) or not _is_finite(self.j1):
            msg = f"j1 must be a finite number, got {self.j1}"
            raise ConfigurationError(msg)
        if not isinstance(self.j2, (int, float)) or not _is_finite(self.j2):
            msg = f"j2 must be a finite number, got {self.j2}"
            raise ConfigurationError(msg)
        if not isinstance(self.j3, (int, float)) or not _is_finite(self.j3):
            msg = f"j3 must be a finite number, got {self.j3}"
            raise ConfigurationError(msg)
        if not isinstance(self.h, (int, float)) or not _is_finite(self.h):
            msg = f"h must be a finite number, got {self.h}"
            raise ConfigurationError(msg)

    @property
    def num_sites(self) -> int:
        """Total number of spins N for this geometry.

        Pure function of ``(lattice_type, size)``; kept in lockstep with
        the Rust constructors by a parity test over every lattice type
        (``tests/test_simulation.py::TestNumSites``).
        """
        length = self.size
        if self.lattice_type in (LatticeType.SQUARE, LatticeType.TRIANGULAR):
            return length * length
        if self.lattice_type is LatticeType.HONEYCOMB:
            return 2 * length * length
        if self.lattice_type is LatticeType.CUBIC:
            return length * length * length
        if self.lattice_type is LatticeType.CHAIN:
            return length
        msg = f"unhandled lattice type: {self.lattice_type!r}"
        raise ConfigurationError(msg)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LatticeConfig:
        """Build a LatticeConfig from a mapping.

        Unknown keys are ignored (forward compatibility with newer file
        schemas) and missing keys take their defaults (older schemas).

        Parameters
        ----------
        data : Mapping[str, Any]
            Field values, e.g. the dict form produced by
            ``dataclasses.asdict``. ``lattice_type`` may be a
            :class:`LatticeType` or its string value.

        Returns
        -------
        LatticeConfig
            A validated configuration.

        Raises
        ------
        ConfigurationError
            If ``data`` is not a mapping or any value is invalid.
        """
        kwargs = _known_fields(cls, data, "lattice config")
        if "lattice_type" in kwargs:
            kwargs["lattice_type"] = _coerce_enum(
                LatticeType, kwargs["lattice_type"], "lattice_type"
            )
        return _construct(cls, kwargs, "lattice config")


@dataclass(frozen=True)
class AdaptiveConfig:
    """Configuration for adaptive thermalization and measurement spacing.

    When enabled, each temperature is annealed with a cool-down ramp and
    then probed with a fixed-temperature diagnostic energy series: MSER
    verifies equilibration and Sokal's windowing method estimates the
    integrated autocorrelation time (tau_int) on the stationary tail of
    that series — never across the ramp, whose energy trace is
    non-stationary by construction. The measurement interval is set to
    ``tau_multiplier * tau_int`` for independent samples.

    Parameters
    ----------
    enabled : bool
        Whether to use adaptive mode. When False (default), the simulation
        uses fixed n_sweeps / measurement_interval / n_thermalization.
    min_thermalization_sweeps : int
        Floor for the annealing-ramp length and the length of each
        fixed-temperature diagnostic block (itself floored at
        ``MIN_DIAGNOSTIC_SWEEPS``).
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
            raise ConfigurationError(msg)
        if self.max_thermalization_sweeps < self.min_thermalization_sweeps:
            msg = (
                f"max_thermalization_sweeps "
                f"({self.max_thermalization_sweeps}) must be >= "
                f"min ({self.min_thermalization_sweeps})"
            )
            raise ConfigurationError(msg)
        if self.c_window <= 0:
            msg = f"c_window must be > 0, got {self.c_window}"
            raise ConfigurationError(msg)
        if self.min_independent_samples < 1:
            msg = (
                "min_independent_samples must be >= 1, "
                f"got {self.min_independent_samples}"
            )
            raise ConfigurationError(msg)
        if self.tau_multiplier <= 0:
            msg = f"tau_multiplier must be > 0, got {self.tau_multiplier}"
            raise ConfigurationError(msg)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AdaptiveConfig:
        """Build an AdaptiveConfig from a mapping.

        Unknown keys are ignored and missing keys take their defaults;
        see :meth:`LatticeConfig.from_dict`.

        Parameters
        ----------
        data : Mapping[str, Any]
            Field values, e.g. the dict form produced by
            ``dataclasses.asdict``.

        Returns
        -------
        AdaptiveConfig
            A validated configuration.

        Raises
        ------
        ConfigurationError
            If ``data`` is not a mapping or any value is invalid.
        """
        kwargs = _known_fields(cls, data, "adaptive config")
        return _construct(cls, kwargs, "adaptive config")


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
        Collect a measurement every this many sweeps. In parallel
        tempering it must be a multiple of ``swap_interval``.
    compute_correlation : bool
        Whether to compute the spin-spin correlation function ``C(r)`` and
        the second-moment correlation length. Each evaluation is a full
        pair sum, ``O(N^2)`` in the number of sites (measured costs are on
        the performance page of the docs), so at ``measurement_interval=1``
        it dominates the run; use ``correlation_interval`` to thin it.
    correlation_interval : int
        Evaluate the correlation observables at every k-th measurement
        (the k-th, 2k-th, ...): ``1`` evaluates at every measurement,
        ``n_sweeps // measurement_interval`` exactly once, at the final
        one. The stored ``C(r)`` is the last evaluation; the correlation
        length series has one entry per evaluation. Ignored by adaptive
        mode, which always takes a single end-of-production snapshot.
    store_configs : bool
        Whether to store spin configurations at every measurement.
        Disable to cut memory and file size when only scalar
        observables are needed.
    mode : ExecutionMode
        Execution strategy. COOLDOWN (default) processes temperatures
        sequentially via cool-down. INDEPENDENT runs each temperature
        in parallel from random initialization using all CPU cores.
        PARALLEL_TEMPERING runs one coupled replica-exchange ensemble.
    swap_interval : int
        Sweeps between replica swap attempts (parallel tempering only).
        Must divide ``measurement_interval``.
    """

    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    algorithm: Algorithm = Algorithm.METROPOLIS
    seed: int = DEFAULT_SEED
    temperatures: tuple[float, ...] = (2.269,)
    n_sweeps: int = DEFAULT_N_SWEEPS
    n_thermalization: int = DEFAULT_N_THERMALIZATION
    measurement_interval: int = DEFAULT_MEASUREMENT_INTERVAL
    compute_correlation: bool = False
    correlation_interval: int = 1
    store_configs: bool = True
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    mode: ExecutionMode = ExecutionMode.COOLDOWN
    swap_interval: int = 1

    def __post_init__(self) -> None:
        if self.n_sweeps < 1:
            msg = f"n_sweeps must be >= 1, got {self.n_sweeps}"
            raise ConfigurationError(msg)
        if self.n_thermalization < 0:
            msg = f"n_thermalization must be >= 0, got {self.n_thermalization}"
            raise ConfigurationError(msg)
        if self.measurement_interval < 1:
            msg = f"measurement_interval must be >= 1, got {self.measurement_interval}"
            raise ConfigurationError(msg)
        if self.correlation_interval < 1:
            msg = f"correlation_interval must be >= 1, got {self.correlation_interval}"
            raise ConfigurationError(msg)
        n_measurements = self.n_sweeps // self.measurement_interval
        if (
            self.compute_correlation
            and not self.adaptive.enabled
            and n_measurements >= 1
            and self.correlation_interval > n_measurements
        ):
            # Every k-th measurement of fewer than k measurements is none:
            # the run would record no correlation sample at all.
            raise ConfigurationError(
                "correlation_interval exceeds the number of measurements "
                f"(n_sweeps // measurement_interval = {n_measurements}), so no "
                "correlation sample would be recorded; lower "
                f"correlation_interval (got {self.correlation_interval}) or "
                "raise n_sweeps."
            )
        if self.swap_interval < 1:
            msg = f"swap_interval must be >= 1, got {self.swap_interval}"
            raise ConfigurationError(msg)
        if (
            self.mode == ExecutionMode.PARALLEL_TEMPERING
            and self.measurement_interval % self.swap_interval != 0
        ):
            # The PT ladder advances in swap_interval-sized chunks, so a
            # measurement can only happen on a chunk boundary; a non-dividing
            # measurement_interval silently drops measurements and the short
            # arrays used to panic at the reshape boundary (B5).
            raise ConfigurationError(
                "Parallel tempering requires measurement_interval to be a "
                "multiple of swap_interval; raise measurement_interval to "
                f"the next multiple of {self.swap_interval} or choose a "
                "swap_interval that divides it. Got "
                f"measurement_interval={self.measurement_interval}, "
                f"swap_interval={self.swap_interval}."
            )
        for temp in self.temperatures:
            if temp <= 0 or not _is_finite(temp):
                msg = f"All temperatures must be positive and finite, got {temp}"
                raise ConfigurationError(msg)
        if len(self.temperatures) == 0:
            msg = "At least one temperature must be specified"
            raise ConfigurationError(msg)
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
            if self.lattice.j1 <= 0.0:
                # The Fortuin-Kasteleyn bond probability 1 - exp(-2*beta*J1)
                # is <= 0 for J1 <= 0: cluster growth never adds a site and
                # the update silently degenerates to random single flips (B1).
                raise ConfigurationError(
                    "Cluster algorithms require J1>0; use metropolis for "
                    "antiferromagnetic couplings; sublattice mapping is "
                    f"future work. Got j1={self.lattice.j1}."
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimulationConfig:
        """Build a SimulationConfig from a mapping.

        The inverse of ``dataclasses.asdict``: nested ``lattice`` and
        ``adaptive`` mappings become their config objects, enum values are
        coerced from strings, and ``temperatures`` becomes a tuple. Unknown
        keys are ignored (forward compatibility with newer file schemas)
        and missing keys take their defaults (older schemas). Validation
        in ``__post_init__`` runs as usual.

        Parameters
        ----------
        data : Mapping[str, Any]
            Field values, e.g. a decoded ``config_json`` record from a
            saved results file.

        Returns
        -------
        SimulationConfig
            A validated configuration.

        Raises
        ------
        ConfigurationError
            If ``data`` is not a mapping or any value is invalid.
        """
        kwargs = _known_fields(cls, data, "simulation config")
        if "lattice" in kwargs:
            kwargs["lattice"] = LatticeConfig.from_dict(kwargs["lattice"])
        if "adaptive" in kwargs:
            kwargs["adaptive"] = AdaptiveConfig.from_dict(kwargs["adaptive"])
        if "algorithm" in kwargs:
            kwargs["algorithm"] = _coerce_enum(
                Algorithm, kwargs["algorithm"], "algorithm"
            )
        if "mode" in kwargs:
            kwargs["mode"] = _coerce_enum(ExecutionMode, kwargs["mode"], "mode")
        if isinstance(kwargs.get("temperatures"), list):
            kwargs["temperatures"] = tuple(kwargs["temperatures"])
        return _construct(cls, kwargs, "simulation config")


def _is_finite(value: float) -> bool:
    """Check if a float is finite (not inf, -inf, or nan)."""
    return math.isfinite(value)


_E = TypeVar("_E", bound=Enum)
_C = TypeVar("_C")


def _known_fields(
    cls: type[Any], data: Mapping[str, Any], context: str
) -> dict[str, Any]:
    """Return the subset of ``data`` matching ``cls``'s dataclass fields."""
    if not isinstance(data, Mapping):
        raise ConfigurationError(
            f"A {context} record must be a mapping of field names to "
            f"values, got {type(data).__name__}"
        )
    names = {f.name for f in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


def _coerce_enum(enum_cls: type[_E], value: object, field_name: str) -> _E:
    """Coerce ``value`` to a member of ``enum_cls``."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError as exc:
        valid = ", ".join(repr(member.value) for member in enum_cls)
        raise ConfigurationError(
            f"Invalid {field_name} {value!r}; valid values are: {valid}"
        ) from exc


def _construct(cls: Callable[..., _C], kwargs: dict[str, Any], context: str) -> _C:
    """Construct a config, mapping constructor errors to ConfigurationError."""
    try:
        return cls(**kwargs)
    except ConfigurationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid {context}: {exc}") from exc
