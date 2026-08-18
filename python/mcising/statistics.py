"""Statistical error analysis for Monte Carlo observables.

Monte Carlo time series are autocorrelated, so the naive standard error
``std(x)/sqrt(n)`` underestimates the true error by ``sqrt(2*tau_int)``.
This module provides autocorrelation-aware estimators (Flyvbjerg-Petersen
blocking, delete-one-block jackknife) and a total, never-raising
``Estimate`` layer used by :class:`mcising.simulation.SimulationResults`,
plotting, and the HDF5 writer.

Layering contract
-----------------
* The **estimator layer** (:func:`naive_se`, :func:`blocking_se`,
  :func:`tau_int_blocking`, :func:`jackknife_se`, :func:`as_float_array`)
  raises :class:`ValueError` on degenerate input (fewer than 2 samples,
  non-finite values). ``tests/_stats.py`` re-exports it for the
  assertion helpers, whose loudness must be preserved.
* The **estimate layer** (:func:`mean_estimate`,
  :func:`jackknife_estimate`, :func:`observable_statistics`) is total:
  it never raises on short or degenerate series and reports what it
  cannot estimate as ``nan`` (never a silent ``0.0``). NaN *is* the
  signal — it renders as an absent error bar and an empty table cell.

This module intentionally shares its name with the standard-library
``statistics`` module; all imports inside the package are absolute, so
there is no shadowing hazard.

This is a leaf module: it must not import from ``mcising.simulation``,
``mcising.io``, or ``mcising.config``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_N_BLOCKS",
    "MAX_JACKKNIFE_BLOCKS",
    "MIN_BLOCKS",
    "MIN_JACKKNIFE_SAMPLES",
    "Estimate",
    "ObservableStatistics",
    "as_float_array",
    "auto_n_blocks",
    "binder_cumulant",
    "blocking_curve",
    "blocking_se",
    "jackknife_estimate",
    "jackknife_se",
    "mean_estimate",
    "naive_se",
    "observable_statistics",
    "specific_heat",
    "susceptibility",
    "tau_int",
    "tau_int_blocking",
]

FloatArray = NDArray[np.float64]
Samples = Sequence[float] | FloatArray

#: Minimum number of blocks a blocking level must hold to be used.
MIN_BLOCKS: Final[int] = 8
#: Default number of delete-one blocks for :func:`jackknife_se`.
DEFAULT_N_BLOCKS: Final[int] = 20
#: Below this many samples the jackknife error is reported as ``nan``:
#: at n=2 the two leave-one-out sets are singletons and the jackknife
#: spread is *exactly* 0.0 — a confidently wrong error bar.
MIN_JACKKNIFE_SAMPLES: Final[int] = 4
#: Upper bound for :func:`auto_n_blocks`; caps jackknife cost at O(32 n).
MAX_JACKKNIFE_BLOCKS: Final[int] = 32


def as_float_array(samples: Samples) -> FloatArray:
    """Validate and flatten a sample series for the estimator layer.

    Parameters
    ----------
    samples : Samples
        Sequence or array of measurements.

    Returns
    -------
    FloatArray
        1-D float64 view of the input.

    Raises
    ------
    ValueError
        If fewer than 2 samples or any value is non-finite.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size < 2:
        msg = f"need at least 2 samples, got {x.size}"
        raise ValueError(msg)
    if not np.all(np.isfinite(x)):
        msg = "samples contain non-finite values"
        raise ValueError(msg)
    return x


# --- estimator layer (raises on degenerate input) ------------------------


def naive_se(samples: Samples) -> float:
    """Standard error of the mean assuming independent samples."""
    x = as_float_array(samples)
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def blocking_se(samples: Samples, *, min_blocks: int = MIN_BLOCKS) -> float:
    """Standard error of the mean corrected for autocorrelation.

    Flyvbjerg-Petersen blocking: average adjacent pairs, recompute the
    naive standard error, repeat. For correlated data the estimate grows
    with block length and plateaus once blocks exceed the
    autocorrelation time. Returns the maximum over all levels that still
    hold ``min_blocks`` blocks — a conservative (upper-bound) plateau
    estimate. For series shorter than ``2 * min_blocks`` no blocking is
    possible and the naive standard error is returned unchanged.
    """
    x = as_float_array(samples)
    se = float(np.std(x, ddof=1) / math.sqrt(x.size))
    while x.size // 2 >= min_blocks:
        n_pairs = x.size // 2
        x = 0.5 * (x[: 2 * n_pairs : 2] + x[1 : 2 * n_pairs : 2])
        se = max(se, float(np.std(x, ddof=1) / math.sqrt(x.size)))
    return se


def blocking_curve(
    samples: Samples, *, min_blocks: int = MIN_BLOCKS
) -> tuple[FloatArray, FloatArray]:
    """Standard-error estimate at every blocking level.

    Parameters
    ----------
    samples : Samples
        Measurement series (at least 2 finite samples).
    min_blocks : int
        Stop once the next level would hold fewer than this many blocks.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        ``(block_length, se)`` arrays, one entry per level starting at
        block length 1 (the naive SE). The curve is the honest,
        inspectable object behind :func:`blocking_se` and
        :func:`tau_int`.
    """
    x = as_float_array(samples)
    length = 1
    lengths = [1.0]
    ses = [float(np.std(x, ddof=1) / math.sqrt(x.size))]
    while x.size // 2 >= min_blocks:
        n_pairs = x.size // 2
        x = 0.5 * (x[: 2 * n_pairs : 2] + x[1 : 2 * n_pairs : 2])
        length *= 2
        lengths.append(float(length))
        ses.append(float(np.std(x, ddof=1) / math.sqrt(x.size)))
    return np.asarray(lengths), np.asarray(ses)


def tau_int_blocking(samples: Samples) -> float:
    """Integrated autocorrelation time implied by :func:`blocking_se`.

    ``(se_blocked / se_naive)**2 = 2 * tau_int`` in units of the
    sampling interval, floored at the uncorrelated value 0.5.

    Because :func:`blocking_se` takes the *maximum* over levels, this
    estimate is deliberately conservative: on synthetic AR(1) data it
    overestimates tau by ~65%. That bias is the safe side for test
    thresholds and for jackknife bin sizing, which is what this function
    is for. For an accuracy-oriented estimate use :func:`tau_int`.
    """
    naive = naive_se(samples)
    if naive == 0.0:
        return 0.5
    ratio = blocking_se(samples) / naive
    return max(0.5, 0.5 * ratio * ratio)


def tau_int(samples: Samples, *, min_blocks: int = 32) -> float:
    """Integrated autocorrelation time from the blocking plateau.

    Reads the standard error at the *deepest* blocking level that still
    holds ``min_blocks`` blocks (no maximum over levels), so level-to-
    level noise does not bias the estimate upward the way
    :func:`tau_int_blocking` does. On synthetic AR(1) data with
    ``n = 2**16`` and ``min_blocks=256`` the bias is a few percent.

    Parameters
    ----------
    samples : Samples
        Measurement series (at least 2 finite samples).
    min_blocks : int
        Minimum blocks the plateau level must hold. Larger values give
        a less noisy but shallower (less bias-corrected) estimate.

    Returns
    -------
    float
        Integrated autocorrelation time in units of the sampling
        interval, floored at the uncorrelated value 0.5.
    """
    naive = naive_se(samples)
    if naive == 0.0:
        return 0.5
    _, ses = blocking_curve(samples, min_blocks=min_blocks)
    ratio = float(ses[-1]) / naive
    return max(0.5, 0.5 * ratio * ratio)


def jackknife_se(
    samples: Samples,
    estimator: Callable[[FloatArray], float],
    *,
    n_blocks: int = DEFAULT_N_BLOCKS,
) -> float:
    """Delete-one-block jackknife error for a nonlinear estimator.

    Contiguous blocks (not single samples) so autocorrelation inside a
    block is absorbed. Use for variances (Cv, chi), ratios, and
    cumulants. Blocks come from ``np.array_split``, so with
    ``n % n_blocks != 0`` block sizes differ by one and the standard
    ``(B-1)/B`` factor is an O(B/n) approximation.
    """
    x = as_float_array(samples)
    n_blocks = max(2, min(n_blocks, x.size))
    blocks = np.array_split(np.arange(x.size), n_blocks)
    values = np.array(
        [estimator(np.delete(x, idx)) for idx in blocks], dtype=np.float64
    )
    spread = float(np.sum((values - values.mean()) ** 2))
    return math.sqrt((n_blocks - 1) / n_blocks * spread)


def auto_n_blocks(samples: Samples, *, max_blocks: int = MAX_JACKKNIFE_BLOCKS) -> int:
    """Choose a jackknife block count from the autocorrelation time.

    Targets a block length of ``2 * tau_int`` (conservative estimate,
    :func:`tau_int_blocking`) so each deleted block is effectively
    independent, clipped to ``[2, max_blocks]`` blocks.
    """
    x = as_float_array(samples)
    bin_length = max(1, math.ceil(2.0 * tau_int_blocking(x)))
    return int(np.clip(x.size // bin_length, 2, max_blocks))


# --- observable estimators -----------------------------------------------


def specific_heat(
    energy: Samples, *, temperature: float, num_sites: float
) -> float:
    """Specific heat per site from a per-site energy series.

    ``Cv = N * Var(e) / T**2`` with the population variance (``ddof=0``,
    the fluctuation-dissipation form; the O(1/n) bias is far below the
    quoted statistical error).
    """
    e = np.asarray(energy, dtype=np.float64).ravel()
    return float(num_sites * np.var(e) / (temperature * temperature))


def susceptibility(
    magnetization: Samples, *, temperature: float, num_sites: float
) -> float:
    """Susceptibility per site from a per-site magnetization series.

    ``chi = N * Var(m) / T`` over the *signed* magnetization. In the
    ordered phase of a finite system the signed variance is inflated by
    global sign flips; the connected form ``Var(|m|)`` is a different
    convention with much smaller values near and below Tc. Changing the
    convention is an estimator-correctness decision deferred past P08 —
    both ``magnetization`` and ``abs_magnetization`` estimates are
    exposed in :class:`ObservableStatistics` so either can be formed.
    """
    m = np.asarray(magnetization, dtype=np.float64).ravel()
    return float(num_sites * np.var(m) / temperature)


def binder_cumulant(magnetization: Samples) -> float:
    """Binder cumulant ``U4 = 1 - <m**4> / (3 <m**2>**2)``.

    Even moments make it sign-agnostic, so the signed per-site series is
    fine. Returns ``nan`` when the second moment vanishes (identically
    zero magnetization), where U4 is undefined.

    Raises
    ------
    ValueError
        If the series is empty or contains non-finite values.
    """
    m = np.asarray(magnetization, dtype=np.float64).ravel()
    if m.size == 0:
        msg = "need at least 1 sample, got 0"
        raise ValueError(msg)
    if not np.all(np.isfinite(m)):
        msg = "samples contain non-finite values"
        raise ValueError(msg)
    m2 = float(np.mean(m * m))
    if m2 == 0.0:
        return math.nan
    m4 = float(np.mean(m**4))
    return 1.0 - m4 / (3.0 * m2 * m2)


# --- total estimate layer (never raises) ---------------------------------


@dataclass(frozen=True)
class Estimate:
    """A point value with its standard error.

    ``error`` is ``nan`` when the series was too short or degenerate to
    quote a principled uncertainty — never a silent ``0.0``.
    """

    value: float
    error: float

    def __str__(self) -> str:
        """Compact notation: ``-1.9563(32)`` = -1.9563 ± 0.0032.

        The parenthesized digits are the two-significant-digit error in
        units of the value's last decimal place. Falls back to
        ``value ± error`` when the error is zero, non-finite, or >= 1.
        """
        if not math.isfinite(self.error):
            return f"{self.value:.6g} ± n/a"
        if self.error <= 0.0 or not math.isfinite(self.value):
            return f"{self.value:.6g} ± {self.error:.2g}"
        decimals = 1 - math.floor(math.log10(self.error))
        if decimals <= 0:
            return f"{round(self.value, decimals):.0f} ± {self.error:.2g}"
        scaled = round(self.error * 10**decimals)
        if scaled >= 100:  # error like 0.0999 rounds up a decade
            scaled = 10
            decimals -= 1
        return f"{self.value:.{decimals}f}({scaled})"


@dataclass(frozen=True)
class ObservableStatistics:
    """Per-temperature observable estimates with standard errors.

    Attributes
    ----------
    temperature : float
        Temperature the series was measured at.
    n_samples : int
        Number of measurements in the energy series.
    tau_int : float
        Conservative integrated autocorrelation time of the energy
        series (:func:`tau_int_blocking`), consistent with the quoted
        blocking errors; ``nan`` if inestimable.
    energy, magnetization, abs_magnetization : Estimate
        Per-site means with blocking standard errors.
    specific_heat, susceptibility, binder_cumulant : Estimate
        Derived quantities with delete-one-block jackknife errors.
        ``specific_heat`` and ``susceptibility`` are ``nan`` when the
        site count is unknown (legacy files without configurations).
    """

    temperature: float
    n_samples: int
    tau_int: float
    energy: Estimate
    magnetization: Estimate
    abs_magnetization: Estimate
    specific_heat: Estimate
    susceptibility: Estimate
    binder_cumulant: Estimate


def mean_estimate(samples: Samples) -> Estimate:
    """Mean with blocking standard error; total (never raises).

    Returns ``Estimate(nan, nan)`` for an empty or non-finite series and
    ``Estimate(mean, nan)`` for a single sample.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        return Estimate(math.nan, math.nan)
    value = float(x.mean())
    if x.size < 2:
        return Estimate(value, math.nan)
    return Estimate(value, blocking_se(x))


def jackknife_estimate(
    samples: Samples,
    estimator: Callable[[FloatArray], float],
    *,
    n_blocks: int | None = None,
) -> Estimate:
    """Estimator value with jackknife error; total (never raises).

    The point value is quoted whenever the estimator can produce one;
    the error is ``nan`` below :data:`MIN_JACKKNIFE_SAMPLES` samples
    (where the delete-one-block spread degenerates). ``n_blocks=None``
    selects :func:`auto_n_blocks` from the series' autocorrelation time.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        return Estimate(math.nan, math.nan)
    value = float(estimator(x))
    if x.size < MIN_JACKKNIFE_SAMPLES:
        return Estimate(value, math.nan)
    blocks = auto_n_blocks(x) if n_blocks is None else n_blocks
    return Estimate(value, jackknife_se(x, estimator, n_blocks=blocks))


def observable_statistics(
    temperature: float,
    energy: Samples,
    magnetization: Samples,
    num_sites: int | None,
) -> ObservableStatistics:
    """Compute all per-temperature estimates from the raw series.

    Parameters
    ----------
    temperature : float
        Temperature the series was measured at (must be positive).
    energy : Samples
        Per-site energy measurements.
    magnetization : Samples
        Per-site signed magnetization measurements.
    num_sites : int | None
        Total site count ``N``; if ``None`` (unknown), specific heat
        and susceptibility are reported as ``nan`` estimates rather
        than silently mis-normalized.

    Returns
    -------
    ObservableStatistics
        Total result — degenerate inputs yield ``nan`` fields, never an
        exception.
    """
    e = np.asarray(energy, dtype=np.float64).ravel()
    m = np.asarray(magnetization, dtype=np.float64).ravel()

    if e.size >= 2 and bool(np.all(np.isfinite(e))):
        tau = tau_int_blocking(e)
    else:
        tau = math.nan

    if num_sites is None:
        cv_est = Estimate(math.nan, math.nan)
        chi_est = Estimate(math.nan, math.nan)
    else:
        n = float(num_sites)
        cv_est = jackknife_estimate(
            e, lambda x: specific_heat(x, temperature=temperature, num_sites=n)
        )
        chi_est = jackknife_estimate(
            m, lambda x: susceptibility(x, temperature=temperature, num_sites=n)
        )

    return ObservableStatistics(
        temperature=temperature,
        n_samples=int(e.size),
        tau_int=tau,
        energy=mean_estimate(e),
        magnetization=mean_estimate(m),
        abs_magnetization=mean_estimate(np.abs(m)),
        specific_heat=cv_est,
        susceptibility=chi_est,
        binder_cumulant=jackknife_estimate(m, binder_cumulant),
    )
