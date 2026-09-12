"""Canonical reweighting of flat-histogram samples.

A multicanonical production run (:class:`mcising.WangLandauSimulation`)
samples configurations with the weight ``W(E) = 1/g(E)``, where ``g`` is
the density of states the Wang-Landau stage estimated. Because the chain
with frozen weights is an exact Markov chain for *any* positive ``W``, a
canonical average at inverse temperature ``beta`` follows from the recorded
series without bias::

    <A>_T = sum_i w_i A_i / sum_i w_i,   ln w_i = ln g(E_i) - beta E_i.

The quality of the weights only sets the variance (through how flat the
production histogram is); the honest error is the delete-one-block
jackknife of the reweighted series, with blocks that never straddle two
walkers. This module holds the pure NumPy primitives behind
:class:`mcising.WangLandauResults`; every function is total and reports
what it cannot estimate as ``nan`` (never a silent ``0.0``).

Leaf module: it imports NumPy and :mod:`mcising.statistics` only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MIN_PEAK_RATIO",
    "BarrierEstimator",
    "bisect_temperature",
    "block_jackknife",
    "block_slices",
    "canonical_histogram",
    "effective_sample_size",
    "log_weights",
    "smooth_histogram",
    "two_peak_barrier",
    "weighted_mean",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
#: How the bottom of the free-energy barrier is estimated: the mean of the
#: canonical distribution over the middle third between the two peaks (the
#: slab plateau of a periodic box), or its minimum.
BarrierEstimator = Literal["plateau", "minimum"]

#: Default number of jackknife blocks over all walkers.
DEFAULT_N_BLOCKS: Final[int] = 20
#: With at least this many walkers, one block per walker is the cleanest
#: error estimate (walkers are independent chains).
BLOCK_PER_WALKER_FROM: Final[int] = 8


def log_weights(
    energies_total: FloatArray | Sequence[float],
    log_g_of_samples: FloatArray | Sequence[float],
    beta: float,
) -> FloatArray:
    """Per-sample log canonical weights ``ln g(E_i) - beta E_i``, shifted.

    ``energies_total`` are total (not per-site) energies. The result is
    shifted so its maximum is ``0`` (no overflow in ``exp``); a sample
    whose ``ln g`` is not finite gets weight zero (``-inf``).

    Parameters
    ----------
    energies_total : array_like
        Total energy of every sample.
    log_g_of_samples : array_like
        ``ln g`` at every sample's energy bin.
    beta : float
        Inverse temperature.

    Returns
    -------
    FloatArray
        Shifted log weights, ``-inf`` where undefined.
    """
    energies = np.asarray(energies_total, dtype=np.float64).ravel()
    log_g = np.asarray(log_g_of_samples, dtype=np.float64).ravel()
    if energies.size == 0:
        return np.empty(0, dtype=np.float64)
    log_w = log_g - beta * energies
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    top = float(log_w.max())
    if not math.isfinite(top):
        return log_w
    return np.asarray(log_w - top, dtype=np.float64)


def weighted_mean(values: FloatArray | Sequence[float], log_w: FloatArray) -> float:
    """``sum_i w_i x_i / sum_i w_i`` from log weights; ``nan`` if no weight."""
    x = np.asarray(values, dtype=np.float64).ravel()
    w = np.exp(np.asarray(log_w, dtype=np.float64).ravel())
    total = float(w.sum())
    if x.size == 0 or not math.isfinite(total) or total <= 0.0:
        return math.nan
    return float((w * x).sum() / total)


def effective_sample_size(log_w: FloatArray) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2``; ``nan`` if no weight.

    Reweighting far from the temperatures the production run covers well
    concentrates the weight on a few samples; this is the number of
    samples the canonical estimate effectively rests on.
    """
    w = np.exp(np.asarray(log_w, dtype=np.float64).ravel())
    total = float(w.sum())
    squares = float((w * w).sum())
    if squares <= 0.0 or not math.isfinite(total):
        return math.nan
    return total * total / squares


def canonical_histogram(
    bin_index: NDArray[np.integer],
    log_w: FloatArray,
    n_bins: int,
) -> FloatArray:
    """Canonical probability per energy bin, ``P_k = sum_{i in k} w_i / sum_i w_i``.

    Returns an all-``nan`` array when the samples carry no weight.
    """
    bins = np.asarray(bin_index).ravel().astype(np.intp)
    w = np.exp(np.asarray(log_w, dtype=np.float64).ravel())
    total = float(w.sum())
    if bins.size == 0 or total <= 0.0 or not math.isfinite(total):
        return np.full(n_bins, math.nan)
    counts = np.bincount(bins, weights=w, minlength=n_bins)
    return np.asarray(counts[:n_bins] / total, dtype=np.float64)


def block_slices(lengths: Sequence[int], n_blocks: int) -> list[slice]:
    """Contiguous jackknife blocks over concatenated walker series.

    ``lengths`` are the series lengths of the walkers in concatenation
    order. Blocks never straddle a walker boundary: each walker receives
    a share of ``n_blocks`` proportional to its length (at least one
    block, at most one block per sample), cut into equal contiguous
    pieces. Returns slices into the concatenated series.
    """
    sizes = [int(n) for n in lengths]
    total = sum(sizes)
    if total <= 0 or n_blocks < 1:
        return []
    slices: list[slice] = []
    offset = 0
    for size in sizes:
        if size <= 0:
            continue
        share = min(size, max(1, round(n_blocks * size / total)))
        edges = np.linspace(0, size, share + 1).astype(int)
        for start, stop in zip(edges[:-1], edges[1:], strict=True):
            if stop > start:
                slices.append(slice(offset + int(start), offset + int(stop)))
        offset += size
    return slices


def block_jackknife(
    estimator: Callable[[BoolArray], FloatArray | Sequence[float] | float],
    n_samples: int,
    blocks: Sequence[slice],
) -> tuple[FloatArray, FloatArray]:
    """Values and delete-one-block jackknife errors of a vector estimator.

    ``estimator`` receives a boolean keep-mask over the ``n_samples``
    samples and returns one or more estimates computed from the kept
    samples. The error of estimate ``a`` is ``sqrt((B - 1) / B *
    sum_j (theta_j - mean_theta)^2)`` over the ``B`` blocks; it is ``nan``
    with fewer than two blocks or when a block estimate is not finite.
    """
    full = np.ones(n_samples, dtype=bool)
    values = np.atleast_1d(np.asarray(estimator(full), dtype=np.float64))
    if len(blocks) < 2:
        return values, np.full(values.shape, math.nan)
    thetas = []
    for block in blocks:
        keep = full.copy()
        keep[block] = False
        thetas.append(np.atleast_1d(np.asarray(estimator(keep), dtype=np.float64)))
    theta = np.asarray(thetas, dtype=np.float64)
    n_blocks = theta.shape[0]
    mean = theta.mean(axis=0)
    spread = ((theta - mean) ** 2).sum(axis=0)
    errors = np.sqrt((n_blocks - 1) / n_blocks * spread)
    errors[~np.all(np.isfinite(theta), axis=0)] = math.nan
    return values, np.asarray(errors, dtype=np.float64)


#: A secondary peak lower than this fraction of the main peak is not a
#: coexisting phase: the low-energy tail of any lattice has dips from the
#: degeneracy structure of its first excitations (on the square lattice
#: the level of two adjacent flipped spins sits below that of one), and
#: those must not be reported as barriers.
MIN_PEAK_RATIO: Final[float] = 1e-3


def smooth_histogram(probability: FloatArray, half_width: int) -> FloatArray:
    """Moving average over ``2 * half_width + 1`` bins, ignoring empty bins.

    Empty bins (unreachable energies, or no weight) contribute nothing
    and stay empty, so the gaps of a spectrum never drag their neighbours
    down. ``half_width = 0`` returns the input unchanged.
    """
    p = np.asarray(probability, dtype=np.float64).ravel()
    if half_width <= 0 or p.size == 0:
        return p
    mask = (p > 0.0).astype(np.float64)
    kernel = np.ones(2 * half_width + 1)
    numerator = np.convolve(p * mask, kernel, mode="same")
    denominator = np.convolve(mask, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        averaged = np.where(denominator > 0.0, numerator / denominator, 0.0)
    return np.asarray(np.where(mask > 0.0, averaged, 0.0), dtype=np.float64)


def two_peak_barrier(
    probability: FloatArray,
    *,
    estimator: BarrierEstimator = "plateau",
    min_peak_ratio: float = MIN_PEAK_RATIO,
    smooth: int = 0,
) -> tuple[float, int, int, int]:
    """Free-energy barrier ``ln(P_peak / P_bottom)`` of a bimodal histogram.

    The main peak is the global maximum; the second peak is the bin best
    separated from it, i.e. the one maximising ``ln min(P_a, P_j) - ln
    min P(between)`` over the bins between them, among the bins holding
    at least ``min_peak_ratio`` of the main peak (bins with ``P = 0`` —
    unreachable energies or weight underflow — carry no information and
    are skipped). The barrier is measured from the *lower* of the two
    peaks (the two are equal at the equal-height temperature) down to
    the bottom between them: the mean over the middle third of the
    bins between the peaks (``"plateau"``, the slab regime of a periodic
    box, Lee & Kosterlitz 1990) or the minimum (``"minimum"``). A bottom
    that is not below both peaks is not a two-phase distribution and
    yields ``nan``.

    ``smooth`` is the half-width, in bins, of the moving average applied
    before the peaks are located (:func:`smooth_histogram`): on a grid of
    thousands of bins the per-bin noise of a finite run has dips of its
    own, while the phases are hundreds of bins wide. The peak heights and
    the ``"minimum"`` bottom are read off the smoothed distribution; the
    ``"plateau"`` bottom averages the raw one.

    Returns
    -------
    tuple[float, int, int, int]
        ``(barrier, low_peak, high_peak, bottom)`` with the bin indices of
        the lower-energy peak, the higher-energy peak and the bottom
        (the argmin, or the centre of the plateau); ``(nan, -1, -1,
        -1)`` when there are not two separated peaks.
    """
    raw = np.asarray(probability, dtype=np.float64).ravel()
    missing = (math.nan, -1, -1, -1)
    if raw.size < 3 or not np.any(raw > 0.0) or not np.all(np.isfinite(raw[raw > 0.0])):
        return missing
    p = smooth_histogram(raw, smooth)
    positive = np.where(p > 0.0, p, np.inf)  # zeros carry no information
    main = int(np.argmax(p))
    floor = min_peak_ratio * p[main]
    best_dip = 0.0
    second = -1
    # Right of the main peak: running minimum over the bins strictly between.
    if main + 2 < p.size:
        between = np.minimum.accumulate(positive[main + 1 : -1])
        for j in range(main + 2, p.size):
            bottom = between[j - main - 2]
            if p[j] < floor or not math.isfinite(bottom):
                continue
            dip = math.log(min(p[main], p[j])) - math.log(bottom)
            if dip > best_dip:
                best_dip, second = dip, j
    if main >= 2:
        between = np.minimum.accumulate(positive[main - 1 : 0 : -1])
        for j in range(main - 2, -1, -1):
            bottom = between[main - j - 2]
            if p[j] < floor or not math.isfinite(bottom):
                continue
            dip = math.log(min(p[main], p[j])) - math.log(bottom)
            if dip > best_dip:
                best_dip, second = dip, j
    if second < 0 or best_dip <= 0.0:
        return missing
    low, high = sorted((main, second))
    inner = positive[low + 1 : high]
    if estimator == "minimum":
        offset = int(np.argmin(inner))
        bottom_value = float(inner[offset])
        bottom_index = low + 1 + offset
    elif estimator == "plateau":
        n_inner = inner.size
        third = max(1, n_inner // 3)
        start = (n_inner - third) // 2
        window = raw[low + 1 + start : low + 1 + start + third]
        finite = window[window > 0.0]
        if finite.size == 0:
            return missing
        bottom_value = float(finite.mean())
        bottom_index = low + 1 + start + third // 2
    else:
        msg = f"estimator must be 'plateau' or 'minimum', got {estimator!r}"
        raise ValueError(msg)
    barrier = math.log(min(p[low], p[high])) - math.log(bottom_value)
    if not barrier > 0.0:
        # The bottom is not below both peaks: a jagged single peak (the
        # degeneracy structure of a small lattice), not two phases.
        return missing
    return barrier, low, high, bottom_index


def bisect_temperature(
    function: Callable[[float], float],
    bracket: tuple[float, float],
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 200,
) -> float:
    """Root of ``function`` on ``bracket`` by bisection; ``nan`` without a sign change.

    Used for the pseudo-transition temperatures (equal peak heights or
    equal phase weights), whose defining function is a reweighted
    quantity that is monotone across the transition in practice.
    """
    low, high = float(bracket[0]), float(bracket[1])
    if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
        return math.nan
    f_low, f_high = function(low), function(high)
    if not (math.isfinite(f_low) and math.isfinite(f_high)) or f_low * f_high > 0.0:
        return math.nan
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        f_mid = function(mid)
        if not math.isfinite(f_mid):
            return math.nan
        if f_mid == 0.0 or high - low <= tolerance:
            return mid
        if f_mid * f_low < 0.0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return 0.5 * (low + high)
