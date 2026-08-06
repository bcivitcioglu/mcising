"""Statistical helpers for seed-robust Monte Carlo tests.

Monte Carlo time series are autocorrelated, so the naive standard error
``std(x)/sqrt(n)`` underestimates the true error by ``sqrt(2*tau_int)``.
Every assertion here uses a blocking (Flyvbjerg-Petersen) standard error
instead, so thresholds keep their meaning when a bug fix changes the RNG
stream.

Layering (see ROADMAP P08): the estimator layer (``naive_se``,
``blocking_se``, ``jackknife_se``, ``tau_int_blocking``) is promoted to
``mcising.statistics``; this module then re-exports it and keeps only
the ``assert_*`` layer. Keep that split intact.

Seed mechanics: use ``@pytest.mark.parametrize("seed", DEFAULT_SEEDS)``
for non-trivial runs where the failing seed should be individually
visible and rerunnable, and :func:`assert_over_seeds` for cheap checks
where the aggregate ("1/5 seeds failed" vs "5/5 failed") is the
diagnostic that matters.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_N_SIGMA",
    "DEFAULT_SEEDS",
    "assert_mean_above",
    "assert_mean_below",
    "assert_ordered_means",
    "assert_over_seeds",
    "assert_samples_agree",
    "assert_within_sigma",
    "blocking_se",
    "jackknife_se",
    "naive_se",
    "tau_int_blocking",
]

FloatArray = NDArray[np.float64]
Samples = Sequence[float] | FloatArray

#: Seeds every statistical test runs over. Fixed by the roadmap: do not
#: add, drop, or reorder entries to make a test pass.
DEFAULT_SEEDS: Final[tuple[int, ...]] = (42, 123, 7, 2024, 31337)
DEFAULT_N_SIGMA: Final[float] = 4.0
MIN_BLOCKS: Final[int] = 8


def _as_array(samples: Samples) -> FloatArray:
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size < 2:
        msg = f"need at least 2 samples, got {x.size}"
        raise ValueError(msg)
    if not np.all(np.isfinite(x)):
        msg = "samples contain non-finite values"
        raise ValueError(msg)
    return x


# --- estimator layer (promoted to mcising.statistics in P08) ------------


def naive_se(samples: Samples) -> float:
    """Standard error of the mean assuming independent samples."""
    x = _as_array(samples)
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def blocking_se(samples: Samples, *, min_blocks: int = MIN_BLOCKS) -> float:
    """Standard error of the mean corrected for autocorrelation.

    Flyvbjerg-Petersen blocking: average adjacent pairs, recompute the
    naive standard error, repeat. For correlated data the estimate grows
    with block length and plateaus once blocks exceed the
    autocorrelation time. Returns the maximum over all levels that still
    hold ``min_blocks`` blocks — a conservative (upper-bound) plateau
    estimate, which is the safe side for a test threshold.
    """
    x = _as_array(samples)
    se = float(np.std(x, ddof=1) / math.sqrt(x.size))
    while x.size // 2 >= min_blocks:
        n_pairs = x.size // 2
        x = 0.5 * (x[: 2 * n_pairs : 2] + x[1 : 2 * n_pairs : 2])
        se = max(se, float(np.std(x, ddof=1) / math.sqrt(x.size)))
    return se


def tau_int_blocking(samples: Samples) -> float:
    """Integrated autocorrelation time implied by the blocking SE.

    ``(se_blocked / se_naive)**2 = 2 * tau_int`` in units of the
    sampling interval. Reported in failure messages so a red test says
    why it is red (bad physics vs undersampled).
    """
    naive = naive_se(samples)
    if naive == 0.0:
        return 0.5
    ratio = blocking_se(samples) / naive
    return max(0.5, 0.5 * ratio * ratio)


def jackknife_se(
    samples: Samples,
    estimator: Callable[[FloatArray], float],
    *,
    n_blocks: int = 20,
) -> float:
    """Delete-one-block jackknife error for a nonlinear estimator.

    Contiguous blocks (not single samples) so autocorrelation inside a
    block is absorbed. Use for variances (Cv, chi), ratios, cumulants.
    Unused by P00 tests; it is the promotion target for P08's error
    analysis.
    """
    x = _as_array(samples)
    n_blocks = max(2, min(n_blocks, x.size))
    blocks = np.array_split(np.arange(x.size), n_blocks)
    values = np.array(
        [estimator(np.delete(x, idx)) for idx in blocks], dtype=np.float64
    )
    spread = float(np.sum((values - values.mean()) ** 2))
    return math.sqrt((n_blocks - 1) / n_blocks * spread)


# --- assertion layer (stays in tests/) ----------------------------------


def _describe(label: str, samples: Samples) -> str:
    x = _as_array(samples)
    return (
        f"{label}: n={x.size} mean={float(x.mean()):.5f} "
        f"blocking_se={blocking_se(x):.5f} naive_se={naive_se(x):.5f} "
        f"tau_int~{tau_int_blocking(x):.1f}"
    )


def _sigmas(deviation: float, err: float) -> str:
    return f"{deviation / err:.1f}" if err > 0.0 else "inf"


def assert_within_sigma(
    samples: Samples,
    expected: float,
    *,
    n_sigma: float = DEFAULT_N_SIGMA,
    se: float | None = None,
    label: str = "sample mean",
) -> None:
    """Assert the sample mean agrees with an exact expected value.

    Two-sided. Only use where ``expected`` is analytically known for
    this finite lattice (e.g. the beta->0 limit), never against a value
    read off a previous run.
    """
    x = _as_array(samples)
    err = blocking_se(x) if se is None else se
    mean = float(x.mean())
    deviation = abs(mean - expected)
    assert deviation <= n_sigma * err, (
        f"{label} is {_sigmas(deviation, err)} sigma from expected "
        f"(limit {n_sigma}): mean={mean:.5f} expected={expected:.5f} "
        f"se={err:.5f} | {_describe(label, x)}"
    )


def assert_samples_agree(
    a: Samples,
    b: Samples,
    *,
    n_sigma: float = DEFAULT_N_SIGMA,
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    """Assert two independent estimates of the same observable agree."""
    xa, xb = _as_array(a), _as_array(b)
    err = math.hypot(blocking_se(xa), blocking_se(xb))
    deviation = abs(float(xa.mean()) - float(xb.mean()))
    assert deviation <= n_sigma * err, (
        f"{label_a} vs {label_b} disagree by "
        f"{_sigmas(deviation, err)} sigma (limit {n_sigma}): "
        f"delta={deviation:.5f} combined_se={err:.5f}\n"
        f"  {_describe(label_a, xa)}\n  {_describe(label_b, xb)}"
    )


def assert_mean_above(
    samples: Samples,
    threshold: float,
    *,
    n_sigma: float = DEFAULT_N_SIGMA,
    se: float | None = None,
    label: str = "sample mean",
) -> None:
    """Assert the mean sits above a physics-decisive threshold.

    One-sided, and the statistical error is granted as slack
    (``mean + n_sigma*se > threshold``). ``threshold`` must be a regime
    boundary placed far from the true value — document the analytic
    value it is compared against at the call site. Statistical noise
    then never flips the test red, but a wrong physical regime does.
    """
    x = _as_array(samples)
    err = blocking_se(x) if se is None else se
    mean = float(x.mean())
    assert mean + n_sigma * err > threshold, (
        f"{label} below threshold: mean={mean:.5f} "
        f"threshold={threshold:.5f} se={err:.5f} "
        f"(short by {_sigmas(threshold - mean, err)} sigma) | "
        f"{_describe(label, x)}"
    )


def assert_mean_below(
    samples: Samples,
    threshold: float,
    *,
    n_sigma: float = DEFAULT_N_SIGMA,
    se: float | None = None,
    label: str = "sample mean",
) -> None:
    """Mirror of :func:`assert_mean_above`."""
    x = _as_array(samples)
    err = blocking_se(x) if se is None else se
    mean = float(x.mean())
    assert mean - n_sigma * err < threshold, (
        f"{label} above threshold: mean={mean:.5f} "
        f"threshold={threshold:.5f} se={err:.5f} "
        f"(over by {_sigmas(mean - threshold, err)} sigma) | "
        f"{_describe(label, x)}"
    )


def assert_ordered_means(
    labelled: Sequence[tuple[str, Samples]],
    *,
    increasing: bool = True,
    n_sigma: float = DEFAULT_N_SIGMA,
) -> None:
    """Assert means are ordered, tolerating n_sigma of statistical noise.

    Fails only when the ordering is violated significantly, so a
    marginal pair cannot go red on an RNG-stream change while a genuine
    inversion (e.g. a broken sign convention) still fails loudly.
    """
    for (label_a, a), (label_b, b) in zip(labelled, labelled[1:]):
        xa, xb = _as_array(a), _as_array(b)
        mean_a, mean_b = float(xa.mean()), float(xb.mean())
        err = math.hypot(blocking_se(xa), blocking_se(xb))
        gap = mean_b - mean_a if increasing else mean_a - mean_b
        assert gap > -n_sigma * err, (
            f"ordering violated: {label_a}={mean_a:.5f} then "
            f"{label_b}={mean_b:.5f} "
            f"({'increasing' if increasing else 'decreasing'} expected), "
            f"off by {_sigmas(-gap, err)} sigma (limit {n_sigma})\n"
            f"  {_describe(label_a, xa)}\n  {_describe(label_b, xb)}"
        )


def assert_over_seeds(
    fn: Callable[[int], None],
    seeds: Iterable[int] = DEFAULT_SEEDS,
    *,
    min_passing: int | None = None,
) -> None:
    """Run ``fn(seed)`` for every seed and report all failures together.

    Use for cheap checks where five pytest ids would be noise, and for
    any check where the aggregate ("1/5 seeds failed" vs "5/5 failed")
    is the diagnostic that matters after an RNG-stream change.
    ``min_passing`` exists for observables with a genuine, quantified
    metastable branch; using it needs a comment naming that branch.
    """
    all_seeds = tuple(seeds)
    failures: list[str] = []
    for seed in all_seeds:
        try:
            fn(seed)
        except AssertionError as exc:
            failures.append(f"  seed={seed}: {exc}")
    required = len(all_seeds) if min_passing is None else min_passing
    n_passed = len(all_seeds) - len(failures)
    assert n_passed >= required, (
        f"{n_passed}/{len(all_seeds)} seeds passed (need {required}):\n"
        + "\n".join(failures)
    )
