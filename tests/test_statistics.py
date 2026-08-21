"""Validation suite for ``mcising.statistics``.

Synthetic-data gates (ROADMAP P08): the jackknife Cv error must agree
with a moving-block bootstrap within 10%, and binning analysis must
recover the known integrated autocorrelation time of an AR(1) process
within 15%. Both are asserted on the mean over ``DEFAULT_SEEDS`` (the
estimator's *bias*), with a looser per-seed bound covering the
single-series noise measured during phase planning (~±10%).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import mcising.statistics as mstats
import numpy as np
import pytest
from mcising.statistics import (
    MIN_JACKKNIFE_SAMPLES,
    Estimate,
    as_float_array,
    auto_n_blocks,
    binder_cumulant,
    blocking_curve,
    blocking_se,
    jackknife_estimate,
    jackknife_se,
    mean_estimate,
    naive_se,
    observable_statistics,
    specific_heat,
    susceptibility,
    tau_int,
    tau_int_blocking,
)
from numpy.typing import NDArray

from tests import _stats
from tests._stats import DEFAULT_SEEDS

FloatArray = NDArray[np.float64]


def _ar1_series(rho: float, n: int, seed: int) -> FloatArray:
    """Stationary AR(1) series with unit variance and known tau_int.

    ``x[t] = rho * x[t-1] + sqrt(1 - rho**2) * eps[t]`` started from a
    stationary draw, so no burn-in is needed. Exact
    ``tau_int = (1 + rho) / (2 * (1 - rho))``.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n) * math.sqrt(1.0 - rho * rho)
    w[0] = rng.standard_normal()
    step = np.frompyfunc(lambda acc, inc: rho * acc + inc, 2, 1)
    return step.accumulate(w, dtype=np.object_).astype(np.float64)


def _exact_tau_int(rho: float) -> float:
    return (1.0 + rho) / (2.0 * (1.0 - rho))


def _moving_block_bootstrap_se(
    x: FloatArray,
    estimator: Callable[[FloatArray], float],
    *,
    block_length: int,
    n_replicates: int,
    rng: np.random.Generator,
) -> float:
    """Test-local moving-block bootstrap reference (not shipped)."""
    n = x.size
    n_blocks = n // block_length
    offsets = np.arange(block_length)
    replicates = np.empty(n_replicates)
    for r in range(n_replicates):
        starts = rng.integers(0, n - block_length + 1, size=n_blocks)
        replicates[r] = estimator(x[(starts[:, None] + offsets).ravel()])
    return float(np.std(replicates, ddof=1))


@pytest.mark.statistical
class TestAr1TauRecovery:
    """Binning analysis recovers the known AR(1) autocorrelation time."""

    @pytest.mark.parametrize("rho", [0.8, 0.9])
    def test_tau_int_within_15_percent(self, rho: float) -> None:
        exact = _exact_tau_int(rho)
        estimates = [
            tau_int(_ar1_series(rho, 2**16, seed), min_blocks=256)
            for seed in DEFAULT_SEEDS
        ]
        # Per-seed noise measured at ~±10% during planning; 30% is the
        # single-series sanity bound, 15% is the roadmap gate on the bias.
        for seed, est in zip(DEFAULT_SEEDS, estimates):
            assert abs(est - exact) <= 0.30 * exact, (
                f"seed={seed}: tau_int={est:.3f} vs exact {exact:.3f}"
            )
        mean_est = float(np.mean(estimates))
        assert abs(mean_est - exact) <= 0.15 * exact, (
            f"seed-mean tau_int={mean_est:.3f} vs exact {exact:.3f} "
            f"(gate: within 15%)"
        )

    def test_conservative_tau_overestimates(self) -> None:
        # tau_int_blocking (max over levels) is documented as biased
        # high; it must bound the plateau estimate from above on
        # strongly correlated data.
        x = _ar1_series(0.9, 2**16, DEFAULT_SEEDS[0])
        assert tau_int_blocking(x) >= tau_int(x, min_blocks=256)


@pytest.mark.statistical
class TestJackknifeVsBootstrap:
    """Jackknife Cv error agrees with a moving-block bootstrap."""

    def test_specific_heat_error_within_10_percent(self) -> None:
        n, n_blocks, n_replicates = 2**14, 64, 500
        block_length = n // n_blocks
        temperature, num_sites = 2.269, 1024.0

        def cv(arr: FloatArray) -> float:
            return specific_heat(
                arr, temperature=temperature, num_sites=num_sites
            )

        ratios = []
        for seed in DEFAULT_SEEDS:
            x = _ar1_series(0.8, n, seed)
            jack = jackknife_se(x, cv, n_blocks=n_blocks)
            boot = _moving_block_bootstrap_se(
                x,
                cv,
                block_length=block_length,
                n_replicates=n_replicates,
                rng=np.random.default_rng(seed + 10_000),
            )
            ratios.append(jack / boot)
            # Per-seed max deviation measured at 6.6% during planning.
            assert abs(ratios[-1] - 1.0) <= 0.25, (
                f"seed={seed}: jackknife/bootstrap={ratios[-1]:.3f}"
            )
        mean_ratio = float(np.mean(ratios))
        assert abs(mean_ratio - 1.0) <= 0.10, (
            f"seed-mean jackknife/bootstrap={mean_ratio:.3f} "
            f"(gate: within 10%)"
        )


class TestDeterministicAnchors:
    """Closed-form identities the estimators must reproduce."""

    def test_delete_one_jackknife_of_mean_equals_naive_se(self) -> None:
        x = np.random.default_rng(42).standard_normal(101)
        jack = jackknife_se(x, lambda a: float(a.mean()), n_blocks=x.size)
        assert jack == pytest.approx(naive_se(x), rel=1e-12)

    @pytest.mark.statistical
    def test_jackknife_variance_of_iid_gaussian(self) -> None:
        # SE of the sample variance of n iid N(0, sigma^2) samples is
        # sigma^2 * sqrt(2 / (n - 1)).
        n = 2048
        exact = math.sqrt(2.0 / (n - 1))
        for seed in DEFAULT_SEEDS:
            x = np.random.default_rng(seed).standard_normal(n)
            jack = jackknife_se(x, lambda a: float(np.var(a)), n_blocks=n)
            assert jack == pytest.approx(exact, rel=0.15), f"seed={seed}"

    def test_binder_cumulant_two_delta_is_two_thirds(self) -> None:
        # m = +/- m0 with equal weight: <m^4> = m0^4 = <m^2>^2, so
        # U4 = 1 - 1/3 = 2/3 exactly, with exactly zero jackknife
        # spread (every delete-one-block value is also 2/3).
        m0 = 0.75
        m = m0 * np.tile([1.0, -1.0], 64)
        est = jackknife_estimate(m, binder_cumulant, n_blocks=16)
        assert est.value == pytest.approx(2.0 / 3.0, abs=1e-12)
        assert est.error == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.statistical
    def test_binder_cumulant_gaussian_is_zero(self) -> None:
        # For Gaussian m: <m^4> = 3 <m^2>^2, so U4 = 0. Checks value
        # and error together: the value must sit within 4 sigma of 0
        # using its own jackknife error.
        for seed in DEFAULT_SEEDS:
            m = np.random.default_rng(seed).standard_normal(8192)
            est = jackknife_estimate(m, binder_cumulant)
            assert est.error > 0.0
            assert abs(est.value) <= 4.0 * est.error, f"seed={seed}"


class TestSusceptibilityKind:
    """P10 (#39): connected chi is the default; signed stays selectable."""

    def test_connected_default_equals_var_of_abs(self) -> None:
        # Sign-flipping ordered series: m = +/- 0.9 with tiny noise.
        rng = np.random.default_rng(42)
        m = 0.9 * np.tile([1.0, -1.0], 512) + 0.01 * rng.standard_normal(1024)
        chi_default = susceptibility(m, temperature=2.0, num_sites=256)
        chi_connected = susceptibility(
            m, temperature=2.0, num_sites=256, kind="connected"
        )
        assert chi_default == chi_connected
        assert chi_default == pytest.approx(256 * np.var(np.abs(m)) / 2.0)

    def test_signed_kind_is_pre_p10_form(self) -> None:
        rng = np.random.default_rng(42)
        m = 0.9 * np.tile([1.0, -1.0], 512) + 0.01 * rng.standard_normal(1024)
        chi_signed = susceptibility(m, temperature=2.0, num_sites=256, kind="signed")
        assert chi_signed == pytest.approx(256 * np.var(m) / 2.0)
        # The whole point of #39: on a sign-flipping ordered series the
        # signed form is inflated by orders of magnitude.
        chi_connected = susceptibility(m, temperature=2.0, num_sites=256)
        assert chi_signed > 100 * chi_connected

    def test_conventions_agree_on_positive_series(self) -> None:
        # Without sign flips, Var(|m|) == Var(m) exactly.
        rng = np.random.default_rng(7)
        m = 0.5 + 0.01 * rng.standard_normal(512)
        chi_c = susceptibility(m, temperature=2.0, num_sites=64)
        chi_s = susceptibility(m, temperature=2.0, num_sites=64, kind="signed")
        assert chi_c == pytest.approx(chi_s, rel=1e-12)

    def test_invalid_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="kind must be"):
            susceptibility(
                [0.1, 0.2],
                temperature=2.0,
                num_sites=4,
                kind="bogus",  # type: ignore[arg-type]
            )


class TestBlockingCurve:
    def test_structure_and_naive_anchor(self) -> None:
        x = _ar1_series(0.8, 4096, 42)
        lengths, ses = blocking_curve(x)
        assert lengths[0] == 1.0
        assert ses[0] == pytest.approx(naive_se(x))
        assert np.all(lengths[1:] == 2.0 * lengths[:-1])
        assert blocking_se(x) == pytest.approx(float(ses.max()))

    def test_correlated_curve_rises(self) -> None:
        x = _ar1_series(0.9, 2**14, 7)
        _, ses = blocking_curve(x)
        # Strong autocorrelation: the plateau SE must exceed the naive
        # SE by roughly sqrt(2 * tau) >> 1.
        assert float(ses[-1]) > 2.0 * float(ses[0])

    def test_auto_n_blocks_bounds(self) -> None:
        iid = np.random.default_rng(42).standard_normal(4096)
        assert auto_n_blocks(iid) == 32  # tau ~ 0.5 -> capped at max
        correlated = _ar1_series(0.95, 4096, 42)
        assert 2 <= auto_n_blocks(correlated) <= 32


class TestEdgePolicy:
    """Raw layer raises; Estimate layer is total and NaN-honest."""

    @pytest.mark.parametrize("bad", [[], [1.0]])
    def test_raw_layer_raises_on_short_series(self, bad: list[float]) -> None:
        for fn in (as_float_array, naive_se, blocking_se, tau_int_blocking):
            with pytest.raises(ValueError, match="at least 2 samples"):
                fn(bad)
        with pytest.raises(ValueError, match="at least 2 samples"):
            jackknife_se(bad, lambda a: float(a.mean()))
        with pytest.raises(ValueError, match="at least 2 samples"):
            tau_int(bad)

    def test_raw_layer_raises_on_non_finite(self) -> None:
        bad = [1.0, math.nan, 2.0]
        for fn in (as_float_array, naive_se, blocking_se, tau_int_blocking):
            with pytest.raises(ValueError, match="non-finite"):
                fn(bad)

    def test_binder_cumulant_edges(self) -> None:
        with pytest.raises(ValueError, match="at least 1 sample"):
            binder_cumulant([])
        with pytest.raises(ValueError, match="non-finite"):
            binder_cumulant([math.nan])
        assert math.isnan(binder_cumulant([0.0, 0.0, 0.0]))
        assert binder_cumulant([0.5]) == pytest.approx(2.0 / 3.0)

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_estimate_layer_is_total_and_never_quotes_zero(
        self, n: int
    ) -> None:
        x = np.linspace(1.0, 2.0, n)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mean_est = mean_estimate(x)
            jack_est = jackknife_estimate(x, lambda a: float(np.var(a)))
        if n == 0:
            assert math.isnan(mean_est.value)
            assert math.isnan(jack_est.value)
        else:
            assert mean_est.value == pytest.approx(float(x.mean()))
            assert jack_est.value == pytest.approx(float(np.var(x)))
        if n < 2:
            assert math.isnan(mean_est.error)
        if n < MIN_JACKKNIFE_SAMPLES:
            # The old code quoted 0.0 here; NaN is the honest answer.
            assert math.isnan(jack_est.error)

    def test_estimate_layer_on_non_finite(self) -> None:
        bad = [1.0, math.inf, 2.0, 3.0]
        assert math.isnan(mean_estimate(bad).error)
        assert math.isnan(
            jackknife_estimate(bad, lambda a: float(np.var(a))).error
        )

    def test_jackknife_error_finite_at_min_samples(self) -> None:
        x = np.array([1.0, 2.0, 4.0, 8.0])
        est = jackknife_estimate(x, lambda a: float(np.var(a)))
        assert math.isfinite(est.error)
        assert est.error > 0.0

    def test_estimate_str(self) -> None:
        # Parenthesized notation: two error digits in units of the
        # value's last decimal place.
        assert str(Estimate(1.2345, 0.012)) == "1.234(12)"
        assert str(Estimate(-1.95633, 0.0032)) == "-1.9563(32)"
        assert str(Estimate(0.666488, 3.1e-05)) == "0.666488(31)"
        # Decade round-up: 0.0995 -> (10) at one fewer decimal.
        assert str(Estimate(1.23456, 0.0995)) == "1.23(10)"
        # Fallbacks: large, zero, and unquotable errors.
        assert str(Estimate(38.7658, 18.0)) == "39 ± 18"
        assert str(Estimate(-2.0, 0.0)) == "-2 ± 0"
        assert "n/a" in str(Estimate(1.2345, math.nan))


class TestObservableStatistics:
    def test_full_statistics_on_synthetic_run(self) -> None:
        rng = np.random.default_rng(42)
        e = -1.5 + 0.05 * rng.standard_normal(400)
        m = 0.9 + 0.02 * rng.standard_normal(400)
        stats = observable_statistics(2.0, e, m, 1024)
        assert stats.n_samples == 400
        assert stats.tau_int >= 0.5
        assert stats.energy.value == pytest.approx(float(e.mean()))
        assert stats.abs_magnetization.value == pytest.approx(
            float(np.abs(m).mean())
        )
        # Value consistency: the quoted point values are exactly the
        # legacy formulas.
        assert stats.specific_heat.value == pytest.approx(
            specific_heat(e, temperature=2.0, num_sites=1024)
        )
        assert stats.susceptibility.value == pytest.approx(
            susceptibility(m, temperature=2.0, num_sites=1024)
        )
        for est in (
            stats.energy,
            stats.magnetization,
            stats.abs_magnetization,
            stats.specific_heat,
            stats.susceptibility,
            stats.binder_cumulant,
        ):
            assert math.isfinite(est.value)
            assert est.error > 0.0

    def test_unknown_num_sites_gives_nan_not_wrong_scale(self) -> None:
        rng = np.random.default_rng(7)
        e, m = rng.standard_normal(100), rng.standard_normal(100)
        stats = observable_statistics(2.0, e, m, None)
        assert math.isnan(stats.specific_heat.value)
        assert math.isnan(stats.susceptibility.value)
        # Quantities that need no site count survive.
        assert math.isfinite(stats.energy.value)
        assert math.isfinite(stats.binder_cumulant.value)

    def test_degenerate_series_is_total(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            stats = observable_statistics(2.0, [], [], 16)
        assert stats.n_samples == 0
        assert math.isnan(stats.energy.value)
        assert math.isnan(stats.tau_int)

    def test_all_zero_magnetization_is_total(self) -> None:
        e = np.full(50, -2.0)
        m = np.zeros(50)
        stats = observable_statistics(2.0, e, m, 16)
        assert stats.energy.value == pytest.approx(-2.0)
        assert stats.energy.error == 0.0  # genuinely constant series
        assert math.isnan(stats.binder_cumulant.value)  # U4 undefined


class TestWrapperFidelity:
    """tests/_stats.py must re-export, not fork, the estimator layer."""

    def test_reexports_are_identical_objects(self) -> None:
        assert _stats.naive_se is mstats.naive_se
        assert _stats.blocking_se is mstats.blocking_se
        assert _stats.jackknife_se is mstats.jackknife_se
        assert _stats.tau_int_blocking is mstats.tau_int_blocking
        assert _stats._as_array is mstats.as_float_array

    def test_wrapper_all_unchanged(self) -> None:
        assert _stats.__all__ == [
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
