"""Unit tests of the reweighting primitives on exact and synthetic data."""

from __future__ import annotations

import math

import numpy as np
import pytest
from mcising import reweighting
from mcising.reweighting import (
    bisect_temperature,
    block_jackknife,
    block_slices,
    canonical_histogram,
    effective_sample_size,
    log_weights,
    smooth_histogram,
    two_peak_barrier,
    weighted_mean,
)


class TestLogWeights:
    def test_shifted_to_zero_maximum(self) -> None:
        lw = log_weights([-4.0, 0.0, 4.0], [1.0, 2.0, 3.0], beta=0.5)
        assert lw.max() == 0.0
        expected = np.array([3.0, 2.0, 1.0])
        np.testing.assert_allclose(lw, expected - expected.max())

    def test_undefined_log_g_gets_zero_weight(self) -> None:
        lw = log_weights([0.0, 0.0], [math.nan, 1.0], beta=1.0)
        assert lw[0] == -math.inf
        assert lw[1] == 0.0

    def test_empty(self) -> None:
        assert log_weights([], [], beta=1.0).size == 0


class TestWeightedMean:
    def test_uniform_weights_are_the_plain_mean(self) -> None:
        values = np.array([1.0, 2.0, 6.0])
        assert weighted_mean(values, np.zeros(3)) == pytest.approx(3.0)

    def test_reweights_a_uniform_sample_to_boltzmann(self) -> None:
        # Two levels with degeneracies 1 and 3 sampled uniformly: the
        # canonical mean energy follows from the weights alone.
        energies = np.array([0.0, 1.0, 1.0, 1.0])
        lw = log_weights(energies, np.zeros(4), beta=1.0)
        expected = 3.0 * math.exp(-1.0) / (1.0 + 3.0 * math.exp(-1.0))
        assert weighted_mean(energies, lw) == pytest.approx(expected)

    def test_no_weight_is_nan(self) -> None:
        assert math.isnan(weighted_mean([1.0], np.array([-math.inf])))
        assert math.isnan(weighted_mean([], np.array([])))


class TestEffectiveSampleSize:
    def test_equal_weights_count_every_sample(self) -> None:
        assert effective_sample_size(np.zeros(50)) == pytest.approx(50.0)

    def test_one_dominant_sample(self) -> None:
        lw = np.full(50, -40.0)
        lw[3] = 0.0
        assert effective_sample_size(lw) == pytest.approx(1.0, rel=1e-6)

    def test_no_weight_is_nan(self) -> None:
        assert math.isnan(effective_sample_size(np.array([-math.inf, -math.inf])))


class TestCanonicalHistogram:
    def test_normalised_over_bins(self) -> None:
        bins = np.array([0, 0, 2, 3])
        lw = np.log(np.array([1.0, 1.0, 2.0, 4.0]))
        prob = canonical_histogram(bins, lw, 5)
        np.testing.assert_allclose(prob, np.array([2.0, 0.0, 2.0, 4.0, 0.0]) / 8.0)

    def test_no_weight_is_nan(self) -> None:
        prob = canonical_histogram(
            np.array([0, 1]), np.array([-math.inf, -math.inf]), 3
        )
        assert np.all(np.isnan(prob))
        assert np.all(
            np.isnan(canonical_histogram(np.array([], dtype=int), np.array([]), 3))
        )


class TestBlockSlices:
    def test_blocks_cover_every_sample_and_never_straddle_walkers(self) -> None:
        lengths = (100, 50, 7)
        slices = block_slices(lengths, 20)
        covered = np.zeros(sum(lengths), dtype=int)
        for block in slices:
            covered[block] += 1
        assert np.all(covered == 1)
        boundaries = np.cumsum(lengths)[:-1]
        for block in slices:
            assert not any(block.start < b < block.stop for b in boundaries)
        # Proportional shares: the long walker gets most blocks, the 7-sample
        # walker at least one and at most seven.
        per_walker = [
            sum(1 for s in slices if lo <= s.start < hi)
            for lo, hi in zip((0, 100, 150), (100, 150, 157), strict=True)
        ]
        assert per_walker[0] > per_walker[1] >= 1 and 1 <= per_walker[2] <= 7

    def test_degenerate_inputs(self) -> None:
        assert block_slices((), 20) == []
        assert block_slices((0, 0), 20) == []
        assert block_slices((10,), 0) == []
        assert block_slices((3,), 20) == [slice(0, 1), slice(1, 2), slice(2, 3)]


class TestBlockJackknife:
    def test_mean_error_matches_the_standard_error(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=4000)

        def estimator(keep: np.ndarray) -> float:
            return float(x[keep].mean())

        values, errors = block_jackknife(estimator, x.size, block_slices((4000,), 20))
        assert values[0] == pytest.approx(x.mean())
        assert errors[0] == pytest.approx(x.std(ddof=1) / math.sqrt(x.size), rel=0.35)

    def test_vector_estimator_and_nan_rules(self) -> None:
        x = np.arange(10, dtype=float)

        def estimator(keep: np.ndarray) -> np.ndarray:
            kept = x[keep]
            return np.array([kept.mean(), math.nan if kept.size < 9 else kept.max()])

        values, errors = block_jackknife(estimator, 10, block_slices((10,), 5))
        assert values.shape == errors.shape == (2,)
        assert math.isfinite(errors[0])
        assert math.isnan(errors[1])
        _, errors = block_jackknife(estimator, 10, [slice(0, 10)])
        assert np.all(np.isnan(errors))


class TestTwoPeakBarrier:
    @staticmethod
    def _bimodal(valley: float, separation: int = 40, width: float = 4.0) -> np.ndarray:
        x = np.arange(120, dtype=float)
        left = np.exp(-((x - 30.0) ** 2) / (2 * width * width))
        right = np.exp(-((x - 30.0 - separation) ** 2) / (2 * width * width))
        prob = left + right
        prob[np.abs(x - 50.0) < 4.0] = np.maximum(prob[np.abs(x - 50.0) < 4.0], valley)
        return prob / prob.sum()

    def test_barrier_of_a_double_gaussian(self) -> None:
        prob = self._bimodal(valley=1e-4)
        barrier, low, high, bottom = two_peak_barrier(prob, estimator="minimum")
        assert (low, high) == (30, 70)
        assert 30 < bottom < 70
        assert barrier == pytest.approx(math.log(prob[30] / prob[bottom]))
        plateau, *_ = two_peak_barrier(prob, estimator="plateau")
        # The plateau mean sits above the minimum, so the barrier is smaller.
        assert 0.0 < plateau <= barrier

    def test_unimodal_is_nan(self) -> None:
        x = np.arange(50, dtype=float)
        prob = np.exp(-((x - 20.0) ** 2) / 8.0)
        barrier, low, high, bottom = two_peak_barrier(prob / prob.sum())
        assert math.isnan(barrier) and (low, high, bottom) == (-1, -1, -1)

    def test_unreachable_zero_bins_do_not_create_a_barrier(self) -> None:
        x = np.arange(50, dtype=float)
        prob = np.exp(-((x - 20.0) ** 2) / 8.0)
        prob[19] = 0.0  # an energy no state has, right next to the peak
        barrier, *_ = two_peak_barrier(prob / prob.sum())
        assert math.isnan(barrier)

    @staticmethod
    def _noisy_double_gaussian(
        centre_high: float, seed: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = np.arange(2000, dtype=float)
        clean = np.exp(-((x - 600.0) ** 2) / (2 * 120.0**2))
        clean += np.exp(-((x - centre_high) ** 2) / (2 * 120.0**2))
        noisy = clean * rng.lognormal(sigma=0.25, size=x.size)
        return clean / clean.sum(), noisy / noisy.sum()

    def test_smoothing_removes_the_noise_bias_of_a_deep_barrier(self) -> None:
        clean, noisy = self._noisy_double_gaussian(1400.0)
        expected = math.log(clean[600] / clean[1000])
        raw, *_ = two_peak_barrier(noisy, estimator="minimum")
        # The minimum of noisy bins is biased low, so the raw barrier is high.
        assert raw > 1.2 * expected
        barrier, low, high, bottom = two_peak_barrier(
            noisy, estimator="minimum", smooth=20
        )
        assert abs(low - 600) < 40 and abs(high - 1400) < 40
        assert 900 < bottom < 1100
        assert barrier == pytest.approx(expected, rel=0.05)

    def test_smoothing_resolves_a_shallow_barrier_from_the_noise(self) -> None:
        clean, noisy = self._noisy_double_gaussian(900.0)
        expected = math.log(clean[600] / clean[750])
        assert 0.05 < expected < 0.3  # a barrier below the per-bin noise
        raw, *_ = two_peak_barrier(noisy, estimator="minimum")
        assert math.isnan(raw) or raw > 2.0 * expected  # noise dips dominate
        barrier, low, high, _ = two_peak_barrier(noisy, estimator="minimum", smooth=30)
        assert abs(low - 600) < 60 and abs(high - 900) < 60, (low, high)
        assert barrier == pytest.approx(expected, abs=0.08)

    def test_smooth_histogram_keeps_gaps(self) -> None:
        p = np.array([1.0, 0.0, 3.0, 5.0, 0.0, 1.0])
        smoothed = smooth_histogram(p, 1)
        assert smoothed[1] == 0.0 and smoothed[4] == 0.0  # gaps stay gaps
        assert smoothed[2] == pytest.approx(4.0)  # (3 + 5) / 2, the gap ignored
        assert smoothed[0] == pytest.approx(1.0)
        np.testing.assert_array_equal(smooth_histogram(p, 0), p)
        assert smooth_histogram(np.array([]), 3).size == 0

    def test_degenerate_inputs(self) -> None:
        assert math.isnan(two_peak_barrier(np.zeros(5))[0])
        assert math.isnan(two_peak_barrier(np.array([1.0, 2.0]))[0])
        assert math.isnan(two_peak_barrier(np.full(5, math.nan))[0])
        with pytest.raises(ValueError, match="estimator"):
            two_peak_barrier(self._bimodal(1e-3), estimator="median")  # type: ignore[arg-type]


class TestBisectTemperature:
    def test_finds_a_root(self) -> None:
        root = bisect_temperature(lambda t: t * t - 2.0, (1.0, 2.0), tolerance=1e-9)
        assert root == pytest.approx(math.sqrt(2.0), abs=1e-8)

    def test_no_sign_change_or_bad_bracket_is_nan(self) -> None:
        assert math.isnan(bisect_temperature(lambda t: t + 1.0, (1.0, 2.0)))
        assert math.isnan(bisect_temperature(lambda t: t, (2.0, 1.0)))
        assert math.isnan(bisect_temperature(lambda t: math.nan, (1.0, 2.0)))

    def test_exact_endpoint_roots(self) -> None:
        assert bisect_temperature(lambda t: t - 1.0, (1.0, 2.0)) == 1.0
        assert bisect_temperature(lambda t: t - 2.0, (1.0, 2.0)) == 2.0


def test_module_is_a_leaf() -> None:
    """The primitives depend on NumPy and the statistics module only."""
    import sys

    loaded = {name for name in sys.modules if name.startswith("mcising.")}
    assert "mcising.reweighting" in loaded
    source = open(reweighting.__file__, encoding="utf-8").read()  # noqa: SIM115
    assert "from mcising.simulation" not in source
    assert "from mcising.wang_landau" not in source
    assert "from mcising.io" not in source
