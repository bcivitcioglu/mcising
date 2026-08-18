"""Tests for correlation function and correlation length."""

from __future__ import annotations

import numpy as np
import pytest
from mcising._core import IsingSimulation

from tests._stats import DEFAULT_SEEDS


class TestCorrelationFunction:
    def test_returns_distances_and_correlations(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        distances, correlations = sim.correlation_function()
        assert len(distances) > 0
        assert len(distances) == len(correlations)

    def test_distances_are_sorted(self) -> None:
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42)
        distances, _ = sim.correlation_function()
        assert all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))

    def test_all_up_zero_connected_correlation(self) -> None:
        """For all-up spins, connected correlation C(r) = <s_i s_j> - <s>^2 = 0."""
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        distances, correlations = sim.correlation_function()
        for c in correlations:
            assert c == pytest.approx(0.0, abs=1e-10)

    def test_checkerboard_negative_nn_correlation(self) -> None:
        """For checkerboard, nearest-neighbor correlation should be negative."""
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 1:
                    spins[i, j] = -1
        sim.set_spins(spins)
        distances, correlations = sim.correlation_function()
        # Find the nearest-neighbor distance (first non-zero distance = 1.0)
        nn_idx = next(i for i, d in enumerate(distances) if d > 0)
        assert distances[nn_idx] == pytest.approx(1.0)
        assert correlations[nn_idx] < 0


class TestCorrelationLength:
    def test_returns_finite_value(self) -> None:
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42)
        xi = sim.correlation_length()
        assert np.isfinite(xi)
        assert xi >= 0

    def test_ordered_state_correlation_length(self) -> None:
        """All-up state: all connected correlations are zero → short xi."""
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42)
        spins = np.ones((8, 8), dtype=np.int8)
        sim.set_spins(spins)
        xi = sim.correlation_length()
        assert np.isfinite(xi)


class TestCorrelationLengthPhysics:
    """Physics ordering for the P09 second-moment estimator (B7, #18).

    Calibrated in-phase on 16x16 Metropolis over DEFAULT_SEEDS (mean of
    50 snapshots): xi(T=2.4) in [1.06, 1.14] vs xi(T=5.0) in
    [0.67, 0.78] — a >7 sigma gap per seed. The deep ordered phase is
    deliberately NOT used as the large-xi side: below Tc the *connected*
    correlations are small localized fluctuations around the ordered
    background (measured xi(T=1.6) ~ 0.2), so xi is compared between the
    hot phase and the near-critical region where it must grow.
    """

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_xi_grows_approaching_tc_from_hot_side(self, seed: int) -> None:
        mean_xi = {}
        for temp in (5.0, 2.4):
            sim = IsingSimulation(16, 1.0, 0.0, 0.0, 0.0, seed)
            beta = 1.0 / temp
            sim.sweep(2000, beta)
            xis = []
            for _ in range(50):
                sim.sweep(10, beta)
                xis.append(sim.correlation_length())
            xis_arr = np.asarray(xis)
            assert np.all(xis_arr >= 0.0)
            assert np.all(xis_arr < 8.0), "xi exceeded L/2 on 16x16"
            mean_xi[temp] = float(xis_arr.mean())
        assert mean_xi[2.4] > mean_xi[5.0], (
            f"xi(2.4)={mean_xi[2.4]:.3f} !> xi(5.0)={mean_xi[5.0]:.3f}"
        )
