"""Tests for correlation function and correlation length."""

from __future__ import annotations

import numpy as np
import pytest
from mcising._core import IsingSimulation


class TestCorrelationFunction:
    def test_returns_distances_and_correlations(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 42)
        distances, correlations = sim.correlation_function()
        assert len(distances) > 0
        assert len(distances) == len(correlations)

    def test_distances_are_sorted(self) -> None:
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 42)
        distances, _ = sim.correlation_function()
        assert all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))

    def test_all_up_zero_connected_correlation(self) -> None:
        """For all-up spins, connected correlation C(r) = <s_i s_j> - <s>^2 = 0."""
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        distances, correlations = sim.correlation_function()
        for c in correlations:
            assert c == pytest.approx(0.0, abs=1e-10)

    def test_checkerboard_negative_nn_correlation(self) -> None:
        """For checkerboard, nearest-neighbor correlation should be negative."""
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 42)
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
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 42)
        xi = sim.correlation_length()
        assert np.isfinite(xi)
        assert xi >= 0

    def test_ordered_state_correlation_length(self) -> None:
        """All-up state: all connected correlations are zero → short xi."""
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 42)
        spins = np.ones((8, 8), dtype=np.int8)
        sim.set_spins(spins)
        xi = sim.correlation_length()
        assert np.isfinite(xi)
