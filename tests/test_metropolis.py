"""Tests for Metropolis algorithm behavior via the Python API."""

from __future__ import annotations

import numpy as np
import pytest
from mcising._core import IsingSimulation


class TestMetropolisSweep:
    def test_sweep_returns_accepted_attempted(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 42)
        accepted, attempted = sim.sweep(1, 1.0)
        assert attempted == 16  # One sweep = N attempted flips
        assert 0 <= accepted <= attempted

    def test_multiple_sweeps(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 42)
        accepted, attempted = sim.sweep(5, 1.0)
        assert attempted == 80  # 5 sweeps * 16 sites

    def test_high_temp_high_acceptance(self) -> None:
        """At very high T (small beta), almost all moves accepted."""
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 42)
        accepted, attempted = sim.sweep(100, 0.001)  # beta ~ 0
        assert accepted / attempted > 0.95

    def test_low_temp_energy_decreases(self) -> None:
        """At very low T, energy should decrease or stay at ground state."""
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 42)
        # Run many sweeps at low T (high beta)
        sim.sweep(500, 10.0)
        energy = sim.energy()
        # Ground state energy is -2.0 per site
        assert energy < -1.5

    def test_ground_state_stability(self) -> None:
        """Starting from ground state at T=0 (large beta), system stays."""
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)

        e_before = sim.energy()
        sim.sweep(100, 100.0)  # Very high beta ≈ T→0
        e_after = sim.energy()

        assert e_after == pytest.approx(e_before)

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces identical results."""
        sim1 = IsingSimulation(8, 1.0, 0.0, 0.0, 123)
        sim2 = IsingSimulation(8, 1.0, 0.0, 0.0, 123)

        sim1.sweep(10, 0.5)
        sim2.sweep(10, 0.5)

        assert np.array_equal(sim1.get_spins(), sim2.get_spins())
        assert sim1.energy() == sim2.energy()

    def test_different_seeds_diverge(self) -> None:
        """Different seeds produce different trajectories."""
        sim1 = IsingSimulation(8, 1.0, 0.0, 0.0, 1)
        sim2 = IsingSimulation(8, 1.0, 0.0, 0.0, 2)

        sim1.sweep(50, 0.5)
        sim2.sweep(50, 0.5)

        assert not np.array_equal(sim1.get_spins(), sim2.get_spins())
