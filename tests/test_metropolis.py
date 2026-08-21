"""Tests for Metropolis algorithm behavior via the Python API."""

from __future__ import annotations

import numpy as np
import pytest
from mcising._core import IsingSimulation

from tests._stats import DEFAULT_SEEDS, assert_mean_above, assert_over_seeds


class TestMetropolisSweep:
    def test_sweep_returns_accepted_attempted(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        accepted, attempted, _ = sim.sweep(1, temperature=1.0)
        assert attempted == 16  # One sweep = N attempted flips
        assert 0 <= accepted <= attempted

    def test_multiple_sweeps(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        accepted, attempted, _ = sim.sweep(5, temperature=1.0)
        assert attempted == 80  # 5 sweeps * 16 sites

    @pytest.mark.statistical
    def test_high_temp_high_acceptance(self) -> None:
        """At T=1000 (beta=0.001) the analytic acceptance floor is exp(-8*beta) = 0.992.

        The worst case on a J1-only square lattice is flipping a fully
        aligned spin, dE = 8*J1, accepted with exp(-beta*dE); every other
        move is at least as likely. The 0.95 threshold sits well below
        that floor.
        """

        def check(seed: int) -> None:
            sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, seed)
            rates = np.empty(100)
            for i in range(100):
                accepted, attempted, _ = sim.sweep(1, temperature=1000.0)
                rates[i] = accepted / attempted
            assert_mean_above(rates, 0.95, label=f"acceptance (seed={seed})")

        assert_over_seeds(check)

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_low_temp_energy_approaches_ground_state(self, seed: int) -> None:
        """Annealed to T->0 the system orders: E/site -> -2.

        A direct quench to beta=10 freezes a sizeable fraction of runs
        into a two-domain-wall stripe at exactly E/site = -1.5 (16 broken
        bonds on an 8x8 torus) — physics, not a bug. Anneal instead, and
        put the threshold at -1.75, between the stripe plateau (-1.5) and
        the ground state (-2.0); at beta=10 no other energies are
        thermally accessible.
        """
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, seed)
        for beta in (0.2, 0.3, 0.4, 0.44, 0.5, 0.6, 0.8, 1.0, 2.0, 10.0):
            sim.sweep(200, temperature=1.0 / beta)
        energy = sim.energy()
        assert energy < -1.75, f"E/site={energy:.4f} after anneal (seed={seed})"

    def test_ground_state_stability(self) -> None:
        """Starting from ground state at T=0 (large beta), system stays."""
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)

        e_before = sim.energy()
        sim.sweep(100, temperature=0.01)  # T→0 limit
        e_after = sim.energy()

        assert e_after == pytest.approx(e_before)

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces identical results."""
        sim1 = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 123)
        sim2 = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 123)

        sim1.sweep(10, temperature=2.0)
        sim2.sweep(10, temperature=2.0)

        assert np.array_equal(sim1.get_spins(), sim2.get_spins())
        assert sim1.energy() == sim2.energy()

    def test_different_seeds_diverge(self) -> None:
        """Different seeds produce different trajectories."""
        sim1 = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 1)
        sim2 = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 2)

        sim1.sweep(50, temperature=2.0)
        sim2.sweep(50, temperature=2.0)

        assert not np.array_equal(sim1.get_spins(), sim2.get_spins())
