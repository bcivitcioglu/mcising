"""Tests for the honeycomb lattice."""

from __future__ import annotations

import numpy as np
from mcising._core import IsingSimulation
from mcising.config import LatticeConfig, LatticeType, SimulationConfig
from mcising.simulation import Simulation


class TestHoneycombEnergy:
    """Test energy computation on honeycomb lattice."""

    def test_all_up_energy(self) -> None:
        """All spins up: E = -J1 * 3 / 2 = -1.5 per site."""
        sim = IsingSimulation(6, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "honeycomb")
        n = sim.num_sites  # 2 * 6 * 6 = 72
        sim.set_spins(np.ones(n, dtype=np.int8))
        assert abs(sim.energy() - (-1.5)) < 1e-10

    def test_all_up_with_field(self) -> None:
        """All up, J1=1, h=1: E = -1.5 - 1.0 = -2.5 per site."""
        sim = IsingSimulation(6, 1.0, 0.0, 0.0, 1.0, 42, "metropolis", "honeycomb")
        sim.set_spins(np.ones(sim.num_sites, dtype=np.int8))
        assert abs(sim.energy() - (-2.5)) < 1e-10


class TestHoneycombSimulation:
    """Test simulation behavior on honeycomb lattice."""

    def test_energy_decreases_at_low_t(self) -> None:
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "honeycomb")
        e_before = sim.energy()
        sim.sweep(200, 10.0)
        e_after = sim.energy()
        assert e_after <= e_before + 1e-10

    def test_deterministic(self) -> None:
        sim1 = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 123, "metropolis", "honeycomb")
        sim2 = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 123, "metropolis", "honeycomb")
        sim1.sweep(10, 0.5)
        sim2.sweep(10, 0.5)
        np.testing.assert_array_equal(sim1.get_spins(), sim2.get_spins())

    def test_spins_shape_is_3d(self) -> None:
        """Honeycomb should return a 3D spin array (L, L, 2)."""
        sim = IsingSimulation(6, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "honeycomb")
        spins = sim.get_spins()
        assert spins.ndim == 3
        assert spins.shape == (6, 6, 2)

    def test_num_sites(self) -> None:
        sim = IsingSimulation(6, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "honeycomb")
        assert sim.num_sites == 72  # 2 * 6 * 6


class TestHoneycombCluster:
    """Test cluster algorithms on honeycomb lattice."""

    def test_wolff_runs(self) -> None:
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42, "wolff", "honeycomb")
        accepted, attempted = sim.sweep(10, 0.5)
        assert attempted > 0

    def test_swendsen_wang_runs(self) -> None:
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42, "swendsen_wang", "honeycomb")
        accepted, attempted = sim.sweep(10, 0.5)
        assert attempted > 0


class TestHoneycombHighLevel:
    """Test high-level Simulation class with honeycomb lattice."""

    def test_run_honeycomb(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(
                lattice_type=LatticeType.HONEYCOMB,
                size=6,
                j1=1.0,
            ),
            temperatures=(2.0, 1.519, 1.0),
            n_sweeps=50,
            measurement_interval=10,
        )
        sim = Simulation(config)
        results = sim.run(show_progress=False)
        assert len(results.temperatures) == 3
        assert all(t in results.energy for t in results.temperatures)
