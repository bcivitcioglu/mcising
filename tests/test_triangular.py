"""Tests for the triangular lattice."""

from __future__ import annotations

import numpy as np
from mcising._core import IsingSimulation
from mcising.config import LatticeConfig, LatticeType, SimulationConfig
from mcising.simulation import Simulation


class TestTriangularEnergy:
    """Test energy computation on triangular lattice."""

    def test_all_up_energy(self) -> None:
        """All spins up: E = -J1 * 6 / 2 = -3.0 per site."""
        sim = IsingSimulation(6, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "triangular")
        sim.set_spins(np.ones((6, 6), dtype=np.int8))
        assert abs(sim.energy() - (-3.0)) < 1e-10

    def test_all_down_energy(self) -> None:
        """All spins down: same energy as all up."""
        sim = IsingSimulation(6, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "triangular")
        sim.set_spins(-np.ones((6, 6), dtype=np.int8))
        assert abs(sim.energy() - (-3.0)) < 1e-10

    def test_all_up_with_field(self) -> None:
        """All up, J1=1, h=1: E = -3.0 - 1.0 = -4.0 per site."""
        sim = IsingSimulation(6, 1.0, 0.0, 0.0, 1.0, 42, "metropolis", "triangular")
        sim.set_spins(np.ones((6, 6), dtype=np.int8))
        assert abs(sim.energy() - (-4.0)) < 1e-10


class TestTriangularSimulation:
    """Test simulation behavior on triangular lattice."""

    def test_energy_decreases_at_low_t(self) -> None:
        """Metropolis on triangular should lower energy at low T."""
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "triangular")
        e_before = sim.energy()
        sim.sweep(200, 10.0)
        e_after = sim.energy()
        assert e_after <= e_before + 1e-10

    def test_deterministic(self) -> None:
        """Same seed → identical results on triangular."""
        sim1 = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 123, "metropolis", "triangular")
        sim2 = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 123, "metropolis", "triangular")
        sim1.sweep(10, 0.5)
        sim2.sweep(10, 0.5)
        np.testing.assert_array_equal(sim1.get_spins(), sim2.get_spins())

    def test_high_temp_high_acceptance(self) -> None:
        """At very high T, acceptance rate should be high."""
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "triangular")
        accepted, attempted = sim.sweep(1, 0.001)
        assert accepted / attempted > 0.5

    def test_ground_state_stable(self) -> None:
        """All-up should remain magnetized at low T."""
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "triangular")
        sim.set_spins(np.ones((8, 8), dtype=np.int8))
        sim.sweep(50, 10.0)
        mag = abs(sim.magnetization())
        assert mag > 0.9, f"Expected magnetized ground state, got |m|={mag}"


class TestTriangularCluster:
    """Test cluster algorithms on triangular lattice."""

    def test_wolff_runs(self) -> None:
        """Wolff should work on triangular (J2=0, h=0)."""
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42, "wolff", "triangular")
        accepted, attempted = sim.sweep(10, 0.5)
        assert attempted > 0

    def test_swendsen_wang_runs(self) -> None:
        """SW should work on triangular (J2=0, h=0)."""
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42, "swendsen_wang", "triangular")
        accepted, attempted = sim.sweep(10, 0.5)
        assert attempted > 0


class TestTriangularHighLevel:
    """Test high-level Simulation class with triangular lattice."""

    def test_run_triangular(self) -> None:
        """Full simulation run on triangular lattice."""
        config = SimulationConfig(
            lattice=LatticeConfig(
                lattice_type=LatticeType.TRIANGULAR,
                size=8,
                j1=1.0,
            ),
            temperatures=(4.0, 3.641, 2.0),
            n_sweeps=50,
            measurement_interval=10,
        )
        sim = Simulation(config)
        results = sim.run(show_progress=False)
        assert len(results.temperatures) == 3
        assert all(t in results.energy for t in results.temperatures)

    def test_config_triangular_type(self) -> None:
        """LatticeConfig with TRIANGULAR type."""
        lc = LatticeConfig(lattice_type=LatticeType.TRIANGULAR, size=8)
        assert lc.lattice_type == LatticeType.TRIANGULAR
        assert lc.size == 8
