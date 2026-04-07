"""Tests for the 1D chain lattice."""

from __future__ import annotations

import numpy as np
from mcising._core import IsingSimulation
from mcising.config import LatticeConfig, LatticeType, SimulationConfig
from mcising.simulation import Simulation


class TestChainEnergy:
    """Test energy computation on chain lattice."""

    def test_all_up_energy(self) -> None:
        """All spins up: E = -J1 * 2 / 2 = -1.0 per site."""
        sim = IsingSimulation(10, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "chain")
        sim.set_spins(np.ones(10, dtype=np.int8))
        assert abs(sim.energy() - (-1.0)) < 1e-10

    def test_all_up_with_field(self) -> None:
        """All up, J1=1, h=1: E = -1.0 - 1.0 = -2.0 per site."""
        sim = IsingSimulation(10, 1.0, 0.0, 0.0, 1.0, 42, "metropolis", "chain")
        sim.set_spins(np.ones(10, dtype=np.int8))
        assert abs(sim.energy() - (-2.0)) < 1e-10

    def test_all_down_energy(self) -> None:
        """All down: same energy as all up."""
        sim = IsingSimulation(10, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "chain")
        sim.set_spins(-np.ones(10, dtype=np.int8))
        assert abs(sim.energy() - (-1.0)) < 1e-10

    def test_alternating_energy(self) -> None:
        """Alternating +1/-1: every NN pair antiparallel, E = +1.0 per site."""
        sim = IsingSimulation(10, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "chain")
        spins = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int8)
        sim.set_spins(spins)
        assert abs(sim.energy() - 1.0) < 1e-10


class TestChainSimulation:
    """Test simulation behavior on chain lattice."""

    def test_energy_decreases_at_low_t(self) -> None:
        sim = IsingSimulation(50, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "chain")
        e_before = sim.energy()
        sim.sweep(200, 10.0)
        e_after = sim.energy()
        assert e_after <= e_before + 1e-10

    def test_deterministic(self) -> None:
        sim1 = IsingSimulation(50, 1.0, 0.0, 0.0, 0.0, 123, "metropolis", "chain")
        sim2 = IsingSimulation(50, 1.0, 0.0, 0.0, 0.0, 123, "metropolis", "chain")
        sim1.sweep(10, 0.5)
        sim2.sweep(10, 0.5)
        np.testing.assert_array_equal(sim1.get_spins(), sim2.get_spins())

    def test_spins_shape_is_1d(self) -> None:
        """Chain should return a 1D spin array."""
        sim = IsingSimulation(20, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "chain")
        spins = sim.get_spins()
        assert spins.ndim == 1
        assert spins.shape == (20,)


class TestChainCluster:
    """Test cluster algorithms on chain lattice."""

    def test_wolff_runs(self) -> None:
        sim = IsingSimulation(20, 1.0, 0.0, 0.0, 0.0, 42, "wolff", "chain")
        accepted, attempted = sim.sweep(10, 0.5)
        assert attempted > 0

    def test_swendsen_wang_runs(self) -> None:
        sim = IsingSimulation(20, 1.0, 0.0, 0.0, 0.0, 42, "swendsen_wang", "chain")
        accepted, attempted = sim.sweep(10, 0.5)
        assert attempted > 0


class TestChainHighLevel:
    """Test high-level Simulation class with chain lattice."""

    def test_run_chain(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(
                lattice_type=LatticeType.CHAIN,
                size=50,
                j1=1.0,
            ),
            temperatures=(2.0, 1.0, 0.5),
            n_sweeps=50,
            measurement_interval=10,
        )
        sim = Simulation(config)
        results = sim.run(show_progress=False)
        assert len(results.temperatures) == 3


class TestChainPhysics:
    """Test 1D chain physics: no ordering at T > 0."""

    def test_no_ordering_at_finite_t(self) -> None:
        """1D Ising has Tc=0: at T=1.0, |m| should be small for large L."""
        sim = IsingSimulation(100, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "chain")
        # Thermalize
        sim.sweep(1000, 1.0)  # beta=1.0 → T=1.0
        # Measure
        mags = []
        for _ in range(100):
            sim.sweep(10, 1.0)
            mags.append(abs(sim.magnetization()))
        avg_mag = np.mean(mags)
        assert avg_mag < 0.5, (
            f"1D chain should not order at T=1.0, got <|m|>={avg_mag:.3f}"
        )
