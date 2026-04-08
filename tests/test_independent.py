"""Tests for independent (parallel) temperature execution mode."""

from __future__ import annotations

import time

import numpy as np
from mcising.config import (
    Algorithm,
    ExecutionMode,
    LatticeConfig,
    LatticeType,
    SimulationConfig,
)
from mcising.simulation import Simulation


class TestIndependentBasic:
    """Test that independent mode produces valid results."""

    def test_produces_all_temperatures(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            temperatures=(3.0, 2.269, 1.5),
            n_sweeps=100,
            measurement_interval=10,
            mode=ExecutionMode.INDEPENDENT,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.temperatures) == 3
        assert all(t in results.energy for t in [3.0, 2.269, 1.5])

    def test_energy_within_bounds(self) -> None:
        """Energy should be within [-2, +2] for J1=1 square lattice."""
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            temperatures=(2.269,),
            n_sweeps=200,
            n_thermalization=100,
            measurement_interval=10,
            mode=ExecutionMode.INDEPENDENT,
        )
        results = Simulation(config).run(show_progress=False)
        e = results.energy[2.269]
        assert np.all(e >= -2.0 - 0.01)
        assert np.all(e <= 2.0 + 0.01)

    def test_configurations_stored(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=4),
            temperatures=(2.269,),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.INDEPENDENT,
        )
        results = Simulation(config).run(show_progress=False)
        configs = results.configurations[2.269]
        assert configs.shape[0] == 5  # 50 / 10
        assert configs.shape[1:] == (4, 4)


class TestIndependentDeterminism:
    """Test reproducibility of independent mode."""

    def test_same_seed_same_results(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            temperatures=(3.0, 2.0),
            n_sweeps=50,
            measurement_interval=10,
            seed=42,
            mode=ExecutionMode.INDEPENDENT,
        )
        r1 = Simulation(config).run(show_progress=False)
        r2 = Simulation(config).run(show_progress=False)
        for t in config.temperatures:
            np.testing.assert_array_equal(r1.energy[t], r2.energy[t])

    def test_different_seeds_different_results(self) -> None:
        base = dict(
            lattice=LatticeConfig(size=8),
            temperatures=(2.269,),
            n_sweeps=100,
            measurement_interval=10,
            mode=ExecutionMode.INDEPENDENT,
        )
        r1 = Simulation(SimulationConfig(**base, seed=1)).run(
            show_progress=False
        )
        r2 = Simulation(SimulationConfig(**base, seed=999)).run(
            show_progress=False
        )
        assert not np.array_equal(
            r1.energy[2.269], r2.energy[2.269]
        )


class TestIndependentLattices:
    """Test independent mode works on all lattice types."""

    def test_triangular(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(
                lattice_type=LatticeType.TRIANGULAR, size=8
            ),
            temperatures=(4.0, 3.641),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.INDEPENDENT,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 2

    def test_cubic(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(
                lattice_type=LatticeType.CUBIC, size=4
            ),
            temperatures=(5.0, 4.5),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.INDEPENDENT,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 2

    def test_honeycomb(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(
                lattice_type=LatticeType.HONEYCOMB, size=6
            ),
            temperatures=(2.0, 1.519),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.INDEPENDENT,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 2

    def test_chain(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(
                lattice_type=LatticeType.CHAIN, size=50
            ),
            temperatures=(2.0, 1.0),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.INDEPENDENT,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 2


class TestIndependentAlgorithms:
    """Test independent mode works with cluster algorithms."""

    def test_wolff(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            algorithm=Algorithm.WOLFF,
            temperatures=(3.0, 2.269),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.INDEPENDENT,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 2

    def test_swendsen_wang(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            algorithm=Algorithm.SWENDSEN_WANG,
            temperatures=(3.0, 2.269),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.INDEPENDENT,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 2


class TestIndependentPerformance:
    """Test that independent mode is actually faster for many temperatures."""

    def test_faster_than_cooldown_many_temps(self) -> None:
        """With 20 temperatures, independent should be faster."""
        temps = tuple(
            float(f"{t:.2f}") for t in np.linspace(1.5, 3.5, 20)
        )
        base = dict(
            lattice=LatticeConfig(size=16),
            temperatures=temps,
            n_sweeps=500,
            n_thermalization=200,
            measurement_interval=10,
            seed=42,
        )

        # Cooldown
        t0 = time.monotonic()
        Simulation(
            SimulationConfig(**base, mode=ExecutionMode.COOLDOWN)
        ).run(show_progress=False)
        cooldown_time = time.monotonic() - t0

        # Independent
        t0 = time.monotonic()
        Simulation(
            SimulationConfig(**base, mode=ExecutionMode.INDEPENDENT)
        ).run(show_progress=False)
        independent_time = time.monotonic() - t0

        # Independent should be faster on multi-core machines.
        # On CI with 1-2 cores it might not be, so use a generous threshold.
        # Just verify it completes and isn't dramatically slower.
        assert independent_time < cooldown_time * 3.0, (
            f"Independent ({independent_time:.2f}s) should not be "
            f"much slower than cooldown ({cooldown_time:.2f}s)"
        )
