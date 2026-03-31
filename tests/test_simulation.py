"""Tests for the high-level Simulation class."""

from __future__ import annotations

import numpy as np
import pytest
from mcising.config import LatticeConfig, SimulationConfig
from mcising.exceptions import SimulationError
from mcising.simulation import Simulation, SimulationResults


class TestSimulationRun:
    def test_run_returns_results(self, default_config: SimulationConfig) -> None:
        sim = Simulation(default_config)
        results = sim.run(show_progress=False)
        assert isinstance(results, SimulationResults)

    def test_results_have_all_temperatures(
        self, default_config: SimulationConfig
    ) -> None:
        sim = Simulation(default_config)
        results = sim.run(show_progress=False)
        # Temperatures sorted descending
        assert len(results.temperatures) == 3
        assert results.temperatures == sorted(results.temperatures, reverse=True)

    def test_results_have_energy_per_temperature(
        self, default_config: SimulationConfig
    ) -> None:
        sim = Simulation(default_config)
        results = sim.run(show_progress=False)
        for temp in results.temperatures:
            assert temp in results.energy
            assert len(results.energy[temp]) > 0

    def test_results_have_magnetization_per_temperature(
        self, default_config: SimulationConfig
    ) -> None:
        sim = Simulation(default_config)
        results = sim.run(show_progress=False)
        for temp in results.temperatures:
            assert temp in results.magnetization
            assert len(results.magnetization[temp]) > 0

    def test_results_have_configurations(
        self, default_config: SimulationConfig
    ) -> None:
        sim = Simulation(default_config)
        results = sim.run(show_progress=False)
        for temp in results.temperatures:
            assert temp in results.configurations
            configs = results.configurations[temp]
            assert configs.ndim == 3
            assert configs.shape[1] == 8
            assert configs.shape[2] == 8

    def test_metadata_has_elapsed(self, default_config: SimulationConfig) -> None:
        sim = Simulation(default_config)
        results = sim.run(show_progress=False)
        assert "elapsed_seconds" in results.metadata
        assert results.metadata["elapsed_seconds"] >= 0  # type: ignore[operator]


class TestSimulationSweep:
    def test_sweep_returns_observables(self) -> None:
        config = SimulationConfig(lattice=LatticeConfig(size=4))
        sim = Simulation(config)
        obs = sim.sweep(2.269, 10)
        assert "energy" in obs
        assert "magnetization" in obs
        assert "acceptance_rate" in obs

    def test_sweep_invalid_temperature_raises(self) -> None:
        config = SimulationConfig(lattice=LatticeConfig(size=4))
        sim = Simulation(config)
        with pytest.raises(SimulationError, match="Temperature must be positive"):
            sim.sweep(-1.0, 10)

    def test_sweep_zero_temperature_raises(self) -> None:
        config = SimulationConfig(lattice=LatticeConfig(size=4))
        sim = Simulation(config)
        with pytest.raises(SimulationError, match="Temperature must be positive"):
            sim.sweep(0.0, 10)


class TestSimulationProperties:
    def test_spins_property(self) -> None:
        config = SimulationConfig(lattice=LatticeConfig(size=4))
        sim = Simulation(config)
        spins = sim.spins
        assert spins.shape == (4, 4)
        assert spins.dtype == np.int8

    def test_energy_property(self) -> None:
        config = SimulationConfig(lattice=LatticeConfig(size=4))
        sim = Simulation(config)
        e = sim.energy
        assert isinstance(e, float)

    def test_magnetization_property(self) -> None:
        config = SimulationConfig(lattice=LatticeConfig(size=4))
        sim = Simulation(config)
        m = sim.magnetization
        assert isinstance(m, float)
        assert -1.0 <= m <= 1.0

    def test_set_spins_via_property(self) -> None:
        config = SimulationConfig(lattice=LatticeConfig(size=4))
        sim = Simulation(config)
        new_spins = np.ones((4, 4), dtype=np.int8)
        sim.spins = new_spins
        assert np.array_equal(sim.spins, new_spins)


class TestCorrelationComputation:
    def test_with_correlation_enabled(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=4),
            temperatures=(2.269,),
            n_sweeps=50,
            measurement_interval=10,
            compute_correlation=True,
        )
        sim = Simulation(config)
        results = sim.run(show_progress=False)
        assert results.correlation_function is not None
        assert results.correlation_length is not None
        assert 2.269 in results.correlation_function
        assert 2.269 in results.correlation_length
