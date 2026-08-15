"""Tests for independent (parallel) temperature execution mode."""

from __future__ import annotations

import numpy as np
import pytest
from mcising.config import (
    Algorithm,
    ExecutionMode,
    LatticeConfig,
    LatticeType,
    SimulationConfig,
)
from mcising.simulation import Simulation

from tests._stats import assert_ordered_means, assert_samples_agree


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
        r1 = Simulation(SimulationConfig(**base, seed=1)).run(show_progress=False)
        r2 = Simulation(SimulationConfig(**base, seed=999)).run(show_progress=False)
        assert not np.array_equal(r1.energy[2.269], r2.energy[2.269])


class TestIndependentLattices:
    """Test independent mode works on all lattice types."""

    def test_triangular(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(lattice_type=LatticeType.TRIANGULAR, size=8),
            temperatures=(4.0, 3.641),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.INDEPENDENT,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 2

    def test_cubic(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(lattice_type=LatticeType.CUBIC, size=4),
            temperatures=(5.0, 4.5),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.INDEPENDENT,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 2

    def test_honeycomb(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(lattice_type=LatticeType.HONEYCOMB, size=6),
            temperatures=(2.0, 1.519),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.INDEPENDENT,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 2

    def test_chain(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(lattice_type=LatticeType.CHAIN, size=50),
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


class TestIndependentManyTemperatures:
    """Test a wide temperature scan for completeness and consistency."""

    @pytest.mark.slow
    @pytest.mark.statistical
    def test_many_temperature_scan_is_complete_and_consistent(self) -> None:
        """A 20-point scan returns complete, physical, cooldown-consistent data.

        This replaces a wall-clock assertion (independent < 3x cooldown),
        which measured the CI runner, not the library. What actually needs
        guarding is that the parallel path drops no temperature and samples
        the same ensemble. Agreement is asserted only for T >= 3.0:
        independent mode thermalizes at the target beta (parallel.rs:83),
        i.e. it quenches, so below Tc it can sit in metastable stripe
        states that cool-down anneals away — a difference of protocol,
        not of physics.
        """
        temps = tuple(float(f"{t:.2f}") for t in np.linspace(1.5, 3.5, 20))
        base = dict(
            lattice=LatticeConfig(size=16),
            temperatures=temps,
            n_sweeps=500,
            n_thermalization=200,
            measurement_interval=10,
            seed=42,
        )
        cooldown = Simulation(
            SimulationConfig(**base, mode=ExecutionMode.COOLDOWN)
        ).run(show_progress=False)
        independent = Simulation(
            SimulationConfig(**base, mode=ExecutionMode.INDEPENDENT)
        ).run(show_progress=False)

        assert set(independent.energy) == set(temps)
        assert set(cooldown.energy) == set(temps)
        for t in temps:
            e = independent.energy[t]
            assert e.size == 50
            assert np.all(np.isfinite(e))
            assert np.all(np.abs(e) <= 2.0 + 1e-9)

        hot = sorted(t for t in temps if t >= 3.0)
        for t in hot:
            assert_samples_agree(
                cooldown.energy[t],
                independent.energy[t],
                label_a=f"cooldown <E>(T={t})",
                label_b=f"independent <E>(T={t})",
            )
        assert_ordered_means(
            [(f"T={t}", independent.energy[t]) for t in hot],
            increasing=True,
        )
        assert_ordered_means(
            [
                (f"T={temps[0]}", independent.energy[temps[0]]),
                (f"T={temps[-1]}", independent.energy[temps[-1]]),
            ],
            increasing=True,
        )


def _small_independent_config(**overrides: object) -> SimulationConfig:
    kwargs: dict[str, object] = {
        "lattice": LatticeConfig(size=4),
        "temperatures": (3.0, 2.0),
        "n_sweeps": 50,
        "measurement_interval": 10,
        "mode": ExecutionMode.INDEPENDENT,
    }
    kwargs.update(overrides)
    return SimulationConfig(**kwargs)  # type: ignore[arg-type]


class TestIndependentCorrelation:
    """compute_correlation must produce data in independent mode (B8).

    Before P06 the flag was accepted, the result dicts were pre-created
    empty, and nothing ever filled them — a silent no-op.
    """

    def test_correlation_populated(self) -> None:
        config = _small_independent_config(compute_correlation=True)
        results = Simulation(config).run(show_progress=False)
        assert results.correlation_function is not None
        assert results.correlation_length is not None
        for temp in (3.0, 2.0):
            distances, correlations = results.correlation_function[temp]
            assert distances.size > 0
            assert distances.shape == correlations.shape
            # One correlation length per measurement (50 // 10).
            assert results.correlation_length[temp].shape == (5,)

    def test_correlation_absent_when_disabled(self) -> None:
        results = Simulation(_small_independent_config()).run(show_progress=False)
        assert results.correlation_function is None
        assert results.correlation_length is None


class TestIndependentStoreConfigs:
    def test_store_configs_false_omits_configurations(self) -> None:
        config = _small_independent_config(store_configs=False)
        results = Simulation(config).run(show_progress=False)
        for temp in (3.0, 2.0):
            assert temp not in results.configurations
            # Scalar observables are unaffected.
            assert results.energy[temp].shape == (5,)


class TestIndependentSkipSeeding:
    """skip_temperatures must not re-seed the surviving temperatures.

    Independent-mode seeds are base_seed + the temperature's index in the
    configured scan; a resumed (skipping) run keeps each survivor's
    original index, so its streams match the uninterrupted run's exactly.
    """

    def test_skip_preserves_streams(self) -> None:
        config = _small_independent_config(temperatures=(3.0, 2.0, 1.0))
        full = Simulation(config).run(show_progress=False)
        resumed = Simulation(config).run(
            show_progress=False, skip_temperatures=frozenset({3.0, 2.0})
        )

        assert resumed.temperatures == [1.0]
        np.testing.assert_array_equal(resumed.energy[1.0], full.energy[1.0])
        np.testing.assert_array_equal(
            resumed.magnetization[1.0], full.magnetization[1.0]
        )
        np.testing.assert_array_equal(
            resumed.configurations[1.0], full.configurations[1.0]
        )
