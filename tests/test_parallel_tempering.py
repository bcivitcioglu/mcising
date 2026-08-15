"""Tests for Parallel Tempering execution mode."""

from __future__ import annotations

import numpy as np
import pytest
from mcising._core import run_independent_temperatures, run_parallel_tempering
from mcising.config import (
    Algorithm,
    ExecutionMode,
    LatticeConfig,
    LatticeType,
    SimulationConfig,
)
from mcising.exceptions import ConfigurationError
from mcising.simulation import Simulation


class TestPTBasic:
    """Test that PT produces valid results."""

    def test_produces_all_temperatures(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            temperatures=(3.0, 2.269, 1.5),
            n_sweeps=100,
            n_thermalization=50,
            measurement_interval=10,
            mode=ExecutionMode.PARALLEL_TEMPERING,
        )
        results = Simulation(config).run(show_progress=False)
        for t in [1.5, 2.269, 3.0]:
            assert t in results.energy, f"Missing T={t}"

    def test_energy_within_bounds(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            temperatures=(3.0, 2.269, 1.5),
            n_sweeps=200,
            n_thermalization=100,
            measurement_interval=10,
            mode=ExecutionMode.PARALLEL_TEMPERING,
        )
        results = Simulation(config).run(show_progress=False)
        for t in results.temperatures:
            e = results.energy[t]
            assert np.all(e >= -2.01), f"E too low at T={t}"
            assert np.all(e <= 2.01), f"E too high at T={t}"

    def test_correct_measurement_count(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=4),
            temperatures=(3.0, 2.0),
            n_sweeps=100,
            measurement_interval=10,
            mode=ExecutionMode.PARALLEL_TEMPERING,
        )
        results = Simulation(config).run(show_progress=False)
        for t in results.temperatures:
            assert len(results.energy[t]) == 10  # 100 / 10

    def test_configurations_stored(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=4),
            temperatures=(3.0, 2.0),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.PARALLEL_TEMPERING,
        )
        results = Simulation(config).run(show_progress=False)
        for t in results.temperatures:
            configs = results.configurations[t]
            assert configs.shape[0] == 5  # 50/10
            assert configs.shape[1:] == (4, 4)


class TestPTDeterminism:
    """Test reproducibility."""

    def test_same_seed_same_results(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            temperatures=(3.0, 2.269, 1.5),
            n_sweeps=100,
            measurement_interval=10,
            seed=42,
            mode=ExecutionMode.PARALLEL_TEMPERING,
        )
        r1 = Simulation(config).run(show_progress=False)
        r2 = Simulation(config).run(show_progress=False)
        for t in config.temperatures:
            np.testing.assert_array_equal(r1.energy[t], r2.energy[t])

    def test_different_seeds_different(self) -> None:
        base = dict(
            lattice=LatticeConfig(size=8),
            temperatures=(3.0, 2.0),
            n_sweeps=100,
            measurement_interval=10,
            mode=ExecutionMode.PARALLEL_TEMPERING,
        )
        r1 = Simulation(SimulationConfig(**base, seed=1)).run(show_progress=False)
        r2 = Simulation(SimulationConfig(**base, seed=999)).run(show_progress=False)
        assert not np.array_equal(r1.energy[2.0], r2.energy[2.0])


class TestPTPhysics:
    """Test physics validity."""

    def test_energy_ordering_by_temperature(self) -> None:
        """Higher T should have higher (less negative) mean energy."""
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            temperatures=(1.5, 2.269, 4.0),
            n_sweeps=2000,
            n_thermalization=500,
            measurement_interval=10,
            mode=ExecutionMode.PARALLEL_TEMPERING,
        )
        results = Simulation(config).run(show_progress=False)
        e_low = np.mean(results.energy[1.5])
        e_mid = np.mean(results.energy[2.269])
        e_high = np.mean(results.energy[4.0])
        # Energy increases (becomes less negative) with temperature
        assert e_low < e_mid < e_high, (
            f"Energy should increase with T: "
            f"E(1.5)={e_low:.3f}, E(2.269)={e_mid:.3f}, E(4.0)={e_high:.3f}"
        )

    def test_swap_interval_parameter(self) -> None:
        """Verify swap_interval > 1 works."""
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            temperatures=(3.0, 2.0),
            n_sweeps=100,
            measurement_interval=10,
            swap_interval=5,
            mode=ExecutionMode.PARALLEL_TEMPERING,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 2


class TestPTLattices:
    """Test PT works on different lattice types."""

    def test_triangular(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(lattice_type=LatticeType.TRIANGULAR, size=8),
            temperatures=(4.0, 3.641, 3.0),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.PARALLEL_TEMPERING,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 3

    def test_cubic(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(lattice_type=LatticeType.CUBIC, size=4),
            temperatures=(5.0, 4.5),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.PARALLEL_TEMPERING,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 2

    def test_honeycomb(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(lattice_type=LatticeType.HONEYCOMB, size=6),
            temperatures=(2.0, 1.519),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.PARALLEL_TEMPERING,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 2


class TestPTAlgorithms:
    """Test PT with different sweep algorithms."""

    def test_wolff(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            algorithm=Algorithm.WOLFF,
            temperatures=(3.0, 2.269),
            n_sweeps=50,
            measurement_interval=10,
            mode=ExecutionMode.PARALLEL_TEMPERING,
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
            mode=ExecutionMode.PARALLEL_TEMPERING,
        )
        results = Simulation(config).run(show_progress=False)
        assert len(results.energy) == 2


def _small_pt_config(**overrides: object) -> SimulationConfig:
    kwargs: dict[str, object] = {
        "lattice": LatticeConfig(size=4),
        "temperatures": (3.0, 2.0),
        "n_sweeps": 100,
        "measurement_interval": 10,
        "mode": ExecutionMode.PARALLEL_TEMPERING,
    }
    kwargs.update(overrides)
    return SimulationConfig(**kwargs)  # type: ignore[arg-type]


_PT_CORE_ARGS = (4, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "square")


class TestPTCadence:
    """Non-dividing swap/measurement cadences are rejected loudly (B5).

    The ladder advances in swap_interval-sized chunks and measures only on
    chunk boundaries; before P06 a non-dividing cadence silently dropped
    measurements and panicked at the configuration reshape.
    """

    def test_nondividing_config_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="multiple of swap_interval"):
            _small_pt_config(measurement_interval=15, swap_interval=10)

    def test_core_rejects_nondividing_cadence(self) -> None:
        # Direct _core call bypasses SimulationConfig — the Rust boundary
        # must reject on its own (ValueError, never PanicException).
        with pytest.raises(ValueError, match="multiple of swap_interval"):
            run_parallel_tempering(
                *_PT_CORE_ARGS, [3.0, 2.0], 10, 90, 15, swap_interval=10
            )

    @pytest.mark.parametrize("swap_interval", [1, 2, 5, 10])
    def test_measurement_count_stable_across_swap_intervals(
        self, swap_interval: int
    ) -> None:
        config = _small_pt_config(swap_interval=swap_interval)
        results = Simulation(config).run(show_progress=False)
        for temp in (3.0, 2.0):
            assert len(results.energy[temp]) == 10  # 100 / 10, never short
            assert results.configurations[temp].shape[0] == 10


class TestPTPanicSafety:
    """Invalid direct _core input raises Python exceptions, not panics."""

    def test_zero_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            run_parallel_tempering(*_PT_CORE_ARGS, [2.0, 0.0], 10, 50, 10)

    def test_nan_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            run_parallel_tempering(*_PT_CORE_ARGS, [2.0, float("nan")], 10, 50, 10)

    def test_empty_temperature_list_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one temperature"):
            run_parallel_tempering(*_PT_CORE_ARGS, [], 10, 50, 10)

    def test_independent_runner_rejects_same_inputs(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            run_independent_temperatures(*_PT_CORE_ARGS, [0.0], 10, 50, 10)
        with pytest.raises(ValueError, match="finite"):
            run_independent_temperatures(*_PT_CORE_ARGS, [float("nan")], 10, 50, 10)
        with pytest.raises(ValueError, match="At least one temperature"):
            run_independent_temperatures(*_PT_CORE_ARGS, [], 10, 50, 10)


class TestPTCorrelation:
    """compute_correlation works in parallel tempering as of P06."""

    def test_correlation_populated(self) -> None:
        config = _small_pt_config(n_sweeps=50, compute_correlation=True)
        results = Simulation(config).run(show_progress=False)
        assert results.correlation_function is not None
        assert results.correlation_length is not None
        for temp in (3.0, 2.0):
            distances, correlations = results.correlation_function[temp]
            assert distances.size > 0
            assert distances.shape == correlations.shape
            assert results.correlation_length[temp].shape == (5,)  # 50 // 10
