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
from mcising.constants import TC_SQUARE_2D
from mcising.exceptions import ConfigurationError
from mcising.simulation import Simulation, SimulationResults

from tests._stats import DEFAULT_SEEDS, assert_samples_agree


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


#: Shared ladder for the PT-vs-independent comparison: straddles Tc so the
#: replica exchange actually mixes ordered and disordered configurations.
PT_AGREEMENT_TEMPERATURES = (2.0, TC_SQUARE_2D, 2.5, 3.0, 3.5)


class TestPTMatchesIndependent:
    """Replica exchange leaves every temperature's marginal untouched.

    PT samples the joint distribution prod_i exp(-beta_i E(s_i)); its
    marginal at each beta_i must therefore equal what an independent run
    at that temperature measures. Any error in the swap criterion (sign,
    missing energy swap, wrong beta pairing) shows up as a disagreement
    at some shared temperature.
    """

    @staticmethod
    def _run(seed: int, mode: ExecutionMode) -> SimulationResults:
        config = SimulationConfig(
            lattice=LatticeConfig(size=16),
            algorithm=Algorithm.SWENDSEN_WANG,
            temperatures=PT_AGREEMENT_TEMPERATURES,
            n_sweeps=4_000,
            n_thermalization=2_000,
            measurement_interval=10,
            mode=mode,
            swap_interval=5 if mode is ExecutionMode.PARALLEL_TEMPERING else 1,
            seed=seed,
        )
        return Simulation(config).run(show_progress=False)

    @pytest.mark.statistical
    def test_pt_matches_independent_at_every_temperature(self) -> None:
        """<E> and <|m|> agree within 3 sigma at every shared temperature.

        Design. Swendsen-Wang in both modes: independent mode quenches
        from a random start at each target T, and Metropolis then freezes
        into stripes below Tc (test_independent.py compares only T >= 3.0
        for that reason); SW reorganizes globally and equilibrates from
        the quench, so the ladder can reach T=2.0 < Tc. n_thermalization
        is load-bearing — 500 sweeps left a +2.8 sigma pooled bias at
        T=2.0 (thermalization, not physics); 2000 removes it. The series
        are pooled across DEFAULT_SEEDS before comparing so every seed
        contributes while the test stays at 10 comparisons (5 T x 2
        observables) at 3 sigma: ~2.7% nominal family false-fail, well
        below 1% given the blocking error's conservatism, versus ~13% for
        50 per-seed comparisons. Runs are seed-deterministic, so the
        calibrated margins hold in CI. Calibration (release build): worst
        of the 10 comparisons 1.48 sigma, ~0.6 s for all runs.
        """
        series: dict[tuple[ExecutionMode, str, float], list[np.ndarray]] = {}
        modes = (ExecutionMode.PARALLEL_TEMPERING, ExecutionMode.INDEPENDENT)
        for seed in DEFAULT_SEEDS:
            for mode in modes:
                results = self._run(seed, mode)
                for t in PT_AGREEMENT_TEMPERATURES:
                    series.setdefault((mode, "E", t), []).append(results.energy[t])
                    series.setdefault((mode, "|m|", t), []).append(
                        np.abs(results.magnetization[t])
                    )
        pt, ind = modes
        for t in PT_AGREEMENT_TEMPERATURES:
            for observable in ("E", "|m|"):
                assert_samples_agree(
                    np.concatenate(series[(pt, observable, t)]),
                    np.concatenate(series[(ind, observable, t)]),
                    n_sigma=3.0,
                    label_a=f"PT <{observable}>(T={t:.3f})",
                    label_b=f"independent <{observable}>(T={t:.3f})",
                )


class TestPTCorrelationInterval:
    """correlation_interval applies to the replica ladder too (P16)."""

    def test_every_kth_measurement(self) -> None:
        dense = Simulation(_small_pt_config(n_sweeps=50, compute_correlation=True)).run(
            show_progress=False
        )
        sparse = Simulation(
            _small_pt_config(
                n_sweeps=50, compute_correlation=True, correlation_interval=5
            )
        ).run(show_progress=False)
        assert dense.correlation_length is not None
        assert sparse.correlation_length is not None
        for temp in (3.0, 2.0):
            assert sparse.correlation_length[temp].shape == (1,)  # 50 // 10 = 5
            assert np.array_equal(
                sparse.correlation_length[temp], dense.correlation_length[temp][4::5]
            )
            assert np.array_equal(sparse.energy[temp], dense.energy[temp])
