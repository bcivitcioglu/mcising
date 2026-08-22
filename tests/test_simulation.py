"""Tests for the high-level Simulation class."""

from __future__ import annotations

import numpy as np
import pytest
from mcising._core import IsingSimulation
from mcising.config import LatticeConfig, LatticeType, SimulationConfig
from mcising.exceptions import ConfigurationError, SimulationError
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
        obs = sim.sweep(10, temperature=2.269)
        assert "energy" in obs
        assert "magnetization" in obs
        assert "acceptance_rate" in obs
        assert obs["n_cluster_flips"] == 0.0  # Metropolis

    def test_sweep_invalid_temperature_raises(self) -> None:
        config = SimulationConfig(lattice=LatticeConfig(size=4))
        sim = Simulation(config)
        with pytest.raises(SimulationError, match="Temperature must be positive"):
            sim.sweep(10, temperature=-1.0)

    def test_sweep_zero_temperature_raises(self) -> None:
        config = SimulationConfig(lattice=LatticeConfig(size=4))
        sim = Simulation(config)
        with pytest.raises(SimulationError, match="Temperature must be positive"):
            sim.sweep(10, temperature=0.0)

    @pytest.mark.parametrize("algorithm", ["metropolis", "wolff", "swendsen_wang"])
    def test_work_accounting_is_honest(self, algorithm: str) -> None:
        # Replaced the "comparable work at equal n_sweeps" gate: the
        # flip-budget Wolff sweep it presumed is size-biased (P10
        # exact-enumeration rejection at 200+ sigma). The honest
        # invariants: Metropolis/SW sweeps are num_sites attempts each;
        # Wolff is one rejection-free cluster per sweep.
        sim = IsingSimulation(8, 1.0, 0.0, 0.0, 0.0, 42, algorithm)
        accepted, attempted, cluster_flips = sim.sweep(20, temperature=2.269)
        if algorithm == "metropolis":
            assert attempted == 20 * sim.num_sites
            assert cluster_flips == 0
            assert accepted <= attempted
        elif algorithm == "swendsen_wang":
            assert attempted == 20 * sim.num_sites
            assert accepted <= attempted
            assert cluster_flips >= 1
        else:  # wolff
            assert attempted == accepted  # rejection-free
            assert cluster_flips == 20  # exactly one cluster per sweep
            assert 20 <= accepted <= 20 * sim.num_sites

    def test_both_entry_points_accept_identical_call_shape(self) -> None:
        # The P10 gate: one signature, one unit — the same argument
        # names and order work on the high-level and core layers.
        config = SimulationConfig(lattice=LatticeConfig(size=4))
        sim = Simulation(config)
        obs = sim.sweep(3, temperature=2.5)
        accepted, attempted, flips = sim._core.sweep(3, temperature=2.5)
        assert {"energy", "magnetization", "acceptance_rate"} <= set(obs)
        assert attempted == 3 * sim.num_sites
        assert flips == 0  # metropolis

    def test_legacy_positional_beta_raises_type_error(self) -> None:
        # The migration guard: the pre-1.0 call shape sweep(n, beta)
        # must fail loudly, never silently reinterpret beta as T.
        config = SimulationConfig(lattice=LatticeConfig(size=4))
        sim = Simulation(config)
        with pytest.raises(TypeError):
            sim.sweep(10, 0.5)  # type: ignore[misc]
        with pytest.raises(TypeError):
            sim._core.sweep(10, 0.5)  # type: ignore[misc]


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


class TestStoreConfigsCooldown:
    def test_store_configs_false_omits_configurations(self) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(size=4),
            temperatures=(2.0,),
            n_sweeps=20,
            measurement_interval=10,
            store_configs=False,
        )
        results = Simulation(config).run(show_progress=False)
        assert 2.0 not in results.configurations
        assert results.energy[2.0].shape == (2,)


class TestNumSites:
    """B11 (#22): the site count is exact or a hard error, never 1."""

    @pytest.mark.parametrize(
        ("lattice_type", "expected"),
        [
            (LatticeType.SQUARE, 1024),
            (LatticeType.TRIANGULAR, 1024),
            (LatticeType.HONEYCOMB, 2048),
            (LatticeType.CUBIC, 32768),
            (LatticeType.CHAIN, 32),
        ],
    )
    def test_per_lattice_site_count_at_l32(
        self, lattice_type: LatticeType, expected: int
    ) -> None:
        assert LatticeConfig(lattice_type, 32).num_sites == expected

    @pytest.mark.parametrize("lattice_type", list(LatticeType))
    @pytest.mark.parametrize("size", [4, 6])
    def test_formula_matches_rust_core(
        self, lattice_type: LatticeType, size: int
    ) -> None:
        # The pure-Python formula is licensed by parity with the Rust
        # constructors, which own the geometry.
        sim = IsingSimulation(
            size, 1.0, 0.0, 0.0, 0.0, 0, "metropolis", lattice_type.value
        )
        assert LatticeConfig(lattice_type, size).num_sites == sim.num_sites

    @pytest.mark.parametrize("lattice_type", list(LatticeType))
    def test_simulation_property_three_way_parity(
        self, lattice_type: LatticeType
    ) -> None:
        # P10: Simulation.num_sites (reads the core) agrees with the
        # config formula and the raw core getter.
        config = SimulationConfig(lattice=LatticeConfig(lattice_type, 4))
        sim = Simulation(config)
        assert sim.num_sites == config.lattice.num_sites
        assert sim.num_sites == sim._core.num_sites

    def test_results_num_sites_from_config(self) -> None:
        config = SimulationConfig(lattice=LatticeConfig(size=4))
        results = SimulationResults(metadata={"config": config})
        assert results.num_sites == 16

    def test_results_num_sites_from_configurations_shape(self) -> None:
        # Config-less legacy files fall back to the stored spin shape.
        results = SimulationResults(
            configurations={2.0: np.ones((3, 4, 4), dtype=np.int8)}
        )
        assert results.num_sites == 16

    def test_results_num_sites_raises_on_missing_info(self) -> None:
        results = SimulationResults(metadata={})
        with pytest.raises(ConfigurationError, match="number of lattice sites"):
            _ = results.num_sites

    def test_no_rust_simulation_constructed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The old code built a throwaway Rust sim per call (B11).
        import mcising.simulation as sim_module

        def _boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("num_sites must not construct a Rust sim")

        monkeypatch.setattr(sim_module, "_RustSim", _boom)
        config = SimulationConfig(lattice=LatticeConfig(size=4))
        results = SimulationResults(metadata={"config": config})
        assert results.num_sites == 16


class TestRunSemantics:
    """P10: run() resets to a fresh core; re-running is legal and
    deterministic."""

    @staticmethod
    def _config() -> SimulationConfig:
        return SimulationConfig(
            lattice=LatticeConfig(size=8),
            temperatures=(2.5, 2.0),
            n_sweeps=50,
            n_thermalization=20,
            measurement_interval=5,
        )

    def test_run_twice_on_same_object_is_identical(self) -> None:
        sim = Simulation(self._config())
        r1 = sim.run(show_progress=False)
        r2 = sim.run(show_progress=False)
        for temp in r1.temperatures:
            np.testing.assert_array_equal(r1.energy[temp], r2.energy[temp])
            np.testing.assert_array_equal(
                r1.magnetization[temp], r2.magnetization[temp]
            )

    def test_manual_sweep_before_run_has_no_effect(self) -> None:
        dirty = Simulation(self._config())
        dirty.sweep(30, temperature=3.0)
        fresh = Simulation(self._config())
        r_dirty = dirty.run(show_progress=False)
        r_fresh = fresh.run(show_progress=False)
        for temp in r_fresh.temperatures:
            np.testing.assert_array_equal(r_dirty.energy[temp], r_fresh.energy[temp])

    def test_reset_false_continues_from_current_state(self) -> None:
        # The reset flag is load-bearing: skipping the reset after a
        # manual sweep must produce a different trajectory.
        dirty = Simulation(self._config())
        dirty.sweep(30, temperature=3.0)
        fresh = Simulation(self._config())
        r_dirty = dirty.run(reset=False, show_progress=False)
        r_fresh = fresh.run(show_progress=False)
        assert any(
            not np.array_equal(r_dirty.energy[t], r_fresh.energy[t])
            for t in r_fresh.temperatures
        )

    def test_reset_restores_initial_spins(self) -> None:
        sim = Simulation(self._config())
        initial = sim.spins.copy()
        sim.sweep(30, temperature=3.0)
        sim.reset()
        np.testing.assert_array_equal(sim.spins, initial)


class TestToDataframe:
    """P11: first coverage for to_dataframe (pandas = `dataframe` extra)."""

    def test_dataframe_has_row_per_temperature(self) -> None:
        pd = pytest.importorskip("pandas")
        config = SimulationConfig(
            lattice=LatticeConfig(size=4),
            temperatures=(3.0, 2.0),
            n_sweeps=20,
            measurement_interval=10,
        )
        results = Simulation(config).run(show_progress=False)
        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert {"T", "E_mean", "chi", "chi_err", "U4", "samples"} <= set(df.columns)
        assert sorted(df["T"]) == [2.0, 3.0]


class TestResultsStatistics:
    """B10 (#21): every quoted observable carries an uncertainty."""

    @pytest.fixture
    def run_results(self) -> SimulationResults:
        config = SimulationConfig(
            lattice=LatticeConfig(size=8),
            temperatures=(2.0,),
            n_sweeps=400,
            measurement_interval=10,
        )
        return Simulation(config).run(show_progress=False)

    def test_statistics_fields_and_consistency(
        self, run_results: SimulationResults
    ) -> None:
        stats = run_results.statistics(2.0)
        assert stats.n_samples == 40
        assert stats.tau_int >= 0.5
        # Point values are exactly the legacy scalar methods.
        assert stats.specific_heat.value == pytest.approx(
            run_results.specific_heat(2.0)
        )
        assert stats.susceptibility.value == pytest.approx(
            run_results.susceptibility(2.0)
        )
        assert stats.binder_cumulant.value == pytest.approx(
            run_results.binder_cumulant(2.0)
        )
        e = run_results.energy[2.0]
        assert stats.energy.value == pytest.approx(float(np.mean(e)))
        for est in (
            stats.energy,
            stats.abs_magnetization,
            stats.specific_heat,
            stats.susceptibility,
        ):
            assert np.isfinite(est.error)
            assert est.error >= 0.0

    def test_statistics_memoized(self, run_results: SimulationResults) -> None:
        assert run_results.statistics(2.0) is run_results.statistics(2.0)

    def test_statistics_total_for_missing_temperature(
        self, run_results: SimulationResults
    ) -> None:
        stats = run_results.statistics(99.0)
        assert stats.n_samples == 0
        assert np.isnan(stats.energy.value)

    def test_binder_cumulant_in_physical_range(
        self, run_results: SimulationResults
    ) -> None:
        # U4 <= 2/3 for any distribution reachable here; at T=2.0 < Tc
        # on an 8x8 square lattice the ordered phase pins it near 2/3.
        u4 = run_results.binder_cumulant(2.0)
        assert 0.0 <= u4 <= 2.0 / 3.0 + 1e-12
