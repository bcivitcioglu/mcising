"""Wang-Landau sampling and multicanonical production: API and physics gates.

Fast gates run on the 3 x 3 and 4 x 4 tori (exact density of states by
enumeration, exact thermodynamics from the Ferdinand-Fisher solution) and
on an 8 x 8 torus; the slow gates cover a 16 x 16 torus, the square J1-J2
model against parallel tempering, and the cubic J1-J2 first-order
transition the method was built for.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from mcising import (
    Algorithm,
    ExecutionMode,
    LatticeConfig,
    LatticeType,
    Simulation,
    SimulationConfig,
    WangLandauConfig,
    WangLandauResults,
    WangLandauSimulation,
)
from mcising.constants import TC_SQUARE_2D
from mcising.exceptions import ConfigurationError
from mcising.statistics import Estimate
from mcising.wang_landau import (
    BarrierEstimate,
    CanonicalEstimates,
    MulticanonicalDiagnostics,
    WalkerSeries,
    WangLandauDiagnostics,
)

from tests._analytic import square_torus_energy_per_site, square_torus_specific_heat
from tests._stats import DEFAULT_SEEDS

#: A 4 x 4 run that converges in a fraction of a second.
SQUARE4 = dict(
    lattice=LatticeConfig(size=4, j1=1.0),
    log_f_final=1e-5,
    check_interval=100,
    production_sweeps=200_000,
)


def _run(**overrides: object) -> WangLandauResults:
    kwargs: dict[str, object] = dict(SQUARE4)
    kwargs.update(overrides)
    config = WangLandauConfig(**kwargs)  # type: ignore[arg-type]
    return WangLandauSimulation(config).run(show_progress=False)


@pytest.fixture(scope="module")
def square4() -> WangLandauResults:
    return _run()


def _assert_agrees(
    estimate: Estimate, exact: float, *, label: str, power: float
) -> None:
    """4-sigma agreement with an exact value plus a power floor on the error.

    The floor stops a shortened run from passing vacuously: the quoted
    error must be at most ``power`` times the magnitude of the exact
    value.
    """
    assert math.isfinite(estimate.error) and estimate.error > 0.0, (
        f"{label}: {estimate}"
    )
    assert estimate.error <= power * abs(exact), (
        f"{label}: error {estimate.error} exceeds the power floor "
        f"{power * abs(exact)} (lengthen the run instead of loosening the gate)"
    )
    deviation = abs(estimate.value - exact)
    assert deviation <= 4.0 * estimate.error, (
        f"{label}: {estimate} is {deviation / estimate.error:.1f} sigma "
        f"from exact {exact}"
    )


class TestConfig:
    def test_defaults_and_measurement_count(self) -> None:
        config = WangLandauConfig()
        assert config.flatness == 0.8
        assert config.log_f_final == 1e-6
        assert config.energy_window is None and config.bin_width is None
        assert config.n_walkers == 1 and config.store_configs is False
        assert (
            WangLandauConfig(
                production_sweeps=95, measurement_interval=10
            ).n_measurements
            == 9
        )

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("energy_window", (1.0, 0.0), "energy_window"),
            ("energy_window", (float("nan"), 1.0), "energy_window"),
            ("energy_window", (0.0, 1.0, 2.0), "energy_window"),
            ("bin_width", 0.0, "bin_width"),
            ("bin_width", float("inf"), "bin_width"),
            ("flatness", 0.0, "flatness"),
            ("flatness", 1.5, "flatness"),
            ("log_f_initial", 0.0, "log_f_initial"),
            ("log_f_final", 0.0, "log_f_final"),
            ("log_f_final", 2.0, "log_f_final"),
            ("check_interval", 0, "check_interval"),
            ("max_wl_sweeps", -1, "max_wl_sweeps"),
            ("production_sweeps", -1, "production_sweeps"),
            ("production_thermalization", -1, "production_thermalization"),
            ("measurement_interval", 0, "measurement_interval"),
            ("n_walkers", 0, "n_walkers"),
            ("drive_beta", -1.0, "drive_beta"),
            ("drive_max_sweeps", 0, "drive_max_sweeps"),
            ("n_windows", 0, "n_windows"),
            ("walkers_per_window", 0, "walkers_per_window"),
            ("window_overlap", 1.0, "window_overlap"),
            ("window_overlap", -0.1, "window_overlap"),
            ("exchange_interval", 0, "exchange_interval"),
        ],
    )
    def test_invalid_values_raise(self, field: str, value: object, match: str) -> None:
        with pytest.raises(ConfigurationError, match=match):
            WangLandauConfig(**{field: value})  # type: ignore[arg-type]

    def test_parallel_stage_cadence(self) -> None:
        assert not WangLandauConfig().parallel_stage
        assert WangLandauConfig(n_windows=2).parallel_stage
        with pytest.raises(ConfigurationError, match="multiple of exchange_interval"):
            WangLandauConfig(n_windows=2, check_interval=150, exchange_interval=100)
        # The serial walker ignores the exchange cadence.
        WangLandauConfig(check_interval=150, exchange_interval=100)

    def test_from_dict_round_trip(self) -> None:
        config = WangLandauConfig(
            lattice=LatticeConfig(lattice_type=LatticeType.CUBIC, size=6, j2=-0.5),
            energy_window=(-1.7, -0.5),
            n_walkers=3,
            seed=9,
        )
        record = dataclasses.asdict(config)
        record["energy_window"] = list(record["energy_window"])
        record["unknown_future_key"] = 1
        assert WangLandauConfig.from_dict(record) == config
        with pytest.raises(ConfigurationError, match="mapping"):
            WangLandauConfig.from_dict("not a mapping")  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError, match="size"):
            WangLandauConfig.from_dict({"lattice": {"size": 1}})

    def test_non_dyadic_couplings_need_a_bin_width(self) -> None:
        with pytest.raises(ValueError, match="bin_width"):
            _run(lattice=LatticeConfig(size=4, j1=1.0, j2=0.3), production_sweeps=10)
        results = _run(
            lattice=LatticeConfig(size=4, j1=1.0, j2=0.3),
            bin_width=0.5,
            log_f_final=1e-3,
            production_sweeps=100,
        )
        assert results.bin_width == 0.5


class TestResultsContainer:
    def test_shapes_and_diagnostics(self, square4: WangLandauResults) -> None:
        r = square4
        assert r.n_bins == 17 and r.bin_width == 4.0
        assert r.energy_bins[0] == -2.0 and r.energy_bins[-1] == 2.0
        assert r.window_bins == (0, 16)
        assert r.visited.sum() == 15  # E = +-28 do not exist on the 4 x 4 torus
        assert np.isnan(r.log_g[1]) and np.isnan(r.log_g[15])
        assert (
            r.wl_histogram.dtype == np.int64
            and r.production_histogram.sum() == 200_000 * 16
        )
        assert r.num_sites == 16 and r.n_samples == 200_000
        assert r.energy.shape == (200_000,) and r.staggered_magnetization.shape == (
            200_000,
            4,
        )
        assert r.bin_index.dtype == np.int64 and r.walker_id.max() == 0
        assert r.final_spins.shape == (16,) and isinstance(r.final_rng_state, bytes)
        assert isinstance(r.wang_landau, WangLandauDiagnostics)
        assert isinstance(r.production, MulticanonicalDiagnostics)
        assert isinstance(r.walkers[0], WalkerSeries)
        wl = r.wang_landau
        assert wl.converged and wl.final_log_f < 1e-5
        assert wl.n_iterations == len(wl.iteration_sweeps) == len(wl.iteration_flatness)
        assert wl.one_over_t_switch_sweep is not None
        assert 0.0 < wl.acceptance < 1.0
        assert sum(wl.iteration_sweeps) == wl.total_sweeps
        prod = r.production
        assert prod.n_walkers == 1 and prod.sweeps_per_walker == 200_000
        assert prod.histogram_flatness > 0.5
        assert prod.total_round_trips > 0 and prod.edge_bins == (0, 16)
        assert prod.acceptance.shape == (1,) and 0.0 < prod.acceptance[0] < 1.0
        assert r.metadata["kind"] == "wang_landau" and r.metadata["seed"] == 42
        assert r.config.lattice.size == 4
        assert r.metadata["elapsed_seconds"] > 0

    def test_config_property_requires_provenance(
        self, square4: WangLandauResults
    ) -> None:
        stripped = dataclasses.replace(square4, metadata={})
        with pytest.raises(ConfigurationError, match="metadata"):
            _ = stripped.config

    def test_order_columns(self, square4: WangLandauResults) -> None:
        assert square4.order_columns() == (1, 2)
        assert square4.order_columns((3,)) == (3,)
        with pytest.raises(ConfigurationError, match="order_parameter"):
            square4.order_columns((4,))
        with pytest.raises(ConfigurationError, match="order_parameter"):
            square4.order_columns(())

    def test_temperature_validation(self, square4: WangLandauResults) -> None:
        with pytest.raises(ConfigurationError, match="temperature"):
            square4.reweight(0.0)
        with pytest.raises(ConfigurationError, match="temperature"):
            square4.effective_sample_size(float("nan"))

    def test_multi_walker_series(self) -> None:
        r = _run(
            n_walkers=3,
            production_sweeps=500,
            measurement_interval=5,
            store_configs=True,
        )
        assert r.walker_lengths == (100, 100, 100)
        assert (
            r.n_samples == 300
            and r.walker_id.tolist() == [0] * 100 + [1] * 100 + [2] * 100
        )
        assert r.walkers[2].configurations is not None
        assert r.walkers[2].configurations.shape == (100, 4, 4)
        assert not np.array_equal(r.walkers[0].energy, r.walkers[1].energy)
        assert len(r.production.round_trips) == 3

    def test_without_production(self) -> None:
        r = _run(production_sweeps=0)
        assert r.n_samples == 0 and r.energy.size == 0
        assert r.staggered_magnetization.shape == (0, 4)
        assert r.walkers[0].n_samples == 0
        assert math.isnan(square_or_nan(r))


def square_or_nan(results: WangLandauResults) -> float:
    """Reweighting an empty series is total: nan, never an exception."""
    estimate = results.reweight(2.0)
    assert math.isnan(estimate.effective_samples)
    return estimate.energy.value


class TestExactDensityOfStates:
    @staticmethod
    def _exact_levels_3x3() -> dict[float, int]:
        """Degeneracy of every energy level of the 3 x 3 torus (J = 1)."""
        size, n = 3, 9
        states = np.arange(1 << n, dtype=np.uint32)
        bits = ((states[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int8)
        grid = (2 * bits - 1).reshape(-1, size, size)
        bonds = (grid * np.roll(grid, 1, axis=1)).sum(axis=(1, 2))
        bonds = bonds + (grid * np.roll(grid, 1, axis=2)).sum(axis=(1, 2))
        energies, counts = np.unique(-bonds, return_counts=True)
        return {float(e) / n: int(c) for e, c in zip(energies, counts, strict=True)}

    def test_wang_landau_reproduces_the_3x3_density_of_states(self) -> None:
        levels = self._exact_levels_3x3()
        r = _run(lattice=LatticeConfig(size=3, j1=1.0), production_sweeps=0)
        visited = {
            float(e): float(lg)
            for e, lg in zip(r.energy_bins, r.log_g, strict=True)
            if math.isfinite(lg)
        }
        assert set(visited) == set(levels)
        reference = max(levels, key=levels.get)
        shift = visited[reference] - math.log(levels[reference])
        deviations = [
            abs(visited[e] - shift - math.log(count)) for e, count in levels.items()
        ]
        assert max(deviations) <= 0.1, deviations
        # Normalised to 2^9 states the ground state is doubly degenerate.
        total = sum(math.exp(lg - shift) for lg in visited.values())
        assert total == pytest.approx(512.0, rel=0.05)


class TestReweighting:
    @pytest.mark.parametrize("temperature", [1.5, TC_SQUARE_2D, 4.0])
    def test_4x4_energy_and_specific_heat_match_ferdinand_fisher(
        self, square4: WangLandauResults, temperature: float
    ) -> None:
        estimate = square4.reweight(temperature)
        _assert_agrees(
            estimate.energy,
            square_torus_energy_per_site(4, temperature),
            label=f"<e> at T={temperature}",
            power=0.01,
        )
        _assert_agrees(
            estimate.specific_heat,
            square_torus_specific_heat(4, temperature),
            label=f"Cv at T={temperature}",
            power=0.10,
        )
        assert estimate.effective_samples > 100
        assert 0.0 <= estimate.edge_weight <= 1.0  # full spectrum: the ground states
        assert 0.0 <= estimate.abs_magnetization.value <= 1.0
        assert estimate.order_parameter.value <= 1.0

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_8x8_matches_ferdinand_fisher_over_seeds(self, seed: int) -> None:
        r = _run(
            lattice=LatticeConfig(size=8, j1=1.0),
            seed=seed,
            n_walkers=2,
            production_sweeps=100_000,
        )
        for temperature in (2.0, TC_SQUARE_2D, 3.0):
            estimate = r.reweight(temperature)
            _assert_agrees(
                estimate.energy,
                square_torus_energy_per_site(8, temperature),
                label=f"seed {seed} <e> at T={temperature}",
                power=0.01,
            )
            _assert_agrees(
                estimate.specific_heat,
                square_torus_specific_heat(8, temperature),
                label=f"seed {seed} Cv at T={temperature}",
                power=0.10,
            )

    def test_wrong_weights_still_give_the_exact_averages(self) -> None:
        """Frozen weights only set the variance: a crude quadratic ln g,
        used unchanged (``max_wl_sweeps=0``), reweights to the exact
        energy within errors."""
        energies = np.linspace(-32.0, 32.0, 17)
        crude = -((energies / 12.0) ** 2)
        config = WangLandauConfig(**SQUARE4, max_wl_sweeps=0)  # type: ignore[arg-type]
        r = WangLandauSimulation(config).run(show_progress=False, initial_log_g=crude)
        assert not r.wang_landau.converged and r.wang_landau.total_sweeps == 0
        np.testing.assert_array_equal(r.log_g, crude)
        for temperature in (TC_SQUARE_2D, 4.0):
            _assert_agrees(
                r.reweight(temperature).energy,
                square_torus_energy_per_site(4, temperature),
                label=f"crude weights <e> at T={temperature}",
                power=0.01,
            )

    def test_curve_dataframe_and_histogram(self, square4: WangLandauResults) -> None:
        temperatures = (2.0, 3.0)
        curve = square4.reweight_curve(temperatures)
        assert [c.temperature for c in curve] == list(temperatures)
        assert all(isinstance(c, CanonicalEstimates) for c in curve)
        frame = square4.to_dataframe(temperatures)
        assert list(frame["T"]) == list(temperatures)
        assert {"E", "E_err", "Cv", "psi", "U4_psi", "n_eff", "edge_weight"} <= set(
            frame.columns
        )
        assert frame["E"][0] == pytest.approx(curve[0].energy.value)
        energies, probability = square4.canonical_energy_histogram(2.0)
        assert energies is square4.energy_bins
        assert probability.sum() == pytest.approx(1.0)
        assert probability[~square4.visited].sum() == 0.0

    def test_refined_density_of_states_agrees_with_the_estimate(
        self, square4: WangLandauResults
    ) -> None:
        refined = square4.log_density_of_states()
        finite = np.isfinite(refined)
        assert finite.sum() == 15
        difference = (refined - square4.log_g)[finite]
        assert np.max(np.abs(difference - difference.mean())) < 0.2

    def test_barrier_and_transition_temperatures_are_total(
        self, square4: WangLandauResults
    ) -> None:
        """A single-peaked (if jagged, at N = 16) distribution never raises:
        the plateau-form barrier is ``nan`` and the temperature solvers
        return an :class:`Estimate` (``nan`` without a root)."""
        barrier = square4.free_energy_barrier(4.0)
        assert isinstance(barrier, BarrierEstimate)
        assert math.isnan(barrier.barrier.value) and math.isnan(barrier.energy_low)
        assert math.isnan(barrier.interface_tension.value)
        for temperature in (1.5, TC_SQUARE_2D):
            estimate = square4.free_energy_barrier(temperature, estimator="minimum")
            assert math.isnan(estimate.barrier.value) or estimate.barrier.value > 0.0
        assert isinstance(square4.equal_height_temperature((1.5, 3.5)), Estimate)
        assert isinstance(square4.equal_weight_temperature((1.5, 3.5), q=2.0), Estimate)
        with pytest.raises(ConfigurationError, match="q must"):
            square4.equal_weight_temperature((1.5, 3.5), q=0.0)

    def test_explicit_blocks_and_order_parameter(
        self, square4: WangLandauResults
    ) -> None:
        a = square4.reweight(2.0, n_blocks=10, order_parameter=(3,))
        b = square4.reweight(2.0, n_blocks=40, order_parameter=(3,))
        assert a.energy.value == pytest.approx(b.energy.value)
        assert a.order_parameter.value == pytest.approx(b.order_parameter.value)
        assert a.order_parameter.value != pytest.approx(
            square4.reweight(2.0).order_parameter.value
        )


class TestDeterminismAndDiagnostics:
    def test_same_seed_is_bit_identical_and_seeds_differ(self) -> None:
        a = _run(production_sweeps=1_000)
        b = _run(production_sweeps=1_000)
        np.testing.assert_array_equal(a.log_g, b.log_g)
        np.testing.assert_array_equal(a.energy, b.energy)
        np.testing.assert_array_equal(a.bin_index, b.bin_index)
        assert a.final_rng_state == b.final_rng_state
        c = _run(production_sweeps=1_000, seed=7)
        assert not np.array_equal(a.energy, c.energy)

    def test_capped_run_reports_not_converged(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        r = _run(max_wl_sweeps=5, production_sweeps=200)
        assert not r.wang_landau.converged and r.wang_landau.total_sweeps == 5
        assert r.production.rejected_unvisited[0] >= 0
        r.summary([2.0])
        out = capsys.readouterr().out
        assert "NOT CONVERGED" in out and "Reweighted" in out
        r.summary()
        assert "Reweighted" not in capsys.readouterr().out

    def test_window_and_drive_in(self) -> None:
        r = _run(
            lattice=LatticeConfig(size=8, j1=1.0),
            energy_window=(-2.0, -1.2),
            production_sweeps=2_000,
            log_f_final=1e-4,
        )
        lo, hi = r.window_bins
        assert lo == 0 and r.energy_bins[hi] <= -1.2
        assert r.wang_landau.drive_in_sweeps > 0
        assert np.all(r.energy <= -1.2 + 1e-12)
        assert r.production_histogram[hi + 1 :].sum() == 0
        with pytest.raises(ValueError, match="energy window"):
            _run(energy_window=(1.74, 1.76), drive_max_sweeps=50, production_sweeps=0)

    def test_staggered_columns_carry_the_order_parameter(self) -> None:
        r = _run(
            lattice=LatticeConfig(
                lattice_type=LatticeType.CUBIC, size=4, j1=1.0, j2=-0.5
            ),
            log_f_final=1e-3,
            production_sweeps=500,
        )
        assert r.order_columns() == (1, 2, 4)
        assert r.staggered_magnetization.shape == (500, 8)
        assert r.bin_width == 2.0 and r.energy_bins[0] == -6.0
        estimate = r.reweight(1.0, order_parameter=(1, 2, 4))
        assert math.isfinite(estimate.order_parameter.value)


@pytest.fixture(scope="module")
def rewl() -> WangLandauResults:
    """A 2 x 2 replica-exchange run on the 8 x 8 torus, shared by the class."""
    return _run(
        lattice=LatticeConfig(size=8, j1=1.0),
        n_windows=2,
        walkers_per_window=2,
        exchange_interval=20,
        check_interval=100,
        log_f_final=1e-5,
        n_walkers=2,
        production_sweeps=50_000,
    )


class TestReplicaExchangeStage:
    """The parallel first stage: windows, walkers, exchanges, one estimate."""

    def test_diagnostics_describe_the_layout(self, rewl: WangLandauResults) -> None:
        wl = rewl.wang_landau
        assert (wl.n_windows, wl.walkers_per_window) == (2, 2)
        assert len(wl.window_bins) == 2
        assert wl.window_bins[0][0] == 0 and wl.window_bins[1][1] == rewl.n_bins - 1
        assert wl.window_bins[1][0] < wl.window_bins[0][1]  # overlap
        assert len(wl.exchange_attempted) == 1 and wl.exchange_attempted[0] > 0
        assert wl.exchange_acceptance.shape == (1,)
        assert 0.0 < wl.exchange_acceptance[0] <= 1.0
        assert len(wl.merge_bins) == 1
        assert wl.window_bins[1][0] <= wl.merge_bins[0] <= wl.window_bins[0][1]
        assert wl.converged
        assert rewl.production.total_round_trips > 0

    @pytest.mark.parametrize("temperature", [2.0, TC_SQUARE_2D, 3.0])
    def test_matches_ferdinand_fisher(
        self, rewl: WangLandauResults, temperature: float
    ) -> None:
        estimate = rewl.reweight(temperature)
        _assert_agrees(
            estimate.energy,
            square_torus_energy_per_site(8, temperature),
            label=f"rewl <e> at T={temperature}",
            power=0.01,
        )
        _assert_agrees(
            estimate.specific_heat,
            square_torus_specific_heat(8, temperature),
            label=f"rewl Cv at T={temperature}",
            power=0.10,
        )

    def test_serial_diagnostics_defaults(self, square4: WangLandauResults) -> None:
        wl = square4.wang_landau
        assert (wl.n_windows, wl.walkers_per_window) == (1, 1)
        assert wl.window_bins == ((0, 16),)
        assert wl.exchange_attempted == () and wl.merge_bins == ()
        assert wl.exchange_acceptance.shape == (0,)

    def test_summary_mentions_the_layout(
        self, rewl: WangLandauResults, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rewl.summary()
        assert "2 window(s) x 2 walker(s)" in capsys.readouterr().out


@pytest.mark.slow
class TestSlowGates:
    def test_16x16_matches_ferdinand_fisher(self) -> None:
        r = _run(
            lattice=LatticeConfig(size=16, j1=1.0),
            log_f_final=1e-6,
            check_interval=500,
            n_walkers=4,
            production_sweeps=50_000,
        )
        assert r.wang_landau.converged and r.production.total_round_trips > 0
        for temperature in (2.0, TC_SQUARE_2D, 2.6):
            estimate = r.reweight(temperature)
            _assert_agrees(
                estimate.energy,
                square_torus_energy_per_site(16, temperature),
                label=f"16x16 <e> T={temperature}",
                power=0.01,
            )
            _assert_agrees(
                estimate.specific_heat,
                square_torus_specific_heat(16, temperature),
                label=f"16x16 Cv T={temperature}",
                power=0.10,
            )

    @pytest.mark.statistical
    def test_square_j1j2_matches_parallel_tempering(self) -> None:
        """Stripe order parameter, energy and Cv of the J1-J2 square lattice
        at |J2|/J1 = 0.625 (a dyadic ratio inside the weakly first-order
        window) agree with a replica-exchange run across the transition
        (T_c ~ 1.1 at L = 16: the Cv peak of a canonical scan)."""
        lattice = LatticeConfig(size=16, j1=1.0, j2=-0.625)
        temperatures = (1.0, 1.1, 1.2, 1.5)
        pt = Simulation(
            SimulationConfig(
                lattice=lattice,
                algorithm=Algorithm.METROPOLIS,
                mode=ExecutionMode.PARALLEL_TEMPERING,
                temperatures=(0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 2.0),
                n_sweeps=40_000,
                n_thermalization=10_000,
                measurement_interval=10,
                swap_interval=5,
                store_configs=False,
            )
        ).run(show_progress=False)
        assert pt.pt_diagnostics is not None
        assert pt.pt_diagnostics.total_round_trips > 0
        # The window covers the ground state (e = 2 J2 = -1.25) up to the
        # energies of the disordered phase at T = 2 (a canonical scan
        # reaches about -0.4 there).
        wl = _run(
            lattice=lattice,
            energy_window=(-1.3, -0.2),
            log_f_final=1e-6,
            check_interval=500,
            n_walkers=8,
            production_sweeps=50_000,
        )
        assert wl.wang_landau.converged and min(wl.production.round_trips) > 0
        from mcising.statistics import mean_estimate

        for temperature in temperatures:
            estimate = wl.reweight(temperature, order_parameter=(1, 2))
            assert estimate.edge_weight < 1e-3, (temperature, estimate.edge_weight)
            stats = pt.statistics(temperature)
            stripe = np.abs(pt.staggered_magnetization[temperature][:, 1:3]).max(axis=1)
            pt_stripe = mean_estimate(stripe)
            for label, a, b in (
                ("Cv", estimate.specific_heat, stats.specific_heat),
                ("<e>", estimate.energy, stats.energy),
                ("stripe", estimate.order_parameter, pt_stripe),
            ):
                combined = math.hypot(a.error, b.error)
                assert math.isfinite(combined) and combined > 0.0, (
                    f"{label} T={temperature}"
                )
                assert abs(a.value - b.value) <= 4.0 * combined, (
                    f"{label} at T={temperature}: WL {a} vs PT {b}"
                )

    @pytest.mark.statistical
    def test_cubic_j1j2_first_order_transition(self) -> None:
        """The layered transition of the cubic J1-J2 model at |J2|/J1 = 0.5,
        the case the method was built for. At L = 16 the canonical energy
        histogram is bimodal: the equal-height and equal-weight
        temperatures exist, agree, and sit at the transition (about 2.40),
        the two phases differ by the latent heat, and the barrier is
        positive. At L = 12 the peaks still overlap, so that size checks
        the reweighted energy, specific heat and layered order parameter
        against canonical runs where canonical Monte Carlo still mixes."""
        cubic = LatticeConfig(lattice_type=LatticeType.CUBIC, size=16, j1=1.0, j2=-0.5)
        r16 = _run(
            lattice=cubic,
            energy_window=(-1.7, -0.5),
            log_f_final=1e-6,
            check_interval=200,
            n_walkers=8,
            production_sweeps=100_000,
            production_thermalization=2_000,
        )
        assert r16.wang_landau.converged
        assert min(r16.production.round_trips) > 0, r16.production.round_trips
        t_h = r16.equal_height_temperature((2.2, 2.6))
        t_w = r16.equal_weight_temperature((2.2, 2.6), q=6.0)
        assert 2.35 < t_h.value < 2.45, t_h
        assert math.isfinite(t_h.error) and t_h.error < 0.05, t_h
        assert abs(t_w.value - t_h.value) < 0.03, (t_h, t_w)
        barrier = r16.free_energy_barrier(t_h.value)
        assert barrier.energy_high - barrier.energy_low > 0.1, barrier  # latent heat
        assert barrier.barrier.value > 0.0, barrier
        assert math.isfinite(barrier.barrier.error), barrier
        assert barrier.interface_tension.value > 0.0
        for temperature in (2.3, 2.5):
            estimate = r16.reweight(temperature)
            assert estimate.edge_weight < 1e-3, (temperature, estimate.edge_weight)
        assert r16.reweight(2.3).order_parameter.value > 0.6
        assert r16.reweight(2.5).order_parameter.value < 0.3

        cubic12 = LatticeConfig(
            lattice_type=LatticeType.CUBIC, size=12, j1=1.0, j2=-0.5
        )
        r12 = _run(
            lattice=cubic12,
            energy_window=(-1.7, -0.5),
            log_f_final=1e-6,
            check_interval=200,
            n_walkers=8,
            production_sweeps=60_000,
            production_thermalization=2_000,
        )
        assert r12.wang_landau.converged and min(r12.production.round_trips) > 0
        canonical = Simulation(
            SimulationConfig(
                lattice=cubic12,
                mode=ExecutionMode.INDEPENDENT,
                temperatures=(2.30, 2.40, 2.50),
                n_sweeps=60_000,
                n_thermalization=20_000,
                measurement_interval=10,
                store_configs=False,
            )
        ).run(show_progress=False)
        from mcising.statistics import mean_estimate

        for temperature in canonical.temperatures:
            estimate = r12.reweight(temperature)
            stats = canonical.statistics(temperature)
            layered = np.abs(
                canonical.staggered_magnetization[temperature][:, [1, 2, 4]]
            ).max(axis=1)
            psi = mean_estimate(layered)
            for label, a, b in (
                ("<e>", estimate.energy, stats.energy),
                ("Cv", estimate.specific_heat, stats.specific_heat),
                ("psi", estimate.order_parameter, psi),
            ):
                combined = math.hypot(a.error, b.error)
                assert math.isfinite(combined) and combined > 0.0, (label, temperature)
                assert abs(a.value - b.value) <= 4.0 * combined, (
                    f"{label} at T={temperature}: WL {a} vs canonical {b}"
                )
