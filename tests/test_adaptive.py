"""Adaptive-mode correctness suite (ROADMAP P09, B9/#20).

The load-bearing guarantee: stationarity (MSER) and tau_int (Sokal) are
estimated exclusively on fixed-temperature data — never across the
cooldown temperature ramp — and the production measurement interval
derives from the stationary tail of that fixed-temperature series only.

The spy tests monkeypatch the module-level ``_analyze_thermalization``
seam (PyO3 classes reject attribute assignment, so the Rust static
method itself cannot be patched) and prove the analyzed series is
bit-exact equal to an independently generated fixed-temperature block.

Thresholds calibrated in-phase on 16x16 Metropolis over DEFAULT_SEEDS
(release build): tau(T=4.0) in [1.16, 1.38] (bound 5.0), tau(T=2.269)
in [4.55, 7.31]; adaptive end-to-end thermalizes on the first 200-sweep
diagnostic block for every seed at T in (2.6, 2.269, 1.8).
"""

from __future__ import annotations

from typing import Any

import mcising.simulation
import numpy as np
import pytest
from mcising import Simulation, SimulationConfig
from mcising._core import IsingSimulation
from mcising.config import AdaptiveConfig, LatticeConfig
from mcising.constants import INF_TEMP
from mcising.simulation import AdaptiveDiagnostics
from mcising.statistics import tau_int as py_tau_int

from tests._stats import DEFAULT_SEEDS, assert_ordered_means

C_WINDOW = 6.0
TAU_MULTIPLIER = 2.0


def _adaptive_config(seed: int, **adaptive_kwargs: Any) -> SimulationConfig:
    """One-temperature 8x8 adaptive config with a 100-sweep ramp/block."""
    adaptive_kwargs.setdefault("enabled", True)
    adaptive_kwargs.setdefault("min_thermalization_sweeps", 100)
    adaptive_kwargs.setdefault("min_independent_samples", 20)
    return SimulationConfig(
        lattice=LatticeConfig(size=8),
        temperatures=(4.0,),
        n_thermalization=100,
        adaptive=AdaptiveConfig(**adaptive_kwargs),
        seed=seed,
    )


class TestAnalyzedSeriesIsFixedTemperatureOnly:
    """The P09 spy gate: the analyzer never sees ramp data."""

    def test_series_is_bit_exact_fixed_temperature_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _adaptive_config(seed=42)
        recorded: list[np.ndarray] = []
        real = mcising.simulation._analyze_thermalization

        def spy(series: np.ndarray, ac: AdaptiveConfig) -> dict[str, Any]:
            recorded.append(np.array(series, copy=True))
            return real(series, ac)

        monkeypatch.setattr(mcising.simulation, "_analyze_thermalization", spy)
        Simulation(config).run(show_progress=False)

        assert recorded, "analyzer was never called"
        # The first analyzed series is exactly one diagnostic block:
        # block = max(min_thermalization_sweeps, MIN_DIAGNOSTIC_SWEEPS).
        assert len(recorded[0]) == 100

        # Bit-exact identity: a twin core (same config, same seed) run
        # through ramp + fixed-T block by hand reproduces the analyzed
        # series byte for byte — so what was analyzed IS the fixed-T
        # block, with zero ramp samples in it. Threshold-free.
        twin = Simulation(config)
        schedule = np.linspace(INF_TEMP, 4.0, num=100).tolist()
        twin._core.thermalize_with_diagnostics(schedule)
        block = np.asarray(twin._core.extend_thermalization(100, 1.0 / 4.0))
        assert np.array_equal(recorded[0], block)

        # Every later analyzed series only extends the fixed-T block.
        for earlier, later in zip(recorded, recorded[1:]):
            assert np.array_equal(later[: len(earlier)], earlier)

    def test_measurement_interval_derives_from_analyzed_series(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _adaptive_config(seed=123)
        recorded: list[np.ndarray] = []
        real = mcising.simulation._analyze_thermalization

        def spy(series: np.ndarray, ac: AdaptiveConfig) -> dict[str, Any]:
            recorded.append(np.array(series, copy=True))
            return real(series, ac)

        monkeypatch.setattr(mcising.simulation, "_analyze_thermalization", spy)
        results = Simulation(config).run(show_progress=False)

        assert results.adaptive_diagnostics is not None
        diag = results.adaptive_diagnostics[4.0]
        # Recomputing the analysis of the LAST recorded (fixed-T) series
        # must reproduce the stored tau and interval exactly: the
        # interval is a pure function of the fixed-temperature series.
        analysis = IsingSimulation.analyze_thermalization_series(
            recorded[-1], C_WINDOW, TAU_MULTIPLIER
        )
        assert diag.tau_int == analysis["tau_int"]
        assert diag.measurement_interval == analysis["recommended_interval"]
        assert diag.measurement_interval == round(TAU_MULTIPLIER * diag.tau_int)


class TestExtensionLoop:
    """The not-thermalized branch is reachable and budget-capped (B9)."""

    @staticmethod
    def _fake(thermalized: bool, interval: int = 1) -> dict[str, Any]:
        return {
            "is_thermalized": thermalized,
            "truncation_point": 0,
            "tau_int": 0.5,
            "window": 0,
            "recommended_interval": interval,
        }

    def test_extension_runs_while_not_thermalized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _adaptive_config(seed=7)
        calls = 0

        def stub(series: np.ndarray, ac: AdaptiveConfig) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return self._fake(thermalized=calls >= 3)

        monkeypatch.setattr(mcising.simulation, "_analyze_thermalization", stub)
        results = Simulation(config).run(show_progress=False)

        assert calls == 3
        assert results.adaptive_diagnostics is not None
        diag = results.adaptive_diagnostics[4.0]
        # Initial 100-sweep block + two 100-sweep extensions, ramp on top.
        assert diag.stationary_sweeps == 300
        assert diag.thermalization_sweeps == 100 + 300
        assert diag.is_thermalized

    def test_cap_terminates_loop_and_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _adaptive_config(seed=2024, max_thermalization_sweeps=350)

        def stub(series: np.ndarray, ac: AdaptiveConfig) -> dict[str, Any]:
            return self._fake(thermalized=False)

        monkeypatch.setattr(mcising.simulation, "_analyze_thermalization", stub)
        with pytest.warns(UserWarning, match="Thermalization not detected"):
            results = Simulation(config).run(show_progress=False)

        assert results.adaptive_diagnostics is not None
        diag = results.adaptive_diagnostics[4.0]
        # Ramp 100 + initial block 100 + extensions 100 + 50 hit the
        # 350-sweep total cap exactly: stationary = 350 - 100.
        assert diag.stationary_sweeps == 250
        assert diag.thermalization_sweeps == 350
        assert not diag.is_thermalized

    def test_budget_starved_production_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An honest interval that the max_total_sweeps budget cannot
        # afford must warn, not silently deliver fewer samples.
        config = _adaptive_config(
            seed=31337,
            min_independent_samples=100,
            max_total_sweeps=1000,
        )

        def stub(series: np.ndarray, ac: AdaptiveConfig) -> dict[str, Any]:
            return self._fake(thermalized=True, interval=50)

        monkeypatch.setattr(mcising.simulation, "_analyze_thermalization", stub)
        with pytest.warns(UserWarning, match="Sweep budget"):
            results = Simulation(config).run(show_progress=False)

        assert results.adaptive_diagnostics is not None
        diag = results.adaptive_diagnostics[4.0]
        # therm 200, remaining 800, interval 50 -> 16 measurements.
        assert diag.n_samples == 16


class TestStationaryTau:
    """tau_int on genuinely stationary series (ROADMAP P09 task)."""

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_tau_short_correlation_regime_at_high_t(self, seed: int) -> None:
        # 16x16 Metropolis at T=4 (well above Tc): calibrated tau in
        # [1.16, 1.38]; the 5.0 bound is > 3x the observed max while
        # still excluding the critical regime (tau >= 4.5 at Tc).
        sim = IsingSimulation(16, 1.0, 0.0, 0.0, 0.0, seed)
        beta = 1.0 / 4.0
        sim.sweep(500, beta)
        series = np.asarray(sim.extend_thermalization(4000, beta))
        analysis = IsingSimulation.analyze_thermalization_series(
            series, C_WINDOW, TAU_MULTIPLIER
        )
        assert analysis["is_thermalized"]
        assert 0.5 <= analysis["tau_int"] <= 5.0

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_critical_slowing_down_ordering(self, seed: int) -> None:
        # Constant-free cross-check: tau at Tc must exceed tau at T=4.
        # Calibrated: per-seed gap at least 4.55 vs at most 1.38.
        taus = {}
        for temp in (4.0, 2.269):
            sim = IsingSimulation(16, 1.0, 0.0, 0.0, 0.0, seed)
            beta = 1.0 / temp
            sim.sweep(500, beta)
            series = np.asarray(sim.extend_thermalization(4000, beta))
            analysis = IsingSimulation.analyze_thermalization_series(
                series, C_WINDOW, TAU_MULTIPLIER
            )
            taus[temp] = analysis["tau_int"]
        assert taus[2.269] > taus[4.0]

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_rust_sokal_and_python_plateau_estimators_agree(
        self, seed: int
    ) -> None:
        # The two tau estimators in the codebase (Rust Sokal windowing,
        # Python blocking plateau rule from P08) must agree on the same
        # AR(1) series, and both with theory: tau = (1+phi)/(2(1-phi)).
        phi = 0.9
        theory = (1.0 + phi) / (2.0 * (1.0 - phi))
        rng = np.random.default_rng(seed)
        n = 100_000
        noise = rng.standard_normal(n)
        series = np.empty(n)
        x = 0.0
        for i in range(n):
            x = phi * x + noise[i]
            series[i] = x

        rust = IsingSimulation.analyze_thermalization_series(
            series, C_WINDOW, TAU_MULTIPLIER
        )["tau_int"]
        # min_blocks=256 is the plateau estimator's documented accuracy
        # setting (statistics.tau_int docstring); the default 32 keeps
        # only ~50 blocks at the deepest level, whose SE-of-SE noise
        # biased tau to 16.2 on this series during calibration.
        python = py_tau_int(series, min_blocks=256)

        assert abs(rust - theory) / theory < 0.25
        assert abs(python - theory) / theory < 0.25
        assert abs(rust - python) / max(rust, python) < 0.25


class TestAdaptiveEndToEnd:
    """The P09 regression gate: adaptive on 16x16 still converges."""

    @pytest.mark.statistical
    @pytest.mark.parametrize("seed", DEFAULT_SEEDS)
    def test_16x16_converges(self, seed: int) -> None:
        temps = (2.6, 2.269, 1.8)
        config = SimulationConfig(
            lattice=LatticeConfig(size=16),
            temperatures=temps,
            adaptive=AdaptiveConfig(enabled=True, min_independent_samples=50),
            seed=seed,
        )
        results = Simulation(config).run(show_progress=False)

        assert results.adaptive_diagnostics is not None
        for temp in temps:
            diag = results.adaptive_diagnostics[temp]
            assert diag.is_thermalized, f"T={temp} not thermalized"
            assert diag.n_samples == 50
            assert 1 <= diag.measurement_interval <= 200
            # Fixed-T diagnostics happened, and the recorded split is
            # consistent: block = max(min_therm, MIN_DIAGNOSTIC) = 200.
            assert diag.stationary_sweeps >= 200
            assert diag.thermalization_sweeps > diag.stationary_sweeps
            assert diag.tau_int > 0

        # Physics: mean energy rises with temperature.
        assert_ordered_means(
            [
                (f"T={t} (seed={seed})", results.energy[t])
                for t in sorted(temps)
            ],
            increasing=True,
        )


class TestDiagnosticsRoundTrip:
    """First HDF5 round-trip coverage for adaptive_diagnostics."""

    def test_hdf5_round_trip_preserves_all_fields(self, tmp_path: Any) -> None:
        from mcising.io import load_hdf5, save_hdf5

        config = _adaptive_config(seed=42)
        results = Simulation(config).run(show_progress=False)
        assert results.adaptive_diagnostics is not None
        path = tmp_path / "adaptive.h5"
        save_hdf5(results, path)

        loaded = load_hdf5(path)
        assert loaded.adaptive_diagnostics is not None
        original = results.adaptive_diagnostics[4.0]
        restored = loaded.adaptive_diagnostics[4.0]
        assert restored == original
        assert restored.stationary_sweeps > 0

    def test_pre_p09_file_without_stationary_sweeps_loads(
        self, tmp_path: Any
    ) -> None:
        import h5py
        from mcising.io import load_hdf5, save_hdf5

        config = _adaptive_config(seed=123)
        results = Simulation(config).run(show_progress=False)
        path = tmp_path / "legacy_adaptive.h5"
        save_hdf5(results, path)

        # Simulate a file written before the attribute existed.
        removed = 0
        with h5py.File(path, "r+") as f:
            for name in f:
                if name.startswith("T=") and "adaptive_diagnostics" in f[name]:
                    del f[name]["adaptive_diagnostics"].attrs[
                        "stationary_sweeps"
                    ]
                    removed += 1
        assert removed == 1

        loaded = load_hdf5(path)
        assert loaded.adaptive_diagnostics is not None
        diag = loaded.adaptive_diagnostics[4.0]
        assert isinstance(diag, AdaptiveDiagnostics)
        assert diag.stationary_sweeps == 0
