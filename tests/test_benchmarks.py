"""Tests for the benchmark baselines in mcising.benchmarks.

The pure-Python and NumPy baselines run at tiny sizes (their hardcoded
100-sweep warmups are ~1600 spin updates at L=4 — negligible); peapods is
a deliberately undeclared competitor package, so its runner is exercised
against an injected stub module and its absence contract is pinned.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest
from mcising.benchmarks import (
    BenchmarkResult,
    bench_mcising,
    bench_numpy,
    bench_peapods,
    bench_pure_python,
)
from mcising.constants import TC_SQUARE_2D


class TestBenchmarkResult:
    """The dataclass arithmetic every CLI table is built from."""

    def test_total_updates_prefers_attempted(self) -> None:
        result = BenchmarkResult(
            name="x",
            lattice_size=4,
            n_sweeps=2,
            elapsed=1.0,
            energy=0.0,
            magnetization=0.0,
            num_sites=54,
            attempted_updates=7,
        )
        assert result.total_updates == 7

    def test_total_updates_num_sites_arm(self) -> None:
        # num_sites beats the L² fallback (54*2=108, not 4²*2=32) — the
        # arm that keeps non-square lattices honest.
        result = BenchmarkResult(
            name="x",
            lattice_size=4,
            n_sweeps=2,
            elapsed=1.0,
            energy=0.0,
            magnetization=0.0,
            num_sites=54,
        )
        assert result.total_updates == 108

    def test_total_updates_square_fallback(self) -> None:
        result = BenchmarkResult(
            name="x",
            lattice_size=4,
            n_sweeps=2,
            elapsed=1.0,
            energy=0.0,
            magnetization=0.0,
        )
        assert result.total_updates == 32

    def test_zero_elapsed_guards(self) -> None:
        result = BenchmarkResult(
            name="x",
            lattice_size=4,
            n_sweeps=10,
            elapsed=0.0,
            energy=0.0,
            magnetization=0.0,
            attempted_updates=100,
        )
        assert result.updates_per_sec == 0.0
        assert result.sweeps_per_sec == 0.0

    def test_rates_with_positive_elapsed(self) -> None:
        result = BenchmarkResult(
            name="x",
            lattice_size=4,
            n_sweeps=10,
            elapsed=2.0,
            energy=0.0,
            magnetization=0.0,
            attempted_updates=100,
        )
        assert result.updates_per_sec == 50.0
        assert result.sweeps_per_sec == 5.0


def _assert_physical(result: BenchmarkResult, name: str, lattice_size: int) -> None:
    """Shared sanity block: labels echoed, observables in physical range."""
    assert result.name == name
    assert result.lattice_size == lattice_size
    assert result.elapsed >= 0.0
    # NN square lattice: |E|/site <= 2, |M|/site <= 1.
    assert -2.0 <= result.energy <= 2.0
    assert -1.0 <= result.magnetization <= 1.0


class TestPurePython:
    def test_bench_pure_python_tiny(self) -> None:
        result = bench_pure_python(4, 5)
        _assert_physical(result, "Pure Python", 4)
        assert result.n_sweeps == 5
        assert result.num_sites is None
        assert result.attempted_updates is None

    def test_pure_python_seed_deterministic(self) -> None:
        first = bench_pure_python(4, 5, seed=123)
        second = bench_pure_python(4, 5, seed=123)
        assert first.energy == second.energy
        assert first.magnetization == second.magnetization


class TestNumpy:
    def test_bench_numpy_tiny(self) -> None:
        result = bench_numpy(4, 5)
        _assert_physical(result, "NumPy", 4)
        assert result.n_sweeps == 5
        assert result.num_sites is None

    def test_numpy_seed_deterministic(self) -> None:
        first = bench_numpy(4, 5, seed=123)
        second = bench_numpy(4, 5, seed=123)
        assert first.energy == second.energy
        assert first.magnetization == second.magnetization


class TestMcising:
    def test_bench_mcising_metropolis_tiny(self) -> None:
        result = bench_mcising(4, 5)
        _assert_physical(result, "mcising (metropolis)", 4)
        assert result.num_sites == 16
        # Metropolis attempts exactly num_sites flips per timed sweep;
        # the warmup's attempts must not leak into the count.
        assert result.attempted_updates == 5 * 16
        assert result.total_updates == 5 * 16

    def test_bench_mcising_wolff_tiny(self) -> None:
        result = bench_mcising(4, 5, algorithm="wolff")
        _assert_physical(result, "mcising (wolff)", 4)
        assert result.num_sites == 16
        # One Wolff sweep is one cluster — attempted is the real work
        # count, not n_sweeps * num_sites; only its presence is pinned.
        assert result.attempted_updates is not None
        assert result.attempted_updates >= 0


def _install_peapods_stub(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Inject a recording ``peapods`` stub into sys.modules.

    ``bench_peapods`` imports function-locally, so the injection takes
    effect at call time. Returns the list that collects every constructed
    ``Ising`` instance.
    """
    instances: list[Any] = []

    class _Sim:
        def __init__(self) -> None:
            self.reset_seeds: list[int] = []

        def reset(self, seed: int) -> None:
            self.reset_seeds.append(seed)

    class _Ising:
        def __init__(
            self, shape: tuple[int, ...], temperatures: Any, **kwargs: Any
        ) -> None:
            self.shape = shape
            self.temperatures = temperatures
            self.ctor_kwargs = dict(kwargs)
            self._sim = _Sim()
            self.sample_calls: list[dict[str, Any]] = []
            self.energies_avg = np.array([-1.4])
            self.mags = np.array([0.5])
            instances.append(self)

        def sample(self, **kwargs: Any) -> None:
            self.sample_calls.append(dict(kwargs))

    module = types.ModuleType("peapods")
    module.Ising = _Ising  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peapods", module)
    return instances


class TestPeapods:
    def test_bench_peapods_stub_square(self, monkeypatch: pytest.MonkeyPatch) -> None:
        instances = _install_peapods_stub(monkeypatch)
        result = bench_peapods(4, 3, seed=7)

        assert result.name == "peapods"
        # peapods reports the opposite energy sign — the negation is
        # load-bearing.
        assert result.energy == pytest.approx(1.4)
        assert result.magnetization == pytest.approx(0.5)
        assert result.num_sites == 16
        assert result.n_sweeps == 3

        (model,) = instances
        assert model.shape == (4, 4)
        assert model.temperatures == pytest.approx([TC_SQUARE_2D])
        # Square is peapods' default geometry — the kwarg must be absent.
        assert model.ctor_kwargs == {"couplings": "ferro"}
        assert model._sim.reset_seeds == [7]
        # Warmup (100 sweeps) then the timed run, no cluster kwargs.
        assert model.sample_calls == [
            {"n_sweeps": 100, "sweep_mode": "metropolis", "warmup_ratio": 0.0},
            {"n_sweeps": 3, "sweep_mode": "metropolis", "warmup_ratio": 0.0},
        ]

    def test_bench_peapods_stub_cluster_and_geometry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        instances = _install_peapods_stub(monkeypatch)
        result = bench_peapods(4, 3, geometry="triangular", dim=3, cluster_mode="wolff")

        assert result.num_sites == 64
        (model,) = instances
        assert model.shape == (4, 4, 4)
        assert model.ctor_kwargs == {"couplings": "ferro", "geometry": "triangular"}
        for call in model.sample_calls:
            assert call["cluster_update_interval"] == 1
            assert call["cluster_mode"] == "wolff"

    def test_bench_peapods_absent_raises_importerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # peapods is deliberately undeclared: without it installed the
        # runner must surface a plain ImportError, not a silent skip.
        # A None sys.modules entry forces the ImportError even on
        # machines where peapods happens to be installed.
        monkeypatch.setitem(sys.modules, "peapods", None)
        with pytest.raises(ImportError):
            bench_peapods(4, 3)
