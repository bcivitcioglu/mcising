"""Tests for configuration dataclasses and validation."""

from __future__ import annotations

import math

import pytest

from mcising.config import Algorithm, LatticeConfig, LatticeType, SimulationConfig


class TestLatticeConfig:
    def test_defaults(self) -> None:
        cfg = LatticeConfig()
        assert cfg.lattice_type == LatticeType.SQUARE
        assert cfg.size == 10
        assert cfg.j1 == 1.0
        assert cfg.j2 == 0.0
        assert cfg.h == 0.0

    def test_custom_values(self) -> None:
        cfg = LatticeConfig(size=32, j1=1.5, j2=0.3, h=0.1)
        assert cfg.size == 32
        assert cfg.j1 == 1.5
        assert cfg.j2 == 0.3
        assert cfg.h == 0.1

    def test_frozen(self) -> None:
        cfg = LatticeConfig()
        with pytest.raises(AttributeError):
            cfg.size = 20  # type: ignore[misc]

    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError, match="Lattice size must be >= 2"):
            LatticeConfig(size=1)

    def test_invalid_j1_raises(self) -> None:
        with pytest.raises(ValueError, match="j1"):
            LatticeConfig(j1=math.inf)

    def test_invalid_j2_raises(self) -> None:
        with pytest.raises(ValueError, match="j2"):
            LatticeConfig(j2=float("nan"))

    def test_invalid_h_raises(self) -> None:
        with pytest.raises(ValueError, match="h"):
            LatticeConfig(h=float("-inf"))


class TestSimulationConfig:
    def test_defaults(self) -> None:
        cfg = SimulationConfig()
        assert cfg.algorithm == Algorithm.METROPOLIS
        assert cfg.seed == 42
        assert cfg.temperatures == (2.269,)
        assert cfg.n_sweeps == 1000
        assert cfg.compute_correlation is False

    def test_frozen(self) -> None:
        cfg = SimulationConfig()
        with pytest.raises(AttributeError):
            cfg.seed = 99  # type: ignore[misc]

    def test_invalid_n_sweeps(self) -> None:
        with pytest.raises(ValueError, match="n_sweeps"):
            SimulationConfig(n_sweeps=0)

    def test_invalid_n_thermalization(self) -> None:
        with pytest.raises(ValueError, match="n_thermalization"):
            SimulationConfig(n_thermalization=-1)

    def test_invalid_measurement_interval(self) -> None:
        with pytest.raises(ValueError, match="measurement_interval"):
            SimulationConfig(measurement_interval=0)

    def test_invalid_temperature_zero(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            SimulationConfig(temperatures=(0.0,))

    def test_invalid_temperature_negative(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            SimulationConfig(temperatures=(-1.0,))

    def test_multiple_temperatures(self) -> None:
        cfg = SimulationConfig(temperatures=(3.0, 2.269, 1.5))
        assert len(cfg.temperatures) == 3


class TestEnums:
    def test_lattice_type_value(self) -> None:
        assert LatticeType.SQUARE.value == "square"

    def test_algorithm_value(self) -> None:
        assert Algorithm.METROPOLIS.value == "metropolis"