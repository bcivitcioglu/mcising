"""Tests for configuration dataclasses and validation."""

from __future__ import annotations

import math
from dataclasses import asdict

import pytest
from mcising.config import (
    AdaptiveConfig,
    Algorithm,
    ExecutionMode,
    LatticeConfig,
    LatticeType,
    SimulationConfig,
)
from mcising.exceptions import ConfigurationError


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


class TestStoreConfigs:
    def test_default_true(self) -> None:
        # Load-bearing default: existing analysis and I/O paths expect
        # configurations to be stored unless explicitly disabled.
        assert SimulationConfig().store_configs is True

    def test_disable(self) -> None:
        cfg = SimulationConfig(store_configs=False)
        assert cfg.store_configs is False


class TestSwapCadence:
    """PT requires measurement_interval to be a multiple of swap_interval (B5)."""

    def test_pt_rejects_nondividing(self) -> None:
        with pytest.raises(ConfigurationError, match="multiple of swap_interval"):
            SimulationConfig(
                mode=ExecutionMode.PARALLEL_TEMPERING,
                measurement_interval=15,
                swap_interval=10,
            )

    @pytest.mark.parametrize("swap_interval", [1, 5, 10])
    def test_pt_accepts_dividing(self, swap_interval: int) -> None:
        cfg = SimulationConfig(
            mode=ExecutionMode.PARALLEL_TEMPERING,
            measurement_interval=10,
            swap_interval=swap_interval,
        )
        assert cfg.swap_interval == swap_interval

    def test_cadence_unchecked_outside_pt(self) -> None:
        # swap_interval is inert outside PT; the guard must not reject
        # configs it does not apply to.
        cfg = SimulationConfig(measurement_interval=15, swap_interval=10)
        assert cfg.measurement_interval == 15


class TestFromDict:
    """SimulationConfig.from_dict inverts dataclasses.asdict."""

    def test_roundtrip_defaults(self) -> None:
        config = SimulationConfig()
        assert SimulationConfig.from_dict(asdict(config)) == config

    @pytest.mark.parametrize("mode", list(ExecutionMode), ids=lambda m: m.value)
    def test_roundtrip_all_modes(self, mode: ExecutionMode) -> None:
        config = SimulationConfig(mode=mode, temperatures=(3.0, 2.0))
        assert SimulationConfig.from_dict(asdict(config)) == config

    @pytest.mark.parametrize("lattice_type", list(LatticeType), ids=lambda t: t.value)
    def test_roundtrip_all_lattices(self, lattice_type: LatticeType) -> None:
        config = SimulationConfig(
            lattice=LatticeConfig(lattice_type=lattice_type, size=4, j1=-1.0, h=0.2)
        )
        assert SimulationConfig.from_dict(asdict(config)) == config

    def test_roundtrip_through_json(self) -> None:
        import json

        config = SimulationConfig(
            lattice=LatticeConfig(size=6, j2=0.5),
            algorithm=Algorithm.METROPOLIS,
            seed=123,
            temperatures=(4.0, 2.269, 1.0),
            adaptive=AdaptiveConfig(enabled=True, c_window=8.0),
        )
        data = json.loads(json.dumps(asdict(config)))
        assert SimulationConfig.from_dict(data) == config

    def test_enums_are_members(self) -> None:
        config = SimulationConfig.from_dict(
            {"algorithm": "wolff", "mode": "independent"}
        )
        assert config.algorithm is Algorithm.WOLFF
        assert config.mode is ExecutionMode.INDEPENDENT

    def test_temperatures_becomes_tuple(self) -> None:
        config = SimulationConfig.from_dict({"temperatures": [3.0, 2.0]})
        assert config.temperatures == (3.0, 2.0)
        assert isinstance(config.temperatures, tuple)

    def test_unknown_keys_ignored(self) -> None:
        # Forward compatibility: a newer schema's extra fields must not
        # prevent loading.
        config = SimulationConfig.from_dict({"seed": 5, "error_model": "jackknife"})
        assert config.seed == 5

    def test_missing_keys_default(self) -> None:
        config = SimulationConfig.from_dict({})
        assert config == SimulationConfig()

    def test_bad_enum_raises_listing_valid(self) -> None:
        with pytest.raises(ConfigurationError, match="metropolis"):
            SimulationConfig.from_dict({"algorithm": "quantum"})

    def test_invalid_value_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="size"):
            SimulationConfig.from_dict({"lattice": {"size": 1}})

    def test_validation_runs(self) -> None:
        with pytest.raises(ConfigurationError, match="swap_interval"):
            SimulationConfig.from_dict(
                {
                    "mode": "parallel_tempering",
                    "measurement_interval": 15,
                    "swap_interval": 10,
                }
            )

    def test_non_mapping_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="mapping"):
            SimulationConfig.from_dict([("seed", 5)])  # type: ignore[arg-type]
