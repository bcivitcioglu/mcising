"""Tests for magnetization computation via the Rust core."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from mcising._core import IsingSimulation
from mcising.config import (
    ExecutionMode,
    LatticeConfig,
    LatticeType,
    SimulationConfig,
)
from mcising.simulation import Simulation


class TestMagnetization:
    def test_all_up_magnetization(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        assert sim.magnetization() == pytest.approx(1.0)

    def test_all_down_magnetization(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        spins = -np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        assert sim.magnetization() == pytest.approx(-1.0)

    def test_checkerboard_magnetization(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 1:
                    spins[i, j] = -1
        sim.set_spins(spins)
        assert sim.magnetization() == pytest.approx(0.0)

    def test_magnetization_range(self, small_sim: IsingSimulation) -> None:
        """Random initial state has |m| <= 1."""
        m = small_sim.magnetization()
        assert -1.0 <= m <= 1.0

    def test_single_flip_changes_magnetization(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)
        spins = np.ones((4, 4), dtype=np.int8)
        sim.set_spins(spins)
        assert sim.magnetization() == pytest.approx(1.0)

        sim.flip_spin(0)
        # One spin flipped: m = (16 - 2) / 16 = 14/16 = 0.875
        assert sim.magnetization() == pytest.approx(0.875)


def _staggered_reference(configurations: np.ndarray) -> np.ndarray:
    """NumPy definition: column k is a bitmask over the axes after the sample axis."""
    n_axes = configurations.ndim - 1
    spins = configurations.astype(np.float64)
    columns = []
    for k in range(1 << n_axes):
        signed = spins
        for axis in range(n_axes):
            if k & (1 << axis):
                extent = configurations.shape[axis + 1]
                sign = (-1.0) ** np.arange(extent)
                shape = [1] * configurations.ndim
                shape[axis + 1] = extent
                signed = signed * sign.reshape(shape)
        columns.append(signed.mean(axis=tuple(range(1, configurations.ndim))))
    return np.stack(columns, axis=1)


class TestStaggeredMagnetization:
    """The bitmask-indexed staggered magnetizations of the Rust core."""

    def _square(self) -> IsingSimulation:
        return IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42)

    def test_all_up_is_uniform_only(self) -> None:
        sim = self._square()
        sim.set_spins(np.ones((4, 4), dtype=np.int8))
        np.testing.assert_array_equal(
            sim.staggered_magnetization(), [1.0, 0.0, 0.0, 0.0]
        )

    def test_row_and_column_stripes_and_neel(self) -> None:
        sim = self._square()
        rows, cols = np.indices((4, 4))
        cases = {
            1: (-1) ** rows,  # alternates from row to row
            2: (-1) ** cols,  # alternates from column to column
            3: (-1) ** (rows + cols),  # Néel
        }
        for column, pattern in cases.items():
            sim.set_spins(pattern.astype(np.int8))
            m = sim.staggered_magnetization()
            expected = np.zeros(4)
            expected[column] = 1.0
            np.testing.assert_array_equal(m, expected)

    def test_component_zero_is_the_magnetization(
        self, small_sim: IsingSimulation
    ) -> None:
        m = small_sim.staggered_magnetization()
        assert m[0] == small_sim.magnetization()

    @pytest.mark.parametrize("lattice_type", list(LatticeType))
    def test_length_and_reference_on_every_lattice(
        self, lattice_type: LatticeType
    ) -> None:
        size = 4
        lattice = LatticeConfig(lattice_type, size)
        sim = IsingSimulation(
            size, 1.0, 0.0, 0.0, 0.0, 3, "metropolis", lattice_type.value
        )
        sim.sweep(5, temperature=2.5)
        m = sim.staggered_magnetization()
        assert m.shape == (2 ** len(lattice.shape),)
        reference = _staggered_reference(sim.get_spins()[None, ...])[0]
        np.testing.assert_allclose(m, reference, atol=1e-12)

    def test_honeycomb_sublattice_bit_is_neel(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "honeycomb")
        spins = np.ones((4, 4, 2), dtype=np.int8)
        spins[:, :, 1] = -1
        sim.set_spins(spins)
        m = sim.staggered_magnetization()
        assert m.shape == (8,)
        assert m[4] == pytest.approx(1.0)
        assert m[0] == pytest.approx(0.0)

    def test_cubic_layered_components(self) -> None:
        sim = IsingSimulation(4, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "cubic")
        x, y, z = np.indices((4, 4, 4))
        for column, pattern in {1: (-1) ** x, 2: (-1) ** y, 4: (-1) ** z}.items():
            sim.set_spins(pattern.astype(np.int8))
            m = sim.staggered_magnetization()
            expected = np.zeros(8)
            expected[column] = 1.0
            np.testing.assert_array_equal(m, expected)


class TestStaggeredMagnetizationResults:
    """Recorded at every measurement by every run path, saved and loaded."""

    @staticmethod
    def _config(lattice_type: LatticeType, mode: ExecutionMode) -> SimulationConfig:
        return SimulationConfig(
            lattice=LatticeConfig(lattice_type, 4),
            temperatures=(3.0, 2.0),
            n_sweeps=40,
            measurement_interval=10,
            mode=mode,
            store_configs=True,
        )

    @pytest.mark.parametrize("mode", list(ExecutionMode))
    @pytest.mark.parametrize("lattice_type", list(LatticeType))
    def test_matches_reference_on_stored_configurations(
        self, lattice_type: LatticeType, mode: ExecutionMode
    ) -> None:
        config = self._config(lattice_type, mode)
        results = Simulation(config).run(show_progress=False)
        n_components = 2 ** len(config.lattice.shape)
        for temp in results.temperatures:
            m = results.staggered_magnetization[temp]
            assert m.shape == (4, n_components)
            # Column 0 is the magnetization series, bit for bit.
            np.testing.assert_array_equal(m[:, 0], results.magnetization[temp])
            reference = _staggered_reference(results.configurations[temp])
            np.testing.assert_allclose(m, reference, atol=1e-12)

    def test_stripe_example_reference_agrees(self) -> None:
        # The example's NumPy stripe order parameter equals the maximum of
        # |columns 1 and 2| of the recorded observable.
        script = (
            Path(__file__).resolve().parents[1] / "examples" / "stripe_phase_diagram.py"
        )
        spec = importlib.util.spec_from_file_location("stripe_phase_diagram", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        config = SimulationConfig(
            lattice=LatticeConfig(size=8, j1=1.0, j2=-0.6),
            temperatures=(2.0, 0.6),
            n_sweeps=100,
            n_thermalization=50,
            measurement_interval=10,
            store_configs=True,
        )
        results = Simulation(config).run(show_progress=False)
        for temp in results.temperatures:
            from_configs = module.stripe_order_parameter(results.configurations[temp])
            recorded = np.abs(results.staggered_magnetization[temp][:, 1:3]).max(axis=1)
            np.testing.assert_allclose(recorded, from_configs, atol=1e-12)
        # Deep in the stripe phase the recorded order parameter is large.
        assert (
            np.abs(results.staggered_magnetization[0.6][:, 1:3]).max(axis=1).mean()
            > 0.8
        )
