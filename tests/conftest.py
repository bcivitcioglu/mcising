"""Shared fixtures for mcising tests."""

from __future__ import annotations

import numpy as np
import pytest
from mcising._core import IsingSimulation
from mcising.config import LatticeConfig, SimulationConfig


@pytest.fixture
def small_sim() -> IsingSimulation:
    """A small 4x4 Rust simulation for fast unit tests."""
    return IsingSimulation(4, 1.0, 0.0, 0.0, 42)


@pytest.fixture
def sim_8x8() -> IsingSimulation:
    """An 8x8 Rust simulation for physics tests."""
    return IsingSimulation(8, 1.0, 0.0, 0.0, 42)


@pytest.fixture
def default_config() -> SimulationConfig:
    """Default simulation config for integration tests."""
    return SimulationConfig(
        lattice=LatticeConfig(size=8, j1=1.0),
        temperatures=(3.0, 2.269, 1.5),
        n_sweeps=100,
        measurement_interval=10,
    )


@pytest.fixture
def all_up_spins_4x4() -> np.ndarray:
    """All-up spin configuration for a 4x4 lattice."""
    return np.ones((4, 4), dtype=np.int8)


@pytest.fixture
def checkerboard_spins_4x4() -> np.ndarray:
    """Checkerboard spin configuration for a 4x4 lattice."""
    spins = np.ones((4, 4), dtype=np.int8)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 1:
                spins[i, j] = -1
    return spins
