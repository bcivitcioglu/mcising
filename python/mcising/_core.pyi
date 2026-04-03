"""Type stubs for the Rust _core extension module."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

class IsingSimulation:
    """Core Ising model simulation engine (Rust/PyO3)."""

    lattice_size: int
    num_sites: int
    j1: float
    j2: float
    j3: float
    h: float

    algorithm_name: str

    def __init__(
        self,
        lattice_size: int,
        j1: float,
        j2: float,
        j3: float,
        h: float,
        seed: int,
        algorithm: str = "metropolis",
    ) -> None: ...
    def sweep(self, n_sweeps: int, beta: float) -> tuple[int, int]: ...
    def energy(self) -> float: ...
    def magnetization(self) -> float: ...
    def get_spins(self) -> NDArray[np.int8]: ...
    def set_spins(self, spins: NDArray[np.int8]) -> None: ...
    def flip_spin(self, row: int, col: int) -> None: ...
    def spin_energy(self, row: int, col: int) -> float: ...
    def correlation_function(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def correlation_length(self) -> float: ...
    def thermalize_with_diagnostics(
        self, temp_schedule: list[float]
    ) -> NDArray[np.float64]: ...
    def extend_thermalization(
        self, n_sweeps: int, beta: float
    ) -> NDArray[np.float64]: ...
    @staticmethod
    def analyze_thermalization_series(
        series: NDArray[np.float64],
        c_window: float,
        tau_multiplier: float,
    ) -> dict[str, Any]: ...
    def production_sweeps(
        self,
        n_measurements: int,
        interval: int,
        beta: float,
        store_configs: bool,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int8] | None,
    ]: ...
    def get_rng_state(self) -> list[int]: ...
    def set_rng_state(self, state: list[int]) -> None: ...
    def __repr__(self) -> str: ...
