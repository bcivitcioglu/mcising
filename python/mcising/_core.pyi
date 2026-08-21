"""Type stubs for the Rust _core extension module."""

from typing import Any, final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "IsingSimulation",
    "run_independent_temperatures",
    "run_parallel_tempering",
]

# @final subsumes @disjoint_base (stubtest rejects the combination).
@final
class IsingSimulation:
    """Core Ising model simulation engine (Rust/PyO3)."""

    def __new__(
        cls,
        lattice_size: int,
        j1: float,
        j2: float,
        j3: float,
        h: float,
        seed: int,
        algorithm: str = "metropolis",
        lattice_type: str = "square",
    ) -> IsingSimulation: ...

    # Read-only PyO3 getters.
    @property
    def lattice_size(self) -> int: ...
    @property
    def num_sites(self) -> int: ...
    @property
    def j1(self) -> float: ...
    @property
    def j2(self) -> float: ...
    @property
    def j3(self) -> float: ...
    @property
    def h(self) -> float: ...
    @property
    def algorithm_name(self) -> str: ...
    def sweep(
        self, n_sweeps: int = 1, *, temperature: float
    ) -> tuple[int, int, int]: ...
    def energy(self) -> float: ...
    def magnetization(self) -> float: ...
    def get_spins(self) -> NDArray[np.int8]: ...
    def set_spins(self, spins: NDArray[np.int8]) -> None: ...
    def flip_spin(self, site: int) -> None: ...
    def spin_energy(self, site: int) -> float: ...
    def correlation_function(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def correlation_length(self) -> float: ...
    def anneal(self, temp_schedule: list[float]) -> None: ...
    def extend_thermalization(
        self, n_sweeps: int, *, temperature: float
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
        *,
        temperature: float,
        store_configs: bool,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int8] | None,  # shape depends on lattice type
        int,  # total cluster flips (0 for Metropolis)
    ]: ...
    def get_rng_state(self) -> list[int]: ...
    def set_rng_state(self, state: list[int]) -> None: ...
    def __repr__(self) -> str: ...

# Both runners return one dict per temperature with keys "temperature",
# "energies", "magnetizations", "n_cluster_flips", plus "configurations"
# when store_configs, and "correlation_distances" /
# "correlation_function" / "correlation_length" when compute_correlation.
def run_parallel_tempering(
    lattice_size: int,
    j1: float,
    j2: float,
    j3: float,
    h: float,
    base_seed: int,
    algorithm: str,
    lattice_type: str,
    temperatures: list[float],
    n_thermalization: int,
    n_sweeps: int,
    measurement_interval: int,
    swap_interval: int = 1,
    store_configs: bool = False,
    compute_correlation: bool = False,
) -> list[dict[str, Any]]: ...
def run_independent_temperatures(
    lattice_size: int,
    j1: float,
    j2: float,
    j3: float,
    h: float,
    base_seed: int,
    algorithm: str,
    lattice_type: str,
    temperatures: list[float],
    n_thermalization: int,
    n_sweeps: int,
    measurement_interval: int,
    store_configs: bool = False,
    compute_correlation: bool = False,
    seed_offsets: list[int] | None = None,
) -> list[dict[str, Any]]: ...
