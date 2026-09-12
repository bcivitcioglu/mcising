"""Type stubs for the Rust ``_core`` extension module.

This stub is the API reference for the compiled classes and functions:
mkdocstrings renders it for ``IsingSimulation`` and editors show it, so
every public symbol carries a docstring. Keep them in step with the ``///``
comments in ``rust/src/simulation.rs``, ``rust/src/parallel.rs`` and
``rust/src/wang_landau/mod.rs`` (``tests/test_core_stub_docs.py`` enforces
their presence, ``stubtest`` their signatures).
"""

from typing import Any, final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "IsingSimulation",
    "run_independent_temperatures",
    "run_parallel_tempering",
    "run_wang_landau",
]

# @final subsumes @disjoint_base (stubtest rejects the combination).
@final
class IsingSimulation:
    """Core Ising model simulation engine (Rust/PyO3).

    One lattice, one coupling set, one update algorithm and one
    random-number stream. :class:`mcising.Simulation` drives it for the
    high-level API; use it directly for sweep-level control — custom
    temperature schedules, per-sweep observables, or exact state
    round-trips through :meth:`get_spins` / :meth:`get_rng_state`.

    The Hamiltonian is ``H = -J1 Σ s_i s_j - J2 Σ s_i s_j - J3 Σ s_i s_j
    - h Σ s_i`` over nearest, next-nearest and third-nearest neighbour
    pairs (each bond once); positive couplings are ferromagnetic.

    Constructor arguments (positional, in this order):

    - ``lattice_size`` — linear size ``L``: sites per side (the honeycomb
      has two sites per cell, the chain ``L`` sites in total).
    - ``j1``, ``j2``, ``j3`` — nearest-, next-nearest- and
      third-nearest-neighbour couplings; the cluster algorithms require
      ``j1 > 0`` and ``j2 = j3 = h = 0``.
    - ``h`` — external field.
    - ``seed`` — seed of the Xoshiro256** generator that draws the initial
      configuration and every update.
    - ``algorithm`` — ``"metropolis"`` (default), ``"wolff"`` or
      ``"swendsen_wang"``.
    - ``lattice_type`` — ``"square"`` (default), ``"triangular"``,
      ``"honeycomb"``, ``"cubic"`` or ``"chain"``; periodic boundaries in
      every direction.
    """

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
    ) -> IsingSimulation:
        """Construct a simulation; the parameters are documented on the class."""

    # Read-only PyO3 getters.
    @property
    def lattice_size(self) -> int:
        """Linear lattice size ``L`` passed at construction."""

    @property
    def num_sites(self) -> int:
        """Number of spins.

        ``L²`` (square, triangular), ``2L²`` (honeycomb), ``L³`` (cubic) or
        ``L`` (chain).
        """

    @property
    def j1(self) -> float:
        """Nearest-neighbour coupling ``J1``."""

    @property
    def j2(self) -> float:
        """Next-nearest-neighbour coupling ``J2``."""

    @property
    def j3(self) -> float:
        """Third-nearest-neighbour coupling ``J3``."""

    @property
    def h(self) -> float:
        """External field ``h``."""

    @property
    def algorithm_name(self) -> str:
        """``"metropolis"``, ``"wolff"`` or ``"swendsen_wang"``."""

    def sweep(self, n_sweeps: int = 1, *, temperature: float) -> tuple[int, int, int]:
        """Run ``n_sweeps`` Monte Carlo sweeps at ``temperature``.

        One Metropolis sweep attempts every site once in a sequential
        scan; one Wolff sweep grows and flips a single cluster; one
        Swendsen-Wang sweep rebuilds every bond and gives every cluster an
        independent flip decision.

        Parameters
        ----------
        n_sweeps : int
            Number of sweeps.
        temperature : float
            Temperature ``T`` (``k_B = 1``); must be positive and finite.

        Returns
        -------
        tuple[int, int, int]
            ``(accepted, attempted, cluster_flips)`` summed over the
            sweeps, with honest work accounting per algorithm. Metropolis:
            accepted single-spin flips, ``n_sweeps * num_sites`` attempts,
            no clusters. Wolff: ``accepted`` and ``attempted`` are both the
            total cluster size (the update is rejection-free) and
            ``cluster_flips`` is ``n_sweeps``. Swendsen-Wang: spins flipped,
            ``n_sweeps * num_sites`` decisions, and the number of clusters
            whose coin came up "flip".
        """

    def energy(self) -> float:
        """Energy per site of the current configuration, ``H / num_sites``."""

    def magnetization(self) -> float:
        """Signed magnetization per site, ``Σ s_i / num_sites``."""

    def staggered_magnetization(self) -> NDArray[np.float64]:
        """Staggered magnetizations per site, ``2 ** n_axes`` components.

        ``n_axes`` is the length of the lattice shape (``get_spins().ndim``:
        1 for the chain, 2 for the square and triangular lattices, 3 for
        the honeycomb and cubic lattices). Component ``k`` is a bitmask
        over the axes: ``(1/N) Σ_i (-1)^(Σ_{a in k} n_a(i)) s_i`` with
        ``n_a(i)`` the coordinate of site ``i`` along axis ``a``; component
        0 is :meth:`magnetization`, the others the stripe, layered and
        Néel order parameters (square: 1 and 2 alternate along rows and
        columns, 3 is Néel; honeycomb: 4 alternates between the
        sublattices).
        """

    def get_spins(self) -> NDArray[np.int8]:
        """Copy of the spin configuration as an ``int8`` array in the lattice's shape.

        ``(L, L)`` for the square and triangular lattices, ``(L, L, 2)``
        for the honeycomb, ``(L, L, L)`` for the cubic lattice and ``(L,)``
        for the chain.
        """

    def set_spins(self, spins: NDArray[np.int8]) -> None:
        """Replace the configuration.

        The array must hold exactly ``num_sites`` values, each ``+1`` or
        ``-1``; it is read in row-major order, so any shape with the right
        size is accepted. Raises ``ValueError`` otherwise.
        """

    def flip_spin(self, site: int) -> None:
        """Flip one spin, addressed by its flat row-major site index.

        Flat indexing is the one scheme every lattice shares (a
        ``(row, col)`` pair cannot address cubic or honeycomb sites).
        Raises ``ValueError`` for an index outside ``range(num_sites)``.
        """

    def spin_energy(self, site: int) -> float:
        """Local energy of one spin, by flat site index.

        ``-s_i (J1 Σ_nn s_j + J2 Σ_nnn s_j + J3 Σ_tnn s_j + h)`` — flipping
        the spin changes the total energy by ``-2 * spin_energy(site)``.
        """

    def correlation_function(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Spin-spin correlation ``C(r) = <s_0 s_r>`` of the current configuration.

        A full ``O(N²)`` pair sum binned by distance; returns
        ``(distances, correlations)`` as two float arrays.
        """

    def correlation_length(self) -> float:
        """Second-moment correlation length of the current configuration.

        Derived from the same ``C(r)`` as :meth:`correlation_function`.
        """

    def anneal(self, temp_schedule: list[float]) -> None:
        """Thermalization ramp: one sweep at each positive entry of ``temp_schedule``.

        Nothing is recorded; non-positive entries are skipped.
        """

    def extend_thermalization(
        self, n_sweeps: int, *, temperature: float
    ) -> NDArray[np.float64]:
        """Sweep ``n_sweeps`` times at ``temperature``, recording the energy.

        Returns the energy per site after every sweep — the series that
        adaptive thermalization analyses with
        :meth:`analyze_thermalization_series`.
        """

    @staticmethod
    def analyze_thermalization_series(
        series: NDArray[np.float64],
        c_window: float,
        tau_multiplier: float,
    ) -> dict[str, Any]:
        """Stationarity and autocorrelation analysis of an energy series.

        Parameters
        ----------
        series : NDArray[np.float64]
            Energy per site after each sweep.
        c_window : float
            Automatic-windowing constant for the integrated autocorrelation
            time (Sokal's ``c``).
        tau_multiplier : float
            Measurement interval recommended as ``tau_multiplier * tau_int``.

        Returns
        -------
        dict[str, Any]
            ``truncation_point`` (index where MSER stationarity begins),
            ``is_thermalized`` (the truncation point lies in the first
            half), ``tau_int`` (integrated autocorrelation time in sweeps,
            estimated on the stationary part), ``window`` (the windowing
            cutoff used) and ``recommended_interval`` (sweeps between
            measurements, at least 1).
        """

    # Returns the same per-temperature dict as the runners below.
    def production_sweeps(
        self,
        n_measurements: int,
        interval: int,
        *,
        temperature: float,
        store_configs: bool,
        compute_correlation: bool = False,
        correlation_interval: int = 1,
    ) -> dict[str, Any]:
        """Run a production block of ``n_measurements`` measurements.

        One FFI crossing for the whole block, with the GIL released while
        sweeping. Energy and magnetization are measured after each block of
        ``interval`` sweeps, a configuration snapshot is stored when
        ``store_configs`` and the correlation observables are evaluated at
        every ``correlation_interval``-th measurement when
        ``compute_correlation``. The random-number stream is consumed
        exactly as ``n_measurements`` separate ``sweep(interval)`` calls
        would consume it.

        Returns
        -------
        dict[str, Any]
            The per-temperature dict the parallel runners return:
            ``temperature``, ``energies``, ``magnetizations``,
            ``staggered_magnetizations`` (shape ``(n_measurements,
            2 ** n_axes)``, see :meth:`staggered_magnetization`),
            ``n_cluster_flips``, plus ``configurations`` when
            ``store_configs`` and ``correlation_distances`` /
            ``correlation_function`` / ``correlation_length`` when
            ``compute_correlation``. Raises ``ValueError`` for a
            non-positive temperature or a zero interval.
        """

    def get_rng_state(self) -> list[int]:
        """Serialized generator state (JSON bytes as ints).

        Lets a checkpoint continue the exact random-number stream through
        :meth:`set_rng_state`.
        """

    def set_rng_state(self, state: list[int]) -> None:
        """Restore a state returned by :meth:`get_rng_state`.

        Malformed input raises ``ValueError``.
        """

    def __repr__(self) -> str: ...

# Both runners produce one dict per temperature with keys "temperature",
# "energies", "magnetizations", "staggered_magnetizations" (2-D, one row
# per measurement), "n_cluster_flips", plus "configurations"
# when store_configs, and "correlation_distances" /
# "correlation_function" / "correlation_length" when compute_correlation
# (evaluated at every correlation_interval-th measurement). The
# parallel-tempering runner pairs that list with a ladder-level
# diagnostics dict.
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
    correlation_interval: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parallel-tempering runner behind ``ExecutionMode.PARALLEL_TEMPERING`` (internal).

    One replica per temperature, seeded ``base_seed + index``, advanced
    together on the Rayon pool with a replica-exchange attempt between
    adjacent temperatures every ``swap_interval`` sweeps (a separate
    swap-decision generator). After ``n_thermalization`` sweeps, measures
    every ``measurement_interval`` sweeps for ``n_sweeps`` sweeps;
    ``measurement_interval`` must be a multiple of ``swap_interval``.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        First, one dict per temperature in ascending temperature order,
        with ``temperature``, ``energies``, ``magnetizations``,
        ``staggered_magnetizations``, ``n_cluster_flips``, plus
        ``configurations`` when
        ``store_configs`` and the ``correlation_*`` arrays when
        ``compute_correlation``. Second, the ladder diagnostics:
        ``temperatures`` (ascending), ``swap_attempted`` and
        ``swap_accepted`` (one entry per adjacent pair, pair ``i``
        coupling rungs ``i`` and ``i + 1``) and ``round_trips`` (one
        entry per replica, coldest → hottest → coldest excursions
        completed during production). Raises ``ValueError`` for an
        unknown algorithm or lattice, an empty or non-positive
        temperature list, a zero interval, or a ``swap_interval`` that
        does not divide ``measurement_interval``.
    """

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
    correlation_interval: int = 1,
) -> list[dict[str, Any]]:
    """Independent-temperatures runner behind ``ExecutionMode.INDEPENDENT`` (internal).

    Every temperature is simulated from a fresh random configuration on
    its own Rayon task, seeded ``base_seed + offset`` where the offsets
    default to the temperature's position in the scan (``seed_offsets``
    lets a resumed checkpoint keep its original streams). After
    ``n_thermalization`` sweeps, measures every ``measurement_interval``
    sweeps for ``n_sweeps`` sweeps.

    Returns
    -------
    list[dict[str, Any]]
        One dict per temperature, in the order given, with the same keys
        as :func:`run_parallel_tempering`. Raises ``ValueError`` for an
        unknown algorithm or lattice, an empty or non-positive temperature
        list, a zero ``measurement_interval``, or ``seed_offsets`` of the
        wrong length.
    """

def run_wang_landau(
    lattice_size: int,
    j1: float,
    j2: float,
    j3: float,
    h: float,
    base_seed: int,
    lattice_type: str,
    production_sweeps: int,
    measurement_interval: int,
    *,
    n_walkers: int = 1,
    production_thermalization: int = 0,
    store_configs: bool = False,
    bin_width: float | None = None,
    energy_window: tuple[float, float] | None = None,
    initial_log_g: NDArray[np.float64] | None = None,
    flatness: float = 0.8,
    log_f_initial: float = 1.0,
    log_f_final: float = 1e-6,
    check_interval: int = 1000,
    max_wl_sweeps: int | None = None,
    drive_beta: float = 1.0,
    drive_max_sweeps: int = 10000,
    n_windows: int = 1,
    walkers_per_window: int = 1,
    window_overlap: float = 0.75,
    exchange_interval: int = 100,
) -> dict[str, Any]:
    """Wang-Landau density of states plus a multicanonical run (internal).

    Behind :class:`mcising.WangLandauSimulation`. Stage 1 is the
    Wang-Landau random walk on the exact energy grid of the couplings
    (single-spin flips at random sites, ``ln g(E) += log_f`` at every
    visit, ``log_f`` halved at every flat histogram — ``min H >= flatness
    * mean H`` checked every ``check_interval`` sweeps — and the
    Belardinelli-Pereyra ``log_f = 1/t`` schedule once ``log_f`` drops
    below ``1/t``, with ``t`` the mean number of visits per bin of the
    window), stopped at ``log_f_final`` or after ``max_wl_sweeps``
    (``max_wl_sweeps=0`` keeps ``initial_log_g`` as given). Stage 2 runs
    ``n_walkers`` chains with the frozen weights ``1/g(E)`` on the Rayon
    pool, seeded ``base_seed + 1000 + k``, discarding
    ``production_thermalization`` sweeps and measuring every
    ``measurement_interval`` sweeps for ``production_sweeps`` sweeps.
    ``energy_window`` is per site, ``bin_width`` in total-energy units
    (required when the couplings are not exactly representable in
    binary), ``initial_log_g`` has one entry per energy bin (NaN =
    unknown). With ``n_windows > 1`` or ``walkers_per_window > 1`` the
    first stage is the replica-exchange Wang-Landau algorithm: the range
    is split into ``n_windows`` windows sharing ``window_overlap`` of
    their length, every window holds ``walkers_per_window`` walkers
    (seeded ``base_seed + 1 + index``) advanced in parallel, adjacent
    windows exchange configurations every ``exchange_interval`` sweeps
    (``check_interval`` must be a multiple of it), the walkers of a
    window average their estimates at every check, and the windows are
    joined at the end. The GIL is released while sampling.

    Returns
    -------
    dict[str, Any]
        ``energy_bins`` (per site), ``bin_width``, ``num_sites``,
        ``window_bins``, ``energy_window``, ``log_g`` (NaN where never
        visited), ``wl_histogram``, ``production_histogram``,
        ``wl_diagnostics`` (per-iteration ``iteration_log_f``,
        ``iteration_sweeps``, ``iteration_flatness``,
        ``iteration_visited_bins``; ``total_sweeps``,
        ``one_over_t_switch_sweep``, ``final_log_f``, ``converged``,
        ``accepted``, ``attempted``, ``drive_in_sweeps``,
        ``visited_bins``, ``n_windows``, ``walkers_per_window``,
        ``window_bins``, ``exchange_attempted``, ``exchange_accepted``,
        ``merge_bins``), ``final_spins``, ``final_rng_state``,
        ``walkers`` (one dict per walker: ``walker``, ``energies``,
        ``magnetizations``, ``staggered_magnetizations``, ``bin_index``,
        ``accepted``, ``attempted``, ``round_trips`` and
        ``configurations`` when stored) and ``production_diagnostics``
        (``n_walkers``, ``sweeps_per_walker``, ``thermalization_sweeps``,
        ``accepted``, ``attempted``, ``rejected_unvisited``,
        ``round_trips`` per walker, ``histogram_flatness``,
        ``edge_bins``). Raises ``ValueError`` for an unknown lattice, a
        non-finite coupling, a zero interval or walker count, a flatness
        outside ``(0, 1]``, a schedule that is not ``0 < log_f_final <
        log_f_initial``, non-dyadic couplings without ``bin_width``, an
        invalid or unreachable ``energy_window``, or a mis-sized
        ``initial_log_g``.
    """
