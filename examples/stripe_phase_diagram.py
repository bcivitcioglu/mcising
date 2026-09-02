#!/usr/bin/env python3
"""Phase diagram of the frustrated J1-J2 Ising model on the square lattice.

Hamiltonian (mcising convention): H = -J1 sum_<ij> s_i s_j - J2 sum_<<ij>> s_i s_j
with J1 = 1 ferromagnetic and J2 <= 0 antiferromagnetic on the diagonals.
The two couplings compete: an aligned pair of diagonal neighbours costs
|J2|, while the four nearest-neighbour bonds want everything aligned.

Ground states (energy per site, periodic lattice):

    ferromagnet   e_FM     = -2 J1 - 2 J2    (all bonds satisfied only for J2 > 0)
    stripe        e_stripe =  2 J2           (rows alternate: nn bonds cancel,
                                              every diagonal bond satisfied)

They cross at J2 = -J1 / 2. For -1/2 < J2/J1 < 0 the low-temperature phase
is ferromagnetic with a transition temperature that falls from Onsager's
2.269 towards zero; for J2/J1 < -1/2 it is the "superantiferromagnetic"
stripe phase, whose order parameter is the staggered magnetization along
one lattice axis,

    m_s = max( |1/N sum_i (-1)^{x_i} s_i| , |1/N sum_i (-1)^{y_i} s_i| ).

Right at J2 = -1/2 the ground state is macroscopically degenerate and
order sets in only at T = 0. (The stripe transition is weakly first order
for 1/2 < |J2|/J1 < ~0.67 and continuous with varying, Ashkin-Teller-like
exponents beyond: Kalz, Honecker & Moliner, Phys. Rev. B 84, 174407 (2011);
Jin, Sen & Sandvik, Phys. Rev. Lett. 108, 045702 (2012).)

The script scans a grid of J2 values, and for each runs a Metropolis
cool-down ladder from high to low temperature (the previous temperature's
final state seeds the next, so the lattice is annealed rather than
quenched into a domain-wall state), storing configurations to evaluate
m_s. Two heatmaps result: <|m|> and <m_s> over the (J2, T) plane, with the
specific-heat peak at each J2 overlaid as a finite-size estimate of the
transition line. Cluster algorithms cannot be used here: Wolff and
Swendsen-Wang are only correct for a single ferromagnetic coupling.

Caveats: this is a single lattice size with a fixed sweep budget per
temperature, meant as a map rather than a finite-size-scaling study.
Close to J2 = -1/2 the two ground states are nearly degenerate and
Metropolis equilibrates slowly at low T, so that column is the least
converged; parallel tempering (``ExecutionMode.PARALLEL_TEMPERING``) is
the tool for a quantitative study there.

Usage:
    python examples/stripe_phase_diagram.py                 # full budget
    python examples/stripe_phase_diagram.py --out figures/  # choose the directory
    python examples/stripe_phase_diagram.py --quick         # seconds, smoke test

Runtime (Apple M4, release build): measured wall time is printed at the end.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from mcising import (
    Algorithm,
    ExecutionMode,
    LatticeConfig,
    Simulation,
    SimulationConfig,
    SimulationResults,
)
from mcising.constants import TC_SQUARE_2D
from numpy.typing import NDArray

FULL_SIZE: Final = 32
FULL_J2: Final[tuple[float, ...]] = tuple(
    float(j) for j in np.round(np.linspace(-1.0, 0.0, 11), 2)
)
FULL_TEMPERATURES: Final[tuple[float, ...]] = tuple(
    float(t) for t in np.round(np.arange(3.0, 0.19, -0.2), 2)
)
QUICK_SIZE: Final = 8
QUICK_J2: Final[tuple[float, ...]] = (-1.0, -0.5, 0.0)
QUICK_TEMPERATURES: Final[tuple[float, ...]] = (3.0, 2.0, 1.0, 0.5)

FloatArray = NDArray[np.float64]


def stripe_order_parameter(configurations: NDArray[np.int8]) -> FloatArray:
    """Per-snapshot stripe order parameter of (n, L, L) square-lattice configurations.

    m_s = max(|<(-1)^x s>|, |<(-1)^y s>|): the staggered magnetization along
    whichever axis the stripes run. Both orientations are degenerate ground
    states, so the maximum makes the order parameter orientation-blind.
    """
    n, rows, cols = configurations.shape
    sign_rows = (-1.0) ** np.arange(rows)
    sign_cols = (-1.0) ** np.arange(cols)
    spins = configurations.astype(np.float64)
    along_rows = np.abs((spins * sign_rows[None, :, None]).mean(axis=(1, 2)))
    along_cols = np.abs((spins * sign_cols[None, None, :]).mean(axis=(1, 2)))
    return np.maximum(along_rows, along_cols)


def run(
    size: int,
    j2: float,
    temperatures: Sequence[float],
    *,
    n_sweeps: int,
    n_thermalization: int,
    seed: int,
) -> SimulationResults:
    """Metropolis cool-down ladder (descending temperatures) at one J2."""
    config = SimulationConfig(
        lattice=LatticeConfig(size=size, j1=1.0, j2=j2),
        algorithm=Algorithm.METROPOLIS,
        mode=ExecutionMode.COOLDOWN,
        temperatures=tuple(temperatures),
        n_sweeps=n_sweeps,
        n_thermalization=n_thermalization,
        measurement_interval=10,
        store_configs=True,
        seed=seed,
    )
    return Simulation(config).run(show_progress=False)


def scan(
    size: int,
    j2_values: Sequence[float],
    temperatures: Sequence[float],
    *,
    n_sweeps: int,
    n_thermalization: int,
    seed: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """<|m|>, <m_s> and the Cv-peak temperature over the (J2, T) grid.

    Returns arrays of shape (n_T, n_J2) for the two order parameters and
    (n_J2,) for the specific-heat peak; rows follow ``temperatures``.
    """
    magnetization = np.zeros((len(temperatures), len(j2_values)))
    stripe = np.zeros_like(magnetization)
    cv_peak = np.zeros(len(j2_values))
    for column, j2 in enumerate(j2_values):
        results = run(
            size,
            j2,
            temperatures,
            n_sweeps=n_sweeps,
            n_thermalization=n_thermalization,
            seed=seed + column,
        )
        specific_heat = np.zeros(len(temperatures))
        for row, t in enumerate(temperatures):
            magnetization[row, column] = np.abs(results.magnetization[t]).mean()
            stripe[row, column] = stripe_order_parameter(
                results.configurations[t]
            ).mean()
            specific_heat[row] = results.specific_heat(t)
        cv_peak[column] = temperatures[int(np.argmax(specific_heat))]
        print(
            f"J2 = {j2:5.2f}: Cv peak at T = {cv_peak[column]:.2f}, "
            f"at T = {temperatures[-1]:.2f} <|m|> = {magnetization[-1, column]:.3f}, "
            f"<m_s> = {stripe[-1, column]:.3f}"
        )
    return magnetization, stripe, cv_peak


def make_figure(
    j2_values: Sequence[float],
    temperatures: Sequence[float],
    magnetization: FloatArray,
    stripe: FloatArray,
    cv_peak: FloatArray,
    output: Path,
    *,
    size: int,
    budget: str,
) -> Path:
    """Two heatmaps over the (J2, T) plane with the Cv-peak line overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j2 = np.asarray(j2_values)
    temps = np.asarray(temperatures)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    panels = (
        (axes[0], magnetization, "ferromagnetic order $\\langle |m| \\rangle$"),
        (axes[1], stripe, "stripe order $\\langle m_s \\rangle$"),
    )
    for ax, values, title in panels:
        mesh = ax.pcolormesh(
            j2, temps, values, shading="nearest", cmap="viridis", vmin=0.0, vmax=1.0
        )
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        ax.plot(j2, cv_peak, "w.-", lw=1, ms=5, label="$C_V$ peak (finite $L$)")
        ax.axvline(-0.5, color="white", ls=":", lw=1)
        ax.plot(
            [0.0],
            [TC_SQUARE_2D],
            marker="*",
            color="C3",
            ms=10,
            ls="none",
            label="exact $T_c$ at $J_2 = 0$",
        )
        ax.set_xlabel("$J_2 / J_1$")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("$T$  ($J_1 = k_B = 1$)")
    for ax in axes:
        ax.text(
            -0.75, temps.max() * 0.93, "stripe", color="white", ha="center", fontsize=8
        )
        ax.text(
            -0.25,
            temps.max() * 0.93,
            "ferromagnet",
            color="white",
            ha="center",
            fontsize=8,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle(
        f"$J_1$-$J_2$ Ising model, square lattice $L={size}$, "
        f"Metropolis cool-down ({budget})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path.cwd(), help="output directory")
    parser.add_argument("--quick", action="store_true", help="tiny budget (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.quick:
        size, j2_values, temps = QUICK_SIZE, QUICK_J2, QUICK_TEMPERATURES
        n_sweeps, n_therm, budget = 500, 100, "quick"
    else:
        size, j2_values, temps = FULL_SIZE, FULL_J2, FULL_TEMPERATURES
        n_sweeps, n_therm, budget = 16_000, 4_000, "1 600 samples/T"

    started = time.perf_counter()
    magnetization, stripe, cv_peak = scan(
        size,
        j2_values,
        temps,
        n_sweeps=n_sweeps,
        n_thermalization=n_therm,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - started
    figure = make_figure(
        j2_values,
        temps,
        magnetization,
        stripe,
        cv_peak,
        args.out / "stripe_phase_diagram.png",
        size=size,
        budget=budget,
    )
    print(f"wrote {figure}  (simulation {elapsed:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
