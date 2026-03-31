"""Matplotlib visualization utilities for Ising model simulations."""

from __future__ import annotations

from typing import Final, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from mcising.simulation import SimulationResults

__all__: Final[list[str]] = [
    "plot_lattice",
    "plot_observables",
    "plot_correlation",
]


def plot_lattice(
    spins: NDArray[np.int8],
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Figure:
    """Plot a spin configuration as a heatmap.

    Parameters
    ----------
    spins : NDArray[np.int8]
        2D array of spin values (+1/-1).
    ax : Axes, optional
        Matplotlib axes to plot on. Creates a new figure if None.
    title : str, optional
        Plot title.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = cast(Figure, ax.get_figure())

    ax.imshow(spins, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal")

    return fig


def plot_observables(
    results: SimulationResults,
    *,
    quantities: tuple[str, ...] = ("energy", "magnetization"),
) -> Figure:
    """Plot observables vs temperature.

    Parameters
    ----------
    results : SimulationResults
        Simulation results containing measurements.
    quantities : tuple[str, ...]
        Which quantities to plot. Options: 'energy', 'magnetization'.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    n_plots = len(quantities)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    temps = sorted(results.temperatures)

    for ax, quantity in zip(axes, quantities):
        means: list[float] = []
        stds: list[float] = []

        for temp in temps:
            if quantity == "energy" and temp in results.energy:
                data = results.energy[temp]
                means.append(float(np.mean(data)))
                stds.append(float(np.std(data)))
            elif quantity == "magnetization" and temp in results.magnetization:
                data = np.abs(results.magnetization[temp])
                means.append(float(np.mean(data)))
                stds.append(float(np.std(data)))

        ax.errorbar(temps, means, yerr=stds, fmt="o-", capsize=3)
        ax.set_xlabel("Temperature")

        if quantity == "energy":
            ax.set_ylabel("Energy per site")
            ax.set_title("Energy vs Temperature")
        elif quantity == "magnetization":
            ax.set_ylabel("|Magnetization| per site")
            ax.set_title("|Magnetization| vs Temperature")

        ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_correlation(
    results: SimulationResults,
    temperature: float,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot the correlation function at a given temperature.

    Parameters
    ----------
    results : SimulationResults
        Simulation results with correlation data.
    temperature : float
        Temperature to plot the correlation function for.
    ax : Axes, optional
        Matplotlib axes. Creates a new figure if None.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if (
        results.correlation_function is None
        or temperature not in results.correlation_function
    ):
        msg = f"No correlation data available for T={temperature}"
        raise ValueError(msg)

    distances, correlations = results.correlation_function[temperature]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = cast(Figure, ax.get_figure())

    ax.plot(distances, correlations, "o-", markersize=4)
    ax.set_xlabel("Distance (lattice units)")
    ax.set_ylabel("C(r)")
    ax.set_title(f"Correlation Function at T={temperature:.3f}")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig
