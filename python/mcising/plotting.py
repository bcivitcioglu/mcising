"""Matplotlib visualization utilities for Ising model simulations.

All plot functions accept either a ``SimulationResults`` object or a
path to an HDF5 file (str or Path). Multiple paths can be passed as a
list to overlay results from different coupling configurations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from mcising.simulation import SimulationResults

__all__: Final[list[str]] = [
    "plot_energy",
    "plot_magnetization",
    "plot_specific_heat",
    "plot_susceptibility",
    "plot_lattice",
    "plot_correlation",
    "plot_energy_timeseries",
    "plot_magnetization_histogram",
    "export_lattices",
    # Legacy alias
    "plot_observables",
]

# Type alias: accepts Results object, file path, or list of either
ResultsInput = (
    SimulationResults
    | str
    | Path
    | list[SimulationResults | str | Path]
)


def _load_if_needed(
    source: SimulationResults | str | Path,
) -> SimulationResults:
    """Load from HDF5 if given a path, otherwise pass through."""
    if isinstance(source, SimulationResults):
        return source
    from mcising.io import load_hdf5

    return load_hdf5(source)


def _to_list(
    source: ResultsInput,
) -> list[SimulationResults]:
    """Normalize input to a list of SimulationResults."""
    if isinstance(source, list):
        return [_load_if_needed(s) for s in source]
    return [_load_if_needed(source)]


def _label_for(results: SimulationResults) -> str:
    """Generate a legend label from the config metadata."""
    config = results.metadata.get("config")
    if config is not None and hasattr(config, "lattice"):
        lc = config.lattice
        parts = []
        if lc.j1 != 0:
            parts.append(f"J1={lc.j1}")
        if lc.j2 != 0:
            parts.append(f"J2={lc.j2}")
        if lc.j3 != 0:
            parts.append(f"J3={lc.j3}")
        if lc.h != 0:
            parts.append(f"h={lc.h}")
        if not parts:
            parts.append("J1=0")
        lt = lc.lattice_type.value
        return f"{lt} {', '.join(parts)}"
    return ""


# ═══════════════════════════════════════════════════════════════════
# Single-quantity vs temperature plots
# ═══════════════════════════════════════════════════════════════════


def plot_energy(
    source: ResultsInput,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot energy per site vs temperature.

    Parameters
    ----------
    source : SimulationResults, path, or list of either
        One or more result sets to plot. Lists produce overlaid curves.
    ax : Axes, optional
        Matplotlib axes. Creates a new figure if None.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    return _plot_quantity(source, "energy", "<E>/N", "Energy per site", ax=ax)


def plot_magnetization(
    source: ResultsInput,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot |magnetization| per site vs temperature.

    Parameters
    ----------
    source : SimulationResults, path, or list of either
        One or more result sets to plot.
    ax : Axes, optional
        Matplotlib axes.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    return _plot_quantity(
        source, "magnetization", "<|M|>/N", "|Magnetization| per site", ax=ax
    )


def plot_specific_heat(
    source: ResultsInput,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot specific heat per site vs temperature.

    Cv = N * Var(E) / T^2.

    Parameters
    ----------
    source : SimulationResults, path, or list of either
        One or more result sets to plot.
    ax : Axes, optional
        Matplotlib axes.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    return _plot_quantity(
        source, "specific_heat", "Cv/N", "Specific heat per site", ax=ax
    )


def plot_susceptibility(
    source: ResultsInput,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot magnetic susceptibility per site vs temperature.

    chi = N * Var(M) / T.

    Parameters
    ----------
    source : SimulationResults, path, or list of either
        One or more result sets to plot.
    ax : Axes, optional
        Matplotlib axes.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    return _plot_quantity(
        source, "susceptibility", r"$\chi$/N",
        "Susceptibility per site", ax=ax,
    )


def _plot_quantity(
    source: ResultsInput,
    quantity: str,
    ylabel: str,
    title: str,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Internal: plot a single quantity vs temperature."""
    results_list = _to_list(source)
    multi = len(results_list) > 1

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = cast(Figure, ax.get_figure())

    for results in results_list:
        temps = sorted(results.temperatures)
        vals: list[float] = []
        errs: list[float] = []

        for t in temps:
            if quantity == "energy" and t in results.energy:
                vals.append(float(np.mean(results.energy[t])))
                errs.append(float(np.std(results.energy[t])))
            elif quantity == "magnetization" and t in results.magnetization:
                vals.append(
                    float(np.mean(np.abs(results.magnetization[t])))
                )
                errs.append(
                    float(np.std(np.abs(results.magnetization[t])))
                )
            elif quantity == "specific_heat" and t in results.energy:
                vals.append(results.specific_heat(t))
                errs.append(0.0)
            elif quantity == "susceptibility" and t in results.magnetization:
                vals.append(results.susceptibility(t))
                errs.append(0.0)

        label = _label_for(results) if multi else None
        if any(e > 0 for e in errs):
            ax.errorbar(
                temps, vals, yerr=errs, fmt="o-", capsize=3, label=label
            )
        else:
            ax.plot(temps, vals, "o-", label=label)

    ax.set_xlabel("Temperature")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if multi:
        ax.legend()

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# Lattice visualization
# ═══════════════════════════════════════════════════════════════════


def _render_spins(ax: Axes, spins: NDArray[np.int8]) -> None:
    """Render a spin array onto an axes (handles 1D/2D/3D shapes)."""
    plot_data = spins
    if spins.ndim == 3:
        plot_data = spins.reshape(spins.shape[0], -1)
    elif spins.ndim == 1:
        plot_data = spins.reshape(1, -1)
    ax.imshow(
        plot_data, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest"
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_lattice(
    source: SimulationResults | str | Path | NDArray[np.int8],
    *,
    temperature: float | None = None,
    n: int | None = None,
    max_cols: int = 10,
    max_panels: int = 100,
) -> Figure:
    """Plot spin configuration(s) as heatmap(s).

    With no ``n``: shows all configurations at the given temperature
    side by side in a grid. With ``n``: shows a single configuration.

    Parameters
    ----------
    source : SimulationResults, path, or NDArray
        Results object, HDF5 path, or a raw spin array.
    temperature : float, optional
        Temperature to plot (required if source is Results/path).
    n : int, optional
        Specific configuration index (0-based). If None, shows all
        configurations at that temperature in a grid.
    max_cols : int
        Maximum columns in the grid (when showing all configs).
    max_panels : int
        Maximum panels before auto-subsampling with a warning.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If ``n`` is out of range for the available configurations.
    """
    # Raw array: just plot it
    if isinstance(source, np.ndarray):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        _render_spins(ax, source)
        return fig

    results = _load_if_needed(source)

    if temperature is None:
        temperature = sorted(results.temperatures)[
            len(results.temperatures) // 2
        ]
    if temperature not in results.configurations:
        msg = f"No configurations stored for T={temperature}"
        raise ValueError(msg)

    configs = results.configurations[temperature]
    n_configs = len(configs)

    # Single config mode
    if n is not None:
        if n < 0 or n >= n_configs:
            msg = (
                f"Config index {n} out of range. "
                f"Temperature T={temperature} has {n_configs} "
                f"configurations (n=0..{n_configs - 1})."
            )
            raise ValueError(msg)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        _render_spins(ax, configs[n])
        ax.set_title(f"T={temperature:.4f}, config {n}")
        return fig

    # All configs mode
    show_configs = list(range(n_configs))
    if n_configs > max_panels:
        indices = np.linspace(0, n_configs - 1, max_panels, dtype=int)
        show_configs = [int(i) for i in indices]

    n_show = len(show_configs)
    ncols = min(n_show, max_cols)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.5 * ncols, 2.5 * nrows),
        squeeze=False,
    )

    for idx, cfg_idx in enumerate(show_configs):
        row, col = divmod(idx, ncols)
        _render_spins(axes[row][col], configs[cfg_idx])
        axes[row][col].set_title(f"{cfg_idx}", fontsize=8)

    for idx in range(n_show, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"T={temperature:.4f}  ({n_configs} configurations)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def export_lattices(
    source: SimulationResults | str | Path,
    output: str | Path,
    *,
    flat: bool = False,
    temperatures: list[float] | None = None,
    dpi: int = 100,
) -> int:
    """Export lattice configurations as PNG images in a zip file.

    Each spin configuration is rendered as a heatmap and saved with
    a descriptive filename encoding lattice type, couplings, algorithm,
    temperature, and configuration index.

    Parameters
    ----------
    source : SimulationResults or path
        Results with stored configurations.
    output : str or Path
        Output zip file path.
    flat : bool
        If False (default), organize in tree: folder per temperature.
        If True, all PNGs in one flat folder.
    temperatures : list[float], optional
        Specific temperatures to export. If None, exports all.
    dpi : int
        Resolution of exported images.

    Returns
    -------
    int
        Number of images exported.
    """
    import io
    import zipfile

    results = _load_if_needed(source)
    output = Path(output)

    # Build descriptive prefix from metadata
    prefix = _export_prefix(results)

    all_temps = sorted(
        t for t in results.temperatures if t in results.configurations
    )
    if temperatures is not None:
        all_temps = [t for t in all_temps if t in temperatures]

    total_images = sum(
        len(results.configurations[t]) for t in all_temps
    )

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    count = 0
    with (
        zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf,
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress,
    ):
        task = progress.add_task(
            f"Exporting {total_images} images", total=total_images
        )
        for temp in all_temps:
            configs = results.configurations[temp]
            for cfg_idx in range(len(configs)):
                # Render to in-memory PNG
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                _render_spins(ax, configs[cfg_idx])
                ax.set_title(
                    f"T={temp:.4f} #{cfg_idx + 1}", fontsize=8
                )
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=dpi)
                plt.close(fig)
                buf.seek(0)

                # Build path inside zip
                t_str = f"T={temp:.4f}"
                cfg_str = f"config_{cfg_idx + 1:03d}.png"

                if flat:
                    arcname = (
                        f"{prefix}/"
                        f"{prefix}_{t_str}_{cfg_str}"
                    )
                else:
                    arcname = (
                        f"{prefix}/{t_str}/{cfg_str}"
                    )

                zf.writestr(arcname, buf.getvalue())
                count += 1
                progress.update(task, advance=1)

    return count


def _export_prefix(results: SimulationResults) -> str:
    """Build a descriptive folder/file prefix from config metadata."""
    config = results.metadata.get("config")
    if config is None or not hasattr(config, "lattice"):
        return "mcising"

    lc = getattr(config, "lattice")
    lt = lc.lattice_type.value

    # Size description
    if lt == "cubic":
        size_str = f"{lc.size}x{lc.size}x{lc.size}"
    elif lt == "chain":
        size_str = str(lc.size)
    else:
        size_str = f"{lc.size}x{lc.size}"

    # Couplings (only non-zero)
    coupling_parts = []
    if lc.j1 != 0:
        coupling_parts.append(f"J1={lc.j1}")
    if lc.j2 != 0:
        coupling_parts.append(f"J2={lc.j2}")
    if lc.j3 != 0:
        coupling_parts.append(f"J3={lc.j3}")
    if lc.h != 0:
        coupling_parts.append(f"h={lc.h}")
    if not coupling_parts:
        coupling_parts.append("J1=0")
    couplings = "_".join(coupling_parts)

    algo = getattr(config, "algorithm", None)
    algo_str = algo.value if algo is not None else "unknown"

    return f"{lt}_{size_str}_{couplings}_{algo_str}"


# ═══════════════════════════════════════════════════════════════════
# Diagnostic plots
# ═══════════════════════════════════════════════════════════════════


def plot_energy_timeseries(
    source: SimulationResults | str | Path,
    temperature: float,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot the energy time series at a given temperature.

    Parameters
    ----------
    source : SimulationResults or path
        Results with energy data.
    temperature : float
        Temperature to plot.
    ax : Axes, optional
        Matplotlib axes.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    results = _load_if_needed(source)
    if temperature not in results.energy:
        msg = f"No energy data for T={temperature}"
        raise ValueError(msg)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    else:
        fig = cast(Figure, ax.get_figure())

    e = results.energy[temperature]
    ax.plot(e, linewidth=0.5, alpha=0.8)
    ax.axhline(
        y=float(np.mean(e)),
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"mean = {float(np.mean(e)):.4f}",
    )
    ax.set_xlabel("Measurement index")
    ax.set_ylabel("Energy per site")
    ax.set_title(f"Energy time series at T={temperature:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_magnetization_histogram(
    source: SimulationResults | str | Path,
    temperature: float,
    *,
    bins: int = 50,
    ax: Axes | None = None,
) -> Figure:
    """Plot the magnetization distribution at a given temperature.

    Below Tc the distribution is bimodal (peaks at +/- M0).
    Above Tc it is a single Gaussian peak near zero.

    Parameters
    ----------
    source : SimulationResults or path
        Results with magnetization data.
    temperature : float
        Temperature to plot.
    bins : int
        Number of histogram bins.
    ax : Axes, optional
        Matplotlib axes.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    results = _load_if_needed(source)
    if temperature not in results.magnetization:
        msg = f"No magnetization data for T={temperature}"
        raise ValueError(msg)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = cast(Figure, ax.get_figure())

    m = results.magnetization[temperature]
    ax.hist(m, bins=bins, density=True, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Magnetization per site")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Magnetization distribution at T={temperature:.3f}")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_correlation(
    source: SimulationResults | str | Path,
    temperature: float,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot the correlation function at a given temperature.

    Parameters
    ----------
    source : SimulationResults or path
        Results with correlation data.
    temperature : float
        Temperature to plot.
    ax : Axes, optional
        Matplotlib axes.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    results = _load_if_needed(source)
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


# ═══════════════════════════════════════════════════════════════════
# Legacy alias
# ═══════════════════════════════════════════════════════════════════


def plot_observables(
    results: SimulationResults,
    *,
    quantities: tuple[str, ...] = ("energy", "magnetization"),
) -> Figure:
    """Plot observables vs temperature (legacy interface).

    Parameters
    ----------
    results : SimulationResults
        Simulation results.
    quantities : tuple[str, ...]
        Quantities to plot: 'energy', 'magnetization',
        'specific_heat', 'susceptibility'.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    n_plots = len(quantities)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    quantity_map = {
        "energy": plot_energy,
        "magnetization": plot_magnetization,
        "specific_heat": plot_specific_heat,
        "susceptibility": plot_susceptibility,
    }

    for ax, quantity in zip(axes, quantities):
        fn = quantity_map.get(quantity)
        if fn is not None:
            fn(results, ax=ax)

    fig.tight_layout()
    return fig
