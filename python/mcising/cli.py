"""Typer CLI for mcising Monte Carlo simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import mcising
from mcising._provenance import HDF5_SCHEMA_VERSION, git_commit
from mcising.benchmarks import BenchmarkResult
from mcising.config import (
    AdaptiveConfig,
    Algorithm,
    ExecutionMode,
    LatticeConfig,
    LatticeType,
    SimulationConfig,
)
from mcising.constants import (
    DEFAULT_ADAPTIVE_C_WINDOW,
    DEFAULT_ADAPTIVE_MAX_THERMALIZATION,
    DEFAULT_ADAPTIVE_MAX_TOTAL_SWEEPS,
    DEFAULT_ADAPTIVE_MIN_INDEPENDENT_SAMPLES,
    DEFAULT_ADAPTIVE_MIN_THERMALIZATION,
    DEFAULT_ADAPTIVE_TAU_MULTIPLIER,
    DEFAULT_MEASUREMENT_INTERVAL,
    DEFAULT_N_SWEEPS,
    DEFAULT_N_THERMALIZATION,
    DEFAULT_SEED,
    TC_CUBIC_3D,
    TC_HONEYCOMB_2D,
    TC_SQUARE_2D,
    TC_TRIANGULAR_2D,
)
from mcising.io import checkpoint_run, save_hdf5, save_json_summary
from mcising.simulation import Simulation

__all__: Final[list[str]] = ["app"]

app = typer.Typer(
    name="mcising",
    help="High-performance Ising model Monte Carlo simulation.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def info() -> None:
    """Display version, build info, and available algorithms."""
    table = Table(title="mcising", show_header=False, border_style="blue")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Version", mcising.__version__)
    commit = git_commit()
    if commit is not None:
        table.add_row("Git commit", commit)
    table.add_row("HDF5 schema", str(HDF5_SCHEMA_VERSION))
    table.add_row(
        "Lattice types",
        ", ".join(lt.value for lt in LatticeType),
    )
    table.add_row(
        "Algorithms",
        ", ".join(a.value for a in Algorithm),
    )
    table.add_row("Rust core", "mcising._core (PyO3)")

    console.print(table)


@app.command()
def run(
    lattice_size: Annotated[
        int,
        typer.Option(
            "-L",
            "--lattice-size",
            help="Lattice size L (L x L; even for triangular/honeycomb).",
        ),
    ] = 16,
    lattice: Annotated[
        LatticeType,
        typer.Option(
            "--lattice",
            help="Lattice geometry.",
        ),
    ] = LatticeType.SQUARE,
    j1: Annotated[float, typer.Option(help="Nearest-neighbor coupling.")] = 1.0,
    j2: Annotated[float, typer.Option(help="Next-nearest-neighbor coupling.")] = 0.0,
    j3: Annotated[float, typer.Option(help="Third-nearest-neighbor coupling.")] = 0.0,
    h: Annotated[float, typer.Option(help="External magnetic field.")] = 0.0,
    temperatures: Annotated[
        list[float] | None,
        typer.Option("-T", "--temperature", help="Temperature(s) to simulate."),
    ] = None,
    t_range: Annotated[
        str | None,
        typer.Option(
            "--T-range",
            help="Temperature range as start:stop:step (e.g. 4.0:0.5:0.1).",
        ),
    ] = None,
    n_sweeps: Annotated[
        int, typer.Option("--sweeps", help="MC sweeps per temperature.")
    ] = DEFAULT_N_SWEEPS,
    n_therm: Annotated[
        int, typer.Option("--therm", help="Thermalization sweeps.")
    ] = DEFAULT_N_THERMALIZATION,
    measurement_interval: Annotated[
        int, typer.Option("--interval", help="Measurement interval.")
    ] = DEFAULT_MEASUREMENT_INTERVAL,
    seed: Annotated[int, typer.Option(help="Random seed.")] = DEFAULT_SEED,
    swap_interval: Annotated[
        int,
        typer.Option(
            "--swap-interval",
            help="Sweeps between replica-swap attempts (parallel tempering).",
        ),
    ] = 1,
    store_configs: Annotated[
        bool,
        typer.Option(
            "--store-configs/--no-store-configs",
            help="Store spin configurations at each measurement.",
        ),
    ] = True,
    correlation: Annotated[
        bool,
        typer.Option("--correlation", help="Compute correlation function."),
    ] = False,
    correlation_interval: Annotated[
        int,
        typer.Option(
            "--correlation-interval",
            help="Evaluate the correlation function every k-th measurement.",
        ),
    ] = 1,
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Output HDF5 file path."),
    ] = None,
    json_summary: Annotated[
        Path | None,
        typer.Option("--json", help="Output JSON summary path."),
    ] = None,
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            help="HDF5 checkpoint file for crash recovery.",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume from an existing checkpoint file."),
    ] = False,
    checkpoint_interval: Annotated[
        int,
        typer.Option(
            "--checkpoint-interval",
            help="Save checkpoint every N temperatures (default: every one).",
        ),
    ] = 1,
    adaptive: Annotated[
        bool,
        typer.Option(
            "--adaptive",
            help="Enable adaptive thermalization and measurement spacing.",
        ),
    ] = False,
    min_samples: Annotated[
        int,
        typer.Option(
            "--min-samples",
            help="Target independent samples per temperature (adaptive mode).",
        ),
    ] = DEFAULT_ADAPTIVE_MIN_INDEPENDENT_SAMPLES,
    max_sweeps: Annotated[
        int,
        typer.Option(
            "--max-sweeps",
            help="Max total sweeps per temperature (adaptive mode).",
        ),
    ] = DEFAULT_ADAPTIVE_MAX_TOTAL_SWEEPS,
    min_therm: Annotated[
        int,
        typer.Option(
            "--min-therm",
            help="Adaptive lower bound on thermalization sweeps "
            "(min_thermalization_sweeps).",
        ),
    ] = DEFAULT_ADAPTIVE_MIN_THERMALIZATION,
    max_therm: Annotated[
        int,
        typer.Option(
            "--max-therm",
            help="Adaptive cap on total thermalization sweeps "
            "(max_thermalization_sweeps).",
        ),
    ] = DEFAULT_ADAPTIVE_MAX_THERMALIZATION,
    c_window: Annotated[
        float,
        typer.Option(
            "--c-window",
            help="Sokal windowing constant for tau_int (adaptive mode).",
        ),
    ] = DEFAULT_ADAPTIVE_C_WINDOW,
    tau_multiplier: Annotated[
        float,
        typer.Option(
            "--tau-multiplier",
            help="Measurement interval = tau_multiplier * tau_int (adaptive mode).",
        ),
    ] = DEFAULT_ADAPTIVE_TAU_MULTIPLIER,
    algorithm: Annotated[
        Algorithm,
        typer.Option(
            "--algorithm",
            help="Monte Carlo update algorithm.",
        ),
    ] = Algorithm.METROPOLIS,
    mode: Annotated[
        ExecutionMode,
        typer.Option(
            "--mode",
            help="Execution mode.",
        ),
    ] = ExecutionMode.COOLDOWN,
) -> None:
    """Run a Monte Carlo simulation of the Ising model."""
    if temperatures and t_range:
        raise typer.BadParameter("Use either -T or --T-range, not both.")

    if resume and checkpoint is None:
        raise typer.BadParameter("--resume requires --checkpoint.")

    if t_range:
        temps = _parse_t_range(t_range)
    elif temperatures:
        temps = tuple(temperatures)
    else:
        temps = (3.0, 2.269, 1.5)

    adaptive_config = AdaptiveConfig(
        enabled=adaptive,
        min_thermalization_sweeps=min_therm,
        max_thermalization_sweeps=max_therm,
        c_window=c_window,
        min_independent_samples=min_samples,
        max_total_sweeps=max_sweeps,
        tau_multiplier=tau_multiplier,
    )

    config = SimulationConfig(
        lattice=LatticeConfig(
            lattice_type=lattice,
            size=lattice_size,
            j1=j1,
            j2=j2,
            j3=j3,
            h=h,
        ),
        algorithm=algorithm,
        temperatures=temps,
        n_sweeps=n_sweeps,
        n_thermalization=n_therm,
        measurement_interval=measurement_interval,
        swap_interval=swap_interval,
        seed=seed,
        store_configs=store_configs,
        compute_correlation=correlation,
        correlation_interval=correlation_interval,
        adaptive=adaptive_config,
        mode=mode,
    )

    _print_config(config)

    sim = Simulation(config)

    if checkpoint is not None:
        results = checkpoint_run(
            sim,
            checkpoint,
            show_progress=True,
            resume=resume,
            checkpoint_interval=checkpoint_interval,
        )
        console.print(f"\n[green]Checkpoint:[/green] {checkpoint}")
    else:
        results = sim.run(show_progress=True)

    _print_results_summary(results)

    if output is not None:
        # Skip redundant save if output is the same as checkpoint
        if checkpoint is None or output.resolve() != checkpoint.resolve():
            save_hdf5(results, output)
            console.print(f"\n[green]Saved HDF5:[/green] {output}")

    if json_summary is not None:
        save_json_summary(results, json_summary)
        console.print(f"[green]Saved JSON:[/green] {json_summary}")

    if output is None and json_summary is None and checkpoint is None:
        console.print(
            "\n[dim]Tip: use -o results.h5 or --json summary.json to save output.[/dim]"
        )


@app.command()
def benchmark(
    lattice_size: Annotated[
        int,
        typer.Option(
            "-L",
            "--lattice-size",
            help="Lattice size L (even sizes cover all lattices).",
        ),
    ] = 32,
    n_sweeps: Annotated[
        int, typer.Option("--sweeps", help="Sweeps to benchmark.")
    ] = 10000,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    scaling: Annotated[
        bool,
        typer.Option(
            "--scaling",
            help="Run scaling benchmark across multiple lattice sizes.",
        ),
    ] = False,
) -> None:
    """Benchmark mcising performance across all lattices, algorithms, and couplings."""
    from mcising.benchmarks import bench_mcising

    if scaling:
        _run_scaling_benchmark(seed)
        return

    cubic_size = min(lattice_size, 16)
    chain_size = lattice_size * lattice_size  # same site count as 2D

    console.print(
        Panel(
            "[bold]mcising Benchmark[/bold]\n"
            f"L={lattice_size} (cubic L={cubic_size}), "
            f"{n_sweeps:,} sweeps per case",
            border_style="blue",
        )
    )

    def _run(
        label: str,
        lt: str,
        alg: str,
        sz: int,
        temp: float,
    ) -> BenchmarkResult:
        return bench_mcising(sz, n_sweeps, seed, alg, lt, temp)

    # ── Table 1: Metropolis across lattices ───────────────────────
    console.print("\n[bold]Metropolis Performance[/bold]")
    metro_table = Table(border_style="green")
    metro_table.add_column("Lattice", style="bold")
    metro_table.add_column("Sites", justify="right")
    metro_table.add_column("Updates/sec", justify="right")
    metro_table.add_column("Sweeps/sec", justify="right")
    metro_table.add_column("E/site", justify="right")

    metro_cases = [
        (f"Square {lattice_size}x{lattice_size}", "square", lattice_size, TC_SQUARE_2D),
        (
            f"Triangular {lattice_size}x{lattice_size}",
            "triangular",
            lattice_size,
            TC_TRIANGULAR_2D,
        ),
        (
            f"Honeycomb {lattice_size}x{lattice_size}",
            "honeycomb",
            lattice_size,
            TC_HONEYCOMB_2D,
        ),
        (f"Chain ({chain_size})", "chain", chain_size, 1.0),
        (f"Cubic {cubic_size}^3", "cubic", cubic_size, TC_CUBIC_3D),
    ]

    with console.status("[bold blue]Metropolis benchmarks..."):
        for label, lt, sz, temp in metro_cases:
            r = _run(label, lt, "metropolis", sz, temp)
            metro_table.add_row(
                label,
                f"{r.total_updates // r.n_sweeps:,}",
                f"{r.updates_per_sec:,.0f}",
                f"{r.sweeps_per_sec:,.0f}",
                f"{r.energy:.4f}",
            )
    console.print(metro_table)

    # ── Table 2: Cluster algorithms ───────────────────────────────
    console.print(
        f"\n[bold]Cluster Algorithms (Square {lattice_size}x{lattice_size})[/bold]"
    )
    cluster_table = Table(border_style="green")
    cluster_table.add_column("Algorithm", style="bold")
    cluster_table.add_column("Updates/sec", justify="right")
    cluster_table.add_column("Sweeps/sec", justify="right")
    cluster_table.add_column("E/site", justify="right")

    with console.status("[bold blue]Cluster benchmarks..."):
        for alg_label, alg in [("Wolff", "wolff"), ("Swendsen-Wang", "swendsen_wang")]:
            r = _run(alg_label, "square", alg, lattice_size, TC_SQUARE_2D)
            cluster_table.add_row(
                alg_label,
                f"{r.updates_per_sec:,.0f}",
                f"{r.sweeps_per_sec:,.0f}",
                f"{r.energy:.4f}",
            )
    console.print(cluster_table)

    # ── Table 3: Coupling strategies ──────────────────────────────
    console.print(
        f"\n[bold]Coupling Strategies (Square {lattice_size}x{lattice_size})[/bold]"
    )
    coupling_table = Table(border_style="green")
    coupling_table.add_column("Strategy", style="bold")
    coupling_table.add_column("Updates/sec", justify="right")
    coupling_table.add_column("Sweeps/sec", justify="right")

    import time as _time

    from mcising._core import IsingSimulation

    coupling_cases = [
        ("J1", 1.0, 0.0, 0.0, 0.0),
        ("J1+J2", 1.0, 1.0, 0.0, 0.0),
        ("J1+J2+J3", 1.0, 1.0, 1.0, 0.0),
        ("J1+J2+J3+H", 1.0, 1.0, 1.0, 1.0),
    ]
    n_sites = lattice_size * lattice_size
    temperature = TC_SQUARE_2D

    with console.status("[bold blue]Coupling benchmarks..."):
        for label, j1, j2, j3, h in coupling_cases:
            sim = IsingSimulation(
                lattice_size,
                j1,
                j2,
                j3,
                h,
                seed,
                "metropolis",
                "square",
            )
            sim.sweep(100, temperature=temperature)
            start = _time.perf_counter()
            sim.sweep(n_sweeps, temperature=temperature)
            elapsed = _time.perf_counter() - start
            ups = n_sweeps * n_sites / elapsed
            sps = n_sweeps / elapsed
            coupling_table.add_row(
                label,
                f"{ups:,.0f}",
                f"{sps:,.0f}",
            )
    console.print(coupling_table)


def _run_scaling_benchmark(seed: int) -> None:
    """Run mcising benchmarks across multiple lattice sizes."""
    from mcising.benchmarks import bench_mcising

    sizes = [8, 16, 32, 64, 128, 256]
    # Scale sweeps down for larger lattices to keep runtime bearable.
    sweep_schedule = {
        8: 5000,
        16: 2000,
        32: 500,
        64: 200,
        128: 50,
        256: 10,
    }

    console.print(
        Panel(
            "[bold]Scaling Benchmark[/bold]: L = "
            + ", ".join(str(s) for s in sizes)
            + f"\nT=T_c={TC_SQUARE_2D:.4g}, Metropolis algorithm",
            border_style="blue",
        )
    )

    table = Table(
        title="Spin Updates / Second (higher is better)",
        border_style="green",
    )
    table.add_column("L", justify="right", style="bold")
    table.add_column("Sweeps", justify="right")
    table.add_column("mcising (Rust)", justify="right")

    with console.status("[bold blue]Running scaling benchmark..."):
        for l_size in sizes:
            sweeps = sweep_schedule[l_size]
            result = bench_mcising(l_size, sweeps, seed)
            table.add_row(str(l_size), f"{sweeps:,}", f"{result.updates_per_sec:,.0f}")

    console.print(table)


def _parse_t_range(value: str) -> tuple[float, ...]:
    """Parse a 'start:stop:step' string into a temperature tuple."""

    parts = value.split(":")
    if len(parts) != 3:
        raise typer.BadParameter(
            f"--T-range must be start:stop:step (e.g. 4.0:0.5:0.1), got '{value}'"
        )
    try:
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        raise typer.BadParameter(f"--T-range values must be numbers, got '{value}'")
    if step <= 0:
        raise typer.BadParameter(f"step must be positive, got {step}")
    if start <= 0 or stop <= 0:
        raise typer.BadParameter("start and stop must be positive temperatures")

    if start > stop:
        temps = np.arange(start, stop - 1e-10, -step)
    else:
        temps = np.arange(start, stop + 1e-10, step)

    if len(temps) == 0:
        raise typer.BadParameter(f"--T-range produced no temperatures: '{value}'")

    return tuple(float(t) for t in temps)


def _print_config(config: SimulationConfig) -> None:
    """Print simulation configuration as a Rich panel."""
    table = Table(show_header=False, border_style="blue", pad_edge=False)
    table.add_column("Param", style="bold")
    table.add_column("Value")

    lc = config.lattice
    table.add_row("Algorithm", config.algorithm.value)
    table.add_row("Lattice", f"L={lc.size} {lc.lattice_type.value}")
    table.add_row("J1 / J2 / J3 / h", f"{lc.j1} / {lc.j2} / {lc.j3} / {lc.h}")
    table.add_row("Temperatures", ", ".join(f"{t:.3f}" for t in config.temperatures))
    table.add_row("Sweeps", str(config.n_sweeps))
    table.add_row("Thermalization", str(config.n_thermalization))
    table.add_row("Measurement interval", str(config.measurement_interval))
    table.add_row("Seed", str(config.seed))
    table.add_row("Correlation", str(config.compute_correlation))
    if config.compute_correlation:
        table.add_row("Correlation interval", str(config.correlation_interval))
    if config.adaptive.enabled:
        table.add_row("Adaptive", "enabled")
        table.add_row("  Min samples", str(config.adaptive.min_independent_samples))
        table.add_row("  Max sweeps", str(config.adaptive.max_total_sweeps))

    console.print(Panel(table, title="[bold]Configuration[/bold]", border_style="blue"))


def _print_results_summary(results: mcising.SimulationResults) -> None:
    """Print a summary table of results."""

    table = Table(title="Results Summary", border_style="green")
    table.add_column("T", justify="right", style="bold")
    table.add_column("<E>/site", justify="right")
    table.add_column("<|M|>/site", justify="right")
    if results.correlation_length is not None:
        table.add_column("xi", justify="right")
    if results.adaptive_diagnostics is not None:
        table.add_column("tau_int", justify="right")
        table.add_column("interval", justify="right")

    for temp in results.temperatures:
        row: list[str] = [f"{temp:.3f}"]
        if temp in results.energy:
            row.append(f"{float(np.mean(results.energy[temp])):.4f}")
        else:
            row.append("-")
        if temp in results.magnetization:
            row.append(f"{float(np.mean(np.abs(results.magnetization[temp]))):.4f}")
        else:
            row.append("-")
        if (
            results.correlation_length is not None
            and temp in results.correlation_length
        ):
            row.append(f"{float(np.mean(results.correlation_length[temp])):.2f}")
        if (
            results.adaptive_diagnostics is not None
            and temp in results.adaptive_diagnostics
        ):
            diag = results.adaptive_diagnostics[temp]
            row.append(f"{diag.tau_int:.1f}")
            row.append(str(diag.measurement_interval))
        table.add_row(*row)

    elapsed = results.metadata.get("elapsed_seconds", 0)
    console.print(table)
    console.print(f"\n[dim]Completed in {float(elapsed):.2f}s[/dim]")  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════
# Post-run commands: summary, plot, export
# ═══════════════════════════════════════════════════════════════════


@app.command()
def summary(
    file: Annotated[Path, typer.Argument(help="HDF5 results file.")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON."),
    ] = False,
    csv_output: Annotated[
        bool,
        typer.Option("--csv", help="Output as CSV."),
    ] = False,
) -> None:
    """Inspect simulation results from an HDF5 file."""
    import json as json_mod
    import math

    from mcising.io import load_hdf5

    results = load_hdf5(file)

    if json_output or csv_output:
        columns = (
            "T",
            "E_mean",
            "E_err",
            "E_std",
            "M_mean",
            "M_err",
            "Cv",
            "Cv_err",
            "chi",
            "chi_err",
            "U4",
            "U4_err",
            "tau_int",
            "samples",
        )
        rows = []
        for t in sorted(results.temperatures):
            if t not in results.energy:
                continue
            e = results.energy[t]
            stats = results.statistics(t)
            row: dict[str, float | int] = {
                "T": t,
                "E_mean": stats.energy.value,
                "E_err": stats.energy.error,
                "E_std": float(np.std(e)),
                "M_mean": stats.abs_magnetization.value,
                "M_err": stats.abs_magnetization.error,
                "Cv": stats.specific_heat.value,
                "Cv_err": stats.specific_heat.error,
                "chi": stats.susceptibility.value,
                "chi_err": stats.susceptibility.error,
                "U4": stats.binder_cumulant.value,
                "U4_err": stats.binder_cumulant.error,
                "tau_int": stats.tau_int,
                "samples": stats.n_samples,
            }
            rows.append(row)

        def _is_nan(value: float | int) -> bool:
            return isinstance(value, float) and math.isnan(value)

        if json_output:
            payload: dict[str, object] = {}
            for key in (
                "version",
                "schema_version",
                "seed",
                "mode",
                "algorithm",
                "git_commit",
            ):
                if key in results.metadata:
                    payload[key] = results.metadata[key]
            # NaN is invalid strict JSON; unknown values are omitted,
            # never written as null (P07 policy).
            payload["results"] = [
                {k: v for k, v in row.items() if not _is_nan(v)} for row in rows
            ]
            print(json_mod.dumps(payload, indent=2))
        else:
            print(",".join(columns))
            for row in rows:
                print(
                    ",".join(
                        "" if _is_nan(row[col]) else str(row[col]) for col in columns
                    )
                )
    else:
        results.summary()


# ── Plot subcommand group ─────────────────────────────────────────

plot_app = typer.Typer(
    help="Generate plots from HDF5 results. Requires -o.",
)
app.add_typer(plot_app, name="plot")


def _save_plot(fig: object, output: Path, dpi: int) -> None:
    """Save a matplotlib figure and close it."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    assert isinstance(fig, Figure)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved:[/green] {output}")


@plot_app.command("energy")
def plot_energy_cmd(
    file: Annotated[list[Path], typer.Argument(help="HDF5 file(s).")],
    output: Annotated[Path, typer.Option("-o", help="Output image.")],
    dpi: Annotated[int, typer.Option(help="DPI.")] = 150,
) -> None:
    """Plot energy per site vs temperature."""
    from mcising.plotting import plot_energy

    paths = [str(f) for f in file]
    src: str | list[str] = paths if len(paths) > 1 else paths[0]
    _save_plot(plot_energy(src), output, dpi)  # type: ignore[arg-type]


@plot_app.command("magnetization")
def plot_magnetization_cmd(
    file: Annotated[list[Path], typer.Argument(help="HDF5 file(s).")],
    output: Annotated[Path, typer.Option("-o", help="Output image.")],
    dpi: Annotated[int, typer.Option(help="DPI.")] = 150,
) -> None:
    """Plot |magnetization| per site vs temperature."""
    from mcising.plotting import plot_magnetization

    paths = [str(f) for f in file]
    src: str | list[str] = paths if len(paths) > 1 else paths[0]
    _save_plot(plot_magnetization(src), output, dpi)  # type: ignore[arg-type]


@plot_app.command("specific-heat")
def plot_specific_heat_cmd(
    file: Annotated[list[Path], typer.Argument(help="HDF5 file(s).")],
    output: Annotated[Path, typer.Option("-o", help="Output image.")],
    dpi: Annotated[int, typer.Option(help="DPI.")] = 150,
) -> None:
    """Plot specific heat per site vs temperature."""
    from mcising.plotting import plot_specific_heat

    paths = [str(f) for f in file]
    src: str | list[str] = paths if len(paths) > 1 else paths[0]
    _save_plot(plot_specific_heat(src), output, dpi)  # type: ignore[arg-type]


@plot_app.command("susceptibility")
def plot_susceptibility_cmd(
    file: Annotated[list[Path], typer.Argument(help="HDF5 file(s).")],
    output: Annotated[Path, typer.Option("-o", help="Output image.")],
    dpi: Annotated[int, typer.Option(help="DPI.")] = 150,
) -> None:
    """Plot susceptibility per site vs temperature."""
    from mcising.plotting import plot_susceptibility

    paths = [str(f) for f in file]
    src: str | list[str] = paths if len(paths) > 1 else paths[0]
    _save_plot(plot_susceptibility(src), output, dpi)  # type: ignore[arg-type]


@plot_app.command("lattice")
def plot_lattice_cmd(
    file: Annotated[Path, typer.Argument(help="HDF5 file.")],
    output: Annotated[Path, typer.Option("-o", help="Output image.")],
    temperature: Annotated[
        float, typer.Option("--temperature", "-T", help="Temperature.")
    ],
    n: Annotated[
        int | None,
        typer.Option(help="Config index (0-based). Omit for all."),
    ] = None,
    dpi: Annotated[int, typer.Option(help="DPI.")] = 150,
) -> None:
    """Plot spin configuration(s) at a temperature."""
    from mcising.plotting import plot_lattice

    _save_plot(
        plot_lattice(str(file), temperature=temperature, n=n),
        output,
        dpi,
    )


@plot_app.command("timeseries")
def plot_timeseries_cmd(
    file: Annotated[Path, typer.Argument(help="HDF5 file.")],
    output: Annotated[Path, typer.Option("-o", help="Output image.")],
    temperature: Annotated[
        float, typer.Option("--temperature", "-T", help="Temperature.")
    ],
    dpi: Annotated[int, typer.Option(help="DPI.")] = 150,
) -> None:
    """Plot energy time series at a temperature."""
    from mcising.plotting import plot_energy_timeseries

    _save_plot(plot_energy_timeseries(str(file), temperature), output, dpi)


@plot_app.command("histogram")
def plot_histogram_cmd(
    file: Annotated[Path, typer.Argument(help="HDF5 file.")],
    output: Annotated[Path, typer.Option("-o", help="Output image.")],
    temperature: Annotated[
        float, typer.Option("--temperature", "-T", help="Temperature.")
    ],
    dpi: Annotated[int, typer.Option(help="DPI.")] = 150,
) -> None:
    """Plot magnetization distribution at a temperature."""
    from mcising.plotting import plot_magnetization_histogram

    _save_plot(
        plot_magnetization_histogram(str(file), temperature),
        output,
        dpi,
    )


# ── Export command ────────────────────────────────────────────────


@app.command()
def export(
    file: Annotated[Path, typer.Argument(help="HDF5 results file.")],
    output: Annotated[Path, typer.Argument(help="Output zip file.")],
    flat: Annotated[
        bool, typer.Option("--flat", help="Flat folder structure.")
    ] = False,
    temperature: Annotated[
        list[float] | None,
        typer.Option("--temperature", "-T", help="Temperature(s)."),
    ] = None,
    dpi: Annotated[int, typer.Option(help="DPI.")] = 100,
) -> None:
    """Export lattice configurations as PNGs in a zip file."""
    from mcising.plotting import export_lattices

    count = export_lattices(
        str(file),
        output,
        flat=flat,
        temperatures=temperature,
        dpi=dpi,
    )
    console.print(f"[green]Exported {count} images to {output}[/green]")


# ── Docs subcommand group ─────────────────────────────────────────

docs_app = typer.Typer(
    help="API and capability reference for agents and developers.",
    invoke_without_command=True,
)
app.add_typer(docs_app, name="docs")


@docs_app.callback()
def docs_default(
    ctx: typer.Context,
) -> None:
    """Show full capabilities overview."""
    if ctx.invoked_subcommand is not None:
        return
    # Default: show everything
    docs_cli()


@docs_app.command("lattices")
def docs_lattices() -> None:
    """List available lattice types."""
    print(
        """LATTICE TYPES
=============
square       2D  coord=4   Tc=2.269   shape=(L,L)      J1,J2,J3,H supported
triangular   2D  coord=6   Tc=3.641   shape=(L,L)      J1,J2,J3,H; even L only
honeycomb    2D  coord=3   Tc=1.519   shape=(L,L,2)    J1,J2,J3,H; even L only
cubic        3D  coord=6   Tc=4.5115  shape=(L,L,L)    J1,J2,J3,H supported
chain        1D  coord=2   Tc=0       shape=(N,)        J1,J2,J3,H supported"""
    )


@docs_app.command("algorithms")
def docs_algorithms() -> None:
    """List available algorithms and constraints."""
    print(
        """ALGORITHMS
==========
metropolis      Single-spin-flip. All couplings. All lattices.
wolff           Cluster flip (DFS). J2=J3=H=0 only. All lattices.
swendsen_wang   Multi-cluster (Union-Find). J2=J3=H=0 only."""
    )


@docs_app.command("couplings")
def docs_couplings() -> None:
    """Show coupling support per lattice."""
    print(
        """COUPLING SUPPORT
================
Lattice      z_NN (J1)  z_NNN (J2)  z_TNN (J3)  H
square       4          4           4           yes
triangular   6          6           6           yes
honeycomb    3          6           3           yes
cubic        6          12          8           yes
chain        2          2           2           yes

15 Metropolis strategies auto-selected based on active couplings:
J1, J2, J3, H, J1H, J2H, J3H, J1J2, J1J3, J2J3, J1J2H, J1J3H, J2J3H, J1J2J3, J1J2J3H"""
    )


@docs_app.command("modes")
def docs_modes() -> None:
    """List execution modes."""
    print(
        """EXECUTION MODES
===============
cooldown            Sequential cool-down. Single-threaded. Default.
independent         Parallel per T via Rayon. ~6x speedup.
parallel_tempering  Parallel + replica swap. Best for frustration."""
    )


@docs_app.command("cli")
def docs_cli() -> None:
    """Show all CLI commands with examples."""
    print(
        """mcising CLI REFERENCE
=====================

mcising info
  Show version and build info.

mcising run [OPTIONS]
  Run a Monte Carlo simulation.
  Examples:
    mcising run -L 32 --T-range 3.5:1.5:0.1 -o results.h5
    mcising run -L 32 --lattice triangular --j2 0.5 -o results.h5
    mcising run -L 32 --algorithm wolff --mode independent -o results.h5
    mcising run -L 32 --mode parallel_tempering --swap-interval 5 \\
        --interval 10 -o results.h5
    mcising run -L 64 --adaptive --min-samples 200 --tau-multiplier 3 \\
        --min-therm 500 --max-therm 20000 --c-window 8 -o results.h5
    mcising run -L 64 --no-store-configs -o results.h5
    mcising run -L 32 --checkpoint sim.h5 --resume

mcising summary <file.h5>
  Print results table from HDF5.
  Examples:
    mcising summary results.h5
    mcising summary results.h5 --json
    mcising summary results.h5 --csv

mcising plot <type> <file.h5> -o <output.png>
  Generate a plot. Types: energy, magnetization, specific-heat,
  susceptibility, lattice, timeseries, histogram.
  Examples:
    mcising plot energy results.h5 -o energy.png
    mcising plot specific-heat results.h5 -o cv.png
    mcising plot lattice results.h5 -o lat.png -T 2.269
    mcising plot lattice results.h5 -o lat.png -T 2.269 --n 3
    mcising plot timeseries results.h5 -o trace.png -T 2.269
    mcising plot histogram results.h5 -o hist.png -T 2.269
    mcising plot energy a.h5 b.h5 c.h5 -o compare.png

mcising export <file.h5> <output.zip>
  Export lattice PNGs to zip.
  Examples:
    mcising export results.h5 lattices.zip
    mcising export results.h5 lattices.zip --flat
    mcising export results.h5 lattices.zip -T 2.269 -T 1.5

mcising benchmark
  Performance benchmark.
  Examples:
    mcising benchmark
    mcising benchmark -L 64 --sweeps 50000
    mcising benchmark --scaling

mcising docs [topic]
  Capability reference. Topics: lattices, algorithms, couplings, modes, cli.
  Examples:
    mcising docs
    mcising docs lattices
    mcising docs algorithms"""
    )
