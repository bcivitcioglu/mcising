"""Typer CLI for mcising Monte Carlo simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import mcising
from mcising.benchmarks import BenchmarkResult
from mcising.config import (
    AdaptiveConfig,
    Algorithm,
    LatticeConfig,
    LatticeType,
    SimulationConfig,
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
        int, typer.Option("-L", "--lattice-size", help="Lattice size L (L x L).")
    ] = 16,
    lattice: Annotated[
        str, typer.Option("--lattice", help="Lattice type: square, triangular, chain.")
    ] = "square",
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
    ] = 1000,
    n_therm: Annotated[
        int, typer.Option("--therm", help="Thermalization sweeps.")
    ] = 100,
    measurement_interval: Annotated[
        int, typer.Option("--interval", help="Measurement interval.")
    ] = 10,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    correlation: Annotated[
        bool,
        typer.Option("--correlation", help="Compute correlation function."),
    ] = False,
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
    ] = 100,
    max_sweeps: Annotated[
        int,
        typer.Option(
            "--max-sweeps",
            help="Max total sweeps per temperature (adaptive mode).",
        ),
    ] = 100_000,
    algorithm: Annotated[
        str,
        typer.Option(
            "--algorithm",
            help="MC algorithm: metropolis, wolff, swendsen_wang.",
        ),
    ] = "metropolis",
) -> None:
    """Run a Monte Carlo simulation of the 2D Ising model."""
    if temperatures and t_range:
        raise typer.BadParameter("Use either -T or --T-range, not both.")

    if t_range:
        temps = _parse_t_range(t_range)
    elif temperatures:
        temps = tuple(temperatures)
    else:
        temps = (3.0, 2.269, 1.5)

    adaptive_config = AdaptiveConfig(
        enabled=adaptive,
        min_independent_samples=min_samples,
        max_total_sweeps=max_sweeps,
    )

    config = SimulationConfig(
        lattice=LatticeConfig(
            lattice_type=LatticeType(lattice),
            size=lattice_size,
            j1=j1,
            j2=j2,
            j3=j3,
            h=h,
        ),
        algorithm=Algorithm(algorithm),
        temperatures=temps,
        n_sweeps=n_sweeps,
        n_thermalization=n_therm,
        measurement_interval=measurement_interval,
        seed=seed,
        compute_correlation=correlation,
        adaptive=adaptive_config,
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
        int, typer.Option("-L", "--lattice-size", help="Lattice size L.")
    ] = 32,
    n_sweeps: Annotated[
        int, typer.Option("--sweeps", help="Sweeps to benchmark.")
    ] = 10000,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    compare: Annotated[
        bool,
        typer.Option(
            "--compare",
            help="Compare mcising against Pure Python and NumPy baselines.",
        ),
    ] = False,
    scaling: Annotated[
        bool,
        typer.Option(
            "--scaling",
            help="Run scaling benchmark across multiple lattice sizes.",
        ),
    ] = False,
) -> None:
    """Benchmark sweep performance.

    By default, benchmarks the mcising Rust core only.
    Use --compare to include Pure Python and NumPy baselines.
    Use --scaling to run across L=8,16,32,64,128,256.
    """
    from mcising.benchmarks import bench_mcising, bench_numpy, bench_pure_python

    if scaling:
        _run_scaling_benchmark(seed, compare)
        return

    console.print(
        Panel(
            f"[bold]Benchmark:[/bold] {lattice_size}x{lattice_size} lattice, "
            f"{n_sweeps:,} sweeps, T=T_c=2.269",
            border_style="blue",
        )
    )

    results: list[BenchmarkResult] = []

    if compare:
        with console.status("[bold blue]Running Pure Python baseline..."):
            results.append(
                bench_pure_python(lattice_size, n_sweeps, seed)
            )

        with console.status("[bold blue]Running NumPy baseline..."):
            results.append(bench_numpy(lattice_size, n_sweeps, seed))

    with console.status("[bold blue]Running mcising (Rust)..."):
        results.append(bench_mcising(lattice_size, n_sweeps, seed))

    if compare:
        peapods_result = _try_bench_peapods(lattice_size, n_sweeps, seed)
        if peapods_result is not None:
            results.append(peapods_result)

    _print_benchmark_results(results, compare)


def _try_bench_peapods(
    lattice_size: int, n_sweeps: int, seed: int
) -> BenchmarkResult | None:
    """Try to run peapods benchmark; return None if not installed."""
    try:
        from mcising.benchmarks import bench_peapods
    except ImportError:
        console.print(
            "[dim]peapods not installed. "
            "Install with: uv sync --group benchmark[/dim]"
        )
        return None

    try:
        with console.status("[bold blue]Running peapods (Rust)..."):
            return bench_peapods(lattice_size, n_sweeps, seed)
    except ImportError:
        console.print(
            "[dim]peapods not installed. "
            "Install with: uv sync --group benchmark[/dim]"
        )
        return None


def _run_scaling_benchmark(seed: int, compare: bool) -> None:
    """Run benchmarks across multiple lattice sizes."""
    from mcising.benchmarks import (
        bench_mcising,
        bench_numpy,
        bench_pure_python,
    )

    has_peapods = False
    bench_peapods_fn = None
    if compare:
        try:
            from mcising.benchmarks import bench_peapods

            bench_peapods_fn = bench_peapods
            has_peapods = True
        except ImportError:
            pass

    sizes = [8, 16, 32, 64, 128, 256]
    # Same sweeps for all implementations per lattice size.
    # Scale down for larger lattices to keep pure Python bearable.
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
            + "\nT=T_c=2.269, Metropolis algorithm"
            + "\n[dim]Same sweep count per L for fair comparison[/dim]",
            border_style="blue",
        )
    )

    # Build the table
    table = Table(
        title="Spin Updates / Second (higher is better)",
        border_style="green",
    )
    table.add_column("L", justify="right", style="bold")
    table.add_column("Sweeps", justify="right")

    runners: list[tuple[str, object]] = []
    if compare:
        runners.append(("Pure Python", bench_pure_python))
        runners.append(("NumPy", bench_numpy))
    runners.append(("mcising (Rust)", bench_mcising))
    if has_peapods and bench_peapods_fn is not None:
        runners.append(("peapods", bench_peapods_fn))

    for name, _ in runners:
        table.add_column(name, justify="right")

    if compare:
        table.add_column(
            "mcising / Python",
            justify="right",
            style="bold green",
        )
        if has_peapods:
            table.add_column(
                "mcising / peapods",
                justify="right",
                style="bold cyan",
            )

    with console.status("[bold blue]Running scaling benchmark..."):
        for l_size in sizes:
            sweeps = sweep_schedule[l_size]
            row: list[str] = [str(l_size), f"{sweeps:,}"]

            row_results: dict[str, BenchmarkResult] = {}

            for name, bench_fn in runners:
                result = bench_fn(l_size, sweeps, seed)  # type: ignore[operator]
                row_results[name] = result
                row.append(f"{result.updates_per_sec:,.0f}")

            if compare:
                rust_ups = row_results["mcising (Rust)"].updates_per_sec
                python_ups = row_results.get("Pure Python")
                if python_ups and python_ups.updates_per_sec > 0:
                    row.append(
                        f"{rust_ups / python_ups.updates_per_sec:,.0f}x"
                    )
                else:
                    row.append("-")
                if has_peapods:
                    peapods_r = row_results.get("peapods")
                    if peapods_r and peapods_r.updates_per_sec > 0:
                        ratio = rust_ups / peapods_r.updates_per_sec
                        row.append(f"{ratio:,.1f}x")
                    else:
                        row.append("-")

            table.add_row(*row)

    console.print(table)
    if has_peapods:
        console.print(
            "\n[dim]Note: mcising/peapods ratio >1 means mcising"
            " is faster per update.[/dim]"
        )


def _print_benchmark_results(
    results: list[BenchmarkResult], compare: bool
) -> None:
    """Print benchmark results as a Rich table."""

    table = Table(title="Benchmark Results", border_style="green")
    table.add_column("Implementation", style="bold")
    table.add_column("Time", justify="right")
    table.add_column("Sweeps", justify="right")
    table.add_column("Updates/sec", justify="right")
    table.add_column("Sweeps/sec", justify="right")
    table.add_column("E/site", justify="right")
    table.add_column("|M|/site", justify="right")
    if compare:
        table.add_column(
            "vs mcising", justify="right", style="bold green"
        )

    rust_ups: float | None = None
    for r in results:
        if "mcising" in r.name:
            rust_ups = r.updates_per_sec

    for r in results:
        vs_mcising = ""
        if compare and rust_ups is not None and r.updates_per_sec > 0:
            if "mcising" in r.name:
                vs_mcising = "1.0x"
            elif r.updates_per_sec > rust_ups:
                ratio = r.updates_per_sec / rust_ups
                vs_mcising = f"[yellow]{ratio:,.1f}x faster[/yellow]"
            else:
                ratio = rust_ups / r.updates_per_sec
                vs_mcising = f"{ratio:,.1f}x slower"

        table.add_row(
            r.name,
            f"{r.elapsed:.3f} s",
            f"{r.n_sweeps:,}",
            f"{r.updates_per_sec:,.0f}",
            f"{r.sweeps_per_sec:,.0f}",
            f"{r.energy:.4f}",
            f"{abs(r.magnetization):.4f}",
            *([vs_mcising] if compare else []),
        )

    console.print(table)

    if compare and len(results) >= 2:
        console.print()
        # Summary line
        slowest = min(r.updates_per_sec for r in results if r.updates_per_sec > 0)
        fastest = max(r.updates_per_sec for r in results)
        if slowest > 0:
            console.print(
                f"  [bold]Overall speedup:[/bold] mcising is "
                f"[bold green]{fastest / slowest:,.0f}x[/bold green] faster "
                f"than the slowest baseline"
            )


def _parse_t_range(value: str) -> tuple[float, ...]:
    """Parse a 'start:stop:step' string into a temperature tuple."""
    import numpy as np

    parts = value.split(":")
    if len(parts) != 3:
        raise typer.BadParameter(
            f"--T-range must be start:stop:step (e.g. 4.0:0.5:0.1), got '{value}'"
        )
    try:
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        raise typer.BadParameter(
            f"--T-range values must be numbers, got '{value}'"
        )
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

    table.add_row("Algorithm", config.algorithm.value)
    table.add_row("Lattice", f"{config.lattice.size}x{config.lattice.size} square")
    lc = config.lattice
    table.add_row("J1 / J2 / J3 / h", f"{lc.j1} / {lc.j2} / {lc.j3} / {lc.h}")
    table.add_row("Temperatures", ", ".join(f"{t:.3f}" for t in config.temperatures))
    table.add_row("Sweeps", str(config.n_sweeps))
    table.add_row("Thermalization", str(config.n_thermalization))
    table.add_row("Measurement interval", str(config.measurement_interval))
    table.add_row("Seed", str(config.seed))
    table.add_row("Correlation", str(config.compute_correlation))
    if config.adaptive.enabled:
        table.add_row("Adaptive", "enabled")
        table.add_row("  Min samples", str(config.adaptive.min_independent_samples))
        table.add_row("  Max sweeps", str(config.adaptive.max_total_sweeps))

    console.print(Panel(table, title="[bold]Configuration[/bold]", border_style="blue"))


def _print_results_summary(results: mcising.SimulationResults) -> None:
    """Print a summary table of results."""
    import numpy as np

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
