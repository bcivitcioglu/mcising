#!/usr/bin/env python3
"""Reproduce mcising vs peapods benchmark comparison.

Requirements:
    pip install mcising peapods rich

Usage:
    python benchmarks/compare_peapods.py
    python benchmarks/compare_peapods.py --sweeps 5000 -L 64
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mcising.benchmarks import (
    BenchmarkResult,
    bench_mcising,
    bench_peapods,
    bench_peapods_cubic,
    bench_peapods_sw,
    bench_peapods_triangular,
    bench_peapods_wolff,
)

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="mcising vs peapods benchmark")
    parser.add_argument("-L", "--lattice-size", type=int, default=32)
    parser.add_argument("--sweeps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lattice_size = args.lattice_size
    n_sweeps = args.sweeps
    seed = args.seed
    cubic_size = min(lattice_size, 16)

    cases: list[
        tuple[str, str, str, int, float, object]
    ] = [
        ("Metropolis: Square", "square", "metropolis", lattice_size, 2.269, bench_peapods),
        ("Metropolis: Triangular", "triangular", "metropolis", lattice_size, 3.641, bench_peapods_triangular),
        ("Metropolis: Cubic", "cubic", "metropolis", cubic_size, 4.5115, bench_peapods_cubic),
        ("Wolff: Square", "square", "wolff", lattice_size, 2.269, bench_peapods_wolff),
        ("Swendsen-Wang: Square", "square", "swendsen_wang", lattice_size, 2.269, bench_peapods_sw),
    ]

    console.print(
        Panel(
            f"[bold]mcising vs peapods[/bold]\n"
            f"L={lattice_size} (cubic L={cubic_size}), "
            f"{n_sweeps:,} sweeps per case",
            border_style="blue",
        )
    )

    for title, lat_type, algorithm, size, temp, peapods_fn in cases:
        dim = f"{size}x{size}" if lat_type != "cubic" else f"{size}^3"
        console.print(f"\n[bold blue]{title}[/bold blue] ({dim}, T={temp})")

        table = Table(border_style="green", show_header=True)
        table.add_column("Implementation", style="bold")
        table.add_column("Time", justify="right")
        table.add_column("Updates/sec", justify="right")
        table.add_column("Sweeps/sec", justify="right")
        table.add_column("E/site", justify="right")
        table.add_column("vs mcising", justify="right", style="bold")

        with console.status(f"[bold blue]{title}..."):
            mc = bench_mcising(size, n_sweeps, seed, algorithm, lat_type, temp)
            pp = peapods_fn(size, n_sweeps, seed)

        rows: list[BenchmarkResult] = [mc, pp]
        mc_ups = mc.updates_per_sec

        for r in rows:
            if "mcising" in r.name:
                vs = "1.0x"
            elif r.updates_per_sec > mc_ups:
                ratio = r.updates_per_sec / mc_ups
                vs = f"[yellow]{ratio:.1f}x faster[/yellow]"
            elif r.updates_per_sec > 0:
                ratio = mc_ups / r.updates_per_sec
                vs = f"{ratio:.1f}x slower"
            else:
                vs = "-"

            table.add_row(
                r.name,
                f"{r.elapsed:.3f} s",
                f"{r.updates_per_sec:,.0f}",
                f"{r.sweeps_per_sec:,.0f}",
                f"{r.energy:.4f}",
                vs,
            )

        console.print(table)


if __name__ == "__main__":
    main()
