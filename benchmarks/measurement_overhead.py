#!/usr/bin/env python3
"""Measurement-path overhead: time ``Simulation.run()`` end to end.

The reference workload is the one named by the P16 roadmap phase: Metropolis
on a 64x64 square lattice at T = 2.269, 100 annealing sweeps, 200 production
sweeps measured at every sweep (``measurement_interval=1``), configurations
stored. It isolates the cost of *measuring* — the per-sweep energy and
magnetization, the spin snapshot and the Python/FFI plumbing around them —
from the cost of the sweep itself. The other rows vary the algorithm and the
snapshot flag so the decomposition is visible.

Each row reports the minimum and median wall time over ``--repeats`` runs
(a fresh ``Simulation`` per run, ``run()`` timed) and the median cost per
measurement.

Usage:
    uv run python benchmarks/measurement_overhead.py
    uv run python benchmarks/measurement_overhead.py --repeats 30 --json out.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mcising
from mcising import Algorithm, LatticeConfig, Simulation, SimulationConfig
from mcising._provenance import git_commit
from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class Workload:
    """One timed configuration."""

    name: str
    algorithm: Algorithm
    store_configs: bool
    lattice_size: int = 64
    n_sweeps: int = 200
    measurement_interval: int = 1
    n_thermalization: int = 100
    temperature: float = 2.269

    def config(self) -> SimulationConfig:
        return SimulationConfig(
            lattice=LatticeConfig(size=self.lattice_size, j1=1.0),
            algorithm=self.algorithm,
            temperatures=(self.temperature,),
            n_sweeps=self.n_sweeps,
            n_thermalization=self.n_thermalization,
            measurement_interval=self.measurement_interval,
            store_configs=self.store_configs,
        )

    @property
    def n_measurements(self) -> int:
        return self.n_sweeps // self.measurement_interval


WORKLOADS: tuple[Workload, ...] = (
    Workload("reference: metropolis, configs stored", Algorithm.METROPOLIS, True),
    Workload("metropolis, configs off", Algorithm.METROPOLIS, False),
    Workload("wolff, configs stored", Algorithm.WOLFF, True),
    Workload("swendsen_wang, configs stored", Algorithm.SWENDSEN_WANG, True),
)


def time_workload(workload: Workload, repeats: int) -> dict[str, Any]:
    """Time ``run()`` ``repeats`` times and summarise."""
    config = workload.config()
    samples: list[float] = []
    for _ in range(repeats):
        sim = Simulation(config)
        start = time.perf_counter()
        sim.run(show_progress=False)
        samples.append(time.perf_counter() - start)
    median = statistics.median(samples)
    return {
        **asdict(workload),
        "algorithm": workload.algorithm.value,
        "repeats": repeats,
        "min_ms": min(samples) * 1e3,
        "median_ms": median * 1e3,
        "us_per_measurement": median / workload.n_measurements * 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--json", type=Path, default=None, help="write rows as JSON")
    args = parser.parse_args()

    rows = [time_workload(w, args.repeats) for w in WORKLOADS]

    table = Table(title=f"Simulation.run() wall time (mcising {mcising.__version__})")
    table.add_column("Workload")
    table.add_column("L", justify="right")
    table.add_column("Sweeps", justify="right")
    table.add_column("min ms", justify="right")
    table.add_column("median ms", justify="right")
    table.add_column("us / measurement", justify="right")
    for row in rows:
        table.add_row(
            row["name"],
            str(row["lattice_size"]),
            f"{row['n_thermalization']}+{row['n_sweeps']}",
            f"{row['min_ms']:.2f}",
            f"{row['median_ms']:.2f}",
            f"{row['us_per_measurement']:.1f}",
        )
    Console().print(table)

    if args.json is not None:
        document = {
            "provenance": {
                "mcising_version": mcising.__version__,
                "git_commit": git_commit(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "rows": rows,
        }
        args.json.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
