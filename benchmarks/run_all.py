#!/usr/bin/env python3
"""Regenerate every published mcising performance number.

One script owns every figure in ``README.md`` and the docs. It measures,
writes ``benchmarks/results.json`` (with provenance) and renders the
marker-delimited blocks ``<!-- benchmarks:<section>:begin/end -->`` in the
pages listed in ``DOC_BLOCKS``. ``tests/test_run_all.py`` checks that the
committed pages equal the render of the committed JSON, so a stale number
fails the canonical suite.

Sections (``--sections`` selects a subset; the rest is merged from the
existing results file):

``lattices``
    Metropolis throughput on every lattice geometry (sweeps only).
``baselines``
    mcising against a pure-Python loop and a NumPy checkerboard
    implementation of the same Metropolis update.
``cluster``
    Metropolis, Wolff and Swendsen-Wang with the integrated autocorrelation
    time, so the cost per *independent* sample is comparable.
``scaling``
    Throughput against lattice size.
``parallel``
    Wall time of the independent and parallel-tempering modes against the
    thread count (each run in a fresh process with ``RAYON_NUM_THREADS``).
``overhead``
    ``Simulation.run()`` end to end on the measure-every-sweep workload.
``correlation``
    Cost of one spin-spin correlation evaluation.
``peapods``
    Matched-physics Metropolis comparison with the peapods package
    (``uv sync --group benchmark``); skipped when peapods is missing.

Every number is stored as measured (medians of repeated runs); ratios are
computed at render time from the stored medians, never rounded by hand.

Usage:
    uv run --group benchmark python benchmarks/run_all.py --write-docs
    uv run python benchmarks/run_all.py --from-json benchmarks/results.json --write-docs
    uv run python benchmarks/run_all.py --check
    uv run python benchmarks/run_all.py --quick --skip-peapods --output /tmp/quick.json
    uv run python benchmarks/run_all.py --sections parallel --write-docs
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import mcising
import numpy as np
from mcising import (
    Algorithm,
    ExecutionMode,
    LatticeConfig,
    Simulation,
    SimulationConfig,
)
from mcising._core import IsingSimulation
from mcising._provenance import git_commit
from mcising.benchmarks import (
    BenchmarkResult,
    bench_mcising,
    bench_numpy,
    bench_pure_python,
)
from mcising.constants import (
    TC_CUBIC_3D,
    TC_HONEYCOMB_2D,
    TC_SQUARE_2D,
    TC_TRIANGULAR_2D,
)
from mcising.statistics import tau_int

#: Bump when the *document layout* changes (not when values change).
SCHEMA_VERSION: Final = 1
REPO_ROOT: Final = Path(__file__).resolve().parents[1]
RESULTS_PATH: Final = REPO_ROOT / "benchmarks" / "results.json"
#: The chain has no finite-temperature transition; a fixed T labels the row.
CHAIN_TEMPERATURE: Final = 1.0
#: Matched-physics rows whose energies differ by more than this are dropped.
AGREEMENT_LIMIT_PERCENT: Final = 0.5
TIMING_SEED: Final = 42
REGENERATE_COMMAND: Final = (
    "uv run --group benchmark python benchmarks/run_all.py --write-docs"
)

Log = Callable[[str], None]
#: ``(config_dict, repeats, threads) -> wall seconds per repeat``.
ChildTimer = Callable[[dict[str, Any], int, int], list[float]]


class BenchmarkError(RuntimeError):
    """A benchmark or docs-rendering step could not be completed."""


# --- budgets ------------------------------------------------------------------


@dataclass(frozen=True)
class LatticesBudget:
    size_2d: int
    chain_size: int
    cubic_size: int
    n_sweeps: int
    repeats: int


@dataclass(frozen=True)
class BaselinesBudget:
    size: int
    large_size: int
    pure_python_sweeps: int
    numpy_sweeps: int
    numpy_large_sweeps: int
    mcising_sweeps: int
    mcising_large_sweeps: int
    repeats: int


@dataclass(frozen=True)
class ClusterBudget:
    size: int
    #: Sweeps in the timed (sweeps-only) block.
    timed_sweeps: int
    #: Thermalization and production sweeps of the series behind tau_int.
    n_thermalization: int
    n_sweeps: int
    repeats: int


@dataclass(frozen=True)
class ScalingBudget:
    #: ``(L, timed sweeps)`` pairs; fixed so the budget is machine-independent.
    sweeps_per_size: tuple[tuple[int, int], ...]
    repeats: int


@dataclass(frozen=True)
class ParallelBudget:
    size: int
    n_temperatures: int
    t_min: float
    t_max: float
    n_thermalization: int
    n_sweeps: int
    measurement_interval: int
    thread_counts: tuple[int, ...]
    #: Also time a row at ``os.cpu_count()`` threads.
    all_cores: bool
    include_parallel_tempering: bool
    repeats: int


@dataclass(frozen=True)
class OverheadBudget:
    lattice_size: int
    n_thermalization: int
    n_sweeps: int
    repeats: int


@dataclass(frozen=True)
class CorrelationBudget:
    sizes: tuple[int, ...]
    repeats: int


@dataclass(frozen=True)
class PeapodsBudget:
    size_2d: int
    cubic_size: int
    n_thermalization: int
    n_sweeps: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class Budget:
    quick: bool
    lattices: LatticesBudget
    baselines: BaselinesBudget
    cluster: ClusterBudget
    scaling: ScalingBudget
    parallel: ParallelBudget
    overhead: OverheadBudget
    correlation: CorrelationBudget
    peapods: PeapodsBudget


FULL_BUDGET: Final = Budget(
    quick=False,
    lattices=LatticesBudget(
        size_2d=32, chain_size=1024, cubic_size=16, n_sweeps=10_000, repeats=5
    ),
    baselines=BaselinesBudget(
        size=32,
        large_size=128,
        pure_python_sweeps=200,
        numpy_sweeps=1_000,
        numpy_large_sweeps=200,
        mcising_sweeps=10_000,
        mcising_large_sweeps=1_000,
        repeats=5,
    ),
    cluster=ClusterBudget(
        size=32,
        timed_sweeps=10_000,
        n_thermalization=5_000,
        n_sweeps=100_000,
        repeats=5,
    ),
    scaling=ScalingBudget(
        sweeps_per_size=(
            (8, 400_000),
            (16, 100_000),
            (32, 25_000),
            (64, 6_000),
            (128, 1_500),
            (256, 400),
        ),
        repeats=3,
    ),
    parallel=ParallelBudget(
        size=128,
        n_temperatures=20,
        t_min=1.5,
        t_max=3.5,
        n_thermalization=500,
        n_sweeps=2_000,
        measurement_interval=10,
        thread_counts=(1, 2, 4, 8),
        all_cores=True,
        include_parallel_tempering=True,
        repeats=3,
    ),
    overhead=OverheadBudget(
        lattice_size=64, n_thermalization=100, n_sweeps=200, repeats=20
    ),
    correlation=CorrelationBudget(sizes=(16, 32, 64), repeats=5),
    peapods=PeapodsBudget(
        size_2d=32,
        cubic_size=16,
        n_thermalization=5_000,
        n_sweeps=100_000,
        seeds=(42, 123, 7),
    ),
)

#: The test suite's budget: every section runs end to end in a few seconds.
QUICK_BUDGET: Final = Budget(
    quick=True,
    lattices=LatticesBudget(
        size_2d=8, chain_size=64, cubic_size=4, n_sweeps=20, repeats=1
    ),
    baselines=BaselinesBudget(
        size=8,
        large_size=16,
        pure_python_sweeps=5,
        numpy_sweeps=10,
        numpy_large_sweeps=5,
        mcising_sweeps=50,
        mcising_large_sweeps=20,
        repeats=1,
    ),
    cluster=ClusterBudget(
        size=8, timed_sweeps=100, n_thermalization=50, n_sweeps=256, repeats=1
    ),
    scaling=ScalingBudget(sweeps_per_size=((8, 50), (16, 20)), repeats=1),
    parallel=ParallelBudget(
        size=16,
        n_temperatures=4,
        t_min=1.5,
        t_max=3.5,
        n_thermalization=10,
        n_sweeps=20,
        measurement_interval=5,
        thread_counts=(1,),
        all_cores=False,
        include_parallel_tempering=False,
        repeats=1,
    ),
    overhead=OverheadBudget(
        lattice_size=16, n_thermalization=10, n_sweeps=20, repeats=1
    ),
    correlation=CorrelationBudget(sizes=(8,), repeats=1),
    peapods=PeapodsBudget(
        size_2d=8, cubic_size=4, n_thermalization=50, n_sweeps=200, seeds=(42,)
    ),
)


# --- provenance ---------------------------------------------------------------


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = out.stdout.strip()
    return value or None


def cpu_brand() -> str | None:
    """Marketing name of the CPU, best effort."""
    if sys.platform == "darwin":
        return _sysctl("machdep.cpu.brand_string")
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or None


def performance_cores() -> int | None:
    """Physical performance cores on Apple silicon, else None."""
    if sys.platform != "darwin":
        return None
    value = _sysctl("hw.perflevel0.physicalcpu")
    return int(value) if value is not None and value.isdigit() else None


def memory_bytes() -> int | None:
    if sys.platform == "darwin":
        value = _sysctl("hw.memsize")
        return int(value) if value is not None and value.isdigit() else None
    if sys.platform != "win32":
        try:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, AttributeError):
            return None
    return None


def peapods_version() -> str | None:
    try:
        return importlib.metadata.version("peapods")
    except importlib.metadata.PackageNotFoundError:
        return None


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def provenance(budget: Budget) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "mcising_version": mcising.__version__,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": {
            "cpu": cpu_brand(),
            "cpu_count": os.cpu_count(),
            "performance_cores": performance_cores(),
            "memory_bytes": memory_bytes(),
        },
        "peapods_version": peapods_version(),
        "budget": asdict(budget),
        "sections": {},
    }


# --- timing helpers -----------------------------------------------------------


def median_of(fn: Callable[[], float], repeats: int) -> float:
    if repeats < 1:
        raise BenchmarkError("repeats must be positive")
    return float(statistics.median(fn() for _ in range(repeats)))


def _median_result(results: Sequence[BenchmarkResult]) -> BenchmarkResult:
    """The run whose wall time is the median (upper median for even counts)."""
    ordered = sorted(results, key=lambda r: r.elapsed)
    return ordered[len(ordered) // 2]


def _timed_rows(
    label: str,
    runner: Callable[[], BenchmarkResult],
    repeats: int,
) -> dict[str, Any]:
    results = [runner() for _ in range(repeats)]
    picked = _median_result(results)
    return {
        "label": label,
        "n_sweeps": int(picked.n_sweeps),
        "sites": int(picked.num_sites or picked.lattice_size**2),
        "attempted_updates": int(picked.total_updates),
        "median_seconds": float(statistics.median(r.elapsed for r in results)),
        "energy_per_site": float(picked.energy),
    }


_PARALLEL_CHILD: Final = """\
import json, sys, time
from mcising import Simulation, SimulationConfig
params = json.load(sys.stdin)
config = SimulationConfig.from_dict(params["config"])
seconds = []
for _ in range(params["repeats"]):
    start = time.perf_counter()
    Simulation(config).run(show_progress=False)
    seconds.append(time.perf_counter() - start)
print(json.dumps({"seconds": seconds}))
"""


def _time_run_in_child(
    config: dict[str, Any], repeats: int, threads: int
) -> list[float]:
    """Time ``Simulation.run()`` in a fresh interpreter with a fixed pool size.

    Rayon sizes its global pool from ``RAYON_NUM_THREADS`` the first time it
    is used, so every thread count needs its own process. All repeats run in
    one child; the pool spin-up is paid once and excluded by the median.
    """
    env = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PARALLEL_CHILD],
            input=json.dumps({"config": config, "repeats": repeats}),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        msg = f"timing child failed (threads={threads}):\n{exc.stderr[-2000:]}"
        raise BenchmarkError(msg) from exc
    payload: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return [float(x) for x in payload["seconds"]]


@dataclass(frozen=True)
class Context:
    """Run-time knobs that are not part of the budget."""

    log: Log = print
    skip_peapods: bool = False
    timer: ChildTimer = field(default=_time_run_in_child)


# --- sections -----------------------------------------------------------------


def lattice_cases(budget: LatticesBudget) -> list[tuple[str, str, int, float]]:
    n, c = budget.size_2d, budget.cubic_size
    return [
        (f"Square {n}x{n}", "square", n, TC_SQUARE_2D),
        (f"Triangular {n}x{n}", "triangular", n, TC_TRIANGULAR_2D),
        (f"Honeycomb {n}x{n}", "honeycomb", n, TC_HONEYCOMB_2D),
        (f"Chain ({budget.chain_size})", "chain", budget.chain_size, CHAIN_TEMPERATURE),
        (f"Cubic {c}^3", "cubic", c, TC_CUBIC_3D),
    ]


def run_lattices(budget: Budget, context: Context) -> dict[str, Any]:
    b = budget.lattices
    rows = []
    for label, lattice, size, temperature in lattice_cases(b):
        context.log(f"[lattices] {label}")
        row = _timed_rows(
            label,
            lambda: bench_mcising(
                size,
                b.n_sweeps,
                TIMING_SEED,
                "metropolis",
                lattice,
                temperature,
            ),
            b.repeats,
        )
        row.update({"lattice": lattice, "size": size, "temperature": temperature})
        rows.append(row)
    square = rows[0]
    if not budget.quick and _updates_per_second(square) < 1e8:
        context.log(
            "[lattices] WARNING: square-lattice throughput is below 100M "
            "updates/s - is this a release build (maturin develop --release)?"
        )
    return {"rows": rows}


def run_baselines(budget: Budget, context: Context) -> dict[str, Any]:
    b = budget.baselines
    cases: list[tuple[str, int, Callable[[], BenchmarkResult]]] = [
        (
            "Pure Python",
            b.size,
            lambda: bench_pure_python(b.size, b.pure_python_sweeps, TIMING_SEED),
        ),
        (
            "NumPy (checkerboard)",
            b.size,
            lambda: bench_numpy(b.size, b.numpy_sweeps, TIMING_SEED),
        ),
        (
            "mcising (Rust)",
            b.size,
            lambda: bench_mcising(b.size, b.mcising_sweeps, TIMING_SEED),
        ),
        (
            "NumPy (checkerboard)",
            b.large_size,
            lambda: bench_numpy(b.large_size, b.numpy_large_sweeps, TIMING_SEED),
        ),
        (
            "mcising (Rust)",
            b.large_size,
            lambda: bench_mcising(b.large_size, b.mcising_large_sweeps, TIMING_SEED),
        ),
    ]
    rows = []
    for label, size, runner in cases:
        context.log(f"[baselines] {label} {size}x{size}")
        row = _timed_rows(label, runner, b.repeats)
        row.update({"implementation": label, "size": size})
        rows.append(row)
    return {"temperature": TC_SQUARE_2D, "rows": rows}


CLUSTER_ALGORITHMS: Final = (
    ("Metropolis", "metropolis"),
    ("Wolff", "wolff"),
    ("Swendsen-Wang", "swendsen_wang"),
)


def run_cluster(budget: Budget, context: Context) -> dict[str, Any]:
    b = budget.cluster
    rows = []
    for label, algorithm in CLUSTER_ALGORITHMS:
        context.log(f"[cluster] {label}: timing")
        row = _timed_rows(
            label,
            lambda: bench_mcising(
                b.size, b.timed_sweeps, TIMING_SEED, algorithm, "square", TC_SQUARE_2D
            ),
            b.repeats,
        )
        context.log(f"[cluster] {label}: autocorrelation series")
        config = SimulationConfig(
            lattice=LatticeConfig(size=b.size),
            algorithm=Algorithm(algorithm),
            temperatures=(TC_SQUARE_2D,),
            n_sweeps=b.n_sweeps,
            n_thermalization=b.n_thermalization,
            measurement_interval=1,
            store_configs=False,
            seed=TIMING_SEED,
        )
        results = Simulation(config).run(show_progress=False)
        energy = np.asarray(results.energy[TC_SQUARE_2D], dtype=np.float64)
        magnetization = np.asarray(
            results.magnetization[TC_SQUARE_2D], dtype=np.float64
        )
        row.update(
            {
                "algorithm": algorithm,
                "size": b.size,
                "temperature": TC_SQUARE_2D,
                "series_sweeps": int(energy.size),
                "tau_int_energy": float(tau_int(energy)),
                "tau_int_abs_magnetization": float(tau_int(np.abs(magnetization))),
            }
        )
        rows.append(row)
    return {"rows": rows}


def run_scaling(budget: Budget, context: Context) -> dict[str, Any]:
    b = budget.scaling
    rows = []
    for size, n_sweeps in b.sweeps_per_size:
        context.log(f"[scaling] L={size}")
        row = _timed_rows(
            f"{size}x{size}",
            lambda: bench_mcising(size, n_sweeps, TIMING_SEED),
            b.repeats,
        )
        row["size"] = size
        rows.append(row)
    return {"temperature": TC_SQUARE_2D, "rows": rows}


def parallel_config(b: ParallelBudget, mode: ExecutionMode) -> SimulationConfig:
    temperatures = tuple(
        float(t) for t in np.linspace(b.t_max, b.t_min, b.n_temperatures)
    )
    return SimulationConfig(
        lattice=LatticeConfig(size=b.size),
        algorithm=Algorithm.METROPOLIS,
        temperatures=temperatures,
        n_sweeps=b.n_sweeps,
        n_thermalization=b.n_thermalization,
        measurement_interval=b.measurement_interval,
        store_configs=False,
        seed=TIMING_SEED,
        mode=mode,
    )


def _config_dict(config: SimulationConfig) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(json.dumps(asdict(config)))
    return document


def run_parallel(budget: Budget, context: Context) -> dict[str, Any]:
    b = budget.parallel
    context.log("[parallel] cooldown (in-process)")
    cooldown = parallel_config(b, ExecutionMode.COOLDOWN)

    def _cooldown_once() -> float:
        start = time.perf_counter()
        Simulation(cooldown).run(show_progress=False)
        return time.perf_counter() - start

    rows: list[dict[str, Any]] = [
        {
            "mode": "cooldown",
            "threads": 1,
            "median_seconds": median_of(_cooldown_once, b.repeats),
        }
    ]
    counts = list(b.thread_counts)
    cpu_count = os.cpu_count() or 1
    if b.all_cores and cpu_count not in counts:
        counts.append(cpu_count)
    independent = _config_dict(parallel_config(b, ExecutionMode.INDEPENDENT))
    for threads in counts:
        context.log(f"[parallel] independent, {threads} thread(s)")
        seconds = context.timer(independent, b.repeats, threads)
        rows.append(
            {
                "mode": "independent",
                "threads": threads,
                "median_seconds": float(statistics.median(seconds)),
            }
        )
    if b.include_parallel_tempering:
        context.log(f"[parallel] parallel tempering, {cpu_count} thread(s)")
        tempering = _config_dict(parallel_config(b, ExecutionMode.PARALLEL_TEMPERING))
        seconds = context.timer(tempering, b.repeats, cpu_count)
        rows.append(
            {
                "mode": "parallel_tempering",
                "threads": cpu_count,
                "median_seconds": float(statistics.median(seconds)),
            }
        )
    return {"cpu_count": cpu_count, "rows": rows}


@dataclass(frozen=True)
class Workload:
    """One timed ``Simulation.run()`` configuration of the overhead section."""

    name: str
    algorithm: Algorithm
    store_configs: bool

    def config(self, budget: OverheadBudget) -> SimulationConfig:
        return SimulationConfig(
            lattice=LatticeConfig(size=budget.lattice_size, j1=1.0),
            algorithm=self.algorithm,
            temperatures=(TC_SQUARE_2D,),
            n_sweeps=budget.n_sweeps,
            n_thermalization=budget.n_thermalization,
            measurement_interval=1,
            store_configs=self.store_configs,
            seed=TIMING_SEED,
        )


WORKLOADS: Final[tuple[Workload, ...]] = (
    Workload("Metropolis, configurations stored", Algorithm.METROPOLIS, True),
    Workload("Metropolis, configurations off", Algorithm.METROPOLIS, False),
    Workload("Wolff, configurations stored", Algorithm.WOLFF, True),
    Workload("Swendsen-Wang, configurations stored", Algorithm.SWENDSEN_WANG, True),
)


def time_workload(workload: Workload, budget: OverheadBudget) -> dict[str, Any]:
    config = workload.config(budget)
    samples: list[float] = []
    for _ in range(budget.repeats):
        sim = Simulation(config)
        start = time.perf_counter()
        sim.run(show_progress=False)
        samples.append(time.perf_counter() - start)
    return {
        "name": workload.name,
        "algorithm": workload.algorithm.value,
        "store_configs": workload.store_configs,
        "n_measurements": budget.n_sweeps,
        "min_seconds": float(min(samples)),
        "median_seconds": float(statistics.median(samples)),
    }


def decompose_reference(budget: OverheadBudget) -> dict[str, float]:
    """Per-sweep and per-measurement cost behind the reference workload."""
    sim = IsingSimulation(
        budget.lattice_size, 1.0, 0.0, 0.0, 0.0, TIMING_SEED, "metropolis", "square"
    )
    sim.sweep(budget.n_thermalization, temperature=TC_SQUARE_2D)

    def _sweeps() -> float:
        start = time.perf_counter()
        sim.sweep(budget.n_sweeps, temperature=TC_SQUARE_2D)
        return (time.perf_counter() - start) / budget.n_sweeps

    def _measurements() -> float:
        start = time.perf_counter()
        for _ in range(budget.n_sweeps):
            sim.energy()
            sim.magnetization()
        return (time.perf_counter() - start) / budget.n_sweeps

    return {
        "sweep_seconds": median_of(_sweeps, budget.repeats),
        "measurement_seconds": median_of(_measurements, budget.repeats),
    }


def run_overhead(budget: Budget, context: Context) -> dict[str, Any]:
    b = budget.overhead
    rows = []
    for workload in WORKLOADS:
        context.log(f"[overhead] {workload.name}")
        rows.append(time_workload(workload, b))
    context.log("[overhead] decomposition of the reference workload")
    return {"rows": rows, "reference": decompose_reference(b)}


def run_correlation(budget: Budget, context: Context) -> dict[str, Any]:
    b = budget.correlation
    rows = []
    for size in b.sizes:
        context.log(f"[correlation] L={size}")
        sim = IsingSimulation(
            size, 1.0, 0.0, 0.0, 0.0, TIMING_SEED, "metropolis", "square"
        )
        sim.sweep(100, temperature=TC_SQUARE_2D)

        def _once() -> float:
            start = time.perf_counter()
            sim.correlation_function()
            return time.perf_counter() - start

        rows.append(
            {
                "size": size,
                "sites": int(sim.num_sites),
                "median_seconds": median_of(_once, b.repeats),
            }
        )
    return {"rows": rows}


def peapods_cases(b: PeapodsBudget) -> list[dict[str, Any]]:
    return [
        {
            "label": "Square",
            "lattice": "square",
            "geometry": None,
            "dim": 2,
            "size": b.size_2d,
            "temperature": TC_SQUARE_2D,
        },
        {
            "label": "Triangular",
            "lattice": "triangular",
            "geometry": "triangular",
            "dim": 2,
            "size": b.size_2d,
            "temperature": TC_TRIANGULAR_2D,
        },
        {
            "label": "Cubic",
            "lattice": "cubic",
            "geometry": None,
            "dim": 3,
            "size": b.cubic_size,
            "temperature": TC_CUBIC_3D,
        },
    ]


def matched_mcising(
    case: dict[str, Any], b: PeapodsBudget, seed: int
) -> tuple[float, float]:
    """``(mean E/site, timed seconds)`` with energy recorded every sweep."""
    sim = IsingSimulation(
        case["size"], 1.0, 0.0, 0.0, 0.0, seed, "metropolis", case["lattice"]
    )
    temperature = float(case["temperature"])
    sim.sweep(b.n_thermalization, temperature=temperature)
    start = time.perf_counter()
    out = sim.production_sweeps(
        b.n_sweeps, 1, temperature=temperature, store_configs=False
    )
    elapsed = time.perf_counter() - start
    energies = np.asarray(out["energies"], dtype=np.float64)
    return float(energies.mean()), elapsed


def matched_peapods(
    case: dict[str, Any], b: PeapodsBudget, seed: int
) -> tuple[float, float]:
    """peapods twin of :func:`matched_mcising` (same sweep, same convention).

    peapods reports ``+sum_bonds J s_i s_j / N`` (the sign of mcising's
    ``H = -J sum s_i s_j`` flipped), so the mean is negated; its Metropolis
    sweep is a sequential scan with one attempt per spin, like mcising's.
    A second ``sample()`` call continues from the current state, so the
    thermalization block is simply discarded. ``sequential=True`` keeps
    the run on one thread. peapods 0.2.0 seeds only through the private
    ``_sim.reset(seed=)`` (the public ``reset()`` takes no seed).
    """
    from peapods import Ising

    shape = (case["size"],) * int(case["dim"])
    kwargs: dict[str, Any] = {"couplings": "ferro"}
    if case["geometry"] is not None:
        kwargs["geometry"] = case["geometry"]
    model = Ising(shape, temperatures=np.array([case["temperature"]]), **kwargs)
    model._sim.reset(seed=seed)
    sample_kwargs: dict[str, Any] = {
        "sweep_mode": "metropolis",
        "warmup_ratio": 0.0,
        "sequential": True,
    }
    model.sample(n_sweeps=b.n_thermalization, **sample_kwargs)
    start = time.perf_counter()
    model.sample(n_sweeps=b.n_sweeps, **sample_kwargs)
    elapsed = time.perf_counter() - start
    return float(-model.energies_avg[0]), elapsed


def _mean_and_error(values: Sequence[float]) -> tuple[float, float | None]:
    if len(values) < 2:
        return float(values[0]), None
    return float(statistics.fmean(values)), float(
        statistics.stdev(values) / len(values) ** 0.5
    )


def run_peapods(budget: Budget, context: Context) -> dict[str, Any]:
    b = budget.peapods
    base: dict[str, Any] = {
        "n_thermalization": b.n_thermalization,
        "n_sweeps": b.n_sweeps,
        "seeds": list(b.seeds),
        "rows": [],
    }
    if context.skip_peapods:
        return {**base, "status": "unavailable", "reason": "skipped"}
    try:
        import peapods  # noqa: F401
    except ImportError:
        return {**base, "status": "unavailable", "reason": "not installed"}
    rows = []
    for case in peapods_cases(b):
        context.log(f"[peapods] {case['label']}: mcising")
        mc = [matched_mcising(case, b, seed) for seed in b.seeds]
        context.log(f"[peapods] {case['label']}: peapods")
        pp = [matched_peapods(case, b, seed) for seed in b.seeds]
        mc_energy, mc_error = _mean_and_error([e for e, _ in mc])
        pp_energy, pp_error = _mean_and_error([e for e, _ in pp])
        delta_percent = 100.0 * abs(mc_energy - pp_energy) / abs(mc_energy)
        rows.append(
            {
                **case,
                "mcising_energy": mc_energy,
                "mcising_energy_error": mc_error,
                "peapods_energy": pp_energy,
                "peapods_energy_error": pp_error,
                "delta_percent": float(delta_percent),
                "agreement": bool(delta_percent <= AGREEMENT_LIMIT_PERCENT),
                "mcising_median_seconds": float(statistics.median(t for _, t in mc)),
                "peapods_median_seconds": float(statistics.median(t for _, t in pp)),
            }
        )
    agreeing = sum(1 for row in rows if row["agreement"])
    if agreeing == len(rows):
        status = "matched"
    elif agreeing:
        status = "partial"
    else:
        status = "deferred"
    return {**base, "status": status, "rows": rows}


@dataclass(frozen=True)
class Section:
    name: str
    run: Callable[[Budget, Context], dict[str, Any]]


SECTIONS: Final[tuple[Section, ...]] = (
    Section("lattices", run_lattices),
    Section("baselines", run_baselines),
    Section("cluster", run_cluster),
    Section("scaling", run_scaling),
    Section("parallel", run_parallel),
    Section("overhead", run_overhead),
    Section("correlation", run_correlation),
    Section("peapods", run_peapods),
)
SECTION_NAMES: Final[tuple[str, ...]] = tuple(s.name for s in SECTIONS)


def run_all(
    budget: Budget,
    context: Context = Context(),
    sections: Sequence[str] = SECTION_NAMES,
) -> dict[str, Any]:
    """Measure the requested sections and return the results document."""
    unknown = [name for name in sections if name not in SECTION_NAMES]
    if unknown:
        raise BenchmarkError(f"unknown section(s): {', '.join(unknown)}")
    document = provenance(budget)
    start = time.perf_counter()
    for section in SECTIONS:
        if section.name not in sections:
            continue
        section_start = time.perf_counter()
        result = section.run(budget, context)
        result["generated_utc"] = _utc_now()
        result["elapsed_seconds"] = time.perf_counter() - section_start
        document["sections"][section.name] = result
    document["elapsed_seconds"] = time.perf_counter() - start
    return document


def merge_sections(existing: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    """Fold a partial run into an existing document (same schema and budget)."""
    if existing.get("schema_version") != fresh.get("schema_version"):
        raise BenchmarkError("schema_version differs; rerun every section")
    if existing.get("budget") != fresh.get("budget"):
        raise BenchmarkError("budget differs from the existing document")
    merged = {**existing, **{k: v for k, v in fresh.items() if k != "sections"}}
    merged["sections"] = {**existing.get("sections", {}), **fresh["sections"]}
    return merged


def load_document(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def write_document(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=1, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


# --- rendering ----------------------------------------------------------------


def _fmt_millions(x: float) -> str:
    return f"{x / 1e6:.0f}M"


def _fmt_int(x: float) -> str:
    return f"{x:,.0f}"


def _fmt_ratio(x: float) -> str:
    return f"{x:.1f}×"


def _fmt_seconds(x: float) -> str:
    return f"{x:.2f} s"


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1e3:.2f} ms"


def _fmt_us(seconds: float) -> str:
    return f"{seconds * 1e6:.2f} µs"


def _fmt_energy(value: float, error: float | None) -> str:
    return f"{value:.4f}" if error is None else f"{value:.4f} ± {error:.4f}"


def _updates_per_second(row: dict[str, Any]) -> float:
    return float(row["attempted_updates"]) / float(row["median_seconds"])


def _sweeps_per_second(row: dict[str, Any]) -> float:
    return float(row["n_sweeps"]) / float(row["median_seconds"])


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        section: dict[str, Any] = document["sections"][name]
    except KeyError as exc:
        raise BenchmarkError(f"results document has no {name!r} section") from exc
    return section


def _machine(document: dict[str, Any]) -> str:
    machine = document["machine"]
    cpu = machine.get("cpu") or "unknown CPU"
    count = machine.get("cpu_count")
    perf = machine.get("performance_cores")
    if count is None:
        return str(cpu)
    if perf:
        return f"{cpu} ({count} cores: {perf} performance + {count - perf} efficiency)"
    return f"{cpu} ({count} cores)"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _baseline_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = _section(document, "baselines")["rows"]
    return rows


def _baseline(
    document: dict[str, Any], implementation: str, size: int
) -> dict[str, Any]:
    for row in _baseline_rows(document):
        if row["implementation"] == implementation and row["size"] == size:
            return row
    raise BenchmarkError(f"no baseline row for {implementation!r} at L={size}")


def _matched_square_ratio(document: dict[str, Any]) -> float | None:
    """peapods speed ratio on the square lattice, when that row matched."""
    peapods = document["sections"].get("peapods")
    if peapods is None or peapods["status"] == "unavailable":
        return None
    for row in peapods["rows"]:
        if row["lattice"] == "square" and row["agreement"]:
            return float(row["peapods_median_seconds"]) / float(
                row["mcising_median_seconds"]
            )
    return None


def render_headline(document: dict[str, Any]) -> str:
    size = int(document["budget"]["baselines"]["size"])
    rust = _baseline(document, "mcising (Rust)", size)
    pure = _baseline(document, "Pure Python", size)
    numpy_row = _baseline(document, "NumPy (checkerboard)", size)
    rust_ups = _updates_per_second(rust)
    text = (
        f"On one core of an {_machine(document)}, mcising performs "
        f"**{_fmt_millions(rust_ups)} Metropolis spin updates per second** on a "
        f"{size}×{size} square lattice at Tc — "
        f"{_fmt_ratio(rust_ups / _updates_per_second(pure))} faster than pure "
        f"Python and {_fmt_ratio(rust_ups / _updates_per_second(numpy_row))} "
        f"faster than a NumPy checkerboard implementation of the same update"
    )
    ratio = _matched_square_ratio(document)
    if ratio is not None:
        text += (
            f", and {_fmt_ratio(ratio)} faster than peapods on a matched "
            f"workload (energy recorded every sweep on both sides)"
        )
    text += "."
    footer = (
        f"mcising {document['mcising_version']} (commit {document['git_commit']}), "
        f"Python {document['python']}, measured {document['generated_utc'][:10]}; "
        f"medians of repeated runs. Regenerate with `{REGENERATE_COMMAND}`."
    )
    return f"{text}\n\n{footer}"


def render_index_card(document: dict[str, Any]) -> str:
    size = int(document["budget"]["baselines"]["size"])
    rust = _baseline(document, "mcising (Rust)", size)
    pure = _baseline(document, "Pure Python", size)
    numpy_row = _baseline(document, "NumPy (checkerboard)", size)
    rust_ups = _updates_per_second(rust)
    cpu = document["machine"].get("cpu") or "one core"
    return (
        f"{_fmt_millions(rust_ups)} Metropolis spin updates per second on one core "
        f"({size}×{size} at Tc, {cpu}) — "
        f"{_fmt_ratio(rust_ups / _updates_per_second(pure))} faster than pure "
        f"Python, {_fmt_ratio(rust_ups / _updates_per_second(numpy_row))} faster "
        f"than a NumPy checkerboard."
    )


def render_lattices(document: dict[str, Any]) -> str:
    section = _section(document, "lattices")
    rows = [
        [
            str(row["label"]),
            _fmt_int(row["sites"]),
            _fmt_int(_sweeps_per_second(row)),
            _fmt_millions(_updates_per_second(row)),
        ]
        for row in section["rows"]
    ]
    n_sweeps = int(section["rows"][0]["n_sweeps"])
    caption = (
        f"Metropolis at each lattice's Tc (chain at T = {CHAIN_TEMPERATURE}), "
        f"{n_sweeps:,} timed sweeps after 100 warm-up sweeps, one thread, "
        f"{_machine(document)}."
    )
    return _table(["Lattice", "Sites", "Sweeps/s", "Spin updates/s"], rows) + (
        f"\n\n{caption}"
    )


def render_baselines(document: dict[str, Any]) -> str:
    rows = []
    for row in _baseline_rows(document):
        size = int(row["size"])
        rust_ups = _updates_per_second(_baseline(document, "mcising (Rust)", size))
        ups = _updates_per_second(row)
        ratio = (
            "—"
            if row["implementation"] == "mcising (Rust)"
            else (_fmt_ratio(rust_ups / ups))
        )
        rows.append(
            [
                str(row["implementation"]),
                f"{size}×{size}",
                _fmt_int(row["n_sweeps"]),
                _fmt_millions(ups) if ups >= 1e6 else _fmt_int(ups),
                ratio,
            ]
        )
    caption = (
        "Single-spin-flip Metropolis on the square lattice at Tc, one thread, "
        f"{_machine(document)}. Pure Python is a plain loop with precomputed "
        "neighbour and Boltzmann tables; the NumPy implementation updates the two "
        "checkerboard sublattices as whole arrays. Spin updates/s counts "
        "attempted flips; timed sweeps exclude 100 warm-up sweeps."
    )
    header = [
        "Implementation",
        "Lattice",
        "Timed sweeps",
        "Spin updates/s",
        "mcising is",
    ]
    return _table(header, rows) + f"\n\n{caption}"


def render_cluster(document: dict[str, Any]) -> str:
    section = _section(document, "cluster")
    rows = []
    for row in section["rows"]:
        us_per_sweep = float(row["median_seconds"]) / float(row["n_sweeps"])
        independent = us_per_sweep * 2.0 * float(row["tau_int_abs_magnetization"])
        rows.append(
            [
                str(row["label"]),
                _fmt_us(us_per_sweep),
                _fmt_millions(_updates_per_second(row)),
                f"{float(row['tau_int_energy']):.1f}",
                f"{float(row['tau_int_abs_magnetization']):.1f}",
                _fmt_us(independent),
            ]
        )
    first = section["rows"][0]
    wolff = next(r for r in section["rows"] if r["algorithm"] == "wolff")
    cluster_size = float(wolff["attempted_updates"]) / float(wolff["n_sweeps"])
    caption = (
        f"{first['size']}×{first['size']} square lattice at Tc, one thread, "
        f"{_machine(document)}. Attempted flips/s counts real work: a Metropolis "
        "sweep attempts every site once, a Swendsen-Wang sweep touches every "
        f"site, and one Wolff sweep is one cluster ({_fmt_int(cluster_size)} "
        "spins on average here). τ_int is the integrated autocorrelation time "
        f"in sweeps from a {int(first['series_sweeps']):,}-sweep series after "
        f"{int(document['budget']['cluster']['n_thermalization']):,} "
        "thermalization sweeps (blocking estimate). µs per independent sample = "
        "µs per sweep × 2 τ_int of the absolute magnetization, the slowest "
        "observable."
    )
    header = [
        "Algorithm",
        "µs per sweep",
        "Attempted flips/s",
        "τ_int (energy)",
        "τ_int (abs. magnetization)",
        "µs per independent sample",
    ]
    return _table(header, rows) + f"\n\n{caption}"


def render_scaling(document: dict[str, Any]) -> str:
    section = _section(document, "scaling")
    rows = [
        [
            str(row["size"]),
            _fmt_int(row["sites"]),
            _fmt_int(row["n_sweeps"]),
            _fmt_millions(_updates_per_second(row)),
            _fmt_us(float(row["median_seconds"]) / float(row["n_sweeps"])),
        ]
        for row in section["rows"]
    ]
    caption = (
        "Metropolis on the square lattice at Tc, one thread, "
        f"{_machine(document)}; timed sweeps exclude 100 warm-up sweeps."
    )
    header = ["L", "Sites", "Timed sweeps", "Spin updates/s", "µs per sweep"]
    return _table(header, rows) + f"\n\n{caption}"


_MODE_LABELS: Final = {
    "cooldown": "Cooldown",
    "independent": "Independent",
    "parallel_tempering": "Parallel tempering",
}


def render_parallel(document: dict[str, Any]) -> str:
    section = _section(document, "parallel")
    b = document["budget"]["parallel"]
    cooldown = next(r for r in section["rows"] if r["mode"] == "cooldown")
    reference = float(cooldown["median_seconds"])
    rows = [
        [
            _MODE_LABELS[str(row["mode"])],
            str(row["threads"]),
            _fmt_seconds(float(row["median_seconds"])),
            _fmt_ratio(reference / float(row["median_seconds"])),
        ]
        for row in section["rows"]
    ]
    caption = (
        f"Metropolis, {b['size']}×{b['size']} square lattice, "
        f"{b['n_temperatures']} temperatures from {b['t_max']} to {b['t_min']}, "
        f"{b['n_thermalization']:,} thermalization + {b['n_sweeps']:,} production "
        f"sweeps per temperature (measured every {b['measurement_interval']}), "
        f"{_machine(document)}; medians of {b['repeats']} runs. The independent "
        "and parallel-tempering rows each run in a fresh process with "
        "`RAYON_NUM_THREADS` set to the thread count; the cooldown mode is "
        "single-threaded by construction."
    )
    header = ["Mode", "Threads", "Wall time", "Speed-up vs cooldown"]
    return _table(header, rows) + f"\n\n{caption}"


def render_overhead(document: dict[str, Any]) -> str:
    section = _section(document, "overhead")
    b = document["budget"]["overhead"]
    rows = [
        [
            str(row["name"]),
            _fmt_ms(float(row["min_seconds"])),
            _fmt_ms(float(row["median_seconds"])),
            _fmt_us(float(row["median_seconds"]) / float(row["n_measurements"])),
        ]
        for row in section["rows"]
    ]
    ref = section["reference"]
    reference_row = section["rows"][0]
    accounted = (
        float(ref["sweep_seconds"]) + float(ref["measurement_seconds"])
    ) * float(b["n_sweeps"])
    remainder = float(reference_row["median_seconds"]) - accounted
    caption = (
        f"`Simulation.run()` end to end: {b['lattice_size']}×{b['lattice_size']} "
        f"square lattice at Tc, {b['n_thermalization']} annealing + "
        f"{b['n_sweeps']} production sweeps measured at every sweep, "
        f"{_machine(document)}; minimum and median of {b['repeats']} runs, a "
        "fresh `Simulation` per run. In the first row a bare Metropolis sweep "
        f"costs {_fmt_us(float(ref['sweep_seconds']))}, an energy + "
        f"magnetization measurement {_fmt_us(float(ref['measurement_seconds']))}, "
        f"and the annealing ramp plus fixed overhead the remaining "
        f"{_fmt_ms(remainder)}."
    )
    header = ["Workload", "min", "median", "µs per measurement"]
    return _table(header, rows) + f"\n\n{caption}"


def render_correlation(document: dict[str, Any]) -> str:
    section = _section(document, "correlation")
    rows = [
        [
            f"{row['size']}×{row['size']}",
            _fmt_int(row["sites"]),
            _fmt_ms(float(row["median_seconds"])),
        ]
        for row in section["rows"]
    ]
    caption = (
        "One `correlation_function()` evaluation (the O(N²) pair sum) on the "
        f"square lattice, {_machine(document)}; medians of "
        f"{document['budget']['correlation']['repeats']} evaluations."
    )
    return _table(["Lattice", "Sites", "Per evaluation"], rows) + f"\n\n{caption}"


_CLUSTER_NOTE: Final = (
    "Wolff and Swendsen-Wang are not compared: peapods interleaves cluster "
    "updates with Metropolis sweeps rather than running cluster-only sweeps, "
    "so no peapods workload matches an mcising cluster sweep."
)


def render_peapods(document: dict[str, Any]) -> str:
    section = _section(document, "peapods")
    if section["status"] == "unavailable":
        return (
            "The peapods comparison was not run "
            f"({section.get('reason', 'unavailable')}); install it with "
            "`uv sync --group benchmark` and rerun the `peapods` section."
        )
    b = document["budget"]["peapods"]
    version = document.get("peapods_version") or "unknown version"
    intro = (
        f"Matched physics against [peapods](https://github.com/PeaBrane/peapods) "
        f"{version} (Rust/PyO3): the same Hamiltonian (H = −J Σ s_i s_j, J = 1, "
        "energies per site, bond counted once), the same Metropolis sweep (a "
        "sequential scan with one attempt per site), the same temperature (Tc), "
        f"{b['n_thermalization']:,} thermalization sweeps, then "
        f"{b['n_sweeps']:,} timed sweeps with the energy recorded every sweep on "
        f"both sides, one thread, {len(b['seeds'])} seeds per side, "
        f"{_machine(document)}. peapods reports +Σ J s_i s_j / N, so its sign "
        "is flipped before comparison. A row is published only when the two "
        f"mean energies agree within {AGREEMENT_LIMIT_PERCENT} %."
    )
    n_sweeps = float(b["n_sweeps"])
    published = [row for row in section["rows"] if row["agreement"]]
    dropped = [row for row in section["rows"] if not row["agreement"]]
    parts = [intro]
    if published:
        rows = [
            [
                f"{row['label']} {_lattice_dims(row)}",
                _fmt_energy(row["mcising_energy"], row["mcising_energy_error"]),
                _fmt_energy(row["peapods_energy"], row["peapods_energy_error"]),
                f"{float(row['delta_percent']):.2f} %",
                _fmt_int(n_sweeps / float(row["mcising_median_seconds"])),
                _fmt_int(n_sweeps / float(row["peapods_median_seconds"])),
                _fmt_ratio(
                    float(row["peapods_median_seconds"])
                    / float(row["mcising_median_seconds"])
                ),
            ]
            for row in published
        ]
        header = [
            "Lattice",
            "E/site mcising",
            "E/site peapods",
            "Δ",
            "Sweeps/s mcising",
            "Sweeps/s peapods",
            "mcising is",
        ]
        parts.append(_table(header, rows))
    if dropped:
        details = "; ".join(
            f"{row['label'].lower()} {_lattice_dims(row)}: "
            f"{_fmt_energy(row['mcising_energy'], row['mcising_energy_error'])} vs "
            f"{_fmt_energy(row['peapods_energy'], row['peapods_energy_error'])} "
            f"({float(row['delta_percent']):.2f} %)"
            for row in dropped
        )
        names = ", ".join(row["label"].lower() for row in dropped)
        parts.append(
            "Not published: the mean energies disagree by more than "
            f"{AGREEMENT_LIMIT_PERCENT} % on the {names} "
            f"lattice{'s' if len(dropped) > 1 else ''} ({details}), so the two "
            "libraries were not simulating the same system there and a speed "
            "ratio would be meaningless. The cross-library comparison on "
            f"{'those lattices' if len(dropped) > 1 else 'that lattice'} is "
            "deferred until the discrepancy is understood."
        )
    parts.append(_CLUSTER_NOTE)
    return "\n\n".join(parts)


def _lattice_dims(row: dict[str, Any]) -> str:
    size = int(row["size"])
    return "×".join([str(size)] * int(row["dim"]))


RENDERERS: Final[dict[str, Callable[[dict[str, Any]], str]]] = {
    "headline": render_headline,
    "index-card": render_index_card,
    "lattices": render_lattices,
    "baselines": render_baselines,
    "cluster": render_cluster,
    "scaling": render_scaling,
    "parallel": render_parallel,
    "overhead": render_overhead,
    "correlation": render_correlation,
    "peapods": render_peapods,
}


def render_section(name: str, document: dict[str, Any]) -> str:
    try:
        renderer = RENDERERS[name]
    except KeyError as exc:
        raise BenchmarkError(f"no renderer for block {name!r}") from exc
    return renderer(document)


# --- docs blocks --------------------------------------------------------------


def markers(section: str) -> tuple[str, str]:
    return (
        f"<!-- benchmarks:{section}:begin -->",
        f"<!-- benchmarks:{section}:end -->",
    )


def _locate(lines: Sequence[str], path: Path, section: str) -> tuple[int, int, str]:
    begin, end = markers(section)
    begins = [i for i, line in enumerate(lines) if line.strip() == begin]
    ends = [i for i, line in enumerate(lines) if line.strip() == end]
    if len(begins) != 1 or len(ends) != 1 or ends[0] <= begins[0]:
        msg = f"{path} must contain exactly one {begin} / {end} pair, in order"
        raise BenchmarkError(msg)
    marker_line = lines[begins[0]]
    indent = marker_line[: len(marker_line) - len(marker_line.lstrip())]
    return begins[0], ends[0], indent


def read_block(path: Path, section: str) -> str:
    """The block currently between the section's markers, de-indented."""
    lines = path.read_text(encoding="utf-8").split("\n")
    begin, end, indent = _locate(lines, path, section)
    block = []
    for line in lines[begin + 1 : end]:
        if not line:
            block.append("")
        elif line.startswith(indent):
            block.append(line[len(indent) :])
        else:
            raise BenchmarkError(f"{path}: block line is not indented like its marker")
    return "\n".join(block)


def write_block(path: Path, section: str, block: str) -> None:
    """Replace the block between the section's markers, keeping their indent.

    Empty lines stay empty (no trailing whitespace) so the pre-commit hooks
    leave the rendered page alone.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    begin, end, indent = _locate(lines, path, section)
    body = [indent + line if line else "" for line in block.split("\n")]
    new_lines = lines[: begin + 1] + body + lines[end:]
    path.write_text("\n".join(new_lines), encoding="utf-8")


DOC_BLOCKS: Final[tuple[tuple[Path, str], ...]] = (
    (REPO_ROOT / "README.md", "headline"),
    (REPO_ROOT / "README.md", "baselines"),
    (REPO_ROOT / "README.md", "peapods"),
    (REPO_ROOT / "docs" / "index.md", "index-card"),
    (REPO_ROOT / "docs" / "advanced" / "performance.md", "headline"),
    (REPO_ROOT / "docs" / "advanced" / "performance.md", "lattices"),
    (REPO_ROOT / "docs" / "advanced" / "performance.md", "baselines"),
    (REPO_ROOT / "docs" / "advanced" / "performance.md", "peapods"),
    (REPO_ROOT / "docs" / "advanced" / "performance.md", "cluster"),
    (REPO_ROOT / "docs" / "advanced" / "performance.md", "scaling"),
    (REPO_ROOT / "docs" / "advanced" / "performance.md", "parallel"),
    (REPO_ROOT / "docs" / "advanced" / "performance.md", "overhead"),
    (REPO_ROOT / "docs" / "advanced" / "performance.md", "correlation"),
    (REPO_ROOT / "docs" / "tutorial" / "cluster-algorithms.md", "cluster"),
    (REPO_ROOT / "docs" / "tutorial" / "parallel-execution.md", "parallel"),
    (REPO_ROOT / "docs" / "guide" / "configuration.md", "correlation"),
)


def write_docs(
    document: dict[str, Any],
    blocks: Sequence[tuple[Path, str]] = DOC_BLOCKS,
) -> None:
    if document["budget"]["quick"]:
        raise BenchmarkError("refusing to write docs from a --quick document")
    for path, section in blocks:
        write_block(path, section, render_section(section, document))


def check_docs(
    document: dict[str, Any],
    blocks: Sequence[tuple[Path, str]] = DOC_BLOCKS,
) -> list[str]:
    """Names of the ``path:section`` blocks that differ from the render."""
    stale = []
    for path, section in blocks:
        if read_block(path, section) != render_section(section, document):
            stale.append(f"{_display_path(path)}:{section}")
    return stale


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


# --- CLI ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--quick", action="store_true", help="tiny budget (the test suite's setting)"
    )
    parser.add_argument(
        "--sections",
        default=",".join(SECTION_NAMES),
        help="comma-separated subset of "
        + ", ".join(SECTION_NAMES)
        + " (merged into the existing results file)",
    )
    parser.add_argument(
        "--skip-peapods", action="store_true", help="do not run the peapods section"
    )
    parser.add_argument(
        "--output", type=Path, default=RESULTS_PATH, help="results JSON path"
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="skip the measurements and render an existing results JSON",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--write-docs",
        action="store_true",
        help="rewrite every marker-delimited block listed in DOC_BLOCKS",
    )
    action.add_argument(
        "--check",
        action="store_true",
        help="verify the docs blocks against the results (no measurements)",
    )
    args = parser.parse_args(argv)

    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    wanted = [name.strip() for name in args.sections.split(",") if name.strip()]
    if args.from_json is not None or args.check:
        source = args.from_json if args.from_json is not None else args.output
        document = load_document(source)
    else:
        budget = QUICK_BUDGET if args.quick else FULL_BUDGET
        context = Context(log=print, skip_peapods=args.skip_peapods)
        document = run_all(budget, context, wanted)
        if set(wanted) != set(SECTION_NAMES) and args.output.exists():
            document = merge_sections(load_document(args.output), document)
        write_document(args.output, document)
        print(f"\nwrote {args.output} ({document['elapsed_seconds']:.0f} s)")

    for name in wanted:
        if name in document["sections"]:
            console.print(Markdown(f"### {name}\n\n{render_section(name, document)}"))

    if args.check:
        stale = check_docs(document)
        if stale:
            print("stale docs blocks: " + ", ".join(stale))
            print(f"regenerate with: {REGENERATE_COMMAND}")
            return 1
        print(f"all {len(DOC_BLOCKS)} docs blocks match {args.output}")
    if args.write_docs:
        write_docs(document)
        print(f"updated {len(DOC_BLOCKS)} docs blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
