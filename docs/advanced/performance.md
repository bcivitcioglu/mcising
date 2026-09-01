# Performance

<!-- benchmarks:headline:begin -->
On one core of an Apple M4 (10 cores: 4 performance + 6 efficiency), mcising performs **351M Metropolis spin updates per second** on a 32×32 square lattice at Tc — 140.4× faster than pure Python and 15.0× faster than a NumPy checkerboard implementation of the same update, and 2.4× faster than peapods on a matched workload (energy recorded every sweep on both sides).

mcising 0.29.0 (commit 2e3548a), Python 3.12.11, measured 2026-09-01; medians of repeated runs. Regenerate with `uv run --group benchmark python benchmarks/run_all.py --write-docs`.
<!-- benchmarks:headline:end -->

Every table on this page is rendered by `benchmarks/run_all.py` from the
committed `benchmarks/results.json`, and `tests/test_run_all.py` fails when
a page and the JSON disagree. Timings are medians of repeated runs on one
machine: expect different absolute numbers elsewhere and the same ratios
within noise.

## Metropolis across lattices

<!-- benchmarks:lattices:begin -->
| Lattice | Sites | Sweeps/s | Spin updates/s |
|---|---|---|---|
| Square 32x32 | 1,024 | 342,361 | 351M |
| Triangular 32x32 | 1,024 | 295,711 | 303M |
| Honeycomb 32x32 | 2,048 | 201,524 | 413M |
| Chain (1024) | 1,024 | 405,197 | 415M |
| Cubic 16^3 | 4,096 | 44,229 | 181M |

Metropolis at each lattice's Tc (chain at T = 1.0), 10,000 timed sweeps after 100 warm-up sweeps, one thread, Apple M4 (10 cores: 4 performance + 6 efficiency).
<!-- benchmarks:lattices:end -->

## Against pure Python and NumPy

<!-- benchmarks:baselines:begin -->
| Implementation | Lattice | Timed sweeps | Spin updates/s | mcising is |
|---|---|---|---|---|
| Pure Python | 32×32 | 200 | 2M | 140.4× |
| NumPy (checkerboard) | 32×32 | 1,000 | 23M | 15.0× |
| mcising (Rust) | 32×32 | 10,000 | 351M | — |
| NumPy (checkerboard) | 128×128 | 200 | 69M | 4.7× |
| mcising (Rust) | 128×128 | 1,000 | 324M | — |

Single-spin-flip Metropolis on the square lattice at Tc, one thread, Apple M4 (10 cores: 4 performance + 6 efficiency). Pure Python is a plain loop with precomputed neighbour and Boltzmann tables; the NumPy implementation updates the two checkerboard sublattices as whole arrays. Spin updates/s counts attempted flips; timed sweeps exclude 100 warm-up sweeps.
<!-- benchmarks:baselines:end -->

## Against peapods

<!-- benchmarks:peapods:begin -->
Matched physics against [peapods](https://github.com/PeaBrane/peapods) 0.2.0 (Rust/PyO3): the same Hamiltonian (H = −J Σ s_i s_j, J = 1, energies per site, bond counted once), the same Metropolis sweep (a sequential scan with one attempt per site), the same temperature (Tc), 5,000 thermalization sweeps, then 100,000 timed sweeps with the energy recorded every sweep on both sides, one thread, 3 seeds per side, Apple M4 (10 cores: 4 performance + 6 efficiency). peapods reports +Σ J s_i s_j / N, so its sign is flipped before comparison. A row is published only when the two mean energies agree within 0.5 %.

| Lattice | E/site mcising | E/site peapods | Δ | Sweeps/s mcising | Sweeps/s peapods | mcising is |
|---|---|---|---|---|---|---|
| Square 32×32 | -1.4325 ± 0.0018 | -1.4346 ± 0.0007 | 0.15 % | 249,160 | 104,786 | 2.4× |
| Triangular 32×32 | -2.0305 ± 0.0012 | -2.0320 ± 0.0014 | 0.07 % | 203,737 | 86,013 | 2.4× |
| Cubic 16×16×16 | -1.0346 ± 0.0010 | -1.0347 ± 0.0002 | 0.01 % | 34,196 | 15,677 | 2.2× |

Wolff and Swendsen-Wang are not compared: peapods interleaves cluster updates with Metropolis sweeps rather than running cluster-only sweeps, so no peapods workload matches an mcising cluster sweep.
<!-- benchmarks:peapods:end -->

## Cluster algorithms: cost per independent sample

Throughput alone misleads for cluster algorithms. A Wolff sweep is one
cluster and a Swendsen-Wang sweep rebuilds every bond, so their attempted
flips per second are lower than Metropolis — but both decorrelate the
magnetization within a few sweeps near Tc, where Metropolis needs many.
The last column is the number to compare.

<!-- benchmarks:cluster:begin -->
| Algorithm | µs per sweep | Attempted flips/s | τ_int (energy) | τ_int (abs. magnetization) | µs per independent sample |
|---|---|---|---|---|---|
| Metropolis | 2.93 µs | 349M | 10.8 | 29.9 | 175.54 µs |
| Wolff | 8.10 µs | 59M | 4.3 | 3.9 | 63.15 µs |
| Swendsen-Wang | 16.96 µs | 60M | 4.5 | 3.7 | 124.26 µs |

32×32 square lattice at Tc, one thread, Apple M4 (10 cores: 4 performance + 6 efficiency). Attempted flips/s counts real work: a Metropolis sweep attempts every site once, a Swendsen-Wang sweep touches every site, and one Wolff sweep is one cluster (475 spins on average here). τ_int is the integrated autocorrelation time in sweeps from a 100,000-sweep series after 5,000 thermalization sweeps (blocking estimate). µs per independent sample = µs per sweep × 2 τ_int of the absolute magnetization, the slowest observable.
<!-- benchmarks:cluster:end -->

## Scaling with lattice size

<!-- benchmarks:scaling:begin -->
| L | Sites | Timed sweeps | Spin updates/s | µs per sweep |
|---|---|---|---|---|
| 8 | 64 | 400,000 | 349M | 0.18 µs |
| 16 | 256 | 100,000 | 350M | 0.73 µs |
| 32 | 1,024 | 25,000 | 347M | 2.96 µs |
| 64 | 4,096 | 6,000 | 338M | 12.12 µs |
| 128 | 16,384 | 1,500 | 326M | 50.31 µs |
| 256 | 65,536 | 400 | 319M | 205.44 µs |

Metropolis on the square lattice at Tc, one thread, Apple M4 (10 cores: 4 performance + 6 efficiency); timed sweeps exclude 100 warm-up sweeps.
<!-- benchmarks:scaling:end -->

## Parallel execution

<!-- benchmarks:parallel:begin -->
| Mode | Threads | Wall time | Speed-up vs cooldown |
|---|---|---|---|
| Cooldown | 1 | 3.01 s | 1.0× |
| Independent | 1 | 3.03 s | 1.0× |
| Independent | 2 | 1.62 s | 1.9× |
| Independent | 4 | 0.84 s | 3.6× |
| Independent | 8 | 0.57 s | 5.3× |
| Independent | 10 | 0.54 s | 5.6× |
| Parallel tempering | 10 | 1.47 s | 2.0× |

Metropolis, 128×128 square lattice, 20 temperatures from 3.5 to 1.5, 500 thermalization + 2,000 production sweeps per temperature (measured every 10), Apple M4 (10 cores: 4 performance + 6 efficiency); medians of 3 runs. The independent and parallel-tempering rows each run in a fresh process with `RAYON_NUM_THREADS` set to the thread count; the cooldown mode is single-threaded by construction.
<!-- benchmarks:parallel:end -->

## Why it's fast

### 15 auto-selected Metropolis strategies

Based on which couplings are active (J1, J2, J3, H), mcising selects the optimal lookup table at construction time. Each strategy has its own dedicated sweep method — no branching in the inner loop.

### Monomorphization

The `McAlgorithm::sweep` method is generic over lattice type. LLVM compiles a separate version for each lattice, allowing loop unrolling and inlining of neighbor accesses.

### Vec-based lookup tables

Acceptance probabilities are precomputed in flat `Vec<f64>` arrays sized by coordination number. One array index per flip — no exp() calls in the hot loop.

### Rayon parallelism

Independent and parallel tempering modes use Rayon's thread pool. Each temperature gets its own simulation instance on a separate core. No shared mutable state, no lock contention.

## Measurement overhead

`Simulation.run()` pays for more than the sweeps: the per-sweep energy and
magnetization, the optional spin snapshot and the Python/FFI plumbing
around them. The workload below — a lattice large enough that a sweep is
not free, measured at every sweep — makes that cost visible next to the
sweep itself.

<!-- benchmarks:overhead:begin -->
| Workload | min | median | µs per measurement |
|---|---|---|---|
| Metropolis, configurations stored | 5.01 ms | 5.08 ms | 25.40 µs |
| Metropolis, configurations off | 5.00 ms | 5.06 ms | 25.29 µs |
| Wolff, configurations stored | 0.93 ms | 0.94 ms | 4.70 µs |
| Swendsen-Wang, configurations stored | 20.11 ms | 20.30 ms | 101.51 µs |

`Simulation.run()` end to end: 64×64 square lattice at Tc, 100 annealing + 200 production sweeps measured at every sweep, Apple M4 (10 cores: 4 performance + 6 efficiency); minimum and median of 20 runs, a fresh `Simulation` per run. In the first row a bare Metropolis sweep costs 12.12 µs, an energy + magnetization measurement 4.22 µs, and the annealing ramp plus fixed overhead the remaining 1.81 ms.
<!-- benchmarks:overhead:end -->

Version 0.29.0 halved this workload. The cooldown path makes two Rust calls
per temperature instead of one per sweep and several per measurement, and
the per-measurement energy skips coupling shells that are exactly zero and
sums dyadic-exact couplings in integer arithmetic, bit-identical to the
serial sum. The before-and-after numbers are in the 0.29.0 changelog entry.

Enabling `compute_correlation` adds a full O(N²) pair sum per evaluation;
`correlation_interval` sets how many measurements apart it runs.

<!-- benchmarks:correlation:begin -->
| Lattice | Sites | Per evaluation |
|---|---|---|
| 16×16 | 256 | 0.35 ms |
| 32×32 | 1,024 | 7.04 ms |
| 64×64 | 4,096 | 134.98 ms |

One `correlation_function()` evaluation (the O(N²) pair sum) on the square lattice, Apple M4 (10 cores: 4 performance + 6 efficiency); medians of 5 evaluations.
<!-- benchmarks:correlation:end -->

## Run your own benchmarks

```bash
# Every published table, into README.md and the docs
# (the peapods rows need the benchmark dependency group)
uv run --group benchmark python benchmarks/run_all.py --write-docs

# One section, merged into benchmarks/results.json
uv run python benchmarks/run_all.py --sections scaling --write-docs

# Quick interactive numbers across lattices, algorithms and couplings
mcising benchmark
mcising benchmark -L 64 --sweeps 50000
```
