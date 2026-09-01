# Performance

mcising's Rust core achieves 268M spin updates/sec on a single core — 3.4x faster than peapods and 430x faster than pure Python.

## Benchmark results

Measured on MacBook Pro 14-inch (2023, Apple M2 Pro, 32 GB), 10,000 sweeps:

### Metropolis across lattices

| Lattice | Sites | Updates/sec |
|---|---|---|
| Square 32x32 | 1,024 | 268M |
| Triangular 32x32 | 1,024 | 221M |
| Honeycomb 32x32 | 2,048 | 304M |
| Chain (1024) | 1,024 | 349M |
| Cubic 16^3 | 4,096 | 145M |

### vs peapods (Rust/PyO3)

| Benchmark | mcising | peapods | Speedup |
|---|---|---|---|
| Metropolis: Square | 269M | 78M | 3.4x |
| Metropolis: Triangular | 223M | 65M | 3.4x |
| Metropolis: Cubic | 147M | 50M | 2.9x |
| Wolff: Square | 100M[^wolff-work] | 30M | 3.3x |
| Swendsen-Wang: Square | 48M | 18M | 2.7x |

[^wolff-work]: Measured before 1.0, when one Wolff sweep (one cluster)
    was counted as `num_sites` updates on both sides of the comparison
    — the *ratio* is meaningful, the absolute Wolff numbers overstate
    real flips by ~`N / ⟨cluster size⟩`. The mcising benchmark now
    counts real attempted flips; re-measurement lands with P17.

Reproduce with [`benchmarks/compare_peapods.py`](https://github.com/bcivitcioglu/mcising/blob/master/benchmarks/compare_peapods.py).

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

`benchmarks/measurement_overhead.py` times `Simulation.run()` end to end on
a 64×64 Metropolis lattice at T = 2.269 with 100 annealing sweeps and 200
production sweeps measured at every sweep — the workload where the cost of
*measuring* (energy, magnetization, the spin snapshot and the plumbing
around them) is most visible next to the sweep itself. Medians over 20
runs (Apple M2 Pro, release build):

| Workload (L=64, 100+200 sweeps, interval 1) | v0.28.0 | unreleased | speed-up |
|---|---|---|---|
| Metropolis, configurations stored | 10.15 ms | 5.06 ms | 2.0× |
| Metropolis, configurations off | 10.05 ms | 5.03 ms | 2.0× |
| Wolff, configurations stored | 6.31 ms | 0.94 ms | 6.7× |
| Swendsen–Wang, configurations stored | 24.97 ms | 19.9 ms | 1.3× |

Two changes account for it. The cooldown path now makes two Rust calls per
temperature (`anneal` for the ramp, `production_sweeps` for the whole
production block) instead of one per sweep and several per measurement; on
its own that is worth only ~5 % here — a Metropolis sweep of 4096 spins
takes 12 µs, so a microsecond of call overhead was never the bottleneck.
The rest is the per-measurement energy: it used to walk the J2, J3 and
field shells even when their couplings are zero, in one serial
floating-point chain (29.5 µs — two and a half sweeps' worth), and now
skips zero shells and sums dyadic-exact couplings (1, 0.5, 0.25, …) in
integer arithmetic (4.2 µs), bit-identical to the old result. What remains
is the physics: 47 % Metropolis sweeps, 31 % the annealing ramp, 17 %
energy measurements.

Enabling `compute_correlation` adds a full O(N²) pair sum per evaluation
(0.3 ms at 16², 7 ms at 32², 130 ms at 64²); `correlation_interval` sets
how many measurements apart it runs.

## Run your own benchmarks

```bash
# Full mcising benchmark
mcising benchmark

# Custom parameters
mcising benchmark -L 64 --sweeps 50000
```
