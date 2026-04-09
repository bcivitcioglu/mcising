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
| Wolff: Square | 100M | 30M | 3.3x |
| Swendsen-Wang: Square | 48M | 18M | 2.7x |

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

## Run your own benchmarks

```bash
# Full mcising benchmark
mcising benchmark

# Custom parameters
mcising benchmark -L 64 --sweeps 50000
```
