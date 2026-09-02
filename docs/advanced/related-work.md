# Related Work

Where mcising sits among the classical spin-model Monte Carlo codes a researcher is likely to consider. The table compares **documented scope**, not speed — the only performance comparison mcising publishes is the matched-physics benchmark against peapods on the [performance page](performance.md), regenerated from a committed script. Every cell was checked against the project's README or documentation on 2026-09-02; "not documented" means the source consulted does not say, not that the feature is absent.

## Feature comparison

| | mcising | [peapods](https://github.com/PeaBrane/peapods) | [ALPS](https://alps.comp-phys.org/) / [ALPSCore](https://github.com/ALPSCore/ALPSCore) | [IsingModels.jl](https://github.com/cossio/IsingModels.jl) | [SpinMC.jl](https://github.com/fbuessen/SpinMC.jl) |
|---|---|---|---|---|---|
| Language | Python API, Rust core (PyO3) | Python API, Rust core (PyO3) | C++ framework; Python tools (`pyalps`) | Julia | Julia |
| Spin model | Ising | Ising (incl. spin glasses) | Ising, XY, Heisenberg (`spinmc` application) | Ising | Classical O(3) Heisenberg |
| Lattices | square, triangular, honeycomb, cubic, chain (periodic) | periodic Bravais lattices: hypercubic in any dimension, triangular, custom neighbour offsets | XML lattice library (arbitrary graphs) | 2D square grid | arbitrary unit cells, any dimension |
| Couplings beyond nearest neighbour | named $J_2$, $J_3$ shells per lattice; external field $h$ | per-bond coupling arrays (uniform, ±J, Gaussian) over custom offsets; no field documented | per-bond-type couplings in the lattice/model XML (not verified here) | not documented | general interaction matrices, further neighbours, external field |
| Update algorithms | Metropolis, Wolff, Swendsen-Wang | Metropolis, Gibbs, Wolff, Swendsen-Wang (interleaved with single-spin sweeps), replica overlap-cluster moves | local and cluster updates | Metropolis, Wolff (from the source tree) | Metropolis |
| Parallel execution | independent temperatures (Rayon), parallel tempering | parallel tempering, replicas, disorder averaging | not documented for `spinmc` | not documented | parallel tempering via MPI |
| Error analysis | blocking and jackknife errors on every observable, $\tau_\mathrm{int}$, adaptive thermalization (MSER + Sokal) | autocorrelation diagnostics, Binder cumulant | ALPSCore accumulators (binning) | not documented | BinningAnalysis.jl |
| Output | HDF5 with provenance (config, seed, version, commit), JSON summary, checkpoint/resume | `.npz` | HDF5 / XML | not documented | HDF5 |
| Install | `pip install mcising` (wheels for Linux, macOS, Windows) | `pip install peapods` | binaries, source or Spack | `Pkg.add("IsingModels")` | `Pkg.add` from GitHub |
| License | MIT | MIT | see repository | MIT | MIT |
| Reference | this documentation; [CITATION.cff](https://github.com/bcivitcioglu/mcising/blob/master/CITATION.cff) | Pei, [arXiv:2602.19045](https://arxiv.org/abs/2602.19045) | Bauer *et al.*, [J. Stat. Mech. (2011) P05001](https://doi.org/10.1088/1742-5468/2011/05/P05001); Gaenko *et al.*, [Comput. Phys. Commun. 213, 235 (2017)](https://doi.org/10.1016/j.cpc.2016.12.009) | Fernandez-de-Cossio-Diaz, Cocco & Monasson, [Phys. Rev. X 13, 021003 (2023)](https://doi.org/10.1103/PhysRevX.13.021003) | not documented |

## How mcising differs

- **Frustration as a first-class, tested feature.** $J_2$ and $J_3$ are named shells with their own neighbour tables on every lattice (including the non-Bravais honeycomb), and antiferromagnetic couplings run through the same exact-enumeration and ground-state tests as the ferromagnet. The [frustration tutorial](../tutorial/frustrated-magnetism.md) reproduces the stripe phase of the $J_1$-$J_2$ model with a committed script.
- **Quoted uncertainties by default.** Every observable of a run carries a blocking or jackknife error and an integrated autocorrelation time, and the [physics page](physics.md) reports the measured critical temperatures of four lattices with statistical and finite-size errors against the exact values.
- **Provenance in the data file.** An HDF5 result reloads with its configuration, seed, execution mode, package version and git commit, and a run can be checkpointed and resumed.
- **A single pip install.** No C++ toolchain, XML lattice definitions or Julia environment; wheels are built for three platforms and four Python versions.

What the others offer that mcising does not: peapods covers disordered couplings, spin-glass replica moves and hypercubic lattices in any dimension; ALPS covers XY and Heisenberg spins, quantum models and arbitrary graphs; SpinMC.jl covers continuous O(3) spins with general interaction matrices and MPI-scale tempering. Wang-Landau sampling, cluster updates for antiferromagnets (via sublattice mapping) and further lattices (kagome, odd-sized triangular and honeycomb) are on mcising's post-1.0 list.
