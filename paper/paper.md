---
title: 'mcising: tested Ising Monte Carlo for frustrated magnets with a Rust core'
tags:
  - Python
  - Rust
  - Monte Carlo
  - Ising model
  - statistical mechanics
  - frustrated magnetism
  - critical phenomena
authors:
  - name: Burak Çivitcioğlu
    orcid: 0000-0001-8433-657X
    affiliation: 1
affiliations:
  - name: "aivancity School of AI & Data for Business & Society, Villejuif, France"
    index: 1
date: 7 September 2026
bibliography: paper.bib
---

# Summary

The Ising model [@ising1925] is one of the simplest models with a phase
transition and an exact solution in two dimensions [@onsager1944]. With
competing further-neighbor couplings it also shows frustration, which remains
actively studied, including with machine-learning methods. Monte Carlo
simulation is the standard numerical tool for the model.

`mcising` is a Python package with a Rust core for classical Ising Monte Carlo
on square, triangular, honeycomb, simple-cubic and chain lattices, with
first-, second- and third-neighbor couplings $J_1$, $J_2$, $J_3$ of either
sign and an external field. It provides Metropolis, Wolff and Swendsen–Wang
updates, three execution modes (sequential cool-down, independent temperatures
in parallel, and parallel tempering), adaptive thermalization, blocking and
jackknife error estimates with integrated autocorrelation times for every
observable, and HDF5 output carrying the configuration, seed, package version
and commit. The Rust core is validated against exact enumeration of small
systems and exact results on large ones: Onsager's solution of the square
lattice and the critical temperatures of four lattices, with antiferromagnetic
and competing couplings passing the same tests as the ferromagnet. `mcising`
installs with `pip` and `uv` and is documented at <https://bcivitcioglu.github.io/mcising/>.

# Statement of need

`mcising` is written for students and researchers in computational, statistical, and condensed matter physics
who need classical Ising Monte Carlo simulation configurable up to third-neighbor interactions. On matched single-thread Metropolis workloads, `mcising` 0.29 ran 2.2 to 2.4 times faster than `peapods` 0.2.0 and about 15 times faster than a NumPy checkerboard implementation on a 32×32 lattice on an Apple M4; the benchmark script and results are committed and regenerated with the documentation. Studies of
frustrated magnetism with competing $J_1$–$J_2$–$J_3$ couplings, studies of
critical phenomena and finite-size scaling are within the scope of `mcising`. `mcising` is used by the author especially to generate phase-labeled data above the size of 100,000 configurations.

# State of the field

Several open-source packages support classical spin-model Monte Carlo simulations. The
closest one is `peapods` [@pei2026], which, similar to `mcising`, pairs a Python API with a
Rust core for Ising systems; it targets spin glasses through per-bond coupling
arrays over custom neighbor offsets, with parallel tempering and replica
overlap-cluster moves. ALPS [@bauer2011] and its successor ALPSCore
[@gaenko2017] take a broader approach: a C++ framework whose `spinmc`
application handles Ising, XY and Heisenberg spins on arbitrary XML-defined
lattices, with binning accumulators for error analysis. In Julia,
`IsingModels.jl` [@fernandez2023] provides Metropolis and Wolff updates on the
square lattice, while `SpinMC.jl` simulates classical O(3) spins with general
interaction matrices and MPI-based parallel tempering. The `mcising`
documentation maintains a feature-by-feature comparison of documented scope,
verified against each project's documentation.

`mcising` differs in three ways. First, frustration is first-class and tested:
$J_2$ and $J_3$ are named neighbor shells with dedicated tables on every
lattice, including the non-Bravais honeycomb, and antiferromagnetic and
competing couplings pass the same exact-enumeration and ground-state tests as
the ferromagnet. Second, uncertainties are default output rather than opt-in:
every observable carries a blocking or jackknife error with an integrated
autocorrelation time. Third, provenance is embedded in the data file, which
records the configuration, seed, version and commit, and runs support
checkpointing and resume. Planned extensions, including Wang-Landau sampling,
sublattice-mapped cluster updates for antiferromagnets, and further lattices
such as kagome, are tracked as post-1.0 scope in the documentation.

# Software design

**Rust core, static dispatch.** The sampler is written in Rust and exposed to
Python through PyO3 [@pyo3], packaged with maturin [@maturin]. Lattices and
update algorithms are Rust traits, and a sweep is generic over both, so each
lattice–algorithm pair compiles to its own inner loop with the choice made once
at the Python boundary. The Metropolis sweep [@metropolis1953] is further
specialized at construction on which Hamiltonian terms are non-zero, with a
cached Boltzmann table per case, so unused $J_2$, $J_3$ or field terms add no
cost. Cluster updates implement Wolff [@wolff1989] and Swendsen–Wang
[@swendsen1987] via the Fortuin–Kasteleyn representation [@fortuin1972]. Both
are valid only for unfrustrated ferromagnetic $J_1$; other couplings are
rejected with an error instead of sampling the wrong distribution. Random
numbers come from xoshiro256** [@blackman2021].

**Exact-enumeration oracle.** A test-only Rust module enumerates small lattices
to build the joint density of states over bond energy and magnetization. From
it come the exact free energy, energy, specific heat and magnetization moments
at any temperature and either sign of $J_1$. Observable definitions match
production bit for bit, so a sampler bug appears as a many-sigma discrepancy
rather than a subtle shift. The oracle rejected an earlier "flip-budget" Wolff
sweep that stopped after a fixed number of flips: the state-dependent stopping
rule biased the final cluster, a fault standard ferromagnetic checks missed.

**Adaptive thermalization and error analysis.** Only a fixed-temperature
diagnostic block is tested for stationarity; the annealing ramp is excluded by
construction. The block goes to the marginal standard error rule [@white1997]
to locate the end of thermalization, and to Sokal's self-consistent windowing
[@sokal1997] to estimate the integrated autocorrelation time
$\tau_\mathrm{int}$, which spaces measurements to de-correlate them. Direct
observables carry blocking standard errors; derived quantities — specific heat,
susceptibility and Binder cumulant [@binder1981] — use delete-one-block
jackknife errors.

**Execution modes and provenance.** A temperature ladder can run as a sequential
cool-down, as independent chains over Rayon [@rayon] threads, or as a parallel
tempering ensemble with the replica-exchange move of @hukushima1996. Production
runs cross the Python–Rust boundary once per measurement block, with the GIL
released while sweeping. Every mode writes the same HDF5 layout, versioned by
a schema number. Attributes record the full configuration, seed, mode,
algorithm, package version and git commit. Older schemas load through explicit
migration paths. Resuming a checkpoint refuses to mix runs with different
settings.

**Testing.** Rust and Python tests run on every push on Linux, macOS and
Windows with an enforced coverage floor. A slower suite of physics validations
runs on every pull request and gates each release. Every documentation code
block is executed in CI, and the three figure-producing example scripts run at
full budget in the physics suite.

# Research impact statement

`mcising` was developed for a study of phase determination
in the frustrated $J_1$–$J_2$ Ising model on the square lattice with and without
deep learning [@civitcioglu2025]. That project needed large, labeled sets of spin
configurations across the ferromagnetic, super-antiferromagnetic and paramagnetic phases. A
subsequent study by another group examined the minimal training set for
convolutional networks on the same model [@li2025]; the wider literature on phase classification commonly uses such datasets. The super-antiferromagnetic phase of the $J_1$–$J_2$ model, the case the original tool was
built for, is reproduced with `mcising` by a committed example script
(\autoref{fig:stripe}).

![The $J_1$–$J_2$ square-lattice phase diagram from the `stripe_phase_diagram.py` example ($L = 32$, Metropolis cool-down): the ferromagnetic order parameter $\langle |m| \rangle$ (left) and the super-antiferromagnetic order parameter $\langle m_s \rangle$ (right) over the $(J_2, T)$ plane, with the specific-heat peak as a finite-size estimate of the transition. Ferromagnetic order for $J_2 > -J_1/2$ and super-antiferromagnetic order below it [@kalz2011; @jin2012].\label{fig:stripe}](figures/stripe_phase_diagram.png)

The validation itself is scripted and committed. One example reproduces Onsager's
exact energy [@onsager1944] and Yang's exact spontaneous magnetization
[@yang1952] on the square lattice with Swendsen–Wang sampling and blocking
errors (\autoref{fig:onsager}), with finite-size rounding near $T_c$ explicitly
labeled. A separate campaign script extracts the critical temperature of four
lattices from Binder-cumulant crossings [@binder1981] with statistical and
finite-size errors. \autoref{tab:tc} compares these estimates to the exact
values [@onsager1944; @houtappel1950; @wannier1950] and, for the cubic lattice,
to the best Monte Carlo estimate [@ferrenberg2018].

![Energy per site and magnetization of the periodic square lattice from the `onsager_reproduction.py` example (Swendsen–Wang, $L = 16$ and $64$) against Onsager's and Yang's exact curves.\label{fig:onsager}](figures/onsager_reproduction.png)

| Lattice | Sizes $L$ | $T_c$, Binder crossing ($\pm$ stat $\pm$ drift) | Reference $T_c$ | $\Delta$ (%) |
|---|---|---|---|---|
| square | 16–64 | 2.2681 $\pm$ 0.0014 $\pm$ 0.0014 | 2.26919 | $-0.05$ |
| triangular | 16–64 | 3.6402 $\pm$ 0.0028 $\pm$ 0.0002 | 3.64096 | $-0.02$ |
| honeycomb | 16–48 | 1.5173 $\pm$ 0.0010 $\pm$ 0.0024 | 1.51865 | $-0.09$ |
| cubic | 8–24 | 4.5128 $\pm$ 0.0013 $\pm$ 0.0064 | 4.51152 | $+0.03$ |

: Critical temperatures from the committed campaign (Swendsen–Wang, independent
mode; crossing of the two largest sizes; "drift" is the change from the previous
size pair). Reference values: exact for the three two-dimensional lattices,
Monte Carlo [@ferrenberg2018] for the cubic lattice.\label{tab:tc}

# AI usage disclosure

From 6 August 2026, the author used Claude Code (Anthropic; Claude Fable 5, and later Claude Fable 5.1
model family) as a coding assistant for the work that took `mcising` from
version 0.21 to 1.0: bug fixes to the sampler and lattices, exact reference
tests and statistical testing, output metadata and checkpointing, documentation
and examples, and a scaffold for this paper: the JOSS section structure, the
bibliography with checked DOIs, the placement of figures and tables, the PDF
build workflow, and provisional prose under each heading. In September 2026, Muse Spark
(`muse-spark-1.3-contributor-free`, via OpenCode) was used for review only; Muse Spark 1.3 made no code edits. Every AI-assisted commit
carries `Assisted-by: Claude Code` in the public git history; all earlier
history, including the original hand-written prototype and the first PyPI releases, was written without AI assistance. The
author specified and reviewed each change, ran and read every physics gate, and made
every release decision. The text of this paper was written by the author, who
reworked the provisional prose section by section. The numerical content, figures and table come from the
committed scripts and results files in the repository.

# Acknowledgements

The author thanks Andreas Honecker and Rudolf A. Römer for the collaboration
on the $J_1$–$J_2$ Ising model study in which `mcising` was first used.

# References
