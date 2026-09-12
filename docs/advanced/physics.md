# Physics Background

## The Ising model

The Ising model describes interacting spins on a lattice. Each site `i` has a spin `s_i = +1` or `-1`. The Hamiltonian is:

$$
H = -J_1 \sum_{\langle i,j \rangle} s_i s_j - J_2 \sum_{\langle\langle i,j \rangle\rangle} s_i s_j - J_3 \sum_{\langle\langle\langle i,j \rangle\rangle\rangle} s_i s_j - h \sum_i s_i
$$

where the sums run over nearest-neighbor (NN), next-nearest-neighbor (NNN), and third-nearest-neighbor (TNN) pairs.

## Critical temperatures

Reference values for the nearest-neighbour ferromagnet (J1=1, J2=J3=h=0),
as shipped in `mcising.constants`:

| Lattice | Tc | Source |
|---|---|---|
| Square 2D | 2 / ln(1 + √2) = 2.26919 | Onsager, Phys. Rev. 65, 117 (1944) — exact, sinh(2/Tc) = 1 |
| Triangular 2D | 4 / ln 3 = 3.64096 | Houtappel, Physica 16, 425 (1950); Wannier, Phys. Rev. 79, 357 (1950) — exact, exp(4/Tc) = 3 |
| Honeycomb 2D | 2 / ln(2 + √3) = 1.51865 | Houtappel (1950); Wannier (1950) — exact, cosh(2/Tc) = 2 |
| Cubic 3D | 1 / 0.221654626(5) = 4.51152 | Ferrenberg, Xu & Landau, Phys. Rev. E 97, 043301 (2018) — Monte Carlo |
| Chain 1D | 0 (no transition) | Ising, Z. Phys. 31, 253 (1925) — exact |

### Measured critical temperatures

The reference values are not taken on trust: `scripts/tc_campaign.py`
measures Tc on every lattice with the library itself. Its results are
committed next to it (`scripts/tc_campaign_results.json`) and checked by
the test suite (`tests/test_tc_campaign.py`): the committed table against
the constants on every CI run, and a full quick-budget rerun on fresh
random streams on every pull request, every push to `master`, and before
every release is published.

**Method.** For each lattice and several linear sizes L, the Binder
cumulant U4 = 1 − ⟨m⁴⟩ / (3 ⟨m²⟩²) and the specific heat are measured on a
grid of temperatures around the reference Tc with Swendsen–Wang updates
in independent mode (each temperature is its own chain from a random
start). The Tc estimate is the crossing of the U4(T) curves of the two
largest sizes (Binder 1981): a weighted quadratic is fitted to the
difference D(T) = U4(L₁; T) − U4(L₂; T) over the grid points within ±1 % of
the reference and its root is the crossing. U4 and Cv carry delete-one-
block jackknife errors; the crossing's statistical error is the spread of
the root over a parametric bootstrap of the U4 values, and the drift
between the last two size pairs is quoted separately as the finite-size
systematic. The specific-heat maximum at each L (vertex of a local
parabola) is listed as a secondary estimator: it converges to Tc only as
L^(−1/ν), and on periodic lattices the 3D peak sits *below* Tc. Every fit
reports χ²/dof as a canary — close to 1 means the jackknife errors are
honest and the local polynomial is adequate.

<!-- tc-campaign:begin -->
| Lattice | L | Tc, Binder crossing (± stat ± drift) | Cv peak at largest L | Reference Tc | Δ (%) |
|---|---|---|---|---|---|
| square | 16, 24, 32, 48, 64 | 2.2681 ± 0.0014 ± 0.0014 (L=48, 64) | 2.2814 ± 0.0008 (L=64) | 2.26919 | -0.05 |
| triangular | 16, 24, 32, 48, 64 | 3.6402 ± 0.0028 ± 0.0002 (L=48, 64) | 3.6585 ± 0.0016 (L=64) | 3.64096 | -0.02 |
| honeycomb | 16, 24, 32, 48 | 1.5173 ± 0.0010 ± 0.0024 (L=32, 48) | 1.5180 ± 0.0006 (L=48) | 1.51865 | -0.09 |
| cubic | 8, 12, 16, 24 | 4.5128 ± 0.0013 ± 0.0064 (L=16, 24) | 4.4663 ± 0.0008 (L=24) | 4.51152 | +0.03 |

| Lattice | L pair | Crossing Tc (± stat) | χ²/dof |
|---|---|---|---|
| square | 16, 24 | 2.2648 ± 0.0030 | 1.82 |
| square | 24, 32 | 2.2690 ± 0.0025 | 0.39 |
| square | 32, 48 | 2.2695 ± 0.0018 | 0.85 |
| square | 48, 64 | 2.2681 ± 0.0014 | 0.90 |
| triangular | 16, 24 | 3.6308 ± 0.0042 | 1.41 |
| triangular | 24, 32 | 3.6381 ± 0.0063 | 0.28 |
| triangular | 32, 48 | 3.6403 ± 0.0024 | 0.14 |
| triangular | 48, 64 | 3.6402 ± 0.0028 | 0.61 |
| honeycomb | 16, 24 | 1.5187 ± 0.0016 | 0.98 |
| honeycomb | 24, 32 | 1.5197 ± 0.0015 | 1.89 |
| honeycomb | 32, 48 | 1.5173 ± 0.0010 | 1.14 |
| cubic | 8, 12 | 4.5052 ± 0.0026 | 0.92 |
| cubic | 12, 16 | 4.5064 ± 0.0028 | 1.65 |
| cubic | 16, 24 | 4.5128 ± 0.0013 | 2.47 |

| Lattice | L | Cv peak T (± stat) | χ²/dof |
|---|---|---|---|
| square | 16 | 2.3170 ± 0.0068 | 1.49 |
| square | 24 | n/a (not bracketed) | — |
| square | 32 | 2.2947 ± 0.0029 | 1.27 |
| square | 48 | 2.2836 ± 0.0026 | 1.34 |
| square | 64 | 2.2814 ± 0.0008 | 1.10 |
| triangular | 16 | n/a (not bracketed) | — |
| triangular | 24 | 3.6844 ± 0.0034 | 0.39 |
| triangular | 32 | 3.6872 ± 0.0118 | 1.15 |
| triangular | 48 | 3.6652 ± 0.0025 | 2.25 |
| triangular | 64 | 3.6585 ± 0.0016 | 1.10 |
| honeycomb | 16 | n/a (not bracketed) | — |
| honeycomb | 24 | 1.5231 ± 0.0017 | 1.23 |
| honeycomb | 32 | 1.5201 ± 0.0010 | 1.17 |
| honeycomb | 48 | 1.5180 ± 0.0006 | 0.64 |
| cubic | 8 | 4.3411 ± 0.0107 | 1.42 |
| cubic | 12 | 4.4192 ± 0.0029 | 0.32 |
| cubic | 16 | 4.4444 ± 0.0017 | 1.89 |
| cubic | 24 | 4.4663 ± 0.0008 | 5.25 |

Swendsen–Wang, independent mode; 2000 thermalization + 40000 measurement sweeps per temperature, sampled every 2 sweeps; grid Tc·[1 ± 0.05] at 0.005 plus Tc·[1 ± 0.015] at 0.0025; crossing fit within ±0.01, peak fit within ±0.015 of the maximum; mcising 0.28.0 (28ebce3), generated 2026-09-01T10:04:07+00:00.
<!-- tc-campaign:end -->

Reading the tables: the first ± is statistical, the second the finite-size
drift; Δ is the deviation of the Binder-crossing estimate from the
reference value in percent. Three features are physics, not defects. The
honeycomb cumulant curves cross near U4 ≈ 0.52 rather than the 0.611 of
the square and triangular lattices (the textbook U* = 0.6107) because its
L × L-cell torus has a 3.46 : 1 aspect ratio — the crossing *value* is
shape dependent, the crossing *temperature* is not. The smallest size
pairs are drift diagnostics only: their crossings can fall outside the
fit window (reported as not bracketed rather than extrapolated). And a
χ²/dof of 2–3 marks a local polynomial at the edge of adequacy rather
than a bad error bar: the 3D scaling variable grows fastest with L, so
the cubic 16/24 quadratic and the sharp cubic specific-heat peak fit
least well (the jackknife errors themselves were cross-checked against
independent-chunk scatter and agree).

Reproduce or refresh the tables with

```bash
uv run python scripts/tc_campaign.py --write-docs   # full budget, ~5 min on 10 cores
uv run python scripts/tc_campaign.py --quick        # the CI slow-suite budget
```

## Staggered magnetization

The uniform magnetization vanishes identically in every phase that breaks
the lattice's translation symmetry — the stripe phase of the square
$J_1$–$J_2$ model, the layered phase of the cubic one, the Néel phase of
any antiferromagnet. Every run therefore also records the staggered
magnetizations, one per combination of lattice axes. With $n$ axes (the
length of the configuration array's shape: 1 for the chain, 2 for the
square and triangular lattices, 3 for the cubic lattice *and* for the
honeycomb, whose third axis is the sublattice index) and $n_a(i)$ the
coordinate of site $i$ along axis $a$, component $k$ — a bitmask over the
axes, $0 \le k < 2^n$ — is

$$
m_k = \frac{1}{N} \sum_i (-1)^{\sum_{a \in k} n_a(i)}\, s_i .
$$

Component 0 is the uniform magnetization. The others are the order
parameters of the commensurate ordered phases:

| Lattice | Component | Pattern | Order |
|---|---|---|---|
| chain | 1 | alternates site to site | Néel |
| square, triangular | 1 | alternates from row to row | stripe |
| square, triangular | 2 | alternates from column to column | stripe |
| square, triangular | 3 | alternates in both directions | Néel (square) |
| honeycomb | 4 | alternates between the two sublattices | Néel |
| cubic | 1, 2, 4 | alternates plane to plane along one axis | layered |
| cubic | 7 | alternates in all three directions | Néel |

`results.staggered_magnetization[T]` holds one row per measurement with
$2^n$ columns; `LatticeConfig.shape` gives the axes. Two caveats: an odd
extent makes $(-1)^{n_a}$ non-periodic across that boundary (the quantity
is still well defined, but no longer a Fourier component of the lattice),
and on the triangular lattice these are the M-point stripe components,
not the three-sublattice order of the antiferromagnet.

## Monte Carlo algorithms

### Metropolis

Single-spin-flip with acceptance probability:

$$
P(\text{accept}) = \min\left(1, e^{-\beta \Delta E}\right)
$$

where `dE = 2 * spin * local_field`. mcising precomputes these probabilities in lookup tables.

### Wolff cluster

1. Pick a random seed spin
2. Grow a cluster via DFS: add aligned neighbors with probability `p = 1 - exp(-2 * beta * J1)`
3. Flip the entire cluster

Dramatically reduces critical slowing down. Autocorrelation time scales as L^0.25 instead of L^2.17 for Metropolis.

### Swendsen-Wang

1. Activate bonds between aligned NN pairs with probability `p = 1 - exp(-2 * beta * J1)`
2. Identify all clusters via Union-Find
3. Flip each cluster independently with 50% probability

Processes the entire lattice per sweep. Uses path compression for O(N * alpha(N)) complexity.

### Parallel Tempering

Run N replicas at different temperatures simultaneously. Every
`swap_interval` sweeps, attempt swaps between adjacent replicas
(alternating between the even and the odd pairs of the ladder):

$$
P(\text{swap}) = \min\left(1, e^{(\beta_i - \beta_j)(E_i - E_j)}\right)
$$

High-temperature replicas explore freely and pass configurations to low-temperature replicas via swaps.

The energies entering the criterion are not re-summed over the lattice
after every round: each sweep reports the exact integer change of the
neighbour-shell sums it caused (every accepted Metropolis flip and every
Wolff cluster knows its own boundary), so the ladder carries every
replica's energy forward at constant cost and the parallel sweeps are
never followed by a serial pass. The energies *recorded* at measurement
points are still evaluated directly from the spins.

#### Diagnostics

Whether a ladder actually mixed cannot be read off the averages it
returns: replicas trapped on one side of a free-energy barrier still give
smooth, plausible numbers. Every parallel-tempering run therefore records
`results.pt_diagnostics`:

- **Swap acceptance** per adjacent pair, `swap_accepted / swap_attempted`.
  A pair that almost never swaps splits the ladder in two; the usual
  target is roughly 20–50 %, reached by spacing the temperatures so that
  the energy histograms of neighbouring rungs overlap.
- **Round trips** per replica: the number of completed coldest → hottest
  → coldest excursions. This is the direct evidence that configurations
  travel the whole ladder. A run whose total is zero has not
  demonstrably crossed its temperature range, whatever its acceptance
  rates say.

Only production rounds are counted; thermalization never swaps.

### Wang-Landau and multicanonical sampling

At a first-order transition the two coexisting phases are separated by a
free-energy barrier $\Delta F \propto \sigma L^{d-1}$, and every canonical
sampler — parallel tempering included — needs a time exponential in that
barrier to cross it. Flat-histogram methods remove the barrier from the
sampling problem by making every energy equally likely.

**Wang-Landau** (Wang & Landau, Phys. Rev. Lett. 86, 2050 (2001)) is a
random walk in energy with the acceptance probability

$$
P(E \to E') = \min\left(1, \frac{g(E)}{g(E')}\right),
$$

where the density of states $g(E)$ is *learned as the walk proceeds*:
after every proposal the current bin receives $\ln g(E) \mathrel{+}= \ln
f$ and one histogram count. When the histogram is flat — its minimum over
the visited bins is at least `flatness` times its mean — the histogram is
reset and $\ln f \to \ln f / 2$. Once $\ln f$ falls below $1/t$, with $t$
the mean number of visits per bin, the schedule switches to $\ln f = 1/t$
(Belardinelli & Pereyra, Phys. Rev. E 75, 046701 (2007)), whose error
decreases as $t^{-1/2}$ instead of saturating. mcising walks on the exact
energy grid of the couplings: the walker carries the integer
neighbour-shell sums, the bin width is the greatest common divisor of all
single-flip energy changes, and the bin of a state is a pure function of
those integers. Proposals pick random sites (a sequential scan is not
ergodic on this walk), a proposal leaving the energy window is rejected
and still counts as a visit of the current bin (Schulz, Binder, Müller &
Landau, Phys. Rev. E 67, 067102 (2003)), and a bin entered for the first
time starts from the $\ln g$ of the bin the walker came from.

**Multicanonical production** (Berg & Neuhaus, Phys. Rev. Lett. 68, 9
(1992)) then freezes the weights $W(E) = 1/g(E)$. With fixed weights the
walk is an exact Markov chain whose stationary distribution is
$\pi(x) \propto W(E(x))$ — for *any* positive $W$ — so a canonical
average at inverse temperature $\beta$ follows from the recorded series
without bias:

$$
\langle A \rangle_T = \frac{\sum_i w_i A_i}{\sum_i w_i},
\qquad
\ln w_i = \ln g(E_i) - \beta E_i .
$$

The Wang-Landau weights only set the variance, through how flat the
production histogram is; the honest uncertainty is the delete-one-block
jackknife of the reweighted series, with blocks that never straddle two
walkers (blocks shorter than a few round-trip times underestimate it).
Every measurement records the energy, the magnetization and the staggered
magnetizations, so the order parameter of a symmetry-broken phase, its
susceptibility and its Binder cumulant are available at any temperature.
The production histogram also refines the estimate,
$\ln g_\mathrm{prod}(E) = \ln H_\mathrm{prod}(E) + \ln g_\mathrm{WL}(E)$,
and its difference to $\ln g_\mathrm{WL}$ is the measured error of the
weights.

The first stage parallelises as replica-exchange Wang-Landau (Vogel, Li,
Wüst & Landau, Phys. Rev. Lett. 110, 210603 (2013)): the range is split
into overlapping windows, each sampled by several walkers; walkers of
adjacent windows exchange configurations with probability
$\min\left(1, \frac{g_i(E_i)\, g_j(E_j)}{g_i(E_j)\, g_j(E_i)}\right)$
when both energies lie in both windows; the walkers of a window average
their estimates at every flatness check; one modification-factor schedule
governs all windows (a check passes when the pooled histogram of every
window is flat, the $1/t$ clock is the slowest window's); and the pieces
are joined at the overlap bin where their slopes $d\ln g/dE$ agree best.

#### Diagnostics

Whether the walk actually connected the two ends of its energy range
cannot be read off the reweighted averages. Every run records
`results.wang_landau` (the $\ln f$ schedule, sweeps and flatness per
iteration, whether $\ln f$ reached `log_f_final`) and
`results.production`: the flip acceptance of every walker, the flatness
of the pooled production histogram, and the number of **round trips** —
excursions from the lowest visited energy bin to the highest and back —
each walker completed, the analogue of a replica's round trip in parallel
tempering. Each reweighted estimate carries the Kish **effective sample
size** $(\sum_i w_i)^2 / \sum_i w_i^2$ and the **edge weight**, the
canonical weight sitting in the two outermost visited bins, which must be
negligible for the estimate to be trusted.

#### Free-energy barriers

At the transition the canonical energy distribution $P_T(E) \propto
g(E)\,e^{-\beta E}$ is bimodal. Following Lee & Kosterlitz (Phys. Rev.
Lett. 65, 137 (1990)), mcising reports

$$
\frac{\Delta F}{T} = \ln \frac{P_\mathrm{peak}}{P_\mathrm{bottom}},
\qquad
\sigma = \frac{T\,(\Delta F / T)}{2 L^{d-1}},
$$

with $P_\mathrm{bottom}$ the mean of $P_T$ over the middle third between
the peaks (the slab plateau of a periodic box, which holds two
interfaces of area $L^{d-1}$) or its minimum, measured from the lower
peak; the peaks are located on the histogram smoothed over one percent
of the visited bins (the phases are hundreds of bins wide on a large
lattice, the per-bin noise of a finite run is not), and a secondary
peak below a thousandth of the main one is not a phase (the low-energy
tail of any lattice has dips from the degeneracies of its first
excitations). The **equal-height temperature** is the
conventional place to quote the barrier; the **equal-weight temperature**
$T_w(q)$, at which the weight of the ordered phase is $q$ times that of
the disordered one ($q$ = number of ordered states), is the finite-size
transition temperature with exponentially small corrections (Borgs &
Kotecký, J. Stat. Phys. 61, 79 (1990)). Both shift as $L^{-d}$. The
barrier estimate is only meaningful once the bottom is a plateau, i.e.
for sizes at which the peaks are well separated.

The cubic $J_1$-$J_2$ model is the case this machinery was built for.
With ferromagnetic $J_1$ and antiferromagnetic next-nearest-neighbour
$J_2$ (twelve face diagonals), the ground state is ferromagnetic for
$|J_2|/J_1 < 1/4$ and layered — ferromagnetic planes alternating along
one axis, $e_0 = -J_1 + 2 J_2$ per site — beyond that; the three axes and
two signs give six ordered states, and the transition out of the layered
phase is first order (the order parameter is a three-component field
with cubic anisotropy whose single-axis state has no stable fixed point:
Aharony, Phys. Rev. B 8, 4270 (1973)). Its order parameter is
$\psi = \max_k |m_k|$ over the layered components $k = 1, 2, 4$ of the
staggered magnetization.

## Correlation length

The correlation length is the second-moment (structure-factor curvature)
estimator over the connected correlation shells:

$$
\xi^2 = \frac{\sum_{r>0} n(r)\, r^2\, C(r)}{2d \sum_{r>0} n(r)\, C(r)}
$$

where `d` is the spatial dimension of the lattice (1 for chain, 2 for
square/triangular/honeycomb, 3 for cubic), `C(r)` is the pair-averaged
connected correlation at distance `r`, and `n(r)` is the number of site
pairs in that shell — the weights that make the sums reproduce the
lattice sum over displacement vectors.

Conventions worth knowing before comparing numbers:

- **The r=0 self-term is excluded.** `C(0) = 1 - m^2` is the on-site
  variance, not a correlation between distinct spins. (The Fourier-space
  `xi_2nd` estimator keeps it; the two definitions differ.)
- **Shells are summed up to the first non-positive `C(r)`.** Beyond the
  noise floor a finite sample's correlations are dominated by noise
  exactly where the `r^2` weight is largest.
- **The estimator is exact for Ornstein–Zernike correlations.** On
  `C(r) = r^{-(d-2)/2} K_{(d-2)/2}(r/\xi)` it returns `xi` in every
  dimension. On a *pure* exponential `e^{-r/\xi}` it returns
  `sqrt((d+1)/2) * xi` for `d > 1` — a pure exponential is not an OZ
  propagator, so this is a property of the definition, not an error.
- **Distances are true Euclidean distances** in lattice-spacing (chain,
  square, cubic), lattice-spacing (triangular, 60-degree basis), or
  NN-bond-length (honeycomb) units. Note the honeycomb `L x L`-cell
  torus is `3L x (sqrt(3)/2)L` in real space — an aspect ratio of about
  3.46:1 — so correlation lengths there are limited by the short axis
  (`~0.87 L`).

## Adaptive thermalization

Adaptive mode separates *annealing* from *diagnosis*. Each temperature
is reached with a cool-down ramp (pure annealing — its energy trace is
non-stationary by construction and is never analyzed), followed by a
fixed-temperature diagnostic series that is the only input to MSER and
Sokal analysis. The production measurement interval derives from the
stationary tail of that fixed-temperature series.

### MSER (Marginal Standard Error Rule)

Finds the truncation point `d` that minimizes `Var(x_d..x_N) / (N - d)`,
discarding the initial transient. Every candidate in the first half of
the series is evaluated exactly (single O(N) pass); following the
classical rule, an argmin at or beyond the midpoint means the data
cannot demonstrate stationarity and the series is reported as **not
thermalized** — the simulation then extends the fixed-temperature run
(up to `max_thermalization_sweeps`) and warns if stationarity is never
detected. Restricting candidates to the first half also guarantees
every evaluated tail keeps at least half the data, which structurally
prevents noisy tiny-tail truncation estimates.

### Sokal windowing

Estimates the integrated autocorrelation time on the stationary tail:

$$
\tau_{\text{int}} = \frac{1}{2} + \sum_{t=1}^{W} C(t)
$$

where the window `W` is determined self-consistently: stop when `t >= c * tau_int(t)` (default c=6).
