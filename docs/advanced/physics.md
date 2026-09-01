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
random streams every night.

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
uv run python scripts/tc_campaign.py --quick        # the nightly test's budget
```

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

Run N replicas at different temperatures simultaneously. After each sweep round, attempt swaps between adjacent replicas:

$$
P(\text{swap}) = \min\left(1, e^{(\beta_i - \beta_j)(E_i - E_j)}\right)
$$

High-temperature replicas explore freely and pass configurations to low-temperature replicas via swaps.

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
