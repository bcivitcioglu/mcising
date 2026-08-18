# Physics Background

## The Ising model

The Ising model describes interacting spins on a lattice. Each site `i` has a spin `s_i = +1` or `-1`. The Hamiltonian is:

$$
H = -J_1 \sum_{\langle i,j \rangle} s_i s_j - J_2 \sum_{\langle\langle i,j \rangle\rangle} s_i s_j - J_3 \sum_{\langle\langle\langle i,j \rangle\rangle\rangle} s_i s_j - h \sum_i s_i
$$

where the sums run over nearest-neighbor (NN), next-nearest-neighbor (NNN), and third-nearest-neighbor (TNN) pairs.

## Critical temperatures

Exact or high-precision values for J1=1, h=0:

| Lattice | Tc | Source |
|---|---|---|
| Square 2D | 2 / ln(1 + sqrt(2)) = 2.269 | Onsager (1944), exact |
| Triangular 2D | 4 / ln(3) = 3.641 | Exact |
| Honeycomb 2D | 2 / ln(2 + sqrt(3)) = 1.519 | Exact |
| Cubic 3D | 4.5115 | High-precision MC estimate |
| Chain 1D | 0 (no transition) | Exact (Ising, 1925) |

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
