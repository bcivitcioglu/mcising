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

## Adaptive thermalization

### MSER (Marginal Standard Error Rule)

Finds the truncation point `d` that minimizes `Var(x_d..x_N) / (N - d)`. Discards the initial transient automatically.

### Sokal windowing

Estimates the integrated autocorrelation time:

$$
\tau_{\text{int}} = \frac{1}{2} + \sum_{t=1}^{W} C(t)
$$

where the window `W` is determined self-consistently: stop when `t >= c * tau_int(t)` (default c=6).
