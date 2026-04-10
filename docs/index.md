<p align="center">
  <img src="assets/logo.svg" alt="mcising" width="300">
</p>

<h1 align="center">mcising</h1>

<p align="center">
  <em>High-performance Ising model Monte Carlo simulation with a Rust core.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/mcising/"><img src="https://img.shields.io/pypi/v/mcising" alt="PyPI"></a>
  <a href="https://pepy.tech/project/mcising"><img src="https://static.pepy.tech/badge/mcising" alt="Downloads"></a>
  <a href="https://github.com/bcivitcioglu/mcising/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

## Install

=== "uv"

    ```bash
    uv add mcising
    ```

=== "pip"

    ```bash
    pip install mcising
    ```

## Quick example

```python
from mcising import Simulation, SimulationConfig, LatticeConfig

# Configure: 32x32 square lattice, three temperatures across Tc
config = SimulationConfig(
    lattice=LatticeConfig(size=32, j1=1.0),
    temperatures=(3.0, 2.269, 1.5),
    n_sweeps=1000,
    seed=42,
)

# Run
results = Simulation(config).run()

# Inspect
for T in results.temperatures:
    E = results.energy[T].mean()
    M = abs(results.magnetization[T]).mean()
    print(f"T={T:.3f}: <E>={E:.4f}, <|M|>={M:.4f}")
```

This runs a Monte Carlo simulation of the 2D Ising model on a 32x32 square lattice, scanning through three temperatures including the critical point Tc = 2.269.

---

## What mcising gives you

<div class="grid cards" markdown>

-   :material-grid:{ .lg .middle } **5 Lattice Geometries**

    ---

    Square, triangular, honeycomb, cubic (3D), and chain (1D). All with periodic boundary conditions.

    [:octicons-arrow-right-24: Lattice types](tutorial/lattice-types.md)

-   :material-lightning-bolt:{ .lg .middle } **268M spin updates/sec**

    ---

    Rust core via PyO3. 3.4x faster than peapods. 430x faster than pure Python.

    [:octicons-arrow-right-24: Performance](advanced/performance.md)

-   :material-atom:{ .lg .middle } **J1-J2-J3 Frustrated Magnetism**

    ---

    Nearest, next-nearest, and third-nearest-neighbor couplings plus external field. 15 auto-optimized Metropolis strategies.

    [:octicons-arrow-right-24: Frustrated magnetism](tutorial/frustrated-magnetism.md)

-   :material-server-network:{ .lg .middle } **3 Execution Modes**

    ---

    Sequential cool-down, independent parallel (Rayon), or parallel tempering with replica exchange.

    [:octicons-arrow-right-24: Parallel execution](tutorial/parallel-execution.md)

-   :material-chart-scatter-plot:{ .lg .middle } **3 MC Algorithms**

    ---

    Metropolis single-spin-flip, Wolff cluster, and Swendsen-Wang cluster. Choose the right tool for your physics.

    [:octicons-arrow-right-24: Algorithms](tutorial/cluster-algorithms.md)

-   :material-auto-fix:{ .lg .middle } **Adaptive Thermalization**

    ---

    MSER equilibration detection + Sokal autocorrelation estimation. No more guessing warmup sweeps.

    [:octicons-arrow-right-24: Adaptive mode](tutorial/adaptive-mode.md)

</div>

## Or use the CLI

```bash
mcising run -L 32 -T 3.0 -T 2.269 -T 1.5 -o results.h5
mcising summary results.h5
mcising plot energy results.h5 -o energy.png
mcising plot specific-heat results.h5 -o cv.png
mcising export results.h5 lattices.zip
```

Full CLI reference: **[CLI Guide](guide/cli.md)**

## Next steps

New to mcising? Start with the **[Tutorial](tutorial/first-simulation.md)** — it walks you through a complete simulation in 5 minutes.

Looking for a specific function or class? Check the **[API Reference](reference/simulation.md)**.

Need CLI commands? See the **[CLI Reference](guide/cli.md)**.
