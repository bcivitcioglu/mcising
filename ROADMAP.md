# mcising v1.0 - Transformation Plan

## Context

**mcising** is a Python library (v0.13) for 2D Ising model Monte Carlo simulation with J1-J2 interactions. It has ~585 lines of pure Python, 5 basic tests, no CI/CD, and hasn't been updated since July 2024. The goal is to transform it into a high-performance, feature-rich, JOSS-publishable library.

**Why this matters:** No general-purpose classical MC Ising model Python library has been published in JOSS. mcising already has unique differentiators (J1-J2 frustrated magnetism, correlation functions, cool-down thermalization) that no competitor offers. With a Rust performance core and modern Python tooling, it can become the definitive tool in this space.

**Key competitor:** peapods (Rust/PyO3, 4 algorithms, N-dimensional) but lacks J1-J2, correlation functions, and structured data export.

---

## Code Quality Standards

### Python
- **Strict mypy** (`strict = true`) - all functions annotated, no `Any` escapes
- **Type stubs** (`_core.pyi`) for the Rust module so mypy can check calls across the boundary
- **Frozen dataclasses** (`frozen=True`) for all config objects - immutable, hashable, safe
- **Custom exception hierarchy**: `MCIsingError` base, `ConfigurationError`, `SimulationError`
- **No magic numbers** - physics constants centralized in `constants.py`
- **Enums** for all categorical choices (`LatticeType`, `Algorithm`)
- **Module-level `__all__`** in every file
- **Strict dependency DAG**: `config <- _core <- simulation <- io/plotting/cli` (no circular imports)
- **Validate early** at the Python boundary, not deep in Rust

### Rust
- **Zero `unsafe`** - PyO3 handles FFI, no manual unsafe blocks
- **No `.unwrap()` in library code** - proper `Result`/`Option`, convert to `PyErr` at boundary
- **`#[deny(clippy::all)]` + `#[warn(clippy::pedantic)]`** at crate level
- **Precomputed neighbor tables** for all lattice types - trades O(N) memory for maximum inner-loop speed
- **Static dispatch** (generics/monomorphization) for the hot path - specialized code per lattice+algorithm combo
- **Dynamic dispatch** only at the PyO3 boundary (enum-based) to select lattice/algorithm at runtime
- **No heap allocation** in the inner Metropolis loop
- **`Xoshiro256StarStar`** RNG - fast, deterministic, reproducible. Owned per simulation instance (no global state)
- **All physics quantities as `f64`**, spins as `i8`

### PyO3 Boundary
- **Thick and intentional**: Rust side does all physics. Python side does config, orchestration, I/O, plotting.
- Only `simulation.rs` imports `pyo3` - internal Rust code is pure Rust with no Python awareness
- Data crosses via NumPy arrays (zero-copy where possible) and scalar primitives
- Same seed + same parameters = identical results, always

---

## Architecture Overview

```
mcising/
├── rust/src/              # Rust core via PyO3 (compiled to mcising._core)
│   ├── lib.rs             # PyO3 module registration
│   ├── lattice/           # Lattice trait + implementations (square, then more)
│   ├── algorithm/         # MC algorithm trait + implementations
│   ├── observables/       # Energy, magnetization, correlation
│   ├── rng.rs             # Xoshiro256** RNG
│   └── simulation.rs      # IsingSimulation pyclass
├── python/mcising/        # Python package (high-level API)
│   ├── __init__.py        # Public exports
│   ├── _core.pyi          # Type stubs for Rust bindings
│   ├── simulation.py      # High-level Simulation class
│   ├── config.py          # Dataclass configs (LatticeConfig, SimulationConfig)
│   ├── observables.py     # Post-processing (heat capacity, susceptibility)
│   ├── io.py              # HDF5/JSON output
│   ├── plotting.py        # Matplotlib visualization
│   └── cli.py             # Typer CLI with Rich output
├── tests/                 # pytest test suite
├── paper/                 # JOSS paper (paper.md, paper.bib)
├── docs/                  # MkDocs documentation
├── Cargo.toml             # Rust crate config
└── pyproject.toml         # Python project config (maturin build)
```

---

## Agile Phases

### Phase 1: Rust Core + Square Lattice + Metropolis (Weeks 1-3)

**Goal:** Replace the Python hot path with Rust. Get a working maturin build that produces identical physics results.

**Sprint 1.1 - Project Scaffolding (Week 1):**

Step 1: **uv project initialization** (do this first to manage all deps)
- Move existing source to `_legacy/` (preserves as physics reference)
- Remove old build artifacts: `setup.py`, `requirements.txt`, `mcising.egg-info/`, `dist/`, `__pycache__/`
- Create `.gitignore` (target/, *.so, *.pyd, __pycache__, .venv/, etc.)
- Rename `LICENCE` -> `LICENSE`
- Run `uv init` to create initial `pyproject.toml`
- Configure `pyproject.toml` with project metadata, dependencies, and dependency groups:
  - Runtime: numpy, matplotlib, h5py, rich, typer
  - Dev: pytest, pytest-cov, ruff, mypy, maturin
- **Checkpoint:** `uv sync` succeeds, `uv run python --version` works, lockfile created

Step 2: **Rust toolchain verification**
- Verify Rust is installed: `rustc --version && cargo --version`
- If not installed, install via rustup
- **Checkpoint:** `rustc --version` prints a version >= 1.70

Step 3: **Maturin + PyO3 setup**
- Add maturin as dev dep: `uv add --dev maturin`
- Update `pyproject.toml` build-system to use maturin
- Create `Cargo.toml` with PyO3, numpy, rand, rand_xoshiro dependencies
- Create minimal `rust/src/lib.rs` with PyO3 "hello world" module (exports a simple function)
- Create `python/mcising/__init__.py` importing from `_core`
- **Checkpoint:** `uv run maturin develop` builds without errors
- **Checkpoint:** `uv run python -c "from mcising._core import hello; print(hello())"` works

**Sprint 1.2 - Rust Core (Week 2):**
- Implement `Lattice` trait in `rust/src/lattice/mod.rs` (num_sites, nearest_neighbors, next_nearest_neighbors, distance_squared)
- Implement `SquareLattice` in `rust/src/lattice/square.rs` with periodic boundary conditions
- **Checkpoint:** `cargo test` passes Rust unit tests for neighbor computation on known small lattice
- Implement `McAlgorithm` trait in `rust/src/algorithm/mod.rs`
- Implement `Metropolis` in `rust/src/algorithm/metropolis.rs`
  - Key optimization: compute `dE = 2 * spin * local_field` directly instead of computing energy before and after flip
- **Checkpoint:** `cargo test` passes Rust unit tests for Metropolis (energy decreases at T=0, acceptance works)
- Implement `IsingSimulation` pyclass in `rust/src/simulation.rs`:
  - `new(lattice_size, j1, j2, h, seed)` - constructor
  - `sweep(n_sweeps, beta)` - Metropolis sweeps
  - `energy()` / `magnetization()` - observables
  - `get_spins()` -> PyArray2<i8> / `set_spins()`
  - `correlation_function()` -> (PyArray1<f64>, PyArray1<f64>)
- Fix existing energy bug: v0.13 double-counts the magnetic field term in `energy()` (line 144 of isinglattice.py)
- **Checkpoint:** `uv run maturin develop && uv run python -c "from mcising._core import IsingSimulation; sim = IsingSimulation(10, 1.0, 0.0, 0.0, 42); print(sim.energy())"` works

**Sprint 1.3 - Python Wrapper + Verification (Week 3):**
- Implement `python/mcising/config.py` - `LatticeConfig`, `SimulationConfig` dataclasses with enums
- **Checkpoint:** `uv run python -c "from mcising.config import SimulationConfig; print(SimulationConfig())"` works
- Implement `python/mcising/simulation.py` - `Simulation` class wrapping `_RustSim`, `SimulationResults` dataclass
- **Checkpoint:** `uv run python -c "from mcising import Simulation, SimulationConfig; s = Simulation(SimulationConfig()); r = s.sweep(2.269, 100); print(r)"` works
- Implement `python/mcising/io.py` - HDF5 output via h5py
- **Checkpoint:** Simulation results can be saved to and loaded from HDF5
- Port and expand tests to pytest (~20 tests):
  - Lattice initialization, neighbors, flip
  - Metropolis: energy decreases at low T, acceptance rate
  - Known analytical: all-up energy, magnetization = 1.0
  - Physics smoke test: <|m|> transition across T_c ~ 2.269 on 32x32 lattice
- **Checkpoint:** `uv run pytest` passes all ~20 tests

---

### Phase 2: Modern Tooling + CI/CD (Weeks 4-5)

**Goal:** Production-grade infrastructure.

**Sprint 2.1 - CLI + Rich Output (Week 4):**
- Implement `python/mcising/cli.py` with Typer:
  - `mcising run` - full simulation with Rich progress bars
  - `mcising benchmark` - performance benchmarks
  - `mcising info` - version, build info, available algorithms
- **Checkpoint:** `uv run mcising info` prints version and available algorithms
- **Checkpoint:** `uv run mcising run -L 10 --seed 42` runs simulation with Rich progress and saves HDF5 output
- Implement `python/mcising/plotting.py` - `plot_lattice()`, `plot_observables()`, `plot_correlation()`
- **Checkpoint:** Plotting functions produce matplotlib figures without errors

**Sprint 2.2 - CI/CD + Code Quality (Week 5):**
- `.github/workflows/ci.yml` - pytest + ruff + mypy + cargo test across Python 3.10-3.13, Linux/macOS/Windows
- `.github/workflows/release.yml` - maturin wheel builds + PyPI publish on tag
- `.pre-commit-config.yaml` - ruff, mypy, rustfmt, clippy
- `python/mcising/_core.pyi` - type stubs for Rust module
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates
- **Checkpoint:** `uv run ruff check python/ tests/` passes with no errors
- **Checkpoint:** `uv run mypy python/mcising/` passes
- **Checkpoint:** Push to GitHub -> CI workflow runs and passes on all platforms

---

### Phase 3: Adaptive Measurement + Cluster Algorithms (Weeks 6-8)

**Goal:** Eliminate wasted sweeps via adaptive thermalization/measurement, then eliminate critical slowing down via cluster algorithms.

**Sprint 3.1 - Rust: Autocorrelation & Adaptive Analysis (Week 6):**
- New `rust/src/autocorrelation.rs` — pure Rust, no PyO3 awareness:
  - **MSER thermalization detection** (`detect_thermalization`): find truncation point d minimizing variance/(N-d) over the energy time series. O(N), no tuning parameters.
  - **Sokal automatic windowing** (`integrated_autocorrelation_time`): compute `tau_int = 0.5 + sum C(t)` with self-consistent cutoff `t >= c * tau_int(t)`, c=6.0 default.
  - **Combined analysis** (`analyze_thermalization`): run MSER on full series, then Sokal on stationary tail (series[d:]). Returns truncation point, tau_int, and recommended measurement interval.
- New PyO3 methods on `IsingSimulation` in `rust/src/simulation.rs`:
  - `thermalize_with_diagnostics(temp_schedule: Vec<f64>) -> PyArray1<f64>` — runs cool-down sweeps, records energy after each sweep, returns the energy time series
  - `extend_thermalization(n_sweeps: usize, beta: f64) -> PyArray1<f64>` — additional sweeps at fixed T, returns energy series
  - `analyze_thermalization_series(series: PyArray1<f64>, c_window: f64) -> dict` — static method wrapping MSER + Sokal
  - `production_sweeps(n_measurements, interval, beta, store_configs) -> (energies, mags, configs)` — batched measurement collection in a single Rust call (also benefits fixed mode)
- **Checkpoint:** `cargo test` passes for known analytical cases (white noise tau~0.5, AR(1) with known tau, step function for MSER)

**Sprint 3.2 - Python: Adaptive Integration (Week 7):**
- `python/mcising/config.py`: Add `AdaptiveConfig` frozen dataclass:
  - `enabled: bool = False` (opt-in; fixed mode stays default for beginners)
  - `min_thermalization_sweeps`, `max_thermalization_sweeps`, `c_window`, `min_independent_samples`, `max_total_sweeps`, `tau_multiplier`
- `python/mcising/simulation.py`:
  - `_thermalize_adaptive()`: calls `thermalize_with_diagnostics`, checks MSER, extends via `extend_thermalization` if not stationary
  - `_collect_at_temperature_adaptive()`: uses tau_int from thermalization analysis to set measurement_interval = `max(1, round(tau_multiplier * tau_int))`
  - `AdaptiveDiagnostics` dataclass: tau_int, truncation_point, interval_used, sweeps_used per temperature
  - Dispatch in `run()`: adaptive vs fixed based on `config.adaptive.enabled`
- Update `_core.pyi` type stubs, `io.py` (HDF5 adaptive diagnostics), `cli.py` (`--adaptive`, `--min-samples`, `--max-sweeps`)
- **Checkpoint:** `uv run pytest` — adaptive mode on 16x16 produces independent samples; fixed mode unchanged
- **Checkpoint:** `uv run mcising run --adaptive -L 32 --seed 42` runs with Rich progress and reports tau_int per temperature

**Sprint 3.3 - Cluster Algorithms (Week 8):**
- **Wolff cluster algorithm** (`rust/src/algorithm/wolff.rs`):
  - BFS-based cluster growth, bond probability p = 1 - exp(-2*beta*J)
  - Constraint: works only for J2=0, h=0 (bond probability derivation assumes nearest-neighbor-only Hamiltonian)
- **Constraint enforcement** (validate early at Python boundary):
  - In `simulation.py`, raise `ConfigurationError` if algorithm is `wolff` or `swendsen_wang` and (`j2 != 0.0` or `h != 0.0`)
  - Clear error message: "Cluster algorithms require J2=0 and h=0. Use algorithm='metropolis' for J1-J2 or external field simulations."
  - Rust-side safety assert as a secondary guard (panic if invariant violated — should never be reached)
- **Checkpoint:** `cargo test` passes Wolff-specific Rust tests (cluster sizes, energy conservation)
- **Checkpoint:** `uv run pytest tests/test_wolff.py` passes (statistics agree with Metropolis)
- **Checkpoint:** Wolff with J2!=0 raises `ConfigurationError` at Python level
- **Swendsen-Wang** (`rust/src/algorithm/swendsen_wang.rs`):
  - Union-Find with path compression, flip each cluster with p=1/2
  - Same constraint enforcement as Wolff (J2=0, h=0)
- **Checkpoint:** `cargo test` passes SW Rust tests
- **Checkpoint:** `uv run pytest tests/test_swendsen_wang.py` passes
- **Benchmark autocorrelation times** using the adaptive infrastructure:
  - Metropolis at T_c on 32x32, 64x64: expect tau ~ L^2.17
  - Wolff at T_c on same lattices: expect tau ~ L^0.25
  - Include benchmark results in documentation
- **Checkpoint:** `uv run mcising run --algorithm wolff -L 32 --seed 42` works
- Add `--algorithm wolff|swendsen_wang|metropolis` to CLI

---

### Phase 4: Additional Lattice Types (Weeks 9-11)

**Goal:** Strengthen the frustrated magnetism story.

- **Triangular** (coordination 6, T_c = 4/ln(3) ~ 3.641)
- **Checkpoint:** `cargo test` passes triangular neighbor tests; physics test confirms T_c
- **Honeycomb** (coordination 3, T_c ~ 1.519)
- **Checkpoint:** `cargo test` passes honeycomb neighbor tests; physics test confirms T_c
- **Kagome** (coordination 4, no finite-T transition for AFM)
- **3D Cubic** (coordination 6, T_c ~ 4.5115)
- **1D Chain** (coordination 2, T_c = 0 - pedagogical)
- **J3 coupling** — third-nearest-neighbor interaction support (J1-J2-J3, J1-J3, J2-J3 combinations)
  - Extend `Lattice` trait with `third_nearest_neighbors()`
  - Add `j3` parameter to `LatticeConfig`, `SimulationConfig`, `IsingSimulation`
  - Add J3 term to Metropolis local field computation
  - **Checkpoint:** J1=0, J2=0, J3=1 simulation produces correct ground state energy
- Refactor `IsingSimulation::new()` to accept lattice_type + shape
- **Checkpoint:** `uv run mcising run --lattice triangular -L 32` works for each new lattice type
- **Checkpoint:** `uv run pytest tests/test_physics_validation.py` passes for all lattices

---

### Phase 5: Advanced Algorithms (Weeks 12-14)

- **Wang-Landau** - flat-histogram sampling of density of states g(E)
- **Checkpoint:** Wang-Landau g(E) reproduces canonical averages from Metropolis within statistical error
- **Parallel Tempering** - multi-replica swaps, Rayon parallelism
- **Checkpoint:** Parallel tempering improves sampling efficiency on frustrated J1-J2 system
- Full benchmark suite across all algorithms and lattice types
- **Checkpoint:** `uv run pytest` passes all tests across all algorithms and lattices

---

### Phase 6: Documentation + JOSS Paper (Weeks 15-18)

- **MkDocs Material** documentation site on GitHub Pages
- API reference via mkdocstrings (NumPy docstring style)
- Tutorials: phase transitions, J1-J2 model, correlation length
- Example scripts in `examples/`
- **paper.md** (750-1750 words) with required JOSS 2026 sections:
  - Summary, Statement of Need, State of the Field, Software Design, Research Impact, AI Usage Disclosure
- **paper.bib** with Metropolis (1953), Wolff (1989), Swendsen-Wang (1987), Wang-Landau (2001), Onsager (1944)
- **Checkpoint:** `mkdocs serve` shows complete docs site locally
- **Checkpoint:** JOSS paper compiles via `openjournals/inara` Docker image
- **Checkpoint:** Full pre-submission checklist passes (see Verification Plan below)

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Performance backend | Rust via PyO3/maturin | Maximum performance, comparable to peapods |
| Python API | Clean break (v1.0), class-based | Old API was function-oriented with 6 commits and limited users |
| Data output | HDF5 primary (h5py) | Standard in computational physics, self-describing, efficient |
| Crash safety | Incremental HDF5 writes | Each temperature is flushed to disk immediately after measurement; completed data survives crashes |
| RNG | Xoshiro256** (rand_xoshiro) | Fast, high-quality, reproducible |
| Spin representation | i8 (+1/-1) flat Vec | Cache-friendly, minimal memory |
| CLI | Typer + Rich | Modern, type-safe, beautiful output |
| Packaging | uv + maturin + pyproject.toml | Modern Python standard |
| Docs | MkDocs Material | Beautiful, widely used, auto-deploy |

---

## Verification Plan

After each phase, verify:

1. **Phase 1:** `maturin develop && pytest` passes. Run 32x32 Metropolis simulation - energy/magnetization match known analytical results at T_c. Benchmark vs old pure Python: expect 50-100x speedup.
2. **Phase 2:** `mcising run -L 16 --seed 42` produces HDF5 output with Rich progress. CI passes on all platforms.
3. **Phase 3:** Run Wolff at T_c on 64x64 lattice - autocorrelation time should be dramatically shorter than Metropolis.
4. **Phase 4:** Each lattice produces correct T_c within statistical error.
5. **Phase 5:** Wang-Landau g(E) reproduces canonical averages. Parallel tempering improves sampling of frustrated systems.
6. **Phase 6:** `mkdocs serve` shows complete docs. `openjournals/inara` compiles paper.pdf successfully.