# Contributing

## Development setup

```bash
git clone https://github.com/bcivitcioglu/mcising.git
cd mcising
uv sync
uv run maturin develop
```

## Running tests

```bash
# Rust tests (141 tests)
cargo test

# Python tests (260 tests)
uv run pytest

# Every documentation code fence (Python fences also run in the canonical
# suite; the shell fences and doctests here take about a minute)
uv run pytest tests/test_docs_snippets.py -m ""

# Slow physics-validation suite (minutes; runs on every pull request and
# push to master via .github/workflows/slow.yml and gates every release)
uv run pytest -m slow

# Linting
uv run ruff check python/ tests/ scripts/ benchmarks/ examples/
uv run mypy python/mcising/ scripts/ benchmarks/ examples/
```

## Project structure

```
mcising/
├── rust/src/              # Rust core (compiled to mcising._core)
│   ├── algorithm/         # Metropolis, Wolff, Swendsen-Wang
│   ├── lattice/           # Square, triangular, honeycomb, cubic, chain
│   ├── parallel.rs        # Rayon parallel execution
│   ├── simulation.rs      # PyO3 boundary class
│   ├── observables.rs     # Energy, magnetization, correlation
│   └── autocorrelation.rs # MSER + Sokal windowing
├── python/mcising/        # Python package
│   ├── simulation.py      # High-level API
│   ├── config.py          # Frozen dataclass configs
│   ├── io.py              # HDF5/JSON I/O
│   ├── cli.py             # Typer CLI
│   └── plotting.py        # Matplotlib visualization
├── tests/                 # 401 tests
├── benchmarks/            # Performance comparison scripts
└── docs/                  # MkDocs documentation
```

## Code quality standards

**Python:** strict mypy, ruff linting, type stubs for Rust bindings.

**Rust:** `#[deny(clippy::all)]`, zero `unsafe`, no `.unwrap()` in library code, proper `Result` handling.

## Examples and figures

The scripts in `examples/` are the runnable research case: each reproduces a
known result and writes one figure. `tests/test_examples.py` runs every
script with `--quick` in the canonical suite and at its full budget in the
slow suite (every pull request), so an example that stops working fails
CI. The committed figures under `docs/assets/figures/` are the full-budget
output; regenerate them after a change that alters what they show:

```bash
python examples/onsager_reproduction.py --out docs/assets/figures
python examples/tc_binder_crossing.py --out docs/assets/figures
python examples/stripe_phase_diagram.py --out docs/assets/figures
```

## Building docs locally

```bash
uv sync --group docs
uv run mkdocs serve
```

Opens at http://localhost:8000 with live reload.
