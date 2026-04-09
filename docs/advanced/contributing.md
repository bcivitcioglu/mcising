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

# Linting
uv run ruff check python/ tests/
uv run mypy python/mcising/
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

## Building docs locally

```bash
uv sync --group docs
uv run mkdocs serve
```

Opens at http://localhost:8000 with live reload.
