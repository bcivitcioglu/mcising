# Contributing to mcising

Thanks for your interest in contributing to mcising!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/bcivitcioglu/mcising.git
   cd mcising
   ```

2. Install dependencies (requires [uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync
   ```

3. Build the Rust extension:
   ```bash
   uv run maturin develop
   ```

4. Verify everything works:
   ```bash
   uv run pytest
   cargo test
   ```

## Code Style

- **Python:** Formatted and linted with [ruff](https://docs.astral.sh/ruff/). Type-checked with [mypy](https://mypy-lang.org/) in strict mode.
- **Rust:** Formatted with `rustfmt`, linted with `clippy`.

Run checks locally:
```bash
uv run ruff check python/ tests/
uv run mypy python/mcising/
cargo fmt -- --check
cargo clippy -- -D warnings
```

## Pre-commit Hooks

Install pre-commit hooks to run checks automatically on each commit:
```bash
uv run pre-commit install
```

## Pull Requests

1. Create a feature branch from `dev`.
2. Make your changes with tests.
3. Ensure all checks pass (pytest, ruff, mypy, cargo test).
4. Open a PR against `dev`.

## Running Tests

```bash
# Python tests
uv run pytest

# Rust tests
cargo test

# With coverage
uv run pytest --cov=mcising
```