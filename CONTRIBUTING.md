# Contributing to mcising

Thanks for your interest in contributing to mcising!

## Getting Help

- **Questions and usage help:** open a
  [GitHub issue](https://github.com/bcivitcioglu/mcising/issues) — questions
  are welcome there; there is no separate forum.
- **Bug reports:** use the bug-report issue template and include the mcising
  version, your platform, and a minimal script that reproduces the problem.
- **Feature proposals:** open an issue describing the use case first, so the
  design can be discussed before any code is written.
- **Private matters:** email bcivitcioglu@gmail.com.

All interactions are covered by the [Code of Conduct](CODE_OF_CONDUCT.md).

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
uv run ruff check python/ tests/ scripts/ benchmarks/
uv run mypy python/mcising/ scripts/ benchmarks/
cargo fmt -- --check
cargo clippy -- -D warnings
```

## Pre-commit Hooks

`pre-commit` is part of the dev dependency group (installed by `uv sync`).
Install the hooks to run checks automatically on each commit:
```bash
uv run pre-commit install
```

## Pull Requests

1. Create a feature branch from `master`.
2. Make your changes with tests, and note user-visible changes in
   `CHANGELOG.md` under `[Unreleased]`.
3. Ensure all checks pass (pytest, ruff, mypy, cargo test, clippy).
4. Open a PR against `master` and fill in the pull-request template.

## Running Tests

```bash
# Python tests
uv run pytest

# Rust tests
cargo test

# With coverage
uv run pytest --cov=mcising

# The slow suite (physics-validation runs taking minutes: the Onsager
# u(T) curve, the five-seed Tc-campaign rerun). The fast CI matrix
# excludes it; .github/workflows/slow.yml runs it on every pull request
# and push to master, release.yml runs it before publishing, and it can
# be dispatched manually.
uv run pytest -m slow
```
