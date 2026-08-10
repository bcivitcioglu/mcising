# What

<!-- What does this PR change? One or two sentences; link the issue(s) it addresses. -->

# Why

<!-- Why is this change needed? What is the user-visible or physics consequence? -->

# Checklist

- [ ] `uv run maturin develop --release` succeeds
- [ ] `cargo fmt -- --check` clean
- [ ] `cargo clippy --all-targets -- -D warnings` clean
- [ ] `cargo test` passes
- [ ] `uv run pytest -q -m "not slow"` passes
- [ ] `uv run ruff check python/ tests/` clean
- [ ] `uv run mypy python/mcising/` clean
- [ ] New behavior is covered by tests
- [ ] `CHANGELOG.md` updated under `[Unreleased]` (user-visible changes)
