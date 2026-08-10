# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CI builds the Rust extension in release mode, measures Python test coverage
  with a ratchet (`--cov-fail-under=39`), and builds the documentation with
  `mkdocs build --strict`.
- Documentation deploys to GitHub Pages automatically on every push to master
  (`.github/workflows/docs.yml`).
- Dependabot updates for pip, cargo, and GitHub Actions dependencies.
- LaTeX in the documentation now renders, via MathJax
  (`pymdownx.arithmatex`); the physics background page displays its equations
  correctly.
- The original v0.13 → v0.21 transformation roadmap is preserved as
  `docs/advanced/history.md`.
- `Cargo.lock` and `uv.lock` are committed for reproducible developer and CI
  environments.

### Fixed

- Antiferromagnetic (J<0) single-coupling Metropolis now samples the
  Boltzmann distribution. The J1-, J2-, and J3-only sweep strategies branched
  on the neighbor-sum sign instead of the energy sign, so every proposed flip
  was accepted and any single-coupling J<0 simulation silently sampled T=∞
  configurations at every requested temperature. Acceptance tables are now
  keyed on |β·J| with the coupling sign folded into the energy branch;
  ferromagnetic (J>0) runs are bit-for-bit unchanged. A new antiferromagnetic
  regression suite covers Néel ground-state energies, low-temperature
  acceptance, and single- vs multi-coupling path agreement.
- Rust sources pass `cargo fmt --check` and
  `cargo clippy --all-targets -- -D warnings` (135 lint errors resolved).
  Sites belonging to known open bugs (B1, B2, B5, B8, B11) are preserved
  behind documented `#[allow]`s so the owning fixes remain visible.
- `mypy python/mcising/` passes: mypy now targets Python 3.12 (numpy's stubs
  use PEP 695 syntax rejected under older targets) and `peapods` is ignored
  via a mypy override instead of a mismatched inline ignore code.

### Removed

- 70 built HTML files (`site/`), the pre-Rust `_legacy/` package, and
  `.DS_Store` files are no longer tracked in the repository.
- The unused `mike` version-provider entry in `mkdocs.yml` (docs deploy as a
  single version).

### Changed

- Test suite hardened for seed robustness: statistical assertions now use
  autocorrelation-aware blocking standard errors (`tests/_stats.py`) and run
  over five fixed seeds instead of one, with thresholds anchored to analytic
  values (Onsager spontaneous magnetization, high-temperature expansion,
  stripe-state energy plateau).
- Sub-critical-temperature tests anneal through T_c via a descending
  temperature ladder instead of quenching, eliminating stripe-state
  false failures.
- Registered `slow` and `statistical` pytest markers (`--strict-markers`);
  the wall-clock performance assertion in the independent-mode test was
  replaced with correctness and cross-mode consistency checks and marked
  `slow`.
