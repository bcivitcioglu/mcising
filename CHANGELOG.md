# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.23.0] - 2026-08-13

### Added

- Cross-lattice geometry test matrix (Rust, test-only): every lattice ×
  shell × size combination is verified for reciprocal bonds, exact
  coordination, shell disjointness, no self-bonds, and — decisively — the
  exact Euclidean distance of every neighbor-table entry in the embedding
  the NN table realizes. This is the assertion class that caught the
  honeycomb defect below.
- A brute-force pair-sum energy reference for the triangular lattice with
  all three couplings active, agreeing with the table-driven energy to
  1e-10 (guards against shell double-counting).
- Exact-enumeration test oracle (Rust, test-only): the full density of
  states of the 4×4 square torus (65 536 states) and the 12-site periodic
  chain (4 096 states), validated against the closed-form transfer matrix
  and high-/zero-temperature limits. Metropolis (both coupling signs),
  Wolff, and Swendsen-Wang long runs now reproduce exact ⟨E⟩ within 0.5%
  at T ∈ {1.0, 2.269, 4.0}, with an enforced statistical power floor.
- True detailed-balance test: the empirical state-visit histogram on a
  3×3 torus is compared against exact Boltzmann weights with a
  KL-divergence threshold derived from the chi-square law of the
  G statistic (previous "detailed balance" test checked stationarity only).

### Changed

- Triangular and honeycomb lattices now require an even size L and reject
  odd L with a clear error (`ConfigurationError` from `LatticeConfig`;
  `ValueError` from the Rust core and both parallel runners). With
  row-parity offset coordinates, rows 0 and L-1 share a parity when L is
  odd, so bonds across the vertical wrap seam were not reciprocal and the
  Hamiltonian was silently invalid. Correct odd-L periodic wraps are
  research-shaped and remain future work.
- Cluster algorithms (Wolff, Swendsen-Wang) now reject `j1 <= 0` with a
  clear error instead of silently degenerating into random single spin
  flips (the bond probability `1 − exp(−2βJ1)` is not a probability for
  antiferromagnetic couplings; the previous guard was a `debug_assert`
  that vanished in release builds). `SimulationConfig` raises
  `ConfigurationError`; the Rust core and both parallel runners raise
  `ValueError`. Use `metropolis` for antiferromagnetic couplings —
  sublattice-mapped cluster updates remain future work.

### Fixed

- Triangular third-neighbor (J3) table: two of six entries per site
  duplicated next-nearest-neighbor sites (double-counting J2 and J3 bonds
  when both couplings were active) and the shell was not reciprocal. The
  corrected shell is parity-independent and sits at the exact TNN
  distance 2.
- Honeycomb second- (J2) and third-neighbor (J3) tables: four of six NNN
  entries and two of three TNN entries per site pointed at sites far
  outside their shells (distances 3, sqrt(13), and sqrt(21) instead of
  sqrt(3) and 2), so any honeycomb run with J2 or J3 active sampled a
  geometrically invalid Hamiltonian. Both tables are rebuilt from the
  armchair-row embedding, with the derivation documented in the source.
- Triangular `distance_squared` now returns the exact Euclidean squared
  distance in the 60-degree basis (an exact integer: d² = 1, 3, 4 for the
  three shells) instead of a square-grid approximation, fixing the
  distance axis of triangular correlation functions.
- Antiferromagnetic (J<0) single-coupling Metropolis now samples the
  Boltzmann distribution. The J1-, J2-, and J3-only sweep strategies branched
  on the neighbor-sum sign instead of the energy sign, so every proposed flip
  was accepted and any single-coupling J<0 simulation silently sampled T=∞
  configurations at every requested temperature. Acceptance tables are now
  keyed on |β·J| with the coupling sign folded into the energy branch;
  ferromagnetic (J>0) runs are bit-for-bit unchanged. A new antiferromagnetic
  regression suite covers Néel ground-state energies, low-temperature
  acceptance, and single- vs multi-coupling path agreement.
- The release workflow now builds wheels for every supported CPython
  (3.10–3.13) on all platforms. Previously the Linux jobs failed outright
  (no interpreter visible inside the manylinux container) and macOS/Windows
  shipped wheels only for CPython 3.12.

## [0.22.0] - 2026-08-08

### Added

- Community and citation files: `CODE_OF_CONDUCT.md` (Contributor Covenant
  2.1), `CITATION.cff`, and a pull-request template with a gates checklist.
- Public issue backlog: all 12 known defects from the pre-1.0 audit are filed
  as GitHub issues (#12–#23) with file:line evidence, ahead of their fixes.
- `pre-commit` added to the dev dependency group; CONTRIBUTING now documents
  support pathways (questions, bug reports, feature proposals) and targets
  `master` instead of the retired `dev` branch.
- Retroactive changelog entries for 0.2.0 through 0.21.0, reconstructed from
  commit messages.
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

- Development-status classifier promoted from Alpha to Beta; license metadata
  expressed as an SPDX string (PEP 639); build requires maturin ≥ 1.8.
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

<!-- Versions below this line predate tagging and publication. They were
     reconstructed retroactively from the commit messages; dates are commit
     dates. Version numbers 0.4–0.7 and 0.16 were never used. -->

## [0.21.0] - 2026-04-11

### Changed

- Improved CLI export progress information.

## [0.20.0] - 2026-04-11

### Changed

- Improved the CLI.

## [0.19.0] - 2026-04-10

### Changed

- Improved plotting functionality and post-processing.

## [0.18.0] - 2026-04-09

### Added

- Documentation site built with MkDocs.

## [0.17.0] - 2026-04-08

### Added

- Parallel tempering and independent-run execution modes.

## [0.15.0] - 2026-04-07

### Changed

- Proper benchmarking; comparative benchmarking separated into the README and
  a dedicated script.

## [0.14.0] - 2026-04-07

### Added

- Full comparative benchmarking.

## [0.13.0] - 2026-04-07

### Added

- Full cross-lattice test coverage.

## [0.12.0] - 2026-04-07

### Added

- Triangular, cubic, and honeycomb lattices, and the 1D chain.

## [0.11.2] - 2026-04-07

- Maintenance release; the commit message records no details.

## [0.11.0] - 2026-04-03

### Added

- Third-neighbor coupling (J3) support.

## [0.10.0] - 2026-04-03

### Changed

- Speed optimization.

## [0.9.0] - 2026-04-02

### Changed

- Energy benchmarking aligned with the other libraries.

## [0.8.0] - 2026-04-02

### Added

- Sequential Metropolis sweep; `--T-range` option in the CLI.

### Changed

- Optimized for speed.

## [0.3.0] - 2026-04-02

### Added

- Benchmarking.

## [0.2.6] - 2026-04-01

### Added

- Adaptive thermalization.

## [0.2.5] - 2026-03-31

### Fixed

- CI/CD corrections.

## [0.2.4] - 2026-03-31

### Added

- CI/CD.

## [0.2.3] - 2026-03-31

### Added

- Restore-and-continue logic.

## [0.2.2] - 2026-03-31

### Added

- CLI rich output and visualizations.

## [0.2.1] - 2026-03-31

### Changed

- mypy, ruff, and black all passing.

## [0.2.0] - 2026-03-31

### Changed

- Rust core implemented for the new mcising — a complete rewrite of the
  original pure-Python package (June 2024, preserved in the git history).
