# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
