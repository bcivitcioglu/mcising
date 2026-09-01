# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Every code fence in the documentation executes in CI. `tests/test_docs_snippets.py`
  runs the Python fences of every page in order (one namespace per page, a
  temporary working directory) in the canonical suite, runs every `mcising`
  command of the shell fences in the slow suite and in a dedicated
  "Doc snippets" CI job, and runs the package's `>>>` docstring examples as
  doctests. A snippet that no longer matches the API now fails a test.
- `IsingSimulation` is in the API reference (Simulation page), rendered from
  `python/mcising/_core.pyi`, whose every public symbol now carries a
  docstring — what `sweep` returns per algorithm, the lattice-dependent
  shape of `get_spins`, the `production_sweeps` dict, the analysis fields
  of `analyze_thermalization_series`, the two parallel runners.
  `tests/test_core_stub_docs.py` enforces the docstrings.

### Fixed

- Documentation snippets that could not run: the saving-results and
  plotting guides assumed a `results` object from another page, the
  plotting examples referenced a temperature outside their own scan and
  HDF5 files nothing had written, the cluster-algorithms example lacked
  its imports, and the CLI page's coupling-comparison plot read files no
  command created. The `SimulationResults.configurations` shape is
  documented per lattice (`(L, L, 2)` honeycomb, `(L, L, L)` cubic,
  `(L,)` chain) instead of `(n_samples, L, L)` for all.

## [0.29.0] - 2026-09-01

### Added

- `benchmarks/run_all.py` regenerates every published performance
  number. It measures eight sections — Metropolis across lattices, the
  pure-Python and NumPy baselines, the cluster algorithms with their
  autocorrelation times, throughput against lattice size, the execution
  modes against thread count, `Simulation.run()` overhead, the
  correlation-function cost and a matched-physics comparison with
  peapods — writes `benchmarks/results.json` with full provenance
  (mcising version and commit, Python, CPU and core counts, memory,
  peapods version, the complete budget) and renders the
  `<!-- benchmarks:<section>:begin/end -->` blocks in `README.md` and the
  docs. `tests/test_run_all.py` fails when a page and the JSON disagree,
  when the committed run used less than the full budget or a debug build,
  and when any page states a speed-up, throughput or timing with a digit
  outside a generated block.
- `benchmark` dependency group (`uv sync --group benchmark`) installs
  peapods for the head-to-head rows; CI does not install it.
- The cluster-algorithm table reports, next to attempted flips per second,
  the integrated autocorrelation time of the energy and of the absolute
  magnetization and the wall time per statistically independent sample
  (Metropolis 176 µs, Wolff 63 µs, Swendsen–Wang 124 µs on 32² at Tc) —
  the quantity that is actually comparable between algorithms.
- Scaling (throughput against L, flat at 320–350M updates/s from 8² to
  256²) and parallel-execution tables (independent mode 5.6× faster than
  the cooldown on 10 cores with 20 temperatures at 128², parallel
  tempering 2.0×) in `docs/advanced/performance.md`.
- `SimulationConfig.correlation_interval` (CLI `--correlation-interval`):
  evaluate the O(N²) spin-spin correlation function and correlation length
  at every k-th measurement instead of every measurement — `1` (the
  default) keeps the previous cadence, `n_sweeps // measurement_interval`
  records exactly one evaluation, at the final measurement. Applies to the
  cooldown, independent and parallel-tempering modes; adaptive mode keeps
  its single end-of-production snapshot. A `k` that would record no sample
  is rejected at construction. The cost is now documented (about 0.3 ms
  per evaluation at 16², 7 ms at 32², 130 ms at 64²).
- Golden replay fixtures: `scripts/capture_golden.py` records eleven
  fixed-seed runs covering every execution path (cooldown on all five
  lattices and three algorithms, J1-J2-h / J1-J3 / antiferromagnetic
  couplings, independent, parallel tempering, adaptive) into
  `tests/data/golden_runs.json`, and `tests/test_golden.py` replays them
  bit for bit in the canonical suite — any change to the random-number
  consumption order or the observable arithmetic now fails a test.
- The `overhead` section of `benchmarks/run_all.py` times
  `Simulation.run()` end to end on the 64², 200-sweep, measure-every-sweep
  workload.

### Changed

- Every benchmark number was re-measured on the current code (Apple M4,
  release build) and every page now carries its measurement context.
  Headline: 351M Metropolis spin updates/s on 32² at Tc, 140× pure Python,
  15× a NumPy checkerboard, 2.4× peapods 0.2.0 on a matched workload. The
  previous "430× faster than pure Python" was measured against the
  mislabelled NumPy baseline (below), and "2.7–3.4× faster than peapods"
  compared a sweeps-only timing with one that measured every sweep, a
  snapshot energy with a block average, and 100 warm-up sweeps at Tc on
  both sides. The matched comparison — same Hamiltonian, same sequential
  Metropolis sweep, 5,000 thermalization sweeps, 100,000 timed sweeps with
  the energy recorded every sweep on both sides, three seeds — agrees on
  the mean energy per site within 0.15 % on the square, triangular and
  cubic lattices and gives 2.2–2.4×. Wolff and Swendsen–Wang are no longer
  compared with peapods, which interleaves cluster updates with Metropolis
  sweeps: no workload matches.
- `mcising.benchmarks.bench_numpy` is a genuine checkerboard Metropolis
  that updates each sublattice as a whole array. The previous
  implementation ran the scalar loop over NumPy scalars — three times
  *slower* than pure Python while labelled "NumPy-vectorized". The result
  label is `NumPy (checkerboard)`; odd lattice sizes are rejected.
- The parallel-execution speed table (millisecond wall times at the noise
  floor) and the README's "~6x faster with 10 cores" comment are replaced
  by the measured thread-count table; the `268M`/`269M` copy drift is gone
  because the pages are generated. The measurement-overhead table shows
  the current numbers only — the 0.28.0 → 0.29.0 before/after is recorded
  below.
- CI lints and type-checks `benchmarks/`.
- The cooldown path (the default mode) makes two Rust calls per
  temperature instead of one per thermalization sweep plus three to five
  per measurement: `IsingSimulation.anneal` walks the ramp and
  `production_sweeps` sweeps, measures, snapshots and — when requested —
  computes the correlation observables, with the GIL released. Fixed-seed
  results are bit-identical (golden suite); the correlation bins are
  computed once per evaluation instead of twice.
- `energy_per_site` skips the J1/J2/J3/field shells whose couplings are
  exactly zero and sums the shells in integer arithmetic when every
  coupling is dyadic-exact (1, 0.5, 0.25, …), proven and tested
  bit-identical to the serial f64 sum; non-dyadic couplings such as 0.3
  keep the serial order. Per-measurement energy at 64² drops from 29.5 µs
  to 4.2 µs (J1 only), 7.1 µs (J1-J2 dyadic) or 19.4 µs (J1-J2-h
  non-dyadic). Every path measures energy — adaptive thermalization and
  the parallel runners benefit too.
- Reference workload (Metropolis 64², 100 annealing + 200 production
  sweeps measured every sweep, configurations stored, Apple M4):
  10.15 ms → 5.06 ms median (2.0×); Wolff 6.31 → 0.94 ms; Swendsen–Wang
  24.97 → 19.9 ms. Of the remaining 5 ms, 47 % is the Metropolis sweeps
  themselves, 31 % the annealing ramp and 17 % the energy measurements
  (`docs/advanced/performance.md`).
- `mcising._core.IsingSimulation.production_sweeps` (internal) returns the
  per-temperature dict the parallel runners return and accepts
  `compute_correlation` / `correlation_interval`; both runners accept
  `correlation_interval`.
- `mcising._provenance.package_version()` is cached — it was a dist-info
  disk read on every `run()`.
- The slow physics-validation suite (the Onsager u(T) curve, the
  five-seed Tc-campaign rerun) now runs on every pull request and every
  push to `master` (`.github/workflows/slow.yml`, replacing the nightly
  schedule) and gates PyPI publication in `release.yml`: a change ships
  only after the physics is re-confirmed, and a wheel is published only
  for a tag that passed.

### Removed

- `benchmarks/compare_peapods.py` and `benchmarks/measurement_overhead.py`,
  folded into `benchmarks/run_all.py` (`--sections peapods` and
  `--sections overhead`).

## [0.28.0] - 2026-09-01

### Added

- Tc campaign: `scripts/tc_campaign.py` measures the critical temperature
  of every 2D/3D lattice with the library itself — Binder-cumulant
  crossings (Swendsen–Wang, independent mode, sizes up to 64² / 24³) plus
  specific-heat peak locations, with bootstrap statistical errors, the
  finite-size drift as a separate systematic, and χ²/dof canaries on every
  fit. The committed results (`scripts/tc_campaign_results.json`) agree
  with the reference values to −0.05 % (square), −0.02 % (triangular),
  −0.09 % (honeycomb) and +0.03 % (cubic), all within one to two
  statistical errors; the table is rendered in `docs/advanced/physics.md`
  and `--reanalyse` recomputes every estimate from the stored tables
  without re-simulating.
- `tests/test_tc_campaign.py`: the committed campaign is checked against
  `mcising.constants` on every CI run (2 % gate, full-budget provenance,
  docs table in sync), and a slow-marked quick-budget rerun over every
  seed reproduces it nightly. `tests/test_constants.py` pins the closed
  forms through their defining identities.
- `.github/workflows/nightly.yml` runs the slow suite every night and on
  manual dispatch (release build, no coverage ratchet).
- Exact-results validation suite. Onsager: `<E>/site` at Tc on a 64x64
  square lattice within 1% of -sqrt(2) with the blocking error quoted in
  the assertion (measured 0.64-0.77% across seeds — the O(1/L)
  finite-size offset; L=32 would fail on that alone), plus the exact
  u(T) curve at five off-critical temperatures (slow-marked). 1D chain:
  `<E>` and signed chi against the finite-N transfer matrix at five
  temperatures within 4 sigma, via Wolff (sequential Metropolis never
  equilibrates the chain, #26). Parallel tempering vs independent mode
  within 3 sigma at every shared temperature. First-principles
  antiferromagnetic ground states on all five lattices, including the
  frustrated triangular stripe bound. Reference formulas live in
  `tests/_analytic.py` (Onsager via an AGM elliptic integral — no scipy).

### Changed

- `TC_CUBIC_3D` is now `1 / 0.221654626` = 4.511523… (Ferrenberg, Xu &
  Landau 2018) instead of the rounded literal 4.5115 — a 5 ppm change —
  and every critical-temperature constant carries its source citation.
- CI lints and type-checks `scripts/` alongside the package and tests.

## [0.27.0] - 2026-08-31

### Fixed

- `mcising run --resume` without `--checkpoint` now exits with a usage
  error instead of silently running a fresh simulation (#45).
- The `run` configuration panel reports the actual lattice type; it
  previously labeled every lattice `"…x… square"` (#46).
- `mcising run --help` no longer claims the model is 2D-only — chain
  (1D) and cubic (3D) lattices are supported (#47).
- `load_hdf5` orders temperatures numerically; lexical group sorting
  previously misordered results whenever any T ≥ 10 (#48).

### Added

- CLI test coverage for every subcommand and flag combination, and
  I/O tests pinning bit-exact round-trips (dtypes and shapes per
  lattice), JSON summary values, and error paths (corrupt/truncated
  files, unwritable paths, degraded config records). Overall coverage
  74% → 90%; the CI coverage ratchet rises from 39 to 89 and now lives
  in `pyproject.toml` (`[tool.coverage.report] fail_under`).
- Plotting figure-content tests: axis labels, error-bar containers,
  legends, the grid/subsampling and raw-array code paths of
  `plot_lattice`, and `plot_correlation` (previously untested).
  `export_lattices` gains a zip/PNG schema contract test — tree and
  flat arcnames, temperature filtering, and PNG signatures.
- Benchmark-module tests (`tests/test_benchmarks.py`): the
  pure-Python/NumPy/Rust runners at tiny sizes, the
  `BenchmarkResult` work-accounting arms, and `bench_peapods`
  against a recording stub, pinning the exact peapods call contract
  and the `ImportError` when peapods is absent.
- 19 new cargo tests for `parallel.rs` and `simulation.rs`:
  parallel-tempering swap acceptance against the Metropolis
  criterion (statistical), bit-identical results under thread-count
  changes, the exact PT ≡ independent equivalence forms,
  RNG-state serialization round-trip with bit-identical continuation
  for all three algorithms, flip/local-energy consistency, and spin
  configuration validation. `set_spins`, `set_rng_state`,
  `flip_spin`, and `spin_energy` now delegate to pure-Rust
  `*_internal` methods (no behavior change) so cargo tests can
  reach them without a Python interpreter.
- Coverage ratchet raised: `fail_under` 89 → 97 (overall coverage
  97.87%; plotting 40% → 97.5% and benchmarks 21% → 100% across the
  two test-debt phases).

## [0.26.0] - 2026-08-22

### Breaking changes in 1.0

- **matplotlib is now the optional `plot` extra** (`pip install
  'mcising[plot]'`): `import mcising` no longer imports matplotlib, and
  the plotting exports resolve lazily (PEP 562), raising a friendly
  `ImportError` naming the extra when matplotlib is absent. pandas —
  used only by `SimulationResults.to_dataframe` — is now declared as
  the `dataframe` extra (it was previously undeclared entirely).
- **`plot_observables` removed** (legacy alias): use `plot_energy` /
  `plot_magnetization` (or the other per-quantity functions) directly.
- **`ConfigurationError` replaces plain `ValueError`** in every config
  validation site, and `ConfigurationError` is now ALSO a `ValueError`
  subclass — existing `except ValueError` code keeps working, and it
  coherently covers the Rust-boundary errors (which remain
  `ValueError`).
- **CLI enum options are typed**: `--lattice`, `--algorithm`, and
  `--mode` are real choice options; an invalid value is an exit-2 usage
  error listing the valid choices instead of a `ValueError` traceback.
- **`bench_peapods_{triangular,cubic,wolff,sw}` removed**: one
  parametrized `bench_peapods(geometry=..., dim=..., temperature=...,
  cluster_mode=...)` replaces the four near-copies.
- Removed the unused `CORRELATION_THRESHOLD` constant and the
  permanently-disabled comparison branch of the CLI scaling benchmark
  (~60 dead lines).
- **One sweep signature, one temperature unit**: both
  `Simulation.sweep` and `mcising._core.IsingSimulation.sweep` are now
  `sweep(n_sweeps=1, *, temperature)`. The core previously took
  `(n_sweeps, beta)` while the high-level API took
  `(temperature, n_sweeps)` — swapped argument order AND different
  units, so mixing layers silently ran wrong physics. `temperature` is
  keyword-only: every pre-1.0 positional call fails loudly with
  `TypeError` instead of reinterpreting a `beta` as a temperature.
  `IsingSimulation.extend_thermalization` and `production_sweeps` take
  keyword-only `temperature` for the same reason; beta is internal.
- **Core `sweep` returns a 3-tuple** `(accepted, attempted,
  n_cluster_flips)`; `production_sweeps` gains the total cluster-flip
  count as a 4th element. `Simulation.sweep`'s dict gains an
  `"n_cluster_flips"` key.
- **Honest Wolff work accounting**: one Wolff sweep remains one cluster
  update, but `attempted` now reports the cluster size (Wolff is
  rejection-free; the old `attempted = num_sites` was fictitious — a
  Wolff "acceptance rate" is identically 1.0 now). Benchmark
  updates/sec counts real attempted flips, which lowers quoted Wolff
  throughput by roughly `N / ⟨cluster size⟩`. A flip-budget sweep that
  would make `n_sweeps` mean equal work across algorithms was
  implemented and rejected: measuring at its state-dependent stopping
  time is size-biased (exact-enumeration rejection at 200+ sigma); the
  unbiased calibrated design is tracked in #42.
- **`run()` resets to a fresh core** (`run(reset=True)` default):
  repeated `run()` calls on one `Simulation` are now identical, and
  manual `sweep()` calls or `spins` assignments before `run()` no
  longer leak into the run. Pass `reset=False` to continue from the
  current state (checkpoint resume does this automatically). New public
  `Simulation.reset()`.
- **Susceptibility default is the connected convention** (#39):
  `chi = N*(<m^2> - <|m|>^2)/T` everywhere chi is quoted (summary,
  DataFrame, CSV/JSON, HDF5 statistics attrs, plots) — the standard
  finite-size-scaling form. The pre-1.0 signed form `N*Var(m)/T`
  (inflated ~14.5x at Tc by global sign flips on finite lattices)
  remains available via `susceptibility(kind="signed")`. Persisted
  summaries/statistics now record `susceptibility_kind`.
- **Removed `IsingSimulation.sweep_measured`** (dead: no callers, no
  stub entry, and its windowed means could never carry error bars).
- **`thermalize_with_diagnostics` renamed to `anneal`** and returns
  `None`: the per-sweep ramp energies it returned were computed and
  discarded (P09 stopped analyzing ramps); the name promised
  diagnostics it no longer produces. RNG streams are unchanged.

### Added

- CLI parity flags for documented workflows: `--swap-interval`
  (parallel tempering — previously documented as tunable but stuck at
  1 via the CLI), `--no-store-configs`, and the adaptive knobs
  `--min-therm`, `--max-therm`, `--c-window`, `--tau-multiplier`. CLI
  defaults now come from `mcising.constants` instead of re-hardcoded
  literals.
- "Stability & versioning" documentation page: the public-API
  contract, the post-1.0 deprecation policy, file-format compatibility
  rules, and the exception contract.
- First test coverage for `AdaptiveConfig` validation and
  `SimulationResults.to_dataframe`.
- `SimulationResults.n_cluster_flips`: per-temperature cluster-flip
  count during measurement sweeps (0 for Metropolis), saved to HDF5 as
  an additive per-temperature attribute (tolerant read; no schema
  bump) — the honest work record for cluster algorithms.
- `Simulation.num_sites` public property (reads the Rust core).
- Adaptive mode now warns (`UserWarning`) when it ignores an explicitly
  set `n_sweeps` or `measurement_interval`, and when
  `adaptive.enabled` is set in independent/parallel-tempering modes
  (where adaptive is not honored). Previously both were silent.
- `py.typed` marker (PEP 561): downstream type checkers now see
  mcising's annotations and the `_core` stubs. CI verifies the stubs
  against the compiled module with `mypy.stubtest`.

### Fixed

- The cubic critical temperature was displayed inconsistently
  (`Tc=4.512` vs `4.5115`) — benchmark cases now use the
  `mcising.constants` Tc values (`TC_TRIANGULAR_2D`, `TC_HONEYCOMB_2D`,
  `TC_CUBIC_3D` are wired in rather than duplicated as literals).
- `bench_peapods` docstring referenced a nonexistent
  `uv sync --group benchmark`.
- `mcising._core.pyi` now matches the runtime module exactly
  (verified by stubtest): read-only getters are properties, `__new__`
  replaces the fictitious `__init__`, and every method signature is
  current.

## [0.25.0] - 2026-08-18

### Added

- `mcising.statistics`: autocorrelation-aware error estimation as a
  public module — blocking (Flyvbjerg–Petersen) standard errors and
  blocking curves, plateau and conservative integrated-autocorrelation-
  time estimators, delete-one-block jackknife for nonlinear estimators,
  and a total `Estimate`/`ObservableStatistics` layer that reports what
  it cannot estimate as NaN, never as a silent `0.0`. Validated against
  a moving-block bootstrap and exact AR(1) autocorrelation times.
- `SimulationResults.statistics(T)`: every observable (E, M, |M|, Cv,
  χ, Binder cumulant U4) with a principled standard error (B10, #21).
  `summary()` and `to_dataframe()` now quote errors; the Rich table
  uses compact `−1.9563(32)` notation.
- Binder cumulant U4 = 1 − ⟨m⁴⟩/(3⟨m²⟩²): `binder_cumulant` in
  `mcising.statistics`, `SimulationResults.binder_cumulant(T)`, and
  U4 ± error in summaries, JSON, CSV, and the HDF5 statistics group.
- HDF5 schema 3: each temperature group gains a derived `statistics`
  subgroup (n_samples, tau_int, value + `*_error` attribute pairs) so
  external tools read uncertainties straight from the file. Loading
  ignores it and recomputes from the raw series — no second source of
  truth; schema 1/2 files load unchanged.
- `LatticeConfig.num_sites` and cached `SimulationResults.num_sites`
  (thanks @ChickenisLegit, #24): the site count is now a pure function
  of the lattice geometry — no throwaway Rust simulation per call.
- `Lattice::dimension()` in the Rust core: the spatial dimension of
  each lattice (chain 1; square/triangular/honeycomb 2; cubic 3),
  feeding the dimension-correct correlation-length constant.
- `AdaptiveDiagnostics.stationary_sweeps`: how many fixed-temperature
  sweeps the stationarity and tau_int estimates are based on (never
  includes the annealing ramp). Written as an additive HDF5 attribute —
  no schema bump; files from older versions load with 0.
- Adaptive mode now warns (`UserWarning`) when `max_total_sweeps`
  cannot afford `min_independent_samples` at the tau-derived interval,
  and when `max_thermalization_sweeps` runs out before stationarity is
  detected — instead of silently delivering less than asked.

### Fixed

- Observable plots hardcoded zero errors for specific heat and
  susceptibility and silently fell back to a bare line plot (B10, #21):
  all four observable plots now always draw real error bars; points
  whose series is too short to quote an uncertainty render without a
  bar instead of with a fake zero-height one.
- Energy/magnetization plot error bars showed the sample spread
  (`np.std` of the series) rather than a standard error of the mean —
  wrong by ~√n and unaware of autocorrelation; they are now blocking
  standard errors.
- `SimulationResults` silently assumed 1 lattice site when the site
  count could not be inferred (B11, #22), mis-scaling Cv and χ by a
  factor of N: it now raises `ConfigurationError`.
- Plotting a results object whose temperature list and series dicts
  disagreed crashed with length-mismatched arrays; missing temperatures
  are now skipped consistently.
- Correlation length was systematically wrong (B7, #18) — three
  compounding defects: the second-moment normalization hardcoded the
  3D constant (6 = 2d) for every lattice, inflating 2D values by √1.5
  and 1D by √3; the r=0 self-term C(0) = 1 − m² entered the
  denominator, biasing ξ low; and shells were summed bin-averaged with
  their pair multiplicities discarded — the dominant error (−14.5% in
  2D even after the other two fixes). The estimator is now
  ξ² = Σ n(r)·r²·C(r) / (2d·Σ n(r)·C(r)) over r > 0, truncated at the
  first non-positive shell, validated by synthetic Ornstein–Zernike
  recovery within 5% in 2D and 3D and by a machine-precision discrete
  closed form in 1D.
- Honeycomb `distance_squared` used a unit-cell square metric that
  ignored the sublattice offset (#35): the same-cell A–B
  nearest-neighbor bond sat in the d²=0 bin, so honeycomb correlation
  output never contained its NN shell and every reported distance was
  a cell index, not a length. Now the exact Euclidean metric in
  NN-bond-length units (integer form 4d² = ΔX² + 3ΔY² from the
  armchair embedding; shells at d² = 1/3/4), verified for every site
  pair against an independent embedding oracle.
- Adaptive mode estimated tau_int across the cooldown temperature ramp
  (B9, #20): the analyzed energy series was non-stationary by
  construction and its tau set the production measurement interval.
  The ramp is now pure annealing; MSER and Sokal's method run
  exclusively on a fixed-temperature diagnostic series recorded after
  it, and the measurement interval derives from that series' stationary
  tail only.
- MSER's not-thermalized verdict was unreachable for most series
  lengths (B9, #20): the truncation search evaluated a 20-point grid
  whose largest candidate reached the classical n/2 decision boundary
  only when 20 divided n/2 — a pure linear ramp of 500 points was
  declared thermalized. Every candidate in [0, n/2] is now evaluated
  exactly (O(n) backward Welford pass), making the adaptive extension
  loop genuinely reachable; tau is additionally estimated on the
  truncation tail even when stationarity is not detected (previously a
  silently optimistic 0.5), with `is_thermalized` as the honesty flag.

### Changed

- HDF5 metadata schema 2 → 3 (see Added; older files keep loading).
- `mcising summary --json/--csv` rows carry the new error columns
  (`E_err`, `M_err`, `Cv_err`, `chi_err`, `U4`, `U4_err`, `tau_int`);
  unquotable values are omitted from JSON and left empty in CSV, never
  emitted as NaN. `save_json_summary` per-temperature entries likewise
  gain error fields.
- Correlation-length values change for every lattice (dimension-correct
  constant, r=0 exclusion, pair-multiplicity weights), and honeycomb
  correlation distances are now true bond-length distances. Raw spin
  trajectories of non-adaptive runs are bit-identical to 0.24.0.
- Adaptive-mode trajectories differ from 0.24.0: each temperature now
  runs fixed-temperature diagnostic sweeps after the annealing ramp
  (the sweeps the estimates are actually based on).

## [0.24.0] - 2026-08-17

### Fixed

- `__version__` reported "0.2.0" regardless of the installed version
  (B12, #23): it is now read from the installed distribution metadata
  via `importlib.metadata`, killing the hardcoded-constant drift class
  permanently. `mcising info` reports the real version.
- Saved files stamped `version="unknown"`: the HDF5 writer now derives
  the writing version itself, so every file (including checkpoints) is
  traceable to the code that wrote it.
- `load_hdf5` never restored the configuration (B12, #23): loaded files
  had empty plot legends, generic `mcising` export prefixes, and — with
  `store_configs=False` — per-site observables (Cv, χ) silently scaled
  by a wrong site count. The `SimulationConfig` object is now restored,
  best-effort even for pre-0.24 files.
- The config serializer's lossy fallback (`str(config)` on
  serialization failure) is gone: an unserializable config record now
  raises `ConfigurationError` at write time instead of writing a
  provenance record that would break the next resume.
- `checkpoint_run` now works in every execution mode (B4, #15). The
  independent and parallel-tempering paths previously dropped the
  checkpoint callback silently: no file was ever created while the CLI
  reported success. Cooldown still saves after each temperature;
  independent mode saves the batch when it returns; parallel tempering is
  all-or-nothing.
- Parallel tempering no longer drops measurements or panics when
  `swap_interval` does not divide `measurement_interval` (B5, #16): the
  cadence is validated at both the config and Rust boundaries, and result
  arrays derive their shape from the actual measurement count, so short
  arrays and the reshape `PanicException` are structurally impossible.
- `flip_spin`/`spin_energy` addressed the wrong site on cubic, honeycomb,
  and chain lattices (B6, #17): both computed `row*L + col` regardless of
  geometry, silently flipping/reading a different in-bounds site on 3D and
  two-sublattice lattices.
- `compute_correlation` was silently ignored in independent mode (B8,
  #19): the flag was accepted and empty result dicts were created, but
  nothing ever filled them. Correlation functions and per-measurement
  correlation lengths are now computed in both parallel modes.
- The parallel runners no longer panic on user input: non-positive, NaN,
  or empty temperature lists and zero intervals now raise `ValueError`
  instead of `PanicException` (or silently sampling at garbage β).
- Resuming a checkpoint now restores the driving simulation's spin/RNG
  state only in cooldown mode, where that state actually advanced.

### Added

- HDF5 metadata schema v2: every file now records `schema_version`,
  `version`, `seed`, `mode`, `algorithm`, the full config, and a
  best-effort `git_commit`; `load_hdf5` restores all of it. Legacy
  (mcising <= 0.23.0) files keep loading through an explicit fallback;
  files from a newer schema are refused with a clear error instead of
  loading incompletely. The same fields appear in `save_json_summary`.
- `SimulationConfig.from_dict` (and `LatticeConfig` / `AdaptiveConfig`
  counterparts): validated reconstruction from plain dicts, the inverse
  of `dataclasses.asdict` — enum coercion, tuple conversion, unknown
  keys ignored, missing keys defaulted.
- `mcising info` shows the git commit (development builds) and the HDF5
  metadata schema version.
- `SimulationConfig.store_configs` (default `True`): disable to skip
  per-measurement spin snapshots in all three modes (previously hardcoded
  on in the parallel runners).
- `compute_correlation` support in parallel tempering (the Rust runner
  gained the parameter; the cost is serial across replicas).
- Resume config guard: `checkpoint_run(..., resume=True)` refuses a
  checkpoint written with a different config (`ConfigurationError`
  naming the mismatched fields). Only `temperatures` may differ, so a
  scan can be extended; independent-mode resume keeps each surviving
  temperature's original RNG stream via seed offsets.

### Changed

- **Breaking**: `checkpoint_run` with `resume=False` (the default) now
  raises `ConfigurationError` when the checkpoint file already exists;
  previously it retained the old file's metadata and either crashed
  mid-write on colliding temperatures or silently merged two runs'
  ensembles into one file.
- **Breaking**: `mcising summary --json` prints an object with
  top-level provenance fields and the per-temperature rows under
  `"results"` (previously a bare array of rows).
- **Breaking**: `_core.IsingSimulation.flip_spin(site)` and
  `spin_energy(site)` now take a single flat site index instead of
  `(row, col)` — the only scheme that addresses every lattice geometry.
- **Breaking**: parallel-tempering configs with
  `measurement_interval % swap_interval != 0` are now rejected with
  `ConfigurationError`; they previously ran and silently lost
  measurements.
- `on_temperature_complete` and `skip_temperatures` are accepted by
  `Simulation.run()` in all three modes (previously cooldown-only,
  silently ignored elsewhere). In parallel tempering, skipping a proper
  subset of the ladder raises (the replicas form one coupled ensemble).

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
