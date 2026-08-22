# Stability & versioning

mcising follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
This page states what that promise covers.

## The public API

The public API is everything importable from the top-level `mcising`
package (see `mcising.__all__`), the documented submodules
(`mcising.statistics`, `mcising.plotting`, `mcising.io`,
`mcising.config`, `mcising.benchmarks`), the `mcising` command-line
interface, and the HDF5/JSON file formats.

**Not** covered:

- `mcising._core` (the Rust extension) and every `_`-prefixed module or
  name. `_core`'s signatures are typed and stub-checked, but its
  surface may change in minor releases; go through `Simulation` unless
  you accept that.
- Exact numerical output of stochastic runs across releases. Fixing a
  correctness bug can legitimately change sampled values and RNG
  streams; such changes are always CHANGELOG-flagged.
- Benchmark numbers in the documentation (they carry their measurement
  context).

## What v1.0 promises

From 1.0.0 onward:

- **Breaking changes only in major releases.** The 0.x line batched its
  breaking changes into the final pre-1.0 releases (see the CHANGELOG's
  "Breaking changes in 1.0" section); after 1.0 the API is frozen
  within a major version.
- **Deprecation policy.** Anything slated for removal is first
  deprecated for at least one minor release: it keeps working, emits a
  `DeprecationWarning` naming the replacement, and is documented in the
  CHANGELOG before it is removed in the next major release.
- **File-format compatibility.** HDF5 files carry a `schema_version`;
  every release reads all older schema versions. Additive fields are
  introduced with tolerant reads and no version bump; incompatible
  changes bump the schema version and are breaking-change flagged.
- **Exception contract.** Invalid configuration raises
  `ConfigurationError`, which is also a `ValueError` — `except
  ValueError` catches invalid-input errors from both the Python layer
  and the Rust core boundary. `MCIsingError` remains the base of all
  mcising-specific exceptions.

## Optional dependencies

The core simulation stack depends only on `numpy`, `h5py`, `rich`, and
`typer`. Plotting requires the `plot` extra (matplotlib);
`SimulationResults.to_dataframe` requires the `dataframe` extra
(pandas). Optional-dependency boundaries are part of the API promise:
`import mcising` will not import an optional dependency.
