//! Parallel execution paths: independent temperatures and parallel tempering.
//!
//! Independent mode gives each temperature its own `IsingSimulation` with a
//! unique RNG seed and runs all of them simultaneously via Rayon — no shared
//! mutable state, pure data parallelism. Parallel tempering runs one coupled
//! replica ladder with periodic swap attempts between adjacent temperatures.
//!
//! All user-reachable failures are rejected up front by the validators in
//! this file; the hot loops themselves are panic-free for validated input.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::Rng;
use rayon::prelude::*;

use crate::error::MCIsingError;
use crate::lattice::{with_lattice, Lattice, LatticeKind};
use crate::measurement::TempResult;
use crate::observables::{self, ShellSums};
use crate::simulation::IsingSimulation;

/// Reject invalid run parameters before entering the parallel sections.
///
/// The per-replica constructors run inside Rayon closures where an error
/// cannot surface as a Python exception, so every user-reachable failure
/// (bad couplings/algorithm/lattice, non-positive or non-finite
/// temperatures, empty temperature list, zero measurement interval) must
/// be caught here at the boundary.
fn validate_run_params(
    lattice_size: usize,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
    base_seed: u64,
    algorithm: &str,
    lattice_type: &str,
    temperatures: &[f64],
    measurement_interval: usize,
    correlation_interval: usize,
) -> Result<(), MCIsingError> {
    IsingSimulation::new_internal(
        lattice_size,
        j1,
        j2,
        j3,
        h,
        base_seed,
        algorithm,
        lattice_type,
    )?;
    if temperatures.is_empty() {
        return Err(MCIsingError::EmptyTemperatureList);
    }
    for &temp in temperatures {
        if !temp.is_finite() || temp <= 0.0 {
            return Err(MCIsingError::InvalidTemperature(temp));
        }
    }
    if measurement_interval < 1 {
        return Err(MCIsingError::InvalidInterval(
            "measurement_interval",
            measurement_interval,
        ));
    }
    if correlation_interval < 1 {
        return Err(MCIsingError::InvalidInterval(
            "correlation_interval",
            correlation_interval,
        ));
    }
    Ok(())
}

/// Reject swap/measurement cadences that would drop measurements (B5).
fn validate_swap_cadence(
    measurement_interval: usize,
    swap_interval: usize,
) -> Result<(), MCIsingError> {
    if swap_interval < 1 {
        return Err(MCIsingError::InvalidInterval(
            "swap_interval",
            swap_interval,
        ));
    }
    if !measurement_interval.is_multiple_of(swap_interval) {
        return Err(MCIsingError::IncompatibleSwapCadence(
            measurement_interval,
            swap_interval,
        ));
    }
    Ok(())
}

/// Independent-temperature runner (pure Rust; the `#[pyfunction]` wrapper
/// releases the GIL around this).
///
/// `seed_offsets` decouples each temperature's RNG seed from its position
/// in `temperatures`: entry `i` is added to `base_seed` for replica `i`.
/// `None` means identity (`seed = base_seed + i`). A resumed run passes the
/// index each surviving temperature had in the original full scan, so its
/// streams are identical to the uninterrupted run's.
fn run_independent_internal(
    lattice_size: usize,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
    base_seed: u64,
    algorithm: &str,
    lattice_type: &str,
    temperatures: &[f64],
    n_thermalization: usize,
    n_sweeps: usize,
    measurement_interval: usize,
    store_configs: bool,
    compute_correlation: bool,
    correlation_interval: usize,
    seed_offsets: Option<&[u64]>,
) -> Result<Vec<TempResult>, MCIsingError> {
    validate_run_params(
        lattice_size,
        j1,
        j2,
        j3,
        h,
        base_seed,
        algorithm,
        lattice_type,
        temperatures,
        measurement_interval,
        correlation_interval,
    )?;
    if let Some(offsets) = seed_offsets {
        if offsets.len() != temperatures.len() {
            return Err(MCIsingError::InvalidSeedOffsets(
                offsets.len(),
                temperatures.len(),
            ));
        }
    }

    let n_measurements = n_sweeps / measurement_interval;

    temperatures
        .par_iter()
        .enumerate()
        .map(|(i, &temp)| {
            let offset = seed_offsets.map_or(i as u64, |offsets| offsets[i]);
            let beta = 1.0 / temp;

            // Each thread gets its own simulation — no shared state.
            let mut sim = IsingSimulation::new_internal(
                lattice_size,
                j1,
                j2,
                j3,
                h,
                base_seed.wrapping_add(offset),
                algorithm,
                lattice_type,
            )?;

            let num_sites = with_lattice!(&sim.lattice, lat => lat.num_sites());
            let shape = with_lattice!(&sim.lattice, lat => lat.shape().to_vec());

            // Thermalize from random initialization at this temperature.
            sim.sweep_internal(n_thermalization, beta);

            let mut result = TempResult::with_capacity(
                temp,
                n_measurements,
                num_sites,
                shape,
                store_configs,
                compute_correlation,
                correlation_interval,
            );

            for _ in 0..n_measurements {
                result.cluster_flips +=
                    sim.sweep_internal(measurement_interval, beta).cluster_flips;
                with_lattice!(&sim.lattice, lat => {
                    let energy =
                        observables::energy_per_site(&sim.spins, lat, j1, j2, j3, h);
                    result.push(&sim.spins, lat, energy);
                });
            }

            Ok(result)
        })
        .collect()
}

/// Run independent simulations at multiple temperatures in parallel.
///
/// Each temperature starts from a random spin configuration with a
/// deterministic seed (`base_seed` + its seed offset; the offset defaults
/// to the temperature's index). All temperatures execute simultaneously
/// via Rayon's thread pool.
///
/// Returns a list of dicts, one per temperature, containing energy and
/// magnetization arrays, plus configuration and correlation arrays when
/// requested.
///
/// # Errors
///
/// Returns an error if `algorithm` or `lattice_type` is not recognized,
/// the lattice cannot be constructed at `lattice_size`, `temperatures` is
/// empty or contains a non-positive/non-finite value,
/// `measurement_interval` is zero, or `seed_offsets` has a different
/// length than `temperatures`.
#[pyfunction]
#[pyo3(signature = (
    lattice_size, j1, j2, j3, h, base_seed, algorithm, lattice_type,
    temperatures, n_thermalization, n_sweeps, measurement_interval,
    store_configs = false, compute_correlation = false, seed_offsets = None,
    correlation_interval = 1
))]
pub fn run_independent_temperatures<'py>(
    py: Python<'py>,
    lattice_size: usize,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
    base_seed: u64,
    algorithm: &str,
    lattice_type: &str,
    temperatures: Vec<f64>,
    n_thermalization: usize,
    n_sweeps: usize,
    measurement_interval: usize,
    store_configs: bool,
    compute_correlation: bool,
    seed_offsets: Option<Vec<u64>>,
    correlation_interval: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    // Clone strings for use inside the GIL-free section (which requires Send).
    let algo = algorithm.to_string();
    let lat_type = lattice_type.to_string();

    // Release the GIL while Rayon does the heavy lifting.
    let results = py.allow_threads(|| {
        run_independent_internal(
            lattice_size,
            j1,
            j2,
            j3,
            h,
            base_seed,
            &algo,
            &lat_type,
            &temperatures,
            n_thermalization,
            n_sweeps,
            measurement_interval,
            store_configs,
            compute_correlation,
            correlation_interval,
            seed_offsets.as_deref(),
        )
    })?;

    convert_results_to_py(py, results)
}

/// Convert `TempResult`s to Python dicts (one per temperature).
fn convert_results_to_py(
    py: Python<'_>,
    results: Vec<TempResult>,
) -> PyResult<Vec<Bound<'_, PyDict>>> {
    results.into_iter().map(|r| r.into_pydict(py)).collect()
}

// ═══════════════════════════════════════════════════════════════════
// Parallel Tempering
// ═══════════════════════════════════════════════════════════════════

/// Build one replica per temperature with consecutive seeds.
fn build_replicas(
    n_temps: usize,
    lattice_size: usize,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
    base_seed: u64,
    algorithm: &str,
    lattice_type: &str,
) -> Result<Vec<IsingSimulation>, MCIsingError> {
    (0..n_temps)
        .map(|i| {
            let mut sim = IsingSimulation::new_internal(
                lattice_size,
                j1,
                j2,
                j3,
                h,
                base_seed.wrapping_add(i as u64),
                algorithm,
                lattice_type,
            )?;
            // The ladder carries the shell sums forward; Wolff reports its
            // cluster boundary only on request.
            sim.set_track_shell_deltas(true);
            Ok(sim)
        })
        .collect()
}

/// Which extreme rung a replica visited last, for round-trip counting.
///
/// A replica is labelled `Up` when it sits on the coldest rung and `Down`
/// when, labelled `Up`, it reaches the hottest one; a round trip
/// (coldest → hottest → coldest) is counted each time a `Down` replica is
/// back on the coldest rung.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Unlabelled,
    Up,
    Down,
}

/// Ladder-level parallel-tempering diagnostics, handed to Python next to
/// the per-temperature results.
///
/// Pair `i` couples rungs `i` and `i + 1` of the ascending temperature
/// ladder (`swap_*` have `n_temps - 1` entries); `round_trips[r]` counts
/// the completed coldest → hottest → coldest excursions of the replica
/// that started on rung `r` (`n_temps` entries). Only production rounds
/// are counted — thermalization never swaps.
pub(crate) struct PtDiagnostics {
    pub(crate) temperatures: Vec<f64>,
    pub(crate) swap_attempted: Vec<usize>,
    pub(crate) swap_accepted: Vec<usize>,
    pub(crate) round_trips: Vec<usize>,
}

impl PtDiagnostics {
    fn new(temperatures: Vec<f64>) -> Self {
        let n = temperatures.len();
        Self {
            temperatures,
            swap_attempted: vec![0; n.saturating_sub(1)],
            swap_accepted: vec![0; n.saturating_sub(1)],
            round_trips: vec![0; n],
        }
    }

    /// Convert to the Python dict `run_parallel_tempering` returns as its
    /// second element. Keys: `temperatures`, `swap_attempted`,
    /// `swap_accepted`, `round_trips`.
    pub(crate) fn into_pydict(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("temperatures", self.temperatures)?;
        dict.set_item("swap_attempted", self.swap_attempted)?;
        dict.set_item("swap_accepted", self.swap_accepted)?;
        dict.set_item("round_trips", self.round_trips)?;
        Ok(dict)
    }
}

/// The replica ladder of a parallel-tempering run.
///
/// Rung `i` is pinned to `betas[i]`; a swap exchanges the spin
/// configurations of two adjacent rungs together with everything derived
/// from them (`shells`, `energies`, `measured`) and the replica identity.
///
/// The swap criterion needs every rung's energy after every round. Rather
/// than re-summing each lattice (an `O(N)` pass per rung that used to run
/// serially after the parallel sweeps), the ladder carries the integer
/// shell sums and applies the exact per-sweep deltas the algorithms report
/// (`SweepResult::delta`), so the swap energy costs `O(1)` per rung. The
/// energies recorded at measurement points are still evaluated directly
/// from the spins — inside the parallel section — so the measurement series
/// is bit-identical to a full recompute.
struct Ladder {
    replicas: Vec<IsingSimulation>,
    /// Ordered-pair shell sums of each rung's current configuration.
    shells: Vec<ShellSums>,
    /// Energy per site of each rung, derived from `shells`.
    energies: Vec<f64>,
    /// Exact energy per site at the end of the latest measurement round;
    /// stale on other rounds. Permuted by swaps like the spins.
    measured: Vec<f64>,
    /// Identity of the configuration on each rung (its starting rung).
    replica_id: Vec<usize>,
    /// Round-trip label of each replica identity.
    direction: Vec<Direction>,
    stats: PtDiagnostics,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
    num_sites: usize,
}

impl Ladder {
    fn new(
        replicas: Vec<IsingSimulation>,
        temperatures: Vec<f64>,
        j1: f64,
        j2: f64,
        j3: f64,
        h: f64,
    ) -> Self {
        let n = replicas.len();
        let num_sites = with_lattice!(&replicas[0].lattice, lat => lat.num_sites());
        let shells: Vec<ShellSums> = replicas
            .iter()
            .map(|sim| Self::shell_sums_of(sim, j1, j2, j3))
            .collect();
        let energies = shells
            .iter()
            .map(|sums| observables::energy_from_shells(sums, j1, j2, j3, h, num_sites))
            .collect();
        let mut ladder = Self {
            replicas,
            shells,
            energies,
            measured: vec![0.0; n],
            replica_id: (0..n).collect(),
            direction: vec![Direction::Unlabelled; n],
            stats: PtDiagnostics::new(temperatures),
            j1,
            j2,
            j3,
            h,
            num_sites,
        };
        // The starting rung counts as a visit for the round-trip labels.
        ladder.record_round_trips();
        ladder
    }

    /// A ladder with rigged energies for direct `attempt_swaps` tests.
    #[cfg(test)]
    fn rigged(replicas: Vec<IsingSimulation>, energies: Vec<f64>) -> Self {
        let n = replicas.len();
        assert_eq!(energies.len(), n);
        let temperatures = (0..n).map(|i| i as f64 + 1.0).collect();
        let mut ladder = Self::new(replicas, temperatures, 1.0, 0.0, 0.0, 0.0);
        ladder.energies = energies;
        ladder
    }

    fn shell_sums_of(sim: &IsingSimulation, j1: f64, j2: f64, j3: f64) -> ShellSums {
        with_lattice!(&sim.lattice, lat => {
            observables::shell_sums(&sim.spins, lat, j1 != 0.0, j2 != 0.0, j3 != 0.0)
        })
    }

    /// One parallel round: `sweeps` sweeps on every rung, the shell sums
    /// and swap energies brought up to date, and — when `measure` — the
    /// exact measurement energy evaluated. Returns the cluster flips per
    /// rung (rung `i` is pinned to `betas[i]`, so the attribution is exact).
    fn sweep_round(&mut self, sweeps: usize, betas: &[f64], measure: bool) -> Vec<usize> {
        let (j1, j2, j3, h, num_sites) = (self.j1, self.j2, self.j3, self.h, self.num_sites);
        self.replicas
            .par_iter_mut()
            .zip(self.shells.par_iter_mut())
            .zip(self.energies.par_iter_mut())
            .zip(self.measured.par_iter_mut())
            .enumerate()
            .map(|(i, (((sim, shell), energy), measured))| {
                let result = sim.sweep_internal(sweeps, betas[i]);
                match result.delta {
                    Some(delta) => *shell += delta,
                    None => *shell = Self::shell_sums_of(sim, j1, j2, j3),
                }
                *energy = observables::energy_from_shells(shell, j1, j2, j3, h, num_sites);
                if measure {
                    debug_assert_eq!(
                        *shell,
                        Self::shell_sums_of(sim, j1, j2, j3),
                        "tracked shell sums drifted on rung {i}"
                    );
                    *measured = with_lattice!(&sim.lattice, lat => {
                        observables::energy_per_site(&sim.spins, lat, j1, j2, j3, h)
                    });
                }
                result.cluster_flips
            })
            .collect()
    }

    /// Attempt swaps between adjacent rungs (even/odd alternation).
    fn attempt_swaps(&mut self, betas: &[f64], round: usize, swap_rng: &mut impl Rng) {
        let offset = usize::from(!round.is_multiple_of(2));
        for i in (offset..self.replicas.len().saturating_sub(1)).step_by(2) {
            let j = i + 1;
            // Standard PT acceptance: P = min(1, exp(delta))
            // where delta = (β_i - β_j) * (E_i - E_j) * N
            // β sorted descending (β_i > β_j), so if E_i < E_j
            // (low-T replica has lower energy), delta > 0 → always accept.
            let delta = (betas[i] - betas[j])
                * (self.energies[i] - self.energies[j])
                * self.num_sites as f64;
            let accept = delta >= 0.0 || swap_rng.gen::<f64>() < delta.exp();
            self.stats.swap_attempted[i] += 1;
            if accept {
                self.stats.swap_accepted[i] += 1;
                // O(1) pointer swap of spin Vecs, plus everything derived
                // from the configuration and its identity.
                let (left, right) = self.replicas.split_at_mut(j);
                std::mem::swap(&mut left[i].spins, &mut right[0].spins);
                self.shells.swap(i, j);
                self.energies.swap(i, j);
                self.measured.swap(i, j);
                self.replica_id.swap(i, j);
            }
        }
    }

    /// Update the round-trip labels from the current rung occupancy.
    fn record_round_trips(&mut self) {
        let n = self.replicas.len();
        if n < 2 {
            return;
        }
        let coldest = self.replica_id[0];
        if self.direction[coldest] == Direction::Down {
            self.stats.round_trips[coldest] += 1;
        }
        self.direction[coldest] = Direction::Up;
        let hottest = self.replica_id[n - 1];
        if self.direction[hottest] == Direction::Up {
            self.direction[hottest] = Direction::Down;
        }
    }

    /// Record one measurement per rung from the energies evaluated in the
    /// latest (measurement) round.
    fn record_measurements(&self, temp_results: &mut [TempResult]) {
        for (i, sim) in self.replicas.iter().enumerate() {
            let energy = self.measured[i];
            with_lattice!(&sim.lattice, lat => {
                temp_results[i].push(&sim.spins, lat, energy);
            });
        }
    }
}

/// Parallel-tempering runner (pure Rust; the `#[pyfunction]` wrapper
/// releases the GIL around this).
fn run_parallel_tempering_internal(
    lattice_size: usize,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
    base_seed: u64,
    algorithm: &str,
    lattice_type: &str,
    temperatures: &[f64],
    n_thermalization: usize,
    n_sweeps: usize,
    measurement_interval: usize,
    swap_interval: usize,
    store_configs: bool,
    compute_correlation: bool,
    correlation_interval: usize,
) -> Result<(Vec<TempResult>, PtDiagnostics), MCIsingError> {
    validate_run_params(
        lattice_size,
        j1,
        j2,
        j3,
        h,
        base_seed,
        algorithm,
        lattice_type,
        temperatures,
        measurement_interval,
        correlation_interval,
    )?;
    validate_swap_cadence(measurement_interval, swap_interval)?;

    let n_temps = temperatures.len();
    let n_measurements = n_sweeps / measurement_interval;

    // Sort temperatures ascending for swap logic. Validation rejected NaN,
    // so total_cmp agrees with the usual order on the remaining values.
    let mut sorted_temps = temperatures.to_vec();
    sorted_temps.sort_by(f64::total_cmp);
    let betas: Vec<f64> = sorted_temps.iter().map(|&t| 1.0 / t).collect();

    // Create one replica per temperature.
    let mut replicas = build_replicas(
        n_temps,
        lattice_size,
        j1,
        j2,
        j3,
        h,
        base_seed,
        algorithm,
        lattice_type,
    )?;

    let num_sites = with_lattice!(&replicas[0].lattice, lat => lat.num_sites());
    let shape = with_lattice!(&replicas[0].lattice, lat => lat.shape().to_vec());

    // Thermalize all replicas in parallel.
    replicas.par_iter_mut().enumerate().for_each(|(i, sim)| {
        sim.sweep_internal(n_thermalization, betas[i]);
    });

    // Separate RNG for swap decisions (deterministic, independent of replica RNGs).
    let mut swap_rng = crate::rng::create_rng(base_seed.wrapping_add(n_temps as u64 + 1000));

    // Pre-allocate result storage.
    let mut temp_results: Vec<TempResult> = sorted_temps
        .iter()
        .map(|&t| {
            TempResult::with_capacity(
                t,
                n_measurements,
                num_sites,
                shape.clone(),
                store_configs,
                compute_correlation,
                correlation_interval,
            )
        })
        .collect();

    let mut ladder = Ladder::new(replicas, sorted_temps, j1, j2, j3, h);
    let mut sweep_count: usize = 0;
    let mut round: usize = 0;

    while sweep_count < n_sweeps {
        // a. Parallel sweeps (with the swap energies and, on measurement
        // rounds, the exact measurement energies brought up to date).
        // With the cadence validated (swap_interval divides
        // measurement_interval), sweep_count hits every multiple of
        // measurement_interval exactly once.
        let sweeps_this_round = swap_interval.min(n_sweeps - sweep_count);
        sweep_count += sweeps_this_round;
        let measure = sweep_count.is_multiple_of(measurement_interval);
        let round_flips = ladder.sweep_round(sweeps_this_round, &betas, measure);
        for (result, flips) in temp_results.iter_mut().zip(&round_flips) {
            result.cluster_flips += flips;
        }

        // b. Swap attempts (even/odd alternation) and round-trip labels.
        ladder.attempt_swaps(&betas, round, &mut swap_rng);
        ladder.record_round_trips();
        round += 1;

        // c. Collect measurements at the right intervals.
        if measure {
            ladder.record_measurements(&mut temp_results);
        }
    }

    Ok((temp_results, ladder.stats))
}

/// Run Parallel Tempering: N replicas at different temperatures with
/// periodic swap attempts between adjacent replicas.
///
/// Swaps use the standard Metropolis criterion:
///   P(swap i,j) = min(1, exp((β_i - β_j) × (E_i - E_j)))
///
/// Even/odd alternation ensures all adjacent pairs get swap opportunities.
///
/// Returns `(results, diagnostics)`: one dict per temperature (ascending)
/// and a ladder-level dict with the swap statistics per adjacent pair and
/// the round-trip count per replica (see `PtDiagnostics`).
///
/// When `compute_correlation` is requested, the correlation observables are
/// computed serially across replicas at each measurement (an `O(N²)`-in-sites
/// cost; parallelizing it is future performance work).
///
/// # Errors
///
/// Returns an error if `algorithm` or `lattice_type` is not recognized,
/// the lattice cannot be constructed at `lattice_size`, `temperatures` is
/// empty or contains a non-positive/non-finite value, an interval is zero,
/// or `swap_interval` does not divide `measurement_interval` (a
/// non-dividing cadence would silently drop measurements; B5).
#[pyfunction]
#[pyo3(signature = (
    lattice_size, j1, j2, j3, h, base_seed, algorithm, lattice_type,
    temperatures, n_thermalization, n_sweeps, measurement_interval,
    swap_interval = 1, store_configs = false, compute_correlation = false,
    correlation_interval = 1
))]
pub fn run_parallel_tempering<'py>(
    py: Python<'py>,
    lattice_size: usize,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
    base_seed: u64,
    algorithm: &str,
    lattice_type: &str,
    temperatures: Vec<f64>,
    n_thermalization: usize,
    n_sweeps: usize,
    measurement_interval: usize,
    swap_interval: usize,
    store_configs: bool,
    compute_correlation: bool,
    correlation_interval: usize,
) -> PyResult<(Vec<Bound<'py, PyDict>>, Bound<'py, PyDict>)> {
    let algo = algorithm.to_string();
    let lat_type = lattice_type.to_string();

    let (results, diagnostics) = py.allow_threads(|| {
        run_parallel_tempering_internal(
            lattice_size,
            j1,
            j2,
            j3,
            h,
            base_seed,
            &algo,
            &lat_type,
            &temperatures,
            n_thermalization,
            n_sweeps,
            measurement_interval,
            swap_interval,
            store_configs,
            compute_correlation,
            correlation_interval,
        )
    })?;

    Ok((
        convert_results_to_py(py, results)?,
        diagnostics.into_pydict(py)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: usize = 4;
    const NUM_SITES: usize = L * L;

    fn independent(
        temps: &[f64],
        seed_offsets: Option<&[u64]>,
        store_configs: bool,
        compute_correlation: bool,
    ) -> Result<Vec<TempResult>, MCIsingError> {
        run_independent_internal(
            L,
            1.0,
            0.0,
            0.0,
            0.0,
            42,
            "metropolis",
            "square",
            temps,
            20,
            50,
            10,
            store_configs,
            compute_correlation,
            1,
            seed_offsets,
        )
    }

    fn pt_with_diagnostics(
        temps: &[f64],
        n_sweeps: usize,
        measurement_interval: usize,
        swap_interval: usize,
    ) -> Result<(Vec<TempResult>, PtDiagnostics), MCIsingError> {
        run_parallel_tempering_internal(
            L,
            1.0,
            0.0,
            0.0,
            0.0,
            42,
            "metropolis",
            "square",
            temps,
            20,
            n_sweeps,
            measurement_interval,
            swap_interval,
            true,
            false,
            1,
        )
    }

    fn pt(
        temps: &[f64],
        n_sweeps: usize,
        measurement_interval: usize,
        swap_interval: usize,
    ) -> Result<Vec<TempResult>, MCIsingError> {
        pt_with_diagnostics(temps, n_sweeps, measurement_interval, swap_interval)
            .map(|(results, _)| results)
    }

    /// `Result::expect_err` needs `Debug` on the Ok type, which the
    /// production structs deliberately do not derive — unwrap manually.
    fn error_of<T>(result: Result<T, MCIsingError>) -> MCIsingError {
        match result {
            Ok(_) => panic!("expected an error, got Ok"),
            Err(e) => e,
        }
    }

    #[test]
    fn test_validate_rejects_nonpositive_temperature() {
        let err = error_of(independent(&[2.0, 0.0], None, false, false));
        assert!(err.to_string().contains("positive"), "got: {err}");
        let err = error_of(independent(&[-1.0], None, false, false));
        assert!(err.to_string().contains("positive"), "got: {err}");
    }

    #[test]
    fn test_validate_rejects_nan_temperature() {
        let err = error_of(independent(&[2.0, f64::NAN], None, false, false));
        assert!(err.to_string().contains("finite"), "got: {err}");
        let err = error_of(pt(&[2.0, f64::NAN], 100, 10, 1));
        assert!(err.to_string().contains("finite"), "got: {err}");
    }

    #[test]
    fn test_validate_rejects_empty_temperature_list() {
        let err = error_of(independent(&[], None, false, false));
        assert!(
            err.to_string().contains("At least one temperature"),
            "got: {err}"
        );
        let err = error_of(pt(&[], 100, 10, 1));
        assert!(
            err.to_string().contains("At least one temperature"),
            "got: {err}"
        );
    }

    #[test]
    fn test_validate_rejects_zero_measurement_interval() {
        let err = error_of(run_independent_internal(
            L,
            1.0,
            0.0,
            0.0,
            0.0,
            42,
            "metropolis",
            "square",
            &[2.0],
            10,
            50,
            0,
            false,
            false,
            1,
            None,
        ));
        assert!(
            err.to_string().contains("measurement_interval"),
            "got: {err}"
        );
    }

    #[test]
    fn test_validate_rejects_zero_correlation_interval() {
        let err = error_of(run_independent_internal(
            L,
            1.0,
            0.0,
            0.0,
            0.0,
            42,
            "metropolis",
            "square",
            &[2.0],
            10,
            50,
            10,
            false,
            true,
            0,
            None,
        ));
        assert!(
            err.to_string().contains("correlation_interval"),
            "got: {err}"
        );
        let err = error_of(run_parallel_tempering_internal(
            L,
            1.0,
            0.0,
            0.0,
            0.0,
            42,
            "metropolis",
            "square",
            &[1.5, 2.5],
            10,
            50,
            10,
            1,
            false,
            true,
            0,
        ));
        assert!(
            err.to_string().contains("correlation_interval"),
            "got: {err}"
        );
    }

    #[test]
    fn test_runners_honor_correlation_interval() {
        // 50 sweeps / interval 10 = 5 measurements; k = 2 evaluates after
        // the 2nd and 4th, k = 5 exactly once at the last.
        let results = run_independent_internal(
            L,
            1.0,
            0.0,
            0.0,
            0.0,
            42,
            "metropolis",
            "square",
            &[2.0],
            20,
            50,
            10,
            false,
            true,
            2,
            None,
        )
        .expect("valid run");
        let corr = results[0]
            .correlation
            .as_ref()
            .expect("compute_correlation=true");
        assert_eq!(corr.lengths.len(), 2);
        assert_eq!(results[0].energies.len(), 5);

        let (results, _) = run_parallel_tempering_internal(
            L,
            1.0,
            0.0,
            0.0,
            0.0,
            42,
            "metropolis",
            "square",
            &[1.5, 2.5],
            10,
            50,
            10,
            5,
            false,
            true,
            5,
        )
        .expect("valid run");
        for r in &results {
            let corr = r.correlation.as_ref().expect("compute_correlation=true");
            assert_eq!(corr.lengths.len(), 1);
            assert_eq!(r.energies.len(), 5);
        }
    }

    #[test]
    fn test_validate_swap_cadence_rejects_nondividing() {
        let err = validate_swap_cadence(15, 10).expect_err("15 % 10 != 0");
        assert!(
            err.to_string().contains("multiple of swap_interval"),
            "got: {err}"
        );
        let err = error_of(pt(&[1.5, 2.5], 90, 15, 10));
        assert!(
            err.to_string().contains("multiple of swap_interval"),
            "got: {err}"
        );
    }

    #[test]
    fn test_validate_swap_cadence_accepts_dividing() {
        for (meas, swap) in [(10, 1), (10, 5), (10, 10), (1, 1)] {
            validate_swap_cadence(meas, swap)
                .unwrap_or_else(|e| panic!("({meas}, {swap}) rejected: {e}"));
        }
        let err = validate_swap_cadence(10, 0).expect_err("swap_interval 0");
        assert!(err.to_string().contains("swap_interval"), "got: {err}");
    }

    #[test]
    fn test_pt_internal_measurement_count_is_never_short() {
        for swap_interval in [1, 2, 5, 10] {
            for n_sweeps in [50, 100] {
                let results = pt(&[1.5, 2.5, 3.5], n_sweeps, 10, swap_interval)
                    .unwrap_or_else(|e| panic!("swap={swap_interval}: {e}"));
                for r in &results {
                    assert_eq!(
                        r.energies.len(),
                        n_sweeps / 10,
                        "swap={swap_interval}, n_sweeps={n_sweeps}, T={}",
                        r.temperature
                    );
                    assert_eq!(r.magnetizations.len(), r.energies.len());
                    let configs = r.configs.as_ref().expect("store_configs=true");
                    assert_eq!(configs.len(), r.energies.len() * NUM_SITES);
                }
            }
        }
    }

    #[test]
    fn test_pt_internal_propagates_constructor_error() {
        let err = error_of(run_parallel_tempering_internal(
            L,
            -1.0,
            0.0,
            0.0,
            0.0,
            42,
            "wolff",
            "square",
            &[1.5, 2.5],
            10,
            50,
            10,
            1,
            false,
            false,
            1,
        ));
        assert!(err.to_string().contains("J1>0"), "got: {err}");
    }

    #[test]
    fn test_independent_internal_honors_compute_correlation() {
        let results = independent(&[2.0, 3.0], None, false, true).expect("valid run");
        for r in &results {
            let corr = r.correlation.as_ref().expect("compute_correlation=true");
            assert_eq!(corr.lengths.len(), 5, "one length per measurement");
            assert!(!corr.distances.is_empty());
            assert_eq!(corr.distances.len(), corr.values.len());
        }

        let results = independent(&[2.0], None, false, false).expect("valid run");
        assert!(results[0].correlation.is_none());
    }

    #[test]
    fn test_pt_internal_honors_compute_correlation() {
        let (results, _) = run_parallel_tempering_internal(
            L,
            1.0,
            0.0,
            0.0,
            0.0,
            42,
            "metropolis",
            "square",
            &[1.5, 2.5],
            10,
            50,
            10,
            5,
            false,
            true,
            1,
        )
        .expect("valid run");
        for r in &results {
            let corr = r.correlation.as_ref().expect("compute_correlation=true");
            assert_eq!(corr.lengths.len(), 5);
            assert_eq!(corr.distances.len(), corr.values.len());
            assert!(!corr.distances.is_empty());
        }
    }

    #[test]
    // Bit-identity IS the contract under test — approximate agreement would
    // hide a reseeded stream.
    #[allow(clippy::float_cmp)]
    fn test_independent_internal_seed_offsets_preserve_streams() {
        // The resume-reproducibility contract: running only the second
        // temperature with its original index as the seed offset must
        // reproduce the full run's streams for that temperature exactly.
        let full = independent(&[2.0, 3.0], None, true, false).expect("full run");
        let resumed = independent(&[3.0], Some(&[1]), true, false).expect("resumed run");

        assert_eq!(full[1].temperature, resumed[0].temperature);
        // Bit-identity is the contract here, not approximate agreement.
        assert_eq!(full[1].energies, resumed[0].energies);
        assert_eq!(full[1].magnetizations, resumed[0].magnetizations);
        assert_eq!(full[1].configs, resumed[0].configs);
    }

    #[test]
    fn test_independent_internal_seed_offsets_length_mismatch() {
        let err = error_of(independent(&[2.0, 3.0], Some(&[0]), false, false));
        assert!(err.to_string().contains("seed_offsets"), "got: {err}");
    }

    #[test]
    fn test_independent_internal_store_configs_false_omits_configs() {
        let results = independent(&[2.0], None, false, false).expect("valid run");
        assert!(results[0].configs.is_none());
        let results = independent(&[2.0], None, true, false).expect("valid run");
        let configs = results[0].configs.as_ref().expect("store_configs=true");
        assert_eq!(configs.len(), 5 * NUM_SITES);
    }

    /// Ferromagnetic 4×4 metropolis replicas for direct `attempt_swaps`
    /// tests. The spin contents are irrelevant there — only the energies
    /// slice drives the criterion — but the sims must exist to be swapped.
    fn metropolis_replicas(n: usize) -> Vec<IsingSimulation> {
        build_replicas(n, L, 1.0, 0.0, 0.0, 0.0, 42, "metropolis", "square")
            .expect("valid replica parameters")
    }

    /// Run a closure inside a dedicated rayon pool with `n_threads`
    /// threads (scoped — the global pool is never touched).
    fn with_pool<T: Send>(n_threads: usize, f: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("pool builds")
            .install(f)
    }

    #[test]
    // Acceptance is detected through the bit-exact movement of the rigged
    // energy value — the swap relocates the float, it never recomputes it.
    #[allow(clippy::float_cmp)]
    fn test_swap_acceptance_matches_metropolis_probability() {
        // P(swap) = min(1, exp(delta)) with delta = (β_i−β_j)(E_i−E_j)·N.
        // Per-site energies are rigged so delta = ln(target_p) exactly,
        // then the empirical acceptance rate over N_TRIALS near-free
        // attempt_swaps calls is compared against target_p. The 5σ band
        // keeps the false-failure probability at ~6e-7 per point, while a
        // missing exp() or a sign flip sits >90σ away — decisive, and two
        // target points pin the exponential shape, not just a coin flip.
        const N_TRIALS: usize = 10_000;
        const N_SIGMA: f64 = 5.0;
        let betas = [1.0, 0.5]; // descending, as production sorts them
        let mut ladder = Ladder::rigged(metropolis_replicas(2), vec![0.0, 0.0]);
        let mut swap_rng = crate::rng::create_rng(999);

        for target_p in [0.5_f64, 0.2] {
            // delta = (β0−β1)·(0 − e1)·N = ln(target_p)
            let e1 = -target_p.ln() / ((betas[0] - betas[1]) * NUM_SITES as f64);
            let mut accepts = 0_usize;
            let attempted_before = ladder.stats.swap_attempted[0];
            let accepted_before = ladder.stats.swap_accepted[0];
            for _ in 0..N_TRIALS {
                ladder.energies = vec![0.0, e1];
                ladder.attempt_swaps(&betas, 0, &mut swap_rng);
                if ladder.energies[0] == e1 {
                    accepts += 1;
                }
            }
            // The counters agree with the bit-exact detection.
            assert_eq!(ladder.stats.swap_attempted[0] - attempted_before, N_TRIALS);
            assert_eq!(ladder.stats.swap_accepted[0] - accepted_before, accepts);
            let p_hat = accepts as f64 / N_TRIALS as f64;
            let sigma = (target_p * (1.0 - target_p) / N_TRIALS as f64).sqrt();
            println!("calib swap-acceptance: p={target_p} p_hat={p_hat} sigma={sigma:.5}");
            assert!(
                (p_hat - target_p).abs() <= N_SIGMA * sigma,
                "p={target_p}: p_hat={p_hat} deviates more than {N_SIGMA}σ (σ={sigma:.5})"
            );
        }
    }

    #[test]
    // The swap moves values bit-exactly; equality is the detection.
    #[allow(clippy::float_cmp)]
    fn test_swap_delta_nonnegative_always_accepts() {
        let betas = [1.0, 0.5];
        let mut ladder = Ladder::rigged(metropolis_replicas(2), vec![0.0, 0.0]);
        let mut swap_rng = crate::rng::create_rng(7);

        // delta > 0 (hotter replica holds the lower energy): deterministic.
        for _ in 0..100 {
            ladder.energies = vec![1.0, 0.0];
            ladder.attempt_swaps(&betas, 0, &mut swap_rng);
            assert_eq!(ladder.energies, [0.0, 1.0], "delta>0 must always swap");
        }
        assert_eq!(ladder.stats.swap_accepted, vec![100]);
        assert_eq!(ladder.stats.swap_attempted, vec![100]);

        // delta == 0 (equal energies): the >= branch swaps unconditionally.
        // Equal energies are indistinguishable after the swap, so marker
        // spins detect it. This always-swap at delta=0 is exactly why
        // "PT at equal betas ≡ independent" holds only as a multiset
        // statement across replicas, never per index.
        ladder.replicas[0].spins = vec![1; NUM_SITES];
        ladder.replicas[1].spins = vec![-1; NUM_SITES];
        ladder.energies = vec![0.5, 0.5];
        ladder.attempt_swaps(&betas, 0, &mut swap_rng);
        assert_eq!(ladder.replicas[0].spins, vec![-1; NUM_SITES]);
        assert_eq!(ladder.replicas[1].spins, vec![1; NUM_SITES]);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_swap_strongly_negative_delta_never_accepts() {
        // delta = −50 → acceptance e^{−50} ≈ 2e-22; over 1000 trials the
        // false-failure probability is ~2e-19.
        let betas = [1.0, 0.5];
        let e1 = 50.0 / ((betas[0] - betas[1]) * NUM_SITES as f64);
        let mut ladder = Ladder::rigged(metropolis_replicas(2), vec![0.0, e1]);
        let mut swap_rng = crate::rng::create_rng(11);
        for _ in 0..1000 {
            ladder.attempt_swaps(&betas, 0, &mut swap_rng);
            assert_eq!(ladder.energies, [0.0, e1], "delta=-50 swap accepted");
        }
        assert_eq!(ladder.stats.swap_attempted, vec![1000]);
        assert_eq!(ladder.stats.swap_accepted, vec![0]);
        assert_eq!(ladder.replica_id, vec![0, 1], "identities never moved");
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_swap_even_odd_pair_alternation() {
        // Three replicas, every adjacent delta rigged positive: the round
        // parity alone selects which pairs may swap — round 0 touches only
        // (0,1), round 1 only (1,2).
        let betas = [1.0, 0.5, 0.25];
        let mut swap_rng = crate::rng::create_rng(13);
        for (round, expected, attempted, ids) in [
            (0_usize, [2.0, 3.0, 1.0], [1, 0], [1, 0, 2]),
            (1, [3.0, 1.0, 2.0], [0, 1], [0, 2, 1]),
        ] {
            let mut ladder = Ladder::rigged(metropolis_replicas(3), vec![3.0, 2.0, 1.0]);
            ladder.attempt_swaps(&betas, round, &mut swap_rng);
            assert_eq!(ladder.energies, expected, "round={round}");
            assert_eq!(ladder.stats.swap_attempted, attempted, "round={round}");
            assert_eq!(ladder.stats.swap_accepted, attempted, "round={round}");
            assert_eq!(ladder.replica_id, ids, "round={round}");
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_swap_two_replicas_odd_round_is_noop() {
        // offset=1 with two replicas yields the empty range 1..1 — no pair
        // exists, so even a rigged delta>0 must not swap.
        let betas = [1.0, 0.5];
        let mut ladder = Ladder::rigged(metropolis_replicas(2), vec![1.0, 0.0]);
        let mut swap_rng = crate::rng::create_rng(17);
        ladder.attempt_swaps(&betas, 1, &mut swap_rng);
        assert_eq!(ladder.energies, [1.0, 0.0]);
        assert_eq!(ladder.stats.swap_attempted, vec![0], "no pair, no attempt");
    }

    #[test]
    fn test_replica_ids_follow_spins() {
        // Marker spins: after an always-accepted swap the identity travels
        // with the configuration, and back again.
        let betas = [1.0, 0.5];
        let mut ladder = Ladder::rigged(metropolis_replicas(2), vec![1.0, 0.0]);
        ladder.replicas[0].spins = vec![1; NUM_SITES];
        ladder.replicas[1].spins = vec![-1; NUM_SITES];
        let mut swap_rng = crate::rng::create_rng(19);
        ladder.attempt_swaps(&betas, 0, &mut swap_rng);
        assert_eq!(ladder.replica_id, vec![1, 0]);
        assert_eq!(ladder.replicas[0].spins, vec![-1; NUM_SITES]);
        ladder.energies = vec![1.0, 0.0];
        ladder.attempt_swaps(&betas, 0, &mut swap_rng);
        assert_eq!(ladder.replica_id, vec![0, 1]);
        assert_eq!(ladder.replicas[0].spins, vec![1; NUM_SITES]);
    }

    #[test]
    fn test_round_trip_counter_sequence() {
        // Three rungs, hand-driven occupancy. A round trip is coldest →
        // hottest → coldest; a replica that only touches the hottest rung
        // without having been coldest first is never counted.
        let mut ladder = Ladder::rigged(metropolis_replicas(3), vec![0.0; 3]);
        // Construction labels replica 0 (coldest) Up; replica 2 stays
        // Unlabelled because it has not been coldest yet.
        assert_eq!(
            ladder.direction,
            vec![Direction::Up, Direction::Unlabelled, Direction::Unlabelled]
        );
        assert_eq!(ladder.stats.round_trips, vec![0, 0, 0]);

        let occupy = |ladder: &mut Ladder, ids: [usize; 3]| {
            ladder.replica_id = ids.to_vec();
            ladder.record_round_trips();
        };
        // Replica 0 climbs to the top: Up → Down.
        occupy(&mut ladder, [2, 1, 0]);
        assert_eq!(ladder.direction[0], Direction::Down);
        assert_eq!(ladder.direction[2], Direction::Up);
        assert_eq!(ladder.stats.round_trips, vec![0, 0, 0]);
        // Sitting in the middle changes nothing.
        occupy(&mut ladder, [2, 0, 1]);
        assert_eq!(ladder.stats.round_trips, vec![0, 0, 0]);
        // Back at the bottom: one round trip for replica 0.
        occupy(&mut ladder, [0, 2, 1]);
        assert_eq!(ladder.stats.round_trips, vec![1, 0, 0]);
        assert_eq!(ladder.direction[0], Direction::Up);
        // Replica 2 (Up since the second step) reaches the top, then the
        // bottom again: its own round trip.
        occupy(&mut ladder, [1, 0, 2]);
        occupy(&mut ladder, [2, 0, 1]);
        assert_eq!(ladder.stats.round_trips, vec![1, 0, 1]);
        // Replica 1 was coldest in the previous step and is now hottest:
        // labelled Down, its round trip only completes back at the bottom.
        assert_eq!(ladder.direction[1], Direction::Down);
        occupy(&mut ladder, [1, 2, 0]);
        assert_eq!(ladder.stats.round_trips, vec![1, 1, 1]);

        // A single rung never counts anything.
        let mut single = Ladder::rigged(metropolis_replicas(1), vec![0.0]);
        single.record_round_trips();
        assert_eq!(single.stats.round_trips, vec![0]);
        assert!(single.stats.swap_attempted.is_empty());
    }

    #[test]
    fn test_pt_diagnostics_shapes_and_bounds() {
        // 100 sweeps at swap_interval 1 → 100 rounds; with 4 rungs the even
        // rounds try pairs (0,1),(2,3) and the odd rounds pair (1,2).
        let (_, d) = pt_with_diagnostics(&[1.5, 2.0, 2.5, 3.5], 100, 10, 1).expect("valid run");
        assert_eq!(d.temperatures, vec![1.5, 2.0, 2.5, 3.5]);
        assert_eq!(d.swap_attempted, vec![50, 50, 50]);
        assert_eq!(d.round_trips.len(), 4);
        for (accepted, attempted) in d.swap_accepted.iter().zip(&d.swap_attempted) {
            assert!(accepted <= attempted);
        }
        // Three rungs at swap_interval 5 over 100 sweeps → 20 rounds.
        let (_, d) = pt_with_diagnostics(&[1.5, 2.5, 3.5], 100, 10, 5).expect("valid run");
        assert_eq!(d.swap_attempted, vec![10, 10]);
    }

    #[test]
    fn test_pt_dense_ladder_completes_round_trips() {
        // Four closely spaced rungs on a 4×4 lattice: swaps are accepted
        // often enough that at least one replica completes a round trip.
        let (_, d) = pt_with_diagnostics(&[2.2, 2.3, 2.4, 2.5], 400, 10, 1).expect("valid run");
        let total: usize = d.round_trips.iter().sum();
        assert!(total > 0, "round trips: {:?}", d.round_trips);
    }

    /// Full PT runs whose measurement rounds check (in debug builds, via
    /// the `debug_assert_eq!` in `sweep_round`) that the tracked shell sums
    /// equal a fresh recount — for every algorithm and for both dyadic and
    /// non-dyadic couplings.
    fn assert_pt_bookkeeping_holds(
        algorithm: &str,
        lattice_type: &str,
        j1: f64,
        j2: f64,
        j3: f64,
        h: f64,
    ) {
        let (results, d) = run_parallel_tempering_internal(
            4,
            j1,
            j2,
            j3,
            h,
            42,
            algorithm,
            lattice_type,
            &[1.5, 2.0, 2.5, 3.0],
            10,
            40,
            4,
            2,
            false,
            false,
            1,
        )
        .unwrap_or_else(|e| panic!("{algorithm}/{lattice_type}: {e}"));
        assert_eq!(results.len(), 4);
        assert_eq!(d.swap_attempted.iter().sum::<usize>(), 20 * 3 / 2);
    }

    #[test]
    fn test_build_replicas_enables_wolff_delta_tracking() {
        let mut replicas = build_replicas(2, L, 1.0, 0.0, 0.0, 0.0, 42, "wolff", "square")
            .expect("valid replica parameters");
        for sim in &mut replicas {
            assert!(sim.sweep_internal(3, 0.5).delta.is_some());
        }
        // Outside the ladder the same simulation does not pay for it.
        let mut plain = IsingSimulation::new_internal(L, 1.0, 0.0, 0.0, 0.0, 42, "wolff", "square")
            .expect("valid constructor arguments");
        assert!(plain.sweep_internal(3, 0.5).delta.is_none());
    }

    #[test]
    fn test_pt_bookkeeping_matches_recompute() {
        assert_pt_bookkeeping_holds("metropolis", "square", 1.0, 0.5, 0.25, 0.5);
        assert_pt_bookkeeping_holds("metropolis", "square", 1.0, -0.3, 0.0, 0.1);
        assert_pt_bookkeeping_holds("metropolis", "cubic", 1.0, 0.5, 0.25, 0.0);
        assert_pt_bookkeeping_holds("metropolis", "honeycomb", -1.0, 0.0, 0.5, 0.0);
        assert_pt_bookkeeping_holds("wolff", "square", 1.0, 0.0, 0.0, 0.0);
        assert_pt_bookkeeping_holds("wolff", "triangular", 0.7, 0.0, 0.0, 0.0);
        assert_pt_bookkeeping_holds("swendsen_wang", "square", 1.0, 0.0, 0.0, 0.0);
    }

    #[test]
    // Bit-identity IS the contract — PT with one replica has no swap pairs,
    // so it must degenerate to the independent runner exactly.
    #[allow(clippy::float_cmp)]
    fn test_pt_single_temperature_equals_independent_bitwise() {
        // Same seed (base+0), same thermalization, same sweep granularity
        // (swap_interval = measurement_interval), and an empty swap loop
        // that never draws from the swap RNG.
        let pt_results = pt(&[2.5], 50, 10, 10).expect("valid run");
        let ind_results = independent(&[2.5], None, true, false).expect("valid run");
        assert_eq!(pt_results.len(), 1);
        assert_eq!(pt_results[0].temperature, ind_results[0].temperature);
        assert_eq!(pt_results[0].energies, ind_results[0].energies);
        assert_eq!(pt_results[0].magnetizations, ind_results[0].magnetizations);
        assert_eq!(pt_results[0].configs, ind_results[0].configs);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_pt_identical_betas_one_round_matches_independent_multiset() {
        // At equal betas delta = 0, so the swap ALWAYS fires and per-index
        // trajectories are permuted — the naive per-index "PT ≡ independent"
        // is false by design. The exact statement is over the multiset of
        // replicas: with one round (swap = measurement = n_sweeps) the swap
        // only permutes already-recorded values, so the sorted measurements
        // must match the independent runner's bit-for-bit (PT replica i and
        // independent temperature i share seed base+i).
        let (pt_results, diagnostics) = run_parallel_tempering_internal(
            L,
            1.0,
            0.0,
            0.0,
            0.0,
            42,
            "metropolis",
            "square",
            &[2.5, 2.5],
            20,
            50,
            50,
            50,
            false,
            false,
            1,
        )
        .expect("valid run");
        // delta = 0 at equal betas: the single round's swap always fires.
        assert_eq!(diagnostics.swap_attempted, vec![1]);
        assert_eq!(diagnostics.swap_accepted, vec![1]);
        let ind_results = run_independent_internal(
            L,
            1.0,
            0.0,
            0.0,
            0.0,
            42,
            "metropolis",
            "square",
            &[2.5, 2.5],
            20,
            50,
            50,
            false,
            false,
            1,
            None,
        )
        .expect("valid run");

        let sorted_single = |results: &[TempResult], pick: fn(&TempResult) -> f64| {
            let mut values: Vec<f64> = results.iter().map(pick).collect();
            values.sort_by(f64::total_cmp);
            values
        };
        assert_eq!(
            sorted_single(&pt_results, |r| r.energies[0]),
            sorted_single(&ind_results, |r| r.energies[0]),
        );
        assert_eq!(
            sorted_single(&pt_results, |r| r.magnetizations[0]),
            sorted_single(&ind_results, |r| r.magnetizations[0]),
        );
    }

    #[test]
    // Determinism under thread-count changes is by construction (per-replica
    // seeds, a serial swap RNG, order-preserving collect) — this pins it.
    #[allow(clippy::float_cmp)]
    fn test_pt_deterministic_under_thread_counts() {
        let (reference, ref_diag) =
            with_pool(1, || pt_with_diagnostics(&[1.5, 2.5, 3.5], 40, 10, 10)).expect("valid run");
        for n_threads in [2, 4] {
            let (run, diag) = with_pool(n_threads, || {
                pt_with_diagnostics(&[1.5, 2.5, 3.5], 40, 10, 10)
            })
            .expect("valid run");
            assert_eq!(
                diag.swap_attempted, ref_diag.swap_attempted,
                "n_threads={n_threads}"
            );
            assert_eq!(
                diag.swap_accepted, ref_diag.swap_accepted,
                "n_threads={n_threads}"
            );
            assert_eq!(
                diag.round_trips, ref_diag.round_trips,
                "n_threads={n_threads}"
            );
            for (r, expected) in run.iter().zip(&reference) {
                assert_eq!(r.temperature, expected.temperature, "n_threads={n_threads}");
                assert_eq!(r.energies, expected.energies, "n_threads={n_threads}");
                assert_eq!(
                    r.magnetizations, expected.magnetizations,
                    "n_threads={n_threads}"
                );
                assert_eq!(r.configs, expected.configs, "n_threads={n_threads}");
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_independent_deterministic_under_thread_counts() {
        let reference =
            with_pool(1, || independent(&[1.5, 2.5, 3.5], None, true, false)).expect("valid run");
        for n_threads in [2, 4] {
            let run = with_pool(n_threads, || {
                independent(&[1.5, 2.5, 3.5], None, true, false)
            })
            .expect("valid run");
            for (r, expected) in run.iter().zip(&reference) {
                assert_eq!(r.temperature, expected.temperature, "n_threads={n_threads}");
                assert_eq!(r.energies, expected.energies, "n_threads={n_threads}");
                assert_eq!(
                    r.magnetizations, expected.magnetizations,
                    "n_threads={n_threads}"
                );
                assert_eq!(r.configs, expected.configs, "n_threads={n_threads}");
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_pt_returns_sorted_temperatures_for_unsorted_input() {
        let results = pt(&[3.5, 1.5, 2.5], 50, 10, 10).expect("valid run");
        let temps: Vec<f64> = results.iter().map(|r| r.temperature).collect();
        assert_eq!(temps, vec![1.5, 2.5, 3.5]);
    }
}
