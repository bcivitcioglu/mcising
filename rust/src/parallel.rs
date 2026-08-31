//! Parallel execution paths: independent temperatures and parallel tempering.
//!
//! Independent mode gives each temperature its own `IsingSimulation` with a
//! unique RNG seed and runs all of them simultaneously via Rayon — no shared
//! mutable state, pure data parallelism. Parallel tempering runs one coupled
//! replica ladder with periodic swap attempts between adjacent temperatures.
//!
//! All user-reachable failures are rejected up front by the validators in
//! this file; the hot loops themselves are panic-free for validated input.

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::Rng;
use rayon::prelude::*;

use crate::error::MCIsingError;
use crate::lattice::{with_lattice, Lattice, LatticeKind};
use crate::observables;
use crate::simulation::IsingSimulation;

/// Correlation data collected for one temperature.
///
/// `distances`/`values` hold the correlation function of the most recent
/// measurement (the representative pair, mirroring the cooldown path);
/// `lengths` collects the correlation length at every measurement.
struct CorrelationRecord {
    distances: Vec<f64>,
    values: Vec<f64>,
    lengths: Vec<f64>,
}

/// Result for a single temperature, returned to Python as a dict.
struct TempResult {
    temperature: f64,
    energies: Vec<f64>,
    magnetizations: Vec<f64>,
    configs: Option<Vec<i8>>,
    correlation: Option<CorrelationRecord>,
    shape: Vec<usize>,
    /// Cluster flips during measurement sweeps only (0 for Metropolis);
    /// thermalization work is never counted.
    cluster_flips: usize,
}

impl TempResult {
    fn with_capacity(
        temperature: f64,
        n_measurements: usize,
        num_sites: usize,
        shape: Vec<usize>,
        store_configs: bool,
        compute_correlation: bool,
    ) -> Self {
        Self {
            temperature,
            energies: Vec::with_capacity(n_measurements),
            magnetizations: Vec::with_capacity(n_measurements),
            configs: store_configs.then(|| Vec::with_capacity(n_measurements * num_sites)),
            correlation: compute_correlation.then(|| CorrelationRecord {
                distances: Vec::new(),
                values: Vec::new(),
                lengths: Vec::with_capacity(n_measurements),
            }),
            shape,
            cluster_flips: 0,
        }
    }

    /// Record one measurement: energy, magnetization, and (when enabled)
    /// a configuration snapshot and correlation data.
    ///
    /// The correlation observables are pure functions of `spins` — they draw
    /// no RNG, so enabling them never perturbs the sampling streams.
    fn push<L: Lattice>(&mut self, spins: &[i8], lattice: &L, energy: f64) {
        self.energies.push(energy);
        self.magnetizations
            .push(observables::magnetization_per_site(spins));
        if let Some(ref mut c) = self.configs {
            c.extend_from_slice(spins);
        }
        if let Some(ref mut corr) = self.correlation {
            let bins = observables::correlation_bins(spins, lattice);
            corr.lengths
                .push(observables::correlation_length(&bins, lattice.dimension()));
            corr.distances = bins.distances();
            corr.values = bins.correlations;
        }
    }
}

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
    store_configs = false, compute_correlation = false, seed_offsets = None
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
            seed_offsets.as_deref(),
        )
    })?;

    convert_results_to_py(py, results)
}

/// Convert `TempResult`s to Python dicts.
///
/// Array shapes are derived from the actual number of collected
/// measurements, never from a precomputed count — a cadence bug upstream
/// can therefore shorten arrays but can no longer cause a reshape panic.
fn convert_results_to_py(
    py: Python<'_>,
    results: Vec<TempResult>,
) -> PyResult<Vec<Bound<'_, PyDict>>> {
    let mut py_results = Vec::with_capacity(results.len());
    for r in results {
        let n_measurements = r.energies.len();
        let dict = PyDict::new(py);
        dict.set_item("temperature", r.temperature)?;
        dict.set_item("energies", r.energies.into_pyarray(py))?;
        dict.set_item("magnetizations", r.magnetizations.into_pyarray(py))?;
        dict.set_item("n_cluster_flips", r.cluster_flips)?;

        if let Some(configs) = r.configs {
            let flat = numpy::PyArray1::from_vec(py, configs);
            let mut reshape_dims: Vec<usize> = vec![n_measurements];
            reshape_dims.extend_from_slice(&r.shape);
            let reshaped = flat.reshape(reshape_dims)?;
            dict.set_item("configurations", reshaped)?;
        }

        if let Some(corr) = r.correlation {
            dict.set_item("correlation_distances", corr.distances.into_pyarray(py))?;
            dict.set_item("correlation_function", corr.values.into_pyarray(py))?;
            dict.set_item("correlation_length", corr.lengths.into_pyarray(py))?;
        }

        py_results.push(dict);
    }
    Ok(py_results)
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
            IsingSimulation::new_internal(
                lattice_size,
                j1,
                j2,
                j3,
                h,
                base_seed.wrapping_add(i as u64),
                algorithm,
                lattice_type,
            )
        })
        .collect()
}

/// Attempt swaps between adjacent replicas (even/odd alternation).
fn attempt_swaps(
    replicas: &mut [IsingSimulation],
    energies: &mut [f64],
    betas: &[f64],
    round: usize,
    num_sites: usize,
    swap_rng: &mut impl Rng,
) {
    let offset = usize::from(!round.is_multiple_of(2));
    for i in (offset..replicas.len().saturating_sub(1)).step_by(2) {
        let j = i + 1;
        // Standard PT acceptance: P = min(1, exp(delta))
        // where delta = (β_i - β_j) * (E_i - E_j) * N
        // β sorted descending (β_i > β_j), so if E_i < E_j
        // (low-T replica has lower energy), delta > 0 → always accept.
        let delta = (betas[i] - betas[j]) * (energies[i] - energies[j]) * num_sites as f64;
        let accept = delta >= 0.0 || swap_rng.gen::<f64>() < delta.exp();
        if accept {
            // O(1) pointer swap of spin Vecs.
            let (left, right) = replicas.split_at_mut(j);
            std::mem::swap(&mut left[i].spins, &mut right[0].spins);
            // Swap cached energies too.
            energies.swap(i, j);
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
            )
        })
        .collect();

    // Cache per-site energies; attempt_swaps multiplies by num_sites to
    // recover the total-energy swap criterion.
    let mut energies: Vec<f64> = replicas
        .iter()
        .map(|sim| {
            with_lattice!(&sim.lattice, lat => {
                observables::energy_per_site(&sim.spins, lat, j1, j2, j3, h)
            })
        })
        .collect();

    let mut sweep_count: usize = 0;
    let mut round: usize = 0;

    while sweep_count < n_sweeps {
        // a. Parallel sweeps. Replica i is pinned to betas[i] (swaps
        // exchange spins, not temperatures), so per-index cluster-flip
        // attribution is correct.
        let sweeps_this_round = swap_interval.min(n_sweeps - sweep_count);
        let round_flips: Vec<usize> = replicas
            .par_iter_mut()
            .enumerate()
            .map(|(i, sim)| {
                sim.sweep_internal(sweeps_this_round, betas[i])
                    .cluster_flips
            })
            .collect();
        for (result, flips) in temp_results.iter_mut().zip(&round_flips) {
            result.cluster_flips += flips;
        }
        sweep_count += sweeps_this_round;

        // Update cached energies.
        for (i, sim) in replicas.iter().enumerate() {
            energies[i] = with_lattice!(&sim.lattice, lat => {
                observables::energy_per_site(&sim.spins, lat, j1, j2, j3, h)
            });
        }

        // b. Swap attempts (even/odd alternation).
        attempt_swaps(
            &mut replicas,
            &mut energies,
            &betas,
            round,
            num_sites,
            &mut swap_rng,
        );
        round += 1;

        // c. Collect measurements at the right intervals. With the cadence
        // validated (swap_interval divides measurement_interval), sweep_count
        // hits every multiple of measurement_interval exactly once.
        if sweep_count.is_multiple_of(measurement_interval) {
            for (i, sim) in replicas.iter().enumerate() {
                let energy = energies[i];
                with_lattice!(&sim.lattice, lat => {
                    temp_results[i].push(&sim.spins, lat, energy);
                });
            }
        }
    }

    Ok(temp_results)
}

/// Run Parallel Tempering: N replicas at different temperatures with
/// periodic swap attempts between adjacent replicas.
///
/// Swaps use the standard Metropolis criterion:
///   P(swap i,j) = min(1, exp((β_i - β_j) × (E_i - E_j)))
///
/// Even/odd alternation ensures all adjacent pairs get swap opportunities.
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
    swap_interval = 1, store_configs = false, compute_correlation = false
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
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let algo = algorithm.to_string();
    let lat_type = lattice_type.to_string();

    let results = py.allow_threads(|| {
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
        )
    })?;

    convert_results_to_py(py, results)
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
            seed_offsets,
        )
    }

    fn pt(
        temps: &[f64],
        n_sweeps: usize,
        measurement_interval: usize,
        swap_interval: usize,
    ) -> Result<Vec<TempResult>, MCIsingError> {
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
        )
    }

    /// `Result::expect_err` needs `Debug` on the Ok type, which the
    /// production structs deliberately do not derive — unwrap manually.
    fn error_of(result: Result<Vec<TempResult>, MCIsingError>) -> MCIsingError {
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
            None,
        ));
        assert!(
            err.to_string().contains("measurement_interval"),
            "got: {err}"
        );
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
        let results = run_parallel_tempering_internal(
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
        let mut replicas = metropolis_replicas(2);
        let mut swap_rng = crate::rng::create_rng(999);

        for target_p in [0.5_f64, 0.2] {
            // delta = (β0−β1)·(0 − e1)·N = ln(target_p)
            let e1 = -target_p.ln() / ((betas[0] - betas[1]) * NUM_SITES as f64);
            let mut accepts = 0_usize;
            for _ in 0..N_TRIALS {
                let mut energies = [0.0, e1];
                attempt_swaps(
                    &mut replicas,
                    &mut energies,
                    &betas,
                    0,
                    NUM_SITES,
                    &mut swap_rng,
                );
                if energies[0] == e1 {
                    accepts += 1;
                }
            }
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
        let mut replicas = metropolis_replicas(2);
        let mut swap_rng = crate::rng::create_rng(7);

        // delta > 0 (hotter replica holds the lower energy): deterministic.
        for _ in 0..100 {
            let mut energies = [1.0, 0.0];
            attempt_swaps(
                &mut replicas,
                &mut energies,
                &betas,
                0,
                NUM_SITES,
                &mut swap_rng,
            );
            assert_eq!(energies, [0.0, 1.0], "delta>0 must always swap");
        }

        // delta == 0 (equal energies): the >= branch swaps unconditionally.
        // Equal energies are indistinguishable after the swap, so marker
        // spins detect it. This always-swap at delta=0 is exactly why
        // "PT at equal betas ≡ independent" holds only as a multiset
        // statement across replicas, never per index.
        replicas[0].spins = vec![1; NUM_SITES];
        replicas[1].spins = vec![-1; NUM_SITES];
        let mut energies = [0.5, 0.5];
        attempt_swaps(
            &mut replicas,
            &mut energies,
            &betas,
            0,
            NUM_SITES,
            &mut swap_rng,
        );
        assert_eq!(replicas[0].spins, vec![-1; NUM_SITES]);
        assert_eq!(replicas[1].spins, vec![1; NUM_SITES]);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_swap_strongly_negative_delta_never_accepts() {
        // delta = −50 → acceptance e^{−50} ≈ 2e-22; over 1000 trials the
        // false-failure probability is ~2e-19.
        let betas = [1.0, 0.5];
        let e1 = 50.0 / ((betas[0] - betas[1]) * NUM_SITES as f64);
        let mut replicas = metropolis_replicas(2);
        let mut swap_rng = crate::rng::create_rng(11);
        for _ in 0..1000 {
            let mut energies = [0.0, e1];
            attempt_swaps(
                &mut replicas,
                &mut energies,
                &betas,
                0,
                NUM_SITES,
                &mut swap_rng,
            );
            assert_eq!(energies, [0.0, e1], "delta=-50 swap accepted");
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_swap_even_odd_pair_alternation() {
        // Three replicas, every adjacent delta rigged positive: the round
        // parity alone selects which pairs may swap — round 0 touches only
        // (0,1), round 1 only (1,2).
        let betas = [1.0, 0.5, 0.25];
        let mut swap_rng = crate::rng::create_rng(13);
        for (round, expected) in [(0_usize, [2.0, 3.0, 1.0]), (1, [3.0, 1.0, 2.0])] {
            let mut replicas = metropolis_replicas(3);
            let mut energies = [3.0, 2.0, 1.0];
            attempt_swaps(
                &mut replicas,
                &mut energies,
                &betas,
                round,
                NUM_SITES,
                &mut swap_rng,
            );
            assert_eq!(energies, expected, "round={round}");
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_swap_two_replicas_odd_round_is_noop() {
        // offset=1 with two replicas yields the empty range 1..1 — no pair
        // exists, so even a rigged delta>0 must not swap.
        let betas = [1.0, 0.5];
        let mut replicas = metropolis_replicas(2);
        let mut swap_rng = crate::rng::create_rng(17);
        let mut energies = [1.0, 0.0];
        attempt_swaps(
            &mut replicas,
            &mut energies,
            &betas,
            1,
            NUM_SITES,
            &mut swap_rng,
        );
        assert_eq!(energies, [1.0, 0.0]);
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
        let pt_results = run_parallel_tempering_internal(
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
        )
        .expect("valid run");
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
        let reference = with_pool(1, || pt(&[1.5, 2.5, 3.5], 40, 10, 10)).expect("valid run");
        for n_threads in [2, 4] {
            let run = with_pool(n_threads, || pt(&[1.5, 2.5, 3.5], 40, 10, 10)).expect("valid run");
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
