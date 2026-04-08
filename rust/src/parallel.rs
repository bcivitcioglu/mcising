//! Parallel independent-temperature execution via Rayon.
//!
//! Each temperature gets its own `IsingSimulation` instance with a unique
//! RNG seed. All temperatures run simultaneously on separate CPU cores.
//! No shared mutable state — pure data parallelism.

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::Rng;
use rayon::prelude::*;

use crate::lattice::{with_lattice, Lattice, LatticeKind};
use crate::observables;
use crate::simulation::IsingSimulation;

/// Result for a single temperature, returned to Python as a dict.
struct TempResult {
    temperature: f64,
    energies: Vec<f64>,
    magnetizations: Vec<f64>,
    configs: Option<Vec<i8>>,
    num_sites: usize,
    shape: Vec<usize>,
}

/// Run independent simulations at multiple temperatures in parallel.
///
/// Each temperature starts from a random spin configuration with a
/// deterministic seed (base_seed + temperature_index). All temperatures
/// execute simultaneously via Rayon's thread pool.
///
/// Returns a list of dicts, one per temperature, containing energy and
/// magnetization arrays.
#[pyfunction]
#[pyo3(signature = (
    lattice_size, j1, j2, j3, h, base_seed, algorithm, lattice_type,
    temperatures, n_thermalization, n_sweeps, measurement_interval,
    store_configs = false, compute_correlation = false
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
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let n_measurements = n_sweeps / measurement_interval.max(1);

    // Clone strings for use inside rayon closure (which requires Send).
    let algo = algorithm.to_string();
    let lat_type = lattice_type.to_string();

    // Release the GIL while Rayon does the heavy lifting.
    let results: Vec<TempResult> = py.allow_threads(|| {
        temperatures
            .par_iter()
            .enumerate()
            .map(|(i, &temp)| {
                let seed = base_seed.wrapping_add(i as u64);
                let beta = 1.0 / temp;

                // Each thread gets its own simulation — no shared state.
                let mut sim = IsingSimulation::new_internal(
                    lattice_size, j1, j2, j3, h, seed, &algo, &lat_type,
                )
                .expect("simulation creation should not fail with validated params");

                let num_sites = with_lattice!(&sim.lattice, lat => lat.num_sites());
                let shape = with_lattice!(&sim.lattice, lat => lat.shape().to_vec());

                // Thermalize from random initialization at this temperature.
                sim.sweep_internal(n_thermalization, beta);

                // Production sweeps — collect measurements.
                let mut energies = Vec::with_capacity(n_measurements);
                let mut magnetizations = Vec::with_capacity(n_measurements);
                let mut configs: Option<Vec<i8>> = if store_configs {
                    Some(Vec::with_capacity(n_measurements * num_sites))
                } else {
                    None
                };

                for _ in 0..n_measurements {
                    sim.sweep_internal(measurement_interval, beta);

                    let energy = with_lattice!(&sim.lattice, lat => {
                        observables::energy_per_site(&sim.spins, lat, j1, j2, j3, h)
                    });
                    energies.push(energy);
                    magnetizations.push(observables::magnetization_per_site(&sim.spins));

                    if let Some(ref mut c) = configs {
                        c.extend_from_slice(&sim.spins);
                    }
                }

                TempResult {
                    temperature: temp,
                    energies,
                    magnetizations,
                    configs,
                    num_sites,
                    shape,
                }
            })
            .collect()
    });

    convert_results_to_py(py, results, n_measurements)
}

/// Convert TempResult Vec to Python list of dicts.
fn convert_results_to_py<'py>(
    py: Python<'py>,
    results: Vec<TempResult>,
    n_measurements: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let mut py_results = Vec::with_capacity(results.len());
    for r in results {
        let dict = PyDict::new(py);
        dict.set_item("temperature", r.temperature)?;
        dict.set_item("energies", r.energies.into_pyarray(py))?;
        dict.set_item("magnetizations", r.magnetizations.into_pyarray(py))?;

        if let Some(configs) = r.configs {
            let flat = numpy::PyArray1::from_vec(py, configs);
            let mut reshape_dims: Vec<usize> = vec![n_measurements];
            reshape_dims.extend_from_slice(&r.shape);
            let reshaped = flat
                .reshape(reshape_dims)
                .expect("reshape should not fail for correct dimensions");
            dict.set_item("configurations", reshaped)?;
        }

        py_results.push(dict);
    }
    Ok(py_results)
}

// ═══════════════════════════════════════════════════════════════════
// Parallel Tempering
// ═══════════════════════════════════════════════════════════════════

/// Run Parallel Tempering: N replicas at different temperatures with
/// periodic swap attempts between adjacent replicas.
///
/// Swaps use the standard Metropolis criterion:
///   P(swap i,j) = min(1, exp((β_i - β_j) × (E_i - E_j)))
///
/// Even/odd alternation ensures all adjacent pairs get swap opportunities.
#[pyfunction]
#[pyo3(signature = (
    lattice_size, j1, j2, j3, h, base_seed, algorithm, lattice_type,
    temperatures, n_thermalization, n_sweeps, measurement_interval,
    swap_interval = 1, store_configs = false
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
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let algo = algorithm.to_string();
    let lat_type = lattice_type.to_string();
    let n_measurements = n_sweeps / measurement_interval.max(1);

    let results: Vec<TempResult> = py.allow_threads(|| {
        let n_temps = temperatures.len();

        // Sort temperatures ascending for swap logic.
        let mut sorted_temps = temperatures.clone();
        sorted_temps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let betas: Vec<f64> = sorted_temps.iter().map(|&t| 1.0 / t).collect();

        // Create one replica per temperature.
        let mut replicas: Vec<IsingSimulation> = (0..n_temps)
            .map(|i| {
                IsingSimulation::new_internal(
                    lattice_size, j1, j2, j3, h,
                    base_seed.wrapping_add(i as u64),
                    &algo, &lat_type,
                )
                .expect("simulation creation should not fail with validated params")
            })
            .collect();

        let num_sites = with_lattice!(&replicas[0].lattice, lat => lat.num_sites());
        let shape = with_lattice!(&replicas[0].lattice, lat => lat.shape().to_vec());

        // Thermalize all replicas in parallel.
        replicas.par_iter_mut().enumerate().for_each(|(i, sim)| {
            sim.sweep_internal(n_thermalization, betas[i]);
        });

        // Separate RNG for swap decisions (deterministic, independent of replica RNGs).
        let mut swap_rng = crate::rng::create_rng(
            base_seed.wrapping_add(n_temps as u64 + 1000),
        );

        // Pre-allocate result storage.
        let mut temp_results: Vec<TempResult> = sorted_temps
            .iter()
            .map(|&t| TempResult {
                temperature: t,
                energies: Vec::with_capacity(n_measurements),
                magnetizations: Vec::with_capacity(n_measurements),
                configs: if store_configs {
                    Some(Vec::with_capacity(n_measurements * num_sites))
                } else {
                    None
                },
                num_sites,
                shape: shape.clone(),
            })
            .collect();

        // Compute initial energies (total, not per-site, for swap criterion).
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
            // a. Parallel sweeps.
            let sweeps_this_round = swap_interval.min(n_sweeps - sweep_count);
            replicas.par_iter_mut().enumerate().for_each(|(i, sim)| {
                sim.sweep_internal(sweeps_this_round, betas[i]);
            });
            sweep_count += sweeps_this_round;

            // Update cached energies.
            for (i, sim) in replicas.iter().enumerate() {
                energies[i] = with_lattice!(&sim.lattice, lat => {
                    observables::energy_per_site(&sim.spins, lat, j1, j2, j3, h)
                });
            }

            // b. Swap attempts (even/odd alternation).
            let offset = if round % 2 == 0 { 0 } else { 1 };
            for i in (offset..n_temps.saturating_sub(1)).step_by(2) {
                let j = i + 1;
                // Standard PT acceptance: P = min(1, exp(delta))
                // where delta = (β_i - β_j) * (E_i - E_j) * N
                // β sorted descending (β_i > β_j), so if E_i < E_j
                // (low-T replica has lower energy), delta > 0 → always accept.
                let delta = (betas[i] - betas[j])
                    * (energies[i] - energies[j])
                    * num_sites as f64;
                let accept = delta >= 0.0
                    || swap_rng.gen::<f64>() < delta.exp();
                if accept {
                    // O(1) pointer swap of spin Vecs.
                    let (left, right) = replicas.split_at_mut(j);
                    std::mem::swap(&mut left[i].spins, &mut right[0].spins);
                    // Swap cached energies too.
                    energies.swap(i, j);
                }
            }
            round += 1;

            // c. Collect measurements at the right intervals.
            if sweep_count % measurement_interval == 0 {
                for (i, sim) in replicas.iter().enumerate() {
                    temp_results[i].energies.push(energies[i]);
                    temp_results[i].magnetizations.push(
                        observables::magnetization_per_site(&sim.spins),
                    );
                    if let Some(ref mut c) = temp_results[i].configs {
                        c.extend_from_slice(&sim.spins);
                    }
                }
            }
        }

        temp_results
    });

    convert_results_to_py(py, results, n_measurements)
}
