//! Per-temperature measurement record shared by every batched run path.
//!
//! The cooldown production block (`IsingSimulation::production_sweeps`), the
//! independent runner and the parallel-tempering runner record the same
//! observables after each measurement block; this module owns that record
//! and its conversion to the per-temperature dict every path hands back to
//! Python.

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::lattice::Lattice;
use crate::observables;

/// Correlation data collected for one temperature.
///
/// `distances`/`values` hold the correlation function of the most recent
/// evaluation (the representative pair); `lengths` collects the correlation
/// length at every evaluation.
pub(crate) struct CorrelationRecord {
    pub(crate) distances: Vec<f64>,
    pub(crate) values: Vec<f64>,
    pub(crate) lengths: Vec<f64>,
}

/// Result for a single temperature, returned to Python as a dict.
pub(crate) struct TempResult {
    pub(crate) temperature: f64,
    pub(crate) energies: Vec<f64>,
    pub(crate) magnetizations: Vec<f64>,
    pub(crate) configs: Option<Vec<i8>>,
    pub(crate) correlation: Option<CorrelationRecord>,
    pub(crate) shape: Vec<usize>,
    /// Cluster flips during measurement sweeps only (0 for Metropolis);
    /// thermalization work is never counted.
    pub(crate) cluster_flips: usize,
    /// Evaluate the correlation observables at every k-th measurement (the
    /// k-th, 2k-th, ...), so `k == 1` records them at every measurement and
    /// `k == n_measurements` exactly once, at the final one. Validated
    /// `>= 1` at the Python boundary; clamped here so the record itself can
    /// never divide by zero.
    correlation_interval: usize,
}

impl TempResult {
    pub(crate) fn with_capacity(
        temperature: f64,
        n_measurements: usize,
        num_sites: usize,
        shape: Vec<usize>,
        store_configs: bool,
        compute_correlation: bool,
        correlation_interval: usize,
    ) -> Self {
        debug_assert!(correlation_interval >= 1, "validated at the boundary");
        let correlation_interval = correlation_interval.max(1);
        Self {
            temperature,
            energies: Vec::with_capacity(n_measurements),
            magnetizations: Vec::with_capacity(n_measurements),
            configs: store_configs.then(|| Vec::with_capacity(n_measurements * num_sites)),
            correlation: compute_correlation.then(|| CorrelationRecord {
                distances: Vec::new(),
                values: Vec::new(),
                lengths: Vec::with_capacity(n_measurements / correlation_interval),
            }),
            shape,
            cluster_flips: 0,
            correlation_interval,
        }
    }

    /// Record one measurement: energy, magnetization, and (when enabled)
    /// a configuration snapshot and, at the cadence, correlation data.
    ///
    /// The correlation observables are pure functions of `spins` — they draw
    /// no RNG, so enabling them never perturbs the sampling streams. The
    /// bins are computed once and feed both the correlation length and the
    /// stored correlation function.
    pub(crate) fn push<L: Lattice>(&mut self, spins: &[i8], lattice: &L, energy: f64) {
        self.energies.push(energy);
        self.magnetizations
            .push(observables::magnetization_per_site(spins));
        if let Some(ref mut c) = self.configs {
            c.extend_from_slice(spins);
        }
        let count = self.energies.len();
        if let Some(ref mut corr) = self.correlation {
            if count.is_multiple_of(self.correlation_interval) {
                let bins = observables::correlation_bins(spins, lattice);
                corr.lengths
                    .push(observables::correlation_length(&bins, lattice.dimension()));
                corr.distances = bins.distances();
                corr.values = bins.correlations;
            }
        }
    }

    /// Convert to the Python dict every run path returns.
    ///
    /// Keys: `temperature`, `energies`, `magnetizations`, `n_cluster_flips`,
    /// plus `configurations` when stored and `correlation_distances` /
    /// `correlation_function` / `correlation_length` when computed. Array
    /// shapes are derived from the actual number of collected measurements,
    /// never from a precomputed count — a cadence bug upstream can therefore
    /// shorten arrays but can no longer cause a reshape panic.
    pub(crate) fn into_pydict(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let n_measurements = self.energies.len();
        let dict = PyDict::new(py);
        dict.set_item("temperature", self.temperature)?;
        dict.set_item("energies", self.energies.into_pyarray(py))?;
        dict.set_item("magnetizations", self.magnetizations.into_pyarray(py))?;
        dict.set_item("n_cluster_flips", self.cluster_flips)?;

        if let Some(configs) = self.configs {
            let flat = numpy::PyArray1::from_vec(py, configs);
            let mut reshape_dims: Vec<usize> = vec![n_measurements];
            reshape_dims.extend_from_slice(&self.shape);
            let reshaped = flat.reshape(reshape_dims)?;
            dict.set_item("configurations", reshaped)?;
        }

        if let Some(corr) = self.correlation {
            dict.set_item("correlation_distances", corr.distances.into_pyarray(py))?;
            dict.set_item("correlation_function", corr.values.into_pyarray(py))?;
            dict.set_item("correlation_length", corr.lengths.into_pyarray(py))?;
        }

        Ok(dict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::square::SquareLattice;

    const L: usize = 4;
    const N: usize = L * L;

    /// Five distinct configurations: all up, then one more spin flipped
    /// each time (so each push sees a different correlation function).
    fn configurations() -> Vec<Vec<i8>> {
        (0..5)
            .map(|k| {
                let mut spins = vec![1i8; N];
                for s in spins.iter_mut().take(k) {
                    *s = -1;
                }
                spins
            })
            .collect()
    }

    fn record(store_configs: bool, compute_correlation: bool, k: usize) -> TempResult {
        let lattice = SquareLattice::new(L).unwrap();
        let mut result =
            TempResult::with_capacity(2.0, 5, N, vec![L, L], store_configs, compute_correlation, k);
        for (i, spins) in configurations().iter().enumerate() {
            result.push(spins, &lattice, -(i as f64));
        }
        result
    }

    #[test]
    fn test_scalar_series_are_recorded_at_every_measurement() {
        let r = record(false, false, 1);
        assert_eq!(r.energies, vec![0.0, -1.0, -2.0, -3.0, -4.0]);
        assert_eq!(r.magnetizations.len(), 5);
        assert!((r.magnetizations[1] - 14.0 / 16.0).abs() < 1e-15);
        assert!(r.configs.is_none());
        assert!(r.correlation.is_none());
        assert_eq!(r.cluster_flips, 0);
    }

    #[test]
    fn test_configurations_are_appended_flat_in_measurement_order() {
        let r = record(true, false, 1);
        let configs = r.configs.expect("store_configs");
        assert_eq!(configs.len(), 5 * N);
        let expected: Vec<i8> = configurations().concat();
        assert_eq!(configs, expected);
    }

    #[test]
    fn test_correlation_interval_one_evaluates_every_measurement() {
        let r = record(false, true, 1);
        let corr = r.correlation.expect("compute_correlation");
        assert_eq!(corr.lengths.len(), 5);
        let lattice = SquareLattice::new(L).unwrap();
        let last = observables::correlation_bins(&configurations()[4], &lattice);
        assert_eq!(corr.values, last.correlations);
        assert_eq!(corr.distances, last.distances());
    }

    #[test]
    fn test_correlation_interval_selects_every_kth_measurement() {
        // k = 2 with 5 measurements: evaluations after measurements 2 and 4,
        // so the stored function is the 4th configuration's.
        let r = record(false, true, 2);
        let corr = r.correlation.expect("compute_correlation");
        assert_eq!(corr.lengths.len(), 2);
        let lattice = SquareLattice::new(L).unwrap();
        let fourth = observables::correlation_bins(&configurations()[3], &lattice);
        assert_eq!(corr.values, fourth.correlations);

        // k = 3: evaluation after measurement 3 only.
        let r = record(false, true, 3);
        let corr = r.correlation.expect("compute_correlation");
        assert_eq!(corr.lengths.len(), 1);
        let third = observables::correlation_bins(&configurations()[2], &lattice);
        assert_eq!(corr.values, third.correlations);

        // k = n_measurements: exactly once, at the final measurement.
        let r = record(false, true, 5);
        let corr = r.correlation.expect("compute_correlation");
        assert_eq!(corr.lengths.len(), 1);
        let fifth = observables::correlation_bins(&configurations()[4], &lattice);
        assert_eq!(corr.values, fifth.correlations);
    }

    #[test]
    fn test_correlation_interval_larger_than_run_records_nothing() {
        let r = record(false, true, 6);
        let corr = r.correlation.expect("compute_correlation");
        assert!(corr.lengths.is_empty());
        assert!(corr.values.is_empty());
        assert!(corr.distances.is_empty());
    }

    #[test]
    fn test_scalar_series_do_not_depend_on_the_correlation_cadence() {
        let every = record(true, true, 1);
        let sparse = record(true, true, 2);
        assert_eq!(every.energies, sparse.energies);
        assert_eq!(every.magnetizations, sparse.magnetizations);
        assert_eq!(every.configs, sparse.configs);
    }
}
