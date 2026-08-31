use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

use crate::algorithm::metropolis::Metropolis;
use crate::algorithm::swendsen_wang::SwendsenWang;
use crate::algorithm::wolff::Wolff;
use crate::algorithm::{AlgorithmKind, AlgorithmState, McAlgorithm, SweepResult};
use crate::autocorrelation;
use crate::error::MCIsingError;
use crate::lattice::{with_lattice, Lattice, LatticeKind};
use crate::observables;
use crate::rng::create_rng;

/// Core Ising model simulation engine.
///
/// This is the PyO3 boundary class that owns the Rust lattice, spins, and RNG.
/// All physics computation happens in Rust; Python calls methods on this class.
#[pyclass]
pub struct IsingSimulation {
    pub(crate) spins: Vec<i8>,
    pub(crate) lattice: LatticeKind,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
    rng: Xoshiro256StarStar,
    lattice_size: usize,
    /// Shape for reshaping spin arrays (e.g. [L, L] for 2D lattices).
    shape: Vec<usize>,
    algorithm: AlgorithmState,
}

/// Per-run measurement arrays:
/// (energies, magnetizations, optional configs, cluster_flips).
type SweepMeasurements<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Option<PyObject>,
    usize,
);

/// Validate a user-supplied temperature and return beta = 1/T.
fn beta_from_temperature(temperature: f64) -> Result<f64, MCIsingError> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(MCIsingError::InvalidTemperature(temperature));
    }
    Ok(1.0 / temperature)
}

impl IsingSimulation {
    /// Create a new simulation (pure Rust, no PyO3 dependency).
    /// Used by both the PyO3 `__new__` and the parallel runner.
    ///
    /// # Errors
    ///
    /// Returns an error if `algorithm` or `lattice_type` is not recognized
    /// or the lattice cannot be constructed at `lattice_size`.
    pub fn new_internal(
        lattice_size: usize,
        j1: f64,
        j2: f64,
        j3: f64,
        h: f64,
        seed: u64,
        algorithm: &str,
        lattice_type: &str,
    ) -> Result<Self, MCIsingError> {
        let lattice = LatticeKind::from_str(lattice_type, lattice_size)?;

        if !j1.is_finite() {
            return Err(MCIsingError::InvalidCoupling("j1", j1));
        }
        if !j2.is_finite() {
            return Err(MCIsingError::InvalidCoupling("j2", j2));
        }
        if !j3.is_finite() {
            return Err(MCIsingError::InvalidCoupling("j3", j3));
        }
        if !h.is_finite() {
            return Err(MCIsingError::InvalidCoupling("h", h));
        }

        let algo_kind = AlgorithmKind::from_str(algorithm)?;

        if algo_kind.requires_no_frustration() && (j2 != 0.0 || j3 != 0.0 || h != 0.0) {
            return Err(MCIsingError::ClusterAlgorithmConstraint(
                algo_kind.name().to_string(),
            ));
        }

        if algo_kind.requires_ferromagnetic_j1() && j1 <= 0.0 {
            return Err(MCIsingError::ClusterCouplingSign(
                algo_kind.name().to_string(),
            ));
        }

        if j2 != 0.0 && lattice.nnn_coordination_number() == 0 {
            return Err(MCIsingError::InvalidCoupling(
                "j2 (no NNN defined for this lattice)",
                j2,
            ));
        }
        if j3 != 0.0 && lattice.tnn_coordination_number() == 0 {
            return Err(MCIsingError::InvalidCoupling(
                "j3 (no TNN defined for this lattice)",
                j3,
            ));
        }

        let num_sites = lattice.num_sites();
        let z_nn = lattice.coordination_number();
        let z_nnn = lattice.nnn_coordination_number();
        let z_tnn = lattice.tnn_coordination_number();
        let shape = lattice.shape().to_vec();

        let mut rng = create_rng(seed);
        let spins: Vec<i8> = (0..num_sites)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();

        let algorithm = match algo_kind {
            AlgorithmKind::Metropolis => AlgorithmState::Metropolis(Box::new(Metropolis::new(
                j1, j2, j3, h, z_nn, z_nnn, z_tnn,
            ))),
            AlgorithmKind::Wolff => AlgorithmState::Wolff(Wolff::new(num_sites)),
            AlgorithmKind::SwendsenWang => {
                AlgorithmState::SwendsenWang(SwendsenWang::new(num_sites))
            }
        };

        Ok(Self {
            spins,
            lattice,
            j1,
            j2,
            j3,
            h,
            rng,
            lattice_size,
            shape,
            algorithm,
        })
    }

    /// Replace the spin configuration (pure Rust, no PyO3).
    ///
    /// # Errors
    ///
    /// Returns an error if `data` has the wrong length for this lattice
    /// or contains a value other than +1/-1.
    pub(crate) fn set_spins_internal(&mut self, data: &[i8]) -> Result<(), MCIsingError> {
        let expected = self.lattice.num_sites();
        if data.len() != expected {
            return Err(MCIsingError::InvalidSpinConfiguration(format!(
                "Expected {expected} spins, got {total}",
                total = data.len()
            )));
        }

        for &val in data {
            if val != 1 && val != -1 {
                return Err(MCIsingError::InvalidSpinConfiguration(format!(
                    "All spins must be +1 or -1, found {val}"
                )));
            }
        }

        self.spins.clear();
        self.spins.extend_from_slice(data);
        Ok(())
    }

    /// Restore the RNG from its serialized state (pure Rust, no PyO3).
    ///
    /// # Errors
    ///
    /// Returns an error if `state` is not a serialized Xoshiro256** state.
    pub(crate) fn set_rng_state_internal(&mut self, state: &[u8]) -> Result<(), MCIsingError> {
        let rng: Xoshiro256StarStar = serde_json::from_slice(state).map_err(|e| {
            MCIsingError::InvalidSpinConfiguration(format!("Invalid RNG state: {e}"))
        })?;
        self.rng = rng;
        Ok(())
    }

    /// Flip the spin at a flat site index (pure Rust, no PyO3).
    ///
    /// # Errors
    ///
    /// Returns an error if `site` is out of bounds.
    pub(crate) fn flip_spin_internal(&mut self, site: usize) -> Result<(), MCIsingError> {
        if site >= self.spins.len() {
            return Err(MCIsingError::InvalidSpinConfiguration(format!(
                "Site index {site} out of bounds for lattice with {n} sites",
                n = self.spins.len()
            )));
        }
        self.spins[site] = -self.spins[site];
        Ok(())
    }

    /// Local energy of the spin at a flat site index (pure Rust, no PyO3).
    ///
    /// # Errors
    ///
    /// Returns an error if `site` is out of bounds.
    pub(crate) fn spin_energy_internal(&self, site: usize) -> Result<f64, MCIsingError> {
        if site >= self.spins.len() {
            return Err(MCIsingError::InvalidSpinConfiguration(format!(
                "Site index {site} out of bounds for lattice with {n} sites",
                n = self.spins.len()
            )));
        }

        with_lattice!(&self.lattice, lat => {
            let spin = f64::from(self.spins[site]);
            let mut local_field: f64 = 0.0;
            for &nbr in lat.nearest_neighbors(site) {
                local_field += self.j1 * f64::from(self.spins[nbr]);
            }
            for &nbr in lat.next_nearest_neighbors(site) {
                local_field += self.j2 * f64::from(self.spins[nbr]);
            }
            for &nbr in lat.third_nearest_neighbors(site) {
                local_field += self.j3 * f64::from(self.spins[nbr]);
            }
            Ok(-spin * local_field - self.h * spin)
        })
    }

    /// Perform sweeps (pure Rust, no PyO3). Used by parallel runner.
    pub fn sweep_internal(&mut self, n_sweeps: usize, beta: f64) -> SweepResult {
        let mut total = SweepResult {
            accepted: 0,
            attempted: 0,
            cluster_flips: 0,
        };
        for _ in 0..n_sweeps {
            let r = self.dispatch_sweep(beta);
            total.accepted += r.accepted;
            total.attempted += r.attempted;
            total.cluster_flips += r.cluster_flips;
        }
        total
    }
}

#[pymethods]
impl IsingSimulation {
    /// Create a new Ising simulation (PyO3 entry point).
    #[new]
    #[pyo3(signature = (lattice_size, j1, j2, j3, h, seed, algorithm = "metropolis", lattice_type = "square"))]
    fn new(
        lattice_size: usize,
        j1: f64,
        j2: f64,
        j3: f64,
        h: f64,
        seed: u64,
        algorithm: &str,
        lattice_type: &str,
    ) -> PyResult<Self> {
        Self::new_internal(lattice_size, j1, j2, j3, h, seed, algorithm, lattice_type)
            .map_err(std::convert::Into::into)
    }

    /// Perform MC sweeps at the given temperature.
    ///
    /// Returns (accepted, attempted, cluster_flips) as a tuple; see
    /// `SweepResult` for the per-algorithm meaning of each counter.
    #[pyo3(signature = (n_sweeps = 1, *, temperature))]
    fn sweep(&mut self, n_sweeps: usize, temperature: f64) -> PyResult<(usize, usize, usize)> {
        let beta = beta_from_temperature(temperature)?;
        let total = self.sweep_internal(n_sweeps, beta);
        Ok((total.accepted, total.attempted, total.cluster_flips))
    }

    #[getter]
    fn algorithm_name(&self) -> &str {
        self.algorithm.name()
    }

    fn energy(&self) -> f64 {
        self.compute_energy()
    }

    fn magnetization(&self) -> f64 {
        observables::magnetization_per_site(&self.spins)
    }

    /// Return the spin configuration as a NumPy array with lattice shape.
    fn get_spins(&self, py: Python<'_>) -> PyResult<PyObject> {
        let flat = numpy::PyArray1::from_vec(py, self.spins.clone());
        let shape: Vec<usize> = self.shape.clone();
        let reshaped = flat
            .reshape(shape)
            .map_err(|e| MCIsingError::InvalidSpinConfiguration(format!("reshape failed: {e}")))?;
        Ok(reshaped.into_any().unbind())
    }

    /// Set the spin configuration from a NumPy array.
    fn set_spins(&mut self, spins: PyReadonlyArrayDyn<'_, i8>) -> PyResult<()> {
        let data = spins.as_slice().map_err(|e| {
            MCIsingError::InvalidSpinConfiguration(format!("Cannot read array: {e}"))
        })?;
        self.set_spins_internal(data).map_err(Into::into)
    }

    /// Flip the spin at a flat site index.
    ///
    /// Flat (row-major) indexing is the one scheme every lattice shares —
    /// a (row, col) pair cannot address cubic ([L, L, L]) or honeycomb
    /// ([L, L, 2]) sites (B6).
    fn flip_spin(&mut self, site: usize) -> PyResult<()> {
        self.flip_spin_internal(site).map_err(Into::into)
    }

    /// Compute the local energy of the spin at a flat site index.
    fn spin_energy(&self, site: usize) -> PyResult<f64> {
        self.spin_energy_internal(site).map_err(Into::into)
    }

    /// Compute the correlation function.
    fn correlation_function<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        with_lattice!(&self.lattice, lat => {
            let bins = observables::correlation_bins(&self.spins, lat);
            (bins.distances().into_pyarray(py), bins.correlations.into_pyarray(py))
        })
    }

    /// Compute the correlation length from the current spin configuration.
    fn correlation_length(&self) -> f64 {
        with_lattice!(&self.lattice, lat => {
            let bins = observables::correlation_bins(&self.spins, lat);
            observables::correlation_length(&bins, lat.dimension())
        })
    }

    #[getter]
    fn lattice_size(&self) -> usize {
        self.lattice_size
    }

    #[getter]
    fn num_sites(&self) -> usize {
        self.lattice.num_sites()
    }

    #[getter]
    fn j1(&self) -> f64 {
        self.j1
    }

    #[getter]
    fn j2(&self) -> f64 {
        self.j2
    }

    #[getter]
    fn j3(&self) -> f64 {
        self.j3
    }

    #[getter]
    fn h(&self) -> f64 {
        self.h
    }

    fn get_rng_state(&self) -> Vec<u8> {
        // Infallible in practice: serializing a fixed 4×u64 generator state
        // to a Vec cannot hit an I/O or unsupported-type error.
        serde_json::to_vec(&self.rng).expect("Xoshiro256StarStar serialization should not fail")
    }

    fn set_rng_state(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.set_rng_state_internal(&state).map_err(Into::into)
    }

    /// Anneal along a temperature schedule: one sweep per positive entry.
    ///
    /// The ramp is pure thermalization — nothing is recorded, and
    /// non-positive schedule entries are skipped silently (a linspace
    /// ramp may in principle cross zero).
    fn anneal(&mut self, temp_schedule: Vec<f64>) {
        for temp in &temp_schedule {
            if *temp <= 0.0 {
                continue;
            }
            let beta = 1.0 / temp;
            self.dispatch_sweep(beta);
        }
    }

    /// Sweep at fixed temperature, recording the energy after every sweep.
    #[pyo3(signature = (n_sweeps, *, temperature))]
    fn extend_thermalization<'py>(
        &mut self,
        py: Python<'py>,
        n_sweeps: usize,
        temperature: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let beta = beta_from_temperature(temperature)?;
        let mut energies = Vec::with_capacity(n_sweeps);
        for _ in 0..n_sweeps {
            self.dispatch_sweep(beta);
            energies.push(self.compute_energy());
        }
        Ok(energies.into_pyarray(py))
    }

    #[staticmethod]
    fn analyze_thermalization_series<'py>(
        py: Python<'py>,
        series: PyReadonlyArray1<'py, f64>,
        c_window: f64,
        tau_multiplier: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let data = series.as_slice().map_err(|e| {
            MCIsingError::InvalidSpinConfiguration(format!("Cannot read array: {e}"))
        })?;
        let analysis = autocorrelation::analyze_thermalization(data, c_window, tau_multiplier);
        let dict = PyDict::new(py);
        dict.set_item("truncation_point", analysis.thermalization.truncation_point)?;
        dict.set_item("is_thermalized", analysis.thermalization.is_thermalized)?;
        dict.set_item("tau_int", analysis.autocorrelation.tau_int)?;
        dict.set_item("window", analysis.autocorrelation.window)?;
        dict.set_item("recommended_interval", analysis.recommended_interval)?;
        Ok(dict)
    }

    /// Run production measurement sweeps, collecting observables at each interval.
    ///
    /// The 4th tuple element is the total number of cluster flips across
    /// all production sweeps (0 for Metropolis).
    #[pyo3(signature = (n_measurements, interval, *, temperature, store_configs))]
    fn production_sweeps<'py>(
        &mut self,
        py: Python<'py>,
        n_measurements: usize,
        interval: usize,
        temperature: f64,
        store_configs: bool,
    ) -> PyResult<SweepMeasurements<'py>> {
        let beta = beta_from_temperature(temperature)?;

        let mut energies = Vec::with_capacity(n_measurements);
        let mut magnetizations = Vec::with_capacity(n_measurements);
        let mut cluster_flips = 0;
        let mut configs: Option<Vec<i8>> = if store_configs {
            Some(Vec::with_capacity(n_measurements * self.spins.len()))
        } else {
            None
        };

        for _ in 0..n_measurements {
            cluster_flips += self.sweep_internal(interval, beta).cluster_flips;
            energies.push(self.compute_energy());
            magnetizations.push(observables::magnetization_per_site(&self.spins));
            if let Some(ref mut c) = configs {
                c.extend_from_slice(&self.spins);
            }
        }

        let py_energies = energies.into_pyarray(py);
        let py_mags = magnetizations.into_pyarray(py);
        let py_configs = match configs {
            Some(c) => {
                let flat = numpy::PyArray1::from_vec(py, c);
                let mut reshape_dims: Vec<usize> = vec![n_measurements];
                reshape_dims.extend_from_slice(&self.shape);
                Some(flat.reshape(reshape_dims)?.into_any().unbind())
            }
            None => None,
        };

        Ok((py_energies, py_mags, py_configs, cluster_flips))
    }

    fn __repr__(&self) -> String {
        format!(
            "IsingSimulation(lattice_size={}, algorithm={}, j1={}, j2={}, j3={}, h={}, energy={:.4}, mag={:.4})",
            self.lattice_size,
            self.algorithm.name(),
            self.j1,
            self.j2,
            self.j3,
            self.h,
            self.energy(),
            self.magnetization()
        )
    }
}

impl IsingSimulation {
    /// Compute energy via with_lattice! dispatch for monomorphization.
    fn compute_energy(&self) -> f64 {
        with_lattice!(&self.lattice, lat => {
            observables::energy_per_site(&self.spins, lat, self.j1, self.j2, self.j3, self.h)
        })
    }

    /// Dispatch a single sweep via with_lattice! × algorithm match.
    /// Each combination is monomorphized — no virtual dispatch.
    fn dispatch_sweep(&mut self, beta: f64) -> SweepResult {
        // We need to split borrows: lattice is immutable, everything else mutable.
        // Use with_lattice! on a reference to avoid moving self.lattice.
        match &mut self.algorithm {
            AlgorithmState::Metropolis(m) => {
                with_lattice!(&self.lattice, lat => {
                    m.sweep(
                        &mut self.spins, lat,
                        self.j1, self.j2, self.j3, self.h, beta, &mut self.rng,
                    )
                })
            }
            AlgorithmState::Wolff(w) => {
                with_lattice!(&self.lattice, lat => {
                    w.sweep(
                        &mut self.spins, lat,
                        self.j1, self.j2, self.j3, self.h, beta, &mut self.rng,
                    )
                })
            }
            AlgorithmState::SwendsenWang(sw) => {
                with_lattice!(&self.lattice, lat => {
                    sw.sweep(
                        &mut self.spins, lat,
                        self.j1, self.j2, self.j3, self.h, beta, &mut self.rng,
                    )
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Result::expect_err` needs `Debug` on the Ok type, which the
    /// production struct deliberately does not derive.
    fn constructor_error(j1: f64, algorithm: &str) -> MCIsingError {
        match IsingSimulation::new_internal(4, j1, 0.0, 0.0, 0.0, 42, algorithm, "square") {
            Ok(_) => panic!("J1={j1} must be rejected for {algorithm}"),
            Err(e) => e,
        }
    }

    #[test]
    fn test_new_internal_rejects_cluster_with_negative_j1() {
        for algorithm in ["wolff", "swendsen_wang"] {
            let err = constructor_error(-1.0, algorithm);
            assert!(err.to_string().contains("requires J1>0"), "{err}");
        }
    }

    #[test]
    fn test_new_internal_rejects_cluster_with_zero_j1() {
        // J1=0 gives p_add=0: cluster growth never adds a site, so the
        // "cluster" update degenerates just like J1<0 does.
        for algorithm in ["wolff", "swendsen_wang"] {
            let err = constructor_error(0.0, algorithm);
            assert!(err.to_string().contains("requires J1>0"), "{err}");
        }
    }

    #[test]
    fn test_new_internal_accepts_cluster_with_positive_j1() {
        for algorithm in ["wolff", "swendsen_wang"] {
            assert!(
                IsingSimulation::new_internal(4, 1.0, 0.0, 0.0, 0.0, 42, algorithm, "square")
                    .is_ok()
            );
        }
    }

    #[test]
    fn test_new_internal_metropolis_accepts_negative_j1() {
        assert!(
            IsingSimulation::new_internal(4, -1.0, 0.0, 0.0, 0.0, 42, "metropolis", "square")
                .is_ok()
        );
    }

    /// A ferromagnetic 4×4 square simulation, the workhorse of the
    /// state-continuation tests below.
    fn square_sim(algorithm: &str) -> IsingSimulation {
        IsingSimulation::new_internal(4, 1.0, 0.0, 0.0, 0.0, 42, algorithm, "square")
            .expect("valid constructor arguments")
    }

    #[test]
    // Bit-identity IS the contract under test — approximate agreement would
    // hide a reseeded or diverged RNG stream.
    #[allow(clippy::float_cmp)]
    fn test_rng_roundtrip_bit_identical_continuation() {
        // The checkpoint contract: capturing (spins, rng_state) mid-run and
        // restoring both into a FRESH simulation must reproduce the original
        // trajectory bit-for-bit. Exercised for all three algorithms — Wolff
        // and Swendsen-Wang carry scratch buffers that are NOT serialized,
        // so this also pins that the scratch is stateless between sweeps.
        for algorithm in ["metropolis", "wolff", "swendsen_wang"] {
            let beta = 0.5;
            let mut original = square_sim(algorithm);
            original.sweep_internal(10, beta);
            let state = original.get_rng_state();
            let snapshot = original.spins.clone();

            original.sweep_internal(20, beta);
            let spins_ahead = original.spins.clone();
            let energy_ahead = original.energy();

            let mut restored = square_sim(algorithm);
            restored
                .set_spins_internal(&snapshot)
                .expect("snapshot has valid length and values");
            restored
                .set_rng_state_internal(&state)
                .expect("state round-trips");
            restored.sweep_internal(20, beta);

            assert_eq!(restored.spins, spins_ahead, "algorithm={algorithm}");
            assert_eq!(restored.energy(), energy_ahead, "algorithm={algorithm}");
        }
    }

    #[test]
    fn test_set_rng_state_restores_old_stream() {
        // Restoring a STALE state must rewind the stream: replaying the
        // same sweeps from the same spins reproduces the first replay even
        // after the generator has been advanced far past the capture point.
        let beta = 0.5;
        let mut sim = square_sim("metropolis");
        sim.sweep_internal(10, beta);
        let state = sim.get_rng_state();
        let snapshot = sim.spins.clone();

        sim.sweep_internal(5, beta);
        let first_replay = sim.spins.clone();

        sim.sweep_internal(7, beta); // advance well past the capture point
        sim.set_spins_internal(&snapshot)
            .expect("snapshot round-trips");
        sim.set_rng_state_internal(&state)
            .expect("state round-trips");
        sim.sweep_internal(5, beta);

        assert_eq!(sim.spins, first_replay);
    }

    #[test]
    fn test_set_rng_state_malformed_is_err() {
        let mut sim = square_sim("metropolis");
        for bad in [&b"not json"[..], b"", b"{}"] {
            let err = sim
                .set_rng_state_internal(bad)
                .expect_err("malformed state must be rejected");
            assert!(err.to_string().contains("Invalid RNG state"), "{err}");
        }
    }

    #[test]
    fn test_flip_energy_identity() {
        // Local/global consistency (B6 regression class): flipping site i
        // changes the total energy by -2·spin_energy(i), i.e. the per-site
        // energy() by -2·spin_energy(i)/N. Couplings cover every shell plus
        // the field so all neighbor tables enter the identity.
        let mut sim =
            IsingSimulation::new_internal(4, 1.0, 0.3, 0.2, 0.1, 42, "metropolis", "square")
                .expect("square supports j2/j3");
        sim.sweep_internal(5, 0.7); // a generic (non-symmetric) configuration
        let n = sim.spins.len() as f64;

        for site in [0, 7, 15] {
            let before = sim.energy();
            let local = sim.spin_energy_internal(site).expect("site in bounds");
            sim.flip_spin_internal(site).expect("site in bounds");
            let after = sim.energy();
            let expected_shift = -2.0 * local / n;
            assert!(
                (after - before - expected_shift).abs() < 1e-12,
                "site={site}: shift {got:.17} vs expected {expected_shift:.17}",
                got = after - before
            );
        }
    }

    #[test]
    // Double flip restores the exact configuration, so the recomputed energy
    // follows the identical summation path — bit-equality is the contract.
    #[allow(clippy::float_cmp)]
    fn test_double_flip_restores_state() {
        let mut sim = square_sim("metropolis");
        sim.sweep_internal(3, 0.7);
        let spins_before = sim.spins.clone();
        let energy_before = sim.energy();

        sim.flip_spin_internal(5).expect("site in bounds");
        sim.flip_spin_internal(5).expect("site in bounds");

        assert_eq!(sim.spins, spins_before);
        assert_eq!(sim.energy(), energy_before);
    }

    #[test]
    fn test_flip_and_spin_energy_out_of_bounds_is_err() {
        let mut sim = square_sim("metropolis");
        let n = sim.spins.len();
        for site in [n, usize::MAX] {
            let err = sim
                .flip_spin_internal(site)
                .expect_err("out-of-bounds flip must be rejected");
            assert!(err.to_string().contains("out of bounds"), "{err}");
            let err = sim
                .spin_energy_internal(site)
                .expect_err("out-of-bounds spin_energy must be rejected");
            assert!(err.to_string().contains("out of bounds"), "{err}");
        }
    }

    #[test]
    // All-up and Néel energies are exact in floating point (spins ±1,
    // couplings that are binary fractions), so equality is exact.
    #[allow(clippy::float_cmp)]
    fn test_set_spins_internal_roundtrip() {
        // All-up ferromagnet with a field: E/site = -(z/2)·j1 - h = -2.5
        // on the square lattice (z=4) with j1=1, h=0.5 — every term is a
        // binary fraction, so the sum is exact.
        let mut sim =
            IsingSimulation::new_internal(4, 1.0, 0.0, 0.0, 0.5, 42, "metropolis", "square")
                .expect("valid constructor arguments");
        let all_up = vec![1_i8; 16];
        sim.set_spins_internal(&all_up).expect("valid spins");
        assert_eq!(sim.spins, all_up);
        assert_eq!(sim.energy(), -2.5);

        // Néel checkerboard at h=0: every NN bond is antialigned, so
        // E/site = +(z/2)·j1 = +2 — pins the sign convention.
        let mut sim = square_sim("metropolis");
        let checkerboard: Vec<i8> = (0..16)
            .map(|i| if (i / 4 + i % 4) % 2 == 0 { 1 } else { -1 })
            .collect();
        sim.set_spins_internal(&checkerboard).expect("valid spins");
        assert_eq!(sim.spins, checkerboard);
        assert_eq!(sim.energy(), 2.0);
    }

    #[test]
    fn test_set_spins_internal_validation() {
        let mut sim = square_sim("metropolis");
        let spins_before = sim.spins.clone();

        let err = sim
            .set_spins_internal(&[1_i8; 15])
            .expect_err("wrong length must be rejected");
        assert!(err.to_string().contains("Expected 16 spins"), "{err}");

        for bad in [0_i8, 2, -2] {
            let mut data = vec![1_i8; 16];
            data[3] = bad;
            let err = sim
                .set_spins_internal(&data)
                .expect_err("non-±1 value must be rejected");
            assert!(err.to_string().contains("must be +1 or -1"), "{err}");
        }

        // Failed calls must not have touched the configuration.
        assert_eq!(sim.spins, spins_before);

        let mixed: Vec<i8> = (0..16).map(|i| if i < 8 { 1 } else { -1 }).collect();
        assert!(sim.set_spins_internal(&mixed).is_ok());
        assert_eq!(sim.spins, mixed);
    }

    #[test]
    // 1/2 and 1/1 are exact in floating point.
    #[allow(clippy::float_cmp)]
    fn test_beta_from_temperature_edges() {
        assert_eq!(beta_from_temperature(2.0).expect("valid T"), 0.5);
        assert_eq!(beta_from_temperature(1.0).expect("valid T"), 1.0);
        assert!(beta_from_temperature(f64::MIN_POSITIVE)
            .expect("tiny positive T is valid")
            .is_finite());

        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                beta_from_temperature(bad).is_err(),
                "T={bad} must be rejected"
            );
        }
    }
}
