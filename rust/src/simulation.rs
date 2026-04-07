use numpy::{
    IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

use crate::algorithm::metropolis::Metropolis;
use crate::algorithm::swendsen_wang::SwendsenWang;
use crate::algorithm::wolff::Wolff;
use crate::algorithm::{AlgorithmKind, McAlgorithm, SweepResult};
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
    spins: Vec<i8>,
    lattice: LatticeKind,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
    rng: Xoshiro256StarStar,
    lattice_size: usize,
    /// Shape for reshaping spin arrays (e.g. [L, L] for 2D lattices).
    shape: Vec<usize>,
    algorithm: AlgorithmKind,
    metropolis: Metropolis,
    wolff: Option<Wolff>,
    swendsen_wang: Option<SwendsenWang>,
}

#[pymethods]
impl IsingSimulation {
    /// Create a new Ising simulation.
    ///
    /// # Arguments
    /// * `lattice_size` - Linear size L of the lattice (must be >= 2)
    /// * `j1` - Nearest-neighbor coupling strength
    /// * `j2` - Next-nearest-neighbor coupling strength
    /// * `j3` - Third-nearest-neighbor coupling strength
    /// * `h` - External magnetic field
    /// * `seed` - Random seed for reproducibility
    /// * `algorithm` - Algorithm name: "metropolis", "wolff", or "swendsen_wang"
    /// * `lattice_type` - Lattice geometry: "square" or "triangular"
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
        let lattice = LatticeKind::from_str(lattice_type, lattice_size)?;

        if !j1.is_finite() {
            return Err(MCIsingError::InvalidCoupling("j1", j1).into());
        }
        if !j2.is_finite() {
            return Err(MCIsingError::InvalidCoupling("j2", j2).into());
        }
        if !j3.is_finite() {
            return Err(MCIsingError::InvalidCoupling("j3", j3).into());
        }
        if !h.is_finite() {
            return Err(MCIsingError::InvalidCoupling("h", h).into());
        }

        let algo_kind = AlgorithmKind::from_str(algorithm)?;

        // Cluster algorithms require J2=0, J3=0, and h=0
        if algo_kind.requires_no_frustration() && (j2 != 0.0 || j3 != 0.0 || h != 0.0) {
            return Err(MCIsingError::ClusterAlgorithmConstraint(
                algo_kind.name().to_string(),
            )
            .into());
        }

        // Validate coupling vs lattice support
        if j2 != 0.0 && lattice.nnn_coordination_number() == 0 {
            return Err(MCIsingError::InvalidCoupling(
                "j2 (no NNN defined for this lattice)", j2,
            ).into());
        }
        if j3 != 0.0 && lattice.tnn_coordination_number() == 0 {
            return Err(MCIsingError::InvalidCoupling(
                "j3 (no TNN defined for this lattice)", j3,
            ).into());
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

        let wolff = if algo_kind == AlgorithmKind::Wolff {
            Some(Wolff::new(num_sites))
        } else {
            None
        };
        let swendsen_wang = if algo_kind == AlgorithmKind::SwendsenWang {
            Some(SwendsenWang::new(num_sites))
        } else {
            None
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
            algorithm: algo_kind,
            metropolis: Metropolis::new(j1, j2, j3, h, z_nn, z_nnn, z_tnn),
            wolff,
            swendsen_wang,
        })
    }

    /// Perform MC sweeps at the given inverse temperature.
    ///
    /// Returns (accepted, attempted) as a tuple.
    fn sweep(&mut self, n_sweeps: usize, beta: f64) -> PyResult<(usize, usize)> {
        if !beta.is_finite() || beta < 0.0 {
            return Err(MCIsingError::InvalidTemperature(if beta == 0.0 {
                0.0
            } else {
                1.0 / beta
            })
            .into());
        }

        let mut total_accepted = 0;
        let mut total_attempted = 0;

        for _ in 0..n_sweeps {
            let result = self.dispatch_sweep(beta);
            total_accepted += result.accepted;
            total_attempted += result.attempted;
        }

        Ok((total_accepted, total_attempted))
    }

    /// Perform MC sweeps while accumulating observable averages.
    fn sweep_measured(
        &mut self,
        n_sweeps: usize,
        beta: f64,
    ) -> PyResult<(f64, f64, usize, usize)> {
        if !beta.is_finite() || beta < 0.0 {
            return Err(MCIsingError::InvalidTemperature(if beta == 0.0 {
                0.0
            } else {
                1.0 / beta
            })
            .into());
        }

        let mut total_accepted = 0;
        let mut total_attempted = 0;
        let mut energy_sum = 0.0;
        let mut mag_sum = 0.0;

        for _ in 0..n_sweeps {
            let result = self.dispatch_sweep(beta);
            total_accepted += result.accepted;
            total_attempted += result.attempted;
            energy_sum += self.compute_energy();
            mag_sum += observables::magnetization_per_site(&self.spins);
        }

        let n = n_sweeps as f64;
        Ok((energy_sum / n, mag_sum / n, total_accepted, total_attempted))
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
    fn get_spins<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let flat = numpy::PyArray1::from_vec(py, self.spins.clone());
        let shape: Vec<usize> = self.shape.clone();
        let reshaped = flat.reshape(shape).map_err(|e| {
            MCIsingError::InvalidSpinConfiguration(format!("reshape failed: {e}"))
        })?;
        Ok(reshaped.into_py(py))
    }

    /// Set the spin configuration from a NumPy array.
    fn set_spins(&mut self, spins: PyReadonlyArrayDyn<'_, i8>) -> PyResult<()> {
        let total: usize = spins.shape().iter().product();
        let expected = self.lattice.num_sites();
        if total != expected {
            return Err(MCIsingError::InvalidSpinConfiguration(format!(
                "Expected {expected} spins, got {total}"
            ))
            .into());
        }

        let data = spins.as_slice().map_err(|e| {
            MCIsingError::InvalidSpinConfiguration(format!("Cannot read array: {e}"))
        })?;

        for &val in data {
            if val != 1 && val != -1 {
                return Err(MCIsingError::InvalidSpinConfiguration(format!(
                    "All spins must be +1 or -1, found {val}"
                ))
                .into());
            }
        }

        self.spins.clear();
        self.spins.extend_from_slice(data);
        Ok(())
    }

    /// Flip the spin at flat index.
    fn flip_spin(&mut self, row: usize, col: usize) -> PyResult<()> {
        let idx = row * self.lattice_size + col;
        if idx >= self.spins.len() {
            return Err(MCIsingError::InvalidSpinConfiguration(format!(
                "Position ({row}, {col}) out of bounds for lattice size {size}",
                size = self.lattice_size
            ))
            .into());
        }
        self.spins[idx] = -self.spins[idx];
        Ok(())
    }

    /// Compute the energy of a single spin at (row, col).
    fn spin_energy(&self, row: usize, col: usize) -> PyResult<f64> {
        let idx = row * self.lattice_size + col;
        if idx >= self.spins.len() {
            return Err(MCIsingError::InvalidSpinConfiguration(format!(
                "Position ({row}, {col}) out of bounds for lattice size {size}",
                size = self.lattice_size
            ))
            .into());
        }

        with_lattice!(&self.lattice, lat => {
            let spin = f64::from(self.spins[idx]);
            let mut local_field: f64 = 0.0;
            for &nbr in lat.nearest_neighbors(idx) {
                local_field += self.j1 * f64::from(self.spins[nbr]);
            }
            for &nbr in lat.next_nearest_neighbors(idx) {
                local_field += self.j2 * f64::from(self.spins[nbr]);
            }
            for &nbr in lat.third_nearest_neighbors(idx) {
                local_field += self.j3 * f64::from(self.spins[nbr]);
            }
            Ok(-spin * local_field - self.h * spin)
        })
    }

    /// Compute the correlation function.
    fn correlation_function<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        with_lattice!(&self.lattice, lat => {
            let (distances, correlations) = observables::correlation_function(&self.spins, lat);
            (distances.into_pyarray(py), correlations.into_pyarray(py))
        })
    }

    /// Compute the correlation length from the current spin configuration.
    fn correlation_length(&self) -> f64 {
        with_lattice!(&self.lattice, lat => {
            let (distances, correlations) = observables::correlation_function(&self.spins, lat);
            observables::correlation_length(&correlations, &distances)
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
        serde_json::to_vec(&self.rng).expect("Xoshiro256StarStar serialization should not fail")
    }

    fn set_rng_state(&mut self, state: Vec<u8>) -> PyResult<()> {
        let rng: Xoshiro256StarStar = serde_json::from_slice(&state).map_err(|e| {
            MCIsingError::InvalidSpinConfiguration(format!("Invalid RNG state: {e}"))
        })?;
        self.rng = rng;
        Ok(())
    }

    fn thermalize_with_diagnostics<'py>(
        &mut self,
        py: Python<'py>,
        temp_schedule: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut energies = Vec::with_capacity(temp_schedule.len());
        for temp in &temp_schedule {
            if *temp <= 0.0 {
                continue;
            }
            let beta = 1.0 / temp;
            self.dispatch_sweep(beta);
            energies.push(self.compute_energy());
        }
        Ok(energies.into_pyarray(py))
    }

    fn extend_thermalization<'py>(
        &mut self,
        py: Python<'py>,
        n_sweeps: usize,
        beta: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(MCIsingError::InvalidTemperature(if beta == 0.0 {
                0.0
            } else {
                1.0 / beta
            })
            .into());
        }
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
    fn production_sweeps<'py>(
        &mut self,
        py: Python<'py>,
        n_measurements: usize,
        interval: usize,
        beta: f64,
        store_configs: bool,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Option<PyObject>,
    )> {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(MCIsingError::InvalidTemperature(if beta == 0.0 {
                0.0
            } else {
                1.0 / beta
            })
            .into());
        }

        let mut energies = Vec::with_capacity(n_measurements);
        let mut magnetizations = Vec::with_capacity(n_measurements);
        let mut configs: Option<Vec<i8>> = if store_configs {
            Some(Vec::with_capacity(n_measurements * self.spins.len()))
        } else {
            None
        };

        for _ in 0..n_measurements {
            for _ in 0..interval {
                self.dispatch_sweep(beta);
            }
            energies.push(self.compute_energy());
            magnetizations.push(observables::magnetization_per_site(&self.spins));
            if let Some(ref mut c) = configs {
                c.extend_from_slice(&self.spins);
            }
        }

        let py_energies = energies.into_pyarray(py);
        let py_mags = magnetizations.into_pyarray(py);
        let py_configs = configs.map(|c| {
            let flat = numpy::PyArray1::from_vec(py, c);
            let mut reshape_dims: Vec<usize> = vec![n_measurements];
            reshape_dims.extend_from_slice(&self.shape);
            flat.reshape(reshape_dims)
                .expect("reshape should not fail for correct dimensions")
                .into_py(py)
        });

        Ok((py_energies, py_mags, py_configs))
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
        match self.algorithm {
            AlgorithmKind::Metropolis => {
                with_lattice!(&self.lattice, lat => {
                    self.metropolis.sweep(
                        &mut self.spins, lat,
                        self.j1, self.j2, self.j3, self.h, beta, &mut self.rng,
                    )
                })
            }
            AlgorithmKind::Wolff => {
                with_lattice!(&self.lattice, lat => {
                    self.wolff.as_mut()
                        .expect("Wolff algorithm not initialized")
                        .sweep(
                            &mut self.spins, lat,
                            self.j1, self.j2, self.j3, self.h, beta, &mut self.rng,
                        )
                })
            }
            AlgorithmKind::SwendsenWang => {
                with_lattice!(&self.lattice, lat => {
                    self.swendsen_wang.as_mut()
                        .expect("Swendsen-Wang algorithm not initialized")
                        .sweep(
                            &mut self.spins, lat,
                            self.j1, self.j2, self.j3, self.h, beta, &mut self.rng,
                        )
                })
            }
        }
    }
}
