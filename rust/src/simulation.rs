use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyUntypedArrayMethods,
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
use crate::lattice::square::SquareLattice;
use crate::lattice::Lattice;
use crate::observables;
use crate::rng::create_rng;

/// Core Ising model simulation engine.
///
/// This is the PyO3 boundary class that owns the Rust lattice, spins, and RNG.
/// All physics computation happens in Rust; Python calls methods on this class.
#[pyclass]
pub struct IsingSimulation {
    spins: Vec<i8>,
    lattice: SquareLattice,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
    rng: Xoshiro256StarStar,
    lattice_size: usize,
    algorithm: AlgorithmKind,
    metropolis: Metropolis,
    wolff: Option<Wolff>,
    swendsen_wang: Option<SwendsenWang>,
}

#[pymethods]
impl IsingSimulation {
    /// Create a new Ising simulation on a square lattice.
    ///
    /// # Arguments
    /// * `lattice_size` - Linear size L of the L x L lattice (must be >= 2)
    /// * `j1` - Nearest-neighbor coupling strength
    /// * `j2` - Next-nearest-neighbor coupling strength
    /// * `j3` - Third-nearest-neighbor coupling strength
    /// * `h` - External magnetic field
    /// * `seed` - Random seed for reproducibility
    /// * `algorithm` - Algorithm name: "metropolis", "wolff", or "swendsen_wang"
    #[new]
    #[pyo3(signature = (lattice_size, j1, j2, j3, h, seed, algorithm = "metropolis"))]
    fn new(
        lattice_size: usize,
        j1: f64,
        j2: f64,
        j3: f64,
        h: f64,
        seed: u64,
        algorithm: &str,
    ) -> PyResult<Self> {
        let lattice = SquareLattice::new(lattice_size)
            .ok_or(MCIsingError::InvalidLatticeSize(lattice_size))?;

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

        let num_sites = lattice.num_sites();
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
            algorithm: algo_kind,
            metropolis: Metropolis::new(j1, j2, j3, h),
            wolff,
            swendsen_wang,
        })
    }

    /// Perform MC sweeps at the given inverse temperature using the
    /// configured algorithm.
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
    ///
    /// Returns (avg_energy, avg_magnetization, accepted, attempted).
    /// Energy and magnetization are per-site, averaged over all sweeps.
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

            energy_sum +=
                observables::energy_per_site(&self.spins, &self.lattice, self.j1, self.j2, self.j3, self.h);
            mag_sum += observables::magnetization_per_site(&self.spins);
        }

        let n = n_sweeps as f64;
        Ok((energy_sum / n, mag_sum / n, total_accepted, total_attempted))
    }

    /// Get the algorithm name.
    #[getter]
    fn algorithm_name(&self) -> &str {
        self.algorithm.name()
    }

    /// Compute the total energy per site.
    fn energy(&self) -> f64 {
        observables::energy_per_site(&self.spins, &self.lattice, self.j1, self.j2, self.j3, self.h)
    }

    /// Compute the magnetization per site.
    fn magnetization(&self) -> f64 {
        observables::magnetization_per_site(&self.spins)
    }

    /// Return the spin configuration as a 2D NumPy array of shape (L, L).
    fn get_spins<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i8>> {
        let shape = [self.lattice_size, self.lattice_size];
        let flat = self.spins.clone();
        numpy::PyArray1::from_vec(py, flat)
            .reshape(shape)
            .expect("reshape should not fail for correct dimensions")
    }

    /// Set the spin configuration from a 2D NumPy array.
    fn set_spins(&mut self, spins: PyReadonlyArray2<'_, i8>) -> PyResult<()> {
        let shape = spins.shape();
        if shape[0] != self.lattice_size || shape[1] != self.lattice_size {
            return Err(MCIsingError::InvalidSpinConfiguration(format!(
                "Expected shape ({}, {}), got ({}, {})",
                self.lattice_size, self.lattice_size, shape[0], shape[1]
            ))
            .into());
        }

        let data = spins.as_slice().map_err(|e| {
            MCIsingError::InvalidSpinConfiguration(format!("Cannot read array: {e}"))
        })?;

        // Validate all values are +1 or -1
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

    /// Flip the spin at position (row, col).
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

        let spin = f64::from(self.spins[idx]);
        let mut local_field: f64 = 0.0;

        for &nbr in self.lattice.nearest_neighbors(idx) {
            local_field += self.j1 * f64::from(self.spins[nbr]);
        }
        for &nbr in self.lattice.next_nearest_neighbors(idx) {
            local_field += self.j2 * f64::from(self.spins[nbr]);
        }
        for &nbr in self.lattice.third_nearest_neighbors(idx) {
            local_field += self.j3 * f64::from(self.spins[nbr]);
        }

        let interaction = -spin * local_field;
        let field = -self.h * spin;

        Ok(interaction + field)
    }

    /// Compute the correlation function.
    ///
    /// Returns a tuple of (distances, correlations) as 1D NumPy arrays.
    fn correlation_function<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let (distances, correlations) =
            observables::correlation_function(&self.spins, &self.lattice);
        (distances.into_pyarray(py), correlations.into_pyarray(py))
    }

    /// Compute the correlation length from the current spin configuration.
    fn correlation_length(&self) -> f64 {
        let (distances, correlations) =
            observables::correlation_function(&self.spins, &self.lattice);
        observables::correlation_length(&correlations, &distances)
    }

    /// Get the lattice size.
    #[getter]
    fn lattice_size(&self) -> usize {
        self.lattice_size
    }

    /// Get the number of sites.
    #[getter]
    fn num_sites(&self) -> usize {
        self.lattice.num_sites()
    }

    /// Get J1 coupling.
    #[getter]
    fn j1(&self) -> f64 {
        self.j1
    }

    /// Get J2 coupling.
    #[getter]
    fn j2(&self) -> f64 {
        self.j2
    }

    /// Get J3 coupling.
    #[getter]
    fn j3(&self) -> f64 {
        self.j3
    }

    /// Get external field h.
    #[getter]
    fn h(&self) -> f64 {
        self.h
    }

    /// Get the RNG internal state as bytes for checkpointing.
    fn get_rng_state(&self) -> Vec<u8> {
        serde_json::to_vec(&self.rng).expect("Xoshiro256StarStar serialization should not fail")
    }

    /// Restore the RNG internal state from bytes previously obtained
    /// via `get_rng_state`.
    fn set_rng_state(&mut self, state: Vec<u8>) -> PyResult<()> {
        let rng: Xoshiro256StarStar = serde_json::from_slice(&state).map_err(|e| {
            MCIsingError::InvalidSpinConfiguration(format!("Invalid RNG state: {e}"))
        })?;
        self.rng = rng;
        Ok(())
    }

    /// Run thermalization sweeps following a temperature schedule, recording
    /// energy after each sweep.
    ///
    /// This replaces the Python-side loop that called `metropolis_sweep(1, beta)`
    /// repeatedly, avoiding N Python-Rust boundary crossings.
    ///
    /// # Arguments
    /// * `temp_schedule` - List of temperatures to sweep through (one sweep per temperature)
    ///
    /// # Returns
    /// 1D NumPy array of energy-per-site values, one per sweep.
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
            energies.push(observables::energy_per_site(
                &self.spins,
                &self.lattice,
                self.j1,
                self.j2,
                self.j3,
                self.h,
            ));
        }

        Ok(energies.into_pyarray(py))
    }

    /// Run additional thermalization sweeps at a fixed temperature,
    /// recording energy after each sweep.
    ///
    /// Used when MSER detects the initial cool-down was insufficient.
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
            energies.push(observables::energy_per_site(
                &self.spins,
                &self.lattice,
                self.j1,
                self.j2,
                self.j3,
                self.h,
            ));
        }

        Ok(energies.into_pyarray(py))
    }

    /// Analyze a thermalization energy series for equilibration and autocorrelation.
    ///
    /// Returns a dict with keys: truncation_point, is_thermalized, tau_int,
    /// window, recommended_interval.
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
    /// Performs n_measurements * interval total sweeps, recording energy,
    /// magnetization, and optionally spin configurations every `interval` sweeps.
    ///
    /// # Returns
    /// Tuple of (energies, magnetizations, configs_or_none).
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
        Option<Bound<'py, PyArray3<i8>>>,
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
            // Run `interval` sweeps between measurements
            for _ in 0..interval {
                self.dispatch_sweep(beta);
            }

            energies.push(observables::energy_per_site(
                &self.spins,
                &self.lattice,
                self.j1,
                self.j2,
                self.j3,
                self.h,
            ));
            magnetizations.push(observables::magnetization_per_site(&self.spins));

            if let Some(ref mut c) = configs {
                c.extend_from_slice(&self.spins);
            }
        }

        let py_energies = energies.into_pyarray(py);
        let py_mags = magnetizations.into_pyarray(py);
        let py_configs = configs.map(|c| {
            let flat = numpy::PyArray1::from_vec(py, c);
            flat.reshape([n_measurements, self.lattice_size, self.lattice_size])
                .expect("reshape should not fail for correct dimensions")
        });

        Ok((py_energies, py_mags, py_configs))
    }

    /// String representation.
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
    /// Dispatch a single sweep to the configured algorithm.
    ///
    /// Monomorphized per-algorithm via match arms — each arm calls the
    /// concrete algorithm type directly, so the hot path is fully specialized.
    fn dispatch_sweep(&mut self, beta: f64) -> SweepResult {
        match self.algorithm {
            AlgorithmKind::Metropolis => self.metropolis.sweep(
                &mut self.spins,
                &self.lattice,
                self.j1,
                self.j2,
                self.j3,
                self.h,
                beta,
                &mut self.rng,
            ),
            AlgorithmKind::Wolff => self
                .wolff
                .as_mut()
                .expect("Wolff algorithm not initialized")
                .sweep(
                    &mut self.spins,
                    &self.lattice,
                    self.j1,
                    self.j2,
                    self.j3,
                    self.h,
                    beta,
                    &mut self.rng,
                ),
            AlgorithmKind::SwendsenWang => self
                .swendsen_wang
                .as_mut()
                .expect("Swendsen-Wang algorithm not initialized")
                .sweep(
                    &mut self.spins,
                    &self.lattice,
                    self.j1,
                    self.j2,
                    self.j3,
                    self.h,
                    beta,
                    &mut self.rng,
                ),
        }
    }
}
