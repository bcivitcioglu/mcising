use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

use crate::algorithm::metropolis::Metropolis;
use crate::algorithm::McAlgorithm;
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
    h: f64,
    rng: Xoshiro256StarStar,
    lattice_size: usize,
}

#[pymethods]
impl IsingSimulation {
    /// Create a new Ising simulation on a square lattice.
    ///
    /// # Arguments
    /// * `lattice_size` - Linear size L of the L x L lattice (must be >= 2)
    /// * `j1` - Nearest-neighbor coupling strength
    /// * `j2` - Next-nearest-neighbor coupling strength
    /// * `h` - External magnetic field
    /// * `seed` - Random seed for reproducibility
    #[new]
    fn new(lattice_size: usize, j1: f64, j2: f64, h: f64, seed: u64) -> PyResult<Self> {
        let lattice = SquareLattice::new(lattice_size)
            .ok_or(MCIsingError::InvalidLatticeSize(lattice_size))?;

        if !j1.is_finite() {
            return Err(MCIsingError::InvalidCoupling("j1", j1).into());
        }
        if !j2.is_finite() {
            return Err(MCIsingError::InvalidCoupling("j2", j2).into());
        }
        if !h.is_finite() {
            return Err(MCIsingError::InvalidCoupling("h", h).into());
        }

        let mut rng = create_rng(seed);
        let spins: Vec<i8> = (0..lattice.num_sites())
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();

        Ok(Self {
            spins,
            lattice,
            j1,
            j2,
            h,
            rng,
            lattice_size,
        })
    }

    /// Perform Metropolis sweeps at the given inverse temperature.
    ///
    /// Returns (accepted, attempted) as a tuple.
    fn metropolis_sweep(&mut self, n_sweeps: usize, beta: f64) -> PyResult<(usize, usize)> {
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
            let result = Metropolis.sweep(
                &mut self.spins,
                &self.lattice,
                self.j1,
                self.j2,
                self.h,
                beta,
                &mut self.rng,
            );
            total_accepted += result.accepted;
            total_attempted += result.attempted;
        }

        Ok((total_accepted, total_attempted))
    }

    /// Compute the total energy per site.
    fn energy(&self) -> f64 {
        observables::energy_per_site(&self.spins, &self.lattice, self.j1, self.j2, self.h)
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

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "IsingSimulation(lattice_size={}, j1={}, j2={}, h={}, energy={:.4}, mag={:.4})",
            self.lattice_size,
            self.j1,
            self.j2,
            self.h,
            self.energy(),
            self.magnetization()
        )
    }
}
