#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod algorithm;
pub mod error;
pub mod lattice;
pub mod observables;
pub mod rng;
pub mod simulation;

use pyo3::prelude::*;

/// The mcising Rust core module.
///
/// Provides high-performance Ising model simulation primitives compiled from Rust.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<simulation::IsingSimulation>()?;
    Ok(())
}