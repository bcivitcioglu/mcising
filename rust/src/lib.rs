#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::too_many_arguments)]

pub mod algorithm;
pub mod autocorrelation;
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
