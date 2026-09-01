#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::too_many_arguments)]

pub mod algorithm;
pub mod autocorrelation;
/// Synthetic-kernel recovery suite for the correlation length (test-only).
#[cfg(test)]
mod correlation_tests;
pub mod error;
/// Exact-enumeration oracle for small systems (test-only).
#[cfg(test)]
mod exact_enumeration;
pub mod lattice;
pub mod measurement;
pub mod observables;
pub mod parallel;
pub mod rng;
pub mod simulation;

#[cfg(not(test))]
use pyo3::prelude::*;

/// The mcising Rust core module.
///
/// Provides high-performance Ising model simulation primitives compiled from Rust.
///
/// Compiled out of test builds: the CPython symbols this module-init
/// references exist only inside a running interpreter, so keeping it in
/// the `cargo test` harness makes release links fail (fat LTO retains the
/// init as a root; debug builds merely dead-strip it at link time). The
/// pure-Rust tests exercise the core directly and never need the module.
#[cfg(not(test))]
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<simulation::IsingSimulation>()?;
    m.add_function(wrap_pyfunction!(parallel::run_independent_temperatures, m)?)?;
    m.add_function(wrap_pyfunction!(parallel::run_parallel_tempering, m)?)?;
    Ok(())
}
