use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use std::fmt;

/// Error types for the mcising simulation core.
#[derive(Debug)]
pub enum MCIsingError {
    InvalidLatticeSize(usize),
    InvalidTemperature(f64),
    InvalidCoupling(&'static str, f64),
    InvalidSpinConfiguration(String),
}

impl fmt::Display for MCIsingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLatticeSize(size) => {
                write!(f, "Lattice size must be >= 2, got {size}")
            }
            Self::InvalidTemperature(temp) => {
                write!(f, "Temperature must be positive and finite, got {temp}")
            }
            Self::InvalidCoupling(name, value) => {
                write!(f, "Coupling {name} must be finite, got {value}")
            }
            Self::InvalidSpinConfiguration(msg) => {
                write!(f, "Invalid spin configuration: {msg}")
            }
        }
    }
}

impl std::error::Error for MCIsingError {}

impl From<MCIsingError> for PyErr {
    fn from(err: MCIsingError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}
