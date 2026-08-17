use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use std::fmt;

/// Error types for the mcising simulation core.
#[derive(Debug)]
pub enum MCIsingError {
    InvalidLatticeSize(usize),
    OddLatticeSize(&'static str, usize),
    InvalidTemperature(f64),
    InvalidCoupling(&'static str, f64),
    InvalidSpinConfiguration(String),
    InvalidAlgorithm(String),
    ClusterAlgorithmConstraint(String),
    ClusterCouplingSign(String),
    InvalidLatticeType(String),
    EmptyTemperatureList,
    InvalidInterval(&'static str, usize),
    IncompatibleSwapCadence(usize, usize),
    InvalidSeedOffsets(usize, usize),
}

impl fmt::Display for MCIsingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLatticeSize(size) => {
                write!(f, "Lattice size must be >= 2, got {size}")
            }
            Self::OddLatticeSize(lattice, size) => {
                write!(
                    f,
                    "The {lattice} lattice requires even size L under periodic \
                     boundary conditions (odd L breaks neighbor-table symmetry \
                     across the wrap seam; odd-L support is future work), \
                     got {size}"
                )
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
            Self::InvalidAlgorithm(name) => {
                write!(
                    f,
                    "Unknown algorithm '{name}'. Valid options: metropolis, wolff, swendsen_wang"
                )
            }
            Self::ClusterAlgorithmConstraint(alg) => {
                write!(
                    f,
                    "Cluster algorithm '{alg}' requires J2=0 and h=0. \
                     Use algorithm='metropolis' for J1-J2 or external field simulations."
                )
            }
            Self::ClusterCouplingSign(alg) => {
                write!(
                    f,
                    "Cluster algorithm '{alg}' requires J1>0; use \
                     algorithm='metropolis' for antiferromagnetic couplings; \
                     sublattice mapping is future work."
                )
            }
            Self::InvalidLatticeType(name) => {
                write!(
                    f,
                    "Unknown lattice type '{name}'. Valid options: square, triangular, chain, honeycomb, cubic"
                )
            }
            Self::EmptyTemperatureList => {
                write!(f, "At least one temperature is required, got an empty list")
            }
            Self::InvalidInterval(name, value) => {
                write!(f, "{name} must be >= 1, got {value}")
            }
            Self::IncompatibleSwapCadence(measurement_interval, swap_interval) => {
                write!(
                    f,
                    "Parallel tempering requires measurement_interval to be a \
                     multiple of swap_interval: the ladder advances in \
                     swap_interval-sized chunks and measures only on chunk \
                     boundaries, so a non-dividing interval silently drops \
                     measurements. Raise measurement_interval to the next \
                     multiple of {swap_interval}, or choose a swap_interval \
                     that divides it. Got \
                     measurement_interval={measurement_interval}, \
                     swap_interval={swap_interval}"
                )
            }
            Self::InvalidSeedOffsets(n_offsets, n_temps) => {
                write!(
                    f,
                    "seed_offsets must have one entry per temperature, \
                     got {n_offsets} offsets for {n_temps} temperatures"
                )
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
