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
    NonDyadicCouplingsNeedBinWidth,
    InvalidBinWidth(f64),
    TooManyEnergyBins(usize, usize),
    DegenerateEnergySpectrum,
    InvalidEnergyWindow(f64, f64),
    EnergyWindowUnreachable {
        e_lo: f64,
        e_hi: f64,
        reached: f64,
        sweeps: u64,
    },
    InvalidFlatness(f64),
    InvalidLogF(&'static str, f64),
    InvalidDriveBeta(f64),
    InvalidInitialLogG(String),
}

impl fmt::Display for MCIsingError {
    // One flat table of messages: splitting it by theme would hide the
    // one place a reviewer reads every user-facing error.
    #[allow(clippy::too_many_lines)]
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
            Self::NonDyadicCouplingsNeedBinWidth => {
                write!(
                    f,
                    "The couplings are not exactly representable in binary at a \
                     usable resolution, so the exact energy grid cannot be \
                     built; pass bin_width (in total-energy units) to bin the \
                     density of states explicitly"
                )
            }
            Self::InvalidBinWidth(width) => {
                write!(f, "bin_width must be positive and finite, got {width}")
            }
            Self::TooManyEnergyBins(requested, max) => {
                write!(
                    f,
                    "The energy grid would need {requested} bins, more than the \
                     {max} supported; pass a coarser bin_width or an energy_window"
                )
            }
            Self::DegenerateEnergySpectrum => {
                write!(
                    f,
                    "Every spin flip is energy-neutral (all couplings are zero), \
                     so there is no density of states to sample"
                )
            }
            Self::InvalidEnergyWindow(lo, hi) => {
                write!(
                    f,
                    "energy_window must be a finite (lo, hi) pair per site with \
                     lo < hi that contains at least one energy bin, got ({lo}, {hi})"
                )
            }
            Self::EnergyWindowUnreachable {
                e_lo,
                e_hi,
                reached,
                sweeps,
            } => {
                write!(
                    f,
                    "Could not drive the configuration into the energy window \
                     ({e_lo}, {e_hi}) per site within {sweeps} sweeps (reached \
                     {reached}); widen the window, raise drive_max_sweeps, or \
                     check that the window is physically reachable"
                )
            }
            Self::InvalidFlatness(value) => {
                write!(f, "flatness must be in (0, 1], got {value}")
            }
            Self::InvalidLogF(name, value) => {
                write!(
                    f,
                    "{name} must be positive and finite with \
                     log_f_final < log_f_initial, got {value}"
                )
            }
            Self::InvalidDriveBeta(value) => {
                write!(f, "drive_beta must be positive and finite, got {value}")
            }
            Self::InvalidInitialLogG(msg) => {
                write!(f, "Invalid initial_log_g: {msg}")
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
