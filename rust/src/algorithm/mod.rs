pub mod metropolis;
pub mod swendsen_wang;
pub mod wolff;

use crate::error::MCIsingError;
use crate::lattice::Lattice;
use rand::Rng;

/// Result of a single Monte Carlo sweep.
///
/// For Metropolis: `accepted` = number of accepted spin flips,
/// `attempted` = total flip attempts (= num_sites).
///
/// For Wolff: `accepted` = cluster size (spins flipped),
/// `attempted` = total sites.
///
/// For Swendsen-Wang: `accepted` = total spins flipped across all clusters,
/// `attempted` = total sites.
#[derive(Debug, Clone, Copy)]
pub struct SweepResult {
    pub accepted: usize,
    pub attempted: usize,
}

impl SweepResult {
    /// Acceptance rate as a fraction in [0, 1].
    ///
    /// For cluster algorithms, this represents the fraction of spins flipped.
    pub fn acceptance_rate(&self) -> f64 {
        if self.attempted == 0 {
            return 0.0;
        }
        self.accepted as f64 / self.attempted as f64
    }
}

/// Trait defining the interface for Monte Carlo update algorithms.
///
/// Uses static dispatch via generics for maximum performance in the hot loop.
/// Each lattice+algorithm combination is monomorphized by the compiler.
///
/// Takes `&mut self` because cluster algorithms (Wolff, Swendsen-Wang) need
/// mutable scratch buffers for reuse across sweeps. Stateless algorithms
/// like Metropolis are unaffected.
pub trait McAlgorithm {
    /// Perform one full sweep of the lattice.
    ///
    /// A "sweep" means N single-spin-flip attempts for Metropolis,
    /// or one cluster construction for cluster algorithms.
    ///
    /// # Arguments
    /// * `spins` - mutable slice of spin values (+1 or -1 as i8)
    /// * `lattice` - the lattice geometry (provides neighbor information)
    /// * `j1` - nearest-neighbor coupling strength
    /// * `j2` - next-nearest-neighbor coupling strength
    /// * `h` - external magnetic field
    /// * `beta` - inverse temperature (1/T)
    /// * `rng` - random number generator
    fn sweep<L: Lattice, R: Rng>(
        &mut self,
        spins: &mut [i8],
        lattice: &L,
        j1: f64,
        j2: f64,
        j3: f64,
        h: f64,
        beta: f64,
        rng: &mut R,
    ) -> SweepResult;

    /// Human-readable name of the algorithm.
    fn name(&self) -> &'static str;
}

/// Runtime algorithm selection for dispatch at the PyO3 boundary.
///
/// Each variant holds the algorithm instance (with its scratch buffers).
/// Dispatch happens via match in `IsingSimulation` methods, preserving
/// monomorphization in the hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmKind {
    Metropolis,
    Wolff,
    SwendsenWang,
}

impl AlgorithmKind {
    /// Parse algorithm name from string (used at PyO3 boundary).
    pub fn from_str(s: &str) -> Result<Self, MCIsingError> {
        match s {
            "metropolis" => Ok(Self::Metropolis),
            "wolff" => Ok(Self::Wolff),
            "swendsen_wang" => Ok(Self::SwendsenWang),
            _ => Err(MCIsingError::InvalidAlgorithm(s.to_string())),
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Metropolis => "Metropolis",
            Self::Wolff => "Wolff",
            Self::SwendsenWang => "Swendsen-Wang",
        }
    }

    /// Whether this algorithm requires J2=0 and h=0.
    pub fn requires_no_frustration(&self) -> bool {
        matches!(self, Self::Wolff | Self::SwendsenWang)
    }
}
