pub mod metropolis;

use crate::lattice::Lattice;
use rand::Rng;

/// Result of a single Monte Carlo sweep.
#[derive(Debug, Clone, Copy)]
pub struct SweepResult {
    pub accepted: usize,
    pub attempted: usize,
}

impl SweepResult {
    /// Acceptance rate as a fraction in [0, 1].
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
        &self,
        spins: &mut [i8],
        lattice: &L,
        j1: f64,
        j2: f64,
        h: f64,
        beta: f64,
        rng: &mut R,
    ) -> SweepResult;

    /// Human-readable name of the algorithm.
    fn name(&self) -> &'static str;
}