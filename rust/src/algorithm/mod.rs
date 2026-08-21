pub mod metropolis;
pub mod swendsen_wang;
pub mod wolff;

use crate::error::MCIsingError;
use crate::lattice::Lattice;
use rand::Rng;

/// Result of a single Monte Carlo sweep, with honest work accounting.
///
/// For Metropolis: `accepted` = number of accepted spin flips,
/// `attempted` = total flip attempts (= num_sites), `cluster_flips` = 0.
///
/// For Wolff: one sweep = ONE cluster; `accepted` = cluster size,
/// `attempted` = cluster size (rejection-free — no fictitious
/// num_sites denominator), `cluster_flips` = 1.
///
/// For Swendsen-Wang: `accepted` = total spins flipped across all clusters,
/// `attempted` = total sites (every site receives a keep/flip decision),
/// `cluster_flips` = clusters whose independent p=1/2 decision came up
/// "flip".
#[derive(Debug, Clone, Copy)]
pub struct SweepResult {
    pub accepted: usize,
    pub attempted: usize,
    pub cluster_flips: usize,
}

impl SweepResult {
    /// Acceptance rate as `accepted / attempted`, always in [0, 1].
    ///
    /// For Metropolis this is the classic acceptance fraction. For
    /// Swendsen-Wang it is the flipped-spin fraction of the lattice.
    /// Wolff is rejection-free, so its rate is identically 1.
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
    /// A "sweep" is `num_sites` attempted-flip equivalents for
    /// Metropolis (N single-spin-flip attempts) and Swendsen-Wang (one
    /// full bond-percolation partition + flip pass), but ONE cluster
    /// construction for Wolff — a per-sweep flip budget was rejected in
    /// P10 because measuring at its state-dependent stopping time is
    /// size-biased (see `wolff.rs`). Callers scale Wolff `n_sweeps`
    /// accordingly; `SweepResult` reports the honest work done.
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

/// Algorithm selection parsed at the PyO3 boundary.
///
/// Pure parse/validation type; the algorithm instance itself lives in
/// [`AlgorithmState`], so a kind can never disagree with its state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmKind {
    Metropolis,
    Wolff,
    SwendsenWang,
}

/// Runtime algorithm state for dispatch in `IsingSimulation`.
///
/// Each variant holds the algorithm instance (with its scratch buffers), so
/// a simulation with a mismatched kind/state pair is unrepresentable — the
/// illegal state that previously required `Option` + `.expect()`. Dispatch
/// happens via match, preserving monomorphization in the hot path.
pub enum AlgorithmState {
    /// Boxed: the lookup-table struct dwarfs the other variants
    /// (`clippy::large_enum_variant`); the deref is once per sweep *call*
    /// (the N-site loop is inside), so it is not on the hot path.
    Metropolis(Box<metropolis::Metropolis>),
    Wolff(wolff::Wolff),
    SwendsenWang(swendsen_wang::SwendsenWang),
}

impl AlgorithmState {
    /// Human-readable name of the held algorithm.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Metropolis(m) => m.name(),
            Self::Wolff(w) => w.name(),
            Self::SwendsenWang(sw) => sw.name(),
        }
    }
}

impl AlgorithmKind {
    /// Parse algorithm name from string (used at PyO3 boundary).
    ///
    /// # Errors
    ///
    /// Returns `MCIsingError::InvalidAlgorithm` for unrecognized names.
    // Inherent `from_str` is the frozen pre-1.0 surface; adopting the
    // `FromStr` trait belongs to the P10/P11 API phases.
    #[allow(clippy::should_implement_trait)]
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

    /// Whether this algorithm's Fortuin-Kasteleyn bond probability
    /// `p_add = 1 - exp(-2*beta*J1)` requires a ferromagnetic J1.
    ///
    /// For J1 <= 0, `p_add <= 0` and cluster growth never adds a site:
    /// the update degenerates to single random spin flips with no
    /// acceptance test — a silently wrong sampler (B1).
    pub fn requires_ferromagnetic_j1(&self) -> bool {
        matches!(self, Self::Wolff | Self::SwendsenWang)
    }
}
