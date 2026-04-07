pub mod chain;
pub mod square;
pub mod triangular;

use crate::error::MCIsingError;
use chain::ChainLattice;
use square::SquareLattice;
use triangular::TriangularLattice;

/// Trait defining the interface for any lattice geometry.
///
/// Lattices precompute neighbor tables at construction time for maximum
/// performance in the inner Monte Carlo loop. Neighbor indices use flat
/// (row-major) indexing into the spin array.
pub trait Lattice: Send + Sync {
    /// Total number of sites in the lattice.
    fn num_sites(&self) -> usize;

    /// Linear extent(s) of the lattice.
    fn shape(&self) -> &[usize];

    /// Number of nearest neighbors per site.
    fn coordination_number(&self) -> usize;

    /// Number of next-nearest neighbors per site.
    fn nnn_coordination_number(&self) -> usize;

    /// Nearest neighbor indices for a given site (precomputed).
    fn nearest_neighbors(&self, idx: usize) -> &[usize];

    /// Next-nearest neighbor indices for a given site (precomputed).
    fn next_nearest_neighbors(&self, idx: usize) -> &[usize];

    /// Squared distance between two sites respecting periodic boundary
    /// conditions. Returns distance in units of lattice spacing squared.
    fn distance_squared(&self, idx_a: usize, idx_b: usize) -> usize;

    /// Convert flat index to (row, col, ...) multi-index.
    fn flat_to_multi(&self, idx: usize) -> Vec<usize>;

    /// Convert (row, col, ...) multi-index to flat index.
    fn multi_to_flat(&self, indices: &[usize]) -> usize;

    /// Number of third-nearest neighbors per site.
    ///
    /// Returns 0 by default (no TNN defined). Override for lattices
    /// that support J3 coupling.
    fn tnn_coordination_number(&self) -> usize {
        0
    }

    /// Third-nearest neighbor indices for a given site (precomputed).
    ///
    /// Returns an empty slice by default. Override for lattices that
    /// support J3 coupling.
    fn third_nearest_neighbors(&self, _idx: usize) -> &[usize] {
        &[]
    }
}

/// Runtime lattice selection for dispatch at the PyO3 boundary.
///
/// Each variant owns its lattice. The `with_lattice!` macro dispatches
/// to generic code, preserving monomorphization on the hot path.
pub enum LatticeKind {
    Square(SquareLattice),
    Triangular(TriangularLattice),
    Chain(ChainLattice),
}

/// Dispatch macro: matches on `LatticeKind` and executes a generic body
/// with the concrete lattice type bound to `$lat`. Each arm is
/// monomorphized independently — no virtual dispatch in the hot path.
///
/// Usage: `with_lattice!(self.lattice, lat => { code using lat })`.
macro_rules! with_lattice {
    ($lattice:expr, $lat:ident => $body:expr) => {
        match $lattice {
            LatticeKind::Square($lat) => $body,
            LatticeKind::Triangular($lat) => $body,
            LatticeKind::Chain($lat) => $body,
        }
    };
}
pub(crate) use with_lattice;

impl LatticeKind {
    /// Parse lattice type from string (used at PyO3 boundary).
    pub fn from_str(s: &str, size: usize) -> Result<Self, MCIsingError> {
        match s {
            "square" => SquareLattice::new(size)
                .map(LatticeKind::Square)
                .ok_or(MCIsingError::InvalidLatticeSize(size)),
            "triangular" => TriangularLattice::new(size)
                .map(LatticeKind::Triangular)
                .ok_or(MCIsingError::InvalidLatticeSize(size)),
            "chain" => ChainLattice::new(size)
                .map(LatticeKind::Chain)
                .ok_or(MCIsingError::InvalidLatticeSize(size)),
            _ => Err(MCIsingError::InvalidLatticeType(s.to_string())),
        }
    }

    /// Delegate to the inner lattice's shape.
    pub fn shape(&self) -> &[usize] {
        with_lattice!(self, lat => lat.shape())
    }

    /// Delegate to the inner lattice's num_sites.
    pub fn num_sites(&self) -> usize {
        with_lattice!(self, lat => lat.num_sites())
    }

    /// Delegate to coordination_number.
    pub fn coordination_number(&self) -> usize {
        with_lattice!(self, lat => lat.coordination_number())
    }

    /// Delegate to nnn_coordination_number.
    pub fn nnn_coordination_number(&self) -> usize {
        with_lattice!(self, lat => lat.nnn_coordination_number())
    }

    /// Delegate to tnn_coordination_number.
    pub fn tnn_coordination_number(&self) -> usize {
        with_lattice!(self, lat => lat.tnn_coordination_number())
    }
}
