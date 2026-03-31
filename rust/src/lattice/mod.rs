pub mod square;

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
}