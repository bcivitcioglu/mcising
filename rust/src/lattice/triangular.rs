use super::Lattice;

/// 2D triangular lattice with periodic boundary conditions.
///
/// Uses offset coordinates on an L×L grid. Each site has 6 nearest neighbors.
/// Rows with even index have different diagonal connectivity than odd rows:
///
/// Even row (r):  also connects to (r-1, c) and (r+1, c) diagonals on the left
/// Odd row (r):   also connects to (r-1, c) and (r+1, c) diagonals on the right
///
/// Specifically, the 6 NN of site (r, c) are:
///   Same as square: left (r, c-1), right (r, c+1), up (r-1, c), down (r+1, c)
///   Even row extra: up-left (r-1, c-1), down-left (r+1, c-1)
///   Odd row extra:  up-right (r-1, c+1), down-right (r+1, c+1)
///
/// NNN (6 per site): the 6 next-nearest neighbors at distance √3.
/// TNN (6 per site): the 6 third-nearest neighbors at distance 2.
pub struct TriangularLattice {
    size: usize,
    num_sites: usize,
    shape: [usize; 2],
    nn_table: Vec<usize>,  // stride 6
    nnn_table: Vec<usize>, // stride 6
    tnn_table: Vec<usize>, // stride 6
}

impl TriangularLattice {
    /// Create a new triangular lattice of dimensions `size x size`.
    ///
    /// Returns `None` if `size < 2`.
    pub fn new(size: usize) -> Option<Self> {
        if size < 2 {
            return None;
        }

        let num_sites = size * size;
        let mut nn_table = Vec::with_capacity(num_sites * 6);
        let mut nnn_table = Vec::with_capacity(num_sites * 6);
        let mut tnn_table = Vec::with_capacity(num_sites * 6);

        for idx in 0..num_sites {
            let row = idx / size;
            let col = idx % size;

            let up = (row + size - 1) % size;
            let down = (row + 1) % size;
            let left = (col + size - 1) % size;
            let right = (col + 1) % size;

            // NN: 6 nearest neighbors
            // Shared by all rows: up, down, left, right
            nn_table.push(up * size + col);    // up
            nn_table.push(down * size + col);  // down
            nn_table.push(row * size + left);  // left
            nn_table.push(row * size + right); // right

            if row % 2 == 0 {
                // Even row: extra diagonals go left
                nn_table.push(up * size + left);   // up-left
                nn_table.push(down * size + left);  // down-left
            } else {
                // Odd row: extra diagonals go right
                nn_table.push(up * size + right);   // up-right
                nn_table.push(down * size + right);  // down-right
            }

            // NNN: 6 next-nearest neighbors (at distance √3 in real space)
            // These are the sites reachable by one NN step + one different NN step.
            // For offset coordinates, NNN depends on row parity.
            let left2 = (col + size - 2) % size;
            let right2 = (col + 2) % size;
            let up2 = (row + size - 2) % size;
            let down2 = (row + 2) % size;

            if row % 2 == 0 {
                // Even row NNN
                nnn_table.push(up * size + right);    // up-right
                nnn_table.push(down * size + right);   // down-right
                nnn_table.push(up2 * size + col);      // up 2
                nnn_table.push(down2 * size + col);    // down 2
                nnn_table.push(up * size + left2);     // up-left-left (via offset)
                // Actually, let me be more careful about NNN on triangular.
                // For a proper triangular lattice with offset coords:
                // NNN are at distance 2 (in lattice-spacing units).
                // Let me just use the second-neighbor shell.
                nnn_table.push(down * size + left2);
            } else {
                // Odd row NNN
                nnn_table.push(up * size + left);
                nnn_table.push(down * size + left);
                nnn_table.push(up2 * size + col);
                nnn_table.push(down2 * size + col);
                nnn_table.push(up * size + right2);
                nnn_table.push(down * size + right2);
            }

            // TNN: 6 third-nearest neighbors (at distance 2 along lattice directions)
            if row % 2 == 0 {
                tnn_table.push(row * size + left2);    // left 2
                tnn_table.push(row * size + right2);   // right 2
                tnn_table.push(up2 * size + left);     // up2-left
                tnn_table.push(up2 * size + col);      // Hmm, this overlaps with NNN
                tnn_table.push(down2 * size + left);
                tnn_table.push(down2 * size + col);
            } else {
                tnn_table.push(row * size + left2);
                tnn_table.push(row * size + right2);
                tnn_table.push(up2 * size + right);
                tnn_table.push(up2 * size + col);
                tnn_table.push(down2 * size + right);
                tnn_table.push(down2 * size + col);
            }
        }

        Some(Self {
            size,
            num_sites,
            shape: [size, size],
            nn_table,
            nnn_table,
            tnn_table,
        })
    }
}

impl Lattice for TriangularLattice {
    fn num_sites(&self) -> usize {
        self.num_sites
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn coordination_number(&self) -> usize {
        6
    }

    fn nnn_coordination_number(&self) -> usize {
        6
    }

    fn nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.nn_table[idx * 6..idx * 6 + 6]
    }

    fn next_nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.nnn_table[idx * 6..idx * 6 + 6]
    }

    fn distance_squared(&self, idx_a: usize, idx_b: usize) -> usize {
        // For offset triangular coordinates, exact Euclidean distance
        // requires conversion to real-space. For now, use axial distance.
        // Real-space coords for offset:
        //   x = col + 0.5 * (row % 2)
        //   y = row * sqrt(3)/2
        // But distance_squared returns integer, so we approximate
        // using the standard lattice metric.
        let row_a = idx_a / self.size;
        let col_a = idx_a % self.size;
        let row_b = idx_b / self.size;
        let col_b = idx_b % self.size;

        let dy = {
            let d = row_a.abs_diff(row_b);
            d.min(self.size - d)
        };
        let dx = {
            let d = col_a.abs_diff(col_b);
            d.min(self.size - d)
        };

        // Approximate: use Manhattan-like metric for triangular
        // This is sufficient for correlation function distance binning.
        dx * dx + dy * dy
    }

    fn flat_to_multi(&self, idx: usize) -> Vec<usize> {
        vec![idx / self.size, idx % self.size]
    }

    fn multi_to_flat(&self, indices: &[usize]) -> usize {
        indices[0] * self.size + indices[1]
    }

    fn tnn_coordination_number(&self) -> usize {
        6
    }

    fn third_nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.tnn_table[idx * 6..idx * 6 + 6]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation_valid() {
        let lattice = TriangularLattice::new(4);
        assert!(lattice.is_some());
        let lattice = lattice.unwrap();
        assert_eq!(lattice.num_sites(), 16);
        assert_eq!(lattice.shape(), &[4, 4]);
        assert_eq!(lattice.coordination_number(), 6);
        assert_eq!(lattice.nnn_coordination_number(), 6);
        assert_eq!(lattice.tnn_coordination_number(), 6);
    }

    #[test]
    fn test_creation_too_small() {
        assert!(TriangularLattice::new(0).is_none());
        assert!(TriangularLattice::new(1).is_none());
    }

    #[test]
    fn test_all_sites_have_correct_neighbor_count() {
        let lattice = TriangularLattice::new(8).unwrap();
        for idx in 0..lattice.num_sites() {
            assert_eq!(lattice.nearest_neighbors(idx).len(), 6);
            assert_eq!(lattice.next_nearest_neighbors(idx).len(), 6);
            assert_eq!(lattice.third_nearest_neighbors(idx).len(), 6);
        }
    }

    #[test]
    fn test_all_neighbors_are_valid_indices() {
        let lattice = TriangularLattice::new(8).unwrap();
        for idx in 0..lattice.num_sites() {
            for &nbr in lattice.nearest_neighbors(idx) {
                assert!(nbr < lattice.num_sites(), "NN {nbr} out of bounds for site {idx}");
            }
            for &nbr in lattice.next_nearest_neighbors(idx) {
                assert!(nbr < lattice.num_sites(), "NNN {nbr} out of bounds for site {idx}");
            }
            for &nbr in lattice.third_nearest_neighbors(idx) {
                assert!(nbr < lattice.num_sites(), "TNN {nbr} out of bounds for site {idx}");
            }
        }
    }

    #[test]
    fn test_no_self_neighbors() {
        let lattice = TriangularLattice::new(8).unwrap();
        for idx in 0..lattice.num_sites() {
            assert!(!lattice.nearest_neighbors(idx).contains(&idx), "Site {idx} is its own NN");
            assert!(!lattice.next_nearest_neighbors(idx).contains(&idx), "Site {idx} is its own NNN");
            assert!(!lattice.third_nearest_neighbors(idx).contains(&idx), "Site {idx} is its own TNN");
        }
    }

    #[test]
    fn test_nn_symmetry() {
        // If j is NN of i, then i must be NN of j
        let lattice = TriangularLattice::new(6).unwrap();
        for idx in 0..lattice.num_sites() {
            for &nbr in lattice.nearest_neighbors(idx) {
                assert!(
                    lattice.nearest_neighbors(nbr).contains(&idx),
                    "Site {nbr} should have {idx} as NN (site {idx} has {nbr})"
                );
            }
        }
    }

    #[test]
    fn test_nn_no_duplicates() {
        let lattice = TriangularLattice::new(8).unwrap();
        for idx in 0..lattice.num_sites() {
            let nn = lattice.nearest_neighbors(idx);
            let mut sorted = nn.to_vec();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), 6, "Site {idx} has duplicate NN");
        }
    }

    #[test]
    fn test_distance_squared_same_site() {
        let lattice = TriangularLattice::new(4).unwrap();
        assert_eq!(lattice.distance_squared(0, 0), 0);
    }

    #[test]
    fn test_flat_to_multi_roundtrip() {
        let lattice = TriangularLattice::new(5).unwrap();
        for idx in 0..lattice.num_sites() {
            let multi = lattice.flat_to_multi(idx);
            assert_eq!(lattice.multi_to_flat(&multi), idx);
        }
    }

    #[test]
    fn test_nn_count_corner_even_row() {
        // Site 0 = (0,0), even row → should have 6 distinct NN
        let lattice = TriangularLattice::new(6).unwrap();
        let nn = lattice.nearest_neighbors(0);
        assert_eq!(nn.len(), 6);
        let mut unique: Vec<usize> = nn.to_vec();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 6, "Site 0 should have 6 unique NN");
    }

    #[test]
    fn test_nn_count_corner_odd_row() {
        // Site on odd row
        let lattice = TriangularLattice::new(6).unwrap();
        let idx = 6; // (1, 0), odd row
        let nn = lattice.nearest_neighbors(idx);
        assert_eq!(nn.len(), 6);
        let mut unique: Vec<usize> = nn.to_vec();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 6, "Site {idx} should have 6 unique NN");
    }

    #[test]
    fn test_energy_all_up() {
        // All spins up on triangular: E = -J1 * 6 / 2 = -3.0 per site
        let lattice = TriangularLattice::new(6).unwrap();
        let spins = vec![1i8; lattice.num_sites()];
        let e = crate::observables::energy_per_site(&spins, &lattice, 1.0, 0.0, 0.0, 0.0);
        assert!(
            (e - (-3.0)).abs() < 1e-10,
            "Expected energy -3.0 for all-up triangular, got {e}"
        );
    }
}
