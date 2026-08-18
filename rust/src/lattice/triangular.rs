use super::Lattice;

/// 2D triangular lattice with periodic boundary conditions.
///
/// Uses offset coordinates on an L×L grid (L even; see [`Self::new`]).
/// Site (r, c) maps to cartesian coordinates, in units of the lattice
/// spacing:
///
/// ```text
///   x = c + 0.5 * (r % 2)
///   y = r * sqrt(3)/2
/// ```
///
/// which realizes the Bravais vectors a1 = (1, 0) and a2 = (1/2, √3/2).
/// The three neighbor shells, with their exact squared distances in this
/// embedding, are:
///
///   NN  (6 per site, d² = 1): ±a1, ±a2, ±(a2 − a1)
///   NNN (6 per site, d² = 3): ±(a1 + a2), ±(2a2 − a1), ±(2a1 − a2)
///   TNN (6 per site, d² = 4): ±2a1, ±2a2, ±2(a2 − a1)
///
/// In (Δrow, Δcol) offset terms the NN and NNN shells depend on row
/// parity (a ±1 row step lands on the other parity, whose x origin is
/// shifted by 1/2), while the TNN shell is parity-independent (row steps
/// of 0 or ±2 preserve parity):
///
///   NN,  even row: (0, ±1), (±1, 0), (−1, −1), (+1, −1)
///   NN,  odd  row: (0, ±1), (±1, 0), (−1, +1), (+1, +1)
///   NNN, even row: (±2, 0), (−1, +1), (+1, +1), (−1, −2), (+1, −2)
///   NNN, odd  row: (±2, 0), (−1, −1), (+1, −1), (−1, +2), (+1, +2)
///   TNN, any  row: (0, ±2), (+2, ±1), (−2, ±1)
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
    /// Returns `None` if `size < 2` or `size` is odd. With row-parity
    /// offset coordinates, rows 0 and L−1 have the same parity when L is
    /// odd, so the diagonal bonds across the vertical wrap seam are not
    /// reciprocal and the Hamiltonian would be invalid (B2, #13). The
    /// row wrap is only a lattice translation, (0, L·√3/2) = (L/2)(2a2 −
    /// a1), for even L.
    pub fn new(size: usize) -> Option<Self> {
        if size < 2 || !size.is_multiple_of(2) {
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
            nn_table.push(up * size + col); // up
            nn_table.push(down * size + col); // down
            nn_table.push(row * size + left); // left
            nn_table.push(row * size + right); // right

            if row.is_multiple_of(2) {
                // Even row: extra diagonals go left
                nn_table.push(up * size + left); // up-left
                nn_table.push(down * size + left); // down-left
            } else {
                // Odd row: extra diagonals go right
                nn_table.push(up * size + right); // up-right
                nn_table.push(down * size + right); // down-right
            }

            // NNN: 6 next-nearest neighbors at distance √3 (d² = 3), the
            // ±(a1+a2), ±(2a2−a1), ±(2a1−a2) shell. Offsets are parity-
            // dependent because a ±1 row step changes the x origin by 1/2
            // (see the struct-level derivation).
            let left2 = (col + size - 2) % size;
            let right2 = (col + 2) % size;
            let up2 = (row + size - 2) % size;
            let down2 = (row + 2) % size;

            if row.is_multiple_of(2) {
                // Even row NNN: (−1,+1), (+1,+1), (±2,0), (−1,−2), (+1,−2)
                nnn_table.push(up * size + right); // up-right
                nnn_table.push(down * size + right); // down-right
                nnn_table.push(up2 * size + col); // up 2
                nnn_table.push(down2 * size + col); // down 2
                nnn_table.push(up * size + left2); // up-left-left
                nnn_table.push(down * size + left2); // down-left-left
            } else {
                // Odd row NNN: (−1,−1), (+1,−1), (±2,0), (−1,+2), (+1,+2)
                nnn_table.push(up * size + left); // up-left
                nnn_table.push(down * size + left); // down-left
                nnn_table.push(up2 * size + col); // up 2
                nnn_table.push(down2 * size + col); // down 2
                nnn_table.push(up * size + right2); // up-right-right
                nnn_table.push(down * size + right2); // down-right-right
            }

            // TNN: 6 third-nearest neighbors at distance 2 (d² = 4), the
            // ±2a1, ±2a2, ±2(a2−a1) shell. Row steps of 0 or ±2 preserve
            // parity, so the offsets are the same for every row:
            // (0, ±2), (+2, ±1), (−2, ±1).
            tnn_table.push(row * size + left2); // left 2
            tnn_table.push(row * size + right2); // right 2
            tnn_table.push(up2 * size + left); // up2-left
            tnn_table.push(up2 * size + right); // up2-right
            tnn_table.push(down2 * size + left); // down2-left
            tnn_table.push(down2 * size + right); // down2-right
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

    fn dimension(&self) -> usize {
        2
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
        // Exact Euclidean squared distance in the 60° basis, in units of
        // the lattice spacing. With x = c + (r % 2)/2 and y = r·√3/2,
        // doubling x to keep integers (X = 2c + r % 2, Y = r) gives
        //
        //   4·d² = ΔX² + 3·ΔY²
        //
        // ΔX and ΔY always have equal parity (X and r have equal parity
        // mod 2 for even L), so ΔX² + 3ΔY² ≡ 0 (mod 4) and d² is an exact
        // integer — the shells come out at d² = 1 (NN), 3 (NNN), 4 (TNN).
        // Periodic images: a column wrap shifts X by 2L, a row wrap shifts
        // Y by L (and preserves parity, L even, so X is unshifted);
        // minimize over the 9 images.
        let l = self.size as isize;
        let row_a = (idx_a / self.size) as isize;
        let col_a = (idx_a % self.size) as isize;
        let row_b = (idx_b / self.size) as isize;
        let col_b = (idx_b % self.size) as isize;

        let dx0 = 2 * (col_b - col_a) + (row_b % 2 - row_a % 2);
        let dy0 = row_b - row_a;

        let mut best = usize::MAX;
        for m in [-1isize, 0, 1] {
            for n in [-1isize, 0, 1] {
                let dx = dx0 + 2 * n * l;
                let dy = dy0 + m * l;
                let four_d2 = (dx * dx + 3 * dy * dy) as usize;
                best = best.min(four_d2);
            }
        }
        best / 4
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
    fn test_creation_odd_rejected() {
        // Odd L breaks diagonal-bond reciprocity across the row wrap (B2).
        assert!(TriangularLattice::new(3).is_none());
        assert!(TriangularLattice::new(5).is_none());
        assert!(TriangularLattice::new(7).is_none());
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
                assert!(
                    nbr < lattice.num_sites(),
                    "NN {nbr} out of bounds for site {idx}"
                );
            }
            for &nbr in lattice.next_nearest_neighbors(idx) {
                assert!(
                    nbr < lattice.num_sites(),
                    "NNN {nbr} out of bounds for site {idx}"
                );
            }
            for &nbr in lattice.third_nearest_neighbors(idx) {
                assert!(
                    nbr < lattice.num_sites(),
                    "TNN {nbr} out of bounds for site {idx}"
                );
            }
        }
    }

    #[test]
    fn test_no_self_neighbors() {
        let lattice = TriangularLattice::new(8).unwrap();
        for idx in 0..lattice.num_sites() {
            assert!(
                !lattice.nearest_neighbors(idx).contains(&idx),
                "Site {idx} is its own NN"
            );
            assert!(
                !lattice.next_nearest_neighbors(idx).contains(&idx),
                "Site {idx} is its own NNN"
            );
            assert!(
                !lattice.third_nearest_neighbors(idx).contains(&idx),
                "Site {idx} is its own TNN"
            );
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
            sorted.sort_unstable();
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
        let lattice = TriangularLattice::new(6).unwrap();
        for idx in 0..lattice.num_sites() {
            let multi = lattice.flat_to_multi(idx);
            assert_eq!(lattice.multi_to_flat(&multi), idx);
        }
    }

    #[test]
    fn test_distance_squared_shells() {
        // Every table entry sits at its shell's exact squared distance:
        // NN d²=1, NNN d²=3, TNN d²=4.
        let lattice = TriangularLattice::new(8).unwrap();
        for idx in 0..lattice.num_sites() {
            for &nbr in lattice.nearest_neighbors(idx) {
                assert_eq!(lattice.distance_squared(idx, nbr), 1, "NN of {idx}");
            }
            for &nbr in lattice.next_nearest_neighbors(idx) {
                assert_eq!(lattice.distance_squared(idx, nbr), 3, "NNN of {idx}");
            }
            for &nbr in lattice.third_nearest_neighbors(idx) {
                assert_eq!(lattice.distance_squared(idx, nbr), 4, "TNN of {idx}");
            }
        }
    }

    #[test]
    fn test_distance_squared_symmetric() {
        let lattice = TriangularLattice::new(6).unwrap();
        for a in 0..lattice.num_sites() {
            for b in 0..lattice.num_sites() {
                assert_eq!(
                    lattice.distance_squared(a, b),
                    lattice.distance_squared(b, a),
                    "d²({a},{b}) != d²({b},{a})"
                );
            }
        }
    }

    #[test]
    fn test_distance_squared_pbc_wrap() {
        // Sites (0,0) and (0,L-1) are NN through the column wrap.
        let lattice = TriangularLattice::new(6).unwrap();
        assert_eq!(lattice.distance_squared(0, 5), 1);
        // Sites (0,0) and (L-1,0): odd row L-1 wraps to a NN diagonal.
        assert_eq!(lattice.distance_squared(0, 30), 1);
        // Sites (0,0) and (L-2,0): two row steps through the wrap = TNN
        // shell partner only via (−2,±1); straight up-2 is NNN (d²=3).
        assert_eq!(lattice.distance_squared(0, 24), 3);
    }

    #[test]
    fn test_nn_count_corner_even_row() {
        // Site 0 = (0,0), even row → should have 6 distinct NN
        let lattice = TriangularLattice::new(6).unwrap();
        let nn = lattice.nearest_neighbors(0);
        assert_eq!(nn.len(), 6);
        let mut unique: Vec<usize> = nn.to_vec();
        unique.sort_unstable();
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
        unique.sort_unstable();
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
