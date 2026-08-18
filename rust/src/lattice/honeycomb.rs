use super::Lattice;

/// 2D honeycomb lattice with periodic boundary conditions.
///
/// Two-sublattice structure with L×L unit cells, each containing 2 sites
/// (A=sublattice 0, B=sublattice 1). Total sites = 2*L*L. L must be even
/// (see [`Self::new`]).
///
/// **Flat indexing:** `idx = row * (2*L) + col * 2 + sublattice`
/// **Shape:** `[L, L, 2]`
///
/// The NN table realizes an armchair-row embedding with NN distance 1
/// and unit-cell width 3. In cartesian coordinates:
///
/// ```text
///   x = 3*col + { A even row: 0,    B even row: +1,
///                 A odd  row: -3/2, B odd  row: -1/2 }
///   y = row * sqrt(3)/2
/// ```
///
/// The A-B bond within each cell is horizontal (A at x, B at x+1); the
/// other two bonds go to the rows above and below (column shift depends
/// on row parity). The three shells, with exact squared distances in
/// this embedding:
///
/// Note the real-space torus of L×L unit cells is 3L × (√3/2)L — an
/// aspect ratio of ~3.46:1 — so observables limited by the shorter
/// (vertical) extent, like the correlation length, saturate near
/// (√3/4)L, not 3L/2.
///
///   NN  (3 per site, opposite sublattice, d² = 1)
///   NNN (6 per site, same sublattice,     d² = 3): the sublattice's
///       own triangular-lattice NN shell; offsets depend only on row
///       parity and are identical for A and B:
///       even rows (Δrow, Δcol): (±2, 0), (−1, 0), (−1, +1), (+1, 0), (+1, +1)
///       odd  rows (Δrow, Δcol): (±2, 0), (−1, −1), (−1, 0), (+1, −1), (+1, 0)
///   TNN (3 per site, opposite sublattice, d² = 4): the site at −2δ for
///       each NN bond vector δ; parity-independent:
///       A: B(r, c−1), B(r±2, c)   B: A(r, c+1), A(r±2, c)
pub struct HoneycombLattice {
    size: usize,
    num_sites: usize,
    shape: [usize; 3],     // [L, L, 2]
    nn_table: Vec<usize>,  // stride 3
    nnn_table: Vec<usize>, // stride 6
    tnn_table: Vec<usize>, // stride 3
}

impl HoneycombLattice {
    /// Create a new honeycomb lattice with L×L unit cells (2*L*L sites).
    ///
    /// Returns `None` if `size < 2` or `size` is odd. With row-parity
    /// offset coordinates, rows 0 and L−1 have the same parity when L is
    /// odd, so bonds across the vertical wrap seam are not reciprocal and
    /// the Hamiltonian would be invalid (B2, #13).
    pub fn new(size: usize) -> Option<Self> {
        if size < 2 || !size.is_multiple_of(2) {
            return None;
        }

        let num_sites = 2 * size * size;
        let stride_row = 2 * size; // number of flat indices per row

        let mut nn_table = Vec::with_capacity(num_sites * 3);
        let mut nnn_table = Vec::with_capacity(num_sites * 6);
        let mut tnn_table = Vec::with_capacity(num_sites * 3);

        for idx in 0..num_sites {
            let row = idx / stride_row;
            let col = (idx % stride_row) / 2;
            let sub = idx % 2;

            let up = (row + size - 1) % size;
            let down = (row + 1) % size;
            let up2 = (row + size - 2) % size;
            let down2 = (row + 2) % size;
            let left = (col + size - 1) % size;
            let right = (col + 1) % size;

            // Helper to compute flat index from (r, c, s)
            let flat = |r: usize, c: usize, s: usize| -> usize { r * stride_row + c * 2 + s };

            // NN (3 neighbors, opposite sublattice, d² = 1): the same-cell
            // horizontal bond plus one bond each to the rows above and
            // below; the column shift of the vertical bonds depends on row
            // parity (see the struct-level embedding).
            if sub == 0 {
                nn_table.push(flat(row, col, 1)); // same-cell B
                if row.is_multiple_of(2) {
                    nn_table.push(flat(up, col, 1)); // up B (no col shift)
                    nn_table.push(flat(down, col, 1)); // down B (no col shift)
                } else {
                    nn_table.push(flat(up, left, 1)); // up-left B
                    nn_table.push(flat(down, left, 1)); // down-left B
                }
            } else {
                nn_table.push(flat(row, col, 0)); // same-cell A
                if row.is_multiple_of(2) {
                    nn_table.push(flat(up, right, 0)); // up-right A
                    nn_table.push(flat(down, right, 0)); // down-right A
                } else {
                    nn_table.push(flat(up, col, 0)); // up A (no col shift)
                    nn_table.push(flat(down, col, 0)); // down A (no col shift)
                }
            }

            // NNN (6 neighbors, same sublattice, d² = 3): each sublattice
            // is a triangular Bravais lattice; this is its NN shell. The
            // offsets depend only on row parity and are identical for A
            // and B: (±2, 0) plus the four diagonal steps whose column
            // shift follows the parity rule.
            nnn_table.push(flat(up2, col, sub)); // up 2
            nnn_table.push(flat(down2, col, sub)); // down 2
            if row.is_multiple_of(2) {
                nnn_table.push(flat(up, col, sub)); // up
                nnn_table.push(flat(up, right, sub)); // up-right
                nnn_table.push(flat(down, col, sub)); // down
                nnn_table.push(flat(down, right, sub)); // down-right
            } else {
                nnn_table.push(flat(up, left, sub)); // up-left
                nnn_table.push(flat(up, col, sub)); // up
                nnn_table.push(flat(down, left, sub)); // down-left
                nnn_table.push(flat(down, col, sub)); // down
            }

            // TNN (3 neighbors, opposite sublattice, d² = 4): the site at
            // −2δ for each NN bond vector δ. Row steps of 0 or ±2 preserve
            // parity, so the offsets are the same for every row.
            if sub == 0 {
                tnn_table.push(flat(row, left, 1)); // left B (far)
                tnn_table.push(flat(up2, col, 1)); // up-2 B
                tnn_table.push(flat(down2, col, 1)); // down-2 B
            } else {
                tnn_table.push(flat(row, right, 0)); // right A (far)
                tnn_table.push(flat(up2, col, 0)); // up-2 A
                tnn_table.push(flat(down2, col, 0)); // down-2 A
            }
        }

        Some(Self {
            size,
            num_sites,
            shape: [size, size, 2],
            nn_table,
            nnn_table,
            tnn_table,
        })
    }
}

impl Lattice for HoneycombLattice {
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
        3
    }

    fn nnn_coordination_number(&self) -> usize {
        6
    }

    fn nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.nn_table[idx * 3..idx * 3 + 3]
    }

    fn next_nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.nnn_table[idx * 6..idx * 6 + 6]
    }

    fn distance_squared(&self, idx_a: usize, idx_b: usize) -> usize {
        // Exact Euclidean squared distance in NN-bond-length units (#35).
        // Doubling the struct-level armchair embedding to keep integers
        // (X = 2x = 6·col + t with t ∈ {0, +2, −3, −1} by (row parity,
        // sublattice); Y = row, so y = Y·√3/2) gives
        //
        //   4·d² = ΔX² + 3·ΔY²
        //
        // X and Y always share the row's parity, so ΔX ≡ ΔY (mod 2) and
        // ΔX² + 3ΔY² ≡ 0 (mod 4): d² is an exact integer — the shells
        // come out at d² = 1 (NN), 3 (NNN), 4 (TNN). Periodic images: a
        // column wrap shifts X by 6L, a row wrap shifts Y by L (parity
        // preserved, L even, so t is unchanged); minimize over the 9
        // images — exact because the torus generators are orthogonal.
        let stride = 2 * self.size;
        let coords = |idx: usize| -> (isize, isize) {
            let row = idx / stride;
            let col = (idx % stride) / 2;
            let sub = idx % 2;
            let t: isize = match (row % 2, sub) {
                (0, 0) => 0,
                (0, _) => 2,
                (_, 0) => -3,
                _ => -1,
            };
            (6 * col as isize + t, row as isize)
        };
        let (x_a, y_a) = coords(idx_a);
        let (x_b, y_b) = coords(idx_b);
        let dx0 = x_b - x_a;
        let dy0 = y_b - y_a;
        let l = self.size as isize;

        let mut best = usize::MAX;
        for m in [-1isize, 0, 1] {
            for n in [-1isize, 0, 1] {
                let dx = dx0 + 6 * n * l;
                let dy = dy0 + m * l;
                let four_d2 = (dx * dx + 3 * dy * dy) as usize;
                best = best.min(four_d2);
            }
        }
        best / 4
    }

    fn flat_to_multi(&self, idx: usize) -> Vec<usize> {
        let stride = 2 * self.size;
        let row = idx / stride;
        let col = (idx % stride) / 2;
        let sub = idx % 2;
        vec![row, col, sub]
    }

    fn multi_to_flat(&self, indices: &[usize]) -> usize {
        indices[0] * (2 * self.size) + indices[1] * 2 + indices[2]
    }

    fn tnn_coordination_number(&self) -> usize {
        3
    }

    fn third_nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.tnn_table[idx * 3..idx * 3 + 3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation_valid() {
        let lat = HoneycombLattice::new(4).unwrap();
        assert_eq!(lat.num_sites(), 32); // 2 * 4 * 4
        assert_eq!(lat.shape(), &[4, 4, 2]);
        assert_eq!(lat.coordination_number(), 3);
        assert_eq!(lat.nnn_coordination_number(), 6);
        assert_eq!(lat.tnn_coordination_number(), 3);
    }

    #[test]
    fn test_creation_too_small() {
        assert!(HoneycombLattice::new(0).is_none());
        assert!(HoneycombLattice::new(1).is_none());
    }

    #[test]
    fn test_creation_odd_rejected() {
        // Odd L breaks bond reciprocity across the row wrap (B2).
        assert!(HoneycombLattice::new(3).is_none());
        assert!(HoneycombLattice::new(5).is_none());
        assert!(HoneycombLattice::new(7).is_none());
    }

    #[test]
    fn test_all_sites_correct_count() {
        let lat = HoneycombLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            assert_eq!(lat.nearest_neighbors(i).len(), 3, "Site {i} wrong NN count");
            assert_eq!(
                lat.next_nearest_neighbors(i).len(),
                6,
                "Site {i} wrong NNN count"
            );
            assert_eq!(
                lat.third_nearest_neighbors(i).len(),
                3,
                "Site {i} wrong TNN count"
            );
        }
    }

    #[test]
    fn test_all_valid_indices() {
        let lat = HoneycombLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            for &n in lat.nearest_neighbors(i) {
                assert!(n < lat.num_sites(), "NN {n} out of bounds for site {i}");
            }
            for &n in lat.next_nearest_neighbors(i) {
                assert!(n < lat.num_sites(), "NNN {n} out of bounds for site {i}");
            }
            for &n in lat.third_nearest_neighbors(i) {
                assert!(n < lat.num_sites(), "TNN {n} out of bounds for site {i}");
            }
        }
    }

    #[test]
    fn test_no_self_neighbors() {
        let lat = HoneycombLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            assert!(
                !lat.nearest_neighbors(i).contains(&i),
                "Site {i} is its own NN"
            );
            assert!(
                !lat.next_nearest_neighbors(i).contains(&i),
                "Site {i} is its own NNN"
            );
            assert!(
                !lat.third_nearest_neighbors(i).contains(&i),
                "Site {i} is its own TNN"
            );
        }
    }

    #[test]
    fn test_nn_symmetry() {
        let lat = HoneycombLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            for &n in lat.nearest_neighbors(i) {
                assert!(
                    lat.nearest_neighbors(n).contains(&i),
                    "Site {n} should have {i} as NN (site {i} has {n} as NN)"
                );
            }
        }
    }

    #[test]
    fn test_nn_no_duplicates() {
        let lat = HoneycombLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            let nn = lat.nearest_neighbors(i);
            let mut sorted = nn.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 3, "Site {i} has duplicate NN: {nn:?}");
        }
    }

    #[test]
    fn test_nn_connects_opposite_sublattice() {
        // All NN should be from the opposite sublattice
        let lat = HoneycombLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            let my_sub = i % 2;
            for &n in lat.nearest_neighbors(i) {
                assert_ne!(
                    n % 2,
                    my_sub,
                    "Site {i} (sub={my_sub}) has NN {n} (sub={}), should be opposite",
                    n % 2
                );
            }
        }
    }

    #[test]
    fn test_nnn_connects_same_sublattice() {
        // All NNN should be from the same sublattice
        let lat = HoneycombLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            let my_sub = i % 2;
            for &n in lat.next_nearest_neighbors(i) {
                assert_eq!(
                    n % 2,
                    my_sub,
                    "Site {i} (sub={my_sub}) has NNN {n} (sub={}), should be same",
                    n % 2
                );
            }
        }
    }

    #[test]
    fn test_nnn_symmetry() {
        let lat = HoneycombLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            for &n in lat.next_nearest_neighbors(i) {
                assert!(
                    lat.next_nearest_neighbors(n).contains(&i),
                    "Site {n} should have {i} as NNN"
                );
            }
        }
    }

    #[test]
    fn test_flat_roundtrip() {
        let lat = HoneycombLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            assert_eq!(lat.multi_to_flat(&lat.flat_to_multi(i)), i);
        }
    }

    #[test]
    fn test_tnn_symmetry() {
        let lat = HoneycombLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            for &n in lat.third_nearest_neighbors(i) {
                assert!(
                    lat.third_nearest_neighbors(n).contains(&i),
                    "Site {n} should have {i} as TNN"
                );
            }
        }
    }

    #[test]
    fn test_tnn_connects_opposite_sublattice() {
        let lat = HoneycombLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            let my_sub = i % 2;
            for &n in lat.third_nearest_neighbors(i) {
                assert_ne!(
                    n % 2,
                    my_sub,
                    "Site {i} (sub={my_sub}) has TNN {n} (sub={}), should be opposite",
                    n % 2
                );
            }
        }
    }

    #[test]
    fn test_shells_disjoint() {
        let lat = HoneycombLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            let nn: std::collections::HashSet<usize> =
                lat.nearest_neighbors(i).iter().copied().collect();
            let nnn: std::collections::HashSet<usize> =
                lat.next_nearest_neighbors(i).iter().copied().collect();
            let tnn: std::collections::HashSet<usize> =
                lat.third_nearest_neighbors(i).iter().copied().collect();
            assert!(nn.is_disjoint(&nnn), "NN∩NNN nonempty for site {i}");
            assert!(nn.is_disjoint(&tnn), "NN∩TNN nonempty for site {i}");
            assert!(nnn.is_disjoint(&tnn), "NNN∩TNN nonempty for site {i}");
        }
    }

    #[test]
    fn test_distance_same_site() {
        let lat = HoneycombLattice::new(4).unwrap();
        assert_eq!(lat.distance_squared(0, 0), 0);
    }

    #[test]
    fn test_distance_squared_shells() {
        // Every table entry sits at its shell's exact squared distance in
        // NN-bond-length units: NN d²=1, NNN d²=3, TNN d²=4.
        let lat = HoneycombLattice::new(8).unwrap();
        for idx in 0..lat.num_sites() {
            for &nbr in lat.nearest_neighbors(idx) {
                assert_eq!(lat.distance_squared(idx, nbr), 1, "NN of {idx}");
            }
            for &nbr in lat.next_nearest_neighbors(idx) {
                assert_eq!(lat.distance_squared(idx, nbr), 3, "NNN of {idx}");
            }
            for &nbr in lat.third_nearest_neighbors(idx) {
                assert_eq!(lat.distance_squared(idx, nbr), 4, "TNN of {idx}");
            }
        }
    }

    #[test]
    fn test_distance_same_cell_ab_is_nn_not_zero() {
        // The #35 regression: the old unit-cell metric ignored the
        // sublattice index, so the same-cell A–B nearest-neighbor bond
        // landed in the d²=0 bin and honeycomb correlation output never
        // reported its NN shell.
        let lat = HoneycombLattice::new(4).unwrap();
        for cell in 0..lat.num_sites() / 2 {
            let a = 2 * cell;
            let b = 2 * cell + 1;
            assert_eq!(lat.distance_squared(a, b), 1, "cell {cell}");
        }
    }

    #[test]
    fn test_distance_squared_symmetric() {
        let lat = HoneycombLattice::new(6).unwrap();
        for a in 0..lat.num_sites() {
            for b in 0..lat.num_sites() {
                assert_eq!(
                    lat.distance_squared(a, b),
                    lat.distance_squared(b, a),
                    "d²({a},{b}) != d²({b},{a})"
                );
            }
        }
    }

    #[test]
    fn test_energy_all_up() {
        // E = -J1 * 3 / 2 = -1.5 per site
        let lat = HoneycombLattice::new(6).unwrap();
        let spins = vec![1i8; lat.num_sites()];
        let e = crate::observables::energy_per_site(&spins, &lat, 1.0, 0.0, 0.0, 0.0);
        assert!(
            (e - (-1.5)).abs() < 1e-10,
            "Expected energy -1.5 for all-up honeycomb, got {e}"
        );
    }
}
