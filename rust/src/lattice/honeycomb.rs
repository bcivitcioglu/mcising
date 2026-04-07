use super::Lattice;

/// 2D honeycomb lattice with periodic boundary conditions.
///
/// Two-sublattice structure with L×L unit cells, each containing 2 sites
/// (A=sublattice 0, B=sublattice 1). Total sites = 2*L*L.
///
/// **Flat indexing:** `idx = row * (2*L) + col * 2 + sublattice`
/// **Shape:** `[L, L, 2]`
///
/// The honeycomb is built from a "brick wall" offset convention:
/// - The A-B bond within each cell is horizontal (A on left, B on right).
/// - Even rows: inter-cell bonds go down-left for A, down-right for B.
/// - Odd rows: inter-cell bonds go down-right for A, down-left for B.
///
/// Each site has coordination 3 (3 NN from the opposite sublattice).
pub struct HoneycombLattice {
    size: usize,
    num_sites: usize,
    shape: [usize; 3],    // [L, L, 2]
    nn_table: Vec<usize>,  // stride 3
    nnn_table: Vec<usize>, // stride 6
    tnn_table: Vec<usize>, // stride 3
}

impl HoneycombLattice {
    /// Create a new honeycomb lattice with L×L unit cells (2*L*L sites).
    ///
    /// Returns `None` if `size < 2`.
    pub fn new(size: usize) -> Option<Self> {
        if size < 2 {
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

            // Helper to compute flat index from (r, c, s)
            let flat = |r: usize, c: usize, s: usize| -> usize {
                r * stride_row + c * 2 + s
            };

            if sub == 0 {
                // ── Sublattice A ──
                // NN (3 neighbors, all sublattice B):
                nn_table.push(flat(row, col, 1));         // same-cell B
                if row % 2 == 0 {
                    nn_table.push(flat(up, col, 1));      // up B (no col shift)
                    nn_table.push(flat(down, col, 1));    // down B (no col shift)
                } else {
                    let left = (col + size - 1) % size;
                    nn_table.push(flat(up, left, 1));     // up-left B
                    nn_table.push(flat(down, left, 1));   // down-left B
                }

                // NNN (6 neighbors, all sublattice A):
                let left = (col + size - 1) % size;
                let right = (col + 1) % size;
                nnn_table.push(flat(row, left, 0));       // left A
                nnn_table.push(flat(row, right, 0));      // right A
                if row % 2 == 0 {
                    nnn_table.push(flat(up, col, 0));     // up A (no shift)
                    nnn_table.push(flat(up, left, 0));    // up-left A
                    nnn_table.push(flat(down, col, 0));   // down A (no shift)
                    nnn_table.push(flat(down, left, 0));  // down-left A
                } else {
                    nnn_table.push(flat(up, right, 0));   // up-right A
                    nnn_table.push(flat(up, col, 0));     // up A
                    nnn_table.push(flat(down, right, 0)); // down-right A
                    nnn_table.push(flat(down, col, 0));   // down A
                }

                // TNN (3 neighbors, all sublattice B at distance 2):
                let left = (col + size - 1) % size;
                let right = (col + 1) % size;
                tnn_table.push(flat(row, left, 1));       // left B (far)
                if row % 2 == 0 {
                    tnn_table.push(flat(up, left, 1));    // up-left B
                    tnn_table.push(flat(down, left, 1));  // down-left B
                } else {
                    let left2 = (col + size - 2) % size;
                    tnn_table.push(flat(up, left2, 1));   // up-far-left B
                    tnn_table.push(flat(down, left2, 1)); // down-far-left B
                }
            } else {
                // ── Sublattice B ──
                // NN (3 neighbors, all sublattice A):
                // Exact mirror of A's connectivity.
                nn_table.push(flat(row, col, 0));         // same-cell A
                if row % 2 == 0 {
                    let right = (col + 1) % size;
                    nn_table.push(flat(up, right, 0));    // up-right A
                    nn_table.push(flat(down, right, 0));  // down-right A
                } else {
                    nn_table.push(flat(up, col, 0));      // up A (no col shift)
                    nn_table.push(flat(down, col, 0));    // down A (no col shift)
                }

                // NNN (6 neighbors, all sublattice B):
                let left = (col + size - 1) % size;
                let right = (col + 1) % size;
                nnn_table.push(flat(row, left, 1));       // left B
                nnn_table.push(flat(row, right, 1));      // right B
                if row % 2 == 0 {
                    nnn_table.push(flat(up, right, 1));   // up-right B
                    nnn_table.push(flat(up, col, 1));     // up B
                    nnn_table.push(flat(down, right, 1)); // down-right B
                    nnn_table.push(flat(down, col, 1));   // down B
                } else {
                    nnn_table.push(flat(up, col, 1));     // up B
                    nnn_table.push(flat(up, left, 1));    // up-left B
                    nnn_table.push(flat(down, col, 1));   // down B
                    nnn_table.push(flat(down, left, 1));  // down-left B
                }

                // TNN (3 neighbors, all sublattice A at distance 2):
                let right = (col + 1) % size;
                tnn_table.push(flat(row, right, 0));      // right A (far)
                if row % 2 == 0 {
                    let right2 = (col + 2) % size;
                    tnn_table.push(flat(up, right2, 0));  // up-far-right A
                    tnn_table.push(flat(down, right2, 0));// down-far-right A
                } else {
                    tnn_table.push(flat(up, right, 0));   // up-right A
                    tnn_table.push(flat(down, right, 0)); // down-right A
                }
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
        // Approximate distance for correlation function binning.
        // Use unit-cell distance (ignoring sublattice offset).
        let stride = 2 * self.size;
        let row_a = idx_a / stride;
        let col_a = (idx_a % stride) / 2;
        let row_b = idx_b / stride;
        let col_b = (idx_b % stride) / 2;

        let dr = {
            let d = row_a.abs_diff(row_b);
            d.min(self.size - d)
        };
        let dc = {
            let d = col_a.abs_diff(col_b);
            d.min(self.size - d)
        };

        dr * dr + dc * dc
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
    fn test_all_sites_correct_count() {
        let lat = HoneycombLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            assert_eq!(lat.nearest_neighbors(i).len(), 3, "Site {i} wrong NN count");
            assert_eq!(lat.next_nearest_neighbors(i).len(), 6, "Site {i} wrong NNN count");
            assert_eq!(lat.third_nearest_neighbors(i).len(), 3, "Site {i} wrong TNN count");
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
            assert!(!lat.nearest_neighbors(i).contains(&i), "Site {i} is its own NN");
            assert!(!lat.next_nearest_neighbors(i).contains(&i), "Site {i} is its own NNN");
            assert!(!lat.third_nearest_neighbors(i).contains(&i), "Site {i} is its own TNN");
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
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), 3, "Site {i} has duplicate NN: {:?}", nn);
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
                    n % 2, my_sub,
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
                    n % 2, my_sub,
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
        let lat = HoneycombLattice::new(5).unwrap();
        for i in 0..lat.num_sites() {
            assert_eq!(lat.multi_to_flat(&lat.flat_to_multi(i)), i);
        }
    }

    #[test]
    fn test_distance_same_site() {
        let lat = HoneycombLattice::new(4).unwrap();
        assert_eq!(lat.distance_squared(0, 0), 0);
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
