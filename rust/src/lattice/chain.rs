use super::Lattice;

/// 1D chain lattice with periodic boundary conditions.
///
/// The simplest lattice: N sites on a ring. Each site has 2 nearest neighbors
/// (left, right), 2 NNN (distance 2), and 2 TNN (distance 3).
///
/// No finite-temperature phase transition (Tc = 0). Pedagogical lattice.
pub struct ChainLattice {
    size: usize,
    shape: [usize; 1],
    nn_table: Vec<usize>,  // stride 2
    nnn_table: Vec<usize>, // stride 2
    tnn_table: Vec<usize>, // stride 2
}

impl ChainLattice {
    /// Create a new chain lattice with `size` sites.
    ///
    /// Returns `None` if `size < 2`.
    pub fn new(size: usize) -> Option<Self> {
        if size < 2 {
            return None;
        }

        let mut nn_table = Vec::with_capacity(size * 2);
        let mut nnn_table = Vec::with_capacity(size * 2);
        let mut tnn_table = Vec::with_capacity(size * 2);

        for i in 0..size {
            // NN: left, right
            nn_table.push((i + size - 1) % size);
            nn_table.push((i + 1) % size);

            // NNN: distance 2
            nnn_table.push((i + size - 2) % size);
            nnn_table.push((i + 2) % size);

            // TNN: distance 3
            tnn_table.push((i + size - 3) % size);
            tnn_table.push((i + 3) % size);
        }

        Some(Self {
            size,
            shape: [size],
            nn_table,
            nnn_table,
            tnn_table,
        })
    }
}

impl Lattice for ChainLattice {
    fn num_sites(&self) -> usize {
        self.size
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn coordination_number(&self) -> usize {
        2
    }

    fn nnn_coordination_number(&self) -> usize {
        2
    }

    fn nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.nn_table[idx * 2..idx * 2 + 2]
    }

    fn next_nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.nnn_table[idx * 2..idx * 2 + 2]
    }

    fn distance_squared(&self, idx_a: usize, idx_b: usize) -> usize {
        let d = idx_a.abs_diff(idx_b);
        let d = d.min(self.size - d);
        d * d
    }

    fn flat_to_multi(&self, idx: usize) -> Vec<usize> {
        vec![idx]
    }

    fn multi_to_flat(&self, indices: &[usize]) -> usize {
        indices[0]
    }

    fn tnn_coordination_number(&self) -> usize {
        2
    }

    fn third_nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.tnn_table[idx * 2..idx * 2 + 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation_valid() {
        let lat = ChainLattice::new(10).unwrap();
        assert_eq!(lat.num_sites(), 10);
        assert_eq!(lat.shape(), &[10]);
        assert_eq!(lat.coordination_number(), 2);
        assert_eq!(lat.nnn_coordination_number(), 2);
        assert_eq!(lat.tnn_coordination_number(), 2);
    }

    #[test]
    fn test_creation_too_small() {
        assert!(ChainLattice::new(0).is_none());
        assert!(ChainLattice::new(1).is_none());
    }

    #[test]
    fn test_nn_correct() {
        let lat = ChainLattice::new(6).unwrap();
        // Site 0: left=5 (PBC), right=1
        let nn = lat.nearest_neighbors(0);
        assert!(nn.contains(&5));
        assert!(nn.contains(&1));
        // Site 3: left=2, right=4
        let nn = lat.nearest_neighbors(3);
        assert!(nn.contains(&2));
        assert!(nn.contains(&4));
    }

    #[test]
    fn test_nnn_correct() {
        let lat = ChainLattice::new(8).unwrap();
        // Site 0: distance 2 left=6, right=2
        let nnn = lat.next_nearest_neighbors(0);
        assert!(nnn.contains(&6));
        assert!(nnn.contains(&2));
    }

    #[test]
    fn test_tnn_correct() {
        let lat = ChainLattice::new(8).unwrap();
        // Site 0: distance 3 left=5, right=3
        let tnn = lat.third_nearest_neighbors(0);
        assert!(tnn.contains(&5));
        assert!(tnn.contains(&3));
    }

    #[test]
    fn test_all_sites_correct_count() {
        let lat = ChainLattice::new(20).unwrap();
        for i in 0..lat.num_sites() {
            assert_eq!(lat.nearest_neighbors(i).len(), 2);
            assert_eq!(lat.next_nearest_neighbors(i).len(), 2);
            assert_eq!(lat.third_nearest_neighbors(i).len(), 2);
        }
    }

    #[test]
    fn test_all_valid_indices() {
        let lat = ChainLattice::new(20).unwrap();
        for i in 0..lat.num_sites() {
            for &n in lat.nearest_neighbors(i) { assert!(n < lat.num_sites()); }
            for &n in lat.next_nearest_neighbors(i) { assert!(n < lat.num_sites()); }
            for &n in lat.third_nearest_neighbors(i) { assert!(n < lat.num_sites()); }
        }
    }

    #[test]
    fn test_no_self_neighbors() {
        let lat = ChainLattice::new(20).unwrap();
        for i in 0..lat.num_sites() {
            assert!(!lat.nearest_neighbors(i).contains(&i));
            assert!(!lat.next_nearest_neighbors(i).contains(&i));
            assert!(!lat.third_nearest_neighbors(i).contains(&i));
        }
    }

    #[test]
    fn test_nn_symmetry() {
        let lat = ChainLattice::new(10).unwrap();
        for i in 0..lat.num_sites() {
            for &n in lat.nearest_neighbors(i) {
                assert!(lat.nearest_neighbors(n).contains(&i),
                    "{n} should have {i} as NN");
            }
        }
    }

    #[test]
    fn test_distance_squared() {
        let lat = ChainLattice::new(10).unwrap();
        assert_eq!(lat.distance_squared(0, 0), 0);
        assert_eq!(lat.distance_squared(0, 1), 1);
        assert_eq!(lat.distance_squared(0, 9), 1); // PBC
        assert_eq!(lat.distance_squared(0, 5), 25); // max distance = 5, 5^2=25
        assert_eq!(lat.distance_squared(3, 7), 16); // d=min(4,6)=4, 4^2=16
    }

    #[test]
    fn test_distance_symmetry() {
        let lat = ChainLattice::new(10).unwrap();
        for i in 0..lat.num_sites() {
            for j in i..lat.num_sites() {
                assert_eq!(lat.distance_squared(i, j), lat.distance_squared(j, i));
            }
        }
    }

    #[test]
    fn test_nn_at_distance_1() {
        let lat = ChainLattice::new(20).unwrap();
        for i in 0..lat.num_sites() {
            for &n in lat.nearest_neighbors(i) {
                assert_eq!(lat.distance_squared(i, n), 1, "NN should be at d²=1");
            }
        }
    }

    #[test]
    fn test_flat_roundtrip() {
        let lat = ChainLattice::new(10).unwrap();
        for i in 0..lat.num_sites() {
            assert_eq!(lat.multi_to_flat(&lat.flat_to_multi(i)), i);
        }
    }

    #[test]
    fn test_energy_all_up() {
        // E = -J1 * 2 / 2 = -1.0 per site
        let lat = ChainLattice::new(10).unwrap();
        let spins = vec![1i8; 10];
        let e = crate::observables::energy_per_site(&spins, &lat, 1.0, 0.0, 0.0, 0.0);
        assert!((e - (-1.0)).abs() < 1e-10, "Expected -1.0, got {e}");
    }
}
