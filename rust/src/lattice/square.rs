use super::Lattice;

/// 2D square lattice with periodic boundary conditions.
///
/// Supports nearest-neighbor (coordination 4), next-nearest-neighbor
/// (coordination 4, diagonal), and third-nearest-neighbor (coordination 4,
/// two steps along axes) interactions. Neighbor tables are precomputed
/// at construction time.
pub struct SquareLattice {
    size: usize,
    num_sites: usize,
    shape: [usize; 2],
    nn_table: Vec<usize>,  // flat, stride 4: nn_table[idx*4 .. idx*4+4]
    nnn_table: Vec<usize>, // flat, stride 4: nnn_table[idx*4 .. idx*4+4]
    tnn_table: Vec<usize>, // flat, stride 4: tnn_table[idx*4 .. idx*4+4]
}

impl SquareLattice {
    /// Create a new square lattice of dimensions `size x size`.
    ///
    /// # Errors
    /// Returns `None` if `size < 2`.
    pub fn new(size: usize) -> Option<Self> {
        if size < 2 {
            return None;
        }

        let num_sites = size * size;
        let mut nn_table = Vec::with_capacity(num_sites * 4);
        let mut nnn_table = Vec::with_capacity(num_sites * 4);
        let mut tnn_table = Vec::with_capacity(num_sites * 4);

        for idx in 0..num_sites {
            let row = idx / size;
            let col = idx % size;

            // Nearest neighbors: up, down, left, right
            nn_table.push(((row + 1) % size) * size + col);        // down
            nn_table.push(((row + size - 1) % size) * size + col); // up
            nn_table.push(row * size + (col + 1) % size);          // right
            nn_table.push(row * size + (col + size - 1) % size);   // left

            // Next-nearest neighbors: four diagonals
            nnn_table.push(((row + 1) % size) * size + (col + 1) % size);         // down-right
            nnn_table.push(((row + 1) % size) * size + (col + size - 1) % size);  // down-left
            nnn_table.push(((row + size - 1) % size) * size + (col + 1) % size);  // up-right
            nnn_table.push(((row + size - 1) % size) * size + (col + size - 1) % size); // up-left

            // Third-nearest neighbors: two steps along each axis
            tnn_table.push(((row + 2) % size) * size + col);        // down 2
            tnn_table.push(((row + size - 2) % size) * size + col); // up 2
            tnn_table.push(row * size + (col + 2) % size);          // right 2
            tnn_table.push(row * size + (col + size - 2) % size);   // left 2
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

impl Lattice for SquareLattice {
    fn num_sites(&self) -> usize {
        self.num_sites
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn coordination_number(&self) -> usize {
        4
    }

    fn nnn_coordination_number(&self) -> usize {
        4
    }

    fn nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.nn_table[idx * 4..idx * 4 + 4]
    }

    fn next_nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.nnn_table[idx * 4..idx * 4 + 4]
    }

    fn distance_squared(&self, idx_a: usize, idx_b: usize) -> usize {
        let row_a = idx_a / self.size;
        let col_a = idx_a % self.size;
        let row_b = idx_b / self.size;
        let col_b = idx_b % self.size;

        let dx = {
            let d = col_a.abs_diff(col_b);
            d.min(self.size - d)
        };
        let dy = {
            let d = row_a.abs_diff(row_b);
            d.min(self.size - d)
        };

        dx * dx + dy * dy
    }

    fn flat_to_multi(&self, idx: usize) -> Vec<usize> {
        vec![idx / self.size, idx % self.size]
    }

    fn multi_to_flat(&self, indices: &[usize]) -> usize {
        indices[0] * self.size + indices[1]
    }

    fn tnn_coordination_number(&self) -> usize {
        4
    }

    fn third_nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.tnn_table[idx * 4..idx * 4 + 4]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation_valid() {
        let lattice = SquareLattice::new(4);
        assert!(lattice.is_some());
        let lattice = lattice.unwrap();
        assert_eq!(lattice.num_sites(), 16);
        assert_eq!(lattice.shape(), &[4, 4]);
        assert_eq!(lattice.coordination_number(), 4);
        assert_eq!(lattice.nnn_coordination_number(), 4);
    }

    #[test]
    fn test_creation_too_small() {
        assert!(SquareLattice::new(0).is_none());
        assert!(SquareLattice::new(1).is_none());
    }

    #[test]
    fn test_nearest_neighbors_corner() {
        // 4x4 lattice, site 0 = (0,0)
        // Neighbors: down=(1,0)=4, up=(3,0)=12, right=(0,1)=1, left=(0,3)=3
        let lattice = SquareLattice::new(4).unwrap();
        let nn = lattice.nearest_neighbors(0);
        assert_eq!(nn.len(), 4);
        assert!(nn.contains(&4)); // down
        assert!(nn.contains(&12)); // up (PBC)
        assert!(nn.contains(&1)); // right
        assert!(nn.contains(&3)); // left (PBC)
    }

    #[test]
    fn test_nearest_neighbors_center() {
        // 4x4 lattice, site 5 = (1,1)
        // Neighbors: down=(2,1)=9, up=(0,1)=1, right=(1,2)=6, left=(1,0)=4
        let lattice = SquareLattice::new(4).unwrap();
        let nn = lattice.nearest_neighbors(5);
        assert_eq!(nn.len(), 4);
        assert!(nn.contains(&9)); // down
        assert!(nn.contains(&1)); // up
        assert!(nn.contains(&6)); // right
        assert!(nn.contains(&4)); // left
    }

    #[test]
    fn test_next_nearest_neighbors_corner() {
        // 4x4 lattice, site 0 = (0,0)
        // NNN: (1,1)=5, (1,3)=7, (3,1)=13, (3,3)=15
        let lattice = SquareLattice::new(4).unwrap();
        let nnn = lattice.next_nearest_neighbors(0);
        assert_eq!(nnn.len(), 4);
        assert!(nnn.contains(&5)); // down-right
        assert!(nnn.contains(&7)); // down-left (PBC)
        assert!(nnn.contains(&13)); // up-right (PBC)
        assert!(nnn.contains(&15)); // up-left (PBC)
    }

    #[test]
    fn test_all_sites_have_correct_neighbor_count() {
        let lattice = SquareLattice::new(8).unwrap();
        for idx in 0..lattice.num_sites() {
            assert_eq!(lattice.nearest_neighbors(idx).len(), 4);
            assert_eq!(lattice.next_nearest_neighbors(idx).len(), 4);
            assert_eq!(lattice.third_nearest_neighbors(idx).len(), 4);
        }
    }

    #[test]
    fn test_all_neighbors_are_valid_indices() {
        let lattice = SquareLattice::new(8).unwrap();
        for idx in 0..lattice.num_sites() {
            for &nbr in lattice.nearest_neighbors(idx) {
                assert!(nbr < lattice.num_sites());
            }
            for &nbr in lattice.next_nearest_neighbors(idx) {
                assert!(nbr < lattice.num_sites());
            }
            for &nbr in lattice.third_nearest_neighbors(idx) {
                assert!(nbr < lattice.num_sites());
            }
        }
    }

    #[test]
    fn test_no_self_neighbors() {
        let lattice = SquareLattice::new(8).unwrap();
        for idx in 0..lattice.num_sites() {
            assert!(!lattice.nearest_neighbors(idx).contains(&idx));
            assert!(!lattice.next_nearest_neighbors(idx).contains(&idx));
            assert!(!lattice.third_nearest_neighbors(idx).contains(&idx));
        }
    }

    #[test]
    fn test_neighbor_symmetry() {
        // If j is a neighbor of i, then i must be a neighbor of j
        let lattice = SquareLattice::new(6).unwrap();
        for idx in 0..lattice.num_sites() {
            for &nbr in lattice.nearest_neighbors(idx) {
                assert!(
                    lattice.nearest_neighbors(nbr).contains(&idx),
                    "Site {nbr} should have {idx} as a neighbor"
                );
            }
            for &nbr in lattice.next_nearest_neighbors(idx) {
                assert!(
                    lattice.next_nearest_neighbors(nbr).contains(&idx),
                    "Site {nbr} should have {idx} as NNN"
                );
            }
            for &nbr in lattice.third_nearest_neighbors(idx) {
                assert!(
                    lattice.third_nearest_neighbors(nbr).contains(&idx),
                    "Site {nbr} should have {idx} as TNN"
                );
            }
        }
    }

    #[test]
    fn test_tnn_coordination_number() {
        let lattice = SquareLattice::new(4).unwrap();
        assert_eq!(lattice.tnn_coordination_number(), 4);
    }

    #[test]
    fn test_third_nearest_neighbors_corner() {
        // 4x4 lattice, site 0 = (0,0)
        // TNN: two steps along axes: (2,0)=8, (0,2)=2, (3-1=2,0) via PBC: up2=(2,0)=8
        // Actually: down2=(2,0)=8, up2=((0+4-2)%4,0)=(2,0)=8... wait.
        // For size=4: down2 = (0+2)%4 = 2, row=2 → idx=8
        //             up2 = (0+4-2)%4 = 2, row=2 → idx=8
        // That means down2 and up2 are the SAME site for size=4!
        // This is expected: on a 4x4 lattice, 2 steps down = 2 steps up (PBC).
        // Use size=6 to avoid this degeneracy.
        let lattice = SquareLattice::new(6).unwrap();
        let tnn = lattice.third_nearest_neighbors(0);
        assert_eq!(tnn.len(), 4);
        // site 0 = (0,0): TNN = (2,0)=12, (4,0)=24, (0,2)=2, (0,4)=4
        assert!(tnn.contains(&12)); // down 2
        assert!(tnn.contains(&24)); // up 2 (PBC: (0+6-2)%6 = 4, idx=4*6+0=24)
        assert!(tnn.contains(&2));  // right 2
        assert!(tnn.contains(&4));  // left 2 (PBC: (0+6-2)%6 = 4)
    }

    #[test]
    fn test_tnn_distance_squared() {
        // TNN are at distance 2 along one axis → d²=4
        let lattice = SquareLattice::new(8).unwrap();
        for idx in 0..lattice.num_sites() {
            for &nbr in lattice.third_nearest_neighbors(idx) {
                assert_eq!(
                    lattice.distance_squared(idx, nbr),
                    4,
                    "TNN of site {idx} at site {nbr} should be at distance²=4"
                );
            }
        }
    }

    #[test]
    fn test_tnn_no_overlap_with_nn_or_nnn() {
        // TNN should not overlap with NN or NNN
        let lattice = SquareLattice::new(8).unwrap();
        for idx in 0..lattice.num_sites() {
            let nn = lattice.nearest_neighbors(idx);
            let nnn = lattice.next_nearest_neighbors(idx);
            for &tnn in lattice.third_nearest_neighbors(idx) {
                assert!(!nn.contains(&tnn), "TNN should not overlap with NN");
                assert!(!nnn.contains(&tnn), "TNN should not overlap with NNN");
            }
        }
    }

    #[test]
    fn test_distance_squared_same_site() {
        let lattice = SquareLattice::new(4).unwrap();
        assert_eq!(lattice.distance_squared(0, 0), 0);
        assert_eq!(lattice.distance_squared(5, 5), 0);
    }

    #[test]
    fn test_distance_squared_nearest_neighbors() {
        let lattice = SquareLattice::new(4).unwrap();
        // Site 0 and site 1 are adjacent -> distance^2 = 1
        assert_eq!(lattice.distance_squared(0, 1), 1);
        // Site 0 and site 4 are adjacent (vertically) -> distance^2 = 1
        assert_eq!(lattice.distance_squared(0, 4), 1);
    }

    #[test]
    fn test_distance_squared_pbc() {
        let lattice = SquareLattice::new(4).unwrap();
        // Site 0=(0,0) and site 3=(0,3): PBC distance = min(3, 4-3) = 1
        assert_eq!(lattice.distance_squared(0, 3), 1);
        // Site 0=(0,0) and site 12=(3,0): PBC distance = min(3, 4-3) = 1
        assert_eq!(lattice.distance_squared(0, 12), 1);
    }

    #[test]
    fn test_distance_squared_symmetry() {
        let lattice = SquareLattice::new(6).unwrap();
        for i in 0..lattice.num_sites() {
            for j in i..lattice.num_sites() {
                assert_eq!(
                    lattice.distance_squared(i, j),
                    lattice.distance_squared(j, i)
                );
            }
        }
    }

    #[test]
    fn test_flat_to_multi_roundtrip() {
        let lattice = SquareLattice::new(5).unwrap();
        for idx in 0..lattice.num_sites() {
            let multi = lattice.flat_to_multi(idx);
            assert_eq!(lattice.multi_to_flat(&multi), idx);
        }
    }
}
