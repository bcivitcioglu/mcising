use super::Lattice;

/// 2D square lattice with periodic boundary conditions.
///
/// Supports nearest-neighbor (coordination 4) and next-nearest-neighbor
/// (coordination 4, diagonal) interactions. Neighbor tables are precomputed
/// at construction time.
pub struct SquareLattice {
    size: usize,
    num_sites: usize,
    shape: [usize; 2],
    nn_table: Vec<Vec<usize>>,
    nnn_table: Vec<Vec<usize>>,
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
        let mut nn_table = Vec::with_capacity(num_sites);
        let mut nnn_table = Vec::with_capacity(num_sites);

        for idx in 0..num_sites {
            let row = idx / size;
            let col = idx % size;

            // Nearest neighbors: up, down, left, right
            nn_table.push(vec![
                ((row + 1) % size) * size + col,       // down
                ((row + size - 1) % size) * size + col, // up
                row * size + (col + 1) % size,          // right
                row * size + (col + size - 1) % size,   // left
            ]);

            // Next-nearest neighbors: four diagonals
            nnn_table.push(vec![
                ((row + 1) % size) * size + (col + 1) % size,          // down-right
                ((row + 1) % size) * size + (col + size - 1) % size,   // down-left
                ((row + size - 1) % size) * size + (col + 1) % size,   // up-right
                ((row + size - 1) % size) * size + (col + size - 1) % size, // up-left
            ]);
        }

        Some(Self {
            size,
            num_sites,
            shape: [size, size],
            nn_table,
            nnn_table,
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
        &self.nn_table[idx]
    }

    fn next_nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.nnn_table[idx]
    }

    fn distance_squared(&self, idx_a: usize, idx_b: usize) -> usize {
        let row_a = idx_a / self.size;
        let col_a = idx_a % self.size;
        let row_b = idx_b / self.size;
        let col_b = idx_b % self.size;

        let dx = {
            let d = if col_a > col_b {
                col_a - col_b
            } else {
                col_b - col_a
            };
            d.min(self.size - d)
        };
        let dy = {
            let d = if row_a > row_b {
                row_a - row_b
            } else {
                row_b - row_a
            };
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
        assert!(nn.contains(&4));  // down
        assert!(nn.contains(&12)); // up (PBC)
        assert!(nn.contains(&1));  // right
        assert!(nn.contains(&3));  // left (PBC)
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
        assert!(nnn.contains(&5));  // down-right
        assert!(nnn.contains(&7));  // down-left (PBC)
        assert!(nnn.contains(&13)); // up-right (PBC)
        assert!(nnn.contains(&15)); // up-left (PBC)
    }

    #[test]
    fn test_all_sites_have_correct_neighbor_count() {
        let lattice = SquareLattice::new(8).unwrap();
        for idx in 0..lattice.num_sites() {
            assert_eq!(lattice.nearest_neighbors(idx).len(), 4);
            assert_eq!(lattice.next_nearest_neighbors(idx).len(), 4);
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
        }
    }

    #[test]
    fn test_no_self_neighbors() {
        let lattice = SquareLattice::new(8).unwrap();
        for idx in 0..lattice.num_sites() {
            assert!(!lattice.nearest_neighbors(idx).contains(&idx));
            assert!(!lattice.next_nearest_neighbors(idx).contains(&idx));
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