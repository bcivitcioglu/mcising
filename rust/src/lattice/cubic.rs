use super::Lattice;

/// 3D simple cubic lattice with periodic boundary conditions.
///
/// L×L×L grid with 6 nearest neighbors (±x, ±y, ±z), 12 next-nearest
/// (face diagonals), and 8 third-nearest (body diagonals).
///
/// **Flat indexing:** `idx = z * L * L + y * L + x`
/// **Shape:** `[L, L, L]`
pub struct CubicLattice {
    size: usize,
    num_sites: usize,
    shape: [usize; 3],
    nn_table: Vec<usize>,  // stride 6
    nnn_table: Vec<usize>, // stride 12
    tnn_table: Vec<usize>, // stride 8
}

impl CubicLattice {
    /// Create a new cubic lattice of dimensions `size x size x size`.
    ///
    /// Returns `None` if `size < 2`.
    pub fn new(size: usize) -> Option<Self> {
        if size < 2 {
            return None;
        }

        let l2 = size * size;
        let num_sites = l2 * size;
        let mut nn_table = Vec::with_capacity(num_sites * 6);
        let mut nnn_table = Vec::with_capacity(num_sites * 12);
        let mut tnn_table = Vec::with_capacity(num_sites * 8);

        for idx in 0..num_sites {
            let z = idx / l2;
            let y = (idx % l2) / size;
            let x = idx % size;

            let xp = (x + 1) % size;
            let xm = (x + size - 1) % size;
            let yp = (y + 1) % size;
            let ym = (y + size - 1) % size;
            let zp = (z + 1) % size;
            let zm = (z + size - 1) % size;

            let flat = |zz: usize, yy: usize, xx: usize| -> usize {
                zz * l2 + yy * size + xx
            };

            // NN: 6 neighbors along axes
            nn_table.push(flat(z, y, xp)); // +x
            nn_table.push(flat(z, y, xm)); // -x
            nn_table.push(flat(z, yp, x)); // +y
            nn_table.push(flat(z, ym, x)); // -y
            nn_table.push(flat(zp, y, x)); // +z
            nn_table.push(flat(zm, y, x)); // -z

            // NNN: 12 face diagonals (pairs of two axes ±1)
            // xy-plane (4)
            nnn_table.push(flat(z, yp, xp));
            nnn_table.push(flat(z, yp, xm));
            nnn_table.push(flat(z, ym, xp));
            nnn_table.push(flat(z, ym, xm));
            // xz-plane (4)
            nnn_table.push(flat(zp, y, xp));
            nnn_table.push(flat(zp, y, xm));
            nnn_table.push(flat(zm, y, xp));
            nnn_table.push(flat(zm, y, xm));
            // yz-plane (4)
            nnn_table.push(flat(zp, yp, x));
            nnn_table.push(flat(zp, ym, x));
            nnn_table.push(flat(zm, yp, x));
            nnn_table.push(flat(zm, ym, x));

            // TNN: 8 body diagonals (all three axes ±1)
            tnn_table.push(flat(zp, yp, xp));
            tnn_table.push(flat(zp, yp, xm));
            tnn_table.push(flat(zp, ym, xp));
            tnn_table.push(flat(zp, ym, xm));
            tnn_table.push(flat(zm, yp, xp));
            tnn_table.push(flat(zm, yp, xm));
            tnn_table.push(flat(zm, ym, xp));
            tnn_table.push(flat(zm, ym, xm));
        }

        Some(Self {
            size,
            num_sites,
            shape: [size, size, size],
            nn_table,
            nnn_table,
            tnn_table,
        })
    }
}

impl Lattice for CubicLattice {
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
        12
    }

    fn nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.nn_table[idx * 6..idx * 6 + 6]
    }

    fn next_nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.nnn_table[idx * 12..idx * 12 + 12]
    }

    fn distance_squared(&self, idx_a: usize, idx_b: usize) -> usize {
        let l2 = self.size * self.size;

        let za = idx_a / l2;
        let ya = (idx_a % l2) / self.size;
        let xa = idx_a % self.size;

        let zb = idx_b / l2;
        let yb = (idx_b % l2) / self.size;
        let xb = idx_b % self.size;

        let dx = { let d = xa.abs_diff(xb); d.min(self.size - d) };
        let dy = { let d = ya.abs_diff(yb); d.min(self.size - d) };
        let dz = { let d = za.abs_diff(zb); d.min(self.size - d) };

        dx * dx + dy * dy + dz * dz
    }

    fn flat_to_multi(&self, idx: usize) -> Vec<usize> {
        let l2 = self.size * self.size;
        vec![idx / l2, (idx % l2) / self.size, idx % self.size]
    }

    fn multi_to_flat(&self, indices: &[usize]) -> usize {
        indices[0] * self.size * self.size + indices[1] * self.size + indices[2]
    }

    fn tnn_coordination_number(&self) -> usize {
        8
    }

    fn third_nearest_neighbors(&self, idx: usize) -> &[usize] {
        &self.tnn_table[idx * 8..idx * 8 + 8]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation_valid() {
        let lat = CubicLattice::new(4).unwrap();
        assert_eq!(lat.num_sites(), 64); // 4^3
        assert_eq!(lat.shape(), &[4, 4, 4]);
        assert_eq!(lat.coordination_number(), 6);
        assert_eq!(lat.nnn_coordination_number(), 12);
        assert_eq!(lat.tnn_coordination_number(), 8);
    }

    #[test]
    fn test_creation_too_small() {
        assert!(CubicLattice::new(0).is_none());
        assert!(CubicLattice::new(1).is_none());
    }

    #[test]
    fn test_all_sites_correct_count() {
        let lat = CubicLattice::new(4).unwrap();
        for i in 0..lat.num_sites() {
            assert_eq!(lat.nearest_neighbors(i).len(), 6, "Site {i} wrong NN count");
            assert_eq!(lat.next_nearest_neighbors(i).len(), 12, "Site {i} wrong NNN count");
            assert_eq!(lat.third_nearest_neighbors(i).len(), 8, "Site {i} wrong TNN count");
        }
    }

    #[test]
    fn test_all_valid_indices() {
        let lat = CubicLattice::new(4).unwrap();
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
        let lat = CubicLattice::new(4).unwrap();
        for i in 0..lat.num_sites() {
            assert!(!lat.nearest_neighbors(i).contains(&i));
            assert!(!lat.next_nearest_neighbors(i).contains(&i));
            assert!(!lat.third_nearest_neighbors(i).contains(&i));
        }
    }

    #[test]
    fn test_nn_symmetry() {
        let lat = CubicLattice::new(4).unwrap();
        for i in 0..lat.num_sites() {
            for &n in lat.nearest_neighbors(i) {
                assert!(lat.nearest_neighbors(n).contains(&i),
                    "Site {n} should have {i} as NN");
            }
        }
    }

    #[test]
    fn test_nn_no_duplicates() {
        let lat = CubicLattice::new(4).unwrap();
        for i in 0..lat.num_sites() {
            let nn = lat.nearest_neighbors(i);
            let mut s = nn.to_vec();
            s.sort();
            s.dedup();
            assert_eq!(s.len(), 6, "Site {i} has duplicate NN");
        }
    }

    #[test]
    fn test_nnn_symmetry() {
        let lat = CubicLattice::new(4).unwrap();
        for i in 0..lat.num_sites() {
            for &n in lat.next_nearest_neighbors(i) {
                assert!(lat.next_nearest_neighbors(n).contains(&i),
                    "Site {n} should have {i} as NNN");
            }
        }
    }

    #[test]
    fn test_tnn_symmetry() {
        let lat = CubicLattice::new(4).unwrap();
        for i in 0..lat.num_sites() {
            for &n in lat.third_nearest_neighbors(i) {
                assert!(lat.third_nearest_neighbors(n).contains(&i),
                    "Site {n} should have {i} as TNN");
            }
        }
    }

    #[test]
    fn test_nn_at_distance_1() {
        let lat = CubicLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            for &n in lat.nearest_neighbors(i) {
                assert_eq!(lat.distance_squared(i, n), 1, "NN should be at d²=1");
            }
        }
    }

    #[test]
    fn test_nnn_at_distance_2() {
        let lat = CubicLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            for &n in lat.next_nearest_neighbors(i) {
                assert_eq!(lat.distance_squared(i, n), 2, "NNN should be at d²=2");
            }
        }
    }

    #[test]
    fn test_tnn_at_distance_3() {
        let lat = CubicLattice::new(6).unwrap();
        for i in 0..lat.num_sites() {
            for &n in lat.third_nearest_neighbors(i) {
                assert_eq!(lat.distance_squared(i, n), 3, "TNN should be at d²=3");
            }
        }
    }

    #[test]
    fn test_distance_squared_same_site() {
        let lat = CubicLattice::new(4).unwrap();
        assert_eq!(lat.distance_squared(0, 0), 0);
    }

    #[test]
    fn test_distance_pbc() {
        let lat = CubicLattice::new(4).unwrap();
        // Site 0=(0,0,0) and site 3=(0,0,3): PBC dx=min(3,1)=1
        assert_eq!(lat.distance_squared(0, 3), 1);
    }

    #[test]
    fn test_distance_symmetry() {
        let lat = CubicLattice::new(4).unwrap();
        for i in 0..lat.num_sites() {
            for j in i..lat.num_sites() {
                assert_eq!(lat.distance_squared(i, j), lat.distance_squared(j, i));
            }
        }
    }

    #[test]
    fn test_flat_roundtrip() {
        let lat = CubicLattice::new(5).unwrap();
        for i in 0..lat.num_sites() {
            assert_eq!(lat.multi_to_flat(&lat.flat_to_multi(i)), i);
        }
    }

    #[test]
    fn test_energy_all_up() {
        // E = -J1 * 6 / 2 = -3.0 per site
        let lat = CubicLattice::new(4).unwrap();
        let spins = vec![1i8; lat.num_sites()];
        let e = crate::observables::energy_per_site(&spins, &lat, 1.0, 0.0, 0.0, 0.0);
        assert!(
            (e - (-3.0)).abs() < 1e-10,
            "Expected energy -3.0 for all-up cubic, got {e}"
        );
    }
}
