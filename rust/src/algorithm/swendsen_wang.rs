use super::{McAlgorithm, SweepResult};
use crate::lattice::Lattice;
use rand::Rng;

/// Swendsen-Wang multi-cluster algorithm.
///
/// Partitions the entire lattice into clusters using Fortuin-Kasteleyn bond
/// activation, then independently flips each cluster with probability 1/2.
/// Uses weighted Union-Find with path compression for efficient cluster
/// identification.
///
/// Only valid for J2=0 and h=0 (nearest-neighbor-only Hamiltonian).
///
/// `accepted` in `SweepResult` is the total number of spins flipped.
pub struct SwendsenWang {
    /// Union-Find parent array.
    parent: Vec<usize>,
    /// Union-Find rank array for weighted union.
    rank: Vec<usize>,
}

impl SwendsenWang {
    /// Create a new Swendsen-Wang algorithm instance with scratch buffers
    /// for the given number of lattice sites.
    pub fn new(num_sites: usize) -> Self {
        Self {
            parent: vec![0; num_sites],
            rank: vec![0; num_sites],
        }
    }

    /// Find root with path compression.
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    /// Union by rank.
    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

impl McAlgorithm for SwendsenWang {
    fn sweep<L: Lattice, R: Rng>(
        &mut self,
        spins: &mut [i8],
        lattice: &L,
        j1: f64,
        j2: f64,
        h: f64,
        beta: f64,
        rng: &mut R,
    ) -> SweepResult {
        debug_assert!(
            j2 == 0.0 && h == 0.0,
            "Swendsen-Wang algorithm requires J2=0 and h=0"
        );

        let n = lattice.num_sites();
        let p_add = 1.0 - (-2.0 * beta * j1).exp();

        // Initialize Union-Find
        for i in 0..n {
            self.parent[i] = i;
            self.rank[i] = 0;
        }

        // Activate bonds between aligned nearest neighbors
        for i in 0..n {
            for &j in lattice.nearest_neighbors(i) {
                // Only process each bond once (i < j)
                if j > i && spins[i] == spins[j] && rng.gen::<f64>() < p_add {
                    self.union(i, j);
                }
            }
        }

        // Assign a random flip decision to each cluster root.
        // We use a simple scheme: for each root, generate a bool.
        // To avoid a separate pass to find roots, we use a lazy approach:
        // store flip decisions indexed by root, using a Vec<i8> where
        // 0 = undecided, 1 = no flip, -1 = flip.
        //
        // But since we're reusing buffers, let's use the rank array
        // (which we no longer need) to store decisions:
        // 0 = undecided, 1 = keep, 2 = flip
        for i in 0..n {
            self.rank[i] = 0; // reset to "undecided"
        }

        let mut total_flipped = 0;

        for i in 0..n {
            let root = self.find(i);

            // Decide for this cluster if not yet decided
            if self.rank[root] == 0 {
                self.rank[root] = if rng.gen::<bool>() { 1 } else { 2 };
            }

            // Apply flip decision
            if self.rank[root] == 2 {
                spins[i] = -spins[i];
                total_flipped += 1;
            }
        }

        SweepResult {
            accepted: total_flipped,
            attempted: n,
        }
    }

    fn name(&self) -> &'static str {
        "Swendsen-Wang"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::square::SquareLattice;
    use crate::rng::create_rng;

    fn all_up_spins(n: usize) -> Vec<i8> {
        vec![1; n]
    }

    #[test]
    fn test_sw_name() {
        let sw = SwendsenWang::new(16);
        assert_eq!(sw.name(), "Swendsen-Wang");
    }

    #[test]
    fn test_sweep_preserves_spin_values() {
        let lattice = SquareLattice::new(4).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);
        let mut sw = SwendsenWang::new(lattice.num_sites());

        sw.sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, 1.0, &mut rng);

        for &s in &spins {
            assert!(s == 1 || s == -1, "Spin must be +1 or -1, got {s}");
        }
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let lattice = SquareLattice::new(4).unwrap();
        let mut spins1 = all_up_spins(lattice.num_sites());
        let mut spins2 = all_up_spins(lattice.num_sites());
        let mut rng1 = create_rng(123);
        let mut rng2 = create_rng(123);
        let mut sw1 = SwendsenWang::new(lattice.num_sites());
        let mut sw2 = SwendsenWang::new(lattice.num_sites());

        sw1.sweep(&mut spins1, &lattice, 1.0, 0.0, 0.0, 0.5, &mut rng1);
        sw2.sweep(&mut spins2, &lattice, 1.0, 0.0, 0.0, 0.5, &mut rng2);

        assert_eq!(spins1, spins2);
    }

    #[test]
    fn test_all_up_ground_state_magnetization() {
        let lattice = SquareLattice::new(8).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);
        let mut sw = SwendsenWang::new(lattice.num_sites());
        let beta_large = 10.0;

        // At low T with all-up, the entire lattice is one cluster.
        // It flips with p=0.5, so |m| should remain 1.0.
        for _ in 0..10 {
            sw.sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, beta_large, &mut rng);
        }

        let mag: f64 = spins.iter().map(|&s| f64::from(s)).sum::<f64>() / spins.len() as f64;
        assert!(
            mag.abs() > 0.99,
            "Ground state should remain fully magnetized at low T, got |m|={}",
            mag.abs()
        );
    }

    #[test]
    fn test_flipped_count_bounds() {
        let lattice = SquareLattice::new(8).unwrap();
        let n = lattice.num_sites();
        let mut rng = create_rng(42);
        let mut spins: Vec<i8> = (0..n)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();
        let mut sw = SwendsenWang::new(n);

        for _ in 0..20 {
            let result = sw.sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, 0.5, &mut rng);
            assert!(result.accepted <= n, "Cannot flip more than N spins");
            assert_eq!(result.attempted, n);
        }
    }

    #[test]
    fn test_union_find_correctness() {
        let mut sw = SwendsenWang::new(10);
        for i in 0..10 {
            sw.parent[i] = i;
            sw.rank[i] = 0;
        }

        // Union 0-1-2 into one cluster
        sw.union(0, 1);
        sw.union(1, 2);
        assert_eq!(sw.find(0), sw.find(1));
        assert_eq!(sw.find(1), sw.find(2));

        // 3 should be separate
        assert_ne!(sw.find(0), sw.find(3));

        // Union 3-4
        sw.union(3, 4);
        assert_eq!(sw.find(3), sw.find(4));
        assert_ne!(sw.find(0), sw.find(3));

        // Union the two clusters
        sw.union(2, 3);
        assert_eq!(sw.find(0), sw.find(4));
    }
}