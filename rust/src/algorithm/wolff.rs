use super::{McAlgorithm, SweepResult};
use crate::lattice::Lattice;
use rand::Rng;

/// Wolff single-cluster algorithm.
///
/// Builds a cluster by BFS/DFS from a random seed site, adding aligned
/// nearest neighbors with probability p_add = 1 - exp(-2 * beta * J1).
/// The entire cluster is then flipped.
///
/// One Wolff "sweep" is ONE cluster update — NOT `num_sites` flip
/// attempts like Metropolis/Swendsen-Wang. A flip-budget sweep (build
/// clusters until >= N spins flipped, then return) was implemented and
/// rejected in P10: returning at that state-dependent stopping time
/// size-biases the final cluster, over-selecting ordered states — the
/// exact-enumeration oracle rejected it at 200-600 sigma (12.5%
/// relative error in <e> at Tc on the 4x4 square). An unbiased
/// equal-work scheme needs a cluster count frozen independently of the
/// measured trajectory (tracked as future work); until then the work
/// accounting below is honest and callers scale `n_sweeps` themselves.
///
/// Only valid for J1>0 with J2=J3=h=0 (ferromagnetic nearest-neighbor-only
/// Hamiltonian); enforced at the boundary in `IsingSimulation::new_internal`.
///
/// `accepted` in `SweepResult` is the cluster size; `attempted` equals
/// it (rejection-free — every site added to the cluster is flipped);
/// `cluster_flips` is 1.
pub struct Wolff {
    /// Reusable visited flags (one per site).
    visited: Vec<bool>,
    /// Reusable stack for DFS cluster growth.
    stack: Vec<usize>,
    /// Sites in the current cluster (for efficient clearing of visited).
    cluster: Vec<usize>,
}

impl Wolff {
    /// Create a new Wolff algorithm instance with scratch buffers for the
    /// given number of lattice sites.
    pub fn new(num_sites: usize) -> Self {
        Self {
            visited: vec![false; num_sites],
            stack: Vec::with_capacity(num_sites),
            cluster: Vec::with_capacity(num_sites),
        }
    }
}

impl McAlgorithm for Wolff {
    fn sweep<L: Lattice, R: Rng>(
        &mut self,
        spins: &mut [i8],
        lattice: &L,
        j1: f64,
        _j2: f64,
        _j3: f64,
        _h: f64,
        beta: f64,
        rng: &mut R,
    ) -> SweepResult {
        let n = lattice.num_sites();
        // p_add in (0,1) requires J1 > 0 — guaranteed by the constructor.
        let p_add = 1.0 - (-2.0 * beta * j1).exp();

        // Pick random seed site
        let seed = rng.gen_range(0..n);
        let cluster_spin = spins[seed];

        // Initialize DFS
        self.stack.clear();
        self.cluster.clear();
        self.stack.push(seed);
        self.visited[seed] = true;
        self.cluster.push(seed);

        // Grow cluster via DFS
        while let Some(site) = self.stack.pop() {
            for &nbr in lattice.nearest_neighbors(site) {
                if !self.visited[nbr] && spins[nbr] == cluster_spin && rng.gen::<f64>() < p_add {
                    self.visited[nbr] = true;
                    self.stack.push(nbr);
                    self.cluster.push(nbr);
                }
            }
        }

        let cluster_size = self.cluster.len();

        // Flip all cluster spins and clear visited flags
        for &site in &self.cluster {
            spins[site] = -spins[site];
            self.visited[site] = false;
        }

        SweepResult {
            accepted: cluster_size,
            attempted: cluster_size,
            cluster_flips: 1,
        }
    }

    fn name(&self) -> &'static str {
        "Wolff"
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
    fn test_wolff_name() {
        let wolff = Wolff::new(16);
        assert_eq!(wolff.name(), "Wolff");
    }

    #[test]
    fn test_sweep_preserves_spin_values() {
        let lattice = SquareLattice::new(4).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);
        let mut wolff = Wolff::new(lattice.num_sites());

        wolff.sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, 0.0, 1.0, &mut rng);

        for &s in &spins {
            assert!(s == 1 || s == -1, "Spin must be +1 or -1, got {s}");
        }
    }

    #[test]
    fn test_accounting_invariants() {
        // Exact, threshold-free invariants of the honest accounting:
        // one cluster per sweep, rejection-free (attempted == accepted),
        // cluster within lattice bounds.
        let lattice = SquareLattice::new(8).unwrap();
        let n = lattice.num_sites();
        let mut rng = create_rng(42);
        let mut spins: Vec<i8> = (0..n)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();
        let mut wolff = Wolff::new(n);

        for _ in 0..20 {
            let result = wolff.sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, 0.0, 0.5, &mut rng);
            assert!(result.accepted >= 1, "Cluster must have at least 1 site");
            assert!(result.accepted <= n, "Cluster cannot exceed lattice size");
            assert_eq!(result.attempted, result.accepted, "Wolff is rejection-free");
            assert_eq!(result.cluster_flips, 1, "One cluster per sweep");
        }
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let lattice = SquareLattice::new(4).unwrap();
        let mut spins1 = all_up_spins(lattice.num_sites());
        let mut spins2 = all_up_spins(lattice.num_sites());
        let mut rng1 = create_rng(123);
        let mut rng2 = create_rng(123);
        let mut wolff1 = Wolff::new(lattice.num_sites());
        let mut wolff2 = Wolff::new(lattice.num_sites());

        wolff1.sweep(&mut spins1, &lattice, 1.0, 0.0, 0.0, 0.0, 0.5, &mut rng1);
        wolff2.sweep(&mut spins2, &lattice, 1.0, 0.0, 0.0, 0.0, 0.5, &mut rng2);

        assert_eq!(spins1, spins2);
    }

    #[test]
    fn test_all_up_ground_state_stable_at_low_t() {
        let lattice = SquareLattice::new(8).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);
        let mut wolff = Wolff::new(lattice.num_sites());
        let beta_large = 10.0; // T = 0.1

        // At low T, p_add is very high, so the entire lattice forms one cluster.
        // Since all spins are aligned, the cluster = entire lattice, and it flips
        // back and forth. Magnetization magnitude should stay 1.0.
        for _ in 0..10 {
            wolff.sweep(
                &mut spins, &lattice, 1.0, 0.0, 0.0, 0.0, beta_large, &mut rng,
            );
        }

        let mag: f64 = spins.iter().map(|&s| f64::from(s)).sum::<f64>() / spins.len() as f64;
        assert!(
            mag.abs() > 0.99,
            "Ground state should remain fully magnetized at low T, got |m|={}",
            mag.abs()
        );
    }

    #[test]
    fn test_high_temp_small_clusters() {
        let lattice = SquareLattice::new(16).unwrap();
        let n = lattice.num_sites();
        let mut spins = all_up_spins(n);
        let mut rng = create_rng(42);
        let mut wolff = Wolff::new(n);
        let beta_small = 0.01; // T = 100

        // At high T, p_add ~ 2*beta*j1 ~ 0.02, clusters should be small.
        // accepted / cluster_flips is the true mean cluster size under
        // the flip-budget sweep.
        let mut total_flipped = 0;
        let mut total_clusters = 0;
        for _ in 0..100 {
            let result = wolff.sweep(
                &mut spins, &lattice, 1.0, 0.0, 0.0, 0.0, beta_small, &mut rng,
            );
            total_flipped += result.accepted;
            total_clusters += result.cluster_flips;
        }
        let avg_cluster_size = total_flipped as f64 / total_clusters as f64;
        assert!(
            avg_cluster_size < n as f64 / 2.0,
            "At high T, average cluster size should be small, got {avg_cluster_size}"
        );
    }
}
