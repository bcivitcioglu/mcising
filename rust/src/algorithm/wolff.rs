use super::{McAlgorithm, SweepResult};
use crate::lattice::Lattice;
use crate::observables::ShellSums;
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
///
/// `SweepResult::delta` is reported only when [`Wolff::set_track_delta`]
/// has enabled it: the boundary pass it needs costs a noticeable fraction
/// of a cluster build, and only a caller that carries the shell sums
/// forward (the parallel-tempering ladder) has a use for it.
pub struct Wolff {
    /// Reusable visited flags (one per site).
    visited: Vec<bool>,
    /// Reusable stack for DFS cluster growth.
    stack: Vec<usize>,
    /// Sites in the current cluster (for efficient clearing of visited).
    cluster: Vec<usize>,
    /// Whether to evaluate the cluster boundary for `SweepResult::delta`.
    track_delta: bool,
}

impl Wolff {
    /// Create a new Wolff algorithm instance with scratch buffers for the
    /// given number of lattice sites.
    pub fn new(num_sites: usize) -> Self {
        Self {
            visited: vec![false; num_sites],
            stack: Vec::with_capacity(num_sites),
            cluster: Vec::with_capacity(num_sites),
            track_delta: false,
        }
    }

    /// Enable or disable reporting the exact shell-sum change of every
    /// cluster flip in `SweepResult::delta` (off by default; the sampling
    /// itself — RNG stream included — is unaffected either way).
    pub fn set_track_delta(&mut self, on: bool) {
        self.track_delta = on;
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

        // Boundary sum before the flip (the flip loop clears `visited`):
        // every bond from a cluster site to a non-cluster neighbour changes
        // sign, and each such bond is seen twice in the ordered-pair shell
        // sum, so the nearest-neighbour shell changes by -4·s·Σ_boundary.
        // Interior bonds are untouched (both ends flip). No RNG is drawn.
        let delta = self.track_delta.then(|| {
            let mut boundary: i64 = 0;
            for &site in &self.cluster {
                for &nbr in lattice.nearest_neighbors(site) {
                    if !self.visited[nbr] {
                        boundary += i64::from(spins[nbr]);
                    }
                }
            }
            ShellSums {
                nn: -4 * i64::from(cluster_spin) * boundary,
                nnn: 0,
                tnn: 0,
                magnetization: -2 * i64::from(cluster_spin) * cluster_size as i64,
            }
        });

        // Flip all cluster spins and clear visited flags
        for &site in &self.cluster {
            spins[site] = -spins[site];
            self.visited[site] = false;
        }

        SweepResult {
            accepted: cluster_size,
            attempted: cluster_size,
            cluster_flips: 1,
            delta,
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

    fn assert_delta_matches_shell_sums<L: Lattice>(lattice: &L, label: &str) {
        use crate::observables::shell_sums;
        let n = lattice.num_sites();
        let mut rng = create_rng(42);
        let mut spins: Vec<i8> = (0..n)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();
        let mut wolff = Wolff::new(n);
        wolff.set_track_delta(true);
        for step in 0..20 {
            let before = shell_sums(&spins, lattice, true, false, false);
            let result = wolff.sweep(&mut spins, lattice, 1.0, 0.0, 0.0, 0.0, 0.5, &mut rng);
            let after = shell_sums(&spins, lattice, true, false, false);
            let delta = result.delta.expect("Wolff tracks its delta");
            assert_eq!(after.nn, before.nn + delta.nn, "{label} step {step}: nn");
            assert_eq!(
                after.magnetization,
                before.magnetization + delta.magnetization,
                "{label} step {step}: magnetization"
            );
            assert_eq!((delta.nnn, delta.tnn), (0, 0), "{label}: unread shells");
        }
    }

    #[test]
    fn test_wolff_delta_is_off_by_default_and_stream_independent() {
        // Tracking changes the report, never the sampling.
        let lattice = SquareLattice::new(8).unwrap();
        let n = lattice.num_sites();
        let mut rng = create_rng(5);
        let spins0: Vec<i8> = (0..n)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();
        let (mut spins_a, mut spins_b) = (spins0.clone(), spins0);
        let (mut rng_a, mut rng_b) = (create_rng(9), create_rng(9));
        let (mut plain, mut tracked) = (Wolff::new(n), Wolff::new(n));
        tracked.set_track_delta(true);
        for _ in 0..10 {
            let a = plain.sweep(&mut spins_a, &lattice, 1.0, 0.0, 0.0, 0.0, 0.5, &mut rng_a);
            let b = tracked.sweep(&mut spins_b, &lattice, 1.0, 0.0, 0.0, 0.0, 0.5, &mut rng_b);
            assert!(a.delta.is_none());
            assert!(b.delta.is_some());
            assert_eq!(a.accepted, b.accepted);
            assert_eq!(spins_a, spins_b);
        }
    }

    #[test]
    fn test_wolff_delta_matches_shell_sums() {
        use crate::lattice::chain::ChainLattice;
        use crate::lattice::honeycomb::HoneycombLattice;
        use crate::lattice::triangular::TriangularLattice;
        assert_delta_matches_shell_sums(&SquareLattice::new(8).unwrap(), "square");
        assert_delta_matches_shell_sums(&ChainLattice::new(32).unwrap(), "chain");
        assert_delta_matches_shell_sums(&TriangularLattice::new(8).unwrap(), "triangular");
        assert_delta_matches_shell_sums(&HoneycombLattice::new(6).unwrap(), "honeycomb");
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
