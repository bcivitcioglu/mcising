use super::{McAlgorithm, SweepResult};
use crate::lattice::Lattice;
use rand::Rng;

/// Metropolis single-spin-flip algorithm.
///
/// At each step, a site is visited and flipped with probability
/// min(1, exp(-beta * dE)), where dE is the energy change from the flip.
///
/// Uses a sequential scan (systematic Metropolis) for cache locality.
/// For the common J2=0, h=0 case, a fast path uses integer arithmetic
/// and a precomputed exp lookup table.
pub struct Metropolis {
    /// Precomputed acceptance probabilities for dE > 0 in the NN-only case.
    /// Index 0 -> dE=4, index 1 -> dE=8. Only positive dE needs the table
    /// (dE <= 0 always accepts).
    exp_table: [f64; 2],
    /// The beta*j1 value for which exp_table was computed.
    cached_beta_j1: f64,
}

impl Metropolis {
    pub fn new() -> Self {
        Self {
            exp_table: [0.0; 2],
            cached_beta_j1: f64::NAN,
        }
    }

    /// Rebuild the exp lookup table if beta*j1 has changed.
    #[inline]
    fn ensure_exp_table(&mut self, beta: f64, j1: f64) {
        let beta_j1 = beta * j1;
        if beta_j1 != self.cached_beta_j1 {
            // For NN-only: dE = 2 * s * j1 * sum_nn, where sum_nn in {-4,-2,0,2,4}
            // Positive dE values (that need exp check): 4*j1 and 8*j1
            // But we factor out j1 by using beta*j1 in the exponent.
            // dE_eff = 2 * s * sum_nn (integer), actual dE = j1 * dE_eff
            // exp(-beta * dE) = exp(-beta * j1 * dE_eff)
            // Positive dE_eff values: 4 and 8
            self.exp_table[0] = (-beta_j1 * 4.0).exp();
            self.exp_table[1] = (-beta_j1 * 8.0).exp();
            self.cached_beta_j1 = beta_j1;
        }
    }
}

impl McAlgorithm for Metropolis {
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
        let n = lattice.num_sites();
        let mut accepted = 0;

        if j2 == 0.0 && h == 0.0 {
            // Fast path: NN-only, integer arithmetic, exp lookup table,
            // sequential sweep for cache locality.
            self.ensure_exp_table(beta, j1);
            let exp_table = &self.exp_table;

            for idx in 0..n {
                let spin = spins[idx] as i32;
                let nn = lattice.nearest_neighbors(idx);

                let sum_nn = spins[nn[0]] as i32
                    + spins[nn[1]] as i32
                    + spins[nn[2]] as i32
                    + spins[nn[3]] as i32;

                // dE_eff = 2 * spin * sum_nn, in {-8,-4,0,4,8}
                let de = 2 * spin * sum_nn;

                let accept = if de <= 0 {
                    true
                } else {
                    // de is 4 or 8; map to table index: 4->0, 8->1
                    rng.gen::<f64>() < exp_table[(de >> 3) as usize]
                };

                if accept {
                    spins[idx] = -spins[idx];
                    accepted += 1;
                }
            }
        } else {
            // General path: supports J2, h, full f64 arithmetic.
            for idx in 0..n {
                let spin = f64::from(spins[idx]);

                let mut local_field: f64 = 0.0;

                for &nbr in lattice.nearest_neighbors(idx) {
                    local_field += j1 * f64::from(spins[nbr]);
                }

                for &nbr in lattice.next_nearest_neighbors(idx) {
                    local_field += j2 * f64::from(spins[nbr]);
                }

                local_field += h;

                let delta_e = 2.0 * spin * local_field;

                if delta_e <= 0.0 || rng.gen::<f64>() < (-beta * delta_e).exp() {
                    spins[idx] = -spins[idx];
                    accepted += 1;
                }
            }
        }

        SweepResult {
            accepted,
            attempted: n,
        }
    }

    fn name(&self) -> &'static str {
        "Metropolis"
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
    fn test_metropolis_name() {
        assert_eq!(Metropolis::new().name(), "Metropolis");
    }

    #[test]
    fn test_sweep_returns_valid_result() {
        let lattice = SquareLattice::new(4).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);
        let result = Metropolis::new().sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, 1.0, &mut rng);
        assert_eq!(result.attempted, 16);
        assert!(result.accepted <= result.attempted);
    }

    #[test]
    fn test_zero_temperature_only_lowers_energy() {
        // At beta = infinity (T=0), only energy-lowering moves are accepted.
        // Start with random config, energy should only decrease or stay same.
        let lattice = SquareLattice::new(8).unwrap();
        let mut rng = create_rng(42);
        let mut spins: Vec<i8> = (0..lattice.num_sites())
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();

        let energy_before = compute_energy(&spins, &lattice, 1.0, 0.0, 0.0);
        let beta_inf = 1e10; // Approximate T=0

        for _ in 0..100 {
            Metropolis::new().sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, beta_inf, &mut rng);
        }

        let energy_after = compute_energy(&spins, &lattice, 1.0, 0.0, 0.0);
        assert!(
            energy_after <= energy_before + 1e-10,
            "Energy should not increase at T=0: before={energy_before}, after={energy_after}"
        );
    }

    #[test]
    fn test_high_temperature_high_acceptance() {
        // At very high temperature (small beta), almost all moves are accepted.
        let lattice = SquareLattice::new(8).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);
        let beta_small = 0.001; // T = 1000

        let result = Metropolis::new().sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, beta_small, &mut rng);
        // At high T, acceptance rate should be very high (> 50%)
        assert!(
            result.acceptance_rate() > 0.5,
            "Acceptance rate at high T should be > 50%, got {}",
            result.acceptance_rate()
        );
    }

    #[test]
    fn test_all_up_ground_state_stable_at_low_t() {
        // All-up configuration is a ground state for ferromagnetic J1>0.
        // At low T, it should remain mostly all-up.
        let lattice = SquareLattice::new(8).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);
        let beta_large = 10.0; // T = 0.1

        for _ in 0..10 {
            Metropolis::new().sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, beta_large, &mut rng);
        }

        let mag: f64 = spins.iter().map(|&s| f64::from(s)).sum::<f64>() / spins.len() as f64;
        assert!(
            mag > 0.9,
            "All-up ground state should remain magnetized at low T, got m={mag}"
        );
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let lattice = SquareLattice::new(4).unwrap();
        let mut spins1 = all_up_spins(lattice.num_sites());
        let mut spins2 = all_up_spins(lattice.num_sites());
        let mut rng1 = create_rng(123);
        let mut rng2 = create_rng(123);

        Metropolis::new().sweep(&mut spins1, &lattice, 1.0, 0.5, 0.0, 0.5, &mut rng1);
        Metropolis::new().sweep(&mut spins2, &lattice, 1.0, 0.5, 0.0, 0.5, &mut rng2);

        assert_eq!(spins1, spins2);
    }

    #[test]
    fn test_deterministic_fast_path() {
        let lattice = SquareLattice::new(4).unwrap();
        let mut spins1 = all_up_spins(lattice.num_sites());
        let mut spins2 = all_up_spins(lattice.num_sites());
        let mut rng1 = create_rng(123);
        let mut rng2 = create_rng(123);

        // Fast path (j2=0, h=0)
        Metropolis::new().sweep(&mut spins1, &lattice, 1.0, 0.0, 0.0, 0.5, &mut rng1);
        Metropolis::new().sweep(&mut spins2, &lattice, 1.0, 0.0, 0.0, 0.5, &mut rng2);
        assert_eq!(spins1, spins2);
    }

    /// Helper: compute total energy per site.
    fn compute_energy(spins: &[i8], lattice: &SquareLattice, j1: f64, j2: f64, h: f64) -> f64 {
        crate::observables::energy_per_site(spins, lattice, j1, j2, h)
    }
}
