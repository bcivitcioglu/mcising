use super::{McAlgorithm, SweepResult};
use crate::lattice::Lattice;
use rand::Rng;

/// Metropolis single-spin-flip algorithm.
///
/// At each step, a random site is selected and flipped with probability
/// min(1, exp(-beta * dE)), where dE is the energy change from the flip.
///
/// The energy change is computed efficiently as dE = 2 * spin * local_field,
/// avoiding a full energy calculation before and after the flip.
pub struct Metropolis;

impl McAlgorithm for Metropolis {
    fn sweep<L: Lattice, R: Rng>(
        &self,
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

        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            let spin = f64::from(spins[idx]);

            // Compute local field from neighbors
            let mut local_field: f64 = 0.0;

            for &nbr in lattice.nearest_neighbors(idx) {
                local_field += j1 * f64::from(spins[nbr]);
            }

            for &nbr in lattice.next_nearest_neighbors(idx) {
                local_field += j2 * f64::from(spins[nbr]);
            }

            local_field += h;

            // Energy change for flipping: dE = 2 * spin * local_field
            let delta_e = 2.0 * spin * local_field;

            if delta_e <= 0.0 || rng.gen::<f64>() < (-beta * delta_e).exp() {
                spins[idx] = -spins[idx];
                accepted += 1;
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
        assert_eq!(Metropolis.name(), "Metropolis");
    }

    #[test]
    fn test_sweep_returns_valid_result() {
        let lattice = SquareLattice::new(4).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);
        let result = Metropolis.sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, 1.0, &mut rng);
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
            Metropolis.sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, beta_inf, &mut rng);
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

        let result = Metropolis.sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, beta_small, &mut rng);
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
            Metropolis.sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, beta_large, &mut rng);
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

        Metropolis.sweep(&mut spins1, &lattice, 1.0, 0.5, 0.0, 0.5, &mut rng1);
        Metropolis.sweep(&mut spins2, &lattice, 1.0, 0.5, 0.0, 0.5, &mut rng2);

        assert_eq!(spins1, spins2);
    }

    /// Helper: compute total energy per site.
    fn compute_energy(spins: &[i8], lattice: &SquareLattice, j1: f64, j2: f64, h: f64) -> f64 {
        crate::observables::energy_per_site(spins, lattice, j1, j2, h)
    }
}
