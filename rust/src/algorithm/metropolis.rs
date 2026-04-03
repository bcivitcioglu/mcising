use super::{McAlgorithm, SweepResult};
use crate::lattice::Lattice;
use rand::Rng;

/// Sweep strategy, selected once at construction based on which Hamiltonian
/// terms are active (nonzero). Named by active terms: J1, J2, H.
/// Each variant uses the smallest possible exp lookup table and skips
/// unnecessary neighbor reads.
#[derive(Clone, Copy, Debug)]
enum SweepStrategy {
    /// J1 only: 2-entry table, integer dE, sign-branch skips RNG call.
    J1,
    /// J1 + field: 10-entry [spin × sum_nn], branchless.
    J1H,
    /// J1 + J2: 25-entry [sum_nn × sum_nnn], spin symmetry trick.
    J1J2,
    /// J1 + J2 + field: 50-entry [spin × sum_nn × sum_nnn], branchless.
    J1J2H,
    /// Field only: 2-entry [spin], no neighbor reads.
    H,
    /// J2 only: 2-entry table, integer dE on NNN. Mirror of J1.
    J2,
    /// J2 + field: 10-entry [spin × sum_nnn], branchless. Mirror of J1H.
    J2H,
}

/// Metropolis single-spin-flip algorithm.
///
/// Uses a sequential scan (systematic Metropolis) for cache locality.
/// The sweep strategy is selected once at construction and dispatched
/// via a single match per sweep call (not per flip).
pub struct Metropolis {
    strategy: SweepStrategy,
    /// 2-entry table for J1, J2, or H strategies.
    table_2: [f64; 2],
    cached_key_2: f64,
    /// 10-entry table for J1H or J2H strategies.
    table_10: [f64; 10],
    cached_key_10: (f64, f64, f64),
    /// 25-entry table for J1J2 strategy (spin symmetry).
    table_25: [f64; 25],
    cached_key_25: (f64, f64, f64),
    /// 50-entry table for J1J2H strategy.
    table_50: [f64; 50],
    cached_key_50: (f64, f64, f64, f64),
}

impl Metropolis {
    pub fn new(j1: f64, j2: f64, h: f64) -> Self {
        let strategy = match (j1 != 0.0, j2 != 0.0, h != 0.0) {
            (true, true, true) => SweepStrategy::J1J2H,
            (true, true, false) => SweepStrategy::J1J2,
            (true, false, true) => SweepStrategy::J1H,
            (true, false, false) => SweepStrategy::J1,
            (false, true, true) => SweepStrategy::J2H,
            (false, true, false) => SweepStrategy::J2,
            (false, false, true) => SweepStrategy::H,
            (false, false, false) => SweepStrategy::J1, // degenerate: no interactions
        };

        Self {
            strategy,
            table_2: [0.0; 2],
            cached_key_2: f64::NAN,
            table_10: [0.0; 10],
            cached_key_10: (f64::NAN, f64::NAN, f64::NAN),
            table_25: [0.0; 25],
            cached_key_25: (f64::NAN, f64::NAN, f64::NAN),
            table_50: [0.0; 50],
            cached_key_50: (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        }
    }

    // ── Table ensure methods ──────────────────────────────────────────

    /// 2-entry table for J1 or J2 strategy (integer dE trick).
    /// Positive dE_eff values: 4*coupling and 8*coupling.
    #[inline]
    fn ensure_table_2_coupling(&mut self, beta: f64, coupling: f64) {
        let key = beta * coupling;
        if key != self.cached_key_2 {
            self.table_2[0] = (-key * 4.0).exp();
            self.table_2[1] = (-key * 8.0).exp();
            self.cached_key_2 = key;
        }
    }

    /// 2-entry table for H strategy (field only, indexed by spin).
    /// table[0] = min(1, exp(2*beta*h)) for spin=-1
    /// table[1] = min(1, exp(-2*beta*h)) for spin=+1
    #[inline]
    fn ensure_table_2_field(&mut self, beta: f64, h: f64) {
        let key = beta * h;
        if key != self.cached_key_2 {
            self.table_2[0] = (2.0 * key).exp().min(1.0);
            self.table_2[1] = (-2.0 * key).exp().min(1.0);
            self.cached_key_2 = key;
        }
    }

    /// 10-entry table for J1H or J2H strategy.
    /// table[spin_idx*5 + sum_idx] = min(1, exp(-beta * 2 * spin * (coupling*sum + h)))
    #[inline]
    fn ensure_table_10(&mut self, beta: f64, coupling: f64, h: f64) {
        let key = (beta, coupling, h);
        if key != self.cached_key_10 {
            for (spin_idx, spin) in [(-1.0_f64), 1.0].iter().enumerate() {
                for i in 0..5_usize {
                    let sum = (i as f64 - 2.0) * 2.0;
                    let de = 2.0 * spin * (coupling * sum + h);
                    self.table_10[spin_idx * 5 + i] = (-beta * de).exp().min(1.0);
                }
            }
            self.cached_key_10 = key;
        }
    }

    /// 25-entry table for J1J2 strategy (spin symmetry).
    /// Precomputed for spin=+1; spin=-1 uses table[24 - idx].
    #[inline]
    fn ensure_table_25(&mut self, beta: f64, j1: f64, j2: f64) {
        let key = (beta, j1, j2);
        if key != self.cached_key_25 {
            for i_nn in 0..5_usize {
                let sum_nn = (i_nn as f64 - 2.0) * 2.0;
                for i_nnn in 0..5_usize {
                    let sum_nnn = (i_nnn as f64 - 2.0) * 2.0;
                    let de = 2.0 * (j1 * sum_nn + j2 * sum_nnn);
                    self.table_25[i_nn * 5 + i_nnn] = (-beta * de).exp().min(1.0);
                }
            }
            self.cached_key_25 = key;
        }
    }

    /// 50-entry table for J1J2H strategy.
    /// table[spin_idx*25 + nn_idx*5 + nnn_idx]
    #[inline]
    fn ensure_table_50(&mut self, beta: f64, j1: f64, j2: f64, h: f64) {
        let key = (beta, j1, j2, h);
        if key != self.cached_key_50 {
            for (spin_idx, spin) in [(-1.0_f64), 1.0].iter().enumerate() {
                for i_nn in 0..5_usize {
                    let sum_nn = (i_nn as f64 - 2.0) * 2.0;
                    for i_nnn in 0..5_usize {
                        let sum_nnn = (i_nnn as f64 - 2.0) * 2.0;
                        let de = 2.0 * spin * (j1 * sum_nn + j2 * sum_nnn + h);
                        self.table_50[spin_idx * 25 + i_nn * 5 + i_nnn] =
                            (-beta * de).exp().min(1.0);
                    }
                }
            }
            self.cached_key_50 = key;
        }
    }

    // ── Sweep methods ─────────────────────────────────────────────────

    /// J1 only (J2=0, h=0): 2-entry table, integer dE, sign branch.
    #[inline]
    fn sweep_j1<L: Lattice, R: Rng>(
        &mut self,
        spins: &mut [i8],
        lattice: &L,
        j1: f64,
        beta: f64,
        rng: &mut R,
    ) -> SweepResult {
        self.ensure_table_2_coupling(beta, j1);
        let table = &self.table_2;
        let n = lattice.num_sites();
        let mut accepted = 0;

        for idx in 0..n {
            let spin = spins[idx] as i32;
            let nn = lattice.nearest_neighbors(idx);
            let sum_nn = spins[nn[0]] as i32
                + spins[nn[1]] as i32
                + spins[nn[2]] as i32
                + spins[nn[3]] as i32;

            let de = 2 * spin * sum_nn;
            let accept = if de <= 0 {
                true
            } else {
                rng.gen::<f64>() < table[(de >> 3) as usize]
            };

            if accept {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }

        SweepResult {
            accepted,
            attempted: n,
        }
    }

    /// J2 only (J1=0, h=0): 2-entry table, integer dE on NNN. Mirror of J1.
    #[inline]
    fn sweep_j2<L: Lattice, R: Rng>(
        &mut self,
        spins: &mut [i8],
        lattice: &L,
        j2: f64,
        beta: f64,
        rng: &mut R,
    ) -> SweepResult {
        self.ensure_table_2_coupling(beta, j2);
        let table = &self.table_2;
        let n = lattice.num_sites();
        let mut accepted = 0;

        for idx in 0..n {
            let spin = spins[idx] as i32;
            let nnn = lattice.next_nearest_neighbors(idx);
            let sum_nnn = spins[nnn[0]] as i32
                + spins[nnn[1]] as i32
                + spins[nnn[2]] as i32
                + spins[nnn[3]] as i32;

            let de = 2 * spin * sum_nnn;
            let accept = if de <= 0 {
                true
            } else {
                rng.gen::<f64>() < table[(de >> 3) as usize]
            };

            if accept {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }

        SweepResult {
            accepted,
            attempted: n,
        }
    }

    /// H only (J1=0, J2=0): 2-entry table indexed by spin, no neighbor reads.
    #[inline]
    fn sweep_h<L: Lattice, R: Rng>(
        &mut self,
        spins: &mut [i8],
        lattice: &L,
        h: f64,
        beta: f64,
        rng: &mut R,
    ) -> SweepResult {
        self.ensure_table_2_field(beta, h);
        let table = &self.table_2;
        let n = lattice.num_sites();
        let mut accepted = 0;

        for idx in 0..n {
            let spin_idx = ((spins[idx] as i32 + 1) >> 1) as usize;
            if rng.gen::<f64>() < table[spin_idx] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }

        SweepResult {
            accepted,
            attempted: n,
        }
    }

    /// J1 + field (J2=0): 10-entry [spin × sum_nn], branchless, NN reads only.
    #[inline]
    fn sweep_j1h<L: Lattice, R: Rng>(
        &mut self,
        spins: &mut [i8],
        lattice: &L,
        j1: f64,
        h: f64,
        beta: f64,
        rng: &mut R,
    ) -> SweepResult {
        self.ensure_table_10(beta, j1, h);
        let table = &self.table_10;
        let n = lattice.num_sites();
        let mut accepted = 0;

        for idx in 0..n {
            let spin = spins[idx] as i32;
            let nn = lattice.nearest_neighbors(idx);
            let sum_nn = spins[nn[0]] as i32
                + spins[nn[1]] as i32
                + spins[nn[2]] as i32
                + spins[nn[3]] as i32;

            let spin_idx = ((spin + 1) >> 1) as usize;
            let nn_idx = ((sum_nn >> 1) + 2) as usize;
            let table_idx = spin_idx * 5 + nn_idx;

            if rng.gen::<f64>() < table[table_idx] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }

        SweepResult {
            accepted,
            attempted: n,
        }
    }

    /// J2 + field (J1=0): 10-entry [spin × sum_nnn], branchless, NNN reads only.
    #[inline]
    fn sweep_j2h<L: Lattice, R: Rng>(
        &mut self,
        spins: &mut [i8],
        lattice: &L,
        j2: f64,
        h: f64,
        beta: f64,
        rng: &mut R,
    ) -> SweepResult {
        self.ensure_table_10(beta, j2, h);
        let table = &self.table_10;
        let n = lattice.num_sites();
        let mut accepted = 0;

        for idx in 0..n {
            let spin = spins[idx] as i32;
            let nnn = lattice.next_nearest_neighbors(idx);
            let sum_nnn = spins[nnn[0]] as i32
                + spins[nnn[1]] as i32
                + spins[nnn[2]] as i32
                + spins[nnn[3]] as i32;

            let spin_idx = ((spin + 1) >> 1) as usize;
            let nnn_idx = ((sum_nnn >> 1) + 2) as usize;
            let table_idx = spin_idx * 5 + nnn_idx;

            if rng.gen::<f64>() < table[table_idx] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }

        SweepResult {
            accepted,
            attempted: n,
        }
    }

    /// J1 + J2 (h=0): 25-entry [nn × nnn], spin symmetry trick.
    #[inline]
    fn sweep_j1j2<L: Lattice, R: Rng>(
        &mut self,
        spins: &mut [i8],
        lattice: &L,
        j1: f64,
        j2: f64,
        beta: f64,
        rng: &mut R,
    ) -> SweepResult {
        self.ensure_table_25(beta, j1, j2);
        let table = &self.table_25;
        let n = lattice.num_sites();
        let mut accepted = 0;

        for idx in 0..n {
            let spin = spins[idx] as i32;
            let nn = lattice.nearest_neighbors(idx);
            let nnn = lattice.next_nearest_neighbors(idx);

            let sum_nn = spins[nn[0]] as i32
                + spins[nn[1]] as i32
                + spins[nn[2]] as i32
                + spins[nn[3]] as i32;
            let sum_nnn = spins[nnn[0]] as i32
                + spins[nnn[1]] as i32
                + spins[nnn[2]] as i32
                + spins[nnn[3]] as i32;

            let nn_idx = ((sum_nn >> 1) + 2) as usize;
            let nnn_idx = ((sum_nnn >> 1) + 2) as usize;
            let base_idx = nn_idx * 5 + nnn_idx;

            // Table precomputed for spin=+1.
            // For spin=-1, dE flips sign → index 24 - base_idx.
            let table_idx = if spin > 0 { base_idx } else { 24 - base_idx };

            if rng.gen::<f64>() < table[table_idx] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }

        SweepResult {
            accepted,
            attempted: n,
        }
    }

    /// J1 + J2 + field: 50-entry [spin × nn × nnn], branchless.
    #[inline]
    fn sweep_j1j2h<L: Lattice, R: Rng>(
        &mut self,
        spins: &mut [i8],
        lattice: &L,
        j1: f64,
        j2: f64,
        h: f64,
        beta: f64,
        rng: &mut R,
    ) -> SweepResult {
        self.ensure_table_50(beta, j1, j2, h);
        let table = &self.table_50;
        let n = lattice.num_sites();
        let mut accepted = 0;

        for idx in 0..n {
            let spin = spins[idx] as i32;
            let nn = lattice.nearest_neighbors(idx);
            let nnn = lattice.next_nearest_neighbors(idx);

            let sum_nn = spins[nn[0]] as i32
                + spins[nn[1]] as i32
                + spins[nn[2]] as i32
                + spins[nn[3]] as i32;
            let sum_nnn = spins[nnn[0]] as i32
                + spins[nnn[1]] as i32
                + spins[nnn[2]] as i32
                + spins[nnn[3]] as i32;

            let spin_idx = ((spin + 1) >> 1) as usize;
            let nn_idx = ((sum_nn >> 1) + 2) as usize;
            let nnn_idx = ((sum_nnn >> 1) + 2) as usize;
            let table_idx = spin_idx * 25 + nn_idx * 5 + nnn_idx;

            if rng.gen::<f64>() < table[table_idx] {
                spins[idx] = -spins[idx];
                accepted += 1;
            }
        }

        SweepResult {
            accepted,
            attempted: n,
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
        match self.strategy {
            SweepStrategy::J1 => self.sweep_j1(spins, lattice, j1, beta, rng),
            SweepStrategy::J1H => self.sweep_j1h(spins, lattice, j1, h, beta, rng),
            SweepStrategy::J1J2 => self.sweep_j1j2(spins, lattice, j1, j2, beta, rng),
            SweepStrategy::J1J2H => self.sweep_j1j2h(spins, lattice, j1, j2, h, beta, rng),
            SweepStrategy::H => self.sweep_h(spins, lattice, h, beta, rng),
            SweepStrategy::J2 => self.sweep_j2(spins, lattice, j2, beta, rng),
            SweepStrategy::J2H => self.sweep_j2h(spins, lattice, j2, h, beta, rng),
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

    fn random_spins(lattice: &SquareLattice, rng: &mut impl Rng) -> Vec<i8> {
        (0..lattice.num_sites())
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect()
    }

    fn compute_energy(spins: &[i8], lattice: &SquareLattice, j1: f64, j2: f64, h: f64) -> f64 {
        crate::observables::energy_per_site(spins, lattice, j1, j2, h)
    }

    // ── Basic tests ───────────────────────────────────────────────────

    #[test]
    fn test_metropolis_name() {
        assert_eq!(Metropolis::new(1.0, 0.0, 0.0).name(), "Metropolis");
    }

    #[test]
    fn test_sweep_returns_valid_result() {
        let lattice = SquareLattice::new(4).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);
        let result =
            Metropolis::new(1.0, 0.0, 0.0).sweep(&mut spins, &lattice, 1.0, 0.0, 0.0, 1.0, &mut rng);
        assert_eq!(result.attempted, 16);
        assert!(result.accepted <= result.attempted);
    }

    // ── Determinism tests (one per strategy) ──────────────────────────

    fn assert_deterministic(j1: f64, j2: f64, h: f64) {
        let lattice = SquareLattice::new(4).unwrap();
        let mut spins1 = all_up_spins(lattice.num_sites());
        let mut spins2 = all_up_spins(lattice.num_sites());
        let mut rng1 = create_rng(123);
        let mut rng2 = create_rng(123);

        Metropolis::new(j1, j2, h).sweep(&mut spins1, &lattice, j1, j2, h, 0.5, &mut rng1);
        Metropolis::new(j1, j2, h).sweep(&mut spins2, &lattice, j1, j2, h, 0.5, &mut rng2);
        assert_eq!(spins1, spins2);
    }

    #[test]
    fn test_deterministic_j1() {
        assert_deterministic(1.0, 0.0, 0.0);
    }

    #[test]
    fn test_deterministic_j1h() {
        assert_deterministic(1.0, 0.0, 0.5);
    }

    #[test]
    fn test_deterministic_j1j2() {
        assert_deterministic(1.0, 0.3, 0.0);
    }

    #[test]
    fn test_deterministic_j1j2h() {
        assert_deterministic(1.0, 0.3, 0.5);
    }

    #[test]
    fn test_deterministic_h() {
        assert_deterministic(0.0, 0.0, 0.5);
    }

    #[test]
    fn test_deterministic_j2() {
        assert_deterministic(0.0, 0.5, 0.0);
    }

    #[test]
    fn test_deterministic_j2h() {
        assert_deterministic(0.0, 0.5, 0.3);
    }

    // ── Physics tests ─────────────────────────────────────────────────

    #[test]
    fn test_zero_temperature_only_lowers_energy() {
        let lattice = SquareLattice::new(8).unwrap();
        let mut rng = create_rng(42);
        let mut spins = random_spins(&lattice, &mut rng);

        let energy_before = compute_energy(&spins, &lattice, 1.0, 0.0, 0.0);
        let beta_inf = 1e10;

        for _ in 0..100 {
            Metropolis::new(1.0, 0.0, 0.0).sweep(
                &mut spins, &lattice, 1.0, 0.0, 0.0, beta_inf, &mut rng,
            );
        }

        let energy_after = compute_energy(&spins, &lattice, 1.0, 0.0, 0.0);
        assert!(
            energy_after <= energy_before + 1e-10,
            "Energy should not increase at T=0: before={energy_before}, after={energy_after}"
        );
    }

    #[test]
    fn test_high_temperature_high_acceptance() {
        let lattice = SquareLattice::new(8).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);

        let result = Metropolis::new(1.0, 0.0, 0.0).sweep(
            &mut spins, &lattice, 1.0, 0.0, 0.0, 0.001, &mut rng,
        );
        assert!(
            result.acceptance_rate() > 0.5,
            "Acceptance rate at high T should be > 50%, got {}",
            result.acceptance_rate()
        );
    }

    #[test]
    fn test_all_up_ground_state_stable_at_low_t() {
        let lattice = SquareLattice::new(8).unwrap();
        let mut spins = all_up_spins(lattice.num_sites());
        let mut rng = create_rng(42);

        for _ in 0..10 {
            Metropolis::new(1.0, 0.0, 0.0).sweep(
                &mut spins, &lattice, 1.0, 0.0, 0.0, 10.0, &mut rng,
            );
        }

        let mag: f64 = spins.iter().map(|&s| f64::from(s)).sum::<f64>() / spins.len() as f64;
        assert!(
            mag > 0.9,
            "All-up ground state should remain magnetized at low T, got m={mag}"
        );
    }

    fn assert_energy_decreases_at_zero_t(j1: f64, j2: f64, h: f64) {
        let lattice = SquareLattice::new(8).unwrap();
        let mut rng = create_rng(42);
        let mut spins = random_spins(&lattice, &mut rng);

        let energy_before = compute_energy(&spins, &lattice, j1, j2, h);
        let beta_inf = 1e10;

        for _ in 0..100 {
            Metropolis::new(j1, j2, h).sweep(
                &mut spins, &lattice, j1, j2, h, beta_inf, &mut rng,
            );
        }

        let energy_after = compute_energy(&spins, &lattice, j1, j2, h);
        assert!(
            energy_after <= energy_before + 1e-10,
            "Energy should not increase at T=0: before={energy_before}, after={energy_after}"
        );
    }

    #[test]
    fn test_energy_decreases_j1j2() {
        assert_energy_decreases_at_zero_t(1.0, 0.5, 0.0);
    }

    #[test]
    fn test_energy_decreases_j1h() {
        assert_energy_decreases_at_zero_t(1.0, 0.0, 0.5);
    }

    #[test]
    fn test_energy_decreases_j2_only() {
        assert_energy_decreases_at_zero_t(0.0, 1.0, 0.0);
    }

    #[test]
    fn test_energy_decreases_j2h() {
        assert_energy_decreases_at_zero_t(0.0, 0.5, 0.3);
    }

    #[test]
    fn test_h_only_aligns_spins() {
        // With only a positive field, all spins should align to +1 at low T.
        let lattice = SquareLattice::new(8).unwrap();
        let mut rng = create_rng(42);
        let mut spins = random_spins(&lattice, &mut rng);
        let beta = 100.0;

        for _ in 0..200 {
            Metropolis::new(0.0, 0.0, 1.0).sweep(
                &mut spins, &lattice, 0.0, 0.0, 1.0, beta, &mut rng,
            );
        }

        let mag: f64 = spins.iter().map(|&s| f64::from(s)).sum::<f64>() / spins.len() as f64;
        assert!(
            mag > 0.99,
            "Field-only at low T should align all spins: got m={mag}"
        );
    }
}