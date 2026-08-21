//! Exact-enumeration oracle for small systems (test-only).
//!
//! Enumerates every spin configuration of a small lattice into a joint
//! density of states g(S, M) over the integer nearest-neighbor bond sum
//! S = sum_<ij> s_i s_j (each bond counted once) and the integer
//! magnetization M = sum_i s_i. At h = J2 = J3 = 0 the total energy of a
//! state is E = -J1 * S, so a single enumeration yields exact
//! thermodynamics for any temperature and either sign of J1.
//!
//! Conventions (chosen to match the production observables exactly):
//!
//! ```text
//! N       = lattice.num_sites()
//! e       = E_total / N       (matches observables::energy_per_site)
//! m       = M / N             (matches observables::magnetization_per_site)
//! <m^2>   = <M^2> / N^2
//! Cv/N    = beta^2 * (<E^2> - <E>^2) / N
//!           (matches SimulationResults.specific_heat: N * var(e) / T^2)
//! log_z   = ln Z, extensive (over all 2^N states)
//! ```
//!
//! The oracle is validated against the closed-form transfer matrix for the
//! periodic chain and against high- and zero-temperature limits; the
//! Monte Carlo samplers are then validated against the oracle. Reusable by
//! later phases (P14 physics validation) as `crate::exact_enumeration`.

use crate::algorithm::metropolis::Metropolis;
use crate::algorithm::swendsen_wang::SwendsenWang;
use crate::algorithm::wolff::Wolff;
use crate::algorithm::{AlgorithmKind, McAlgorithm};
use crate::lattice::chain::ChainLattice;
use crate::lattice::square::SquareLattice;
use crate::lattice::Lattice;
use crate::rng::create_rng;
use rand::Rng;
use std::sync::OnceLock;

/// Joint density of states over (bond sum S, magnetization M).
///
/// Dense storage: `counts[(s + num_bonds) * (2N + 1) + (m + N)]`.
pub(crate) struct DensityOfStates {
    num_sites: usize,
    num_bonds: usize,
    counts: Vec<u64>,
}

/// Exact thermodynamics at one temperature, per-site conventions above.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Exact {
    /// ln Z (extensive).
    pub(crate) log_z: f64,
    /// <E>/N.
    pub(crate) energy: f64,
    /// <(M/N)^2>.
    pub(crate) m_squared: f64,
    /// Cv/N = beta^2 (<E^2> - <E>^2) / N.
    pub(crate) specific_heat: f64,
    /// Var(E/N) — sizes the Monte Carlo runs that target `energy`.
    pub(crate) energy_variance: f64,
}

/// Nearest-neighbor bond sum S = sum_<ij> s_i s_j, each bond counted once.
///
/// The neighbor tables list every bond twice (once from each end), hence
/// the division by 2; the sum over both directions is always even.
pub(crate) fn nn_bond_sum<L: Lattice>(spins: &[i8], lattice: &L) -> i32 {
    let mut twice_sum = 0i32;
    for (i, &s) in spins.iter().enumerate() {
        for &nbr in lattice.nearest_neighbors(i) {
            twice_sum += i32::from(s) * i32::from(spins[nbr]);
        }
    }
    twice_sum / 2
}

/// Enumerate all 2^N spin states of `lattice` into a density of states.
///
/// # Panics
///
/// Panics for more than 20 sites (2^N states becomes unreasonable, and the
/// state bitmask is a u32).
pub(crate) fn enumerate_states<L: Lattice>(lattice: &L) -> DensityOfStates {
    let n = lattice.num_sites();
    assert!(n <= 20, "exact enumeration is 2^N states; refusing N={n}");
    let num_bonds = (0..n)
        .map(|i| lattice.nearest_neighbors(i).len())
        .sum::<usize>()
        / 2;

    let width = 2 * n + 1;
    let mut counts = vec![0u64; (2 * num_bonds + 1) * width];
    let mut spins = vec![1i8; n];
    for state in 0u32..(1u32 << n) {
        for (i, s) in spins.iter_mut().enumerate() {
            *s = if (state >> i) & 1 == 1 { 1 } else { -1 };
        }
        let bond_sum = nn_bond_sum(&spins, lattice);
        let m: i32 = spins.iter().map(|&x| i32::from(x)).sum();
        let row = usize::try_from(bond_sum + num_bonds as i32).expect("bond sum in range");
        let col = usize::try_from(m + n as i32).expect("magnetization in range");
        counts[row * width + col] += 1;
    }

    DensityOfStates {
        num_sites: n,
        num_bonds,
        counts,
    }
}

impl DensityOfStates {
    /// Total number of enumerated states (must equal 2^N).
    pub(crate) fn total_states(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Iterate populated cells as (S, M, count).
    fn populated(&self) -> impl Iterator<Item = (i32, i32, u64)> + '_ {
        let width = 2 * self.num_sites + 1;
        let nb = self.num_bonds as i32;
        let n = self.num_sites as i32;
        self.counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(move |(idx, &c)| ((idx / width) as i32 - nb, (idx % width) as i32 - n, c))
    }

    /// Exact thermodynamics at `temperature` for coupling `j1` (h=J2=J3=0).
    ///
    /// Weights are accumulated relative to the ground-state energy
    /// (shift by `E_min` before exponentiating), so low temperatures
    /// underflow gracefully to the ground-state manifold instead of
    /// overflowing.
    pub(crate) fn exact_at(&self, temperature: f64, j1: f64) -> Exact {
        let beta = 1.0 / temperature;
        let n = self.num_sites as f64;

        let e_min = self
            .populated()
            .map(|(s, _, _)| -j1 * f64::from(s))
            .fold(f64::INFINITY, f64::min);

        // Accumulate moments of d = E - E_min (better conditioned than raw E).
        let mut z = 0.0;
        let mut sum_d = 0.0;
        let mut sum_d2 = 0.0;
        let mut sum_m2 = 0.0;
        for (s, m, count) in self.populated() {
            let d = -j1 * f64::from(s) - e_min;
            let w = count as f64 * (-beta * d).exp();
            z += w;
            sum_d += w * d;
            sum_d2 += w * d * d;
            sum_m2 += w * f64::from(m) * f64::from(m);
        }

        let mean_d = sum_d / z;
        let var_e_total = sum_d2 / z - mean_d * mean_d;
        Exact {
            log_z: z.ln() - beta * e_min,
            energy: (e_min + mean_d) / n,
            m_squared: (sum_m2 / z) / (n * n),
            specific_heat: beta * beta * var_e_total / n,
            energy_variance: var_e_total / (n * n),
        }
    }
}

/// Cached density of states for the 4x4 square torus (65 536 states).
pub(crate) fn square4_dos() -> &'static DensityOfStates {
    static DOS: OnceLock<DensityOfStates> = OnceLock::new();
    DOS.get_or_init(|| enumerate_states(&SquareLattice::new(4).expect("4x4 square is valid")))
}

/// Cached density of states for the 12-site periodic chain (4 096 states).
pub(crate) fn chain12_dos() -> &'static DensityOfStates {
    static DOS: OnceLock<DensityOfStates> = OnceLock::new();
    DOS.get_or_init(|| enumerate_states(&ChainLattice::new(12).expect("chain-12 is valid")))
}

/// The three gate temperatures fixed by the roadmap.
pub(crate) const GATE_TEMPERATURES: [f64; 3] = [1.0, 2.269, 4.0];

#[cfg(test)]
mod tests {
    use super::*;

    // ── Oracle self-validation ────────────────────────────────────────

    /// (ln Z, <E>/N) for the periodic Ising chain of `n` sites at h=0,
    /// from the transfer-matrix eigenvalues lambda± = 2cosh(bJ), 2sinh(bJ):
    /// Z = lambda+^n + lambda-^n. Valid for either sign of J (lambda- < 0
    /// enters Z at an even power for even n; the energy formula follows by
    /// -d(ln Z)/d(beta) and holds for odd powers too).
    fn chain_transfer_matrix(n: i32, temperature: f64, j: f64) -> (f64, f64) {
        let beta = 1.0 / temperature;
        let lp = 2.0 * (beta * j).cosh();
        let lm = 2.0 * (beta * j).sinh();
        let z = lp.powi(n) + lm.powi(n);
        let e_per_site = -j * (lp.powi(n - 1) * lm + lm.powi(n - 1) * lp) / z;
        (z.ln(), e_per_site)
    }

    #[test]
    fn test_exact_dos_counts_every_state() {
        assert_eq!(square4_dos().total_states(), 65_536);
        assert_eq!(chain12_dos().total_states(), 4_096);
    }

    #[test]
    fn test_exact_bond_sum_matches_energy_observable() {
        // The bridge that makes the oracle valid for the production energy
        // definition: -J1*S/N must equal observables::energy_per_site for
        // random configurations, both lattices, both coupling signs.
        let square = SquareLattice::new(4).unwrap();
        let chain = ChainLattice::new(12).unwrap();
        let mut rng = create_rng(42);
        for _ in 0..200 {
            let s_spins: Vec<i8> = (0..square.num_sites())
                .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
                .collect();
            let c_spins: Vec<i8> = (0..chain.num_sites())
                .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
                .collect();
            for j1 in [1.0, -1.0] {
                let e_square =
                    -j1 * f64::from(nn_bond_sum(&s_spins, &square)) / square.num_sites() as f64;
                let e_chain =
                    -j1 * f64::from(nn_bond_sum(&c_spins, &chain)) / chain.num_sites() as f64;
                let obs_square =
                    crate::observables::energy_per_site(&s_spins, &square, j1, 0.0, 0.0, 0.0);
                let obs_chain =
                    crate::observables::energy_per_site(&c_spins, &chain, j1, 0.0, 0.0, 0.0);
                assert!((e_square - obs_square).abs() < 1e-12);
                assert!((e_chain - obs_chain).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_exact_chain12_matches_transfer_matrix() {
        for temperature in GATE_TEMPERATURES {
            for j1 in [1.0, -1.0] {
                let ex = chain12_dos().exact_at(temperature, j1);
                let (log_z_tm, e_tm) = chain_transfer_matrix(12, temperature, j1);
                assert!(
                    (ex.log_z - log_z_tm).abs() < 1e-10,
                    "ln Z mismatch at T={temperature}, J1={j1}: \
                     enumeration {} vs transfer matrix {log_z_tm}",
                    ex.log_z
                );
                assert!(
                    (ex.energy - e_tm).abs() < 1e-10,
                    "<E>/N mismatch at T={temperature}, J1={j1}: \
                     enumeration {} vs transfer matrix {e_tm}",
                    ex.energy
                );
            }
        }
    }

    #[test]
    fn test_exact_square4_energy_symmetric_under_coupling_sign() {
        // The 4x4 torus is bipartite: flipping one sublattice maps S -> -S,
        // so <E> and Cv are identical for J1 = +1 and J1 = -1.
        for temperature in GATE_TEMPERATURES {
            let fm = square4_dos().exact_at(temperature, 1.0);
            let afm = square4_dos().exact_at(temperature, -1.0);
            assert!(
                ((fm.energy - afm.energy) / fm.energy).abs() < 1e-12,
                "energy not sign-symmetric at T={temperature}"
            );
            assert!(
                ((fm.specific_heat - afm.specific_heat) / fm.specific_heat).abs() < 1e-12,
                "Cv not sign-symmetric at T={temperature}"
            );
        }
    }

    #[test]
    fn test_exact_square4_m2_distinguishes_coupling_sign() {
        // <m^2> is NOT sign-symmetric: at T=1 the FM is nearly saturated
        // while the AFM (Neel-ordered) has near-zero uniform magnetization.
        // Proves the enumerator is not sign-blind everywhere.
        let fm = square4_dos().exact_at(1.0, 1.0);
        let afm = square4_dos().exact_at(1.0, -1.0);
        assert!(fm.m_squared > 0.99, "FM <m^2> = {}", fm.m_squared);
        assert!(afm.m_squared < 1e-3, "AFM <m^2> = {}", afm.m_squared);
    }

    #[test]
    fn test_exact_high_temperature_limits() {
        // At T -> infinity every state is equiprobable: ln Z -> N ln 2,
        // <E> -> 0, and <M^2> = N (iid spins) so <m^2> = 1/N.
        for (dos, n) in [(square4_dos(), 16.0), (chain12_dos(), 12.0)] {
            for j1 in [1.0, -1.0] {
                let ex = dos.exact_at(1e6, j1);
                assert!((ex.log_z - n * std::f64::consts::LN_2).abs() < 1e-4);
                assert!(ex.energy.abs() < 1e-4);
                assert!((ex.m_squared - 1.0 / n).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_exact_high_temperature_expansion_first_order() {
        // First-order high-temperature expansion: <e> = -(z/2) J1 tanh(bJ1)
        // ~ -(z/2) J1 b. Independently checks the per-site normalization
        // (z = 4 for the square lattice, z = 2 for the chain).
        let temperature = 100.0;
        let beta = 1.0 / temperature;
        for (dos, z) in [(square4_dos(), 4.0), (chain12_dos(), 2.0)] {
            for j1 in [1.0, -1.0] {
                let ex = dos.exact_at(temperature, j1);
                let first_order = -(z / 2.0) * j1 * j1 * beta;
                assert!(
                    ((ex.energy - first_order) / first_order).abs() < 1e-3,
                    "z={z}, J1={j1}: <e> = {} vs first order {first_order}",
                    ex.energy
                );
            }
        }
    }

    #[test]
    fn test_exact_ground_state_limits() {
        // At T=0.01 every excitation weight underflows to exactly zero:
        // <e> = -2|J1| on the square torus (z/2 bonds per site, all
        // satisfied — FM all-up/all-down or AFM Neel pair), Cv -> 0, and
        // exp(ln Z + beta*E_0) recovers the ground-state degeneracy 2.
        let temperature = 0.01;
        let beta = 1.0 / temperature;
        for j1 in [1.0, -1.0] {
            let ex = square4_dos().exact_at(temperature, j1);
            assert!(
                (ex.energy - (-2.0)).abs() < 1e-9,
                "J1={j1}: <e>={}",
                ex.energy
            );
            assert!(ex.specific_heat < 1e-8);
            let e0_total = -2.0 * 16.0;
            let degeneracy = (ex.log_z + beta * e0_total).exp();
            assert!(
                (degeneracy - 2.0).abs() < 1e-9,
                "J1={j1}: ground-state degeneracy {degeneracy}"
            );
        }
    }

    #[test]
    fn test_exact_specific_heat_matches_energy_derivative() {
        // Cv = d<E>/dT: the fluctuation formula must agree with a central
        // difference of the exact energy. Catches any N-factor slip.
        let dt = 1e-4;
        for temperature in GATE_TEMPERATURES {
            for (dos, label) in [(square4_dos(), "square4"), (chain12_dos(), "chain12")] {
                let ex = dos.exact_at(temperature, 1.0);
                let e_plus = dos.exact_at(temperature + dt, 1.0).energy;
                let e_minus = dos.exact_at(temperature - dt, 1.0).energy;
                let cv_diff = (e_plus - e_minus) / (2.0 * dt);
                assert!(
                    ((ex.specific_heat - cv_diff) / cv_diff).abs() < 1e-5,
                    "{label} at T={temperature}: fluctuation Cv/N = {} vs \
                     d<e>/dT = {cv_diff}",
                    ex.specific_heat
                );
            }
        }
    }

    // ── Monte Carlo vs exact ──────────────────────────────────────────
    //
    // Each cell runs `N_SEEDS` independent chains and compares the mean of
    // the per-seed means against the exact value. The standard error over
    // seeds is an honest SE regardless of intra-chain autocorrelation.
    //
    // There is deliberately NO chain-12 Metropolis comparison: the chain
    // never equilibrates under sequential-sweep Metropolis at any beta
    // (pre-existing #26, typewriter domain-wall surfing). The chain oracle
    // is validated against the transfer matrix above and sampled with
    // Wolff below.

    const N_SEEDS: u64 = 10;
    /// The roadmap gate: relative agreement with the exact value.
    const REL_TOL: f64 = 0.005;
    /// Power floor: SE must be <= 0.125% of |exact| so the 0.5% gate sits
    /// at >= 4 sigma. Without this, shortening the runs would make the
    /// gate pass vacuously (same idea as the 0.04 floor in test_wolff.py).
    const SE_REL_MAX: f64 = 0.001_25;

    fn mean_and_se(values: &[f64]) -> (f64, f64) {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, (var / n).sqrt())
    }

    /// One independent chain: random init, `n_therm` discarded sweeps, then
    /// `n_measure` samples spaced `interval` sweeps. Returns per-seed
    /// (<e>, <m^2>). Energy uses the same integer bond-sum path as the
    /// enumerator, which `test_exact_bond_sum_matches_energy_observable`
    /// pins to the production definition.
    fn run_chain<L: Lattice, A: McAlgorithm>(
        algo: &mut A,
        lattice: &L,
        j1: f64,
        beta: f64,
        seed: u64,
        n_therm: usize,
        n_measure: usize,
        interval: usize,
    ) -> (f64, f64) {
        let n = lattice.num_sites();
        let nf = n as f64;
        let mut rng = create_rng(seed);
        let mut spins: Vec<i8> = (0..n)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();
        for _ in 0..n_therm {
            algo.sweep(&mut spins, lattice, j1, 0.0, 0.0, 0.0, beta, &mut rng);
        }
        let mut sum_e = 0.0;
        let mut sum_m2 = 0.0;
        for _ in 0..n_measure {
            for _ in 0..interval {
                algo.sweep(&mut spins, lattice, j1, 0.0, 0.0, 0.0, beta, &mut rng);
            }
            sum_e += -j1 * f64::from(nn_bond_sum(&spins, lattice)) / nf;
            let m: i32 = spins.iter().map(|&x| i32::from(x)).sum();
            sum_m2 += (f64::from(m) / nf).powi(2);
        }
        (sum_e / n_measure as f64, sum_m2 / n_measure as f64)
    }

    /// Per-seed (<e>, <m^2>) means for one (algorithm, lattice, J1, T) cell.
    fn means_over_seeds<L: Lattice>(
        lattice: &L,
        kind: AlgorithmKind,
        j1: f64,
        temperature: f64,
        seed_base: u64,
        n_therm: usize,
        n_measure: usize,
        interval: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let beta = 1.0 / temperature;
        let mut e_means = Vec::new();
        let mut m2_means = Vec::new();
        for offset in 0..N_SEEDS {
            let seed = seed_base + offset;
            let (e, m2) = match kind {
                AlgorithmKind::Metropolis => {
                    let mut algo = Metropolis::new(
                        j1,
                        0.0,
                        0.0,
                        0.0,
                        lattice.coordination_number(),
                        lattice.nnn_coordination_number(),
                        lattice.tnn_coordination_number(),
                    );
                    run_chain(
                        &mut algo, lattice, j1, beta, seed, n_therm, n_measure, interval,
                    )
                }
                AlgorithmKind::Wolff => {
                    let mut algo = Wolff::new(lattice.num_sites());
                    run_chain(
                        &mut algo, lattice, j1, beta, seed, n_therm, n_measure, interval,
                    )
                }
                AlgorithmKind::SwendsenWang => {
                    let mut algo = SwendsenWang::new(lattice.num_sites());
                    run_chain(
                        &mut algo, lattice, j1, beta, seed, n_therm, n_measure, interval,
                    )
                }
            };
            e_means.push(e);
            m2_means.push(m2);
        }
        (e_means, m2_means)
    }

    /// The gate assertion: strictly relative agreement plus the power floor.
    fn assert_within_relative(
        label: &str,
        per_seed_means: &[f64],
        exact: f64,
        rel_tol: f64,
        se_rel_max: f64,
    ) {
        let (mean, se) = mean_and_se(per_seed_means);
        let rel_err = ((mean - exact) / exact).abs();
        let se_rel = (se / exact).abs();
        let sigma_dev = (mean - exact).abs() / se;
        println!(
            "calib {label}: mean={mean:.6} exact={exact:.6} \
             rel_err={rel_err:.2e} se_rel={se_rel:.2e} dev={sigma_dev:.2}sigma"
        );
        assert!(
            se_rel <= se_rel_max,
            "{label}: power floor violated: se/|exact| = {se_rel:.2e} > \
             {se_rel_max:.2e} — lengthen the run instead of trusting an \
             underpowered pass"
        );
        assert!(
            rel_err <= rel_tol,
            "{label}: |mean-exact|/|exact| = {rel_err:.2e} > {rel_tol:.2e} \
             (mean={mean}, exact={exact}, se={se}, {sigma_dev:.1} sigma)"
        );
    }

    /// Per-temperature (n_therm, n_measure, interval, seed_base), sized so
    /// the SE lands under the power floor with margin. The T=4.0 cells
    /// dominate the cost: the gate is relative and |<e>| is smallest there
    /// while sd(e) is not.
    struct Cell {
        temperature: f64,
        n_therm: usize,
        n_measure: usize,
        interval: usize,
        seed_base: u64,
    }

    fn compare_cells(
        label: &str,
        lattice: &impl Lattice,
        dos: &DensityOfStates,
        kind: AlgorithmKind,
        j1: f64,
        cells: &[Cell],
        check_m2: bool,
    ) {
        for cell in cells {
            let ex = dos.exact_at(cell.temperature, j1);
            let (e_means, m2_means) = means_over_seeds(
                lattice,
                kind,
                j1,
                cell.temperature,
                cell.seed_base,
                cell.n_therm,
                cell.n_measure,
                cell.interval,
            );
            println!(
                "calib {label} T={} exact: e={:.6} m2={:.6} sd(e)={:.4}",
                cell.temperature,
                ex.energy,
                ex.m_squared,
                ex.energy_variance.sqrt()
            );
            assert_within_relative(
                &format!("{label} <e> T={}", cell.temperature),
                &e_means,
                ex.energy,
                REL_TOL,
                SE_REL_MAX,
            );
            if check_m2 {
                // <m^2> at 3% relative — cheap, coarse cross-check of a
                // second observable. Not asserted for the AFM square: its
                // exact <m^2> is ~8e-5 at T=1 and a relative tolerance is
                // meaningless there (sign discrimination is covered by
                // test_exact_square4_m2_distinguishes_coupling_sign).
                assert_within_relative(
                    &format!("{label} <m2> T={}", cell.temperature),
                    &m2_means,
                    ex.m_squared,
                    0.03,
                    0.0075,
                );
            }
        }
    }

    #[test]
    fn test_metropolis_matches_exact_ferromagnetic() {
        let lattice = SquareLattice::new(4).unwrap();
        let cells = [
            Cell {
                temperature: 1.0,
                n_therm: 2_000,
                n_measure: 20_000,
                interval: 1,
                seed_base: 7_000,
            },
            Cell {
                temperature: 2.269,
                n_therm: 5_000,
                n_measure: 60_000,
                interval: 1,
                seed_base: 7_100,
            },
            Cell {
                temperature: 4.0,
                n_therm: 2_000,
                n_measure: 180_000,
                interval: 1,
                seed_base: 7_200,
            },
        ];
        compare_cells(
            "metropolis square4 J1=+1",
            &lattice,
            square4_dos(),
            AlgorithmKind::Metropolis,
            1.0,
            &cells,
            true,
        );
    }

    #[test]
    fn test_metropolis_matches_exact_antiferromagnetic() {
        let lattice = SquareLattice::new(4).unwrap();
        let cells = [
            Cell {
                temperature: 1.0,
                n_therm: 2_000,
                n_measure: 20_000,
                interval: 1,
                seed_base: 7_300,
            },
            Cell {
                temperature: 2.269,
                n_therm: 5_000,
                n_measure: 60_000,
                interval: 1,
                seed_base: 7_400,
            },
            Cell {
                temperature: 4.0,
                n_therm: 2_000,
                n_measure: 180_000,
                interval: 1,
                seed_base: 7_500,
            },
        ];
        compare_cells(
            "metropolis square4 J1=-1",
            &lattice,
            square4_dos(),
            AlgorithmKind::Metropolis,
            -1.0,
            &cells,
            false,
        );
    }

    #[test]
    fn test_wolff_matches_exact_ferromagnetic() {
        let lattice = SquareLattice::new(4).unwrap();
        // interval is in cluster moves (one Wolff "sweep" = one cluster);
        // the values approximate one lattice-sweep-equivalent per sample.
        // Fixed spacing is load-bearing: measuring at a flip-budget
        // stopping time is size-biased (P10 oracle rejection, 200+ sigma).
        let cells = [
            Cell {
                temperature: 1.0,
                n_therm: 2_000,
                n_measure: 20_000,
                interval: 4,
                seed_base: 7_600,
            },
            Cell {
                temperature: 2.269,
                n_therm: 2_000,
                n_measure: 40_000,
                interval: 8,
                seed_base: 7_610,
            },
            Cell {
                temperature: 4.0,
                n_therm: 2_000,
                n_measure: 120_000,
                interval: 10,
                seed_base: 7_620,
            },
        ];
        compare_cells(
            "wolff square4 J1=+1",
            &lattice,
            square4_dos(),
            AlgorithmKind::Wolff,
            1.0,
            &cells,
            true,
        );
    }

    #[test]
    fn test_swendsen_wang_matches_exact_ferromagnetic() {
        let lattice = SquareLattice::new(4).unwrap();
        let cells = [
            Cell {
                temperature: 1.0,
                n_therm: 1_000,
                n_measure: 20_000,
                interval: 1,
                seed_base: 7_700,
            },
            Cell {
                temperature: 2.269,
                n_therm: 1_000,
                n_measure: 90_000,
                interval: 1,
                seed_base: 7_710,
            },
            Cell {
                temperature: 4.0,
                n_therm: 1_000,
                n_measure: 220_000,
                interval: 1,
                seed_base: 7_720,
            },
        ];
        compare_cells(
            "swendsen_wang square4 J1=+1",
            &lattice,
            square4_dos(),
            AlgorithmKind::SwendsenWang,
            1.0,
            &cells,
            true,
        );
    }

    #[test]
    fn test_wolff_matches_exact_chain12_ferromagnetic() {
        let lattice = ChainLattice::new(12).unwrap();
        let cells = [
            Cell {
                temperature: 1.0,
                n_therm: 2_000,
                n_measure: 20_000,
                interval: 3,
                seed_base: 7_800,
            },
            Cell {
                temperature: 2.269,
                n_therm: 2_000,
                n_measure: 60_000,
                interval: 6,
                seed_base: 7_810,
            },
            Cell {
                temperature: 4.0,
                n_therm: 2_000,
                n_measure: 150_000,
                interval: 6,
                seed_base: 7_820,
            },
        ];
        compare_cells(
            "wolff chain12 J1=+1",
            &lattice,
            chain12_dos(),
            AlgorithmKind::Wolff,
            1.0,
            &cells,
            true,
        );
    }
}
