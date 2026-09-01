use crate::lattice::Lattice;
use std::collections::BTreeMap;

/// Compute the total energy per site of the spin configuration.
///
/// E/N = (-sum_{<i,j>} J1*si*sj - sum_{<<i,j>>} J2*si*sj
///        - sum_{<<<i,j>>>} J3*si*sj - h*sum_i si) / N
///
/// Interaction terms are divided by 2 to correct for double-counting.
///
/// The reference evaluation is `energy_per_site_serial` (test-only): one
/// serial f64 accumulation over every shell of every site. This entry point
/// returns the same bits by a cheaper route, pinned by `to_bits` tests:
///
/// * when every nonzero coupling is dyadic-exact for this lattice
///   ([`dyadic_exact`]), every partial sum of the serial chain is exactly
///   representable, so the serial result *is* the exact value — which the
///   integer shell sums reproduce in a single pass;
/// * otherwise the serial chain runs with the exactly-zero shells skipped:
///   a `-(±0.0)` term never changes an accumulator that starts at `+0.0`
///   and can never become `-0.0` (`x - x = +0.0`, `+0.0 - (±0.0) = +0.0`).
pub fn energy_per_site<L: Lattice>(
    spins: &[i8],
    lattice: &L,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
) -> f64 {
    let n = lattice.num_sites();
    if dyadic_exact(lattice, j1, j2, j3, h) {
        let sums = shell_sums(spins, lattice, j1 != 0.0, j2 != 0.0, j3 != 0.0);
        // `0.0 - x` (not `-x`) keeps a zero shell at +0.0, as the serial
        // accumulator does; every product and difference below is exact.
        let interaction = 0.0 - j1 * sums.nn as f64 - j2 * sums.nnn as f64 - j3 * sums.tnn as f64;
        let field = 0.0 - h * sums.magnetization as f64;
        return (interaction / 2.0 + field) / n as f64;
    }
    energy_per_site_sparse(spins, lattice, j1, j2, j3, h)
}

/// Ordered-pair spin products per shell, plus the total magnetization.
struct ShellSums {
    nn: i64,
    nnn: i64,
    tnn: i64,
    magnetization: i64,
}

/// Single integer pass over the requested shells (`|sum| <= z * N`).
fn shell_sums<L: Lattice>(spins: &[i8], lattice: &L, nn: bool, nnn: bool, tnn: bool) -> ShellSums {
    let mut sums = ShellSums {
        nn: 0,
        nnn: 0,
        tnn: 0,
        magnetization: 0,
    };
    for (idx, &spin) in spins.iter().enumerate() {
        let spin = i64::from(spin);
        sums.magnetization += spin;
        if nn {
            let s: i64 = lattice
                .nearest_neighbors(idx)
                .iter()
                .map(|&j| i64::from(spins[j]))
                .sum();
            sums.nn += spin * s;
        }
        if nnn {
            let s: i64 = lattice
                .next_nearest_neighbors(idx)
                .iter()
                .map(|&j| i64::from(spins[j]))
                .sum();
            sums.nnn += spin * s;
        }
        if tnn {
            let s: i64 = lattice
                .third_nearest_neighbors(idx)
                .iter()
                .map(|&j| i64::from(spins[j]))
                .sum();
            sums.tnn += spin * s;
        }
    }
    sums
}

/// The serial accumulation with exactly-zero shells skipped (bit-identical
/// to `energy_per_site_serial`, see [`energy_per_site`]).
fn energy_per_site_sparse<L: Lattice>(
    spins: &[i8],
    lattice: &L,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
) -> f64 {
    let n = lattice.num_sites();
    let (use1, use2, use3, use_h) = (j1 != 0.0, j2 != 0.0, j3 != 0.0, h != 0.0);
    let mut interaction = 0.0;
    let mut field = 0.0;

    for idx in 0..n {
        let spin = f64::from(spins[idx]);

        if use1 {
            for &nbr in lattice.nearest_neighbors(idx) {
                interaction -= j1 * spin * f64::from(spins[nbr]);
            }
        }
        if use2 {
            for &nbr in lattice.next_nearest_neighbors(idx) {
                interaction -= j2 * spin * f64::from(spins[nbr]);
            }
        }
        if use3 {
            for &nbr in lattice.third_nearest_neighbors(idx) {
                interaction -= j3 * spin * f64::from(spins[nbr]);
            }
        }
        if use_h {
            field -= h * spin;
        }
    }

    (interaction / 2.0 + field) / n as f64
}

/// The original evaluation, kept verbatim as the bit-level reference for
/// the tests: every shell of every site in one serial f64 chain.
#[cfg(test)]
fn energy_per_site_serial<L: Lattice>(
    spins: &[i8],
    lattice: &L,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
) -> f64 {
    let n = lattice.num_sites();
    let mut interaction = 0.0;
    let mut field = 0.0;

    for idx in 0..n {
        let spin = f64::from(spins[idx]);

        for &nbr in lattice.nearest_neighbors(idx) {
            interaction -= j1 * spin * f64::from(spins[nbr]);
        }
        for &nbr in lattice.next_nearest_neighbors(idx) {
            interaction -= j2 * spin * f64::from(spins[nbr]);
        }
        for &nbr in lattice.third_nearest_neighbors(idx) {
            interaction -= j3 * spin * f64::from(spins[nbr]);
        }
        field -= h * spin;
    }

    // Interaction terms double-counted (each pair counted twice), field is not
    (interaction / 2.0 + field) / n as f64
}

/// Whether the integer-shell evaluation is bit-identical to the serial one.
///
/// Every nonzero coupling `c` is an integer multiple of its ulp `2^(e_c)`.
/// With `e = min e_c`, each partial sum of the serial chain — in any
/// interleaving of shells — is an integer multiple of `2^e` bounded by
/// `B = Σ_k z_k·N·|J_k| + N·|h|`, and any multiple of `2^e` below
/// `2^53·2^e` is representable, so every intermediate rounding is exact and
/// the chain returns the exact value. The bound is compared against `2^51`
/// (a factor-4 margin over the f64 rounding of the check itself); an
/// overflow to `inf` fails the check, which is the safe direction.
/// Subnormal couplings fall back to the serial path.
fn dyadic_exact<L: Lattice>(lattice: &L, j1: f64, j2: f64, j3: f64, h: f64) -> bool {
    let n = lattice.num_sites() as f64;
    let terms = [
        (j1, lattice.coordination_number() as f64 * n),
        (j2, lattice.nnn_coordination_number() as f64 * n),
        (j3, lattice.tnn_coordination_number() as f64 * n),
        (h, n),
    ];
    let mut min_exp: Option<i32> = None;
    for &(c, _) in &terms {
        if c == 0.0 {
            continue;
        }
        match ulp_exponent(c) {
            Some(e) => min_exp = Some(min_exp.map_or(e, |m| m.min(e))),
            None => return false,
        }
    }
    let Some(e) = min_exp else {
        // Every coupling is zero: both paths return exactly +0.0.
        return true;
    };
    let scale = 2f64.powi(-e);
    let bound: f64 = terms
        .iter()
        .filter(|&&(c, _)| c != 0.0)
        .map(|&(c, count)| c.abs() * scale * count)
        .sum();
    bound < 2f64.powi(51)
}

/// Exponent of the ulp of a normal, nonzero f64 (`c = m·2^e` with `m` odd);
/// `None` for zero, subnormal, infinite or NaN inputs.
fn ulp_exponent(c: f64) -> Option<i32> {
    let bits = c.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 || biased == 0x7ff {
        return None;
    }
    let significand = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    Some(biased - 1023 - 52 + significand.trailing_zeros() as i32)
}

/// Compute the magnetization per site.
///
/// M/N = sum_i si / N
pub fn magnetization_per_site(spins: &[i8]) -> f64 {
    let sum: i64 = spins.iter().map(|&s| i64::from(s)).sum();
    sum as f64 / spins.len() as f64
}

/// Connected spin-spin correlations binned by exact squared distance.
///
/// One entry per unique squared distance, ascending; `d_sq[0] == 0` is
/// the on-site bin. Produced by [`correlation_bins`].
pub struct CorrelationBins {
    /// Exact squared distance key of each bin (lattice-spacing units²).
    pub d_sq: Vec<usize>,
    /// Pair-averaged connected correlation `<si*sj> - <m>²` per bin.
    pub correlations: Vec<f64>,
    /// Number of ordered site pairs contributing to each bin
    /// (shell multiplicity × N).
    pub counts: Vec<usize>,
}

impl CorrelationBins {
    /// Bin distances in lattice-spacing units.
    pub fn distances(&self) -> Vec<f64> {
        self.d_sq.iter().map(|&d| (d as f64).sqrt()).collect()
    }
}

/// Compute the spin-spin correlation function averaged over distance.
///
/// Correlations are `<si*sj> - <m>^2` averaged over all ordered pairs at
/// each unique squared distance; the pair counts are kept because the
/// second-moment correlation length needs them as shell weights.
pub fn correlation_bins<L: Lattice>(spins: &[i8], lattice: &L) -> CorrelationBins {
    let n = lattice.num_sites();
    let mag = magnetization_per_site(spins);
    let mag_sq = mag * mag;

    // Accumulate correlations by squared distance
    let mut corr_sum: BTreeMap<usize, f64> = BTreeMap::new();
    let mut corr_count: BTreeMap<usize, usize> = BTreeMap::new();

    for i in 0..n {
        for j in i..n {
            let d_sq = lattice.distance_squared(i, j);
            let corr = f64::from(spins[i]) * f64::from(spins[j]);

            if i == j {
                *corr_sum.entry(d_sq).or_insert(0.0) += corr;
                *corr_count.entry(d_sq).or_insert(0) += 1;
            } else {
                // Add twice to account for both (i,j) and (j,i)
                *corr_sum.entry(d_sq).or_insert(0.0) += 2.0 * corr;
                *corr_count.entry(d_sq).or_insert(0) += 2;
            }
        }
    }

    let mut d_sq = Vec::with_capacity(corr_sum.len());
    let mut correlations = Vec::with_capacity(corr_sum.len());
    let mut counts = Vec::with_capacity(corr_sum.len());

    for (&key, &sum) in &corr_sum {
        let count = corr_count[&key];
        d_sq.push(key);
        correlations.push(sum / count as f64 - mag_sq);
        counts.push(count);
    }

    CorrelationBins {
        d_sq,
        correlations,
        counts,
    }
}

/// Second-moment correlation length from the connected correlation shells.
///
/// ξ² = Σ_{r>0} n(r)·r²·C(r) / (2d · Σ_{r>0} n(r)·C(r))
///
/// where `n(r)` is the number of site pairs at distance r, so the sums
/// reproduce the lattice sum over displacement vectors — the definition
/// implied by the structure-factor curvature Ĝ(k) ≈ Ĝ(0)·(1 − k²ξ²).
///
/// Conventions:
/// * The r=0 self-term C(0) = 1 − m² is the on-site variance, not a
///   correlation between distinct spins; it is excluded from the
///   denominator (this real-space estimator differs from the Fourier
///   ξ₂nd, which keeps it). Excluding it is what makes the estimator
///   return ξ on Ornstein–Zernike data.
/// * Shells are summed in ascending r up to the first non-positive
///   shell. A finite sample's C(r) is noise-dominated exactly where the
///   r² weight is largest; truncating at the noise floor is
///   deterministic, whereas keeping isolated positive outliers at
///   arbitrary r is not.
/// * On the OZ propagator C(r) = r^{−(d−2)/2}·K_{(d−2)/2}(r/ξ) this
///   returns ξ exactly in the continuum for every d; on a pure
///   exponential it returns √((d+1)/2)·ξ — a pure exponential is not an
///   OZ propagator for d > 1.
pub fn correlation_length(bins: &CorrelationBins, dimension: usize) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for ((&d_sq, &c), &count) in bins.d_sq.iter().zip(&bins.correlations).zip(&bins.counts) {
        if d_sq == 0 {
            continue;
        }
        if c <= 0.0 {
            break;
        }
        let weight = count as f64 * c;
        numerator += weight * d_sq as f64;
        denominator += weight;
    }

    if denominator <= 0.0 {
        return 0.0;
    }

    (numerator / (2.0 * dimension as f64 * denominator)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::chain::ChainLattice;
    use crate::lattice::cubic::CubicLattice;
    use crate::lattice::honeycomb::HoneycombLattice;
    use crate::lattice::square::SquareLattice;
    use crate::lattice::triangular::TriangularLattice;
    use crate::rng::create_rng;
    use rand::Rng;

    /// Dyadic-exact sets (integer path), non-dyadic sets (sparse serial
    /// path), a mixed-ulp set, the all-zero set and a field-only set.
    const COUPLING_SETS: [(f64, f64, f64, f64); 9] = [
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.5, 0.0, 0.0),
        (-1.0, 0.0, 0.25, 0.0),
        (1.0, 0.0, 0.0, 0.5),
        (1.0, 0.0, 0.0, 1.0),
        (1.0, -0.3, 0.0, 0.1),
        (0.3, 0.0, 0.0, 0.0),
        (1.0, 9.094_947_017_729_282e-13, 0.0, 0.0), // 2^-40: mixed ulps
        (0.0, 0.0, 0.0, 0.0),
    ];

    fn random_spins(n: usize, seed: u64) -> Vec<i8> {
        let mut rng = create_rng(seed);
        (0..n)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect()
    }

    fn assert_fast_path_bit_identical<L: Lattice>(lattice: &L, label: &str) {
        for seed in 0..8u64 {
            let spins = random_spins(lattice.num_sites(), seed);
            for &(j1, j2, j3, h) in &COUPLING_SETS {
                let fast = energy_per_site(&spins, lattice, j1, j2, j3, h);
                let reference = energy_per_site_serial(&spins, lattice, j1, j2, j3, h);
                assert_eq!(
                    fast.to_bits(),
                    reference.to_bits(),
                    "{label} seed={seed} J=({j1},{j2},{j3},{h}): {fast:e} vs {reference:e}"
                );
            }
        }
    }

    #[test]
    fn test_energy_fast_path_bit_identical_square() {
        assert_fast_path_bit_identical(&SquareLattice::new(6).unwrap(), "square");
    }

    #[test]
    fn test_energy_fast_path_bit_identical_triangular() {
        assert_fast_path_bit_identical(&TriangularLattice::new(6).unwrap(), "triangular");
    }

    #[test]
    fn test_energy_fast_path_bit_identical_honeycomb() {
        assert_fast_path_bit_identical(&HoneycombLattice::new(4).unwrap(), "honeycomb");
    }

    #[test]
    fn test_energy_fast_path_bit_identical_cubic() {
        assert_fast_path_bit_identical(&CubicLattice::new(4).unwrap(), "cubic");
    }

    #[test]
    fn test_energy_fast_path_bit_identical_chain() {
        assert_fast_path_bit_identical(&ChainLattice::new(16).unwrap(), "chain");
    }

    #[test]
    fn test_dyadic_exact_classifies_couplings() {
        let lattice = SquareLattice::new(64).unwrap();
        assert!(dyadic_exact(&lattice, 1.0, 0.0, 0.0, 0.0));
        assert!(dyadic_exact(&lattice, 1.0, 0.5, 0.25, 0.0));
        assert!(dyadic_exact(&lattice, -1.0, 0.0, 0.0, 1.5));
        assert!(dyadic_exact(&lattice, 0.0, 0.0, 0.0, 0.0));
        assert!(!dyadic_exact(&lattice, 0.3, 0.0, 0.0, 0.0));
        assert!(!dyadic_exact(&lattice, 1.0, -0.3, 0.0, 0.0));
        // Mixed ulps: 4·4096·2^40 exceeds the 2^51 budget at L = 64 ...
        assert!(!dyadic_exact(&lattice, 1.0, 2f64.powi(-40), 0.0, 0.0));
        // ... but fits at L = 6 (the property test above takes that path).
        assert!(dyadic_exact(
            &SquareLattice::new(6).unwrap(),
            1.0,
            2f64.powi(-40),
            0.0,
            0.0
        ));
        // Subnormal couplings are never classified exact.
        assert!(!dyadic_exact(
            &lattice,
            f64::MIN_POSITIVE / 2.0,
            0.0,
            0.0,
            0.0
        ));
    }

    #[test]
    fn test_ulp_exponent() {
        assert_eq!(ulp_exponent(1.0), Some(0));
        assert_eq!(ulp_exponent(0.5), Some(-1));
        assert_eq!(ulp_exponent(-0.25), Some(-2));
        assert_eq!(ulp_exponent(3.0), Some(0));
        assert_eq!(ulp_exponent(1.5), Some(-1));
        assert_eq!(ulp_exponent(2f64.powi(-40)), Some(-40));
        assert_eq!(ulp_exponent(0.3), Some(-54));
        assert_eq!(ulp_exponent(0.0), None);
        assert_eq!(ulp_exponent(f64::MIN_POSITIVE / 2.0), None);
        assert_eq!(ulp_exponent(f64::INFINITY), None);
        assert_eq!(ulp_exponent(f64::NAN), None);
    }

    #[test]
    fn test_energy_exactly_zero_keeps_positive_zero() {
        // Alternating rows on a square lattice: every site has two aligned
        // and two anti-aligned nearest neighbours, so S1 = 0 and, for a
        // dyadic J1, E is exactly +0.0 on both paths.
        let lattice = SquareLattice::new(4).unwrap();
        let spins: Vec<i8> = (0..16)
            .map(|idx| if (idx / 4) % 2 == 0 { 1 } else { -1 })
            .collect();
        for &j1 in &[1.0, -1.0, 0.5] {
            let e = energy_per_site(&spins, &lattice, j1, 0.0, 0.0, 0.0);
            assert_eq!(e.to_bits(), 0.0f64.to_bits(), "J1={j1} gave {e:?}");
            let reference = energy_per_site_serial(&spins, &lattice, j1, 0.0, 0.0, 0.0);
            assert_eq!(e.to_bits(), reference.to_bits());
        }
        // Non-dyadic J1 takes the sparse path; still bit-identical.
        let e = energy_per_site(&spins, &lattice, 0.3, 0.0, 0.0, 0.0);
        let reference = energy_per_site_serial(&spins, &lattice, 0.3, 0.0, 0.0, 0.0);
        assert_eq!(e.to_bits(), reference.to_bits());
    }

    #[test]
    fn test_energy_all_up_ferromagnetic() {
        // All spins up, J1=1, J2=0, h=0 on 4x4 square lattice
        // Each site has 4 NN, each pair contributes -J1*1*1 = -1
        // Total interaction = -4*16/2 = -32 (with double-counting correction)
        // Energy per site = -32/16 = -2.0
        let lattice = SquareLattice::new(4).unwrap();
        let spins = vec![1i8; 16];
        let e = energy_per_site(&spins, &lattice, 1.0, 0.0, 0.0, 0.0);
        assert!(
            (e - (-2.0)).abs() < 1e-10,
            "Expected energy -2.0 for all-up ferromagnet, got {e}"
        );
    }

    #[test]
    fn test_energy_all_up_with_j2() {
        // All spins up, J1=1, J2=0.5, h=0 on 4x4 square lattice
        // NN contribution: -J1 * 4 * 16 / 2 / 16 = -2.0
        // NNN contribution: -J2 * 4 * 16 / 2 / 16 = -1.0
        // Total: -3.0
        let lattice = SquareLattice::new(4).unwrap();
        let spins = vec![1i8; 16];
        let e = energy_per_site(&spins, &lattice, 1.0, 0.5, 0.0, 0.0);
        assert!((e - (-3.0)).abs() < 1e-10, "Expected energy -3.0, got {e}");
    }

    #[test]
    fn test_energy_with_field() {
        // All spins up, J1=1, J2=0, h=1 on 4x4
        // Interaction: -2.0 per site
        // Field: -h * m = -1.0 * 1.0 = -1.0 per site
        // Total: -3.0
        let lattice = SquareLattice::new(4).unwrap();
        let spins = vec![1i8; 16];
        let e = energy_per_site(&spins, &lattice, 1.0, 0.0, 0.0, 1.0);
        assert!((e - (-3.0)).abs() < 1e-10, "Expected energy -3.0, got {e}");
    }

    #[test]
    fn test_energy_checkerboard_antiferromagnetic() {
        // Checkerboard pattern on 4x4: alternating +1/-1
        // Every NN pair has opposite spins: si*sj = -1
        // Interaction: -J1 * (-1) * 4 * 16 / 2 / 16 = +2.0
        let lattice = SquareLattice::new(4).unwrap();
        let mut spins = vec![1i8; 16];
        for (idx, spin) in spins.iter_mut().enumerate().take(16) {
            let row = idx / 4;
            let col = idx % 4;
            if (row + col) % 2 == 1 {
                *spin = -1;
            }
        }
        let e = energy_per_site(&spins, &lattice, 1.0, 0.0, 0.0, 0.0);
        assert!(
            (e - 2.0).abs() < 1e-10,
            "Expected energy +2.0 for checkerboard, got {e}"
        );
    }

    #[test]
    fn test_magnetization_all_up() {
        let spins = vec![1i8; 16];
        assert!((magnetization_per_site(&spins) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_magnetization_all_down() {
        let spins = vec![-1i8; 16];
        assert!((magnetization_per_site(&spins) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_magnetization_checkerboard() {
        let mut spins = vec![1i8; 16];
        for (idx, spin) in spins.iter_mut().enumerate().take(16) {
            let row = idx / 4;
            let col = idx % 4;
            if (row + col) % 2 == 1 {
                *spin = -1;
            }
        }
        assert!((magnetization_per_site(&spins)).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_bins_all_up() {
        // All spins up: <si*sj> = 1 for all pairs, <m>^2 = 1
        // So C(r) = 1 - 1 = 0 for all distances
        let lattice = SquareLattice::new(4).unwrap();
        let spins = vec![1i8; 16];
        let bins = correlation_bins(&spins, &lattice);
        for c in &bins.correlations {
            assert!(
                c.abs() < 1e-10,
                "Expected zero connected correlation for all-up, got {c}"
            );
        }
    }

    #[test]
    fn test_correlation_bins_returns_sorted_distances() {
        let lattice = SquareLattice::new(4).unwrap();
        let spins = vec![1i8; 16];
        let bins = correlation_bins(&spins, &lattice);
        for window in bins.distances().windows(2) {
            assert!(window[0] <= window[1], "Distances should be sorted");
        }
    }

    #[test]
    fn test_correlation_bins_counts_are_pair_counts() {
        // Ordered pair counts: N self-pairs at d²=0, 4N at d²=1 (each site
        // has 4 NN), 4N at d²=2 (diagonals); all counts sum to N².
        let lattice = SquareLattice::new(8).unwrap();
        let spins = vec![1i8; 64];
        let bins = correlation_bins(&spins, &lattice);
        let n = 64usize;
        assert_eq!(bins.counts.iter().sum::<usize>(), n * n);
        assert_eq!(bins.d_sq[0], 0);
        assert_eq!(bins.counts[0], n);
        assert_eq!(bins.d_sq[1], 1);
        assert_eq!(bins.counts[1], 4 * n);
        assert_eq!(bins.d_sq[2], 2);
        assert_eq!(bins.counts[2], 4 * n);
    }

    #[test]
    // 0.0 is the exact early-return value, not a computed float.
    #[allow(clippy::float_cmp)]
    fn test_correlation_length_zero_for_uncorrelated() {
        // If all correlations beyond r=0 are zero or negative, xi = 0
        let bins = CorrelationBins {
            d_sq: vec![0, 1, 4],
            correlations: vec![1.0, 0.0, -0.1],
            counts: vec![16, 64, 64],
        };
        assert_eq!(correlation_length(&bins, 2), 0.0);
    }

    #[test]
    fn test_correlation_length_ignores_r0_bin() {
        // The r=0 self-term must not enter the denominator: bins with and
        // without a huge C(0) give the same xi.
        let with_r0 = CorrelationBins {
            d_sq: vec![0, 1, 2],
            correlations: vec![1e6, 0.5, 0.2],
            counts: vec![16, 64, 64],
        };
        let without_r0 = CorrelationBins {
            d_sq: vec![1, 2],
            correlations: vec![0.5, 0.2],
            counts: vec![64, 64],
        };
        let a = correlation_length(&with_r0, 2);
        let b = correlation_length(&without_r0, 2);
        assert!((a - b).abs() < 1e-12, "xi with r0 {a} != without {b}");
        assert!(a > 0.0);
    }

    #[test]
    fn test_correlation_length_truncates_at_first_nonpositive_shell() {
        // Shells after the first non-positive one are noise and must be
        // ignored, even if hugely positive.
        let truncated = CorrelationBins {
            d_sq: vec![0, 1, 2, 4, 5],
            correlations: vec![1.0, 0.5, 0.2, -0.01, 1e6],
            counts: vec![16, 64, 64, 32, 128],
        };
        let clean = CorrelationBins {
            d_sq: vec![0, 1, 2],
            correlations: vec![1.0, 0.5, 0.2],
            counts: vec![16, 64, 64],
        };
        let a = correlation_length(&truncated, 2);
        let b = correlation_length(&clean, 2);
        assert!((a - b).abs() < 1e-12, "truncated {a} != clean {b}");
    }

    #[test]
    fn test_correlation_length_dimension_constant_scales() {
        // Same bins, d=2 vs d=3: xi ratio must be exactly sqrt(3/2) — the
        // direct regression on the hardcoded-6 defect (B7).
        let bins = CorrelationBins {
            d_sq: vec![1, 2, 4],
            correlations: vec![0.6, 0.3, 0.1],
            counts: vec![64, 64, 32],
        };
        let xi2 = correlation_length(&bins, 2);
        let xi3 = correlation_length(&bins, 3);
        assert!(((xi2 / xi3) - 1.5f64.sqrt()).abs() < 1e-12);
    }
}
