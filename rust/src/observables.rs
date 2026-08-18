use crate::lattice::Lattice;
use std::collections::BTreeMap;

/// Compute the total energy per site of the spin configuration.
///
/// E/N = (-sum_{<i,j>} J1*si*sj - sum_{<<i,j>>} J2*si*sj
///        - sum_{<<<i,j>>>} J3*si*sj - h*sum_i si) / N
///
/// Interaction terms are divided by 2 to correct for double-counting.
pub fn energy_per_site<L: Lattice>(
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
    use crate::lattice::square::SquareLattice;

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
