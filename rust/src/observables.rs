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

/// Compute the spin-spin correlation function averaged over distance.
///
/// Returns (distances, correlations) where distances are in lattice spacing
/// units and correlations are `<si*sj> - <m>^2` averaged over all pairs
/// at each unique distance.
pub fn correlation_function<L: Lattice>(spins: &[i8], lattice: &L) -> (Vec<f64>, Vec<f64>) {
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

    let mut distances = Vec::with_capacity(corr_sum.len());
    let mut correlations = Vec::with_capacity(corr_sum.len());

    for (&d_sq, &sum) in &corr_sum {
        let count = corr_count[&d_sq];
        distances.push((d_sq as f64).sqrt());
        correlations.push(sum / count as f64 - mag_sq);
    }

    (distances, correlations)
}

/// Compute the correlation length from a correlation function.
///
/// Uses the second-moment definition:
/// xi = sqrt(sum(C(r) * r^2) / sum(6 * C(r)))
///
/// Only includes positive correlation values above a threshold.
pub fn correlation_length(correlations: &[f64], distances: &[f64]) -> f64 {
    let threshold = 1e-8;
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (c, d) in correlations.iter().zip(distances.iter()) {
        if *c > threshold {
            numerator += c * d * d;
            denominator += 6.0 * c;
        }
    }

    if denominator <= 0.0 || numerator / denominator < 0.0 {
        return 0.0;
    }

    (numerator / denominator).sqrt()
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
        for idx in 0..16 {
            let row = idx / 4;
            let col = idx % 4;
            if (row + col) % 2 == 1 {
                spins[idx] = -1;
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
        for idx in 0..16 {
            let row = idx / 4;
            let col = idx % 4;
            if (row + col) % 2 == 1 {
                spins[idx] = -1;
            }
        }
        assert!((magnetization_per_site(&spins)).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_function_all_up() {
        // All spins up: <si*sj> = 1 for all pairs, <m>^2 = 1
        // So C(r) = 1 - 1 = 0 for all distances
        let lattice = SquareLattice::new(4).unwrap();
        let spins = vec![1i8; 16];
        let (_, correlations) = correlation_function(&spins, &lattice);
        for c in &correlations {
            assert!(
                c.abs() < 1e-10,
                "Expected zero connected correlation for all-up, got {c}"
            );
        }
    }

    #[test]
    fn test_correlation_function_returns_sorted_distances() {
        let lattice = SquareLattice::new(4).unwrap();
        let spins = vec![1i8; 16];
        let (distances, _) = correlation_function(&spins, &lattice);
        for window in distances.windows(2) {
            assert!(window[0] <= window[1], "Distances should be sorted");
        }
    }

    #[test]
    fn test_correlation_length_zero_for_uncorrelated() {
        // If all correlations are zero or negative, xi should be 0
        let correlations = vec![0.0, 0.0, -0.1];
        let distances = vec![0.0, 1.0, 2.0];
        assert_eq!(correlation_length(&correlations, &distances), 0.0);
    }
}
