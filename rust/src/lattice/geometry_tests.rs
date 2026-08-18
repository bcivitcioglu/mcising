//! Cross-lattice neighbor-table geometry matrix (P05).
//!
//! Every lattice × shell × size combination is checked for the invariants
//! a valid Hamiltonian needs: reciprocal bonds, exact coordination, shell
//! disjointness, no self-bonds, and — the assertion class that actually
//! catches wrong tables (#14, #34) — the exact Euclidean distance of every
//! table entry in the embedding the NN table realizes.
//!
//! Matrix: {chain, square, cubic, triangular, honeycomb} × {NN, NNN, TNN}
//! × L ∈ {4, 6, 8} = 45 combinations.
//!
//! ## Torus-validity guard
//!
//! On a small torus the periodic wrap itself identifies shell sites: on
//! the L=4 chain, i+3 ≡ i−1 *is* a nearest neighbor, and (+2) ≡ (−2)
//! collapses radius-2 shells to duplicate entries. These identifications
//! are geometric facts of the quotient lattice, not table bugs, so the
//! affected assertions are guarded by the shell radius r (the maximum
//! |offset component| of the shell):
//!
//! - reciprocity (multiset), table stride, index bounds, no self-bonds:
//!   asserted for all 45 combinations, unconditionally;
//! - exact shell distance: asserted when `2r <= L` (min-image cannot
//!   shorten an offset with components ≤ L/2) — this skips only chain TNN
//!   (r=3) at L=4;
//! - unique-neighbor count == coordination number, and shell
//!   disjointness: asserted when `2r + 1 <= L` with r the largest radius
//!   involved (all offsets then distinct mod L) — this skips the radius-2
//!   shells at L=4 and chain TNN below L=8.

use super::chain::ChainLattice;
use super::cubic::CubicLattice;
use super::honeycomb::HoneycombLattice;
use super::square::SquareLattice;
use super::triangular::TriangularLattice;
use super::Lattice;
use crate::observables;
use crate::rng::create_rng;
use rand::Rng;
use std::collections::HashSet;

const SIZES: [usize; 3] = [4, 6, 8];
const SHELL_NAMES: [&str; 3] = ["NN", "NNN", "TNN"];

/// One lattice under test: the instance plus its embedding.
struct GeometryCase {
    name: &'static str,
    lattice: Box<dyn Lattice>,
    /// Cartesian position of a flat site index, NN distance = 1.
    pos: Box<dyn Fn(usize) -> [f64; 3]>,
    /// Torus translation vectors in the same cartesian frame.
    periods: Vec<[f64; 3]>,
    /// Exact squared shell distances {NN, NNN, TNN}.
    d2: [f64; 3],
    /// Shell radius: max |offset component|, for the torus-validity guard.
    radius: [usize; 3],
}

fn cases(l: usize) -> Vec<GeometryCase> {
    let fl = l as f64;
    let sq32 = 3f64.sqrt() / 2.0;

    let chain = ChainLattice::new(l).unwrap();
    let square = SquareLattice::new(l).unwrap();
    let cubic = CubicLattice::new(l).unwrap();
    let triangular = TriangularLattice::new(l).unwrap();
    let honeycomb = HoneycombLattice::new(l).unwrap();

    vec![
        GeometryCase {
            name: "chain",
            lattice: Box::new(chain),
            pos: Box::new(|i| [i as f64, 0.0, 0.0]),
            periods: vec![[fl, 0.0, 0.0]],
            d2: [1.0, 4.0, 9.0],
            radius: [1, 2, 3],
        },
        GeometryCase {
            name: "square",
            lattice: Box::new(square),
            pos: Box::new(move |i| [(i % l) as f64, (i / l) as f64, 0.0]),
            periods: vec![[fl, 0.0, 0.0], [0.0, fl, 0.0]],
            d2: [1.0, 2.0, 4.0],
            radius: [1, 1, 2],
        },
        GeometryCase {
            name: "cubic",
            lattice: Box::new(cubic),
            pos: Box::new(move |i| [(i % l) as f64, ((i / l) % l) as f64, (i / (l * l)) as f64]),
            periods: vec![[fl, 0.0, 0.0], [0.0, fl, 0.0], [0.0, 0.0, fl]],
            d2: [1.0, 2.0, 3.0],
            radius: [1, 1, 1],
        },
        GeometryCase {
            name: "triangular",
            // x = c + (r % 2)/2, y = r*sqrt(3)/2; even-L row wrap is the
            // lattice translation (0, L*sqrt(3)/2).
            lattice: Box::new(triangular),
            pos: Box::new(move |i| {
                let row = i / l;
                let col = i % l;
                [
                    col as f64 + 0.5 * ((row % 2) as f64),
                    row as f64 * sq32,
                    0.0,
                ]
            }),
            periods: vec![[fl, 0.0, 0.0], [0.0, fl * sq32, 0.0]],
            d2: [1.0, 3.0, 4.0],
            radius: [1, 2, 2],
        },
        GeometryCase {
            name: "honeycomb",
            // Armchair rows, unit-cell width 3 (see honeycomb.rs):
            // x = 3c + {A even: 0, B even: +1, A odd: -3/2, B odd: -1/2}.
            lattice: Box::new(honeycomb),
            pos: Box::new(move |i| {
                let stride = 2 * l;
                let row = i / stride;
                let col = (i % stride) / 2;
                let sub = i % 2;
                let off = match (row % 2, sub) {
                    (0, 0) => 0.0,
                    (0, 1) => 1.0,
                    (1, 0) => -1.5,
                    _ => -0.5,
                };
                [3.0 * col as f64 + off, row as f64 * sq32, 0.0]
            }),
            periods: vec![[3.0 * fl, 0.0, 0.0], [0.0, fl * sq32, 0.0]],
            d2: [1.0, 3.0, 4.0],
            radius: [1, 2, 2],
        },
    ]
}

fn shell(lat: &dyn Lattice, k: usize, idx: usize) -> &[usize] {
    match k {
        0 => lat.nearest_neighbors(idx),
        1 => lat.next_nearest_neighbors(idx),
        2 => lat.third_nearest_neighbors(idx),
        _ => unreachable!(),
    }
}

fn coordination(lat: &dyn Lattice, k: usize) -> usize {
    match k {
        0 => lat.coordination_number(),
        1 => lat.nnn_coordination_number(),
        2 => lat.tnn_coordination_number(),
        _ => unreachable!(),
    }
}

/// Minimum-image squared distance between two sites over all torus images.
fn min_image_d2(case: &GeometryCase, a: usize, b: usize) -> f64 {
    let pa = (case.pos)(a);
    let pb = (case.pos)(b);
    let d0 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    let mut best = f64::INFINITY;
    // Up to 3 period vectors; iterate shift coefficients in {-1, 0, 1}.
    let shifts: [f64; 3] = [-1.0, 0.0, 1.0];
    for &s0 in &shifts {
        for &s1 in &shifts {
            for &s2 in &shifts {
                let coeff = [s0, s1, s2];
                let mut d = d0;
                for (pi, p) in case.periods.iter().enumerate() {
                    for x in 0..3 {
                        d[x] += coeff[pi] * p[x];
                    }
                }
                let n2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                best = best.min(n2);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Unconditional assertions: all 45 combinations ────────────────

    #[test]
    fn test_matrix_reciprocity_stride_bounds_no_self() {
        for l in SIZES {
            for case in cases(l) {
                let lat = case.lattice.as_ref();
                let n = lat.num_sites();
                for (k, shell_name) in SHELL_NAMES.iter().enumerate() {
                    let z = coordination(lat, k);
                    for i in 0..n {
                        let nbrs = shell(lat, k, i);
                        assert_eq!(
                            nbrs.len(),
                            z,
                            "{} L={} {shell_name}: site {i} table stride != coordination",
                            case.name,
                            l,
                        );
                        for &j in nbrs {
                            assert!(
                                j < n,
                                "{} L={} {shell_name}: site {i} neighbor {j} out of bounds",
                                case.name,
                                l,
                            );
                            assert_ne!(
                                j, i,
                                "{} L={} {shell_name}: site {i} is its own neighbor",
                                case.name, l,
                            );
                            // Multiset reciprocity: j appears in shell(i) as
                            // often as i appears in shell(j), so duplicate
                            // wrap entries at small L stay balanced.
                            let fwd = nbrs.iter().filter(|&&x| x == j).count();
                            let bwd = shell(lat, k, j).iter().filter(|&&x| x == i).count();
                            assert_eq!(
                                fwd, bwd,
                                "{} L={} {shell_name}: bond {i}<->{j} not reciprocal \
                                 ({fwd} vs {bwd})",
                                case.name, l,
                            );
                        }
                    }
                }
            }
        }
    }

    // ── Exact shell distances: guarded by 2r <= L ────────────────────

    #[test]
    fn test_matrix_exact_shell_distances() {
        let mut checked = 0usize;
        for l in SIZES {
            for case in cases(l) {
                let lat = case.lattice.as_ref();
                for (k, shell_name) in SHELL_NAMES.iter().enumerate() {
                    if 2 * case.radius[k] > l {
                        continue; // torus-validity guard (chain TNN at L=4)
                    }
                    checked += 1;
                    for i in 0..lat.num_sites() {
                        for &j in shell(lat, k, i) {
                            let d2 = min_image_d2(&case, i, j);
                            assert!(
                                (d2 - case.d2[k]).abs() < 1e-9,
                                "{} L={} {shell_name}: site {i} neighbor {j} at d²={d2}, \
                                 expected {}",
                                case.name,
                                l,
                                case.d2[k]
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 44, "expected 45 combos minus chain TNN at L=4");
    }

    // ── Metric agreement: every pair, every lattice (P09/#35) ────────

    #[test]
    fn test_matrix_distance_squared_matches_embedding() {
        // `distance_squared` must equal the min-image distance of the
        // embedding the NN table realizes — for ALL site pairs, not just
        // shell entries. Both sides minimize over the same 3×3(×3) torus
        // images, so no torus-validity guard is needed. This is the
        // oracle that catches #35 (honeycomb ignored the sublattice
        // offset, so the same-cell A–B NN bond sat in the d²=0 bin).
        for l in SIZES {
            for case in cases(l) {
                let lat = case.lattice.as_ref();
                let n = lat.num_sites();
                for i in 0..n {
                    for j in 0..n {
                        let expected = min_image_d2(&case, i, j);
                        let actual = lat.distance_squared(i, j) as f64;
                        assert!(
                            (actual - expected).abs() < 1e-9,
                            "{} L={}: d²({i},{j}) = {actual}, embedding says {expected}",
                            case.name,
                            l,
                        );
                    }
                }
            }
        }
    }

    // ── Uniqueness + disjointness: guarded by 2r + 1 <= L ────────────

    #[test]
    fn test_matrix_unique_neighbor_counts() {
        for l in SIZES {
            for case in cases(l) {
                let lat = case.lattice.as_ref();
                for (k, shell_name) in SHELL_NAMES.iter().enumerate() {
                    if 2 * case.radius[k] + 1 > l {
                        continue; // wrap-duplicate entries are expected
                    }
                    let z = coordination(lat, k);
                    for i in 0..lat.num_sites() {
                        let unique: HashSet<usize> = shell(lat, k, i).iter().copied().collect();
                        assert_eq!(
                            unique.len(),
                            z,
                            "{} L={} {shell_name}: site {i} has duplicate neighbors",
                            case.name,
                            l,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_matrix_shell_disjointness() {
        for l in SIZES {
            for case in cases(l) {
                let lat = case.lattice.as_ref();
                let r_max = *case.radius.iter().max().unwrap();
                if 2 * r_max + 1 > l {
                    continue; // torus identifies shells at this size
                }
                for i in 0..lat.num_sites() {
                    let sets: Vec<HashSet<usize>> = (0..3)
                        .map(|k| shell(lat, k, i).iter().copied().collect())
                        .collect();
                    for a in 0..3 {
                        for b in a + 1..3 {
                            assert!(
                                sets[a].is_disjoint(&sets[b]),
                                "{} L={}: site {i} shells {} and {} overlap",
                                case.name,
                                l,
                                SHELL_NAMES[a],
                                SHELL_NAMES[b]
                            );
                        }
                    }
                }
            }
        }
    }

    // ── Brute-force pair-sum energy reference (triangular, B3 gate) ──

    /// Reference total energy by classifying every site pair through the
    /// (fixed) exact `distance_squared`: d²=1 → J1, d²=3 → J2, d²=4 → J3.
    /// Any shell double-counting (the B3 defect) shifts this away from
    /// the table-driven `energy_per_site`.
    fn pair_sum_energy(
        lat: &TriangularLattice,
        spins: &[i8],
        j1: f64,
        j2: f64,
        j3: f64,
        h: f64,
    ) -> f64 {
        let n = lat.num_sites();
        let mut e = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let coupling = match lat.distance_squared(i, j) {
                    1 => j1,
                    3 => j2,
                    4 => j3,
                    _ => continue,
                };
                e -= coupling * f64::from(spins[i]) * f64::from(spins[j]);
            }
        }
        let m: f64 = spins.iter().map(|&s| f64::from(s)).sum();
        e - h * m
    }

    #[test]
    fn test_triangular_pair_sum_energy_all_up() {
        // The roadmap gate: all-up, J1=J2=J3=1, agreement to 1e-10.
        let lat = TriangularLattice::new(6).unwrap();
        let n = lat.num_sites();
        let spins = vec![1i8; n];
        let e_ref = pair_sum_energy(&lat, &spins, 1.0, 1.0, 1.0, 0.0) / n as f64;
        let e_tab = observables::energy_per_site(&spins, &lat, 1.0, 1.0, 1.0, 0.0);
        assert!(
            (e_ref - e_tab).abs() < 1e-10,
            "all-up pair sum {e_ref} vs table {e_tab}"
        );
        // Each shell has 6 bonds/site counted half: e = -(6+6+6)/2 = -9.
        assert!((e_ref - (-9.0)).abs() < 1e-10, "expected -9.0, got {e_ref}");
    }

    #[test]
    fn test_triangular_pair_sum_energy_random_configs() {
        let lat = TriangularLattice::new(6).unwrap();
        let n = lat.num_sites();
        let couplings = [
            (1.0, 1.0, 1.0, 0.0),
            (1.0, 0.5, -0.3, 0.2),
            (-1.0, 0.7, 0.2, -0.4),
        ];
        let mut rng = create_rng(42);
        for cfg_idx in 0..20 {
            let spins: Vec<i8> = (0..n)
                .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
                .collect();
            for &(j1, j2, j3, h) in &couplings {
                let e_ref = pair_sum_energy(&lat, &spins, j1, j2, j3, h) / n as f64;
                let e_tab = observables::energy_per_site(&spins, &lat, j1, j2, j3, h);
                assert!(
                    (e_ref - e_tab).abs() < 1e-10,
                    "config {cfg_idx} (J1={j1}, J2={j2}, J3={j3}, h={h}): \
                     pair sum {e_ref} vs table {e_tab}"
                );
            }
        }
    }
}
