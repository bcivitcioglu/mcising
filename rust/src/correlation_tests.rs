//! Synthetic-kernel recovery suite for the second-moment correlation
//! length (P09, B7/#18).
//!
//! Strategy: place a known correlation kernel C(r) on the *actual*
//! distance bins of each lattice (real `distance_squared`, real shell
//! multiplicities) and require `correlation_length` to return the known
//! ξ₀. This exercises the dimension constant, the pair-count weights,
//! the r=0 exclusion, and the lattice metric in one assertion.
//!
//! ## Which kernel recovers ξ₀ (the load-bearing derivation)
//!
//! The estimator is the structure-factor curvature
//! ξ² = Σ_r r²C(r) / (2d Σ_r C(r)), summed over displacement vectors.
//! In the continuum with pair density r^{d−1} dr and C(r) = r^{−a}e^{−r/ξ₀},
//!
//!   ξ_est² = ξ₀² · (d−a)(d−a+1) / (2d).
//!
//! It returns ξ₀ exactly when C is the Ornstein–Zernike propagator
//! (a = (d−1)/2 asymptotically; the exact OZ form e^{−r/ξ}/r in d=3),
//! because Ĝ(k) = 1/(k² + ξ₀^{−2}) has curvature ξ₀² identically. A *pure*
//! exponential is not an OZ propagator for d > 1: its second-moment
//! length is exactly √((d+1)/2)·ξ₀ (√1.5 ξ₀ in 2D, √2 ξ₀ in 3D). The
//! original roadmap gate asked for ξ₀ from a pure exponential — that is
//! mathematically unreachable with the (correct) 2d constant, so the gate
//! was corrected to the OZ kernel and the pure-exponential expectation is
//! asserted at its exact analytic value instead (see ROADMAP P09 log).
//!
//! In d=1 the discrete sum has a closed form with no continuum
//! approximation at all: for C(r) = q^{|r|}, q = e^{−1/ξ₀},
//!
//!   ξ_est² = Σ r²q^r / (2 Σ q^r) = (1+q) / (2(1−q)²),
//!
//! a machine-precision oracle for the constant, the counts, and the r=0
//! exclusion at once.

use crate::lattice::chain::ChainLattice;
use crate::lattice::cubic::CubicLattice;
use crate::lattice::honeycomb::HoneycombLattice;
use crate::lattice::square::SquareLattice;
use crate::lattice::triangular::TriangularLattice;
use crate::lattice::Lattice;
use crate::observables::{correlation_length, CorrelationBins};
use std::collections::BTreeMap;

/// Build bins with a synthetic kernel on a lattice's true distance
/// structure. Counts come from a single-site scan: every lattice here is
/// vertex-transitive (honeycomb's sublattices are related by inversion),
/// so the distance multiset from site 0 replicates at every site and the
/// global factor N cancels in the estimator.
fn synthetic_bins<L: Lattice>(lattice: &L, kernel: impl Fn(f64) -> f64) -> CorrelationBins {
    let mut count_map: BTreeMap<usize, usize> = BTreeMap::new();
    for j in 0..lattice.num_sites() {
        *count_map.entry(lattice.distance_squared(0, j)).or_insert(0) += 1;
    }

    let mut d_sq = Vec::with_capacity(count_map.len());
    let mut correlations = Vec::with_capacity(count_map.len());
    let mut counts = Vec::with_capacity(count_map.len());
    for (&key, &count) in &count_map {
        d_sq.push(key);
        counts.push(count);
        let r = (key as f64).sqrt();
        // The r=0 bin value is irrelevant (excluded by the estimator);
        // 1.0 is the physical on-site variance of ±1 spins at m=0.
        correlations.push(if key == 0 { 1.0 } else { kernel(r) });
    }

    CorrelationBins {
        d_sq,
        correlations,
        counts,
    }
}

fn assert_within(xi: f64, expected: f64, rel_tol: f64, label: &str) {
    let rel = (xi - expected) / expected;
    // Calibration record; visible under `cargo test -- --nocapture`.
    eprintln!("{label}: xi = {xi:.6}, expected {expected:.6}, rel dev {rel:+.5}");
    assert!(
        rel.abs() < rel_tol,
        "{label}: xi = {xi:.6}, expected {expected:.6} (rel dev {rel:+.4}, tol {rel_tol})"
    );
}

#[test]
fn xi_chain_closed_form() {
    // d=1 discrete oracle: no continuum limit, exact geometric sums.
    // L=64 truncates at r=32 where q^32 = e^{-16} ≈ 1e-7.
    let lattice = ChainLattice::new(64).unwrap();
    let xi0: f64 = 2.0;
    let q = (-1.0 / xi0).exp();
    let bins = synthetic_bins(&lattice, |r| q.powf(r));
    let xi = correlation_length(&bins, lattice.dimension());
    let exact = ((1.0 + q) / (2.0 * (1.0 - q) * (1.0 - q))).sqrt();
    assert_within(xi, exact, 1e-4, "chain L=64 q=e^{-1/2}");
}

#[test]
fn xi_square_oz_within_5pct() {
    // GATE (corrected form): OZ asymptotic kernel in 2D recovers xi0.
    let lattice = SquareLattice::new(128).unwrap();
    let xi0 = 6.0;
    let bins = synthetic_bins(&lattice, |r| r.powf(-0.5) * (-r / xi0).exp());
    let xi = correlation_length(&bins, lattice.dimension());
    assert_within(xi, xi0, 0.05, "square L=128 OZ xi0=6");
    // Sharper pin: the 2D OZ *asymptotic* kernel's continuum second
    // moment is 0.96825 xi0 (= sqrt(15/16)); the lattice sum must sit
    // within 2% of that value.
    assert_within(xi, 0.968_246 * xi0, 0.02, "square L=128 OZ continuum pin");
}

#[test]
fn xi_cubic_oz_within_5pct() {
    // GATE (corrected form): the exact OZ propagator e^{-r/xi}/r in 3D —
    // continuum second moment is xi0 exactly.
    let lattice = CubicLattice::new(32).unwrap();
    let xi0 = 2.5;
    let bins = synthetic_bins(&lattice, |r| (-r / xi0).exp() / r);
    let xi = correlation_length(&bins, lattice.dimension());
    assert_within(xi, xi0, 0.05, "cubic L=32 OZ xi0=2.5");
    assert_within(xi, xi0, 0.02, "cubic L=32 OZ continuum pin");
}

#[test]
fn xi_pure_exponential_is_sqrt_d_plus_1_over_2() {
    // Executable documentation of the corrected gate: a pure exponential
    // is NOT an OZ propagator for d > 1; its second-moment length is
    // exactly sqrt((d+1)/2)·xi0. Asserting the exact analytic value is
    // strictly harder than the original (unreachable) xi0 assertion.
    let square = SquareLattice::new(128).unwrap();
    let xi0_2d = 6.0;
    let bins = synthetic_bins(&square, |r| (-r / xi0_2d).exp());
    let xi = correlation_length(&bins, square.dimension());
    assert_within(xi, 1.5f64.sqrt() * xi0_2d, 0.02, "square pure-exp");

    let cubic = CubicLattice::new(32).unwrap();
    let xi0_3d = 2.0;
    let bins = synthetic_bins(&cubic, |r| (-r / xi0_3d).exp());
    let xi = correlation_length(&bins, cubic.dimension());
    assert_within(xi, 2f64.sqrt() * xi0_3d, 0.03, "cubic pure-exp");
}

#[test]
fn xi_triangular_metric_flows_through() {
    // The P05-corrected triangular metric (integer 4d² = ΔX² + 3ΔY²)
    // feeds the estimator; OZ recovery on it proves metric + d=2 + counts
    // compose correctly.
    let lattice = TriangularLattice::new(64).unwrap();
    let xi0 = 3.0;
    let bins = synthetic_bins(&lattice, |r| r.powf(-0.5) * (-r / xi0).exp());
    let xi = correlation_length(&bins, lattice.dimension());
    assert_within(xi, xi0, 0.05, "triangular L=64 OZ xi0=3");
}

#[test]
fn xi_honeycomb_metric_flows_through() {
    // Runs on the #35-corrected honeycomb metric. The L×L unit-cell
    // honeycomb torus is 3L × (√3/2)L in real space, so the short axis
    // limits usable xi0; L=48 with xi0=2.5 is safely isotropic.
    let lattice = HoneycombLattice::new(48).unwrap();
    let xi0 = 2.5;
    let bins = synthetic_bins(&lattice, |r| r.powf(-0.5) * (-r / xi0).exp());
    let xi = correlation_length(&bins, lattice.dimension());
    assert_within(xi, xi0, 0.05, "honeycomb L=48 OZ xi0=2.5");
}
