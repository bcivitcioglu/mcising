//! Exact energy grid for flat-histogram sampling.
//!
//! The walkers index their density-of-states arrays by an integer bin. With
//! dyadic couplings every energy is an exact integer in units of `2^e`
//! (`observables::dyadic_exact`), the walker carries the exact ordered-pair
//! shell sums, and the bin of a state is a pure function of those integers —
//! never of the path taken to reach it. The bin width is the greatest common
//! divisor of every single-flip energy change, so no reachable energy falls
//! between two bins. Couplings that are not dyadic-exact need an explicit
//! `bin_width` and take a float path that rounds the exact-shell energy to
//! the nearest bin centre.

use crate::error::MCIsingError;
use crate::lattice::Lattice;
use crate::observables::{self, ShellSums};

/// Hard cap on the bin count: the walker keeps three arrays per bin and the
/// runner returns copies; above this the arrays, not the lattice, own the
/// memory and the caller should pass a coarser `bin_width` or a window.
pub(crate) const MAX_BINS: usize = 1 << 24;

/// Couplings as integers in units of `2^scale_exp`, where `scale_exp` is the
/// smallest ulp exponent over the nonzero couplings. Exists only when
/// [`observables::dyadic_exact`] holds, so every product below is exact in
/// both `i64` and `f64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IntegerCouplings {
    pub(crate) scale_exp: i32,
    pub(crate) j1: i64,
    pub(crate) j2: i64,
    pub(crate) j3: i64,
    pub(crate) h: i64,
}

impl IntegerCouplings {
    /// `None` when the couplings are not dyadic-exact for this lattice or
    /// are all zero.
    pub(crate) fn try_new<L: Lattice>(
        lattice: &L,
        j1: f64,
        j2: f64,
        j3: f64,
        h: f64,
    ) -> Option<Self> {
        if !observables::dyadic_exact(lattice, j1, j2, j3, h) {
            return None;
        }
        let mut min_exp: Option<i32> = None;
        for coupling in [j1, j2, j3, h] {
            if coupling == 0.0 {
                continue;
            }
            let exp = observables::ulp_exponent(coupling)?;
            min_exp = Some(min_exp.map_or(exp, |m| m.min(exp)));
        }
        let scale_exp = min_exp?;
        let scale = 2f64.powi(-scale_exp);
        // Exact: each coupling is an integer multiple of 2^scale_exp and the
        // dyadic bound keeps |c · 2^-e| far below 2^53.
        let to_units = |coupling: f64| (coupling * scale) as i64;
        Some(Self {
            scale_exp,
            j1: to_units(j1),
            j2: to_units(j2),
            j3: to_units(j3),
            h: to_units(h),
        })
    }

    /// The energy unit `2^scale_exp`.
    pub(crate) fn unit(&self) -> f64 {
        2f64.powi(self.scale_exp)
    }

    /// Total energy in units: `-(j1·nn + j2·nnn + j3·tnn)/2 - h·M`. The
    /// ordered-pair sums are even, so the division is exact.
    pub(crate) fn energy_units(&self, sums: &ShellSums) -> i64 {
        let twice = self.j1 * sums.nn + self.j2 * sums.nnn + self.j3 * sums.tnn;
        debug_assert!(twice % 2 == 0, "ordered-pair shell sums are even");
        -(twice / 2) - self.h * sums.magnetization
    }

    /// Change of [`Self::energy_units`] when a spin `spin` whose local shell
    /// sums are `(s1, s2, s3)` flips: `2·spin·(j1·s1 + j2·s2 + j3·s3 + h)`.
    pub(crate) fn flip_delta_units(&self, spin: i64, s1: i64, s2: i64, s3: i64) -> i64 {
        2 * spin * (self.j1 * s1 + self.j2 * s2 + self.j3 * s3 + self.h)
    }

    /// Greatest common divisor of every nonzero single-flip |ΔE| in units,
    /// over `S_k ∈ {-z_k, -z_k + 2, …, z_k}` of the shells with a nonzero
    /// coupling (an inactive shell contributes nothing). Zero when every
    /// flip is energy-neutral.
    pub(crate) fn flip_quantum_units(&self, z: [usize; 3]) -> i64 {
        let range = |zk: usize, active: bool| -> Vec<i64> {
            if active {
                (0..=zk).map(|i| 2 * i as i64 - zk as i64).collect()
            } else {
                vec![0]
            }
        };
        let range1 = range(z[0], self.j1 != 0);
        let range2 = range(z[1], self.j2 != 0);
        let range3 = range(z[2], self.j3 != 0);
        let mut quantum = 0i64;
        for &s1 in &range1 {
            for &s2 in &range2 {
                for &s3 in &range3 {
                    quantum = gcd(quantum, self.flip_delta_units(1, s1, s2, s3));
                }
            }
        }
        quantum
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `ceil(a / b)` for `b > 0`.
fn ceil_div(a: i64, b: i64) -> i64 {
    (a + b - 1).div_euclid(b)
}

/// Inclusive range of global bin indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BinWindow {
    pub(crate) lo: usize,
    pub(crate) hi: usize,
}

impl BinWindow {
    pub(crate) fn len(self) -> usize {
        self.hi - self.lo + 1
    }

    pub(crate) fn contains(self, bin: i64) -> bool {
        bin >= self.lo as i64 && bin <= self.hi as i64
    }
}

/// The integer form of the grid (exact path only).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExactBinning {
    pub(crate) couplings: IntegerCouplings,
    pub(crate) e_min_units: i64,
    pub(crate) width_units: i64,
}

/// Global energy partition: bin `k` is centred on the total energy
/// `e_min + k·width`. Every reachable energy sits exactly on a centre in
/// the exact form and within `width/2` of one in the float form.
#[derive(Clone, Debug)]
pub(crate) struct EnergyBinning {
    pub(crate) num_sites: usize,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
    /// Total energy of bin 0.
    pub(crate) e_min: f64,
    /// Bin width in total-energy units.
    pub(crate) width: f64,
    pub(crate) n_bins: usize,
    pub(crate) exact: Option<ExactBinning>,
}

impl EnergyBinning {
    /// Build the grid: the exact grid from the flip quantum when the
    /// couplings are dyadic-exact and no `bin_width` is given, the float
    /// grid otherwise.
    ///
    /// # Errors
    ///
    /// `DegenerateEnergySpectrum` when every coupling is zero,
    /// `NonDyadicCouplingsNeedBinWidth` when the couplings are not
    /// dyadic-exact and no width is given, `InvalidBinWidth` for a
    /// non-positive or non-finite width, `TooManyEnergyBins` above
    /// [`MAX_BINS`].
    pub(crate) fn detect<L: Lattice>(
        lattice: &L,
        j1: f64,
        j2: f64,
        j3: f64,
        h: f64,
        bin_width: Option<f64>,
    ) -> Result<Self, MCIsingError> {
        let num_sites = lattice.num_sites();
        let z = [
            lattice.coordination_number(),
            lattice.nnn_coordination_number(),
            lattice.tnn_coordination_number(),
        ];
        // Ordered pairs per shell: twice the bond count, hence even.
        let pairs = [z[0] * num_sites, z[1] * num_sites, z[2] * num_sites];
        let n = num_sites as f64;
        let e_ref = 0.0
            - (j1 * pairs[0] as f64 + j2 * pairs[1] as f64 + j3 * pairs[2] as f64) / 2.0
            - h * n;
        let bound =
            (j1.abs() * pairs[0] as f64 + j2.abs() * pairs[1] as f64 + j3.abs() * pairs[2] as f64)
                / 2.0
                + h.abs() * n;
        if bound == 0.0 {
            return Err(MCIsingError::DegenerateEnergySpectrum);
        }
        if let Some(width) = bin_width {
            return Self::with_width(num_sites, [j1, j2, j3, h], e_ref, bound, width);
        }
        let Some(couplings) = IntegerCouplings::try_new(lattice, j1, j2, j3, h) else {
            return Err(MCIsingError::NonDyadicCouplingsNeedBinWidth);
        };
        let quantum = couplings.flip_quantum_units(z);
        if quantum == 0 {
            return Err(MCIsingError::DegenerateEnergySpectrum);
        }
        let all_up = ShellSums {
            nn: pairs[0] as i64,
            nnn: pairs[1] as i64,
            tnn: pairs[2] as i64,
            magnetization: num_sites as i64,
        };
        let e_ref_units = couplings.energy_units(&all_up);
        let bound_units = (couplings.j1.abs() * pairs[0] as i64
            + couplings.j2.abs() * pairs[1] as i64
            + couplings.j3.abs() * pairs[2] as i64)
            / 2
            + couplings.h.abs() * num_sites as i64;
        let below = ceil_div(e_ref_units + bound_units, quantum);
        let e_min_units = e_ref_units - quantum * below;
        let n_bins_i = (bound_units - e_min_units) / quantum + 1;
        let n_bins = usize::try_from(n_bins_i).unwrap_or(usize::MAX);
        if n_bins > MAX_BINS {
            return Err(MCIsingError::TooManyEnergyBins(n_bins, MAX_BINS));
        }
        let unit = couplings.unit();
        Ok(Self {
            num_sites,
            j1,
            j2,
            j3,
            h,
            e_min: e_min_units as f64 * unit,
            width: quantum as f64 * unit,
            n_bins,
            exact: Some(ExactBinning {
                couplings,
                e_min_units,
                width_units: quantum,
            }),
        })
    }

    fn with_width(
        num_sites: usize,
        couplings: [f64; 4],
        e_ref: f64,
        bound: f64,
        width: f64,
    ) -> Result<Self, MCIsingError> {
        if !width.is_finite() || width <= 0.0 {
            return Err(MCIsingError::InvalidBinWidth(width));
        }
        let below = ((e_ref + bound) / width).ceil();
        let e_min = e_ref - width * below;
        let n_bins_f = ((bound - e_min) / width).floor() + 1.0;
        if !n_bins_f.is_finite() || n_bins_f > MAX_BINS as f64 {
            let requested = if n_bins_f.is_finite() {
                n_bins_f as usize
            } else {
                usize::MAX
            };
            return Err(MCIsingError::TooManyEnergyBins(requested, MAX_BINS));
        }
        Ok(Self {
            num_sites,
            j1: couplings[0],
            j2: couplings[1],
            j3: couplings[2],
            h: couplings[3],
            e_min,
            width,
            n_bins: n_bins_f as usize,
            exact: None,
        })
    }

    /// Total energy of a configuration from its exact shell sums (the
    /// `energy_from_shells` expression without the per-site division).
    pub(crate) fn total_energy(&self, sums: &ShellSums) -> f64 {
        let interaction =
            0.0 - self.j1 * sums.nn as f64 - self.j2 * sums.nnn as f64 - self.j3 * sums.tnn as f64;
        interaction / 2.0 + (0.0 - self.h * sums.magnetization as f64)
    }

    /// Total energy change of flipping a spin with local sums `(s1, s2, s3)`.
    pub(crate) fn flip_energy(&self, spin: i64, s1: i64, s2: i64, s3: i64) -> f64 {
        2.0 * spin as f64
            * (self.j1 * s1 as f64 + self.j2 * s2 as f64 + self.j3 * s3 as f64 + self.h)
    }

    /// Bin of a configuration: a pure function of its exact shell sums.
    pub(crate) fn bin_of_shells(&self, sums: &ShellSums) -> usize {
        if let Some(exact) = &self.exact {
            let offset = exact.couplings.energy_units(sums) - exact.e_min_units;
            debug_assert!(
                offset >= 0 && offset % exact.width_units == 0,
                "every reachable energy is a bin centre"
            );
            (offset / exact.width_units) as usize
        } else {
            let energy = self.total_energy(sums);
            let bin = ((energy - self.e_min) / self.width).round().max(0.0) as usize;
            bin.min(self.n_bins - 1)
        }
    }

    /// Total energy at the centre of bin `bin`.
    pub(crate) fn energy_of_bin(&self, bin: usize) -> f64 {
        match &self.exact {
            Some(exact) => {
                (exact.e_min_units + bin as i64 * exact.width_units) as f64 * exact.couplings.unit()
            }
            None => self.e_min + bin as f64 * self.width,
        }
    }

    /// Per-site energy of every bin centre.
    pub(crate) fn energies_per_site(&self) -> Vec<f64> {
        let n = self.num_sites as f64;
        (0..self.n_bins)
            .map(|bin| self.energy_of_bin(bin) / n)
            .collect()
    }

    /// The bins whose centre lies in the per-site window `[e_lo, e_hi]`.
    ///
    /// # Errors
    ///
    /// `InvalidEnergyWindow` when the bounds are not finite, not ordered,
    /// or select no bin.
    pub(crate) fn window_bins(&self, e_lo: f64, e_hi: f64) -> Result<BinWindow, MCIsingError> {
        if !e_lo.is_finite() || !e_hi.is_finite() || e_lo >= e_hi {
            return Err(MCIsingError::InvalidEnergyWindow(e_lo, e_hi));
        }
        let n = self.num_sites as f64;
        let tolerance = 1e-9 * self.width;
        let lo = ((e_lo * n - self.e_min - tolerance) / self.width)
            .ceil()
            .max(0.0);
        let hi = ((e_hi * n - self.e_min + tolerance) / self.width)
            .floor()
            .min((self.n_bins - 1) as f64);
        if lo > hi {
            return Err(MCIsingError::InvalidEnergyWindow(e_lo, e_hi));
        }
        Ok(BinWindow {
            lo: lo as usize,
            hi: hi as usize,
        })
    }

    /// Every bin of the grid.
    pub(crate) fn full_window(&self) -> BinWindow {
        BinWindow {
            lo: 0,
            hi: self.n_bins - 1,
        }
    }
}

/// Exact-path lookup `Δbin(spin, S1, S2, S3)`, laid out like the Metropolis
/// three-coupling table: the index is `spin_idx·block + Σ_k i_k·stride_k`
/// with `i_k = (S_k + z_k)/2`. A shell with a zero coupling has extent 1
/// and stride 0, so its (always zero) local sum never moves the index.
pub(crate) struct DeltaBinTable {
    z: [i64; 3],
    strides: [usize; 3],
    block: usize,
    delta_bin: Vec<i32>,
}

impl DeltaBinTable {
    pub(crate) fn build(exact: &ExactBinning, z: [usize; 3], active: [bool; 3]) -> Self {
        let extent = |k: usize| if active[k] { z[k] + 1 } else { 1 };
        let ext = [extent(0), extent(1), extent(2)];
        let strides = [
            if active[0] { ext[1] * ext[2] } else { 0 },
            if active[1] { ext[2] } else { 0 },
            usize::from(active[2]),
        ];
        let block = ext[0] * ext[1] * ext[2];
        let local_sum = |k: usize, i: usize| -> i64 {
            if active[k] {
                2 * i as i64 - z[k] as i64
            } else {
                0
            }
        };
        let mut delta_bin = vec![0i32; 2 * block];
        for (spin_idx, spin) in [-1i64, 1].into_iter().enumerate() {
            for i1 in 0..ext[0] {
                for i2 in 0..ext[1] {
                    for i3 in 0..ext[2] {
                        let delta = exact.couplings.flip_delta_units(
                            spin,
                            local_sum(0, i1),
                            local_sum(1, i2),
                            local_sum(2, i3),
                        );
                        debug_assert_eq!(delta % exact.width_units, 0);
                        let idx =
                            spin_idx * block + i1 * strides[0] + i2 * strides[1] + i3 * strides[2];
                        delta_bin[idx] = (delta / exact.width_units) as i32;
                    }
                }
            }
        }
        Self {
            z: [z[0] as i64, z[1] as i64, z[2] as i64],
            strides,
            block,
            delta_bin,
        }
    }

    #[inline]
    pub(crate) fn index(&self, spin: i64, s1: i64, s2: i64, s3: i64) -> usize {
        let spin_idx = ((spin + 1) >> 1) as usize;
        let i1 = ((s1 + self.z[0]) >> 1) as usize;
        let i2 = ((s2 + self.z[1]) >> 1) as usize;
        let i3 = ((s3 + self.z[2]) >> 1) as usize;
        spin_idx * self.block + i1 * self.strides[0] + i2 * self.strides[1] + i3 * self.strides[2]
    }

    #[inline]
    pub(crate) fn delta(&self, idx: usize) -> i64 {
        i64::from(self.delta_bin[idx])
    }
}

#[cfg(test)]
mod tests {
    // Bit-identity of grid energies and weights is the contract these
    // tests pin, so exact float comparison is the point.
    #![allow(clippy::float_cmp)]

    use super::*;
    use crate::lattice::chain::ChainLattice;
    use crate::lattice::cubic::CubicLattice;
    use crate::lattice::honeycomb::HoneycombLattice;
    use crate::lattice::square::SquareLattice;
    use crate::lattice::triangular::TriangularLattice;
    use crate::rng::create_rng;
    use rand::Rng;

    fn random_spins(n: usize, seed: u64) -> Vec<i8> {
        let mut rng = create_rng(seed);
        (0..n)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect()
    }

    fn all_up(n: usize) -> Vec<i8> {
        vec![1; n]
    }

    /// Bit-identity is the contract between the grid and the energy
    /// observable, so exact float comparison is the point of these tests.
    #[allow(clippy::float_cmp)]
    fn assert_consistent<L: Lattice>(lattice: &L, couplings: [f64; 4], bin_width: Option<f64>) {
        let [j1, j2, j3, h] = couplings;
        let binning = EnergyBinning::detect(lattice, j1, j2, j3, h, bin_width).expect("valid grid");
        let n = lattice.num_sites();
        let mut configs = vec![all_up(n)];
        for seed in 0..20 {
            configs.push(random_spins(n, 100 + seed));
        }
        for spins in &configs {
            let sums = observables::shell_sums(spins, lattice, j1 != 0.0, j2 != 0.0, j3 != 0.0);
            let bin = binning.bin_of_shells(&sums);
            assert!(bin < binning.n_bins);
            let total = observables::energy_per_site(spins, lattice, j1, j2, j3, h) * n as f64;
            if binning.exact.is_some() {
                assert_eq!(binning.energy_of_bin(bin), total);
                assert_eq!(binning.total_energy(&sums), total);
            } else {
                assert!((binning.energy_of_bin(bin) - total).abs() <= binning.width / 2.0 + 1e-9);
            }
        }
    }

    #[test]
    fn test_exact_grid_maps_every_configuration_to_its_energy() {
        let square = SquareLattice::new(4).unwrap();
        let chain = ChainLattice::new(12).unwrap();
        let cubic = CubicLattice::new(4).unwrap();
        let triangular = TriangularLattice::new(4).unwrap();
        let honeycomb = HoneycombLattice::new(4).unwrap();
        let sets = [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [1.0, -0.5, 0.0, 0.0],
            [1.0, 0.0, 0.25, 0.0],
            [1.0, 0.0, 0.0, 0.5],
            [1.0, 0.5, 0.25, -0.125],
            [0.0, 1.0, 0.0, 0.0],
        ];
        for set in sets {
            assert_consistent(&square, set, None);
            assert_consistent(&chain, set, None);
            assert_consistent(&cubic, set, None);
            assert_consistent(&triangular, set, None);
            assert_consistent(&honeycomb, set, None);
        }
    }

    #[test]
    fn test_flip_quantum_and_bin_counts() {
        let square = SquareLattice::new(4).unwrap();
        let grid = EnergyBinning::detect(&square, 1.0, 0.0, 0.0, 0.0, None).unwrap();
        assert_eq!(grid.width, 4.0);
        assert_eq!(grid.e_min, -32.0);
        assert_eq!(grid.n_bins, 17);

        let chain = ChainLattice::new(12).unwrap();
        let grid = EnergyBinning::detect(&chain, 1.0, 0.0, 0.0, 0.0, None).unwrap();
        assert_eq!(grid.width, 4.0);
        assert_eq!(grid.n_bins, 7);

        let cubic = CubicLattice::new(4).unwrap();
        let grid = EnergyBinning::detect(&cubic, 1.0, -0.5, 0.0, 0.0, None).unwrap();
        assert_eq!(grid.width, 2.0);
        // All-up sits exactly at the FM/layered frustration energy 0.
        let sums = observables::shell_sums(&all_up(64), &cubic, true, true, false);
        assert_eq!(grid.energy_of_bin(grid.bin_of_shells(&sums)), 0.0);

        let grid = EnergyBinning::detect(&square, 1.0, 0.0, 0.25, 0.0, None).unwrap();
        assert_eq!(grid.width, 1.0);
        let grid = EnergyBinning::detect(&square, 1.0, 0.0, 0.0, 0.5, None).unwrap();
        assert_eq!(grid.width, 1.0);

        let honeycomb = HoneycombLattice::new(4).unwrap();
        let grid = EnergyBinning::detect(&honeycomb, 1.0, 0.0, 0.0, 0.0, None).unwrap();
        assert_eq!(grid.width, 2.0);
        let triangular = TriangularLattice::new(4).unwrap();
        let grid = EnergyBinning::detect(&triangular, 1.0, 0.0, 0.0, 0.0, None).unwrap();
        assert_eq!(grid.width, 4.0);
        let grid = EnergyBinning::detect(&cubic, 1.0, 0.5, 0.25, 0.0, None).unwrap();
        assert_eq!(grid.width, 1.0);
    }

    #[test]
    fn test_non_dyadic_couplings_need_a_width() {
        let square = SquareLattice::new(4).unwrap();
        let err = EnergyBinning::detect(&square, 1.0, 0.3, 0.0, 0.0, None).unwrap_err();
        assert!(matches!(err, MCIsingError::NonDyadicCouplingsNeedBinWidth));
        assert!(err.to_string().contains("bin_width"));
        assert_consistent(&square, [1.0, 0.3, 0.0, 0.0], Some(0.2));
        let grid = EnergyBinning::detect(&square, 1.0, 0.3, 0.0, 0.0, Some(0.2)).unwrap();
        assert!(grid.exact.is_none());
    }

    #[test]
    fn test_width_override_coarsens_the_exact_grid() {
        let square = SquareLattice::new(4).unwrap();
        let grid = EnergyBinning::detect(&square, 1.0, 0.0, 0.0, 0.0, Some(8.0)).unwrap();
        assert!(grid.exact.is_none());
        assert_eq!(grid.n_bins, 9);
        assert_consistent(&square, [1.0, 0.0, 0.0, 0.0], Some(8.0));
    }

    #[test]
    fn test_degenerate_and_oversized_grids_are_rejected() {
        let square = SquareLattice::new(4).unwrap();
        assert!(matches!(
            EnergyBinning::detect(&square, 0.0, 0.0, 0.0, 0.0, None).unwrap_err(),
            MCIsingError::DegenerateEnergySpectrum
        ));
        assert!(matches!(
            EnergyBinning::detect(&square, 1.0, 0.0, 0.0, 0.0, Some(1e-9)).unwrap_err(),
            MCIsingError::TooManyEnergyBins(_, MAX_BINS)
        ));
        assert!(matches!(
            EnergyBinning::detect(&square, 1.0, 0.0, 0.0, 0.0, Some(-1.0)).unwrap_err(),
            MCIsingError::InvalidBinWidth(_)
        ));
        assert!(matches!(
            EnergyBinning::detect(&square, 1.0, 0.0, 0.0, 0.0, Some(f64::NAN)).unwrap_err(),
            MCIsingError::InvalidBinWidth(_)
        ));
    }

    #[test]
    fn test_delta_bin_table_matches_direct_evaluation() {
        let cubic = CubicLattice::new(4).unwrap();
        let sets = [
            [1.0, -0.5, 0.0, 0.0],
            [1.0, 0.5, 0.25, -0.125],
            [0.0, 1.0, 0.0, 0.5],
        ];
        for [j1, j2, j3, h] in sets {
            let grid = EnergyBinning::detect(&cubic, j1, j2, j3, h, None).unwrap();
            let exact = grid.exact.as_ref().unwrap();
            let z = [6, 12, 8];
            let active = [j1 != 0.0, j2 != 0.0, j3 != 0.0];
            let table = DeltaBinTable::build(exact, z, active);
            let mut rng = create_rng(9);
            for trial in 0..200 {
                let spins = random_spins(64, 500 + trial);
                let site = rng.gen_range(0..64);
                let before =
                    observables::shell_sums(&spins, &cubic, active[0], active[1], active[2]);
                let spin = i64::from(spins[site]);
                let local = |nbrs: &[usize], on: bool| -> i64 {
                    if on {
                        nbrs.iter().map(|&j| i64::from(spins[j])).sum()
                    } else {
                        0
                    }
                };
                let s1 = local(cubic.nearest_neighbors(site), active[0]);
                let s2 = local(cubic.next_nearest_neighbors(site), active[1]);
                let s3 = local(cubic.third_nearest_neighbors(site), active[2]);
                let mut flipped = spins.clone();
                flipped[site] = -flipped[site];
                let after =
                    observables::shell_sums(&flipped, &cubic, active[0], active[1], active[2]);
                let direct = grid.bin_of_shells(&after) as i64 - grid.bin_of_shells(&before) as i64;
                assert_eq!(table.delta(table.index(spin, s1, s2, s3)), direct);
            }
        }
    }

    #[test]
    fn test_window_bins_and_errors() {
        let square = SquareLattice::new(4).unwrap();
        let grid = EnergyBinning::detect(&square, 1.0, 0.0, 0.0, 0.0, None).unwrap();
        // Per-site (-1.5, 0.5) selects total energies -24 ..= 8: bins 2 ..= 10.
        let window = grid.window_bins(-1.5, 0.5).unwrap();
        assert_eq!(window, BinWindow { lo: 2, hi: 10 });
        assert_eq!(window.len(), 9);
        assert!(window.contains(2) && window.contains(10));
        assert!(!window.contains(1) && !window.contains(11) && !window.contains(-1));
        // Exactly on bin centres is inclusive on both sides.
        assert_eq!(grid.window_bins(-2.0, 2.0).unwrap(), grid.full_window());
        for (lo, hi) in [
            (-2.5, -2.2),
            (0.3, 0.2),
            (f64::NAN, 1.0),
            (0.0, f64::INFINITY),
        ] {
            assert!(matches!(
                grid.window_bins(lo, hi).unwrap_err(),
                MCIsingError::InvalidEnergyWindow(_, _)
            ));
        }
    }
}
