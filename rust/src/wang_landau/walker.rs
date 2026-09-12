//! Single-spin-flip walkers on the energy grid: the Wang-Landau estimator,
//! the frozen-weight multicanonical chain, and the drive-in used to reach an
//! energy window. All three share one proposal: a random site (never the
//! sequential scan, whose deterministic 2-cycles make it non-ergodic — #32,
//! #26), exact integer shell updates, and a bin computed from the state.

use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

use crate::lattice::Lattice;
use crate::observables::{self, ShellSums};

use super::binning::{BinWindow, DeltaBinTable, EnergyBinning};

/// Mutable state of one walker. Owns its generator, so an orchestration
/// layer can drive many in parallel and exchange configurations between
/// them ([`WalkerState::swap_configuration`]).
pub(crate) struct WalkerState {
    pub(crate) spins: Vec<i8>,
    pub(crate) rng: Xoshiro256StarStar,
    /// Exact ordered-pair shell sums of `spins`.
    pub(crate) shells: ShellSums,
    /// Global bin index; always equal to `binning.bin_of_shells(&shells)`.
    pub(crate) bin: usize,
    pub(crate) accepted: u64,
    pub(crate) attempted: u64,
    /// Production only: proposals into a bin the Wang-Landau stage never
    /// entered (no weight exists there, so the move is rejected).
    pub(crate) rejected_unvisited: u64,
}

impl WalkerState {
    pub(crate) fn new<L: Lattice>(
        spins: Vec<i8>,
        rng: Xoshiro256StarStar,
        lattice: &L,
        binning: &EnergyBinning,
        active: [bool; 3],
    ) -> Self {
        let shells = observables::shell_sums(&spins, lattice, active[0], active[1], active[2]);
        let bin = binning.bin_of_shells(&shells);
        Self {
            spins,
            rng,
            shells,
            bin,
            accepted: 0,
            attempted: 0,
            rejected_unvisited: 0,
        }
    }

    /// A walker starting from this configuration with a fresh generator
    /// and zeroed counters.
    pub(crate) fn spawn(&self, rng: Xoshiro256StarStar) -> Self {
        Self {
            spins: self.spins.clone(),
            rng,
            shells: self.shells,
            bin: self.bin,
            accepted: 0,
            attempted: 0,
            rejected_unvisited: 0,
        }
    }

    /// Exchange configurations (spins, shells, bin) with another walker;
    /// generators and counters stay put (the replica-exchange move).
    pub(crate) fn swap_configuration(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.spins, &mut other.spins);
        std::mem::swap(&mut self.shells, &mut other.shells);
        std::mem::swap(&mut self.bin, &mut other.bin);
    }

    #[inline]
    fn apply(&mut self, proposal: &Proposal) {
        self.spins[proposal.site] = -self.spins[proposal.site];
        self.shells.nn -= 4 * proposal.spin * proposal.s1;
        self.shells.nnn -= 4 * proposal.spin * proposal.s2;
        self.shells.tnn -= 4 * proposal.spin * proposal.s3;
        self.shells.magnetization -= 2 * proposal.spin;
        self.bin = proposal.bin as usize;
    }
}

/// Immutable per-run context shared by every walker.
pub(crate) struct WalkerContext<'a> {
    pub(crate) binning: &'a EnergyBinning,
    /// The exact Δbin table (`None` on the float path).
    pub(crate) table: Option<&'a DeltaBinTable>,
    pub(crate) window: BinWindow,
    pub(crate) num_sites: usize,
    pub(crate) use_nn: bool,
    pub(crate) use_nnn: bool,
    pub(crate) use_tnn: bool,
}

/// One flip proposal: the site, its spin, the local shell sums, and the
/// bin the flipped configuration would occupy (possibly outside the grid's
/// window; on the exact path always inside the grid).
#[derive(Clone, Copy, Debug)]
struct Proposal {
    site: usize,
    spin: i64,
    s1: i64,
    s2: i64,
    s3: i64,
    bin: i64,
}

#[inline]
fn propose<L: Lattice>(
    state: &WalkerState,
    ctx: &WalkerContext<'_>,
    lattice: &L,
    site: usize,
) -> Proposal {
    let spins = &state.spins;
    let spin = i64::from(spins[site]);
    let local =
        |neighbours: &[usize]| -> i64 { neighbours.iter().map(|&j| i64::from(spins[j])).sum() };
    let s1 = if ctx.use_nn {
        local(lattice.nearest_neighbors(site))
    } else {
        0
    };
    let s2 = if ctx.use_nnn {
        local(lattice.next_nearest_neighbors(site))
    } else {
        0
    };
    let s3 = if ctx.use_tnn {
        local(lattice.third_nearest_neighbors(site))
    } else {
        0
    };
    let bin = if let Some(table) = ctx.table {
        state.bin as i64 + table.delta(table.index(spin, s1, s2, s3))
    } else {
        let mut next = state.shells;
        next.nn -= 4 * spin * s1;
        next.nnn -= 4 * spin * s2;
        next.tnn -= 4 * spin * s3;
        next.magnetization -= 2 * spin;
        ctx.binning.bin_of_shells(&next) as i64
    };
    Proposal {
        site,
        spin,
        s1,
        s2,
        s3,
        bin,
    }
}

/// Wang-Landau accumulators of one window (window-local arrays).
pub(crate) struct WlEstimate {
    window: BinWindow,
    /// `ln g` up to a constant; NaN marks a bin never entered.
    log_g: Vec<f64>,
    hist: Vec<u64>,
    n_visited: usize,
}

impl WlEstimate {
    /// A fresh estimate with only `start_bin` known (`ln g = 0`).
    pub(crate) fn new(window: BinWindow, start_bin: usize) -> Self {
        let mut log_g = vec![f64::NAN; window.len()];
        log_g[start_bin - window.lo] = 0.0;
        Self {
            window,
            log_g,
            hist: vec![0; window.len()],
            n_visited: 1,
        }
    }

    /// Start from a supplied global `ln g` (NaN = unknown). A start bin the
    /// supplied estimate does not cover takes the smallest known value so
    /// the walker can leave it.
    pub(crate) fn from_initial(window: BinWindow, initial: &[f64], start_bin: usize) -> Self {
        let mut log_g: Vec<f64> = initial[window.lo..=window.hi].to_vec();
        let start = start_bin - window.lo;
        if log_g[start].is_nan() {
            let floor = log_g
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(f64::INFINITY, f64::min);
            log_g[start] = if floor.is_finite() { floor } else { 0.0 };
        }
        let n_visited = log_g.iter().filter(|v| v.is_finite()).count();
        Self {
            window,
            log_g,
            hist: vec![0; window.len()],
            n_visited,
        }
    }

    pub(crate) fn log_g(&self) -> &[f64] {
        &self.log_g
    }

    pub(crate) fn window(&self) -> BinWindow {
        self.window
    }

    pub(crate) fn histogram(&self) -> &[u64] {
        &self.hist
    }

    /// `ln g` at a global bin: `None` outside the window or never visited.
    pub(crate) fn log_g_at(&self, bin: usize) -> Option<f64> {
        if !self.window.contains(bin as i64) {
            return None;
        }
        let value = self.log_g[bin - self.window.lo];
        value.is_finite().then_some(value)
    }

    /// Replace the estimate by the element-wise mean over `others` and
    /// itself, ignoring unvisited entries (the replica-exchange rule: every
    /// walker of a window carries the window's pooled knowledge).
    pub(crate) fn average_with(estimates: &mut [&mut WlEstimate]) {
        if estimates.len() < 2 {
            return;
        }
        let len = estimates[0].log_g.len();
        let mut mean = vec![f64::NAN; len];
        for (bin, slot) in mean.iter_mut().enumerate() {
            let mut sum = 0.0;
            let mut count = 0usize;
            for estimate in estimates.iter() {
                let value = estimate.log_g[bin];
                if value.is_finite() {
                    sum += value;
                    count += 1;
                }
            }
            if count > 0 {
                *slot = sum / count as f64;
            }
        }
        let n_visited = mean.iter().filter(|v| v.is_finite()).count();
        for estimate in estimates.iter_mut() {
            estimate.log_g.copy_from_slice(&mean);
            estimate.n_visited = n_visited;
        }
    }

    pub(crate) fn n_visited(&self) -> usize {
        self.n_visited
    }

    /// `min H / mean H` over the visited bins (0 when none is visited).
    pub(crate) fn flatness(&self) -> f64 {
        flatness_over_visited(&self.hist, &self.log_g)
    }

    pub(crate) fn reset_histogram(&mut self) {
        self.hist.iter_mut().for_each(|h| *h = 0);
    }

    /// Expand to the full grid: NaN / 0 outside the window.
    pub(crate) fn into_global(self, n_bins: usize) -> (Vec<f64>, Vec<u64>) {
        let mut log_g = vec![f64::NAN; n_bins];
        let mut hist = vec![0u64; n_bins];
        log_g[self.window.lo..=self.window.hi].copy_from_slice(&self.log_g);
        hist[self.window.lo..=self.window.hi].copy_from_slice(&self.hist);
        (log_g, hist)
    }
}

/// `min H / mean H` over the bins whose `ln g` is finite; 0 when none is.
pub(crate) fn flatness_over_visited(hist: &[u64], log_g: &[f64]) -> f64 {
    let mut min = u64::MAX;
    let mut sum: u64 = 0;
    let mut count: u64 = 0;
    for (&h, &lg) in hist.iter().zip(log_g) {
        if lg.is_finite() {
            min = min.min(h);
            sum += h;
            count += 1;
        }
    }
    if count == 0 || sum == 0 {
        return 0.0;
    }
    min as f64 * count as f64 / sum as f64
}

/// Energy-space round-trip counter between two bins, the analogue of the
/// parallel-tempering round trip: a walker is labelled `Up` when it sits
/// on the low edge and `Down` when, labelled `Up`, it reaches the high
/// edge; a round trip is counted each time a `Down` walker is back on the
/// low edge.
pub(crate) struct EdgeTracker {
    lo: usize,
    hi: usize,
    state: Edge,
    pub(crate) round_trips: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Edge {
    Unlabelled,
    Up,
    Down,
}

impl EdgeTracker {
    pub(crate) fn new(lo: usize, hi: usize) -> Self {
        Self {
            lo,
            hi,
            state: Edge::Unlabelled,
            round_trips: 0,
        }
    }

    #[inline]
    pub(crate) fn observe(&mut self, bin: usize) {
        if bin == self.lo {
            if self.state == Edge::Down {
                self.round_trips += 1;
            }
            self.state = Edge::Up;
        } else if bin == self.hi && self.state == Edge::Up {
            self.state = Edge::Down;
        }
    }
}

/// One Wang-Landau sweep: `num_sites` random-site proposals with acceptance
/// `min(1, g(E)/g(E'))`, updating `ln g` and the histogram of the current
/// bin after every proposal. A proposal leaving the window is rejected and
/// still counts as a visit of the current bin; a bin entered for the first
/// time starts from the `ln g` of the bin the walker came from.
pub(crate) fn wl_sweep<L: Lattice>(
    state: &mut WalkerState,
    estimate: &mut WlEstimate,
    ctx: &WalkerContext<'_>,
    lattice: &L,
    log_f: f64,
) {
    let lo = estimate.window.lo;
    for _ in 0..ctx.num_sites {
        let site = state.rng.gen_range(0..ctx.num_sites);
        let proposal = propose(state, ctx, lattice, site);
        state.attempted += 1;
        if ctx.window.contains(proposal.bin) {
            let new_bin = proposal.bin as usize;
            let log_g_old = estimate.log_g[state.bin - lo];
            let slot = &mut estimate.log_g[new_bin - lo];
            if slot.is_nan() {
                *slot = log_g_old;
                estimate.n_visited += 1;
            }
            let log_g_new = *slot;
            let accept =
                log_g_new <= log_g_old || state.rng.gen::<f64>() < (log_g_old - log_g_new).exp();
            if accept {
                state.apply(&proposal);
                state.accepted += 1;
            }
        }
        let current = state.bin - lo;
        estimate.log_g[current] += log_f;
        estimate.hist[current] += 1;
    }
}

/// One multicanonical sweep with frozen weights `W(E) = 1/g(E)`: the same
/// proposal and acceptance as [`wl_sweep`] but `ln g` is read-only. A bin
/// the Wang-Landau stage never entered has no weight and is rejected.
pub(crate) fn muca_sweep<L: Lattice>(
    state: &mut WalkerState,
    ctx: &WalkerContext<'_>,
    lattice: &L,
    log_g: &[f64],
    hist: &mut [u64],
    edges: &mut EdgeTracker,
) {
    let lo = ctx.window.lo;
    for _ in 0..ctx.num_sites {
        let site = state.rng.gen_range(0..ctx.num_sites);
        let proposal = propose(state, ctx, lattice, site);
        state.attempted += 1;
        if ctx.window.contains(proposal.bin) {
            let log_g_new = log_g[proposal.bin as usize - lo];
            if log_g_new.is_nan() {
                state.rejected_unvisited += 1;
            } else {
                let log_g_old = log_g[state.bin - lo];
                let accept = log_g_new <= log_g_old
                    || state.rng.gen::<f64>() < (log_g_old - log_g_new).exp();
                if accept {
                    state.apply(&proposal);
                    state.accepted += 1;
                }
            }
        }
        hist[state.bin - lo] += 1;
        edges.observe(state.bin);
    }
}

/// One Metropolis sweep at inverse temperature `beta` (negative to raise
/// the energy) that returns as soon as the walker is inside the window.
pub(crate) fn drive_sweep<L: Lattice>(
    state: &mut WalkerState,
    ctx: &WalkerContext<'_>,
    lattice: &L,
    beta: f64,
) -> bool {
    if ctx.window.contains(state.bin as i64) {
        return true;
    }
    for _ in 0..ctx.num_sites {
        let site = state.rng.gen_range(0..ctx.num_sites);
        let proposal = propose(state, ctx, lattice, site);
        let delta = ctx
            .binning
            .flip_energy(proposal.spin, proposal.s1, proposal.s2, proposal.s3);
        let accept = beta * delta <= 0.0 || state.rng.gen::<f64>() < (-beta * delta).exp();
        if accept {
            state.apply(&proposal);
            if ctx.window.contains(state.bin as i64) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    // Bit-identity of grid energies and weights is the contract these
    // tests pin, so exact float comparison is the point.
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn test_flatness_definition() {
        let visited = [0.0, 0.0, 0.0];
        assert!((flatness_over_visited(&[10, 8, 12], &visited) - 0.8).abs() < 1e-12);
        assert_eq!(flatness_over_visited(&[10, 0, 12], &visited), 0.0);
        // Unvisited (NaN) bins are ignored entirely.
        assert!(
            (flatness_over_visited(&[10, 0, 12], &[0.0, f64::NAN, 0.0]) - 10.0 / 11.0).abs()
                < 1e-12
        );
        assert_eq!(flatness_over_visited(&[0, 0], &[f64::NAN, f64::NAN]), 0.0);
        assert_eq!(flatness_over_visited(&[0, 0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_late_discovery_starts_from_the_current_bin_and_edges() {
        let window = BinWindow { lo: 3, hi: 7 };
        let mut est = WlEstimate::new(window, 5);
        assert_eq!(est.n_visited(), 1);
        est.log_g[2] = 4.5;
        // A neighbour discovered now inherits 4.5, not 0.
        let origin = est.log_g[2];
        let slot = &mut est.log_g[3];
        assert!(slot.is_nan());
        *slot = origin;
        est.n_visited += 1;
        assert_eq!(est.log_g[3], 4.5);
        let (log_g, hist) = est.into_global(10);
        assert_eq!(log_g.len(), 10);
        assert!(log_g[0].is_nan() && log_g[9].is_nan());
        assert_eq!(log_g[5], 4.5);
        assert_eq!(hist, vec![0; 10]);
    }

    #[test]
    fn test_from_initial_fills_an_unknown_start_bin() {
        let window = BinWindow { lo: 0, hi: 3 };
        let initial = [f64::NAN, 2.0, 5.0, f64::NAN];
        let est = WlEstimate::from_initial(window, &initial, 0);
        assert_eq!(est.log_g()[0], 2.0);
        assert_eq!(est.n_visited(), 3);
        let est = WlEstimate::from_initial(window, &[f64::NAN; 4], 3);
        assert_eq!(est.log_g()[3], 0.0);
        assert_eq!(est.n_visited(), 1);
    }

    #[test]
    fn test_swap_configuration_exchanges_state_but_not_generators() {
        use crate::lattice::square::SquareLattice;
        use crate::rng::create_rng;
        let lattice = SquareLattice::new(4).unwrap();
        let binning = EnergyBinning::detect(&lattice, 1.0, 0.0, 0.0, 0.0, None).unwrap();
        let active = [true, false, false];
        let mut up = WalkerState::new(vec![1; 16], create_rng(1), &lattice, &binning, active);
        let mut mixed: Vec<i8> = vec![1; 16];
        mixed[0] = -1;
        let mut other = WalkerState::new(mixed.clone(), create_rng(2), &lattice, &binning, active);
        let (bin_up, bin_other) = (up.bin, other.bin);
        up.attempted = 7;
        up.swap_configuration(&mut other);
        assert_eq!(up.spins, mixed);
        assert_eq!(other.spins, vec![1; 16]);
        assert_eq!((up.bin, other.bin), (bin_other, bin_up));
        assert_eq!(
            up.shells,
            observables::shell_sums(&up.spins, &lattice, true, false, false)
        );
        assert_eq!(up.attempted, 7);
        assert_eq!(other.attempted, 0);
    }

    #[test]
    fn test_edge_tracker_counts_low_high_low_excursions() {
        let mut edges = EdgeTracker::new(2, 9);
        for bin in [5, 9, 9, 5, 2] {
            edges.observe(bin);
        }
        // Reaching the high edge before ever touching the low one labels
        // nothing; the first visit of the low edge merely labels Up.
        assert_eq!(edges.round_trips, 0);
        for bin in [4, 9, 6, 2] {
            edges.observe(bin);
        }
        assert_eq!(edges.round_trips, 1);
        for bin in [2, 2, 9, 2, 9, 2] {
            edges.observe(bin);
        }
        assert_eq!(edges.round_trips, 3);
    }
}
