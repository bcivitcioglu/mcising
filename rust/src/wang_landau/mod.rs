//! Wang-Landau density-of-states sampling followed by a multicanonical
//! production run with frozen weights.
//!
//! Stage 1 (Wang-Landau) estimates `ln g(E)` on the exact energy grid of
//! `binning` with the flat-histogram random walk of Wang & Landau (PRL 86,
//! 2050 (2001)): every visit adds `ln f` to the current bin, the histogram
//! is flat when its minimum reaches `flatness` times its mean, `ln f`
//! halves at each flat histogram, and once `ln f` falls below `1/t` the
//! Belardinelli–Pereyra schedule `ln f = 1/t` takes over (PRE 75, 046701
//! (2007); `t` counts mean visits per bin of the window). Stage 2 freezes
//! the weights `W(E) = 1/g(E)` and runs `n_walkers` independent
//! multicanonical chains (Berg & Neuhaus, PRL 68, 9 (1992)) that record the
//! energy, magnetization and staggered magnetizations of every measurement;
//! the frozen-weight chain is an exact Markov chain, so canonical averages
//! at any temperature follow by reweighting the series in Python.
//!
//! Everything user-reachable is validated before any work starts; the hot
//! loops are panic-free for validated input.

mod binning;
mod walker;

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::error::MCIsingError;
use crate::lattice::{with_lattice, Lattice, LatticeKind};
use crate::measurement::TempResult;
use crate::observables;
use crate::rng::create_rng;
use crate::simulation::IsingSimulation;

use binning::{BinWindow, DeltaBinTable, EnergyBinning};
use walker::{
    drive_sweep, flatness_over_visited, muca_sweep, wl_sweep, EdgeTracker, WalkerContext,
    WalkerState, WlEstimate,
};

/// Production walker `k` is seeded `base_seed + PRODUCTION_SEED_OFFSET + k`
/// (the parallel-tempering swap generator uses the same convention), so
/// its stream never coincides with the Wang-Landau walker's.
pub(crate) const PRODUCTION_SEED_OFFSET: u64 = 1000;

/// The Wang-Landau modification-factor schedule.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WangLandauSchedule {
    pub(crate) log_f_initial: f64,
    pub(crate) log_f_final: f64,
    pub(crate) flatness: f64,
    /// Sweeps between flatness checks.
    pub(crate) check_interval: u64,
    /// Cap on the stage; `None` runs to `log_f_final`.
    pub(crate) max_sweeps: Option<u64>,
}

/// Every input of one run (validated by [`run_wang_landau_internal`]).
pub(crate) struct WangLandauParams<'a> {
    pub(crate) lattice_size: usize,
    pub(crate) j1: f64,
    pub(crate) j2: f64,
    pub(crate) j3: f64,
    pub(crate) h: f64,
    pub(crate) base_seed: u64,
    pub(crate) lattice_type: &'a str,
    /// Total-energy units; required for non-dyadic couplings.
    pub(crate) bin_width: Option<f64>,
    /// Per-site (lo, hi).
    pub(crate) energy_window: Option<(f64, f64)>,
    /// Global `ln g` to start from (NaN = unknown), e.g. a previous run's.
    pub(crate) initial_log_g: Option<&'a [f64]>,
    pub(crate) schedule: WangLandauSchedule,
    pub(crate) drive_beta: f64,
    pub(crate) drive_max_sweeps: u64,
    /// Overlapping energy windows of the replica-exchange stage (1 = the
    /// serial walker).
    pub(crate) n_windows: usize,
    /// Walkers per window; their estimates are averaged at every check.
    pub(crate) walkers_per_window: usize,
    /// Fraction of a window's length shared with its neighbour.
    pub(crate) window_overlap: f64,
    /// Sweeps between replica-exchange attempts.
    pub(crate) exchange_interval: u64,
    pub(crate) n_walkers: usize,
    pub(crate) production_sweeps: u64,
    pub(crate) production_thermalization: u64,
    pub(crate) measurement_interval: u64,
    pub(crate) store_configs: bool,
}

/// Stage-1 diagnostics: one record per modification-factor value (the 1/t
/// stage is the last record).
#[derive(Debug, Default)]
pub(crate) struct WlDiagnostics {
    pub(crate) iteration_log_f: Vec<f64>,
    pub(crate) iteration_sweeps: Vec<u64>,
    pub(crate) iteration_flatness: Vec<f64>,
    pub(crate) iteration_visited_bins: Vec<usize>,
    pub(crate) total_sweeps: u64,
    pub(crate) one_over_t_switch_sweep: Option<u64>,
    pub(crate) final_log_f: f64,
    pub(crate) converged: bool,
    pub(crate) accepted: u64,
    pub(crate) attempted: u64,
    pub(crate) drive_in_sweeps: u64,
    pub(crate) visited_bins: usize,
    pub(crate) n_windows: usize,
    pub(crate) walkers_per_window: usize,
    /// Inclusive bin range of every window.
    pub(crate) window_bins: Vec<(usize, usize)>,
    /// Replica-exchange attempts and acceptances per adjacent window pair.
    pub(crate) exchange_attempted: Vec<u64>,
    pub(crate) exchange_accepted: Vec<u64>,
    /// Bin at which each window's estimate was joined to the previous one.
    pub(crate) merge_bins: Vec<usize>,
}

impl WlDiagnostics {
    fn close_iteration(&mut self, log_f: f64, sweeps: u64, flatness: f64, visited: usize) {
        self.iteration_log_f.push(log_f);
        self.iteration_sweeps.push(sweeps);
        self.iteration_flatness.push(flatness);
        self.iteration_visited_bins.push(visited);
    }

    fn into_pydict(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("iteration_log_f", self.iteration_log_f)?;
        dict.set_item("iteration_sweeps", self.iteration_sweeps)?;
        dict.set_item("iteration_flatness", self.iteration_flatness)?;
        dict.set_item("iteration_visited_bins", self.iteration_visited_bins)?;
        dict.set_item("total_sweeps", self.total_sweeps)?;
        dict.set_item("one_over_t_switch_sweep", self.one_over_t_switch_sweep)?;
        dict.set_item("final_log_f", self.final_log_f)?;
        dict.set_item("converged", self.converged)?;
        dict.set_item("accepted", self.accepted)?;
        dict.set_item("attempted", self.attempted)?;
        dict.set_item("drive_in_sweeps", self.drive_in_sweeps)?;
        dict.set_item("visited_bins", self.visited_bins)?;
        dict.set_item("n_windows", self.n_windows)?;
        dict.set_item("walkers_per_window", self.walkers_per_window)?;
        dict.set_item("window_bins", self.window_bins)?;
        dict.set_item("exchange_attempted", self.exchange_attempted)?;
        dict.set_item("exchange_accepted", self.exchange_accepted)?;
        dict.set_item("merge_bins", self.merge_bins)?;
        Ok(dict)
    }
}

/// Stage-2 diagnostics, one entry per walker where applicable.
#[derive(Debug)]
pub(crate) struct ProductionDiagnostics {
    pub(crate) n_walkers: usize,
    pub(crate) sweeps_per_walker: u64,
    pub(crate) thermalization_sweeps: u64,
    pub(crate) accepted: Vec<u64>,
    pub(crate) attempted: Vec<u64>,
    pub(crate) rejected_unvisited: Vec<u64>,
    pub(crate) round_trips: Vec<u64>,
    /// `min H / mean H` of the pooled production histogram over the bins
    /// the Wang-Landau stage visited.
    pub(crate) histogram_flatness: f64,
    /// The bins between which round trips are counted (lowest and highest
    /// Wang-Landau-visited bins).
    pub(crate) edge_bins: Option<(usize, usize)>,
}

impl ProductionDiagnostics {
    fn into_pydict(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("n_walkers", self.n_walkers)?;
        dict.set_item("sweeps_per_walker", self.sweeps_per_walker)?;
        dict.set_item("thermalization_sweeps", self.thermalization_sweeps)?;
        dict.set_item("accepted", self.accepted)?;
        dict.set_item("attempted", self.attempted)?;
        dict.set_item("rejected_unvisited", self.rejected_unvisited)?;
        dict.set_item("round_trips", self.round_trips)?;
        dict.set_item("histogram_flatness", self.histogram_flatness)?;
        dict.set_item("edge_bins", self.edge_bins)?;
        Ok(dict)
    }
}

/// The production series of one walker.
pub(crate) struct MucaSeries {
    pub(crate) walker: usize,
    /// Built with `temperature = NaN`; only its series are reported.
    pub(crate) series: TempResult,
    /// Global bin index at every measurement.
    pub(crate) bins: Vec<u32>,
    pub(crate) accepted: u64,
    pub(crate) attempted: u64,
    pub(crate) round_trips: u64,
}

impl MucaSeries {
    fn into_pydict(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("walker", self.walker)?;
        dict.set_item("bin_index", self.bins.into_pyarray(py))?;
        dict.set_item("accepted", self.accepted)?;
        dict.set_item("attempted", self.attempted)?;
        dict.set_item("round_trips", self.round_trips)?;
        self.series.write_series(py, &dict)?;
        Ok(dict)
    }
}

/// Everything one run produces.
pub(crate) struct WangLandauResult {
    pub(crate) binning: EnergyBinning,
    pub(crate) window: BinWindow,
    pub(crate) energy_window: Option<(f64, f64)>,
    /// Global; NaN where the Wang-Landau stage never entered.
    pub(crate) log_g: Vec<f64>,
    /// Wang-Landau histogram since its last reset.
    pub(crate) wl_histogram: Vec<u64>,
    /// Production histogram pooled over walkers (production sweeps only).
    pub(crate) production_histogram: Vec<u64>,
    pub(crate) wl: WlDiagnostics,
    /// The Wang-Landau walker's final configuration and generator state
    /// (the resume hooks).
    pub(crate) final_spins: Vec<i8>,
    pub(crate) final_rng_state: Vec<u8>,
    pub(crate) production: Vec<MucaSeries>,
    pub(crate) production_diagnostics: ProductionDiagnostics,
}

impl WangLandauResult {
    /// Convert to the dict `run_wang_landau` returns. Keys: `energy_bins`
    /// (per site), `bin_width` (total energy), `num_sites`, `window_bins`,
    /// `energy_window`, `log_g`, `wl_histogram`, `production_histogram`,
    /// `wl_diagnostics`, `final_spins`, `final_rng_state`, `walkers` (one
    /// dict per walker with `walker`, `energies`, `magnetizations`,
    /// `staggered_magnetizations`, `bin_index`, `accepted`, `attempted`,
    /// `round_trips` and `configurations` when stored) and
    /// `production_diagnostics`.
    pub(crate) fn into_pydict(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item(
            "energy_bins",
            self.binning.energies_per_site().into_pyarray(py),
        )?;
        dict.set_item("bin_width", self.binning.width)?;
        dict.set_item("num_sites", self.binning.num_sites)?;
        dict.set_item("window_bins", (self.window.lo, self.window.hi))?;
        dict.set_item("energy_window", self.energy_window)?;
        dict.set_item("log_g", self.log_g.into_pyarray(py))?;
        dict.set_item("wl_histogram", self.wl_histogram.into_pyarray(py))?;
        dict.set_item(
            "production_histogram",
            self.production_histogram.into_pyarray(py),
        )?;
        dict.set_item("wl_diagnostics", self.wl.into_pydict(py)?)?;
        dict.set_item("final_spins", self.final_spins.into_pyarray(py))?;
        dict.set_item("final_rng_state", self.final_rng_state)?;
        let walkers = self
            .production
            .into_iter()
            .map(|series| series.into_pydict(py))
            .collect::<PyResult<Vec<_>>>()?;
        dict.set_item("walkers", walkers)?;
        dict.set_item(
            "production_diagnostics",
            self.production_diagnostics.into_pydict(py)?,
        )?;
        Ok(dict)
    }
}

/// Reject every user-reachable parameter error before any work starts.
fn validate(p: &WangLandauParams<'_>) -> Result<(), MCIsingError> {
    if p.measurement_interval < 1 {
        return Err(MCIsingError::InvalidInterval("measurement_interval", 0));
    }
    if p.n_walkers < 1 {
        return Err(MCIsingError::InvalidInterval("n_walkers", 0));
    }
    if p.schedule.check_interval < 1 {
        return Err(MCIsingError::InvalidInterval("check_interval", 0));
    }
    if p.drive_max_sweeps < 1 {
        return Err(MCIsingError::InvalidInterval("drive_max_sweeps", 0));
    }
    let flatness = p.schedule.flatness;
    if !(flatness.is_finite() && flatness > 0.0 && flatness <= 1.0) {
        return Err(MCIsingError::InvalidFlatness(flatness));
    }
    let initial = p.schedule.log_f_initial;
    if !(initial.is_finite() && initial > 0.0) {
        return Err(MCIsingError::InvalidLogF("log_f_initial", initial));
    }
    let final_log_f = p.schedule.log_f_final;
    if !(final_log_f.is_finite() && final_log_f > 0.0 && final_log_f < initial) {
        return Err(MCIsingError::InvalidLogF("log_f_final", final_log_f));
    }
    if !(p.drive_beta.is_finite() && p.drive_beta > 0.0) {
        return Err(MCIsingError::InvalidDriveBeta(p.drive_beta));
    }
    if p.n_windows < 1 {
        return Err(MCIsingError::InvalidInterval("n_windows", 0));
    }
    if p.walkers_per_window < 1 {
        return Err(MCIsingError::InvalidInterval("walkers_per_window", 0));
    }
    if p.exchange_interval < 1 {
        return Err(MCIsingError::InvalidInterval("exchange_interval", 0));
    }
    if !(p.window_overlap.is_finite() && (0.0..1.0).contains(&p.window_overlap)) {
        return Err(MCIsingError::InvalidWindowOverlap(p.window_overlap));
    }
    let parallel = p.n_windows > 1 || p.walkers_per_window > 1;
    if parallel
        && !p
            .schedule
            .check_interval
            .is_multiple_of(p.exchange_interval)
    {
        return Err(MCIsingError::IncompatibleCheckCadence(
            p.schedule.check_interval,
            p.exchange_interval,
        ));
    }
    Ok(())
}

/// What either first stage hands to production: the seed configurations,
/// the global `ln g`, the histogram since the last reset, and the record.
struct StageOutput {
    seeds: Vec<WalkerState>,
    log_g: Vec<f64>,
    histogram: Vec<u64>,
    diagnostics: WlDiagnostics,
}

/// The modification-factor schedule shared by the serial and the
/// replica-exchange stage: flatness-driven halving of `ln f`, then the
/// Belardinelli–Pereyra `1/t` law once `ln f` falls below `1/t` while the
/// set of visited bins is stable.
struct ScheduleState {
    log_f: f64,
    one_over_t: bool,
    iteration_start: u64,
    iteration_log_f: f64,
    visited_at_last_check: usize,
}

/// Outcome of a flatness check.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// Nothing changed.
    Continue,
    /// The histograms were flat: `ln f` halved, histograms must be reset.
    Flat,
    /// The `1/t` schedule took over (no reset).
    Switched,
    /// `ln f` reached its final value.
    Converged,
}

impl ScheduleState {
    fn new(schedule: &WangLandauSchedule, visited: usize) -> Self {
        Self {
            log_f: schedule.log_f_initial,
            one_over_t: false,
            iteration_start: 0,
            iteration_log_f: schedule.log_f_initial,
            visited_at_last_check: visited,
        }
    }

    /// In the `1/t` stage: refresh `ln f`; `true` once it is final.
    fn tick(&mut self, t_inv: f64, schedule: &WangLandauSchedule) -> bool {
        self.log_f = t_inv;
        self.log_f < schedule.log_f_final
    }

    fn switch(&mut self, sweeps: u64, t_inv: f64, diagnostics: &mut WlDiagnostics) {
        self.one_over_t = true;
        diagnostics.one_over_t_switch_sweep = Some(sweeps);
        self.log_f = t_inv;
        self.iteration_start = sweeps;
        self.iteration_log_f = self.log_f;
    }

    /// A flatness check at sweep `sweeps` (not in the `1/t` stage).
    fn check(
        &mut self,
        sweeps: u64,
        t_inv: f64,
        flat: f64,
        visited: usize,
        schedule: &WangLandauSchedule,
        diagnostics: &mut WlDiagnostics,
    ) -> Verdict {
        let visited_stable = visited == self.visited_at_last_check;
        self.visited_at_last_check = visited;
        if self.log_f <= t_inv && visited_stable {
            diagnostics.close_iteration(
                self.iteration_log_f,
                sweeps - self.iteration_start,
                flat,
                visited,
            );
            self.switch(sweeps, t_inv, diagnostics);
            return if self.log_f < schedule.log_f_final {
                Verdict::Converged
            } else {
                Verdict::Switched
            };
        }
        if flat >= schedule.flatness {
            diagnostics.close_iteration(
                self.iteration_log_f,
                sweeps - self.iteration_start,
                flat,
                visited,
            );
            self.log_f *= 0.5;
            self.iteration_start = sweeps;
            self.iteration_log_f = self.log_f;
            if self.log_f < schedule.log_f_final {
                return Verdict::Converged;
            }
            if self.log_f <= t_inv && visited_stable {
                self.switch(sweeps, t_inv, diagnostics);
            }
            return Verdict::Flat;
        }
        Verdict::Continue
    }

    /// Close the last record.
    fn finish(self, sweeps: u64, flat: f64, visited: usize, diagnostics: &mut WlDiagnostics) {
        diagnostics.close_iteration(
            self.iteration_log_f,
            sweeps - self.iteration_start,
            flat,
            visited,
        );
        diagnostics.total_sweeps = sweeps;
        diagnostics.final_log_f = self.log_f;
    }
}

/// The runner (pure Rust; the `#[pyfunction]` wrapper releases the GIL
/// around it).
///
/// # Errors
///
/// Every validation error of the parameters, the lattice and the couplings
/// (`IsingSimulation::new_internal`), the energy grid
/// (`EnergyBinning::detect`, `window_bins`), a mis-sized `initial_log_g`,
/// and `EnergyWindowUnreachable` when the drive-in cannot enter the window.
pub(crate) fn run_wang_landau_internal(
    p: &WangLandauParams<'_>,
) -> Result<WangLandauResult, MCIsingError> {
    validate(p)?;
    let sim = IsingSimulation::new_internal(
        p.lattice_size,
        p.j1,
        p.j2,
        p.j3,
        p.h,
        p.base_seed,
        "metropolis",
        p.lattice_type,
    )?;
    let (spins, lattice, rng) = sim.into_parts();
    with_lattice!(&lattice, lat => run_with_lattice(lat, spins, rng, p))
}

fn run_with_lattice<L: Lattice>(
    lattice: &L,
    spins: Vec<i8>,
    rng: rand_xoshiro::Xoshiro256StarStar,
    p: &WangLandauParams<'_>,
) -> Result<WangLandauResult, MCIsingError> {
    let binning = EnergyBinning::detect(lattice, p.j1, p.j2, p.j3, p.h, p.bin_width)?;
    let window = match p.energy_window {
        Some((lo, hi)) => binning.window_bins(lo, hi)?,
        None => binning.full_window(),
    };
    if let Some(initial) = p.initial_log_g {
        if initial.len() != binning.n_bins {
            return Err(MCIsingError::InvalidInitialLogG(format!(
                "expected one entry per energy bin ({}), got {}",
                binning.n_bins,
                initial.len()
            )));
        }
        if initial.iter().any(|v| v.is_infinite()) {
            return Err(MCIsingError::InvalidInitialLogG(
                "entries must be finite or NaN (unknown)".to_string(),
            ));
        }
    }
    let z = [
        lattice.coordination_number(),
        lattice.nnn_coordination_number(),
        lattice.tnn_coordination_number(),
    ];
    let active = [p.j1 != 0.0, p.j2 != 0.0, p.j3 != 0.0];
    let table = binning
        .exact
        .as_ref()
        .map(|exact| DeltaBinTable::build(exact, z, active));
    let ctx = WalkerContext {
        binning: &binning,
        table: table.as_ref(),
        window,
        num_sites: lattice.num_sites(),
        use_nn: active[0],
        use_nnn: active[1],
        use_tnn: active[2],
    };
    let parallel = p.n_windows > 1 || p.walkers_per_window > 1;
    let stage = if parallel {
        run_parallel_stage(spins, rng, lattice, &binning, &ctx, window, active, p)?
    } else {
        let mut state = WalkerState::new(spins, rng, lattice, &binning, active);
        let drive_in_sweeps = drive_in(&mut state, &ctx, lattice, p)?;
        let mut estimate = match p.initial_log_g {
            Some(initial) => WlEstimate::from_initial(window, initial, state.bin),
            None => WlEstimate::new(window, state.bin),
        };
        let mut wl = run_wang_landau_stage(&mut state, &mut estimate, &ctx, lattice, &p.schedule);
        wl.drive_in_sweeps = drive_in_sweeps;
        wl.accepted = state.accepted;
        wl.attempted = state.attempted;
        wl.visited_bins = estimate.n_visited();
        wl.n_windows = 1;
        wl.walkers_per_window = 1;
        wl.window_bins = vec![(window.lo, window.hi)];
        let (log_g, wl_histogram) = estimate.into_global(binning.n_bins);
        StageOutput {
            seeds: vec![state],
            log_g,
            histogram: wl_histogram,
            diagnostics: wl,
        }
    };

    let shape = lattice.shape().to_vec();
    let frozen = &stage.log_g[window.lo..=window.hi];
    let (production, production_histogram, production_diagnostics) =
        run_production(&stage.seeds, frozen, &ctx, lattice, p, &shape);
    // Infallible in practice: a fixed 4×u64 generator state (see
    // `IsingSimulation::get_rng_state`).
    let final_rng_state = serde_json::to_vec(&stage.seeds[0].rng)
        .expect("Xoshiro256StarStar serialization should not fail");
    let final_spins = stage
        .seeds
        .into_iter()
        .next()
        .map(|s| s.spins)
        .unwrap_or_default();
    Ok(WangLandauResult {
        binning,
        window,
        energy_window: p.energy_window,
        log_g: stage.log_g,
        wl_histogram: stage.histogram,
        production_histogram,
        wl: stage.diagnostics,
        final_spins,
        final_rng_state,
        production,
        production_diagnostics,
    })
}

/// Seed offset of the Wang-Landau walkers of the parallel stage (walker `i`
/// draws from `base_seed + 1 + i`; production walkers start at
/// `PRODUCTION_SEED_OFFSET`, the exchange generator uses `EXCHANGE_SEED_OFFSET`).
const WL_WALKER_SEED_OFFSET: u64 = 1;
const EXCHANGE_SEED_OFFSET: u64 = 2000;

/// Overlapping windows of equal length covering `window`: consecutive
/// windows start `(1 - overlap)` window lengths apart and the last one
/// ends at the top of the range.
///
/// # Errors
///
/// `InvalidWindowSplit` when a window would hold fewer than two bins or
/// adjacent windows would not share a bin.
fn split_windows(
    window: BinWindow,
    n_windows: usize,
    overlap: f64,
) -> Result<Vec<BinWindow>, MCIsingError> {
    if n_windows <= 1 {
        return Ok(vec![window]);
    }
    let span = window.len() as f64;
    let width = span / (1.0 + (n_windows as f64 - 1.0) * (1.0 - overlap));
    let step = width * (1.0 - overlap);
    let mut windows = Vec::with_capacity(n_windows);
    for index in 0..n_windows {
        let lo = (window.lo as f64 + index as f64 * step).round() as usize;
        let hi = if index == n_windows - 1 {
            window.hi
        } else {
            ((lo as f64 + width - 1.0).round() as usize).min(window.hi)
        };
        if hi < lo + 1 {
            return Err(MCIsingError::InvalidWindowSplit(format!(
                "{n_windows} windows over {} bins leave window {index} with fewer \
                 than two bins; use fewer windows or a wider energy range",
                window.len()
            )));
        }
        windows.push(BinWindow { lo, hi });
    }
    for pair in windows.windows(2) {
        if pair[1].lo > pair[0].hi {
            return Err(MCIsingError::InvalidWindowSplit(format!(
                "windows {:?} and {:?} do not overlap; raise window_overlap",
                (pair[0].lo, pair[0].hi),
                (pair[1].lo, pair[1].hi)
            )));
        }
    }
    Ok(windows)
}

/// One walker of the replica-exchange stage.
struct RewlWalker {
    state: WalkerState,
    estimate: WlEstimate,
    window: usize,
}

/// Join the per-window estimates into one global `ln g`: each window is
/// shifted onto the previous one at the overlap bin where their slopes
/// `d ln g / dE` agree best, and takes over from that bin on (Vogel, Li,
/// Wüst & Landau, PRL 110, 210603 (2013)).
fn merge_windows(pieces: &[(BinWindow, Vec<f64>)], n_bins: usize) -> (Vec<f64>, Vec<usize>) {
    let mut global = vec![f64::NAN; n_bins];
    let mut merge_bins = Vec::new();
    let Some((first_window, first)) = pieces.first() else {
        return (global, merge_bins);
    };
    global[first_window.lo..=first_window.hi].copy_from_slice(first);
    let mut previous_hi = first_window.hi;
    for (window, piece) in &pieces[1..] {
        let local = |bin: usize| piece[bin - window.lo];
        // Candidate join bins: overlap bins where both slopes exist.
        let mut best: Option<(f64, usize)> = None;
        for bin in window.lo..previous_hi.min(window.hi) {
            let slope_prev = global[bin + 1] - global[bin];
            let slope_new = local(bin + 1) - local(bin);
            if !(slope_prev.is_finite() && slope_new.is_finite()) {
                continue;
            }
            let mismatch = (slope_prev - slope_new).abs();
            if best.is_none_or(|(m, _)| mismatch < m) {
                best = Some((mismatch, bin));
            }
        }
        let join = best.map(|(_, bin)| bin).or_else(|| {
            // No common slope: join at the first overlap bin both know.
            (window.lo..=previous_hi.min(window.hi))
                .find(|&bin| global[bin].is_finite() && local(bin).is_finite())
        });
        let Some(join) = join else {
            // Nothing in common: keep the piece as is (its offset is arbitrary).
            let range = window.lo..=window.hi;
            for (bin, slot) in global.iter_mut().enumerate() {
                if range.contains(&bin) && slot.is_nan() {
                    *slot = local(bin);
                }
            }
            merge_bins.push(window.lo);
            previous_hi = window.hi;
            continue;
        };
        let shift = global[join] - local(join);
        let range = join..=window.hi;
        for (bin, slot) in global.iter_mut().enumerate() {
            if !range.contains(&bin) {
                continue;
            }
            let value = local(bin);
            if value.is_finite() {
                *slot = value + shift;
            }
        }
        merge_bins.push(join);
        previous_hi = window.hi;
    }
    (global, merge_bins)
}

/// Attempt one replica exchange between random walkers of each adjacent
/// window pair of the current parity.
fn attempt_exchanges(
    walkers: &mut [RewlWalker],
    windows: &[BinWindow],
    round: u64,
    rng: &mut rand_xoshiro::Xoshiro256StarStar,
    attempted: &mut [u64],
    accepted: &mut [u64],
) {
    use rand::Rng;
    let per_window = walkers.len() / windows.len();
    let offset = usize::from(!round.is_multiple_of(2));
    for pair in (offset..windows.len().saturating_sub(1)).step_by(2) {
        let a = pair * per_window + rng.gen_range(0..per_window);
        let b = (pair + 1) * per_window + rng.gen_range(0..per_window);
        attempted[pair] += 1;
        let (bin_a, bin_b) = (walkers[a].state.bin, walkers[b].state.bin);
        if !(windows[pair].contains(bin_b as i64) && windows[pair + 1].contains(bin_a as i64)) {
            continue;
        }
        let (lo, hi) = walkers.split_at_mut(b);
        let (walker_a, walker_b) = (&mut lo[a], &mut hi[0]);
        let (Some(a_at_a), Some(a_at_b), Some(b_at_b), Some(b_at_a)) = (
            walker_a.estimate.log_g_at(bin_a),
            walker_a.estimate.log_g_at(bin_b),
            walker_b.estimate.log_g_at(bin_b),
            walker_b.estimate.log_g_at(bin_a),
        ) else {
            continue;
        };
        // P = min(1, g_a(E_a) g_b(E_b) / (g_a(E_b) g_b(E_a))).
        let log_p = (a_at_a - a_at_b) + (b_at_b - b_at_a);
        if log_p >= 0.0 || rng.gen::<f64>() < log_p.exp() {
            walker_a.state.swap_configuration(&mut walker_b.state);
            accepted[pair] += 1;
        }
    }
}

/// Average the estimates of the walkers of every window.
fn average_windows(walkers: &mut [RewlWalker], n_windows: usize) {
    let per_window = walkers.len() / n_windows;
    for group in walkers.chunks_mut(per_window) {
        let mut estimates: Vec<&mut WlEstimate> =
            group.iter_mut().map(|w| &mut w.estimate).collect();
        WlEstimate::average_with(&mut estimates);
    }
}

/// Build the walkers of the replica-exchange stage: every window gets
/// `walkers_per_window` copies of the seeded configuration with their own
/// generators, each driven into its window. Returns the walkers and the
/// summed drive-in sweeps.
fn build_rewl_walkers<L: Lattice>(
    template: &WalkerState,
    windows: &[BinWindow],
    contexts: &[WalkerContext<'_>],
    lattice: &L,
    p: &WangLandauParams<'_>,
) -> Result<(Vec<RewlWalker>, u64), MCIsingError> {
    let per_window = p.walkers_per_window;
    let mut walkers: Vec<RewlWalker> = (0..windows.len() * per_window)
        .map(|index| RewlWalker {
            state: template.spawn(create_rng(
                p.base_seed
                    .wrapping_add(WL_WALKER_SEED_OFFSET)
                    .wrapping_add(index as u64),
            )),
            // Placeholder until the walker is inside its window.
            estimate: WlEstimate::new(windows[0], windows[0].lo),
            window: index / per_window,
        })
        .collect();
    let drive_in_sweeps: Vec<u64> = walkers
        .par_iter_mut()
        .map(|walker| drive_in(&mut walker.state, &contexts[walker.window], lattice, p))
        .collect::<Result<_, _>>()?;
    for walker in &mut walkers {
        let window = windows[walker.window];
        walker.estimate = match p.initial_log_g {
            Some(initial) => WlEstimate::from_initial(window, initial, walker.state.bin),
            None => WlEstimate::new(window, walker.state.bin),
        };
    }
    Ok((walkers, drive_in_sweeps.iter().sum()))
}

fn total_visited(walkers: &[RewlWalker]) -> usize {
    walkers.iter().map(|w| w.estimate.n_visited()).sum()
}

/// Flatness of every window's pooled histogram (the walkers of a window
/// share one estimate after averaging, so their visits count together),
/// minimised over windows.
fn pooled_flatness(walkers: &[RewlWalker], n_windows: usize) -> f64 {
    let per_window = walkers.len() / n_windows;
    walkers
        .chunks(per_window)
        .map(|group| {
            let first = &group[0].estimate;
            let mut pooled = vec![0u64; first.histogram().len()];
            for walker in group {
                for (slot, &count) in pooled.iter_mut().zip(walker.estimate.histogram()) {
                    *slot += count;
                }
            }
            flatness_over_visited(&pooled, first.log_g())
        })
        .fold(f64::INFINITY, f64::min)
}

/// Drive the replica-exchange walkers through the modification-factor
/// schedule: parallel sweeps in `exchange_interval` chunks, an exchange
/// round after each chunk, one shared `ln f` (a check passes when every
/// window's pooled histogram is flat; the `1/t` clock is the slowest
/// window's),
/// and estimates averaged within each window at every check.
fn run_rewl_schedule<L: Lattice>(
    walkers: &mut [RewlWalker],
    windows: &[BinWindow],
    contexts: &[WalkerContext<'_>],
    lattice: &L,
    p: &WangLandauParams<'_>,
    diagnostics: &mut WlDiagnostics,
) {
    let schedule = &p.schedule;
    let per_window = p.walkers_per_window as f64;
    let proposals_per_sweep = contexts[0].num_sites as f64;
    let clock = windows
        .iter()
        .map(|w| w.len() as f64 / (per_window * proposals_per_sweep))
        .fold(0.0_f64, f64::max);
    let mut exchange_rng = create_rng(p.base_seed.wrapping_add(EXCHANGE_SEED_OFFSET));
    let mut state = ScheduleState::new(schedule, total_visited(walkers));
    let mut sweeps: u64 = 0;
    let mut round: u64 = 0;
    loop {
        if schedule.max_sweeps.is_some_and(|max| sweeps >= max) {
            break;
        }
        let chunk = match schedule.max_sweeps {
            Some(max) => p.exchange_interval.min(max - sweeps),
            None => p.exchange_interval,
        };
        let log_f = state.log_f;
        walkers.par_iter_mut().for_each(|walker| {
            let context = &contexts[walker.window];
            for _ in 0..chunk {
                wl_sweep(
                    &mut walker.state,
                    &mut walker.estimate,
                    context,
                    lattice,
                    log_f,
                );
            }
        });
        sweeps += chunk;
        attempt_exchanges(
            walkers,
            windows,
            round,
            &mut exchange_rng,
            &mut diagnostics.exchange_attempted,
            &mut diagnostics.exchange_accepted,
        );
        round += 1;
        let t_inv = clock / sweeps as f64;
        let at_check = sweeps.is_multiple_of(schedule.check_interval);
        if state.one_over_t {
            if at_check {
                average_windows(walkers, windows.len());
            }
            if state.tick(t_inv, schedule) {
                diagnostics.converged = true;
                break;
            }
            continue;
        }
        if !at_check {
            continue;
        }
        let verdict = state.check(
            sweeps,
            t_inv,
            pooled_flatness(walkers, windows.len()),
            total_visited(walkers),
            schedule,
            diagnostics,
        );
        if verdict != Verdict::Continue {
            average_windows(walkers, windows.len());
        }
        match verdict {
            Verdict::Flat => {
                for walker in walkers.iter_mut() {
                    walker.estimate.reset_histogram();
                }
            }
            Verdict::Converged => {
                diagnostics.converged = true;
                break;
            }
            Verdict::Switched | Verdict::Continue => {}
        }
    }
    average_windows(walkers, windows.len());
    state.finish(
        sweeps,
        pooled_flatness(walkers, windows.len()),
        total_visited(walkers),
        diagnostics,
    );
}

/// The replica-exchange Wang-Landau stage (Vogel, Li, Wüst & Landau, PRL
/// 110, 210603 (2013)): overlapping windows, `walkers_per_window` walkers
/// each, configuration exchanges between adjacent windows, and one merged
/// estimate at the end.
fn run_parallel_stage<L: Lattice>(
    spins: Vec<i8>,
    rng: rand_xoshiro::Xoshiro256StarStar,
    lattice: &L,
    binning: &EnergyBinning,
    ctx: &WalkerContext<'_>,
    window: BinWindow,
    active: [bool; 3],
    p: &WangLandauParams<'_>,
) -> Result<StageOutput, MCIsingError> {
    let windows = split_windows(window, p.n_windows, p.window_overlap)?;
    let contexts: Vec<WalkerContext<'_>> = windows
        .iter()
        .map(|&w| WalkerContext {
            binning: ctx.binning,
            table: ctx.table,
            window: w,
            num_sites: ctx.num_sites,
            use_nn: ctx.use_nn,
            use_nnn: ctx.use_nnn,
            use_tnn: ctx.use_tnn,
        })
        .collect();
    let template = WalkerState::new(spins, rng, lattice, binning, active);
    let (mut walkers, drive_in_sweeps) =
        build_rewl_walkers(&template, &windows, &contexts, lattice, p)?;
    let mut diagnostics = WlDiagnostics {
        n_windows: windows.len(),
        walkers_per_window: p.walkers_per_window,
        window_bins: windows.iter().map(|w| (w.lo, w.hi)).collect(),
        exchange_attempted: vec![0; windows.len().saturating_sub(1)],
        exchange_accepted: vec![0; windows.len().saturating_sub(1)],
        drive_in_sweeps,
        ..WlDiagnostics::default()
    };
    run_rewl_schedule(
        &mut walkers,
        &windows,
        &contexts,
        lattice,
        p,
        &mut diagnostics,
    );
    diagnostics.accepted = walkers.iter().map(|w| w.state.accepted).sum();
    diagnostics.attempted = walkers.iter().map(|w| w.state.attempted).sum();

    // The estimates within a window are averaged, hence identical.
    let per_window = p.walkers_per_window;
    let pieces: Vec<(BinWindow, Vec<f64>)> = windows
        .iter()
        .enumerate()
        .map(|(index, &w)| (w, walkers[index * per_window].estimate.log_g().to_vec()))
        .collect();
    let (log_g, merge_bins) = merge_windows(&pieces, binning.n_bins);
    diagnostics.merge_bins = merge_bins;
    diagnostics.visited_bins = log_g.iter().filter(|v| v.is_finite()).count();
    let mut histogram = vec![0u64; binning.n_bins];
    for walker in &walkers {
        let w = walker.estimate.window();
        for (offset, &count) in walker.estimate.histogram().iter().enumerate() {
            histogram[w.lo + offset] += count;
        }
    }
    Ok(StageOutput {
        seeds: walkers.into_iter().map(|w| w.state).collect(),
        log_g,
        histogram,
        diagnostics,
    })
}

/// Move the initial configuration into the window with Metropolis sweeps at
/// `±drive_beta` (the sign re-evaluated every sweep, so an overshoot turns
/// around). Returns the number of sweeps used.
fn drive_in<L: Lattice>(
    state: &mut WalkerState,
    ctx: &WalkerContext<'_>,
    lattice: &L,
    p: &WangLandauParams<'_>,
) -> Result<u64, MCIsingError> {
    let mut sweeps = 0u64;
    loop {
        if ctx.window.contains(state.bin as i64) {
            return Ok(sweeps);
        }
        if sweeps >= p.drive_max_sweeps {
            let n = ctx.num_sites as f64;
            let (e_lo, e_hi) = (
                ctx.binning.energy_of_bin(ctx.window.lo) / n,
                ctx.binning.energy_of_bin(ctx.window.hi) / n,
            );
            return Err(MCIsingError::EnergyWindowUnreachable {
                e_lo,
                e_hi,
                reached: ctx.binning.energy_of_bin(state.bin) / n,
                sweeps,
            });
        }
        let beta = if state.bin > ctx.window.hi {
            p.drive_beta
        } else {
            -p.drive_beta
        };
        sweeps += 1;
        if drive_sweep(state, ctx, lattice, beta) {
            return Ok(sweeps);
        }
    }
}

/// Stage 1 of the serial walker: flatness-driven halving of `ln f`, then
/// the 1/t schedule (see [`ScheduleState`]).
fn run_wang_landau_stage<L: Lattice>(
    state: &mut WalkerState,
    estimate: &mut WlEstimate,
    ctx: &WalkerContext<'_>,
    lattice: &L,
    schedule: &WangLandauSchedule,
) -> WlDiagnostics {
    let bins = ctx.window.len() as f64;
    let proposals_per_sweep = ctx.num_sites as f64;
    let mut diagnostics = WlDiagnostics::default();
    let mut clock = ScheduleState::new(schedule, estimate.n_visited());
    let mut sweeps: u64 = 0;
    loop {
        if schedule.max_sweeps.is_some_and(|max| sweeps >= max) {
            break;
        }
        wl_sweep(state, estimate, ctx, lattice, clock.log_f);
        sweeps += 1;
        // Mean visits per bin of the window: the Belardinelli–Pereyra
        // clock (sweeps for a full-range run whose bin count is ~N).
        let t_inv = bins / (sweeps as f64 * proposals_per_sweep);
        if clock.one_over_t {
            if clock.tick(t_inv, schedule) {
                diagnostics.converged = true;
                break;
            }
            continue;
        }
        if !sweeps.is_multiple_of(schedule.check_interval) {
            continue;
        }
        let verdict = clock.check(
            sweeps,
            t_inv,
            estimate.flatness(),
            estimate.n_visited(),
            schedule,
            &mut diagnostics,
        );
        match verdict {
            Verdict::Flat => estimate.reset_histogram(),
            Verdict::Converged => {
                diagnostics.converged = true;
                break;
            }
            Verdict::Switched | Verdict::Continue => {}
        }
    }
    clock.finish(
        sweeps,
        estimate.flatness(),
        estimate.n_visited(),
        &mut diagnostics,
    );
    diagnostics
}

/// One production walker's output before pooling.
struct WalkerOutput {
    series: MucaSeries,
    histogram: Vec<u64>,
    rejected_unvisited: u64,
}

/// Run one production walker: thermalize (unrecorded), then measure.
fn run_one_walker<L: Lattice>(
    k: usize,
    seed_states: &[WalkerState],
    frozen: &[f64],
    edges: (usize, usize),
    ctx: &WalkerContext<'_>,
    lattice: &L,
    p: &WangLandauParams<'_>,
    shape: &[usize],
) -> WalkerOutput {
    let n_measurements = (p.production_sweeps / p.measurement_interval) as usize;
    let window_len = ctx.window.len();
    let rng = create_rng(
        p.base_seed
            .wrapping_add(PRODUCTION_SEED_OFFSET)
            .wrapping_add(k as u64),
    );
    let mut walker = seed_states[k % seed_states.len()].spawn(rng);
    let mut histogram = vec![0u64; window_len];
    let mut edge_tracker = EdgeTracker::new(edges.0, edges.1);
    {
        // Thermalization in the multicanonical ensemble: nothing is
        // recorded and the counters restart afterwards.
        let mut scratch_histogram = vec![0u64; window_len];
        let mut scratch_edges = EdgeTracker::new(edges.0, edges.1);
        for _ in 0..p.production_thermalization {
            muca_sweep(
                &mut walker,
                ctx,
                lattice,
                frozen,
                &mut scratch_histogram,
                &mut scratch_edges,
            );
        }
    }
    walker.accepted = 0;
    walker.attempted = 0;
    walker.rejected_unvisited = 0;
    let mut series = TempResult::with_capacity(
        f64::NAN,
        n_measurements,
        ctx.num_sites,
        shape.to_vec(),
        p.store_configs,
        false,
        1,
    );
    let mut bins = Vec::with_capacity(n_measurements);
    for _ in 0..n_measurements {
        for _ in 0..p.measurement_interval {
            muca_sweep(
                &mut walker,
                ctx,
                lattice,
                frozen,
                &mut histogram,
                &mut edge_tracker,
            );
        }
        let energy = observables::energy_per_site(&walker.spins, lattice, p.j1, p.j2, p.j3, p.h);
        debug_assert_eq!(
            walker.shells,
            observables::shell_sums(&walker.spins, lattice, ctx.use_nn, ctx.use_nnn, ctx.use_tnn)
        );
        debug_assert_eq!(walker.bin, ctx.binning.bin_of_shells(&walker.shells));
        series.push(&walker.spins, lattice, energy);
        bins.push(walker.bin as u32);
    }
    WalkerOutput {
        series: MucaSeries {
            walker: k,
            series,
            bins,
            accepted: walker.accepted,
            attempted: walker.attempted,
            round_trips: edge_tracker.round_trips,
        },
        histogram,
        rejected_unvisited: walker.rejected_unvisited,
    }
}

/// Stage 2: `n_walkers` frozen-weight chains in parallel, each starting
/// from the Wang-Landau walker's final configuration with its own
/// generator.
fn run_production<L: Lattice>(
    seed_states: &[WalkerState],
    frozen: &[f64],
    ctx: &WalkerContext<'_>,
    lattice: &L,
    p: &WangLandauParams<'_>,
    shape: &[usize],
) -> (Vec<MucaSeries>, Vec<u64>, ProductionDiagnostics) {
    let visited_edges = {
        let lo = frozen.iter().position(|v| v.is_finite());
        let hi = frozen.iter().rposition(|v| v.is_finite());
        match (lo, hi) {
            (Some(lo), Some(hi)) => Some((ctx.window.lo + lo, ctx.window.lo + hi)),
            _ => None,
        }
    };
    let edges = visited_edges.unwrap_or((seed_states[0].bin, seed_states[0].bin));
    let n_measurements = (p.production_sweeps / p.measurement_interval) as usize;
    let outputs: Vec<WalkerOutput> = (0..p.n_walkers)
        .into_par_iter()
        .map(|k| run_one_walker(k, seed_states, frozen, edges, ctx, lattice, p, shape))
        .collect();

    let mut pooled = vec![0u64; ctx.binning.n_bins];
    let mut diagnostics = ProductionDiagnostics {
        n_walkers: p.n_walkers,
        sweeps_per_walker: n_measurements as u64 * p.measurement_interval,
        thermalization_sweeps: p.production_thermalization,
        accepted: Vec::with_capacity(p.n_walkers),
        attempted: Vec::with_capacity(p.n_walkers),
        rejected_unvisited: Vec::with_capacity(p.n_walkers),
        round_trips: Vec::with_capacity(p.n_walkers),
        histogram_flatness: 0.0,
        edge_bins: visited_edges,
    };
    let mut production = Vec::with_capacity(p.n_walkers);
    for output in outputs {
        for (i, &count) in output.histogram.iter().enumerate() {
            pooled[ctx.window.lo + i] += count;
        }
        diagnostics.accepted.push(output.series.accepted);
        diagnostics.attempted.push(output.series.attempted);
        diagnostics
            .rejected_unvisited
            .push(output.rejected_unvisited);
        diagnostics.round_trips.push(output.series.round_trips);
        production.push(output.series);
    }
    diagnostics.histogram_flatness =
        flatness_over_visited(&pooled[ctx.window.lo..=ctx.window.hi], frozen);
    (production, pooled, diagnostics)
}

/// Wang-Landau density of states followed by a frozen-weight multicanonical
/// production run (see the module documentation).
///
/// `energy_window` is per site, `bin_width` in total-energy units;
/// `initial_log_g` (one entry per energy bin, NaN = unknown) seeds the
/// estimate, and `max_wl_sweeps = 0` freezes it as given. With
/// `n_windows > 1` or `walkers_per_window > 1` the first stage runs the
/// replica-exchange Wang-Landau algorithm (overlapping windows sharing
/// `window_overlap` of their length, exchanges every `exchange_interval`
/// sweeps, estimates averaged within a window and joined at the end);
/// otherwise the serial walker runs. Returns the dict described at
/// `WangLandauResult::into_pydict`.
///
/// # Errors
///
/// Returns a `ValueError` for an unknown lattice, a non-finite coupling, a
/// zero interval or walker count, a flatness outside `(0, 1]`, a
/// modification-factor schedule that is not `0 < log_f_final <
/// log_f_initial`, a non-positive `drive_beta`, couplings that are not
/// dyadic-exact without a `bin_width`, a non-positive `bin_width`, a grid
/// of more than 2^24 bins, an invalid or unreachable `energy_window`, or an
/// `initial_log_g` of the wrong length or with infinite entries.
#[pyfunction]
#[pyo3(signature = (
    lattice_size, j1, j2, j3, h, base_seed, lattice_type,
    production_sweeps, measurement_interval, *,
    n_walkers = 1, production_thermalization = 0, store_configs = false,
    bin_width = None, energy_window = None, initial_log_g = None,
    flatness = 0.8, log_f_initial = 1.0, log_f_final = 1e-6,
    check_interval = 1000, max_wl_sweeps = None,
    drive_beta = 1.0, drive_max_sweeps = 10_000,
    n_windows = 1, walkers_per_window = 1, window_overlap = 0.75,
    exchange_interval = 100,
))]
pub fn run_wang_landau<'py>(
    py: Python<'py>,
    lattice_size: usize,
    j1: f64,
    j2: f64,
    j3: f64,
    h: f64,
    base_seed: u64,
    lattice_type: &str,
    production_sweeps: u64,
    measurement_interval: u64,
    n_walkers: usize,
    production_thermalization: u64,
    store_configs: bool,
    bin_width: Option<f64>,
    energy_window: Option<(f64, f64)>,
    initial_log_g: Option<PyReadonlyArray1<'py, f64>>,
    flatness: f64,
    log_f_initial: f64,
    log_f_final: f64,
    check_interval: u64,
    max_wl_sweeps: Option<u64>,
    drive_beta: f64,
    drive_max_sweeps: u64,
    n_windows: usize,
    walkers_per_window: usize,
    window_overlap: f64,
    exchange_interval: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let lat_type = lattice_type.to_string();
    let initial: Option<Vec<f64>> = match initial_log_g {
        Some(array) => Some(
            array
                .as_slice()
                .map_err(|e| MCIsingError::InvalidInitialLogG(format!("cannot read array: {e}")))?
                .to_vec(),
        ),
        None => None,
    };
    let result = py.allow_threads(|| {
        let params = WangLandauParams {
            lattice_size,
            j1,
            j2,
            j3,
            h,
            base_seed,
            lattice_type: &lat_type,
            bin_width,
            energy_window,
            initial_log_g: initial.as_deref(),
            schedule: WangLandauSchedule {
                log_f_initial,
                log_f_final,
                flatness,
                check_interval,
                max_sweeps: max_wl_sweeps,
            },
            drive_beta,
            drive_max_sweeps,
            n_windows,
            walkers_per_window,
            window_overlap,
            exchange_interval,
            n_walkers,
            production_sweeps,
            production_thermalization,
            measurement_interval,
            store_configs,
        };
        run_wang_landau_internal(&params)
    })?;
    result.into_pydict(py)
}

#[cfg(test)]
mod tests {
    // Bit-identity of grid energies and weights is the contract these
    // tests pin, so exact float comparison is the point.
    #![allow(clippy::float_cmp)]

    use super::*;
    use crate::exact_enumeration::{chain12_dos, square4_dos, GATE_TEMPERATURES};

    fn schedule(log_f_final: f64) -> WangLandauSchedule {
        WangLandauSchedule {
            log_f_initial: 1.0,
            log_f_final,
            flatness: 0.8,
            check_interval: 100,
            max_sweeps: None,
        }
    }

    fn params(
        lattice_type: &'static str,
        size: usize,
        couplings: [f64; 4],
        log_f_final: f64,
        production_sweeps: u64,
    ) -> WangLandauParams<'static> {
        WangLandauParams {
            lattice_size: size,
            j1: couplings[0],
            j2: couplings[1],
            j3: couplings[2],
            h: couplings[3],
            base_seed: 42,
            lattice_type,
            bin_width: None,
            energy_window: None,
            initial_log_g: None,
            schedule: schedule(log_f_final),
            drive_beta: 1.0,
            drive_max_sweeps: 10_000,
            n_windows: 1,
            walkers_per_window: 1,
            window_overlap: 0.75,
            exchange_interval: 100,
            n_walkers: 1,
            production_sweeps,
            production_thermalization: 0,
            measurement_interval: 1,
            store_configs: false,
        }
    }

    fn run(p: &WangLandauParams<'_>) -> WangLandauResult {
        run_wang_landau_internal(p).expect("valid run")
    }

    /// `Result::unwrap_err` needs `Debug` on the Ok type, which the
    /// production result deliberately does not derive.
    fn error_of(p: &WangLandauParams<'_>) -> MCIsingError {
        match run_wang_landau_internal(p) {
            Err(err) => err,
            Ok(_) => panic!("expected an error"),
        }
    }

    fn bin_of_energy(binning: &EnergyBinning, energy: f64) -> usize {
        ((energy - binning.e_min) / binning.width).round() as usize
    }

    /// Maximum |Δ ln g| against the exact levels after aligning at the
    /// level with the largest count; asserts the visited support equals
    /// the exact support first.
    fn aligned_deviation(result: &WangLandauResult, exact: &[(f64, f64)]) -> f64 {
        let visited: Vec<usize> = result
            .log_g
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, _)| i)
            .collect();
        let expected: Vec<usize> = exact
            .iter()
            .map(|&(energy, _)| bin_of_energy(&result.binning, energy))
            .collect();
        assert_eq!(
            visited, expected,
            "visited bins differ from the exact support"
        );
        let (ref_bin, ref_log_g) = exact
            .iter()
            .map(|&(energy, log_g)| (bin_of_energy(&result.binning, energy), log_g))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("non-empty support");
        let shift = result.log_g[ref_bin] - ref_log_g;
        exact
            .iter()
            .map(|&(energy, log_g)| {
                (result.log_g[bin_of_energy(&result.binning, energy)] - shift - log_g).abs()
            })
            .fold(0.0, f64::max)
    }

    /// Canonical (<e>, Cv/N) at `beta` from a global `ln g` by exact
    /// reweighting of the bins.
    fn canonical_from_log_g(binning: &EnergyBinning, log_g: &[f64], beta: f64) -> (f64, f64) {
        let n = binning.num_sites as f64;
        let terms: Vec<(f64, f64)> = log_g
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(k, &lg)| {
                let energy = binning.energy_of_bin(k);
                (energy, lg - beta * energy)
            })
            .collect();
        let max = terms.iter().map(|t| t.1).fold(f64::NEG_INFINITY, f64::max);
        let (mut z, mut e1, mut e2) = (0.0, 0.0, 0.0);
        for (energy, log_w) in terms {
            let w = (log_w - max).exp();
            z += w;
            e1 += w * energy;
            e2 += w * energy * energy;
        }
        let mean = e1 / z;
        let var = e2 / z - mean * mean;
        (mean / n, beta * beta * var / n)
    }

    /// Canonical <e> at `beta` by reweighting the production series with
    /// the frozen weights.
    fn reweight_series(result: &WangLandauResult, beta: f64) -> f64 {
        let n = result.binning.num_sites as f64;
        let mut log_w = Vec::new();
        let mut energies = Vec::new();
        for walker in &result.production {
            for (i, &e) in walker.series.energies.iter().enumerate() {
                let bin = walker.bins[i] as usize;
                log_w.push(result.log_g[bin] - beta * e * n);
                energies.push(e);
            }
        }
        let max = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (mut z, mut sum) = (0.0, 0.0);
        for (lw, e) in log_w.iter().zip(&energies) {
            let w = (lw - max).exp();
            z += w;
            sum += w * e;
        }
        sum / z
    }

    /// The fast gate: `tolerance` on ln g is 0.1 at `log_f_final = 1e-5`,
    /// where the halving stage still leaves a few 1e-2 (a wrong acceptance
    /// rule is O(1) off); the canonical averages are held to 1 % (<e>)
    /// and 5 % (Cv). The ignored release-mode calibration test tracks the
    /// finer 1/t scaling.
    fn assert_matches_exact(
        label: &str,
        result: &WangLandauResult,
        exact: &[(f64, f64)],
        canonical: impl Fn(f64) -> (f64, f64),
        tolerance: f64,
    ) {
        let deviation = aligned_deviation(result, exact);
        println!(
            "calib {label}: max|dln g|={deviation:.4} sweeps={} iterations={} switch={:?}",
            result.wl.total_sweeps,
            result.wl.iteration_log_f.len(),
            result.wl.one_over_t_switch_sweep
        );
        assert!(result.wl.converged, "{label}: did not converge");
        assert!(
            deviation <= tolerance,
            "{label}: max |dln g| = {deviation} > {tolerance}"
        );
        for temperature in GATE_TEMPERATURES {
            let (e_exact, cv_exact) = canonical(temperature);
            let (e_wl, cv_wl) =
                canonical_from_log_g(&result.binning, &result.log_g, 1.0 / temperature);
            println!(
                "calib {label} T={temperature}: e={e_wl:.6} exact={e_exact:.6} cv={cv_wl:.6} exact={cv_exact:.6}"
            );
            assert!(
                ((e_wl - e_exact) / e_exact).abs() <= 0.01,
                "{label} T={temperature}: <e> {e_wl} vs exact {e_exact}"
            );
            assert!(
                ((cv_wl - cv_exact) / cv_exact).abs() <= 0.05,
                "{label} T={temperature}: Cv {cv_wl} vs exact {cv_exact}"
            );
        }
    }

    #[test]
    fn test_wang_landau_chain12_matches_exact_log_g() {
        // The chain is the canary for site selection: the sequential
        // Metropolis scan never equilibrates it (#26), the random-site
        // walk must.
        let p = params("chain", 12, [1.0, 0.0, 0.0, 0.0], 1e-5, 0);
        let result = run(&p);
        let dos = chain12_dos();
        assert_matches_exact(
            "chain12 J1=+1",
            &result,
            &dos.log_g_by_energy(1.0, 0.0, 0.0),
            |t| {
                let ex = dos.exact_at(t, 1.0);
                (ex.energy, ex.specific_heat)
            },
            0.1,
        );
        assert!(result.production.is_empty() || result.production[0].series.energies.is_empty());
    }

    #[test]
    fn test_wang_landau_square4_matches_exact_log_g_both_signs() {
        let dos = square4_dos();
        for j1 in [1.0, -1.0] {
            let p = params("square", 4, [j1, 0.0, 0.0, 0.0], 1e-5, 0);
            let result = run(&p);
            let exact = dos.log_g_by_energy(j1, 0.0, 0.0);
            assert_eq!(exact.len(), 15, "the 4x4 torus has 15 energy levels");
            assert_matches_exact(
                &format!("square4 J1={j1}"),
                &result,
                &exact,
                |t| {
                    let ex = dos.exact_at(t, j1);
                    (ex.energy, ex.specific_heat)
                },
                0.1,
            );
            // Normalised to 2^16 states, the ground state is doubly degenerate.
            let max = result
                .log_g
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = result
                .log_g
                .iter()
                .filter(|v| v.is_finite())
                .map(|&v| (v - max).exp())
                .sum();
            let shift = max + total.ln() - 16.0 * std::f64::consts::LN_2;
            let ground = bin_of_energy(&result.binning, exact[0].0);
            let ground_log_g = result.log_g[ground] - shift;
            assert!(
                (ground_log_g - std::f64::consts::LN_2).abs() <= 0.1,
                "J1={j1}: normalised ln g(E_min) = {ground_log_g}"
            );
        }
    }

    #[test]
    fn test_wang_landau_square4_j2_and_field_match_exact_log_g() {
        let dos = square4_dos();
        for (j2, h) in [(-0.5, 0.0), (0.0, 0.5)] {
            let p = params("square", 4, [1.0, j2, 0.0, h], 1e-5, 0);
            let result = run(&p);
            assert_matches_exact(
                &format!("square4 J2={j2} h={h}"),
                &result,
                &dos.log_g_by_energy(1.0, j2, h),
                |t| {
                    let ex = dos.exact_at_couplings(t, 1.0, j2, h);
                    (ex.energy, ex.specific_heat)
                },
                0.1,
            );
        }
    }

    #[test]
    #[ignore = "calibration: run with --release -- --ignored"]
    fn test_wang_landau_square4_error_scaling_calibration() {
        let dos = square4_dos();
        for log_f_final in [1e-4, 1e-5, 1e-6, 1e-7] {
            let p = params("square", 4, [1.0, 0.0, 0.0, 0.0], log_f_final, 0);
            let result = run(&p);
            let deviation = aligned_deviation(&result, &dos.log_g_by_energy(1.0, 0.0, 0.0));
            println!(
                "calib square4 log_f_final={log_f_final:.0e}: max|dln g|={deviation:.5} sweeps={}",
                result.wl.total_sweeps
            );
            // The halving stage alone (1e-4) leaves ~0.2; the 1/t stage
            // brings it down as t^-1/2.
            if log_f_final <= 1e-5 {
                assert!(
                    deviation <= 0.1,
                    "log_f_final={log_f_final:.0e}: {deviation}"
                );
            }
        }
    }

    #[test]
    fn test_one_over_t_schedule_diagnostics() {
        let p = params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-5, 0);
        let result = run(&p);
        let wl = &result.wl;
        let switch = wl
            .one_over_t_switch_sweep
            .expect("the 1/t stage was entered");
        assert!(switch <= wl.total_sweeps);
        assert!(wl.converged && wl.final_log_f < 1e-5);
        assert!(wl.total_sweeps >= 10_000, "{}", wl.total_sweeps);
        // Strict halving before the switch; the last record is the 1/t stage.
        let n = wl.iteration_log_f.len();
        assert!(n >= 3);
        for pair in wl.iteration_log_f[..n - 1].windows(2) {
            assert!((pair[1] - pair[0] * 0.5).abs() <= 1e-12, "{pair:?}");
        }
        assert!(wl.iteration_log_f[n - 1] <= wl.iteration_log_f[n - 2]);
        assert_eq!(wl.iteration_sweeps.iter().sum::<u64>(), wl.total_sweeps);
        assert_eq!(wl.iteration_sweeps.len(), n);
        assert_eq!(wl.iteration_flatness.len(), n);
        assert_eq!(wl.iteration_visited_bins.len(), n);
        assert_eq!(*wl.iteration_visited_bins.last().unwrap(), 15);
        assert_eq!(wl.visited_bins, 15);
        assert_eq!(wl.attempted, wl.total_sweeps * 16);
        assert!(wl.accepted > 0 && wl.accepted < wl.attempted);
        assert_eq!(wl.drive_in_sweeps, 0);
        let hist_total: u64 = result.wl_histogram.iter().sum();
        assert!(hist_total.is_multiple_of(16));
        assert!(hist_total <= 16 * wl.total_sweeps);
        assert!(hist_total >= 16 * wl.iteration_sweeps[n - 1]);
    }

    #[test]
    fn test_max_wl_sweeps_returns_unconverged() {
        let mut p = params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-5, 100);
        p.schedule.max_sweeps = Some(50);
        let result = run(&p);
        assert!(!result.wl.converged);
        assert_eq!(result.wl.total_sweeps, 50);
        assert_eq!(result.production.len(), 1);
        assert_eq!(result.production[0].series.energies.len(), 100);
        p.schedule.max_sweeps = Some(0);
        let result = run(&p);
        assert_eq!(result.wl.total_sweeps, 0);
        assert_eq!(result.wl.iteration_sweeps, vec![0]);
        assert_eq!(result.wl.visited_bins, 1);
    }

    #[test]
    fn test_production_histogram_flat_with_round_trips_and_exact_reweighting() {
        let p = params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-5, 50_000);
        let result = run(&p);
        let diag = &result.production_diagnostics;
        println!(
            "calib production: flatness={:.3} round_trips={:?} acceptance={:.3}",
            diag.histogram_flatness,
            diag.round_trips,
            diag.accepted[0] as f64 / diag.attempted[0] as f64
        );
        assert!(
            diag.histogram_flatness >= 0.5,
            "{}",
            diag.histogram_flatness
        );
        assert!(diag.round_trips[0] > 0);
        assert_eq!(diag.sweeps_per_walker, 50_000);
        assert_eq!(diag.attempted[0], 50_000 * 16);
        assert_eq!(diag.rejected_unvisited[0], 0);
        assert_eq!(
            diag.edge_bins,
            result.window.lo.checked_add(0).map(|_| (0, 16))
        );
        let series = &result.production[0];
        assert_eq!(series.series.energies.len(), 50_000);
        assert_eq!(series.bins.len(), 50_000);
        assert!(series
            .bins
            .iter()
            .all(|&b| result.log_g[b as usize].is_finite()));
        assert_eq!(result.production_histogram.iter().sum::<u64>(), 50_000 * 16);
        let dos = square4_dos();
        for temperature in GATE_TEMPERATURES {
            let exact = dos.exact_at(temperature, 1.0).energy;
            let e = reweight_series(&result, 1.0 / temperature);
            println!("calib production T={temperature}: e={e:.5} exact={exact:.5}");
            assert!(
                ((e - exact) / exact).abs() <= 0.02,
                "T={temperature}: reweighted <e> {e} vs exact {exact}"
            );
        }
    }

    #[test]
    fn test_production_never_updates_log_g() {
        let without = run(&params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-4, 0));
        let with = run(&params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-4, 5_000));
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&without.log_g), bits(&with.log_g));
        assert_eq!(without.wl_histogram, with.wl_histogram);
    }

    #[test]
    fn test_wrong_weights_still_reproduce_the_exact_canonical_averages() {
        // The frozen-weight chain samples pi_W(x) ∝ W(E(x)) for any positive
        // W, and reweighting removes W exactly; only the variance depends
        // on the weights. A linear tilt and a crude quadratic both have to
        // give the exact <e>. (Exactly flat weights are excluded on
        // purpose: with every move accepted the walk on the hypercube has
        // period 2, and sampling every N flips would see one flip-parity
        // class only — the weights merely have to be non-constant for the
        // rejections to break that periodicity.)
        let dos = square4_dos();
        let tilted: Vec<f64> = (0..17_usize).map(|k| 0.05 * k as f64).collect();
        let quadratic: Vec<f64> = (0..17_usize)
            .map(|k| {
                let energy = -32.0 + 4.0 * k as f64;
                -(energy / 12.0).powi(2)
            })
            .collect();
        for (label, initial) in [("tilted", &tilted), ("quadratic", &quadratic)] {
            let mut p = params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-5, 100_000);
            p.initial_log_g = Some(initial);
            p.schedule.max_sweeps = Some(0);
            let result = run(&p);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&result.log_g),
                bits(initial),
                "{label}: weights were modified"
            );
            for temperature in [4.0, 2.269] {
                let exact = dos.exact_at(temperature, 1.0).energy;
                let e = reweight_series(&result, 1.0 / temperature);
                println!("calib wrong-weights {label} T={temperature}: e={e:.5} exact={exact:.5}");
                assert!(
                    ((e - exact) / exact).abs() <= 0.03,
                    "{label} T={temperature}: {e} vs {exact}"
                );
            }
        }
    }

    #[test]
    fn test_production_rejects_wang_landau_unvisited_bins() {
        let mut p = params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-5, 2_000);
        p.schedule.max_sweeps = Some(2);
        let result = run(&p);
        assert!(result.wl.visited_bins < 15);
        let series = &result.production[0];
        assert!(series
            .bins
            .iter()
            .all(|&b| result.log_g[b as usize].is_finite()));
        assert!(result.production_diagnostics.rejected_unvisited[0] > 0);
    }

    #[test]
    fn test_window_restricted_wang_landau_matches_exact_inside() {
        let dos = square4_dos();
        let mut p = params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-5, 2_000);
        p.energy_window = Some((-1.5, 0.5));
        let result = run(&p);
        assert_eq!(result.window, BinWindow { lo: 2, hi: 10 });
        let exact: Vec<(f64, f64)> = dos
            .log_g_by_energy(1.0, 0.0, 0.0)
            .into_iter()
            .filter(|&(energy, _)| (-24.0..=8.0).contains(&energy))
            .collect();
        assert_eq!(exact.len(), 9);
        let deviation = aligned_deviation(&result, &exact);
        println!("calib window: max|dln g|={deviation:.4}");
        assert!(result.wl.converged && deviation <= 0.05);
        for (k, &count) in result.production_histogram.iter().enumerate() {
            assert!(count == 0 || (2..=10).contains(&k));
        }
        assert!(result.production[0]
            .series
            .energies
            .iter()
            .all(|&e| (-1.5..=0.5).contains(&e)));
    }

    #[test]
    fn test_window_drive_in_lowers_and_raises_the_energy() {
        let dos = square4_dos();
        for (window, range) in [((-2.0, -1.2), (-32.0, -19.2)), ((1.2, 2.0), (19.2, 32.0))] {
            let mut p = params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-4, 0);
            p.energy_window = Some(window);
            let result = run(&p);
            assert!(result.wl.drive_in_sweeps > 0, "{window:?}");
            let exact: Vec<(f64, f64)> = dos
                .log_g_by_energy(1.0, 0.0, 0.0)
                .into_iter()
                .filter(|&(energy, _)| energy >= range.0 && energy <= range.1)
                .collect();
            assert_eq!(exact.len(), 3);
            let deviation = aligned_deviation(&result, &exact);
            println!(
                "calib drive-in {window:?}: sweeps={} max|dln g|={deviation:.4}",
                result.wl.drive_in_sweeps
            );
            assert!(deviation <= 0.1);
        }
    }

    #[test]
    fn test_window_without_a_reachable_state_is_an_error() {
        // Per-site (1.74, 1.76) selects only the bin at E = 28, which no
        // state of the 4x4 torus occupies.
        let mut p = params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-4, 0);
        p.energy_window = Some((1.74, 1.76));
        p.drive_max_sweeps = 200;
        let err = error_of(&p);
        assert!(
            matches!(
                err,
                MCIsingError::EnergyWindowUnreachable { sweeps: 200, .. }
            ),
            "{err}"
        );
        // The message names the window's bin energies (both edges map to
        // the E = 28 bin, 1.75 per site).
        assert!(err.to_string().contains("(1.75, 1.75)"), "{err}");
    }

    #[test]
    fn test_same_seed_is_bit_identical_and_seeds_differ() {
        let p = params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-3, 2_000);
        let a = run(&p);
        let b = run(&p);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.log_g), bits(&b.log_g));
        assert_eq!(a.wl_histogram, b.wl_histogram);
        assert_eq!(a.production_histogram, b.production_histogram);
        assert_eq!(
            bits(&a.production[0].series.energies),
            bits(&b.production[0].series.energies)
        );
        assert_eq!(a.production[0].bins, b.production[0].bins);
        assert_eq!(a.final_spins, b.final_spins);
        assert_eq!(a.final_rng_state, b.final_rng_state);
        let mut other = params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-3, 2_000);
        other.base_seed = 43;
        let c = run(&other);
        assert_ne!(
            bits(&a.production[0].series.energies),
            bits(&c.production[0].series.energies)
        );
    }

    fn with_pool<T: Send>(n_threads: usize, f: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("pool builds")
            .install(f)
    }

    #[test]
    fn test_multi_walker_production_is_deterministic_across_thread_counts() {
        let mut p = params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-3, 2_000);
        p.n_walkers = 3;
        p.measurement_interval = 10;
        p.store_configs = true;
        let serial = with_pool(1, || run(&p));
        let parallel = with_pool(4, || run(&p));
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(serial.production.len(), 3);
        for (a, b) in serial.production.iter().zip(&parallel.production) {
            assert_eq!(a.walker, b.walker);
            assert_eq!(bits(&a.series.energies), bits(&b.series.energies));
            assert_eq!(a.bins, b.bins);
            assert_eq!(a.round_trips, b.round_trips);
            assert_eq!(a.series.energies.len(), 200);
            assert_eq!(a.series.staggered.len(), 200 * 4);
            assert_eq!(a.series.configs.as_ref().map(Vec::len), Some(200 * 16));
        }
        assert_ne!(
            bits(&serial.production[0].series.energies),
            bits(&serial.production[1].series.energies)
        );
        let diag = &serial.production_diagnostics;
        assert_eq!(diag.n_walkers, 3);
        assert_eq!(diag.accepted.len(), 3);
        assert_eq!(diag.round_trips.len(), 3);
        assert_eq!(diag.sweeps_per_walker, 2_000);
        assert_eq!(
            serial.production_histogram.iter().sum::<u64>(),
            3 * 2_000 * 16
        );
        // Walker 0 of a three-walker run is the single-walker run.
        p.n_walkers = 1;
        let single = run(&p);
        assert_eq!(
            bits(&single.production[0].series.energies),
            bits(&serial.production[0].series.energies)
        );
        assert_eq!(single.production[0].bins, serial.production[0].bins);
    }

    #[test]
    fn test_production_thermalization_is_not_recorded() {
        let mut p = params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-3, 1_000);
        p.production_thermalization = 500;
        let result = run(&p);
        let diag = &result.production_diagnostics;
        assert_eq!(diag.thermalization_sweeps, 500);
        assert_eq!(diag.attempted[0], 1_000 * 16);
        assert_eq!(result.production_histogram.iter().sum::<u64>(), 1_000 * 16);
        assert_eq!(result.production[0].series.energies.len(), 1_000);
    }

    #[test]
    fn test_validation_errors() {
        let base = || params("square", 4, [1.0, 0.0, 0.0, 0.0], 1e-3, 10);
        let err = |p: WangLandauParams<'_>| error_of(&p).to_string();
        let mut p = base();
        p.measurement_interval = 0;
        assert!(err(p).contains("measurement_interval"));
        let mut p = base();
        p.n_walkers = 0;
        assert!(err(p).contains("n_walkers"));
        let mut p = base();
        p.schedule.check_interval = 0;
        assert!(err(p).contains("check_interval"));
        let mut p = base();
        p.drive_max_sweeps = 0;
        assert!(err(p).contains("drive_max_sweeps"));
        for flatness in [0.0, 1.5, f64::NAN] {
            let mut p = base();
            p.schedule.flatness = flatness;
            assert!(err(p).contains("flatness"));
        }
        let mut p = base();
        p.schedule.log_f_initial = 0.0;
        assert!(err(p).contains("log_f_initial"));
        let mut p = base();
        p.schedule.log_f_final = 2.0;
        assert!(err(p).contains("log_f_final"));
        let mut p = base();
        p.drive_beta = -1.0;
        assert!(err(p).contains("drive_beta"));
        let mut p = base();
        p.lattice_type = "kagome";
        assert!(err(p).contains("Unknown lattice type"));
        let mut p = base();
        p.j2 = 0.3;
        assert!(err(p).contains("bin_width"));
        let mut p = base();
        p.bin_width = Some(0.0);
        assert!(err(p).contains("bin_width"));
        let mut p = base();
        p.energy_window = Some((1.0, 0.0));
        assert!(err(p).contains("energy_window"));
        let short = vec![0.0; 3];
        let mut p = base();
        p.initial_log_g = Some(&short);
        assert!(err(p).contains("initial_log_g"));
        let infinite = vec![f64::INFINITY; 17];
        let mut p = base();
        p.initial_log_g = Some(&infinite);
        assert!(err(p).contains("finite"));
        let mut p = base();
        p.j1 = 0.0;
        assert!(err(p).contains("energy-neutral"));
        let mut p = params("honeycomb", 3, [1.0, 0.0, 0.0, 0.0], 1e-3, 10);
        p.lattice_size = 3;
        assert!(err(p).contains("even size"));
    }

    #[test]
    fn test_non_dyadic_couplings_run_on_the_float_path() {
        let mut p = params("square", 4, [1.0, 0.3, 0.0, 0.0], 1e-3, 500);
        p.bin_width = Some(0.5);
        let result = run(&p);
        assert!(result.binning.exact.is_none());
        assert!(result.wl.converged);
        assert!(result.production[0]
            .bins
            .iter()
            .all(|&b| result.log_g[b as usize].is_finite()));
    }

    #[test]
    fn test_split_windows_cover_the_range_with_overlap() {
        let window = BinWindow { lo: 3, hi: 102 };
        let windows = split_windows(window, 4, 0.75).unwrap();
        assert_eq!(windows.len(), 4);
        assert_eq!(windows[0].lo, 3);
        assert_eq!(windows[3].hi, 102);
        for pair in windows.windows(2) {
            assert!(pair[1].lo > pair[0].lo && pair[1].lo <= pair[0].hi);
            assert!(pair[1].hi > pair[0].hi);
            // Roughly three quarters shared.
            let shared = (pair[0].hi - pair[1].lo + 1) as f64 / pair[0].len() as f64;
            assert!((shared - 0.75).abs() < 0.1, "{shared}");
        }
        assert_eq!(split_windows(window, 1, 0.75).unwrap(), vec![window]);
        // No overlap means no exchanges: rejected.
        assert!(matches!(
            split_windows(window, 2, 0.0).unwrap_err(),
            MCIsingError::InvalidWindowSplit(_)
        ));
        // Windows narrower than two bins: rejected.
        assert!(matches!(
            split_windows(BinWindow { lo: 0, hi: 2 }, 8, 0.75).unwrap_err(),
            MCIsingError::InvalidWindowSplit(_)
        ));
    }

    #[test]
    fn test_merge_windows_joins_pieces_at_matching_slopes() {
        // An exact parabola split into two overlapping pieces with an
        // arbitrary offset on the second: the merge recovers it.
        let truth: Vec<f64> = (0..40_usize)
            .map(|k| -((k as f64 - 20.0) / 6.0).powi(2))
            .collect();
        let first = BinWindow { lo: 0, hi: 24 };
        let second = BinWindow { lo: 15, hi: 39 };
        let piece_a = truth[0..=24].to_vec();
        let piece_b: Vec<f64> = truth[15..=39].iter().map(|v| v + 7.5).collect();
        let (merged, joins) = merge_windows(&[(first, piece_a), (second, piece_b)], 40);
        assert_eq!(joins.len(), 1);
        assert!((15..24).contains(&joins[0]));
        for (k, (m, t)) in merged.iter().zip(&truth).enumerate() {
            assert!((m - t).abs() < 1e-9, "bin {k}: {m} vs {t}");
        }
        // Unvisited entries stay unvisited, pieces without overlap keep their offset.
        let (merged, joins) = merge_windows(
            &[
                (BinWindow { lo: 0, hi: 3 }, vec![0.0, 1.0, f64::NAN, 3.0]),
                (BinWindow { lo: 6, hi: 8 }, vec![10.0, 11.0, 12.0]),
            ],
            10,
        );
        assert!(merged[2].is_nan() && merged[4].is_nan() && merged[9].is_nan());
        assert_eq!(merged[6], 10.0);
        assert_eq!(joins, vec![6]);
    }

    fn parallel_params(
        n_windows: usize,
        walkers_per_window: usize,
        log_f_final: f64,
        production_sweeps: u64,
    ) -> WangLandauParams<'static> {
        let mut p = params(
            "square",
            4,
            [1.0, 0.0, 0.0, 0.0],
            log_f_final,
            production_sweeps,
        );
        p.n_windows = n_windows;
        p.walkers_per_window = walkers_per_window;
        p.exchange_interval = 20;
        p.schedule.check_interval = 100;
        p
    }

    #[test]
    fn test_replica_exchange_stage_matches_exact_log_g() {
        let dos = square4_dos();
        let exact = dos.log_g_by_energy(1.0, 0.0, 0.0);
        for (n_windows, per_window) in [(1, 3), (2, 2), (3, 1)] {
            let p = parallel_params(n_windows, per_window, 1e-5, 20_000);
            let result = run(&p);
            let wl = &result.wl;
            assert_eq!(wl.n_windows, n_windows);
            assert_eq!(wl.walkers_per_window, per_window);
            assert_eq!(wl.window_bins.len(), n_windows);
            assert_eq!(wl.exchange_attempted.len(), n_windows - 1);
            assert_eq!(wl.merge_bins.len(), n_windows - 1);
            if n_windows > 1 {
                assert!(wl.exchange_attempted.iter().all(|&a| a > 0));
                assert!(wl.exchange_accepted.iter().sum::<u64>() > 0);
            }
            assert_eq!(
                wl.attempted,
                wl.total_sweeps * 16 * (n_windows * per_window) as u64
            );
            assert!(wl.converged);
            let deviation = aligned_deviation(&result, &exact);
            println!(
                "calib rewl {n_windows}x{per_window}: max|dln g|={deviation:.4} sweeps={} \
                 exchanges={:?}/{:?}",
                wl.total_sweeps, wl.exchange_accepted, wl.exchange_attempted
            );
            // Pooled flatness ends the halving stage sooner than a single
            // walker's, so at log_f_final = 1e-5 the residual is a little larger.
            assert!(deviation <= 0.15, "{n_windows}x{per_window}: {deviation}");
            // The parallel schedule ends after fewer sweeps per walker (its
            // clock pools the walkers), so the canonical gate is 2 % / 5 %.
            for temperature in GATE_TEMPERATURES {
                let ex = dos.exact_at(temperature, 1.0);
                let (e, cv) =
                    canonical_from_log_g(&result.binning, &result.log_g, 1.0 / temperature);
                assert!(
                    ((e - ex.energy) / ex.energy).abs() <= 0.02,
                    "T={temperature}: {e} vs {}",
                    ex.energy
                );
                assert!(
                    ((cv - ex.specific_heat) / ex.specific_heat).abs() <= 0.05,
                    "T={temperature}: {cv} vs {}",
                    ex.specific_heat
                );
            }
            // Production ran on the merged estimate from every walker's end state.
            assert_eq!(result.production.len(), 1);
            assert_eq!(result.production[0].series.energies.len(), 20_000);
            assert!(result.production_diagnostics.histogram_flatness > 0.5);
            let e = reweight_series(&result, 1.0 / 2.269);
            let exact_e = dos.exact_at(2.269, 1.0).energy;
            assert!(((e - exact_e) / exact_e).abs() <= 0.02, "{e} vs {exact_e}");
        }
    }

    #[test]
    fn test_replica_exchange_stage_is_deterministic_across_thread_counts() {
        let p = parallel_params(2, 2, 1e-3, 1_000);
        let serial = with_pool(1, || run(&p));
        let parallel = with_pool(4, || run(&p));
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&serial.log_g), bits(&parallel.log_g));
        assert_eq!(serial.wl_histogram, parallel.wl_histogram);
        assert_eq!(serial.wl.exchange_accepted, parallel.wl.exchange_accepted);
        assert_eq!(
            bits(&serial.production[0].series.energies),
            bits(&parallel.production[0].series.energies)
        );
        assert_eq!(serial.final_spins, parallel.final_spins);
    }

    #[test]
    fn test_replica_exchange_validation() {
        let err = |p: WangLandauParams<'_>| error_of(&p).to_string();
        let mut p = parallel_params(2, 2, 1e-3, 10);
        p.window_overlap = 1.0;
        assert!(err(p).contains("window_overlap"));
        let mut p = parallel_params(2, 2, 1e-3, 10);
        p.exchange_interval = 30; // 100 is not a multiple of 30
        assert!(err(p).contains("exchange_interval"));
        let p = parallel_params(0, 1, 1e-3, 10);
        assert!(err(p).contains("n_windows"));
        let p = parallel_params(1, 0, 1e-3, 10);
        assert!(err(p).contains("walkers_per_window"));
        let mut p = parallel_params(1, 1, 1e-3, 10);
        p.exchange_interval = 0;
        assert!(err(p).contains("exchange_interval"));
        // The serial path ignores the cadence rule.
        let mut p = parallel_params(1, 1, 1e-3, 10);
        p.exchange_interval = 30;
        assert!(run_wang_landau_internal(&p).is_ok());
        // Too many windows for a three-bin range.
        let mut p = parallel_params(8, 1, 1e-3, 10);
        p.energy_window = Some((-2.0, -1.5));
        assert!(err(p).contains("windows"));
    }

    #[test]
    fn test_replica_exchange_with_window_and_cap() {
        let mut p = parallel_params(2, 2, 1e-3, 500);
        p.energy_window = Some((-1.5, 0.5));
        p.schedule.max_sweeps = Some(150);
        let result = run(&p);
        assert!(!result.wl.converged);
        assert_eq!(result.wl.total_sweeps, 150);
        assert_eq!(result.wl.window_bins[0].0, 2);
        assert_eq!(result.wl.window_bins[1].1, 10);
        assert!(result.production[0]
            .bins
            .iter()
            .all(|&b| (2..=10).contains(&(b as usize))));
    }

    #[test]
    fn test_honeycomb_and_cubic_smoke() {
        let mut p = params("honeycomb", 4, [1.0, 0.0, 0.0, 0.0], 1e-3, 200);
        p.schedule.max_sweeps = Some(200);
        let result = run(&p);
        assert_eq!(result.binning.width, 2.0);
        assert_eq!(result.binning.e_min, -48.0);
        assert_eq!(result.binning.n_bins, 49);
        assert!(!result.wl.converged);
        assert!(result.production[0].bins.iter().all(|&b| (b as usize) < 49));

        let mut p = params("cubic", 4, [1.0, -0.5, 0.0, 0.0], 1e-3, 200);
        p.schedule.max_sweeps = Some(300);
        let result = run(&p);
        assert_eq!(result.binning.width, 2.0);
        assert_eq!(result.binning.num_sites, 64);
        assert!(result.production[0]
            .bins
            .iter()
            .all(|&b| result.log_g[b as usize].is_finite()));
        assert_eq!(result.production[0].series.staggered.len(), 200 * 8);
    }
}
