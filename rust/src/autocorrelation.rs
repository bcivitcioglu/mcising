//! Autocorrelation analysis and thermalization detection.
//!
//! Provides:
//! - MSER (Marginal Standard Error Rule) for detecting when a time series
//!   has reached stationarity (thermalization).
//! - Sokal's automatic windowing method for estimating the integrated
//!   autocorrelation time of a stationary time series.
//! - Combined analysis that runs MSER first, then Sokal on the stationary
//!   tail.

/// Result of MSER thermalization detection.
#[derive(Debug, Clone)]
pub struct ThermalizationResult {
    /// Index where stationarity begins (truncation point).
    pub truncation_point: usize,
    /// Whether the series appears thermalized (truncation_point < len/2).
    pub is_thermalized: bool,
}

/// Result of integrated autocorrelation time estimation.
#[derive(Debug, Clone)]
pub struct AutocorrelationResult {
    /// Estimated integrated autocorrelation time.
    pub tau_int: f64,
    /// The windowing cutoff used.
    pub window: usize,
}

/// Combined thermalization + autocorrelation analysis result.
#[derive(Debug, Clone)]
pub struct ThermalizationAnalysis {
    pub thermalization: ThermalizationResult,
    pub autocorrelation: AutocorrelationResult,
    /// Recommended measurement interval in sweeps.
    pub recommended_interval: usize,
}

/// Detect thermalization using the Marginal Standard Error Rule (MSER).
///
/// For a time series x_0, ..., x_{N-1}, finds the truncation point d
/// that minimizes the standard error of the mean of x_d, ..., x_{N-1}.
///
/// MSER statistic at truncation point d: Var(x_d..x_N) / (N - d)
///
/// Every candidate d in [0, N/2] is evaluated exactly (single backward
/// Welford pass, O(N)). The classical acceptance rule (White 1997) is
/// kept: the series is thermalized only if the minimizing d falls in the
/// first half — an argmin at the boundary means the data cannot
/// demonstrate stationarity ("insufficient data"). Restricting candidates
/// to d ≤ N/2 also guarantees every evaluated tail has ≥ N/2 points, so
/// the small-tail pathology (a noisy variance estimate on a tiny tail
/// winning the argmin) is structurally impossible and no minimum-tail
/// constant is needed. Ties resolve to the smallest d, so a zero-variance
/// (frozen) series is thermalized at d = 0, not data-starved.
///
/// Before P09 the search used a 20-point grid whose largest candidate,
/// `20·⌊(N/2)/20⌋`, reached the boundary only when 20 | (N/2) — making
/// the not-thermalized verdict unreachable for most N (B9, #20).
pub fn detect_thermalization(series: &[f64]) -> ThermalizationResult {
    let n = series.len();
    if n < 4 {
        return ThermalizationResult {
            truncation_point: 0,
            is_thermalized: true,
        };
    }

    let mut best_d = 0;
    let mut best_mser = f64::MAX;

    // Backward Welford pass: extend the tail one sample leftward per
    // step; at d ≤ n/2 the tail has n−d ≥ 2 samples, so the variance is
    // always defined.
    let mut count = 0usize;
    let mut mean = 0.0;
    let mut m2 = 0.0;

    for d in (0..n).rev() {
        count += 1;
        let x = series[d];
        let delta = x - mean;
        mean += delta / count as f64;
        m2 += delta * (x - mean);

        if d <= n / 2 {
            let var = m2 / (count - 1) as f64;
            let mser = var / count as f64;
            // `<=`: iterating d downward, so on ties the smallest d wins.
            if mser <= best_mser {
                best_mser = mser;
                best_d = d;
            }
        }
    }

    ThermalizationResult {
        truncation_point: best_d,
        is_thermalized: best_d < n / 2,
    }
}

/// Compute integrated autocorrelation time using Sokal's automatic windowing.
///
/// Given a stationary time series, computes:
///   C(t) = (<x(i)*x(i+t)> - <x>^2) / (<x^2> - <x>^2)
///   tau_int = 0.5 + sum_{t=1}^{M} C(t)
///
/// where M is the smallest t such that t >= c_window * tau_int(t).
///
/// # Arguments
/// * `series` - Stationary time series (should already be truncated past transient)
/// * `c_window` - Windowing constant (typically 6.0, per Sokal's recommendation)
pub fn integrated_autocorrelation_time(series: &[f64], c_window: f64) -> AutocorrelationResult {
    let n = series.len();
    if n < 4 {
        return AutocorrelationResult {
            tau_int: 0.5,
            window: 0,
        };
    }

    let mean = series.iter().sum::<f64>() / n as f64;
    let var = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-15 {
        return AutocorrelationResult {
            tau_int: 0.5,
            window: 0,
        };
    }

    let max_lag = n / 2;
    let mut tau_int = 0.5;
    let mut window = 0;

    for t in 1..max_lag {
        // Direct computation of C(t) — O(N) per lag, but we truncate early via windowing
        let mut autocov = 0.0;
        for i in 0..(n - t) {
            autocov += (series[i] - mean) * (series[i + t] - mean);
        }
        autocov /= n as f64;

        let rho_t = autocov / var;
        tau_int += rho_t;

        // Sokal's self-consistent windowing criterion
        if (t as f64) >= c_window * tau_int {
            window = t;
            break;
        }
    }

    // If we never hit the criterion, use the full range (noisy estimate)
    if window == 0 {
        window = max_lag.saturating_sub(1).max(1);
    }

    // tau_int should be at least 0.5 (uncorrelated limit)
    tau_int = tau_int.max(0.5);

    AutocorrelationResult { tau_int, window }
}

/// Combined analysis: detect thermalization, then estimate autocorrelation
/// on the stationary tail.
///
/// # Arguments
/// * `series` - Full thermalization energy time series
/// * `c_window` - Sokal windowing constant (typically 6.0)
/// * `tau_multiplier` - Multiplier for tau_int to get recommended interval (typically 2.0)
pub fn analyze_thermalization(
    series: &[f64],
    c_window: f64,
    tau_multiplier: f64,
) -> ThermalizationAnalysis {
    let therm = detect_thermalization(series);

    // Estimate tau on the post-truncation tail whenever it has enough
    // points — even when not thermalized: the caller may have exhausted
    // its sweep budget and must act on the best available estimate
    // rather than a silently optimistic tau = 0.5. `is_thermalized`
    // remains the honesty flag.
    let tail = &series[therm.truncation_point.min(series.len())..];
    let autocorr = if tail.len() >= 4 {
        integrated_autocorrelation_time(tail, c_window)
    } else {
        AutocorrelationResult {
            tau_int: 0.5,
            window: 0,
        }
    };

    let recommended_interval = (tau_multiplier * autocorr.tau_int).round().max(1.0) as usize;

    ThermalizationAnalysis {
        thermalization: therm,
        autocorrelation: autocorr,
        recommended_interval,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_white_noise_tau_near_half() {
        // Uncorrelated white noise should have tau_int ~ 0.5
        use rand::Rng;
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256StarStar;

        let mut rng = Xoshiro256StarStar::seed_from_u64(42);
        let series: Vec<f64> = (0..10_000).map(|_| rng.gen::<f64>()).collect();

        let result = integrated_autocorrelation_time(&series, 6.0);
        assert!(
            result.tau_int < 1.5,
            "White noise tau_int should be near 0.5, got {}",
            result.tau_int
        );
    }

    #[test]
    fn test_correlated_series_higher_tau() {
        // AR(1) process: x_{t+1} = phi * x_t + noise
        // Theoretical tau_int = (1 + phi) / (2 * (1 - phi))
        // For phi=0.9: tau_int = 1.9 / 0.2 = 9.5
        use rand::Rng;
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256StarStar;

        let phi = 0.9;
        let mut rng = Xoshiro256StarStar::seed_from_u64(42);
        let n = 100_000;
        let mut series = Vec::with_capacity(n);
        let mut x = 0.0;
        for _ in 0..n {
            x = phi * x + rng.gen::<f64>() - 0.5;
            series.push(x);
        }

        let result = integrated_autocorrelation_time(&series, 6.0);
        let theoretical = (1.0 + phi) / (2.0 * (1.0 - phi));
        assert!(
            (result.tau_int - theoretical).abs() < 3.0,
            "AR(1) phi=0.9: expected tau~{theoretical:.1}, got {:.1}",
            result.tau_int
        );
    }

    #[test]
    fn test_mser_detects_stationary_series() {
        // Already stationary series — truncation point should be near 0
        use rand::Rng;
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256StarStar;

        let mut rng = Xoshiro256StarStar::seed_from_u64(42);
        let series: Vec<f64> = (0..1000).map(|_| rng.gen::<f64>()).collect();

        let result = detect_thermalization(&series);
        assert!(
            result.is_thermalized,
            "Stationary series should be detected as thermalized"
        );
        assert!(
            result.truncation_point < 100,
            "Truncation point should be near start for stationary data, got {}",
            result.truncation_point
        );
    }

    #[test]
    fn test_mser_detects_transient() {
        // Series with a clear transient: first 200 values at high level, then settle to 0
        let mut series = Vec::with_capacity(1000);
        for i in 0..1000 {
            if i < 200 {
                series.push(10.0 - f64::from(i) * 0.05); // Decaying from 10 to 0
            } else {
                series.push(0.1 * (f64::from(i) * 0.01).sin()); // Small fluctuations around 0
            }
        }

        let result = detect_thermalization(&series);
        assert!(
            result.is_thermalized,
            "Series with transient + stationary tail should be thermalized"
        );
        // Truncation point should be somewhere around 150-250
        assert!(
            result.truncation_point >= 100 && result.truncation_point <= 350,
            "Truncation point should be near the transient end (~200), got {}",
            result.truncation_point
        );
    }

    #[test]
    fn test_mser_rejects_linear_ramp() {
        // The P09 gate: a pure linear ramp is never stationary, so MSER
        // must return not-thermalized — at EVERY length. The variance of
        // a length-k ramp tail grows like k², so MSER(d) ~ (n-d) is
        // minimized at the largest candidate d = n/2, which is exactly
        // the "insufficient data" boundary. The pre-P09 20-point grid
        // reached n/2 only when 20 | (n/2): n=500 (max candidate 240 <
        // 250) and n=317 (140 < 158) were declared thermalized.
        for n in [200usize, 317, 500, 1000] {
            let series: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let result = detect_thermalization(&series);
            assert!(
                !result.is_thermalized,
                "n={n}: linear ramp declared thermalized (d={})",
                result.truncation_point
            );
        }
    }

    #[test]
    fn test_mser_rejects_cooldown_shaped_ramp() {
        // The actual annealing shape the adaptive path produces: a
        // monotone drift from a hot value toward a cold one.
        let n = 200;
        let series: Vec<f64> = (0..n)
            .map(|i| 100.0 + (2.269 - 100.0) * f64::from(i) / f64::from(n - 1))
            .collect();
        let result = detect_thermalization(&series);
        assert!(
            !result.is_thermalized,
            "cooldown ramp declared thermalized (d={})",
            result.truncation_point
        );
    }

    #[test]
    fn test_mser_boundary_argmin_is_not_thermalized() {
        // First half transient, second half stationary: the argmin sits
        // exactly at d = n/2, the classical "insufficient data" verdict.
        // n = 502 makes n/2 = 251 unreachable for the old 20-point grid.
        let n = 502;
        let series: Vec<f64> = (0..n)
            .map(|i| {
                if i < n / 2 {
                    100.0
                } else {
                    (i % 7) as f64 * 0.01
                }
            })
            .collect();
        let result = detect_thermalization(&series);
        assert_eq!(result.truncation_point, n / 2);
        assert!(!result.is_thermalized);
    }

    #[test]
    fn test_mser_accepts_stationary_ar1() {
        // Correlated but stationary series must stay accepted, over
        // several correlation strengths and seeds.
        use rand::Rng;
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256StarStar;

        for phi in [0.5, 0.8, 0.9] {
            for seed in [42u64, 123, 7, 2024, 31337] {
                let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
                let n = 1000;
                let mut series = Vec::with_capacity(n);
                let mut x = 0.0;
                for _ in 0..n {
                    x = phi * x + rng.gen::<f64>() - 0.5;
                    series.push(x);
                }
                let result = detect_thermalization(&series);
                assert!(
                    result.is_thermalized,
                    "phi={phi} seed={seed}: stationary AR(1) rejected (d={})",
                    result.truncation_point
                );
            }
        }
    }

    #[test]
    fn test_mser_constant_series_is_thermalized() {
        // Zero variance everywhere: every truncation point ties at
        // MSER = 0. Ties must resolve to the smallest d (a frozen,
        // perfectly equilibrated series is thermalized, not data-starved).
        let series = vec![42.0; 1000];
        let result = detect_thermalization(&series);
        assert_eq!(result.truncation_point, 0);
        assert!(result.is_thermalized);
    }

    #[test]
    fn test_mser_not_thermalized_all_drift() {
        // Monotonically drifting series — never thermalized
        let series: Vec<f64> = (0..1000).map(f64::from).collect();
        let result = detect_thermalization(&series);
        // For monotonic drift, the best truncation point is near the end
        // because variance is minimized when we take the shortest tail
        assert!(
            !result.is_thermalized || result.truncation_point > 400,
            "Drifting series should not be detected as thermalized or have late truncation, got d={}",
            result.truncation_point
        );
    }

    #[test]
    fn test_analyze_combined() {
        // Stationary series — should detect thermalization and estimate tau
        use rand::Rng;
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256StarStar;

        let mut rng = Xoshiro256StarStar::seed_from_u64(42);
        let series: Vec<f64> = (0..5000).map(|_| rng.gen::<f64>()).collect();

        let result = analyze_thermalization(&series, 6.0, 2.0);
        assert!(result.thermalization.is_thermalized);
        assert!(result.autocorrelation.tau_int >= 0.5);
        assert!(result.recommended_interval >= 1);
    }

    #[test]
    // tau_int == 0.5 is an exact literal returned by the degenerate path.
    #[allow(clippy::float_cmp)]
    fn test_short_series_does_not_panic() {
        let series = vec![1.0, 2.0];
        let result = analyze_thermalization(&series, 6.0, 2.0);
        assert!(result.thermalization.is_thermalized);
        assert_eq!(result.autocorrelation.tau_int, 0.5);
    }

    #[test]
    // tau_int == 0.5 is an exact literal returned by the zero-variance path.
    #[allow(clippy::float_cmp)]
    fn test_constant_series() {
        // All identical values — zero variance, tau should be 0.5
        let series = vec![42.0; 1000];
        let result = integrated_autocorrelation_time(&series, 6.0);
        assert_eq!(result.tau_int, 0.5, "Constant series should have tau=0.5");
    }
}
