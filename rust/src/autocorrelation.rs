/// Autocorrelation analysis and thermalization detection.
///
/// Provides:
/// - MSER (Marginal Standard Error Rule) for detecting when a time series
///   has reached stationarity (thermalization).
/// - Sokal's automatic windowing method for estimating the integrated
///   autocorrelation time of a stationary time series.
/// - Combined analysis that runs MSER first, then Sokal on the stationary tail.

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
/// The series is considered thermalized if d < N/2.
pub fn detect_thermalization(series: &[f64]) -> ThermalizationResult {
    let n = series.len();
    if n < 4 {
        return ThermalizationResult {
            truncation_point: 0,
            is_thermalized: true,
        };
    }

    // Evaluate candidate truncation points at regular intervals.
    // Use ~20 candidates to keep cost O(N), not O(N^2).
    let n_candidates = 20.min(n / 2);
    let step = (n / 2).max(1) / n_candidates.max(1);

    let mut best_d = 0;
    let mut best_mser = f64::MAX;

    for k in 0..=n_candidates {
        let d = (k * step).min(n - 2);
        let tail = &series[d..];
        let tail_n = tail.len();
        if tail_n < 2 {
            continue;
        }

        let mean = tail.iter().sum::<f64>() / tail_n as f64;
        let var = tail.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (tail_n - 1) as f64;
        let mser = var / tail_n as f64;

        if mser < best_mser {
            best_mser = mser;
            best_d = d;
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

    let autocorr = if therm.is_thermalized && therm.truncation_point < series.len() {
        let tail = &series[therm.truncation_point..];
        if tail.len() >= 4 {
            integrated_autocorrelation_time(tail, c_window)
        } else {
            AutocorrelationResult {
                tau_int: 0.5,
                window: 0,
            }
        }
    } else {
        AutocorrelationResult {
            tau_int: 0.5,
            window: 0,
        }
    };

    let recommended_interval = (tau_multiplier * autocorr.tau_int)
        .round()
        .max(1.0) as usize;

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
        use rand_xoshiro::Xoshiro256StarStar;
        use rand::SeedableRng;

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
        use rand_xoshiro::Xoshiro256StarStar;
        use rand::SeedableRng;

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
        use rand_xoshiro::Xoshiro256StarStar;
        use rand::SeedableRng;

        let mut rng = Xoshiro256StarStar::seed_from_u64(42);
        let series: Vec<f64> = (0..1000).map(|_| rng.gen::<f64>()).collect();

        let result = detect_thermalization(&series);
        assert!(result.is_thermalized, "Stationary series should be detected as thermalized");
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
                series.push(10.0 - (i as f64) * 0.05); // Decaying from 10 to 0
            } else {
                series.push(0.1 * ((i as f64) * 0.01).sin()); // Small fluctuations around 0
            }
        }

        let result = detect_thermalization(&series);
        assert!(result.is_thermalized, "Series with transient + stationary tail should be thermalized");
        // Truncation point should be somewhere around 150-250
        assert!(
            result.truncation_point >= 100 && result.truncation_point <= 350,
            "Truncation point should be near the transient end (~200), got {}",
            result.truncation_point
        );
    }

    #[test]
    fn test_mser_not_thermalized_all_drift() {
        // Monotonically drifting series — never thermalized
        let series: Vec<f64> = (0..1000).map(|i| i as f64).collect();
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
        use rand_xoshiro::Xoshiro256StarStar;
        use rand::SeedableRng;

        let mut rng = Xoshiro256StarStar::seed_from_u64(42);
        let series: Vec<f64> = (0..5000).map(|_| rng.gen::<f64>()).collect();

        let result = analyze_thermalization(&series, 6.0, 2.0);
        assert!(result.thermalization.is_thermalized);
        assert!(result.autocorrelation.tau_int >= 0.5);
        assert!(result.recommended_interval >= 1);
    }

    #[test]
    fn test_short_series_does_not_panic() {
        let series = vec![1.0, 2.0];
        let result = analyze_thermalization(&series, 6.0, 2.0);
        assert!(result.thermalization.is_thermalized);
        assert_eq!(result.autocorrelation.tau_int, 0.5);
    }

    #[test]
    fn test_constant_series() {
        // All identical values — zero variance, tau should be 0.5
        let series = vec![3.14; 1000];
        let result = integrated_autocorrelation_time(&series, 6.0);
        assert_eq!(result.tau_int, 0.5, "Constant series should have tau=0.5");
    }
}