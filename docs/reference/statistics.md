# Statistics

Autocorrelation-aware error estimation for Monte Carlo observables.
Means carry blocking (Flyvbjerg–Petersen) standard errors; derived
quantities (specific heat, susceptibility, Binder cumulant) carry
delete-one-block jackknife errors. The high-level entry point is
`SimulationResults.statistics(temperature)`, which returns an
[`ObservableStatistics`][mcising.ObservableStatistics].

::: mcising.Estimate

---

::: mcising.ObservableStatistics

---

::: mcising.statistics.blocking_se

---

::: mcising.statistics.blocking_curve

---

::: mcising.statistics.naive_se

---

::: mcising.statistics.tau_int

---

::: mcising.statistics.tau_int_blocking

---

::: mcising.statistics.jackknife_se

---

::: mcising.statistics.auto_n_blocks

---

::: mcising.statistics.binder_cumulant

---

::: mcising.statistics.specific_heat

---

::: mcising.statistics.susceptibility

---

::: mcising.statistics.mean_estimate

---

::: mcising.statistics.jackknife_estimate

---

::: mcising.statistics.observable_statistics
