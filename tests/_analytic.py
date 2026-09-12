"""Closed-form reference values for the exact-results validation suite.

Test-local companion to :mod:`tests._stats`: these are referee-facing
oracles, not library features, so they stay out of ``mcising`` (feature
freeze). Every function states the formula it implements so a reviewer
can check the derivation without running anything.

Units throughout: J = 1, k_B = 1, energies per site, h = 0.
"""

from __future__ import annotations

import math

__all__ = [
    "chain_energy_per_site",
    "chain_susceptibility_signed",
    "complete_elliptic_k",
    "onsager_energy_per_site",
    "square_torus_energy_per_site",
    "square_torus_log_z",
    "square_torus_specific_heat",
]


def complete_elliptic_k(k: float) -> float:
    """Complete elliptic integral of the first kind, K(k).

    K(k) = int_0^{pi/2} dtheta / sqrt(1 - k^2 sin^2 theta), evaluated via
    the arithmetic-geometric mean, K(k) = pi / (2 AGM(1, sqrt(1 - k^2))).
    The AGM iteration converges quadratically, so a handful of steps
    reach double precision for any 0 <= k < 1. K diverges
    logarithmically as k -> 1, which is why the domain excludes it.
    Implemented here because numpy is the package's only numeric
    dependency (no scipy).
    """
    if not 0.0 <= k < 1.0:
        raise ValueError(f"K(k) requires 0 <= k < 1, got {k}")
    a, b = 1.0, math.sqrt(1.0 - k * k)
    for _ in range(64):
        if abs(a - b) <= 1e-15 * a:
            break
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    return math.pi / (2.0 * a)


def onsager_energy_per_site(temperature: float) -> float:
    """Exact internal energy per site of the infinite square lattice.

    Onsager (1944), with beta = 1/T and J = 1:

        u(T) = -coth(2 beta) [1 + (2/pi) (2 tanh^2(2 beta) - 1) K(k1)],
        k1 = 2 sinh(2 beta) / cosh^2(2 beta).

    Checks: as T -> infinity the bracket is 4 beta^2 + O(beta^4), giving
    u -> -2 beta = -(z/2) tanh(beta), the high-temperature expansion. At
    Tc, sinh(2 beta_c) = 1 so tanh^2(2 beta_c) = 1/2 exactly and the
    bracket is 1, giving u(Tc) = -coth(2 beta_c) = -sqrt(2).

    Do not evaluate at exactly Tc: k1 = 1 there and the product is
    0 * infinity in floating point (:func:`complete_elliptic_k` raises).
    The limit from either side is -sqrt(2); use that constant instead.
    """
    two_beta = 2.0 / temperature
    sinh, cosh = math.sinh(two_beta), math.cosh(two_beta)
    k1 = 2.0 * sinh / (cosh * cosh)
    tanh_sq = (sinh / cosh) ** 2
    bracket = 1.0 + (2.0 / math.pi) * (2.0 * tanh_sq - 1.0) * complete_elliptic_k(k1)
    return -(cosh / sinh) * bracket


def chain_energy_per_site(n: int, temperature: float) -> float:
    """Exact energy per site of the N-site periodic Ising chain.

    Transfer-matrix eigenvalues lambda_+ = 2 cosh(beta), lambda_- =
    2 sinh(beta) give Z = lambda_+^N + lambda_-^N, and with
    t = tanh(beta) = lambda_- / lambda_+:

        e(N, T) = -(1/N) d ln Z / d beta = -(t + t^(N-1)) / (1 + t^N).

    The N -> infinity limit is Ising's (1925) -tanh(beta). The finite-N
    correction is O(t^N): at T = 0.8, N = 64 it is ~2e-5 relative to a
    statistical error of the same order, so the finite form is the honest
    reference for the sizes the tests run.
    """
    t = math.tanh(1.0 / temperature)
    return -(t + t ** (n - 1)) / (1.0 + t**n)


def chain_susceptibility_signed(n: int, temperature: float) -> float:
    """Exact signed susceptibility per site of the N-site periodic chain.

    With t = tanh(beta) the periodic two-point function is
    <s_0 s_r> = (t^r + t^(N-r)) / (1 + t^N), and translation invariance
    gives N <m^2> = sum_{r=0}^{N-1} <s_0 s_r>. Both geometric sums close:

        chi(N, T) = N <m^2> / T = (1/T) (1 + t) (1 - t^N) / ((1 - t) (1 + t^N)).

    The N -> infinity limit is beta exp(2 beta). This is the *signed*
    convention N Var(m) / T (``kind="signed"`` in
    :func:`mcising.statistics.susceptibility`): <m> = 0 in the full
    trace, so N Var(m) = N <m^2>. The package's default "connected"
    form N Var(|m|) / T involves <|m|>, which has no closed form.
    """
    t = math.tanh(1.0 / temperature)
    t_n = t**n
    return (1.0 + t) * (1.0 - t_n) / ((1.0 - t) * (1.0 + t_n)) / temperature


def _log_2cosh(x: float) -> float:
    """ln(2 cosh x) without overflow: |x| + ln(1 + exp(-2|x|))."""
    a = abs(x)
    return a + math.log1p(math.exp(-2.0 * a))


def _log_abs_2sinh(x: float) -> float:
    """ln|2 sinh x| without overflow: |x| + ln(1 - exp(-2|x|)); -inf at 0."""
    a = abs(x)
    if a == 0.0:
        return -math.inf
    return a + math.log(-math.expm1(-2.0 * a))


def square_torus_log_z(size: int, temperature: float) -> float:
    """Exact ln Z of the L x L periodic square-lattice Ising model (J = 1).

    Ferdinand & Fisher, Phys. Rev. 185, 832 (1969), eqs. (2.1)-(2.5), the
    finite-lattice form of Kaufman's solution; with K = beta J, n = m = L:

        Z = (1/2) (2 sinh 2K)^(L^2/2) (Z_1 + Z_2 + Z_3 + Z_4),
        Z_1 = prod_{r=0}^{L-1} 2 cosh(L gamma_{2r+1} / 2),
        Z_2 = prod_{r=0}^{L-1} 2 sinh(L gamma_{2r+1} / 2),
        Z_3 = prod_{r=0}^{L-1} 2 cosh(L gamma_{2r} / 2),
        Z_4 = prod_{r=0}^{L-1} 2 sinh(L gamma_{2r} / 2),
        cosh gamma_l = cosh 2K coth 2K - cos(pi l / L)   (l >= 1),
        gamma_0 = 2K + ln tanh K   (negative below T_c: Z_4 changes sign).

    Evaluated in log space (the products overflow above L ~ 30). The
    sum is positive because Z_1 >= |Z_2| and Z_3 >= |Z_4| term by term.
    Checks: L = 4 against a brute-force enumeration of the 65 536 states,
    and L = 64 against Onsager's energy (``tests/test_analytic.py``).
    """
    if size < 2:
        raise ValueError(f"square_torus_log_z needs L >= 2, got {size}")
    k = 1.0 / temperature
    cosh2k, sinh2k = math.cosh(2.0 * k), math.sinh(2.0 * k)
    coth2k = cosh2k / sinh2k

    def gamma(index: int) -> float:
        if index == 0:
            return 2.0 * k + math.log(math.tanh(k))
        return math.acosh(cosh2k * coth2k - math.cos(math.pi * index / size))

    half = 0.5 * size
    log_z1 = sum(_log_2cosh(half * gamma(2 * r + 1)) for r in range(size))
    log_z2 = sum(_log_abs_2sinh(half * gamma(2 * r + 1)) for r in range(size))
    log_z3 = sum(_log_2cosh(half * gamma(2 * r)) for r in range(size))
    log_z4 = sum(_log_abs_2sinh(half * gamma(2 * r)) for r in range(size))
    sign_z4 = -1.0 if gamma(0) < 0.0 else 1.0
    top = max(log_z1, log_z2, log_z3, log_z4)
    total = (
        math.exp(log_z1 - top)
        + math.exp(log_z2 - top)
        + math.exp(log_z3 - top)
        + sign_z4 * math.exp(log_z4 - top)
    )
    prefactor = 0.5 * size * size * math.log(2.0 * sinh2k)
    return -math.log(2.0) + prefactor + top + math.log(total)


def square_torus_energy_per_site(size: int, temperature: float) -> float:
    """Exact energy per site of the L x L torus, -(1/N) d ln Z / d beta.

    Central difference of :func:`square_torus_log_z` in beta with step
    1e-4 (truncation error ~1e-8 relative, far below any Monte Carlo
    error bar). Converges to :func:`onsager_energy_per_site` as L grows;
    at T_c the finite-L value sits ~0.6/L above -sqrt(2).
    """
    beta = 1.0 / temperature
    step = 1e-4
    upper = square_torus_log_z(size, 1.0 / (beta + step))
    lower = square_torus_log_z(size, 1.0 / (beta - step))
    return -(upper - lower) / (2.0 * step) / (size * size)


def square_torus_specific_heat(size: int, temperature: float) -> float:
    """Exact specific heat per site of the L x L torus, beta^2 d^2 ln Z / d beta^2 / N.

    Second central difference in beta with step 1e-3 (truncation ~1e-6
    relative, rounding ~1e-7); matches the fluctuation form
    ``N Var(e) / T^2`` of :func:`mcising.statistics.specific_heat`.
    """
    beta = 1.0 / temperature
    step = 1e-3
    upper = square_torus_log_z(size, 1.0 / (beta + step))
    centre = square_torus_log_z(size, temperature)
    lower = square_torus_log_z(size, 1.0 / (beta - step))
    second = (upper - 2.0 * centre + lower) / (step * step)
    return beta * beta * second / (size * size)
