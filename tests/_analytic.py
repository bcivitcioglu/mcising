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
