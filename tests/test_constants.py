"""The critical-temperature constants satisfy their cited closed forms."""

from __future__ import annotations

import math

import pytest
from mcising.constants import (
    TC_CUBIC_3D,
    TC_HONEYCOMB_2D,
    TC_SQUARE_2D,
    TC_TRIANGULAR_2D,
)


class TestCriticalTemperatures:
    """Exact 2D solutions (Onsager 1944; Houtappel 1950; Wannier 1950) and
    the cubic Monte Carlo value (Ferrenberg, Xu & Landau 2018).

    The 2D constants are pinned through the defining identities rather
    than decimal literals, so a typo in the closed form cannot hide behind
    a matching typo in the test. The campaign agreement test lives in
    ``tests/test_tc_campaign.py``.
    """

    def test_square_is_onsagers_self_dual_point(self) -> None:
        assert math.sinh(2.0 / TC_SQUARE_2D) == pytest.approx(1.0, abs=1e-12)

    def test_triangular_is_houtappels_point(self) -> None:
        assert math.exp(4.0 / TC_TRIANGULAR_2D) == pytest.approx(3.0, abs=1e-12)

    def test_honeycomb_is_houtappels_point(self) -> None:
        assert math.cosh(2.0 / TC_HONEYCOMB_2D) == pytest.approx(2.0, abs=1e-12)

    def test_honeycomb_and_triangular_are_duals(self) -> None:
        """The two lattices are Kramers-Wannier duals: their critical
        couplings satisfy sinh(2 K_tri) sinh(2 K_hc) = 1 (Wannier 1950), a
        cross-check that the two closed forms belong to the same pair."""
        product = math.sinh(2.0 / TC_TRIANGULAR_2D) * math.sinh(2.0 / TC_HONEYCOMB_2D)
        assert product == pytest.approx(1.0, abs=1e-12)

    def test_cubic_is_the_ferrenberg_xu_landau_value(self) -> None:
        # beta_c = 0.221654626(5); the constant is its reciprocal exactly.
        assert 1.0 / TC_CUBIC_3D == pytest.approx(0.221654626, abs=1e-12)
        # The rounded pre-0.28.0 literal remains the correct 4-decimal value,
        # so nothing quoting "4.5115" (CLI help, README) went stale.
        assert round(TC_CUBIC_3D, 4) == 4.5115

    def test_ordering(self) -> None:
        """More neighbours, higher Tc: honeycomb (z=3) < square (4) <
        triangular and cubic (6); the 3D lattice orders above the 2D one at
        equal coordination."""
        assert TC_HONEYCOMB_2D < TC_SQUARE_2D < TC_TRIANGULAR_2D < TC_CUBIC_3D
