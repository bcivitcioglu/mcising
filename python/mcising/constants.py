"""Centralized physics constants and simulation defaults."""

import math
from typing import Final

# Critical temperatures of the nearest-neighbour ferromagnet (J1=1, J2=J3=h=0).
# Sources — exact 2D solutions: L. Onsager, Phys. Rev. 65, 117 (1944) [square,
# sinh(2/Tc) = 1]; R. M. F. Houtappel, Physica 16, 425 (1950) and G. H. Wannier,
# Phys. Rev. 79, 357 (1950) [triangular exp(4/Tc) = 3, honeycomb cosh(2/Tc) = 2].
# Cubic: Monte Carlo, beta_c = 0.221654626(5), A. M. Ferrenberg, J. Xu &
# D. P. Landau, Phys. Rev. E 97, 043301 (2018). All four are re-measured by
# scripts/tc_campaign.py (Binder crossings; results in
# scripts/tc_campaign_results.json, checked by tests/test_tc_campaign.py).
TC_SQUARE_2D: Final[float] = 2.0 / math.log(1.0 + math.sqrt(2.0))  # 2.269185...
TC_TRIANGULAR_2D: Final[float] = 4.0 / math.log(3.0)  # 3.640957...
TC_HONEYCOMB_2D: Final[float] = 2.0 / math.log(2.0 + math.sqrt(3.0))  # 1.518651...
# 1/beta_c (Ferrenberg, Xu & Landau 2018); the rounded literal 4.5115 used
# before 0.28.0 differs by 5 ppm.
TC_CUBIC_3D: Final[float] = 1.0 / 0.221654626  # 4.511523...

# High temperature used for cool-down initialization
INF_TEMP: Final[float] = 100.0


# Default simulation parameters
DEFAULT_LATTICE_SIZE: Final[int] = 10
DEFAULT_J1: Final[float] = 1.0
DEFAULT_J2: Final[float] = 0.0
DEFAULT_J3: Final[float] = 0.0
DEFAULT_H: Final[float] = 0.0
DEFAULT_SEED: Final[int] = 42
DEFAULT_N_SWEEPS: Final[int] = 1000
DEFAULT_N_THERMALIZATION: Final[int] = 100
DEFAULT_MEASUREMENT_INTERVAL: Final[int] = 10

# Adaptive measurement defaults
DEFAULT_ADAPTIVE_MIN_THERMALIZATION: Final[int] = 200
DEFAULT_ADAPTIVE_MAX_THERMALIZATION: Final[int] = 10_000
DEFAULT_ADAPTIVE_C_WINDOW: Final[float] = 6.0
DEFAULT_ADAPTIVE_MIN_INDEPENDENT_SAMPLES: Final[int] = 100
DEFAULT_ADAPTIVE_MAX_TOTAL_SWEEPS: Final[int] = 100_000
DEFAULT_ADAPTIVE_TAU_MULTIPLIER: Final[float] = 2.0
# Floor on the fixed-temperature diagnostic block the adaptive scheme
# analyzes for stationarity and tau_int: MSER's boundary verdict needs
# enough points to be meaningful. A floor, not a target (B9, #20).
MIN_DIAGNOSTIC_SWEEPS: Final[int] = 64
