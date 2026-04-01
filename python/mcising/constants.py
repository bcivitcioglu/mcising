"""Centralized physics constants and simulation defaults."""

import math
from typing import Final

# Critical temperatures for known lattice types (J1=1, J2=0, h=0)
TC_SQUARE_2D: Final[float] = 2.0 / math.log(1.0 + math.sqrt(2.0))  # ~2.269
TC_TRIANGULAR_2D: Final[float] = 4.0 / math.log(3.0)  # ~3.641
TC_HONEYCOMB_2D: Final[float] = 2.0 / math.log(2.0 + math.sqrt(3.0))  # ~1.519
TC_CUBIC_3D: Final[float] = 4.5115  # High-precision MC estimate

# High temperature used for cool-down initialization
INF_TEMP: Final[float] = 100.0

# Correlation function threshold for truncation
CORRELATION_THRESHOLD: Final[float] = 1e-8

# Default simulation parameters
DEFAULT_LATTICE_SIZE: Final[int] = 10
DEFAULT_J1: Final[float] = 1.0
DEFAULT_J2: Final[float] = 0.0
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
