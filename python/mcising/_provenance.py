"""Provenance facts stamped into every saved result.

Stdlib-only leaf module: both :mod:`mcising.simulation` and :mod:`mcising.io`
need these values, and ``io`` imports ``simulation``, so neither can host them
without a cycle.
"""

import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

# On-disk metadata layout of HDF5 files written by this code. Files without a
# schema_version attribute are schema 1 (mcising <= 0.23.0).
HDF5_SCHEMA_VERSION: Final[int] = 2


def package_version() -> str:
    """Return the installed mcising version.

    Reads the installed distribution metadata, the single source of truth
    for the package version (``pyproject.toml`` at build time).

    Returns
    -------
    str
        The installed version, or the PEP 440 sentinel ``"0.0.0+unknown"``
        when mcising is imported from a source tree without an install.
    """
    try:
        return version("mcising")
    except PackageNotFoundError:
        return "0.0.0+unknown"


@lru_cache(maxsize=1)
def git_commit() -> str | None:
    """Return the short git commit of the mcising source, best effort.

    Resolves against the package directory, not the caller's working
    directory, so a wheel install never picks up an unrelated repository.

    Returns
    -------
    str or None
        Short commit hash when the package lives in a git checkout,
        otherwise None.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.decode("ascii", errors="replace").strip()
    return commit or None
