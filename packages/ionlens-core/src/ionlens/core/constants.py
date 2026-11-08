"""Fundamental physical constants — doc 09 §2.5.

    CODATA 2022, via ``scipy.constants``. Never hand-typed. A mistyped electron mass
    is a plausible, undetectable and catastrophic error.

Every value here is re-exported from :mod:`scipy.constants` and wrapped in the shared
unit registry. Nothing in this module contains a numeric literal with physical meaning,
which is what makes the doc 08 §5 literal-lint rule enforceable in the packages that
consume it: physics code asks for :data:`ELEMENTARY_CHARGE`, never for ``1.602e-19``.
"""

from __future__ import annotations

import scipy
import scipy.constants as sc

from ionlens.core.units import Q_, Quantity

__all__ = [
    "ATOMIC_MASS",
    "BOLTZMANN",
    "CLASSICAL_ELECTRON_RADIUS",
    "CODATA_SOURCE",
    "ELECTRON_MASS",
    "ELEMENTARY_CHARGE",
    "PLANCK",
    "SPEED_OF_LIGHT",
    "VACUUM_PERMITTIVITY",
]


def _codata_release() -> str:
    """Name the evaluated constant set the running SciPy actually exposes.

    SciPy keeps this in a private attribute. It is read defensively rather than
    hard-coded, because doc 09 §2.5 pins CODATA 2022 specifically and a provenance
    record that *asserts* the release without checking it is worth nothing. If a future
    SciPy moves the attribute, the record degrades to the SciPy version — still
    resolvable — instead of silently claiming a release it did not verify.
    """
    release = getattr(sc._codata, "_current_codata", None)
    return str(release) if release else f"unknown release (SciPy {scipy.__version__})"


CODATA_SOURCE: str = f"{_codata_release()} via scipy.constants (SciPy {scipy.__version__})"

ELEMENTARY_CHARGE: Quantity = Q_(sc.elementary_charge, "C")

BOLTZMANN: Quantity = Q_(sc.Boltzmann, "J / K")

ELECTRON_MASS: Quantity = Q_(sc.electron_mass, "kg")

ATOMIC_MASS: Quantity = Q_(sc.atomic_mass, "kg")

VACUUM_PERMITTIVITY: Quantity = Q_(sc.epsilon_0, "F / m")

SPEED_OF_LIGHT: Quantity = Q_(sc.speed_of_light, "m / s")

PLANCK: Quantity = Q_(sc.Planck, "J * s")

CLASSICAL_ELECTRON_RADIUS: Quantity = Q_(sc.physical_constants["classical electron radius"][0], "m")
