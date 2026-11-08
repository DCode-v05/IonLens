"""Fundamental constants — doc 09 §2.5.

    CODATA 2022, via scipy.constants. Never hand-typed. A mistyped electron mass is a
    plausible, undetectable and catastrophic error.

These tests exist to keep that promise checkable. They compare against
:mod:`scipy.constants` rather than against literals, because a test containing the
hand-typed value it is guarding would guard nothing.
"""

from __future__ import annotations

import pytest
import scipy.constants as sc

from ionlens.core import constants as k
from ionlens.core.units import DimensionalityError, magnitude_in


class TestValuesComeFromCodata:
    @pytest.mark.parametrize(
        ("constant", "scipy_value", "units"),
        [
            ("ELEMENTARY_CHARGE", sc.elementary_charge, "C"),
            ("BOLTZMANN", sc.Boltzmann, "J / K"),
            ("ELECTRON_MASS", sc.electron_mass, "kg"),
            ("ATOMIC_MASS", sc.atomic_mass, "kg"),
            ("VACUUM_PERMITTIVITY", sc.epsilon_0, "F / m"),
            ("SPEED_OF_LIGHT", sc.speed_of_light, "m / s"),
            ("PLANCK", sc.Planck, "J * s"),
        ],
    )
    def test_matches_scipy_exactly(self, constant: str, scipy_value: float, units: str) -> None:
        assert magnitude_in(getattr(k, constant), units) == scipy_value

    def test_classical_electron_radius_matches_scipy(self) -> None:
        expected = sc.physical_constants["classical electron radius"][0]

        assert magnitude_in(k.CLASSICAL_ELECTRON_RADIUS, "m") == expected

    def test_the_documented_reference_value_is_reproduced(self) -> None:
        assert magnitude_in(k.CLASSICAL_ELECTRON_RADIUS, "m") == pytest.approx(2.818e-15, rel=1e-3)


class TestConstantsAreDimensional:
    def test_every_constant_carries_units(self) -> None:
        for name in k.__all__:
            if name == "CODATA_SOURCE":
                continue
            value = getattr(k, name)
            assert hasattr(value, "units"), f"{name} is not a dimensional quantity"

    def test_a_constant_refuses_the_wrong_dimensionality(self) -> None:
        with pytest.raises(DimensionalityError):
            magnitude_in(k.BOLTZMANN, "m")


class TestProvenance:
    def test_records_which_codata_release_is_in_use(self) -> None:
        assert "CODATA" in k.CODATA_SOURCE
        assert sc.__name__.split(".")[0] in k.CODATA_SOURCE.lower() or "SciPy" in k.CODATA_SOURCE

    def test_the_release_in_use_is_the_one_the_specification_names(self) -> None:
        assert "2022" in k.CODATA_SOURCE
