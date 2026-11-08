"""Fidelity, species and control parameters — doc 03 §1 and doc 05 §2.1."""

from __future__ import annotations

import pytest

from ionlens.core.state import Fidelity, PlasmaParams, Species
from ionlens.core.units import Q_, DimensionalityError, magnitude_in


class TestFidelity:
    def test_names_the_four_levels_of_the_hierarchy(self) -> None:
        assert {f.name for f in Fidelity} == {"L0", "L1", "L2", "L3"}

    def test_each_level_carries_its_description(self) -> None:
        assert "analytic" in Fidelity.L0.description.lower()
        assert "fluid" in Fidelity.L1.description.lower()
        assert "kinetic" in Fidelity.L2.description.lower()
        assert "surrogate" in Fidelity.L3.description.lower()

    def test_levels_are_not_ordered(self) -> None:
        with pytest.raises(TypeError):
            _ = Fidelity.L0 < Fidelity.L2  # type: ignore[operator]

    def test_only_the_kinetic_level_produces_an_iedf_directly(self) -> None:
        assert Fidelity.L2.resolves_distribution is True
        assert Fidelity.L1.resolves_distribution is False
        assert Fidelity.L0.resolves_distribution is False

    def test_the_surrogate_records_what_it_emulates(self) -> None:
        assert Fidelity.L3.emulates is Fidelity.L2
        assert Fidelity.L2.emulates is None

    def test_round_trips_through_its_string_value(self) -> None:
        assert Fidelity("L2") is Fidelity.L2
        assert Fidelity.L2.value == "L2"


class TestSpecies:
    def test_carries_a_name_mass_and_charge(self) -> None:
        argon = Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1)

        assert argon.name == "Ar+"
        assert magnitude_in(argon.mass, "u") == pytest.approx(39.948)
        assert argon.charge_number == 1

    def test_exposes_mass_in_kilograms_for_hot_loops(self) -> None:
        argon = Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1)

        assert argon.mass_kg == pytest.approx(6.6335e-26, rel=1e-4)

    def test_rejects_a_non_mass_quantity(self) -> None:
        with pytest.raises(DimensionalityError):
            Species(name="Ar+", mass=Q_(39.948, "m"), charge_number=1)

    def test_rejects_a_non_positive_mass(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Species(name="Ar+", mass=Q_(0.0, "u"), charge_number=1)

    def test_rejects_an_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Species(name="", mass=Q_(39.948, "u"), charge_number=1)

    def test_is_immutable(self) -> None:
        argon = Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1)

        with pytest.raises(AttributeError):
            argon.name = "Xe+"  # type: ignore[misc]


def _rp1(**overrides: object) -> PlasmaParams:
    """The reference operating point RP-1 of doc 01 §2.1."""
    defaults: dict[str, object] = {
        "species": Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1),
        "n_0": Q_(1e17, "m**-3"),
        "T_e": Q_(3.0, "eV"),
        "T_i": Q_(0.05, "eV"),
        "T_g": Q_(300.0, "K"),
        "pressure": Q_(5.0, "mTorr"),
        "bias": Q_(-250.0, "V"),
        "gamma_se": 0.10,
        "kappa": 1.0,
    }
    return PlasmaParams(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestPlasmaParams:
    def test_holds_the_reference_operating_point(self) -> None:
        params = _rp1()

        assert magnitude_in(params.n_0, "m**-3") == pytest.approx(1e17)
        assert magnitude_in(params.T_e, "eV") == pytest.approx(3.0)
        assert magnitude_in(params.pressure, "mTorr") == pytest.approx(5.0)

    def test_bias_is_signed_and_negative_for_a_biased_electrode(self) -> None:
        params = _rp1()

        assert magnitude_in(params.bias, "V") == pytest.approx(-250.0)
        assert magnitude_in(params.bias_magnitude, "V") == pytest.approx(250.0)

    def test_bias_magnitude_is_positive_for_a_positive_bias_too(self) -> None:
        params = _rp1(bias=Q_(50.0, "V"))

        assert magnitude_in(params.bias_magnitude, "V") == pytest.approx(50.0)

    def test_rejects_a_non_positive_density(self) -> None:
        with pytest.raises(ValueError, match="n_0"):
            _rp1(n_0=Q_(0.0, "m**-3"))

    def test_rejects_a_non_positive_electron_temperature(self) -> None:
        with pytest.raises(ValueError, match="T_e"):
            _rp1(T_e=Q_(-1.0, "eV"))

    def test_rejects_a_non_positive_pressure(self) -> None:
        with pytest.raises(ValueError, match="pressure"):
            _rp1(pressure=Q_(0.0, "mTorr"))

    def test_rejects_a_negative_secondary_emission_yield(self) -> None:
        with pytest.raises(ValueError, match="gamma_se"):
            _rp1(gamma_se=-0.01)

    def test_accepts_a_kappa_outside_its_prior_range(self) -> None:
        assert _rp1(kappa=7.5).kappa == pytest.approx(7.5)

    def test_rejects_wrong_dimensions_on_every_quantity(self) -> None:
        with pytest.raises(DimensionalityError):
            _rp1(n_0=Q_(1e17, "m**-2"))
        with pytest.raises(DimensionalityError):
            _rp1(pressure=Q_(5.0, "V"))
        with pytest.raises(DimensionalityError):
            _rp1(bias=Q_(-250.0, "mTorr"))

    def test_accepts_electron_temperature_in_kelvin(self) -> None:
        params = _rp1(T_e=Q_(34813.6, "K"))

        assert magnitude_in(params.T_e_eV, "eV") == pytest.approx(3.0, rel=1e-4)

    def test_exposes_si_magnitudes_for_hot_loops(self) -> None:
        params = _rp1()

        assert params.n_0_per_m3 == pytest.approx(1e17)
        assert params.T_e_J == pytest.approx(3.0 * 1.602176634e-19, rel=1e-9)

    def test_neutral_density_follows_from_pressure_and_gas_temperature(self) -> None:
        params = _rp1()

        assert magnitude_in(params.n_g, "m**-3") == pytest.approx(1.61e20, rel=2e-3)

    def test_is_immutable(self) -> None:
        params = _rp1()

        with pytest.raises(AttributeError):
            params.n_0 = Q_(1e18, "m**-3")  # type: ignore[misc]

    def test_replace_returns_a_new_object(self) -> None:
        params = _rp1()

        hotter = params.replace(T_e=Q_(5.0, "eV"))

        assert magnitude_in(hotter.T_e, "eV") == pytest.approx(5.0)
        assert magnitude_in(params.T_e, "eV") == pytest.approx(3.0)
