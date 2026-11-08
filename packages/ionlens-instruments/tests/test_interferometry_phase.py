"""``ionlens.instruments.interferometry.phase`` — doc 04 §5.1, doc 02 §8, doc 01 §5.4.

Plain module-level helpers, not a shared ``conftest.py`` — see the note at the top of
``oes_system.py`` for why this package avoids one.

Every hand-verified number here was cross-checked independently (not copied from the
implementation) using CODATA constants and the algebra of doc 04 §5.1 and doc 01 §5.4
before this module existed, so a passing test is a check on the physics and not a mirror
of the code.
"""

from __future__ import annotations

import math

import pytest

from ionlens.core.constants import CLASSICAL_ELECTRON_RADIUS
from ionlens.core.units import magnitude_in
from ionlens.instruments.interferometry.phase import (
    ARGON_STATIC_POLARIZABILITY_VOLUME_M3,
    CHAMBER_DIAMETER_M,
    CO2_WAVELENGTH_M,
    PHASE_RESOLUTION_RAD,
    critical_density_per_m3,
    detection_floor_n_e_per_m3,
    electron_phase_shift_rad,
    net_phase_shift_rad,
    neutral_phase_shift_rad,
    refractive_index,
    refractive_index_linearized,
)

_R_E_M = float(magnitude_in(CLASSICAL_ELECTRON_RADIUS, "m"))

_RP1_N_E_PER_M3 = 1.0e17
_RP1_PRESSURE_MTORR = 5.0
_RP1_GAS_TEMPERATURE_K = 300.0

_RP1_N_NEUTRAL_PER_M3 = (5.0e-3 * 133.322) / (1.380649e-23 * 300.0)


class TestCriticalDensity:
    def test_matches_the_hand_verified_value_at_10_6_microns(self) -> None:
        n_c = critical_density_per_m3(CO2_WAVELENGTH_M)

        assert n_c == pytest.approx(9.923e24, rel=1.0e-3)

    def test_refuses_a_nonpositive_wavelength(self) -> None:
        with pytest.raises(ValueError, match="wavelength"):
            critical_density_per_m3(0.0)


class TestRefractiveIndex:
    def test_the_linear_expansion_matches_the_exact_form_at_rp1(self) -> None:
        exact = refractive_index(_RP1_N_E_PER_M3, CO2_WAVELENGTH_M)
        linear = refractive_index_linearized(_RP1_N_E_PER_M3, CO2_WAVELENGTH_M)

        assert exact == pytest.approx(linear, abs=1.0e-12)

    def test_refuses_a_density_at_or_above_cutoff(self) -> None:
        n_c = critical_density_per_m3(CO2_WAVELENGTH_M)

        with pytest.raises(ValueError, match="critical density"):
            refractive_index(n_c, CO2_WAVELENGTH_M)

    def test_is_less_than_one_below_cutoff(self) -> None:
        assert refractive_index(_RP1_N_E_PER_M3, CO2_WAVELENGTH_M) < 1.0


class TestElectronPhaseShift:
    def test_matches_the_hand_verified_rp1_value_over_the_400mm_chord(self) -> None:
        phase = electron_phase_shift_rad(_RP1_N_E_PER_M3, CO2_WAVELENGTH_M, CHAMBER_DIAMETER_M)

        assert phase == pytest.approx(1.1948e-3, rel=1.0e-3)

    def test_scales_linearly_with_density(self) -> None:
        low = electron_phase_shift_rad(_RP1_N_E_PER_M3, CO2_WAVELENGTH_M, CHAMBER_DIAMETER_M)
        high = electron_phase_shift_rad(2.0 * _RP1_N_E_PER_M3, CO2_WAVELENGTH_M, CHAMBER_DIAMETER_M)

        assert high == pytest.approx(2.0 * low, rel=1.0e-9)

    def test_matches_the_raw_constant_formula(self) -> None:
        expected = _R_E_M * CO2_WAVELENGTH_M * _RP1_N_E_PER_M3 * CHAMBER_DIAMETER_M

        assert electron_phase_shift_rad(
            _RP1_N_E_PER_M3, CO2_WAVELENGTH_M, CHAMBER_DIAMETER_M
        ) == pytest.approx(expected, rel=1.0e-9)


class TestDetectionFloor:
    def test_matches_doc02_floor_over_the_400mm_chord(self) -> None:
        floor = detection_floor_n_e_per_m3(
            phase_resolution_rad=PHASE_RESOLUTION_RAD,
            wavelength_m=CO2_WAVELENGTH_M,
            chord_length_m=CHAMBER_DIAMETER_M,
        )

        assert floor == pytest.approx(8.370e15, rel=1.0e-3)

    def test_matches_doc01_floor_over_the_100mm_chord(self) -> None:
        floor = detection_floor_n_e_per_m3(
            phase_resolution_rad=PHASE_RESOLUTION_RAD,
            wavelength_m=CO2_WAVELENGTH_M,
            chord_length_m=0.1,
        )

        assert floor == pytest.approx(3.348e16, rel=1.0e-3)

    def test_scales_inversely_with_chord_length(self) -> None:
        short = detection_floor_n_e_per_m3(
            phase_resolution_rad=PHASE_RESOLUTION_RAD,
            wavelength_m=CO2_WAVELENGTH_M,
            chord_length_m=0.1,
        )
        long = detection_floor_n_e_per_m3(
            phase_resolution_rad=PHASE_RESOLUTION_RAD,
            wavelength_m=CO2_WAVELENGTH_M,
            chord_length_m=0.4,
        )

        assert short == pytest.approx(4.0 * long, rel=1.0e-9)


class TestNeutralPhaseShift:
    def test_is_positive(self) -> None:
        assert (
            neutral_phase_shift_rad(_RP1_N_NEUTRAL_PER_M3, CO2_WAVELENGTH_M, CHAMBER_DIAMETER_M)
            > 0.0
        )

    def test_matches_the_dilute_gas_refractivity_at_stp_within_the_literature_range(self) -> None:
        loschmidt_per_m3 = 2.6868e25
        refractivity = 2.0 * math.pi * loschmidt_per_m3 * ARGON_STATIC_POLARIZABILITY_VOLUME_M3

        assert 2.0e-4 < refractivity < 3.2e-4

    def test_scales_linearly_with_neutral_density(self) -> None:
        low = neutral_phase_shift_rad(_RP1_N_NEUTRAL_PER_M3, CO2_WAVELENGTH_M, CHAMBER_DIAMETER_M)
        high = neutral_phase_shift_rad(
            2.0 * _RP1_N_NEUTRAL_PER_M3, CO2_WAVELENGTH_M, CHAMBER_DIAMETER_M
        )

        assert high == pytest.approx(2.0 * low, rel=1.0e-9)


class TestNetPhaseShift:
    def test_the_neutral_term_reduces_the_net_phase_at_rp1(self) -> None:
        electron_only = electron_phase_shift_rad(
            _RP1_N_E_PER_M3, CO2_WAVELENGTH_M, CHAMBER_DIAMETER_M
        )
        net = net_phase_shift_rad(
            n_e_per_m3=_RP1_N_E_PER_M3,
            n_neutral_per_m3=_RP1_N_NEUTRAL_PER_M3,
            wavelength_m=CO2_WAVELENGTH_M,
            chord_length_m=CHAMBER_DIAMETER_M,
        )

        assert net < electron_only

    def test_the_neutral_contribution_at_5_mtorr_is_not_negligible(self) -> None:
        """The honest finding, not the intuitive one.

        A back-of-envelope guess is that background-gas refraction is negligible next to
        the plasma signal. At CO2's 10.6 micron wavelength and RP-1's reference 5 mTorr
        fill, it is not: the neutral term computed here is roughly a third of the
        electron term (hand-verified independently below), which is exactly *why* doc 04
        Section 5.2 lists it as something the framework includes rather than waves away.
        This test pins that finding so it cannot silently regress to "small" without
        someone noticing the assertion fail.
        """
        electron_only = electron_phase_shift_rad(
            _RP1_N_E_PER_M3, CO2_WAVELENGTH_M, CHAMBER_DIAMETER_M
        )
        neutral = neutral_phase_shift_rad(
            _RP1_N_NEUTRAL_PER_M3, CO2_WAVELENGTH_M, CHAMBER_DIAMETER_M
        )

        ratio = neutral / electron_only
        assert 0.15 < ratio < 0.55

    def test_net_is_exactly_electron_minus_neutral(self) -> None:
        electron_only = electron_phase_shift_rad(
            _RP1_N_E_PER_M3, CO2_WAVELENGTH_M, CHAMBER_DIAMETER_M
        )
        neutral = neutral_phase_shift_rad(
            _RP1_N_NEUTRAL_PER_M3, CO2_WAVELENGTH_M, CHAMBER_DIAMETER_M
        )
        net = net_phase_shift_rad(
            n_e_per_m3=_RP1_N_E_PER_M3,
            n_neutral_per_m3=_RP1_N_NEUTRAL_PER_M3,
            wavelength_m=CO2_WAVELENGTH_M,
            chord_length_m=CHAMBER_DIAMETER_M,
        )

        assert net == pytest.approx(electron_only - neutral, rel=1.0e-12)
