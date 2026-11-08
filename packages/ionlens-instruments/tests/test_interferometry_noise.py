"""``ionlens.instruments.interferometry.noise`` — doc 04 §5.2, doc 02 §8.1-8.2.

Plain module-level helpers, not a shared ``conftest.py`` — see the note at the top of
``oes_system.py``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import ClassVar

import numpy as np
import pytest

from ionlens.core.params import ParameterRegistry, default_registry
from ionlens.instruments.interferometry.noise import (
    FRINGE_JUMP_MAGNITUDE_RAD,
    independent_phase_std_rad,
    sample_fringe_jumps,
    vibration_phase_std_rad,
)
from ionlens.instruments.interferometry.phase import PHASE_RESOLUTION_RAD


def _registry_with(overrides: dict[str, float]) -> ParameterRegistry:
    """The shipped registry with one or more entries' ``value`` swapped out.

    Used to prove a value is actually *read* from the registry rather than copied into a
    local literal at import time (doc 00 C1): if changing the registry entry changes the
    computed result, the code path goes through the registry; if it does not, the
    registry entry is decorative.
    """
    entries = dict(default_registry().entries)
    for entry_id, value in overrides.items():
        entries[entry_id] = replace(entries[entry_id], value=value)
    return ParameterRegistry(entries)


class TestVibrationPhaseStd:
    def test_is_positive_for_a_representative_window(self) -> None:
        assert vibration_phase_std_rad(1.0e-3) > 0.0

    def test_is_below_the_full_phase_resolution_budget(self) -> None:
        assert vibration_phase_std_rad(1.0e-3) < PHASE_RESOLUTION_RAD

    def test_a_longer_window_sees_more_of_the_1_over_f_tail(self) -> None:
        short = vibration_phase_std_rad(1.0e-4)
        long = vibration_phase_std_rad(1.0e-1)

        assert long > short

    def test_returns_zero_when_the_window_is_too_short_to_resolve_any_vibration(self) -> None:
        assert vibration_phase_std_rad(1.0e-9) == 0.0

    def test_refuses_a_nonpositive_duration(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            vibration_phase_std_rad(0.0)


class TestIndependentPhaseStd:
    def test_is_positive(self) -> None:
        assert independent_phase_std_rad() > 0.0

    def test_combines_with_the_vibration_term_to_the_calibrated_phase_resolution(self) -> None:
        reference_window_s = 1.0e-3
        total_variance = (
            vibration_phase_std_rad(reference_window_s) ** 2 + independent_phase_std_rad() ** 2
        )

        assert math.sqrt(total_variance) == pytest.approx(PHASE_RESOLUTION_RAD, rel=1.0e-9)


class TestFringeJumps:
    def test_a_certain_rate_produces_a_jump_of_exactly_two_pi_on_every_chord(self) -> None:
        rng = np.random.default_rng(0)

        jumps = sample_fringe_jumps(rng, n_chords=8, rate=1.0)

        assert jumps.shape == (8,)
        np.testing.assert_allclose(jumps, FRINGE_JUMP_MAGNITUDE_RAD)
        assert pytest.approx(2.0 * math.pi) == FRINGE_JUMP_MAGNITUDE_RAD

    def test_a_zero_rate_produces_no_jumps(self) -> None:
        rng = np.random.default_rng(0)

        jumps = sample_fringe_jumps(rng, n_chords=8, rate=0.0)

        np.testing.assert_array_equal(jumps, np.zeros(8))

    def test_refuses_a_rate_outside_zero_one(self) -> None:
        rng = np.random.default_rng(0)

        with pytest.raises(ValueError, match="probability"):
            sample_fringe_jumps(rng, n_chords=8, rate=1.5)
        with pytest.raises(ValueError, match="probability"):
            sample_fringe_jumps(rng, n_chords=8, rate=-0.1)

    def test_is_deterministic_for_a_seeded_generator(self) -> None:
        first = sample_fringe_jumps(np.random.default_rng(42), n_chords=8, rate=0.5)
        second = sample_fringe_jumps(np.random.default_rng(42), n_chords=8, rate=0.5)

        np.testing.assert_array_equal(first, second)


class TestVibrationModelParametersAreRegistered:
    """doc 00 C1: the vibration model's shape and split were invented numbers with no
    citable source (see the module docstring) that never reached the registry. The six
    ``IF.vibration_*`` entries fix that — registered ASSUMED, and read from the registry
    at call time rather than held as bare module literals.
    """

    _IDS_AND_DEFAULTS: ClassVar[dict[str, float]] = {
        "IF.vibration_common_mode_fraction": 0.9,
        "IF.vibration_resonance_frequency_hz": 50.0,
        "IF.vibration_resonance_width_hz": 5.0,
        "IF.vibration_one_over_f_share": 0.5,
        "IF.vibration_upper_bandwidth_hz": 1.0e4,
        "IF.vibration_reference_window_s": 1.0e-3,
    }

    @pytest.mark.parametrize(("entry_id", "expected"), sorted(_IDS_AND_DEFAULTS.items()))
    def test_each_entry_resolves_and_is_classed_assumed(
        self, entry_id: str, expected: float
    ) -> None:
        entry = default_registry()[entry_id]
        assert entry.provenance_class.value == "ASSUMED"
        assert entry.value == pytest.approx(expected)

    def test_the_common_mode_fraction_is_read_from_the_registry_not_hardcoded(self) -> None:
        no_vibration_registry = _registry_with({"IF.vibration_common_mode_fraction": 0.0})

        vibration = vibration_phase_std_rad(1.0e-3, registry=no_vibration_registry)
        independent = independent_phase_std_rad(registry=no_vibration_registry)

        assert vibration == pytest.approx(0.0)
        assert independent == pytest.approx(PHASE_RESOLUTION_RAD)

    def test_the_resonance_frequency_is_read_from_the_registry_not_hardcoded(self) -> None:
        default_result = vibration_phase_std_rad(1.0e-2)
        moved_registry = _registry_with({"IF.vibration_resonance_frequency_hz": 500.0})

        moved_result = vibration_phase_std_rad(1.0e-2, registry=moved_registry)

        assert moved_result != pytest.approx(default_result)

    def test_the_reference_window_is_read_from_the_registry_not_hardcoded(self) -> None:
        moved_registry = _registry_with({"IF.vibration_reference_window_s": 5.0e-3})

        vibration = vibration_phase_std_rad(5.0e-3, registry=moved_registry)
        independent = independent_phase_std_rad(registry=moved_registry)

        total_variance = vibration**2 + independent**2
        assert math.sqrt(total_variance) == pytest.approx(PHASE_RESOLUTION_RAD, rel=1.0e-9)
