"""The sealed-truth barrier and tier labelling — doc 05 §7, doc 07 §3.

doc 05 §7 opens with the sentence this module exists to enforce:

    An inverse crime is committed when the same model and discretisation generate the data
    and perform the inversion. The result is an artificially perfect recovery that proves
    nothing. This is the most common way computational inverse-problem work is invalidated,
    and it is guarded against **structurally rather than by good intentions**.

"Structurally rather than by good intentions" is a design instruction. An inverse crime is
not usually committed deliberately; it happens because the truth is *in scope* — sitting in
a variable, one attribute access away from the code computing the estimate. Nothing stops a
plotting routine, a convergence check or an initial guess from reaching for it, and once one
of them does, the recovery is perfect and the result is worthless.

So the truth is sealed behind an object that will not hand it over until the estimate has
been committed, and every result carries the tier it was produced at:

- **T0** — same model, no noise. Recovery to numerical tolerance; failing it is a bug, and
  nothing else means anything until it passes.
- **T1** — same model, with noise. Optimistic: the upper bound on achievable accuracy.
- **T2** — mismatched models, noise, imperfect calibration. The number quoted publicly.

doc 05 §7.2: "**Reporting T1 as if it were T2 is treated as a project defect**, and the CI
enforces that any figure showing accuracy carries its tier label." A label that can be
omitted is not enforcement, so it is a required constructor argument rather than a default.
"""

from __future__ import annotations

import numpy as np
import pytest

from ionlens.validation.sealed import (
    InverseCrimeError,
    SealedTruth,
    Tier,
    TierMismatchError,
    tier_of_configuration,
)


class TestTheSeal:
    def test_the_truth_cannot_be_read_before_an_estimate_is_committed(self) -> None:
        sealed = SealedTruth(value=6577.0, name="Gamma_E")

        with pytest.raises(InverseCrimeError, match="before an estimate has been committed"):
            _ = sealed.value

    def test_the_truth_is_readable_once_an_estimate_is_committed(self) -> None:
        sealed = SealedTruth(value=6577.0, name="Gamma_E")

        sealed.commit_estimate(6510.0, tier=Tier.T0)

        assert sealed.value == 6577.0

    def test_an_estimate_cannot_be_changed_once_committed(self) -> None:
        sealed = SealedTruth(value=6577.0, name="Gamma_E")
        sealed.commit_estimate(6510.0, tier=Tier.T0)

        with pytest.raises(InverseCrimeError, match="already committed"):
            sealed.commit_estimate(6577.0, tier=Tier.T0)

    def test_the_error_names_the_quantity_so_the_message_is_actionable(self) -> None:
        sealed = SealedTruth(value=1.0, name="Gamma_E")

        with pytest.raises(InverseCrimeError, match="Gamma_E"):
            _ = sealed.value

    def test_repr_does_not_leak_the_value_before_commitment(self) -> None:
        sealed = SealedTruth(value=6577.0, name="Gamma_E")

        assert "6577" not in repr(sealed)
        assert "sealed" in repr(sealed).lower()

    def test_repr_reveals_the_value_after_commitment(self) -> None:
        sealed = SealedTruth(value=6577.0, name="Gamma_E")
        sealed.commit_estimate(6510.0, tier=Tier.T0)

        assert "6577" in repr(sealed)

    def test_an_array_valued_truth_is_sealed_the_same_way(self) -> None:
        sealed = SealedTruth(value=np.array([1.0, 2.0, 3.0]), name="IEDF")

        with pytest.raises(InverseCrimeError):
            _ = sealed.value

        sealed.commit_estimate(np.array([1.1, 2.1, 2.9]), tier=Tier.T2)
        np.testing.assert_array_equal(sealed.value, [1.0, 2.0, 3.0])


class TestTheError:
    def test_the_relative_error_is_computed_only_after_commitment(self) -> None:
        sealed = SealedTruth(value=6577.0, name="Gamma_E")

        with pytest.raises(InverseCrimeError):
            _ = sealed.relative_error

        sealed.commit_estimate(6510.0, tier=Tier.T0)

        assert sealed.relative_error == pytest.approx(abs(6510.0 / 6577.0 - 1.0))

    def test_a_zero_truth_reports_absolute_error_rather_than_dividing(self) -> None:
        sealed = SealedTruth(value=0.0, name="drift")
        sealed.commit_estimate(0.01, tier=Tier.T0)

        assert sealed.relative_error == pytest.approx(0.01)


class TestTierLabelling:
    def test_a_result_must_carry_a_tier(self) -> None:
        sealed = SealedTruth(value=1.0, name="x")

        with pytest.raises(TypeError):
            sealed.commit_estimate(1.0)  # type: ignore[call-arg]

    def test_the_committed_tier_is_reported(self) -> None:
        sealed = SealedTruth(value=1.0, name="x")
        sealed.commit_estimate(1.0, tier=Tier.T2)

        assert sealed.tier is Tier.T2

    def test_tiers_are_ordered_by_honesty(self) -> None:
        assert Tier.T0 < Tier.T1 < Tier.T2

    def test_the_configuration_maps_to_the_right_tier(self) -> None:
        assert (
            tier_of_configuration(same_model=True, noise=False, imperfect_calibration=False)
            is Tier.T0
        )
        assert (
            tier_of_configuration(same_model=True, noise=True, imperfect_calibration=False)
            is Tier.T1
        )
        assert (
            tier_of_configuration(same_model=False, noise=True, imperfect_calibration=True)
            is Tier.T2
        )

    def test_a_mismatched_model_without_noise_is_not_claimable_as_t2(self) -> None:
        with pytest.raises(TierMismatchError, match="noise"):
            tier_of_configuration(same_model=False, noise=False, imperfect_calibration=True)

    def test_claiming_t2_without_imperfect_calibration_is_refused(self) -> None:
        with pytest.raises(TierMismatchError, match="calibration"):
            tier_of_configuration(same_model=False, noise=True, imperfect_calibration=False)


class TestTheReportingRule:
    def test_a_t1_result_cannot_be_relabelled_as_t2(self) -> None:
        sealed = SealedTruth(value=1.0, name="x")
        sealed.commit_estimate(1.0, tier=Tier.T1)

        with pytest.raises(TierMismatchError, match="T1"):
            sealed.assert_at_least(Tier.T2)

    def test_a_t2_result_satisfies_a_t1_requirement(self) -> None:
        sealed = SealedTruth(value=1.0, name="x")
        sealed.commit_estimate(1.0, tier=Tier.T2)

        sealed.assert_at_least(Tier.T1)

    def test_t0_failure_is_reported_as_a_bug_not_a_result(self) -> None:
        sealed = SealedTruth(value=6577.0, name="Gamma_E")
        sealed.commit_estimate(4000.0, tier=Tier.T0)

        with pytest.raises(AssertionError, match="bug"):
            sealed.assert_t0_consistency(tolerance=1e-6)

    def test_t0_passes_when_recovery_is_to_numerical_tolerance(self) -> None:
        sealed = SealedTruth(value=6577.0, name="Gamma_E")
        sealed.commit_estimate(6577.0 * (1 + 1e-12), tier=Tier.T0)

        sealed.assert_t0_consistency(tolerance=1e-6)

    def test_asserting_t0_consistency_on_a_non_t0_result_is_refused(self) -> None:
        sealed = SealedTruth(value=1.0, name="x")
        sealed.commit_estimate(1.0, tier=Tier.T1)

        with pytest.raises(TierMismatchError, match="T0"):
            sealed.assert_t0_consistency(tolerance=1e-6)
