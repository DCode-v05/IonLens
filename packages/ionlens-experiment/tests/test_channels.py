"""What :mod:`ionlens.experiment.channels` must be true for — doc 05 §3.2, doc 11 §9 item 3.

The project's honest end-to-end error is 36.5 %, diagnosed as a single optical channel
sliding along the ``n_0``-``T_e`` degeneracy. Connecting the second, already-built channel
is the designed fix, and it is only a fix if the second channel sees something the first
cannot. ``TestWhatLifAddsThatOesCannotSee`` is therefore not a nice-to-have test in this
file: it is the test that decides whether the whole exercise is worth doing, and it is
written to fail loudly if the answer is no.

``TestTheTuningRangeBlocker`` records the answer this file actually found at the RP-1
operating point, so that it is a checked property of the code rather than a discovery
somebody makes again later.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ionlens.core.params import default_registry
from ionlens.core.state import PlasmaState
from ionlens.core.units import magnitude_in
from ionlens.experiment.channels import (
    CHANNEL_NAMES,
    INTERFEROMETRY_CHANNEL,
    LIF_CHANNEL,
    OES_CHANNEL,
    THOMSON_CHANNEL,
    ChannelSet,
    build_channels,
    reconstruct_ivdf,
)
from ionlens.experiment.closed_loop import (
    _argon_ion,
    _fixed_spatial_grid,
    _reference_theta,
    _to_plasma_params,
)
from ionlens.instruments.lif.instrument import _sheath_edge_index
from ionlens.inverse.fusion import JointLikelihood
from ionlens.inverse.parameters import ControlParameters
from ionlens.physics.analytic.sheath import AnalyticSheathSolver

_REACHABLE_BIAS_V = -100.0

_SEED = 7


def _solver() -> AnalyticSheathSolver:
    return AnalyticSheathSolver()


def _state(theta: ControlParameters) -> PlasmaState:
    registry = default_registry()
    species = _argon_ion(registry)
    solver = _solver()
    grid = _fixed_spatial_grid(
        solver, reference=_reference_theta(), species=species, registry=registry
    )
    params = _to_plasma_params(theta, species=species, registry=registry)
    return solver.solve(params, grid=grid)


def _reference_state() -> PlasmaState:
    return _state(_reference_theta())


def _operating_theta() -> ControlParameters:
    return _reference_theta().replace(V_w=_REACHABLE_BIAS_V)


@pytest.fixture
def channels() -> ChannelSet:
    return build_channels(reference_state=_reference_state(), seed=_SEED)


@pytest.fixture
def truth_state() -> PlasmaState:
    return _state(_operating_theta())


def _joint_and_truth(channels: ChannelSet, truth_state: PlasmaState) -> JointLikelihood:
    return channels.joint(channels.observe(truth_state))


def _only(joint: JointLikelihood, name: str) -> JointLikelihood:
    """The joint likelihood ablated down to one named channel.

    The set grew from two channels to four (Thomson and interferometry were connected for
    the reason :mod:`ionlens.experiment.channels`'s docstring gives: with honest calibration
    uncertainty, OES's brightness scale and the plasma density are the same unknown). Every
    "this channel alone" assertion below used to be spelled ``joint.without(the_other_one)``,
    which silently became "three channels" when the set grew. Routing them through here
    keeps each of those tests asking the question it was written to ask rather than a
    weaker one that happens to still pass.
    """
    ablated = joint
    for other in joint.names:
        if other != name:
            ablated = ablated.without(other)
    return ablated


class TestTheChannelSet:
    def test_the_joint_likelihood_contains_every_channel_by_name(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        joint = channels.joint(channels.observe(truth_state))

        assert joint.names == CHANNEL_NAMES
        assert set(joint.names) == {
            OES_CHANNEL,
            LIF_CHANNEL,
            THOMSON_CHANNEL,
            INTERFEROMETRY_CHANNEL,
        }

    def test_the_channel_names_match_what_the_instruments_stamp_on_their_output(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        observations = channels.observe(truth_state)

        for name in CHANNEL_NAMES:
            assert observations[name].instrument_id == name

    def test_each_channel_alone_produces_a_finite_log_likelihood(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        joint = _joint_and_truth(channels, truth_state)

        for name in CHANNEL_NAMES:
            alone = _only(joint, name)

            detail = alone.detail(truth_state)

            assert detail.contributing == (name,)
            assert math.isfinite(detail.log_prob)

    def test_ablation_by_name_leaves_a_likelihood_that_still_scores(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        joint = _joint_and_truth(channels, truth_state)

        without_lif = joint.without(LIF_CHANNEL)

        assert without_lif.names == tuple(n for n in CHANNEL_NAMES if n != LIF_CHANNEL)
        assert math.isfinite(without_lif.log_prob(truth_state))
        assert joint.names == CHANNEL_NAMES, "without() must not mutate"

    def test_the_two_channels_integrate_for_genuinely_different_times(
        self, channels: ChannelSet
    ) -> None:
        oes_s = float(magnitude_in(channels.oes_window.duration, "s"))
        lif_s = float(magnitude_in(channels.lif_window.duration, "s"))

        assert lif_s > 100.0 * oes_s
        assert float(magnitude_in(channels.oes_window.start, "s")) == 0.0
        assert float(magnitude_in(channels.lif_window.start, "s")) == 0.0

    def test_a_missing_observation_is_refused_rather_than_silently_dropped(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        observations = channels.observe(truth_state)
        del observations[LIF_CHANNEL]

        with pytest.raises(KeyError, match="no observation supplied for lif"):
            channels.joint(observations)

    def test_the_four_channels_integrate_for_genuinely_different_times(
        self, channels: ChannelSet
    ) -> None:
        oes_s = float(magnitude_in(channels.oes_window.duration, "s"))
        thomson_s = float(magnitude_in(channels.thomson_window.duration, "s"))

        assert thomson_s > 1.0e9 * oes_s
        for window in (
            channels.oes_window,
            channels.lif_window,
            channels.thomson_window,
            channels.interferometry_window,
        ):
            assert float(magnitude_in(window.start, "s")) == 0.0


class TestTheReconstructedIvdf:
    """doc 03 §6's assumed drifting Maxwellian, checked against its own moments."""

    def test_it_carries_the_states_ion_density(self, truth_state: PlasmaState) -> None:
        reconstructed = reconstruct_ivdf(truth_state)

        assert reconstructed.ion_distribution is not None
        np.testing.assert_allclose(
            reconstructed.ion_distribution.density_per_m3(),
            truth_state.field("n_i").values,
            rtol=1e-6,
        )

    def test_the_drift_is_toward_the_wall(self, truth_state: PlasmaState) -> None:
        reconstructed = reconstruct_ivdf(truth_state)

        assert reconstructed.ion_distribution is not None
        np.testing.assert_allclose(
            reconstructed.ion_distribution.mean_velocity_m_per_s(),
            -truth_state.field("u_i").values,
            rtol=1e-6,
        )
        assert np.all(reconstructed.ion_distribution.particle_flux_toward_wall_per_m2_s() > 0.0)

    def test_a_state_that_already_resolves_f_i_is_returned_untouched(
        self, truth_state: PlasmaState
    ) -> None:
        once = reconstruct_ivdf(truth_state)

        assert reconstruct_ivdf(once) is once


class TestWhatLifAddsThatOesCannotSee:
    """The test that decides whether connecting LIF is worth doing at all.

    ``T_i`` is the sharpest available probe of the question. ``closed_loop``'s own module
    docstring states that ``T_i`` "has exactly zero effect on this configuration's
    prediction" because L0 runs at ``gamma_i = GAMMA_I_COLD_ION = 0``, so the OES channel
    is not merely weakly sensitive to it — it is bit-for-bit blind. LIF's line width is
    ``sqrt(T_i)`` by construction. If the LIF response is not large here, the second
    channel is carrying no information the first lacks and the whole fusion exercise is
    pointless; the assertions are written to say so.
    """

    def test_oes_is_bit_for_bit_blind_to_the_ion_temperature(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        theta = _operating_theta()
        joint = _only(_joint_and_truth(channels, truth_state), OES_CHANNEL)

        at_truth = joint.log_prob(_state(theta))
        at_four_times_t_i = joint.log_prob(_state(theta.replace(T_i=theta.T_i * 4.0)))

        assert at_four_times_t_i == at_truth

    def test_lif_responds_sharply_to_the_ion_temperature_oes_cannot_see(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        theta = _operating_theta()
        observations = channels.observe(truth_state)
        lif = _only(channels.joint(observations), LIF_CHANNEL)

        at_truth = lif.log_prob(_state(theta))
        hotter = lif.log_prob(_state(theta.replace(T_i=theta.T_i * 1.5)))
        colder = lif.log_prob(_state(theta.replace(T_i=theta.T_i / 1.5)))

        assert at_truth - hotter > 100.0
        assert at_truth - colder > 100.0

    def test_both_channels_constrain_the_degeneracy_direction(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        theta = _operating_theta()
        observations = channels.observe(truth_state)
        joint = channels.joint(observations)
        oes = _only(joint, OES_CHANNEL)
        lif = _only(joint, LIF_CHANNEL)

        k = 1.05
        along_ridge = _state(theta.replace(n_0=theta.n_0 * k, T_e=theta.T_e / k**2))
        oes_penalty = oes.log_prob(_state(theta)) - oes.log_prob(along_ridge)
        lif_penalty = lif.log_prob(_state(theta)) - lif.log_prob(along_ridge)

        assert oes_penalty > 100.0
        assert lif_penalty > 100.0


class TestTheTuningRange:
    """doc 01 §5.1 and doc 14 RS-03 at RP-1 — a constraint that existed and then lifted.

    **This class used to assert the opposite of what it asserts now, and the history is
    the point.** When this module was written, ``LifInstrument._expected_ion_speed``
    anchored its gate on the free-fall speed across the *whole* sheath drop: 34.9 km/s for
    argon through RP-1's -250 V, against the 25.8 km/s ceiling a 20 GHz mode-hop-free
    laser can reach. LIF therefore declared itself blind at RP-1, the fusion layer excluded
    it, and connecting the channel did nothing at the operating point the project actually
    cares about. That was recorded here as a blocker, and in ``channels.py``'s module
    docstring as needing "a refinement of that gate inside ``ionlens-instruments``".

    That refinement has since landed: the gate is now anchored at the sheath-edge Bohm
    speed, and ``_resolve_z_index`` defaults the measurement volume to the sheath-edge grid
    node, on doc 01 §5.1's own stated mitigation — measure at the edge and *propagate* to
    the wall, rather than pretend the wall IVDF is directly reachable. So LIF is now
    informative at RP-1 and these tests say so.

    They are kept, re-based rather than deleted, because a reader who finds only the
    passing assertion has no way to know the constraint was ever real or why
    ``_REACHABLE_BIAS_V`` exists at all.
    """

    def test_lif_is_now_informative_at_the_rp1_wall_bias(self, channels: ChannelSet) -> None:
        registry = default_registry()
        rp1 = _to_plasma_params(_reference_theta(), species=_argon_ion(registry), registry=registry)

        assert channels.lif.is_informative(rp1)

    def test_lif_is_reachable_at_the_reduced_bias_these_tests_run_at(
        self, channels: ChannelSet
    ) -> None:
        registry = default_registry()
        reachable = _to_plasma_params(
            _operating_theta(), species=_argon_ion(registry), registry=registry
        )

        assert channels.lif.is_informative(reachable)

    def test_at_rp1_every_channel_now_contributes(self, channels: ChannelSet) -> None:
        rp1_state = _reference_state()
        joint = channels.joint(channels.observe(rp1_state))

        detail = joint.detail(rp1_state)

        assert set(detail.contributing) == set(CHANNEL_NAMES)
        assert detail.excluded == ()
        assert math.isfinite(detail.log_prob)

    def test_the_measurement_volume_is_pinned_and_does_not_move_with_theta(
        self, channels: ChannelSet
    ) -> None:
        theta = _operating_theta()
        sparse = _state(theta)
        dense = _state(theta.replace(n_0=theta.n_0 * 2.0))

        tolerance = float(default_registry().value_in("sheath.edge_tolerance", "dimensionless"))
        assert _sheath_edge_index(sparse, tolerance=tolerance) != _sheath_edge_index(
            dense, tolerance=tolerance
        )

        assert channels.lif._resolve_z_index(sparse) == channels.lif._resolve_z_index(dense)


class TestThomsonIsExcludedRatherThanCrashingWhenTheSheathSwallowsItsVolume:
    """The invariant a MAP search actually depends on — see `_ThomsonOutsideTheSheath`.

    `JointLikelihood.detail` asks `is_informative` and, if the answer is yes, calls
    `forward` with no `except` anywhere in the path. So for every theta the optimiser can
    propose, **either the channel must report itself blind or `forward` must return finite
    values**. Anything else is not a bad fit, it is a crashed run.

    Thomson violated that before the gate existed, and not marginally: its own
    params-only floor admits `n_0` down to 2.42e15, while the pinned measurement volume
    sits at a *fixed* 2.28 mm and the sheath it has to be outside of grows as
    `sqrt(T_e / n_0)`. Measured, `forward` raised `ThomsonBlindWindowError` for 16 of 81
    points across the prior — continuously for `n_0` in roughly [2.6e15, 1.7e16] at
    `T_e = 3` eV. doc 01 IF-6 makes exclusion the right answer there, and
    `FusionDetail.excluded` is what keeps a dropped channel recorded rather than silent.
    """

    def test_no_theta_in_the_map_search_region_either_raises_or_goes_unreported(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        joint = _joint_and_truth(channels, truth_state)
        theta = _operating_theta()

        excluded_somewhere = False
        for n_0 in np.geomspace(1.0e15, 1.0e19, 9):
            for T_e in np.geomspace(1.0, 10.0, 5):
                state = _state(theta.replace(n_0=float(n_0), T_e=float(T_e)))

                detail = joint.detail(state)

                assert math.isfinite(detail.log_prob)
                if THOMSON_CHANNEL in detail.excluded:
                    excluded_somewhere = True

        assert excluded_somewhere, (
            "no theta in the swept region excluded Thomson, so this test is not exercising "
            "the gate it was written for"
        )

    def test_the_gate_is_what_is_doing_it_rather_than_luck(self, channels: ChannelSet) -> None:
        registry = default_registry()
        species = _argon_ion(registry)
        swallowed = _to_plasma_params(
            _reference_theta().replace(n_0=1.0e16), species=species, registry=registry
        )
        healthy = _to_plasma_params(_reference_theta(), species=species, registry=registry)

        assert not channels.thomson.inversion.is_informative(swallowed)
        assert channels.thomson.inversion.is_informative(healthy)
        assert channels.thomson.inversion.instrument.is_informative(swallowed)


class TestDiscrepancyReachesEachChannel:
    """``build_channels(discrepancy=...)`` — keyed by channel name, pinned onto each
    wrapper's constructor state rather than threaded through a call.

    ``ionlens.inverse.fusion.Channel`` calls ``instrument.likelihood(obs, pred)`` with exactly
    the two positional arguments its own ``_Instrument`` protocol declares, so a per-channel
    discrepancy has nowhere to travel except as state each of ``_OesWithCalibrationUncertainty``,
    ``_LifOnReconstructedIvdf``, ``_ThomsonOutsideTheSheath`` and interferometry's
    ``_InterferometryWithOptionalCoherence`` carries — exactly the same problem
    ``calibration_uncertainty`` already solves for each of them. This class is the
    end-to-end check that ``build_channels``'s mapping actually reaches every one of the
    four wrappers, that an unrecognised key is refused rather than silently doing nothing,
    and that ``discrepancy=None`` disturbs nothing.
    """

    def test_an_unrecognised_channel_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="oes_typo"):
            build_channels(
                reference_state=_reference_state(),
                seed=_SEED,
                discrepancy={"oes_typo": np.array([1.0])},
            )

    def test_discrepancy_none_is_bit_for_bit_identical_to_the_default(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        explicit_none = build_channels(
            reference_state=_reference_state(), seed=_SEED, discrepancy=None
        )
        observations = channels.observe(truth_state)

        default_log_prob = channels.joint(observations).log_prob(truth_state)
        explicit_log_prob = explicit_none.joint(observations).log_prob(truth_state)

        assert default_log_prob == explicit_log_prob

    @pytest.mark.parametrize("name", [OES_CHANNEL, LIF_CHANNEL, INTERFEROMETRY_CHANNEL])
    def test_a_discrepancy_on_one_channel_changes_only_that_channels_contribution(
        self, name: str, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        observations = channels.observe(truth_state)
        values = np.abs(np.asarray(observations[name].values))
        discrepancy_value = 0.02 * values + 1.0e-12

        with_discrepancy = build_channels(
            reference_state=_reference_state(),
            seed=_SEED,
            discrepancy={name: discrepancy_value},
        )

        without_term = _only(channels.joint(observations), name).log_prob(truth_state)
        with_term = _only(with_discrepancy.joint(observations), name).log_prob(truth_state)

        assert math.isfinite(with_term)
        assert with_term != without_term

        for other in CHANNEL_NAMES:
            if other == name:
                continue
            assert _only(channels.joint(observations), other).log_prob(truth_state) == _only(
                with_discrepancy.joint(observations), other
            ).log_prob(truth_state)

    def test_oes_discrepancy_widens_the_curvature_through_the_wrapper(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        """The same widens-never-tightens property the instrument-level tests check,
        reproduced through ``build_channels``'s wrapper to confirm the wiring does not
        introduce the mean-variance-coupling defect this project has already hit once."""
        observations = channels.observe(truth_state)
        oes_values = np.abs(np.asarray(observations[OES_CHANNEL].values))
        discrepancy_value = 0.05 * oes_values + 1.0e-12

        with_discrepancy = build_channels(
            reference_state=_reference_state(),
            seed=_SEED,
            discrepancy={OES_CHANNEL: discrepancy_value},
        )

        def curvature(built: ChannelSet) -> float:
            step = 1.0e-3

            def score(theta_n0_factor: float) -> float:
                scaled_state = _state(_operating_theta().replace(n_0=theta_n0_factor))
                return _only(built.joint(observations), OES_CHANNEL).log_prob(scaled_state)

            centre_n0 = float(_operating_theta().n_0)
            centre = score(centre_n0)
            up = score(centre_n0 * (1.0 + step))
            down = score(centre_n0 * (1.0 - step))
            return -(up - 2.0 * centre + down) / (centre_n0 * step) ** 2

        without = curvature(channels)
        with_term = curvature(with_discrepancy)

        assert without > 0.0
        assert with_term > 0.0
        assert with_term < without

    def test_a_thomson_discrepancy_is_refused_at_rp1s_photon_starved_channels(
        self, channels: ChannelSet, truth_state: PlasmaState
    ) -> None:
        """The documented, measured limitation: doc 02 §7.1's 0.008 pe/channel/shot means
        several Thomson channels sit far below the Gaussian normal-approximation floor
        ``ThomsonInstrument.likelihood`` requires before it will accept a coherent
        discrepancy — see that method's own docstring. This is not a bug in the wiring;
        it is the wiring correctly refusing to approximate past a floor doc 02 §7.1's own
        numbers put out of reach here."""
        observations = channels.observe(truth_state)
        thomson_values = np.abs(np.asarray(observations[THOMSON_CHANNEL].values))
        discrepancy_value = 0.02 * thomson_values + 1.0e-12

        with_discrepancy = build_channels(
            reference_state=_reference_state(),
            seed=_SEED,
            discrepancy={THOMSON_CHANNEL: discrepancy_value},
        )

        with pytest.raises(ValueError, match="floor"):
            _only(with_discrepancy.joint(observations), THOMSON_CHANNEL).log_prob(truth_state)
