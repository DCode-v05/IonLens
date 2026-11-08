"""Monte-Carlo collisions — doc 03 §4.2 step 5, doc 03 §4.5.

A Monte-Carlo collision module cannot be verified the way the push was. There is no single
trajectory to compare against a closed form; what the module produces is a *distribution*,
and a distribution can be wrong in ways that every individual sample looks right.

ADR-011 is the reason this file is written the way it is. The EEDF solver passed 1 488
internal tests and was 52 % wrong on the Reid benchmark, because every one of those tests
compared the solver against itself. So the assertions below are against closed forms the
sampler cannot influence:

1. **The rate.** ``nu = n_gas sigma(v) v`` is the definition of a collision frequency. The
   sampler's measured rate is compared to it with a tolerance *derived* from the binomial
   variance of the estimator plus the known ``O(nu_max dt)`` discretisation bias — not
   eyeballed, and not taken from a previous run of this code.
2. **The free-path distribution.** Constant cross section, no field: the collision-free
   path length is exponential with mean ``1/(n sigma)``. Tested by Kolmogorov-Smirnov
   against the analytic CDF, not by a moment.
3. **Null collision == direct rejection.** The equivalence is the entire justification for
   the null-collision method, so it is asserted rather than assumed: an independent direct
   rejection sampler is written here, in the test, and the two are compared channel by
   channel with a two-proportion z test.
4. **The charge-exchange IEDF.** For a uniform field and a constant CX cross section the
   ion energy distribution at the wall is exponential with mean ``e E lambda``. That is a
   textbook result (Davis & Vanderslice 1963 in its collisional limit) which no part of
   this implementation can bend, and it is the observable doc 03 §4.5 says the whole
   project turns on.
5. **Conservation.** CX is an identity swap between the ion and the neutral it hits, so the
   pair's momentum and energy are unchanged *exactly*. Ion elastic scattering conserves
   both in the centre-of-mass frame, which is asserted through the relative speed rather
   than through a total that could be conserved by two cancelling errors.

Point 5 deserves a note. A test that only asked "is the post-CX ion cold?" would pass an
implementation that gave the ion a freshly sampled thermal velocity uncorrelated with the
neutral it actually struck. That implementation produces a correct-looking IEDF and
violates momentum conservation in every single event. The swap is therefore asserted
against the recorded target velocity, not against a temperature.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from ionlens.core.constants import BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE
from ionlens.core.params import default_registry
from ionlens.core.random import Stream, generator
from ionlens.core.units import magnitude_in
from ionlens.physics.atomic.lxcat import CrossSection, ProcessType
from ionlens.physics.kinetic.collisions import (
    BackgroundGas,
    CollisionKind,
    EnergyCeilingExceededError,
    NullCollisionSampler,
    Process,
    collision_generator,
    neutral_density_m3,
)
from ionlens.physics.kinetic.push import Species, push_leapfrog

_E = float(magnitude_in(ELEMENTARY_CHARGE, "C"))
_M_E = float(magnitude_in(ELECTRON_MASS, "kg"))
_K_B = float(magnitude_in(BOLTZMANN, "J/K"))
_M_AR = float(default_registry().value_in("species.Ar.mass", "kg"))

_SEED = 20260805

_SIGMA = 4.0

_KS_CRITICAL = 1.95


def _section(
    process: ProcessType,
    *,
    sigma_m2: float,
    max_energy_ev: float,
    threshold_ev: float | None = None,
    reactants: tuple[str, ...] = ("Ar^+", "Ar"),
    products: tuple[str, ...] = ("Ar", "Ar^+"),
) -> CrossSection:
    """A constant cross section, as a :class:`CrossSection` the production path accepts."""
    lowest = 0.0 if threshold_ev is None else threshold_ev
    return CrossSection(
        process=process,
        database="synthetic",
        projectile=reactants[0],
        target=reactants[-1],
        reactants=reactants,
        products=products,
        threshold_ev=threshold_ev,
        mass_ratio=None,
        energy_ev=np.array([lowest, max_energy_ev], dtype=np.float64),
        sigma_m2=np.array([sigma_m2, sigma_m2], dtype=np.float64),
        parameters={},
    )


def _ramp_section(*, sigma_low: float, sigma_high: float, max_energy_ev: float) -> CrossSection:
    """A linearly varying cross section, for the ``nu_max`` maximisation test."""
    return CrossSection(
        process=ProcessType.CHARGE_EXCHANGE,
        database="synthetic",
        projectile="Ar^+",
        target="Ar",
        reactants=("Ar^+", "Ar"),
        products=("Ar", "Ar^+"),
        threshold_ev=None,
        mass_ratio=None,
        energy_ev=np.array([0.0, max_energy_ev], dtype=np.float64),
        sigma_m2=np.array([sigma_low, sigma_high], dtype=np.float64),
        parameters={},
    )


def _argon_ions() -> Species:
    return Species(name="Ar+", charge_c=_E, mass_kg=_M_AR, weight=1.0)


def _electrons() -> Species:
    return Species(name="e", charge_c=-_E, mass_kg=_M_E, weight=1.0)


def _gas(*, density_m3: float = 1e21, temperature_k: float = 300.0) -> BackgroundGas:
    return BackgroundGas(mass_kg=_M_AR, density_m3=density_m3, temperature_k=temperature_k)


def _beam(n: int, *, energy_ev: float, mass_kg: float) -> NDArray[np.float64]:
    """``n`` particles, all at ``energy_ev``, all travelling along ``+z``."""
    speed = math.sqrt(2.0 * energy_ev * _E / mass_kg)
    velocities = np.zeros((n, 3), dtype=np.float64)
    velocities[:, 2] = speed
    return velocities


def _speed_of(energy_ev: float, mass_kg: float) -> float:
    return math.sqrt(2.0 * energy_ev * _E / mass_kg)


def _kinetic_energy_ev(velocities: NDArray[np.float64], mass_kg: float) -> NDArray[np.float64]:
    return 0.5 * mass_kg * np.sum(velocities**2, axis=1) / _E


def _rate_tolerance(*, trials: int, probability: float, nu_max_dt: float) -> float:
    """The relative tolerance a measured rate is entitled to, from closed forms only.

    Two terms, both derived rather than tuned:

    * **Monte-Carlo.** The count is binomial ``(trials, probability)``, so the relative
      standard error of ``count / trials`` is ``sqrt((1 - p) / (trials p))``.
    * **Discretisation.** The scheme's real-collision probability per step is
      ``(1 - exp(-nu_max dt)) nu / nu_max``, which is ``nu dt (1 - nu_max dt / 2 + ...)``.
      The first-order deficit ``nu_max dt / 2`` is a bias, not noise, so it adds.
    """
    monte_carlo = math.sqrt((1.0 - probability) / (trials * probability))
    return _SIGMA * monte_carlo + nu_max_dt / 2.0


class TestTheNullCollisionFrequencyCeiling:
    """``nu_max = max over the energy range of n_gas sum_k sigma_k(E) v(E)`` — doc 03 §4.5.

    Getting this wrong is the failure mode the null-collision method is most exposed to,
    and it is silent in one direction: a ``nu_max`` that is too *small* caps every
    particle's collision probability at ``1 - exp(-nu_max dt)`` and the run simply
    under-collides, producing a plausible, collisionless-looking IEDF.
    """

    def test_a_constant_cross_section_puts_the_maximum_at_the_energy_ceiling(self) -> None:
        sigma = 5e-19
        ceiling = 100.0
        gas = _gas()
        sampler = NullCollisionSampler(
            species=_argon_ions(),
            gas=gas,
            processes=(
                Process(
                    kind=CollisionKind.ION_CHARGE_EXCHANGE,
                    section=_section(
                        ProcessType.CHARGE_EXCHANGE, sigma_m2=sigma, max_energy_ev=ceiling
                    ),
                ),
            ),
            energy_ceiling_ev=ceiling,
        )

        expected = gas.density_m3 * sigma * _speed_of(ceiling, _M_AR)

        assert sampler.nu_max_per_s == pytest.approx(expected, rel=1e-12)

    @pytest.mark.physics
    def test_a_falling_cross_section_has_its_maximum_inside_the_interval(self) -> None:
        sigma_low, sigma_high, ceiling = 1e-18, 0.0, 90.0
        gas = _gas()
        sampler = NullCollisionSampler(
            species=_argon_ions(),
            gas=gas,
            processes=(
                Process(
                    kind=CollisionKind.ION_CHARGE_EXCHANGE,
                    section=_ramp_section(
                        sigma_low=sigma_low, sigma_high=sigma_high, max_energy_ev=ceiling
                    ),
                ),
            ),
            energy_ceiling_ev=ceiling,
        )

        e_star = 30.0
        sigma_star = sigma_low + (sigma_high - sigma_low) * e_star / ceiling
        expected = gas.density_m3 * sigma_star * _speed_of(e_star, _M_AR)

        assert sampler.nu_max_per_s == pytest.approx(expected, rel=1e-10)
        scan = np.linspace(0.0, ceiling, 10_001)
        assert float(sampler.total_frequency_per_s(scan).max()) <= sampler.nu_max_per_s

    def test_the_ceiling_may_not_exceed_the_tabulated_range(self) -> None:
        with pytest.raises(ValueError, match="tabulated"):
            NullCollisionSampler(
                species=_argon_ions(),
                gas=_gas(),
                processes=(
                    Process(
                        kind=CollisionKind.ION_CHARGE_EXCHANGE,
                        section=_section(
                            ProcessType.CHARGE_EXCHANGE, sigma_m2=5e-19, max_energy_ev=50.0
                        ),
                    ),
                ),
                energy_ceiling_ev=100.0,
            )

    def test_a_particle_above_the_ceiling_is_a_runtime_error(self) -> None:
        ceiling = 10.0
        sampler = NullCollisionSampler(
            species=_argon_ions(),
            gas=_gas(),
            processes=(
                Process(
                    kind=CollisionKind.ION_CHARGE_EXCHANGE,
                    section=_section(
                        ProcessType.CHARGE_EXCHANGE, sigma_m2=5e-19, max_energy_ev=ceiling
                    ),
                ),
            ),
            energy_ceiling_ev=ceiling,
        )

        with pytest.raises(EnergyCeilingExceededError, match="ceiling"):
            sampler.collide(
                collision_generator(_SEED),
                positions_m=np.zeros(4),
                velocities_m_per_s=_beam(4, energy_ev=50.0, mass_kg=_M_AR),
                dt_s=1e-12,
            )


class TestTheSampledRateMatchesTheClosedForm:
    """The sampled collision rate against ``n_gas sigma(v) v``.

    Every trial is one particle for one step with its velocity restored afterwards, so the
    trials are independent Bernoulli draws at a *known* energy and the estimator's variance
    is the binomial one written down in :func:`_rate_tolerance`. Restoring the velocity is
    what makes the closed form exact: a particle that collided would otherwise carry a
    different energy — and therefore a different ``sigma v`` — into the next step.
    """

    @staticmethod
    def _electron_sampler(ceiling: float) -> NullCollisionSampler:
        return NullCollisionSampler(
            species=_electrons(),
            gas=_gas(),
            processes=(
                Process(
                    kind=CollisionKind.ELECTRON_ELASTIC,
                    section=_section(
                        ProcessType.EFFECTIVE,
                        sigma_m2=1e-19,
                        max_energy_ev=ceiling,
                        reactants=("e", "Ar"),
                        products=("e", "Ar"),
                    ),
                ),
                Process(
                    kind=CollisionKind.ELECTRON_EXCITATION,
                    section=_section(
                        ProcessType.EXCITATION,
                        sigma_m2=5e-20,
                        max_energy_ev=ceiling,
                        threshold_ev=11.5,
                        reactants=("e", "Ar"),
                        products=("e", "Ar*"),
                    ),
                ),
                Process(
                    kind=CollisionKind.ELECTRON_IONISATION,
                    section=_section(
                        ProcessType.IONIZATION,
                        sigma_m2=3e-20,
                        max_energy_ev=ceiling,
                        threshold_ev=15.76,
                        reactants=("e", "Ar"),
                        products=("e", "e", "Ar^+"),
                    ),
                ),
            ),
            energy_ceiling_ev=ceiling,
        )

    @pytest.mark.physics
    @pytest.mark.parametrize("beam_ev", [4.0, 20.0, 60.0])
    def test_every_open_channel_collides_at_n_sigma_v(self, beam_ev: float) -> None:
        ceiling = 100.0
        sampler = self._electron_sampler(ceiling)
        rng = collision_generator(_SEED)

        nu_max_dt = 1e-2
        dt = nu_max_dt / sampler.nu_max_per_s
        n_particles, n_repeats = 400_000, 20
        trials = n_particles * n_repeats

        positions = np.zeros(n_particles)
        totals = np.zeros(len(sampler.processes), dtype=np.int64)
        for _ in range(n_repeats):
            outcome = sampler.collide(
                rng,
                positions_m=positions,
                velocities_m_per_s=_beam(n_particles, energy_ev=beam_ev, mass_kg=_M_E),
                dt_s=dt,
            )
            totals += np.asarray(outcome.counts_by_process, dtype=np.int64)

        speed = _speed_of(beam_ev, _M_E)
        p_null = sampler.null_probability(dt)
        expected_nu = np.asarray(sampler.frequencies_per_s(beam_ev)).ravel()

        for index, process in enumerate(sampler.processes):
            threshold = process.threshold_ev
            if threshold is not None and beam_ev < threshold:
                assert totals[index] == 0, f"{process.label} fired below its threshold"
                assert expected_nu[index] == 0.0
                continue

            sigma = float(process.section.sigma_m2[-1])
            physical_nu = sampler.gas.density_m3 * sigma * speed
            assert expected_nu[index] == pytest.approx(physical_nu, rel=1e-12)

            measured_nu = totals[index] / (trials * dt)
            probability = p_null * physical_nu / sampler.nu_max_per_s
            tolerance = _rate_tolerance(trials=trials, probability=probability, nu_max_dt=nu_max_dt)
            assert measured_nu == pytest.approx(physical_nu, rel=tolerance), (
                f"{process.label}: measured {measured_nu:.4e} /s against the closed form "
                f"n sigma v = {physical_nu:.4e} /s, tolerance {tolerance:.3%}"
            )

    @pytest.mark.physics
    def test_the_rate_is_right_when_the_real_probability_is_a_percent_of_the_ceiling(
        self,
    ) -> None:
        ceiling = 400.0
        sampler = self._electron_sampler(ceiling)
        beam_ev = 4.0
        speed = _speed_of(beam_ev, _M_E)
        physical_nu = sampler.gas.density_m3 * 1e-19 * speed
        assert physical_nu / sampler.nu_max_per_s < 0.06

        nu_max_dt = 1e-2
        dt = nu_max_dt / sampler.nu_max_per_s
        rng = collision_generator(_SEED)
        n_particles, n_repeats = 400_000, 100
        trials = n_particles * n_repeats

        positions = np.zeros(n_particles)
        collisions = 0
        for _ in range(n_repeats):
            outcome = sampler.collide(
                rng,
                positions_m=positions,
                velocities_m_per_s=_beam(n_particles, energy_ev=beam_ev, mass_kg=_M_E),
                dt_s=dt,
            )
            collisions += int(outcome.n_collisions)

        measured_nu = collisions / (trials * dt)
        probability = sampler.null_probability(dt) * physical_nu / sampler.nu_max_per_s
        tolerance = _rate_tolerance(trials=trials, probability=probability, nu_max_dt=nu_max_dt)

        assert measured_nu == pytest.approx(physical_nu, rel=tolerance)


class TestTheCollisionFreePathIsExponential:
    """Constant cross section, no field: the free path is ``Exp(mean = 1 / (n sigma))``.

    This is the one property that pins the *distribution* rather than its mean, and it is
    what a rate test cannot see: a sampler that collided every particle deterministically
    once per ``1/nu`` steps would reproduce the rate exactly and the physics not at all.
    """

    @pytest.mark.physics
    @pytest.mark.slow
    def test_the_free_path_distribution_matches_the_analytic_cdf(self) -> None:
        sigma, density = 5e-19, 1e21
        mean_free_path = 1.0 / (density * sigma)

        ceiling = 20.0
        sampler = NullCollisionSampler(
            species=_electrons(),
            gas=_gas(density_m3=density),
            processes=(
                Process(
                    kind=CollisionKind.ELECTRON_ELASTIC,
                    section=_section(
                        ProcessType.EFFECTIVE,
                        sigma_m2=sigma,
                        max_energy_ev=ceiling,
                        reactants=("e", "Ar"),
                        products=("e", "Ar"),
                    ),
                ),
            ),
            energy_ceiling_ev=ceiling,
        )

        beam_ev = ceiling
        speed = _speed_of(beam_ev, _M_E)
        nu = density * sigma * speed
        dt = 5e-3 / nu
        n_particles, max_steps = 5_000, 3_000

        rng = collision_generator(_SEED)
        first_step = np.full(n_particles, -1, dtype=np.int64)
        positions = np.zeros(n_particles)
        velocities = _beam(n_particles, energy_ev=beam_ev, mass_kg=_M_E)

        for step in range(max_steps):
            outcome = sampler.collide(
                rng, positions_m=positions, velocities_m_per_s=velocities, dt_s=dt
            )
            hit = (np.asarray(outcome.process_index) >= 0) & (first_step < 0)
            if np.any(hit):
                first_step[hit] = step + 1
                velocities = velocities.copy()
                velocities[hit] = 0.0
            if (first_step > 0).all():
                break

        collided = first_step > 0
        assert collided.all(), f"{int((~collided).sum())} particles never collided"

        path = first_step[collided] * speed * dt
        assert path.mean() == pytest.approx(
            mean_free_path, rel=_SIGMA / math.sqrt(n_particles) + 5e-3
        )

        result = stats.kstest(path, stats.expon(scale=mean_free_path).cdf)
        assert result.statistic < _KS_CRITICAL / math.sqrt(n_particles), (
            f"KS distance {result.statistic:.4f} against Exp(mean = {mean_free_path:.4e} m)"
        )


class TestNullCollisionEqualsDirectRejection:
    """The equivalence that justifies the whole method — doc 03 §4.5.

    The null-collision method is an optimisation: it replaces a per-particle evaluation of
    the total collision frequency with a single Bernoulli draw against a constant ceiling.
    That is only legitimate if it samples the same distribution. The comparison is against
    a direct rejection sampler written here from the definition, so the two implementations
    share no code and cannot fail together.
    """

    @staticmethod
    def _direct_rejection(
        sampler: NullCollisionSampler,
        rng: np.random.Generator,
        *,
        energy_ev: float,
        n_particles: int,
        dt_s: float,
    ) -> NDArray[np.int64]:
        """Per-channel counts from the textbook direct method: no ceiling, no null process."""
        nu = np.asarray(sampler.frequencies_per_s(energy_ev)).ravel()
        nu_total = float(nu.sum())
        probability = -math.expm1(-nu_total * dt_s)

        draw = rng.random(n_particles)
        collides = draw < probability
        selector = draw[collides] / probability
        edges = np.cumsum(nu) / nu_total
        chosen = np.searchsorted(edges, selector, side="right")
        return np.bincount(chosen, minlength=nu.size).astype(np.int64)

    @pytest.mark.physics
    def test_the_two_samplers_agree_channel_by_channel(self) -> None:
        ceiling, beam_ev = 100.0, 30.0
        sampler = TestTheSampledRateMatchesTheClosedForm._electron_sampler(ceiling)

        nu_max_dt = 1e-2
        dt = nu_max_dt / sampler.nu_max_per_s
        n_particles, n_repeats = 400_000, 20
        trials = n_particles * n_repeats

        null_rng = collision_generator(_SEED)
        direct_rng = generator(_SEED + 1, Stream.COLLISIONS)
        positions = np.zeros(n_particles)

        null_counts = np.zeros(len(sampler.processes), dtype=np.int64)
        direct_counts = np.zeros(len(sampler.processes), dtype=np.int64)
        for _ in range(n_repeats):
            outcome = sampler.collide(
                null_rng,
                positions_m=positions,
                velocities_m_per_s=_beam(n_particles, energy_ev=beam_ev, mass_kg=_M_E),
                dt_s=dt,
            )
            null_counts += np.asarray(outcome.counts_by_process, dtype=np.int64)
            direct_counts += self._direct_rejection(
                sampler, direct_rng, energy_ev=beam_ev, n_particles=n_particles, dt_s=dt
            )

        for index, process in enumerate(sampler.processes):
            a, b = int(null_counts[index]), int(direct_counts[index])
            assert a > 0, f"{process.label} never fired"
            pooled = (a + b) / (2.0 * trials)
            standard_error = math.sqrt(2.0 * pooled * (1.0 - pooled) / trials)
            z = (a - b) / (trials * standard_error)
            assert abs(z) < _SIGMA, f"{process.label}: null {a} vs direct {b}, z = {z:.2f}"


class TestChargeExchange:
    """doc 03 §4.5 — "the single most important collision for this project".

    Three separate claims, tested separately because they fail separately:

    * the ion and the neutral **swap identities**, so the pair conserves momentum and
      energy exactly;
    * the surviving ion is **cold**, at the gas temperature, wherever it happens to be —
      which is what "a slow ion at the local potential" means in a code where the potential
      is a property of position;
    * the outcome is **independent of the incoming ion energy**, which is what makes CX
      produce structure rather than merely drag.
    """

    @staticmethod
    def _sampler(ceiling: float = 300.0) -> NullCollisionSampler:
        return NullCollisionSampler(
            species=_argon_ions(),
            gas=_gas(),
            processes=(
                Process(
                    kind=CollisionKind.ION_CHARGE_EXCHANGE,
                    section=_section(
                        ProcessType.CHARGE_EXCHANGE, sigma_m2=5e-19, max_energy_ev=ceiling
                    ),
                ),
            ),
            energy_ceiling_ev=ceiling,
        )

    @staticmethod
    def _collide_everything(
        sampler: NullCollisionSampler, *, n: int, energy_ev: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """One step at ``nu_max dt = 20``, so every candidate drawn is tested.

        Not every ion collides: with a constant cross section the real frequency is
        ``nu_max sqrt(E / E_ceiling)``, so a 1 eV beam under a 300 eV ceiling collides at
        6 % per step and the rest are null collisions. That is the method working, and the
        assertion below is on the sample size the statistics need rather than on a rate the
        other tests in this file already own.
        """
        before = _beam(n, energy_ev=energy_ev, mass_kg=_M_AR)
        outcome = sampler.collide(
            collision_generator(_SEED),
            positions_m=np.zeros(n),
            velocities_m_per_s=before,
            dt_s=20.0 / sampler.nu_max_per_s,
        )
        hit = np.asarray(outcome.process_index) >= 0
        assert int(hit.sum()) > 1_000
        return (
            before[hit],
            np.asarray(outcome.velocities_m_per_s)[hit],
            np.asarray(outcome.target_velocities_m_per_s)[hit],
        )

    @pytest.mark.physics
    def test_the_ion_and_the_neutral_swap_identities(self) -> None:
        sampler = self._sampler()
        before, after, target = self._collide_everything(sampler, n=20_000, energy_ev=100.0)

        np.testing.assert_array_equal(after, target)

        neutral_after = before
        momentum_before = _M_AR * (before + target)
        momentum_after = _M_AR * (after + neutral_after)
        np.testing.assert_allclose(momentum_after, momentum_before, rtol=0.0, atol=0.0)

        energy_before = np.sum(before**2, axis=1) + np.sum(target**2, axis=1)
        energy_after = np.sum(after**2, axis=1) + np.sum(neutral_after**2, axis=1)
        np.testing.assert_allclose(energy_after, energy_before, rtol=1e-15)

    @pytest.mark.physics
    def test_the_surviving_ion_is_at_the_gas_temperature(self) -> None:
        sampler = self._sampler()
        _, after, _ = self._collide_everything(sampler, n=50_000, energy_ev=250.0)

        energies = _kinetic_energy_ev(after, _M_AR)
        expected = 1.5 * _K_B * sampler.gas.temperature_k / _E
        sem = math.sqrt(1.5) * (_K_B * sampler.gas.temperature_k / _E) / math.sqrt(after.shape[0])
        assert energies.mean() == pytest.approx(expected, abs=_SIGMA * sem)

        scale = sampler.gas.thermal_speed_m_per_s
        result = stats.kstest(after[:, 2], stats.norm(loc=0.0, scale=scale).cdf)
        assert result.statistic < _KS_CRITICAL / math.sqrt(after.shape[0])

    @pytest.mark.physics
    def test_the_outcome_does_not_depend_on_the_incoming_energy(self) -> None:
        sampler = self._sampler()
        _, cold_after, _ = self._collide_everything(sampler, n=40_000, energy_ev=1.0)
        _, hot_after, _ = self._collide_everything(sampler, n=40_000, energy_ev=250.0)

        result = stats.ks_2samp(
            _kinetic_energy_ev(cold_after, _M_AR), _kinetic_energy_ev(hot_after, _M_AR)
        )
        assert result.pvalue > 1e-3


class TestTheChargeExchangeIedf:
    """The observable doc 03 §4.5 says the project turns on, against its closed form.

    In a uniform field with a constant CX cross section the ion flux is conserved, so CX
    events are uniform in path length and each ion's *last* CX before the wall lies at a
    distance ``s`` with density ``exp(-s/lambda)/lambda``. A uniform field turns that into
    an energy: ``E = e |E_z| s``, so

        ``f(E) = (1 / <E>) exp(-E / <E>)``     with     ``<E> = e |E_z| lambda``.

    Nothing in the sampler can produce that by construction. It requires the free path to
    be exponential *and* the post-CX ion to be cold *and* the rate to be right, all at
    once, inside a real push loop — which is why this is the test that would have caught
    the class of error ADR-011 records.
    """

    @pytest.mark.physics
    @pytest.mark.slow
    def test_the_wall_iedf_is_exponential_with_mean_e_field_times_mean_free_path(
        self,
    ) -> None:
        length = 0.02
        bias_v = 250.0
        field_v_per_m = -bias_v / length
        sigma, density = 5e-19, 1e21
        mean_free_path = 1.0 / (density * sigma)
        assert length / mean_free_path == pytest.approx(10.0)

        ceiling = 300.0
        sampler = NullCollisionSampler(
            species=_argon_ions(),
            gas=_gas(density_m3=density),
            processes=(
                Process(
                    kind=CollisionKind.ION_CHARGE_EXCHANGE,
                    section=_section(
                        ProcessType.CHARGE_EXCHANGE, sigma_m2=sigma, max_energy_ev=ceiling
                    ),
                ),
            ),
            energy_ceiling_ev=ceiling,
        )
        ions = _argon_ions()

        nu_max_dt = 1e-2
        dt = nu_max_dt / sampler.nu_max_per_s
        n_particles = 20_000

        rng = collision_generator(_SEED)
        positions = np.full(n_particles, length)
        velocities: object = jnp.zeros((n_particles, 3))
        field = np.full(n_particles, field_v_per_m)
        arrival_ev = np.zeros(n_particles)
        in_flight = np.ones(n_particles, dtype=np.bool_)
        parking = length / 2.0

        for _ in range(12_000):
            pushed_positions, pushed_velocities = push_leapfrog(
                ions,
                positions_m=positions,
                velocities_m_per_s=velocities,
                e_field_v_per_m=field,
                dt_s=dt,
            )
            positions = np.array(pushed_positions)
            moved = np.array(pushed_velocities)

            escaped = in_flight & (positions > length)
            positions[escaped] = 2.0 * length - positions[escaped]
            moved[escaped, 2] *= -1.0

            absorbed = in_flight & (positions <= 0.0)
            if np.any(absorbed):
                arrival_ev[absorbed] = _kinetic_energy_ev(moved[absorbed], _M_AR)
                in_flight &= ~absorbed
                positions[absorbed] = parking
                moved[absorbed] = 0.0
                field[absorbed] = 0.0
            if not in_flight.any():
                break

            velocities = sampler.collide(
                rng, positions_m=positions, velocities_m_per_s=moved, dt_s=dt
            ).velocities_m_per_s

        assert not in_flight.any(), f"{int(in_flight.sum())} ions never reached the wall"
        energies = arrival_ev

        expected_mean = bias_v / length * mean_free_path
        assert expected_mean == pytest.approx(25.0)

        tolerance = _SIGMA / math.sqrt(energies.size) + nu_max_dt / 2.0 + 2e-3
        assert energies.mean() == pytest.approx(expected_mean, rel=tolerance), (
            f"mean IEDF energy {energies.mean():.3f} eV against e|E| lambda = "
            f"{expected_mean:.3f} eV"
        )

        result = stats.kstest(energies, stats.expon(scale=energies.mean()).cdf)
        assert result.statistic < _KS_CRITICAL / math.sqrt(energies.size), (
            f"KS distance {result.statistic:.4f} against an exponential IEDF"
        )


class TestElectronKinematics:
    """Energy bookkeeping for the three electron channels — doc 03 §4.5.

    Each assertion is per-particle and exact, not statistical. A factor of two in an
    inelastic energy loss is invisible in a mean and fatal in an EEDF.
    """

    @staticmethod
    def _one_channel(kind: CollisionKind, section: CrossSection) -> NullCollisionSampler:
        ceiling = float(section.energy_ev[-1])
        return NullCollisionSampler(
            species=_electrons(),
            gas=_gas(),
            processes=(Process(kind=kind, section=section),),
            energy_ceiling_ev=ceiling,
        )

    @staticmethod
    def _collide_all(
        sampler: NullCollisionSampler, *, n: int, energy_ev: float
    ) -> tuple[NDArray[np.float64], object]:
        before = _beam(n, energy_ev=energy_ev, mass_kg=_M_E)
        outcome = sampler.collide(
            collision_generator(_SEED),
            positions_m=np.linspace(0.0, 1.0, n),
            velocities_m_per_s=before,
            dt_s=20.0 / sampler.nu_max_per_s,
        )
        return before, outcome

    @pytest.mark.physics
    def test_elastic_loses_exactly_the_two_m_over_mass_fraction(self) -> None:
        ceiling = 20.0
        sampler = self._one_channel(
            CollisionKind.ELECTRON_ELASTIC,
            _section(
                ProcessType.EFFECTIVE,
                sigma_m2=5e-19,
                max_energy_ev=ceiling,
                reactants=("e", "Ar"),
                products=("e", "Ar"),
            ),
        )
        before, outcome = self._collide_all(sampler, n=20_000, energy_ev=ceiling)
        hit = np.asarray(outcome.process_index) >= 0  # type: ignore[attr-defined]
        after = np.asarray(outcome.velocities_m_per_s)[hit]  # type: ignore[attr-defined]
        incoming = before[hit]

        cos_chi = np.sum(incoming * after, axis=1) / (
            np.linalg.norm(incoming, axis=1) * np.linalg.norm(after, axis=1)
        )
        ratio = _M_E / _M_AR
        expected = _kinetic_energy_ev(incoming, _M_E) * (1.0 - 2.0 * ratio * (1.0 - cos_chi))
        np.testing.assert_allclose(_kinetic_energy_ev(after, _M_E), expected, rtol=1e-12)

        loss = 1.0 - _kinetic_energy_ev(after, _M_E) / _kinetic_energy_ev(incoming, _M_E)
        sem = 2.0 * ratio / math.sqrt(3.0 * after.shape[0])
        assert loss.mean() == pytest.approx(2.0 * ratio, abs=_SIGMA * sem)

    @pytest.mark.physics
    def test_excitation_removes_exactly_the_threshold_energy(self) -> None:
        threshold, ceiling = 11.5, 30.0
        sampler = self._one_channel(
            CollisionKind.ELECTRON_EXCITATION,
            _section(
                ProcessType.EXCITATION,
                sigma_m2=5e-19,
                max_energy_ev=ceiling,
                threshold_ev=threshold,
                reactants=("e", "Ar"),
                products=("e", "Ar*"),
            ),
        )
        before, outcome = self._collide_all(sampler, n=10_000, energy_ev=ceiling)
        hit = np.asarray(outcome.process_index) >= 0  # type: ignore[attr-defined]
        after = np.asarray(outcome.velocities_m_per_s)[hit]  # type: ignore[attr-defined]

        expected = _kinetic_energy_ev(before[hit], _M_E) - threshold
        np.testing.assert_allclose(_kinetic_energy_ev(after, _M_E), expected, rtol=1e-11)

    @pytest.mark.physics
    def test_ionisation_conserves_energy_and_emits_one_electron_and_one_ion(self) -> None:
        threshold, ceiling = 15.76, 60.0
        sampler = self._one_channel(
            CollisionKind.ELECTRON_IONISATION,
            _section(
                ProcessType.IONIZATION,
                sigma_m2=5e-19,
                max_energy_ev=ceiling,
                threshold_ev=threshold,
                reactants=("e", "Ar"),
                products=("e", "e", "Ar^+"),
            ),
        )
        n = 10_000
        positions = np.linspace(0.0, 1.0, n)
        before = _beam(n, energy_ev=ceiling, mass_kg=_M_E)
        outcome = sampler.collide(
            collision_generator(_SEED),
            positions_m=positions,
            velocities_m_per_s=before,
            dt_s=20.0 / sampler.nu_max_per_s,
        )

        index = np.asarray(outcome.process_index)
        hit = index >= 0
        n_events = int(hit.sum())
        assert n_events > 0.99 * n

        secondary_v = np.asarray(outcome.new_electron_velocities_m_per_s)
        new_ion_v = np.asarray(outcome.new_ion_velocities_m_per_s)
        assert secondary_v.shape == (n_events, 3)
        assert new_ion_v.shape == (n_events, 3)

        scattered = _kinetic_energy_ev(np.asarray(outcome.velocities_m_per_s)[hit], _M_E)
        ejected = _kinetic_energy_ev(secondary_v, _M_E)
        incoming = _kinetic_energy_ev(before[hit], _M_E)
        np.testing.assert_allclose(scattered + ejected + threshold, incoming, rtol=1e-11)

        np.testing.assert_allclose(
            np.asarray(outcome.new_electron_positions_m), positions[hit], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(outcome.new_ion_positions_m), positions[hit], rtol=0.0, atol=0.0
        )

        ion_energy = _kinetic_energy_ev(new_ion_v, _M_AR)
        expected = 1.5 * _K_B * sampler.gas.temperature_k / _E
        sem = math.sqrt(1.5) * (_K_B * sampler.gas.temperature_k / _E) / math.sqrt(n_events)
        assert ion_energy.mean() == pytest.approx(expected, abs=_SIGMA * sem)


class TestIonElasticKinematics:
    """Ion elastic scattering, "isotropic" as doc 03 §4.5's table specifies it.

    Momentum and energy are conserved for the (ion, neutral) pair iff the centre-of-mass
    velocity is unchanged and the relative speed is preserved. Both are asserted directly:
    a total-energy check over the ensemble could be satisfied by two errors cancelling,
    a per-pair centre-of-mass check cannot.
    """

    @pytest.mark.physics
    def test_the_centre_of_mass_velocity_and_relative_speed_are_preserved(self) -> None:
        ceiling = 100.0
        sampler = NullCollisionSampler(
            species=_argon_ions(),
            gas=_gas(),
            processes=(
                Process(
                    kind=CollisionKind.ION_ELASTIC,
                    section=_section(ProcessType.ELASTIC, sigma_m2=5e-19, max_energy_ev=ceiling),
                ),
            ),
            energy_ceiling_ev=ceiling,
        )

        n = 20_000
        before = _beam(n, energy_ev=50.0, mass_kg=_M_AR)
        outcome = sampler.collide(
            collision_generator(_SEED),
            positions_m=np.zeros(n),
            velocities_m_per_s=before,
            dt_s=20.0 / sampler.nu_max_per_s,
        )
        hit = np.asarray(outcome.process_index) >= 0
        incoming = before[hit]
        target = np.asarray(outcome.target_velocities_m_per_s)[hit]
        after = np.asarray(outcome.velocities_m_per_s)[hit]

        total_mass = _M_AR + sampler.gas.mass_kg
        v_cm = (_M_AR * incoming + sampler.gas.mass_kg * target) / total_mass
        relative_speed = np.linalg.norm(incoming - target, axis=1)

        expected = sampler.gas.mass_kg / total_mass * relative_speed
        np.testing.assert_allclose(np.linalg.norm(after - v_cm, axis=1), expected, rtol=1e-12)

        direction = (after - v_cm) / expected[:, None]
        incoming_direction = (incoming - v_cm) / expected[:, None]
        cos_chi = np.sum(direction * incoming_direction, axis=1)
        assert cos_chi.mean() == pytest.approx(0.0, abs=_SIGMA / math.sqrt(3.0 * cos_chi.size))


class TestReproducibility:
    """doc 00 E3, doc 10 §5 — bit-for-bit reproduction from a manifest.

    The collision module is the noisiest thing in the kernel, so it is the one most able
    to break the promise. It draws from :attr:`Stream.COLLISIONS` and from nothing else.
    """

    @staticmethod
    def _run(rng: np.random.Generator) -> NDArray[np.float64]:
        sampler = TestChargeExchange._sampler()
        outcome = sampler.collide(
            rng,
            positions_m=np.zeros(2_000),
            velocities_m_per_s=_beam(2_000, energy_ev=100.0, mass_kg=_M_AR),
            dt_s=1.0 / sampler.nu_max_per_s,
        )
        return np.asarray(outcome.velocities_m_per_s)

    def test_the_same_root_seed_reproduces_the_run_bit_for_bit(self) -> None:
        first = self._run(collision_generator(_SEED))
        second = self._run(collision_generator(_SEED))

        np.testing.assert_array_equal(first, second)

    def test_the_generator_is_the_collisions_stream(self) -> None:
        from_helper = self._run(collision_generator(_SEED))
        from_stream = self._run(generator(_SEED, Stream.COLLISIONS))

        np.testing.assert_array_equal(from_helper, from_stream)

    def test_a_different_stream_gives_a_different_realisation(self) -> None:
        collisions = self._run(generator(_SEED, Stream.COLLISIONS))
        photons = self._run(generator(_SEED, Stream.PHOTONS))

        assert not np.array_equal(collisions, photons)


class TestTheBackgroundGas:
    """The neutral reservoir, against the ideal gas law and doc 01 §2.1's RP-1."""

    def test_the_neutral_density_matches_the_ideal_gas_law_at_rp1(self) -> None:
        pascals_per_mtorr = 133.322 / 1000.0
        density = neutral_density_m3(pressure_pa=5.0 * pascals_per_mtorr, temperature_k=300.0)

        assert density == pytest.approx(1.61e20, rel=1e-2)

    def test_the_thermal_speed_is_the_per_component_maxwellian_width(self) -> None:
        gas = _gas(temperature_k=300.0)

        assert gas.thermal_speed_m_per_s == pytest.approx(
            math.sqrt(_K_B * 300.0 / _M_AR), rel=1e-12
        )
        assert gas.thermal_speed_m_per_s == pytest.approx(251.0, rel=1e-2)


class TestTheGuards:
    """Configuration errors that would otherwise produce a plausible, wrong run."""

    def test_a_process_whose_kind_contradicts_its_cross_section_is_refused(self) -> None:
        with pytest.raises(ValueError, match="CHARGE EXCHANGE"):
            Process(
                kind=CollisionKind.ION_CHARGE_EXCHANGE,
                section=_section(ProcessType.ELASTIC, sigma_m2=5e-19, max_energy_ev=10.0),
            )

    def test_mixing_electron_and_ion_processes_in_one_sampler_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one species"):
            NullCollisionSampler(
                species=_argon_ions(),
                gas=_gas(),
                processes=(
                    Process(
                        kind=CollisionKind.ION_CHARGE_EXCHANGE,
                        section=_section(
                            ProcessType.CHARGE_EXCHANGE, sigma_m2=5e-19, max_energy_ev=10.0
                        ),
                    ),
                    Process(
                        kind=CollisionKind.ELECTRON_ELASTIC,
                        section=_section(
                            ProcessType.EFFECTIVE,
                            sigma_m2=1e-19,
                            max_energy_ev=10.0,
                            reactants=("e", "Ar"),
                            products=("e", "Ar"),
                        ),
                    ),
                ),
                energy_ceiling_ev=10.0,
            )

    def test_an_empty_process_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            NullCollisionSampler(
                species=_argon_ions(), gas=_gas(), processes=(), energy_ceiling_ev=10.0
            )

    def test_a_non_positive_timestep_is_refused(self) -> None:
        sampler = TestChargeExchange._sampler(ceiling=10.0)

        with pytest.raises(ValueError, match="positive"):
            sampler.collide(
                collision_generator(_SEED),
                positions_m=np.zeros(2),
                velocities_m_per_s=np.zeros((2, 3)),
                dt_s=0.0,
            )

    def test_velocities_must_be_three_dimensional(self) -> None:
        sampler = TestChargeExchange._sampler(ceiling=10.0)

        with pytest.raises(ValueError, match="1D3V"):
            sampler.collide(
                collision_generator(_SEED),
                positions_m=np.zeros(2),
                velocities_m_per_s=np.zeros((2, 2)),
                dt_s=1e-12,
            )

    def test_no_particles_is_not_an_error(self) -> None:
        sampler = TestChargeExchange._sampler(ceiling=10.0)
        outcome = sampler.collide(
            collision_generator(_SEED),
            positions_m=np.zeros(0),
            velocities_m_per_s=np.zeros((0, 3)),
            dt_s=1e-12,
        )

        assert outcome.n_collisions == 0
        assert np.asarray(outcome.velocities_m_per_s).shape == (0, 3)
