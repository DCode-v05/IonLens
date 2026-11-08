"""The solver contract the manifest engine calls, and the one adapter that satisfies it.

## Why this is not simply ``ForwardSolver``

doc 08 §4 declares ``ForwardSolver.solve(self, params: PlasmaParams, t: TimeGrid)``. That
signature cannot express a **steady** solve, and steady is what L0 and L1 mostly are:

- :class:`~ionlens.core.state.ScalarField` indexes a time-dependent field ``(n_t, n_z)`` and a
  steady one ``(n_z,)``, so a solver handed a ``TimeGrid`` is being asked for the former;
- :attr:`~ionlens.core.state.PlasmaState.time` is already ``TimeGrid | None``;
- :class:`~ionlens.core.protocols.forward.IonEnergyFlux` already has an ``is_steady`` view.

The *data model* therefore already says the time grid is optional, and only the protocol
signature disagrees with it. :class:`ManifestSolver` is doc 08 §4's contract with ``t``
widened to ``TimeGrid | None``, minus ``cost_estimate``, and the widening is the correction
the data model implies. Both departures are stated rather than assumed:

- **``t: TimeGrid | None``.** A solver that implements doc 08 §4 exactly — ``t: TimeGrid``
  — is *not* assignable to this protocol, because parameter types are contravariant. That
  is a real consequence and the reason it is recorded here and in ADR-008 rather than
  quietly worked around: doc 08 §4 should adopt the optional form when it is next revised,
  and until it does, a solver written for the manifest engine widens ``t``.
- **No ``cost_estimate``.** doc 08 §4 has it for the doc 10 §6 work queue, which is not
  this WBS item. Requiring it here would have made the L0 adapter invent a wall-clock
  figure with no measurement behind it, and doc 00 C1 counts a fabricated number as a
  defect whether or not anything reads it.

## Why the adapter lives in this package

``ionlens.physics.analytic.AnalyticSheathSolver`` is committed and predates the protocol; its
own docstring says it is "shaped to the ``ForwardSolver`` contract ... but not declared to
implement it". The shapes differ in more than an annotation — ``solve`` takes keyword
``grid``/``time``, ``flux`` takes no position and returns a bare ``Quantity`` rather than
the doc 01 §1.2 decomposition — so an adapter is needed. It belongs in ionlens-physics and
moves there when that package next implements the protocol natively; the entry-point
declaration in this package's ``pyproject.toml`` moves with it, and the manifest keeps
saying ``ionlens.physics.analytic.sheath`` throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

import numpy as np

from ionlens.core.protocols.config import SolverConfig
from ionlens.core.protocols.forward import IonEnergyFlux
from ionlens.core.protocols.metadata import Citation, SolverMetadata
from ionlens.core.state import Fidelity, PlasmaParams, PlasmaState, TimeGrid
from ionlens.core.units import magnitude_in
from ionlens.physics import __version__ as _physics_version
from ionlens.physics.analytic import (
    DEFAULT_DOMAIN_SHEATHS,
    DEFAULT_EDGE_TO_CENTRE_RATIO,
    DEFAULT_SAMPLES_PER_SHEATH,
    GAMMA_I_COLD_ION,
    AnalyticSheathSolver,
    SheathModel,
    ion_energy_flux,
    ion_flux,
)

__all__ = ["SOLVER_NAME", "AnalyticSheathForwardSolver", "ManifestSolver"]

SOLVER_NAME: Final[str] = "ionlens.physics.analytic.sheath"

_WALL_POSITION_M: Final[float] = 0.0

_CONFIG_KEYS: Final[tuple[str, ...]] = (
    "model",
    "h_l",
    "gamma_i",
    "domain_sheaths",
    "samples_per_sheath",
)


@runtime_checkable
class ManifestSolver(Protocol):
    """What ``ionlens run`` requires of a ``forward.solver`` — doc 08 §4, as amended above.

    Runtime checking is by *name only*, which is what :class:`typing.Protocol` gives and
    all that is wanted here: it turns a plugin that is not a solver at all into a loud
    failure at load time rather than an ``AttributeError`` after the run directory has
    been created. Signatures are mypy's job, run against the plugin's own source.
    """

    def configure(self, cfg: SolverConfig) -> None:
        """Apply the ``forward:`` block, refusing any key this solver does not own."""
        ...

    def solve(self, params: PlasmaParams, t: TimeGrid | None) -> PlasmaState:
        """Produce the state. ``t`` is ``None`` for a steady solve — see the module docstring."""
        ...

    def flux(self, state: PlasmaState, z: float) -> IonEnergyFlux:
        """Apply the doc 03 §6 flux functional at position ``z``, in metres."""
        ...

    def fidelity(self) -> Fidelity:
        """Which level of the doc 03 §1 hierarchy this is."""
        ...

    def metadata(self) -> SolverMetadata:
        """Name, version and citations — doc 00 C2, and doc 08 §7's ``solver_versions``."""
        ...


_CITATIONS: Final[tuple[Citation, ...]] = (
    Citation(
        key="langmuir1913",
        reference=(
            "I. Langmuir, 'The effect of space charge and residual gases on thermionic "
            "currents in high vacuum', Phys. Rev. 2, 450 (1913)"
        ),
        doi="10.1103/PhysRev.2.450",
    ),
    Citation(
        key="lieberman2005",
        reference=(
            "M. A. Lieberman and A. J. Lichtenberg, 'Principles of Plasma Discharges and "
            "Materials Processing', 2nd ed., Wiley (2005), ch. 6"
        ),
    ),
)


@dataclass(slots=True)
class AnalyticSheathForwardSolver:
    """``ionlens.physics.analytic.sheath`` as a manifest-resolvable solver.

    Mutable, and deliberately: doc 08 §4 declares ``configure`` as returning ``None`` and
    mutating, because a solver is constructed by the plugin loader (doc 08 §10) and
    configured by the manifest engine (doc 08 §6) at two different moments. It is the one
    place this package departs from immutability, and the departure stops here — the
    wrapped :class:`~ionlens.physics.analytic.AnalyticSheathSolver` is frozen, and
    :meth:`configure` replaces it rather than editing it.
    """

    solver: AnalyticSheathSolver = field(default_factory=AnalyticSheathSolver)


    def configure(self, cfg: SolverConfig) -> None:
        """Apply the ``forward:`` block, minus ``solver``.

        Raises:
            ValueError: If the block sets a key this solver does not own, or a value the
                analytic model rejects. The unknown-key rule of doc 08 §6 is enforced here
                rather than in the manifest schema because the solver is the only thing
                that knows its own vocabulary.
        """
        unknown = sorted(set(cfg.values) - set(_CONFIG_KEYS))
        if unknown:
            raise ValueError(
                f"forward: {SOLVER_NAME} does not read {', '.join(unknown)}. It reads: "
                f"{', '.join(_CONFIG_KEYS)}. A forward block written for another solver "
                "would otherwise run here with its settings silently ignored."
            )

        model = self.solver.model
        if "model" in cfg:
            name = cfg.require_str("model")
            try:
                model = SheathModel(name)
            except ValueError as exc:
                permitted = ", ".join(item.value for item in SheathModel)
                raise ValueError(
                    f"forward.model: {name!r} is not a doc 03 §2 sheath model; "
                    f"expected one of [{permitted}]"
                ) from exc

        self.solver = AnalyticSheathSolver(
            h_l=cfg.require_float("h_l") if "h_l" in cfg else DEFAULT_EDGE_TO_CENTRE_RATIO,
            gamma_i=cfg.require_float("gamma_i") if "gamma_i" in cfg else GAMMA_I_COLD_ION,
            model=model,
            domain_sheaths=(
                cfg.require_int("domain_sheaths")
                if "domain_sheaths" in cfg
                else DEFAULT_DOMAIN_SHEATHS
            ),
            samples_per_sheath=(
                cfg.require_int("samples_per_sheath")
                if "samples_per_sheath" in cfg
                else DEFAULT_SAMPLES_PER_SHEATH
            ),
        )

    def solve(self, params: PlasmaParams, t: TimeGrid | None) -> PlasmaState:
        """The L0 state on the solver's own grid.

        Raises:
            ValueError: If a time grid is supplied. doc 03 §2's expressions are steady,
                and attaching a time axis to them would claim a time dependence this level
                does not model — :class:`~ionlens.core.state.ScalarField` would then require
                ``(n_t, n_z)`` values that nothing here computes.
        """
        if t is not None:
            raise ValueError(
                f"{SOLVER_NAME} is steady: doc 03 §2's sheath expressions have no time "
                f"dependence, so a {t.n_points}-point time grid cannot be honoured. "
                "Use an L1 or L2 solver for a time-resolved run."
            )
        return self.solver.solve(params)

    def flux(self, state: PlasmaState, z: float) -> IonEnergyFlux:
        """``Gamma_E`` and ``Gamma_i`` at the wall, with the doc 01 §1.2 split kept.

        Both moments come from :mod:`ionlens.physics.analytic` rather than from integrating
        the state's fields, so this adapter cannot develop its own opinion about the
        quantity of interest — doc 03 §6 requires "the same functional" at every level.

        Raises:
            ValueError: If ``z`` is not the wall. doc 03 §2.3 states ``Gamma_E`` at the
                wall and nowhere else, and returning the wall value for a position inside
                the sheath would be a plausible number about the wrong place — the
                registration error doc 02 §2 calls the most damaging and least modelled
                systematic.
        """
        if z != _WALL_POSITION_M:
            raise ValueError(
                f"{SOLVER_NAME} evaluates the doc 03 §6 flux functional at the wall "
                f"(z = 0, doc 02 §2) and nowhere else; got z = {z} m. L0 has no in-sheath "
                "flux profile to sample."
            )

        params = state.params
        energy = float(
            magnitude_in(
                ion_energy_flux(params, h_l=self.solver.h_l, gamma_i=self.solver.gamma_i),
                "W/m**2",
            )
        )
        particles = float(
            magnitude_in(
                ion_flux(params, h_l=self.solver.h_l, gamma_i=self.solver.gamma_i),
                "1/(m**2*s)",
            )
        )

        return IonEnergyFlux(
            position=state.grid.z[0],
            species=params.species,
            energy_flux_toward_wall_watt_per_m2=np.asarray(energy, dtype=np.float64),
            particle_flux_toward_wall_per_m2_s=np.asarray(particles, dtype=np.float64),
            fidelity=self.solver.fidelity(),
            time=state.time,
        )

    def fidelity(self) -> Fidelity:
        return self.solver.fidelity()

    def metadata(self) -> SolverMetadata:
        """Name, version and citations.

        The version is ``ionlens-physics``'s, not this package's: the number that determines
        the result is the version of the code that computed it, and this adapter computes
        nothing. doc 08 §7's ``solver_versions`` is read by whoever is trying to regenerate
        an archived artifact, and pointing them at the wrapper would send them to the wrong
        changelog.
        """
        return SolverMetadata(
            name=SOLVER_NAME,
            version=_physics_version,
            citations=_CITATIONS,
            description=(
                "L0 analytic sheath — Child-Langmuir and matrix profiles, Bohm flux "
                "(doc 03 §2). The V-03 verification anchor."
            ),
        )
