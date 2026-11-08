"""The experiment manifest — doc 08 §6.

    One file, one experiment, fully reproducible.

Three modules, in the order a manifest passes through them:

- :mod:`~ionlens.experiment.manifest.parse` — strict block reading; unknown keys raise.
- :mod:`~ionlens.experiment.manifest.schema` — the blocks doc 08 §6 defines, and the doc 05 §7
  consistency checks between them.
- :mod:`~ionlens.experiment.manifest.load` — the OmegaConf front end (ADR-008).

and one that turns the ``plasma:`` block into what a solver takes:
:mod:`~ionlens.experiment.manifest.plasma`.
"""

from ionlens.experiment.manifest.load import load_manifest
from ionlens.experiment.manifest.parse import UnknownKeyError
from ionlens.experiment.manifest.plasma import MAXWELLIAN_KAPPA, ResolvedPlasma, resolve_plasma
from ionlens.experiment.manifest.schema import (
    ArtifactRequest,
    BiasMode,
    BiasSpec,
    CalibrationMode,
    ExperimentSpec,
    ForwardSpec,
    InstrumentSpec,
    InverseSpec,
    Manifest,
    ManifestConsistencyError,
    NoiseSpec,
    OutputSpec,
    PlasmaSpec,
    ValidationSpec,
    manifest_from_document,
)

__all__ = [
    "MAXWELLIAN_KAPPA",
    "ArtifactRequest",
    "BiasMode",
    "BiasSpec",
    "CalibrationMode",
    "ExperimentSpec",
    "ForwardSpec",
    "InstrumentSpec",
    "InverseSpec",
    "Manifest",
    "ManifestConsistencyError",
    "NoiseSpec",
    "OutputSpec",
    "PlasmaSpec",
    "ResolvedPlasma",
    "UnknownKeyError",
    "ValidationSpec",
    "load_manifest",
    "manifest_from_document",
    "resolve_plasma",
]
