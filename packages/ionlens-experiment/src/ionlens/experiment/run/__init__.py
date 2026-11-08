"""Running a manifest and verifying that it reproduces — doc 08 §6, doc 13.

Four modules:

- :mod:`~ionlens.experiment.run.store` — the directory layout and the run index (doc 13 §2).
- :mod:`~ionlens.experiment.run.record` — the doc 13 §2 record itself.
- :mod:`~ionlens.experiment.run.engine` — ``ionlens run``.
- :mod:`~ionlens.experiment.run.reproduce` — ``ionlens reproduce``, which is gate G-1.3.
"""

from ionlens.experiment.run.engine import (
    IMPLEMENTED_ARTIFACTS,
    METRICS_FILENAME,
    PLASMA_STATE_FILENAME,
    StageNotImplementedError,
    execute,
)
from ionlens.experiment.run.record import RunRecord, RunStatus, stream_seeds
from ionlens.experiment.run.reproduce import ReproductionResult, reproduce
from ionlens.experiment.run.store import (
    ARTIFACTS_DIRNAME,
    MANIFEST_FILENAME,
    PROVENANCE_FILENAME,
    RECORD_FILENAME,
    IndexEntry,
    RunDirectory,
    RunNotFoundError,
    RunStore,
    run_id_for,
)

__all__ = [
    "ARTIFACTS_DIRNAME",
    "IMPLEMENTED_ARTIFACTS",
    "MANIFEST_FILENAME",
    "METRICS_FILENAME",
    "PLASMA_STATE_FILENAME",
    "PROVENANCE_FILENAME",
    "RECORD_FILENAME",
    "IndexEntry",
    "ReproductionResult",
    "RunDirectory",
    "RunNotFoundError",
    "RunRecord",
    "RunStatus",
    "RunStore",
    "StageNotImplementedError",
    "execute",
    "reproduce",
    "run_id_for",
    "stream_seeds",
]
