"""Per-stream random number generation — doc 10 §5.

    Seeds are per-stream, not global. A change to the noise model must not perturb the
    plasma solve's random sequence.

    Per-stream seeding is a small decision with large consequences. With a single global
    RNG, adding one noise source shifts every subsequent random draw, and two runs that
    should be comparable are not. Separate streams for plasma initialisation, collisions,
    photon statistics, detector noise and sampler proposals make the runs independently
    perturbable — which the ablation matrix (doc 07 §5.2) requires.

That last clause is the whole justification. The ablation matrix switches noise sources on
and off one at a time and compares the results; doc 07 §5.2's F-01 through F-19 are all of
this form. If switching one source off shifted the plasma solve's draws, the comparison
would be measuring the random number generator.

## Why seeds are derived from the stream *name*

A stream's seed is a keyed hash of its name, not a spawn index. If streams were numbered
by declaration order, inserting a new one would renumber every stream after it and
silently change every run that used them — which is the exact failure doc 10 §5 forbids,
reintroduced by the mechanism meant to prevent it. Deriving from the name means a new
stream is genuinely new and disturbs nothing.

Python's built-in :func:`hash` is salted per process and cannot be used: doc 00 E3 promises
bit-for-bit reproducibility, and a seed that changed between interpreter launches would
make that unverifiable. BLAKE2b is used instead, and the derivation is pinned by a
regression test carrying recorded values.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Final

import numpy as np

__all__ = ["Stream", "generator", "stream_seed"]


class Stream(StrEnum):
    """The independent random streams a run may draw from — doc 10 §5.

    Enumerated rather than free-form. A typo'd stream name would silently create an
    independent stream that nothing else ever uses, the run would finish, and the
    resulting artifact would be reproducible but wrong.
    """

    PLASMA_INIT = "plasma_init"
    COLLISIONS = "collisions"
    PHOTONS = "photons"
    DETECTOR_NOISE = "detector_noise"
    CALIBRATION = "calibration"
    SAMPLER = "sampler"
    EXPERIMENT_DESIGN = "experiment_design"
    SUBSAMPLING = "subsampling"
    MAP_MULTISTART = "map_multistart"


_DOMAIN: Final[bytes] = b"ionlens/stream-seed/v1"

_SEED_BYTES: Final[int] = 8


def stream_seed(root_seed: int, stream: Stream) -> int:
    """Derive a stream's seed from the run's root seed.

    Args:
        root_seed: The run's single recorded seed. doc 08 §7 stores exactly one per
            artifact; every stream derives from it, so the artifact stays honest about
            what determines the run.
        stream: Which stream.

    Returns:
        A 64-bit seed, stable across processes and interpreter versions.

    Raises:
        ValueError: If ``stream`` is not a registered :class:`Stream`, or ``root_seed``
            is negative.
    """
    if not isinstance(stream, Stream):
        raise ValueError(
            f"unknown stream {stream!r}. Streams are enumerated (doc 10 §5) because a "
            f"typo would silently create an independent stream nothing else uses: "
            f"{', '.join(s.value for s in Stream)}"
        )
    if root_seed < 0:
        raise ValueError(f"root seed must be non-negative, got {root_seed}")

    digest = hashlib.blake2b(
        f"{root_seed}:{stream.value}".encode(),
        digest_size=_SEED_BYTES,
        person=_DOMAIN[:16],
    ).digest()
    return int.from_bytes(digest, "big")


def generator(root_seed: int, stream: Stream) -> np.random.Generator:
    """A generator for one stream, independent of every other stream.

    Independent means what it says: exhausting the detector-noise stream leaves the
    plasma solve's sequence untouched, because they are separate objects seeded from
    separate derived seeds.
    """
    return np.random.default_rng(np.random.SeedSequence(stream_seed(root_seed, stream)))
