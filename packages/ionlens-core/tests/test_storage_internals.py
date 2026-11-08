"""The refusal paths inside the storage layer.

Every branch here exists because some *other* tool could have produced the file being
read, or because a caller could hand this layer something it cannot represent. doc 13 §5
retains reduced artifacts forever and doc 13 §6 re-reads archived manifests nightly, so
"a file this package did not write" is an ordinary condition rather than a hypothetical —
and guessing at its layout is how a plausible but wrong array reaches a figure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr
from numpy.typing import NDArray

from ionlens.core.provenance import EnvironmentLockSource, Provenance, Tier
from ionlens.core.state import ParameterLevel, Posterior, SamplerDiagnostics
from ionlens.core.storage import PosteriorSummary, read_posterior, write_posterior
from ionlens.core.storage.hdf5 import write_quantity
from ionlens.core.storage.metadata import provenance_from_attributes
from ionlens.core.units import Q_

_NAMES = ("V_w",)


@pytest.fixture
def record() -> Provenance:
    return Provenance(
        manifest_sha256="4a7f2e91" + "0" * 56,
        git_commit="9c1d8b3" + "0" * 33,
        git_dirty=False,
        seed=20260804,
        environment_lock_hash="e81c" + "f" * 60,
        environment_lock_source=EnvironmentLockSource.UV_LOCK,
        created_utc=datetime(2026, 8, 4, 12, 30, 15, 123456, tzinfo=UTC),
        ionlens_version="0.1.0",
        solver_versions={"numpyro": "0.15"},
        tier=Tier.T2,
    )


def _samples(seed: int) -> NDArray[np.float64]:
    return np.random.default_rng(seed).standard_normal((4, 1000, 1)) * np.pi


def _posterior(seed: int = 0) -> Posterior:
    return Posterior(
        samples=_samples(seed),
        names=_NAMES,
        levels={"V_w": ParameterLevel.CONTROL},
        tier=Tier.T2,
        diagnostics=SamplerDiagnostics(r_hat={"V_w": 1.001}, ess={"V_w": 3200.0}, divergences=0),
    )


class TestQuantitiesThatCannotBeStoredAsScalars:
    def test_an_array_valued_quantity_is_refused(self, tmp_path: Path) -> None:
        with (
            h5py.File(tmp_path / "x.h5", "w") as handle,
            pytest.raises(ValueError, match="scalar"),
        ):
            write_quantity(handle, "n_0", Q_(np.ones(3), "m**-3"))


class TestAttributesWrittenByAnotherTool:
    def test_a_provenance_field_stored_as_a_utf_8_blob_still_reads(self) -> None:
        record = Provenance(
            manifest_sha256="4a7f2e91" + "0" * 56,
            git_commit="9c1d8b3" + "0" * 33,
            git_dirty=False,
            seed=20260804,
            environment_lock_hash="e81c" + "f" * 60,
            environment_lock_source=EnvironmentLockSource.UV_LOCK,
            created_utc=datetime(2026, 8, 4, tzinfo=UTC),
            ionlens_version="0.1.0",
            solver_versions={"petsc": "3.20"},
            tier=Tier.T2,
        )
        attributes: dict[str, object] = {
            key: value.encode("utf-8") if isinstance(value, str) else value
            for key, value in record.to_dict().items()
        }
        attributes["solver_versions"] = b'{"petsc": "3.20"}'

        assert provenance_from_attributes(attributes) == record


class TestPosteriorSummaryInvariants:
    def test_a_mean_without_a_credible_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="credible interval"):
            PosteriorSummary(
                mean={"V_w": np.asarray(1.0)},
                credible={},
                diagnostics=SamplerDiagnostics(
                    r_hat={"V_w": 1.0}, ess={"V_w": 500.0}, divergences=0
                ),
                n_draws=1000,
            )

    def test_two_archives_of_the_same_posterior_summarise_identically(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        posterior = _posterior()
        write_posterior(tmp_path / "a.zarr", posterior, provenance=record)
        write_posterior(tmp_path / "b.zarr", posterior, provenance=record)

        assert read_posterior(tmp_path / "a.zarr").summary == (
            read_posterior(tmp_path / "b.zarr").summary
        )

    def test_archives_of_different_posteriors_do_not_summarise_identically(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        write_posterior(tmp_path / "a.zarr", _posterior(seed=0), provenance=record)
        write_posterior(tmp_path / "b.zarr", _posterior(seed=1), provenance=record)

        assert read_posterior(tmp_path / "a.zarr").summary != (
            read_posterior(tmp_path / "b.zarr").summary
        )

    def test_a_summary_is_not_equal_to_an_unrelated_object(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        write_posterior(tmp_path / "a.zarr", _posterior(), provenance=record)

        assert read_posterior(tmp_path / "a.zarr").summary != "a summary"

    def test_a_summary_says_what_it_covers(self, tmp_path: Path, record: Provenance) -> None:
        write_posterior(tmp_path / "a.zarr", _posterior(), provenance=record)

        rendered = repr(read_posterior(tmp_path / "a.zarr").summary)

        assert "PosteriorSummary" in rendered
        assert "n_draws=1000" in rendered


class TestAStoreThisPackageDidNotWrite:
    def test_a_derived_member_that_is_an_array_rather_than_a_group_is_refused(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        write_posterior(tmp_path / "post.zarr", _posterior(), provenance=record)
        group = zarr.open_group(store=tmp_path / "post.zarr", mode="a")
        del group["derived"]
        group.create_array("derived", shape=(2,), chunks=(2,), dtype="float64")

        with pytest.raises(ValueError, match="derived"):
            read_posterior(tmp_path / "post.zarr")

    def test_a_samples_member_that_is_a_group_rather_than_an_array_is_refused(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        write_posterior(tmp_path / "post.zarr", _posterior(), provenance=record)
        group = zarr.open_group(store=tmp_path / "post.zarr", mode="a")
        del group["samples"]
        group.create_group("samples")

        with pytest.raises(ValueError, match="samples"):
            read_posterior(tmp_path / "post.zarr")
