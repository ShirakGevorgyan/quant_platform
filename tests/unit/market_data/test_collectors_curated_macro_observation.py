"""Curated macro observation model + dataset manifest tests (Milestone
10, Phase 4B)."""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

import pytest
from _curated_test_helpers import T0

from quant_platform.core.exceptions import CollectorError
from quant_platform.market_data.collectors.curated.datasets import (
    CombinedUniverseManifestStore,
    CompletenessStatus,
    ComponentDatasetManifestStore,
    create_combined_universe_manifest,
    create_component_dataset_manifest,
)
from quant_platform.market_data.collectors.curated.macro_observation import (
    CuratedObservationStore,
    create_curated_macro_observation,
)


def _obs(date: str, value: str | None, *, is_missing: bool = False, realtime_start: str | None = "2024-01-02", row: int = 0, series_id: str = "DGS10"):
    return create_curated_macro_observation(
        series_id=series_id, canonical_series_name="us_10y_nominal_yield", target_macro_instrument_id="us_10y_nominal_yield",
        observation_date=date, value=(None if value is None else Decimal(value)), is_missing=is_missing, normalized_unit="percent",
        native_unit="%", native_frequency="D", realtime_start=realtime_start, realtime_end="9999-12-31", availability_time=T0,
        availability_policy_id="a" * 64, request_manifest_id="b" * 64, response_manifest_id="c" * 64, source_manifest_id="d" * 64, source_row_index=row,
    )


class TestObservationValidation:
    def test_value_required_when_not_missing(self) -> None:
        with pytest.raises(CollectorError):
            _obs("2024-01-02", None, is_missing=False)

    def test_value_forbidden_when_missing(self) -> None:
        with pytest.raises(CollectorError):
            create_curated_macro_observation(
                series_id="DGS10", canonical_series_name="x", target_macro_instrument_id="x", observation_date="2024-01-02", value=Decimal("1"),
                is_missing=True, normalized_unit="percent", native_unit="%", native_frequency="D", realtime_start=None, realtime_end=None,
                availability_time=T0, availability_policy_id="a" * 64, request_manifest_id="b" * 64, response_manifest_id="c" * 64,
                source_manifest_id="d" * 64, source_row_index=0,
            )

    def test_negative_row_index_rejected(self) -> None:
        with pytest.raises(CollectorError):
            _obs("2024-01-02", "4.02", row=-1)


class TestObservationIdentity:
    def test_same_content_same_id(self) -> None:
        a = _obs("2024-01-02", "4.02")
        b = _obs("2024-01-02", "4.02")
        assert a.observation_id == b.observation_id

    def test_revision_produces_different_id(self) -> None:
        a = _obs("2024-01-02", "4.02", realtime_start="2024-01-02")
        b = _obs("2024-01-02", "4.05", realtime_start="2024-06-01")
        assert a.observation_id != b.observation_id

    def test_same_date_different_value_never_collapsed(self) -> None:
        """Two vintages of the same observation_date: identity must
        distinguish them even if a caller were to (incorrectly) treat
        them as "the same slot" -- they are two DIFFERENT durable facts."""
        a = _obs("2024-01-02", "4.02", realtime_start="2024-01-02")
        b = _obs("2024-01-02", "4.03", realtime_start="2024-01-03")
        assert a.observation_id != b.observation_id
        assert a.observation_date == b.observation_date


class TestObservationStore:
    def test_append_and_read(self) -> None:
        root = Path(tempfile.mkdtemp())
        store = CuratedObservationStore(root)
        a = _obs("2024-01-02", "4.02")
        store.append("fred", a)
        assert store.read_observations("fred", "DGS10") == [a]

    def test_exact_retry_is_idempotent(self) -> None:
        root = Path(tempfile.mkdtemp())
        store = CuratedObservationStore(root)
        a = _obs("2024-01-02", "4.02")
        store.append("fred", a)
        store.append("fred", a)
        assert len(store.read_observations("fred", "DGS10")) == 1

    def test_unknown_series_returns_empty(self) -> None:
        root = Path(tempfile.mkdtemp())
        store = CuratedObservationStore(root)
        assert store.read_observations("fred", "NOPE") == []


class TestComponentDatasetManifest:
    def test_coverage_and_counts(self) -> None:
        obs1 = _obs("2024-01-02", "4.02", row=0)
        obs2 = _obs("2024-01-03", "4.05", row=1)
        manifest = create_component_dataset_manifest(series_id="DGS10", canonical_series_name="us_10y_nominal_yield", observations=(obs1, obs2), missing_count=0, creation_time=T0)
        assert manifest.coverage_start == "2024-01-02"
        assert manifest.coverage_end == "2024-01-03"
        assert manifest.observation_count == 2
        assert manifest.revision_count == 0

    def test_revision_count_detects_multiple_vintages_same_date(self) -> None:
        obs1 = _obs("2024-01-02", "4.02", realtime_start="2024-01-02", row=0)
        obs2 = _obs("2024-01-02", "4.05", realtime_start="2024-06-01", row=1)
        manifest = create_component_dataset_manifest(series_id="DGS10", canonical_series_name="x", observations=(obs1, obs2), missing_count=0, creation_time=T0)
        assert manifest.revision_count == 1
        assert manifest.observation_count == 2

    def test_empty_observations(self) -> None:
        manifest = create_component_dataset_manifest(series_id="DGS10", canonical_series_name="x", observations=(), missing_count=0, creation_time=T0)
        assert manifest.coverage_start is None and manifest.coverage_end is None and manifest.observation_count == 0

    def test_mixed_frequency_component_manifests_are_independent(self) -> None:
        """Two DIFFERENT series' manifests must never be conflated into
        one physically regular time series -- each carries its own
        `native_frequency` untouched."""
        daily_obs = _obs("2024-01-02", "4.02", series_id="DGS10")
        monthly_manifest = create_component_dataset_manifest(series_id="CPIAUCSL", canonical_series_name="us_cpi_all_urban", observations=(), missing_count=0, creation_time=T0)
        daily_manifest = create_component_dataset_manifest(series_id="DGS10", canonical_series_name="us_10y_nominal_yield", observations=(daily_obs,), missing_count=0, creation_time=T0)
        assert daily_manifest.native_frequency == "D"
        assert monthly_manifest.series_id != daily_manifest.series_id


class TestComponentDatasetManifestStore:
    def test_versioning_increments_on_change(self) -> None:
        root = Path(tempfile.mkdtemp())
        store = ComponentDatasetManifestStore(root)
        m1 = create_component_dataset_manifest(series_id="DGS10", canonical_series_name="x", observations=(_obs("2024-01-02", "4.02"),), missing_count=0, creation_time=T0)
        _, v1 = store.append("fred", m1)
        assert v1 == 1
        m2 = create_component_dataset_manifest(series_id="DGS10", canonical_series_name="x", observations=(_obs("2024-01-02", "4.02"), _obs("2024-01-03", "4.05", row=1)), missing_count=0, creation_time=T0)
        _, v2 = store.append("fred", m2)
        assert v2 == 2

    def test_exact_no_op_update_mints_no_new_version(self) -> None:
        root = Path(tempfile.mkdtemp())
        store = ComponentDatasetManifestStore(root)
        m = create_component_dataset_manifest(series_id="DGS10", canonical_series_name="x", observations=(_obs("2024-01-02", "4.02"),), missing_count=0, creation_time=T0)
        store.append("fred", m)
        _, v = store.append("fred", m)  # identical content again
        assert v == 1


class TestCombinedUniverseManifest:
    def test_binds_exact_component_versions(self) -> None:
        component = create_component_dataset_manifest(series_id="DGS10", canonical_series_name="x", observations=(_obs("2024-01-02", "4.02"),), missing_count=0, creation_time=T0)
        combined = create_combined_universe_manifest(
            curated_registry_id="a" * 64, backfill_plan_id="b" * 64, target_dataset_namespace="ns1", component_manifests={"DGS10": component},
            availability_policy_ids_by_series={"DGS10": "c" * 64}, revision_policy_id="d" * 64, completeness_status=CompletenessStatus.COMPLETE, creation_time=T0,
        )
        assert combined.component_manifest_ids["DGS10"] == component.component_manifest_id

    def test_completeness_status_validated(self) -> None:
        component = create_component_dataset_manifest(series_id="DGS10", canonical_series_name="x", observations=(), missing_count=0, creation_time=T0)
        with pytest.raises(Exception):  # noqa: B017 -- CombinedManifestError
            create_combined_universe_manifest(
                curated_registry_id="a" * 64, backfill_plan_id="b" * 64, target_dataset_namespace="ns1", component_manifests={"DGS10": component},
                availability_policy_ids_by_series={}, revision_policy_id="d" * 64, completeness_status="not_a_real_status", creation_time=T0,
            )


class TestCombinedUniverseManifestStore:
    def test_versioning_and_no_op(self) -> None:
        root = Path(tempfile.mkdtemp())
        component = create_component_dataset_manifest(series_id="DGS10", canonical_series_name="x", observations=(_obs("2024-01-02", "4.02"),), missing_count=0, creation_time=T0)
        combined = create_combined_universe_manifest(
            curated_registry_id="a" * 64, backfill_plan_id="b" * 64, target_dataset_namespace="ns1", component_manifests={"DGS10": component},
            availability_policy_ids_by_series={"DGS10": "c" * 64}, revision_policy_id="d" * 64, completeness_status=CompletenessStatus.COMPLETE, creation_time=T0,
        )
        store = CombinedUniverseManifestStore(root)
        _, v1 = store.append(combined)
        assert v1 == 1
        _, v2 = store.append(combined)
        assert v2 == 1  # no-op
