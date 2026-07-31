"""Unit tests for `market_data.provenance` (Milestone 10, Phase 3):
bidirectional row<->event provenance identity, idempotent append,
conflicting-append fail-closed, and cross-record conflict detection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from quant_platform.core.exceptions import ProvenanceError
from quant_platform.market_data.adapters import SourceRowCoordinate
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.provenance import (
    ProvenanceRecord,
    ProvenanceStore,
    create_provenance_record,
    find_provenance_conflicts,
)

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="XAUUSD", provider="offline_csv")


def _record(*, row_index: int = 0, event_id: str = "event_a", dataset_id: str = "ds1", source_manifest_id: str = "a" * 64) -> ProvenanceRecord:
    return create_provenance_record(
        source_manifest_id=source_manifest_id, source_row_index=row_index, source_record_digest="b" * 64,
        original_timestamp_text="2024-01-01T00:00:00Z", normalized_event_time=_T0, instrument_mapping_id="c" * 64,
        resolved_instrument_id="XAUUSD", timeframe_mapping_id="d" * 64, timezone_policy_id="e" * 64,
        ingestion_batch_id="batch1", event_id=event_id, dataset_id=dataset_id, recorded_time=_T0,
    )


class TestProvenanceRecordIdentity:
    def test_deterministic(self) -> None:
        assert _record().provenance_id == _record().provenance_id

    def test_recorded_time_excluded_from_identity(self) -> None:
        r1 = create_provenance_record(
            source_manifest_id="a" * 64, source_row_index=0, source_record_digest="b" * 64, original_timestamp_text="x",
            normalized_event_time=_T0, instrument_mapping_id="c" * 64, resolved_instrument_id="XAUUSD", timeframe_mapping_id=None,
            timezone_policy_id="e" * 64, ingestion_batch_id="batch1", event_id="event_a", dataset_id="ds1", recorded_time=_T0,
        )
        r2 = create_provenance_record(
            source_manifest_id="a" * 64, source_row_index=0, source_record_digest="b" * 64, original_timestamp_text="x",
            normalized_event_time=_T0, instrument_mapping_id="c" * 64, resolved_instrument_id="XAUUSD", timeframe_mapping_id=None,
            timezone_policy_id="e" * 64, ingestion_batch_id="batch1", event_id="event_a", dataset_id="ds1", recorded_time=_T0.replace(year=2030),
        )
        assert r1.provenance_id == r2.provenance_id

    def test_changed_event_id_changes_provenance_id(self) -> None:
        assert _record(event_id="event_a").provenance_id != _record(event_id="event_b").provenance_id

    def test_dataset_id_excluded_from_identity(self) -> None:
        # Regression: `dataset_id` is a repository-state SNAPSHOT that can
        # legitimately drift for reasons unrelated to this row (other data
        # landing in the same dataset between an original attempt and a
        # later idempotent retry) -- it must not participate in identity,
        # or an otherwise-exact retry would spuriously conflict.
        r1 = _record(event_id="event_a", dataset_id="ds1")
        r2 = _record(event_id="event_a", dataset_id="ds2_after_unrelated_ingestion")
        assert r1.provenance_id == r2.provenance_id

    def test_negative_row_index_rejected(self) -> None:
        with pytest.raises(ProvenanceError):
            create_provenance_record(
                source_manifest_id="a" * 64, source_row_index=-1, source_record_digest="b" * 64, original_timestamp_text="x",
                normalized_event_time=_T0, instrument_mapping_id="c" * 64, resolved_instrument_id="XAUUSD", timeframe_mapping_id=None,
                timezone_policy_id="e" * 64, ingestion_batch_id="batch1", event_id="event_a", dataset_id="ds1", recorded_time=_T0,
            )

    def test_round_trip(self) -> None:
        record = _record()
        assert ProvenanceRecord.from_json_dict(record.to_json_dict()) == record

    def test_source_coordinate(self) -> None:
        record = _record(row_index=5)
        assert record.source_coordinate() == SourceRowCoordinate(source_manifest_id="a" * 64, row_index=5)


class TestProvenanceStore:
    def test_append_and_read(self, tmp_path: Path) -> None:
        store = ProvenanceStore(tmp_path)
        record = _record()
        store.append(_KEY, record)
        assert store.read_all(_KEY) == [record]

    def test_exact_retry_idempotent(self, tmp_path: Path) -> None:
        store = ProvenanceStore(tmp_path)
        record = _record()
        store.append(_KEY, record)
        store.append(_KEY, _record())
        assert len(store.read_all(_KEY)) == 1

    def test_conflicting_retry_fails_closed(self, tmp_path: Path) -> None:
        store = ProvenanceStore(tmp_path)
        store.append(_KEY, _record(event_id="event_a"))
        with pytest.raises(ProvenanceError):
            store.append(_KEY, _record(event_id="event_DIFFERENT"))
        assert len(store.read_all(_KEY)) == 1

    def test_read_by_source_coordinate(self, tmp_path: Path) -> None:
        store = ProvenanceStore(tmp_path)
        record = _record(row_index=3)
        store.append(_KEY, record)
        found = store.read_by_source_coordinate(_KEY, SourceRowCoordinate(source_manifest_id="a" * 64, row_index=3))
        assert found == record
        assert store.read_by_source_coordinate(_KEY, SourceRowCoordinate(source_manifest_id="a" * 64, row_index=999)) is None

    def test_read_by_event_id(self, tmp_path: Path) -> None:
        store = ProvenanceStore(tmp_path)
        store.append(_KEY, _record(row_index=0, event_id="shared"))
        store.append(_KEY, _record(row_index=1, event_id="shared"))
        assert len(store.read_by_event_id(_KEY, "shared")) == 2

    def test_different_rows_do_not_conflict(self, tmp_path: Path) -> None:
        store = ProvenanceStore(tmp_path)
        store.append(_KEY, _record(row_index=0, event_id="event_a"))
        store.append(_KEY, _record(row_index=1, event_id="event_b"))
        assert len(store.read_all(_KEY)) == 2


class TestFindProvenanceConflicts:
    def test_clean_state_has_no_conflicts(self) -> None:
        records = [_record(row_index=0, event_id="a"), _record(row_index=1, event_id="b")]
        assert find_provenance_conflicts(records) == ()

    def test_event_bound_to_multiple_source_rows_reported(self) -> None:
        # Legitimate exact-duplicate absorption: two different rows, same event.
        records = [_record(row_index=0, event_id="a"), _record(row_index=1, event_id="a")]
        conflicts = find_provenance_conflicts(records)
        assert len(conflicts) == 1
        assert conflicts[0].issue_code == "event_bound_to_multiple_source_rows"

    def test_coordinate_bound_to_multiple_events_reported(self) -> None:
        # Simulated corruption: same coordinate, two different events (the
        # durable store itself refuses this at append time; this proves
        # the detector still catches it in an already-corrupted read).
        records = [_record(row_index=0, event_id="a"), _record(row_index=0, event_id="b")]
        conflicts = find_provenance_conflicts(records)
        assert len(conflicts) == 1
        assert conflicts[0].issue_code == "coordinate_bound_to_multiple_events"
