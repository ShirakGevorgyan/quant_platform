"""Unit tests for the Milestone 10, Phase 3 addition to
`market_data.reconciliation`: `reconcile_historical_ingestion_operation`
cross-store integrity across the operation ledger, quarantine store,
provenance store, checkpoint store, and dataset manifest history."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from quant_platform.market_data.adapters import create_in_memory_adapter
from quant_platform.market_data.backfill import GapPolicy, OverlapPolicy, create_backfill_plan
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    PartitionGranularity,
    PartitioningSpec,
)
from quant_platform.market_data.mappings import InstrumentMappingEntry, create_instrument_mapping_spec
from quant_platform.market_data.orchestration import run_ingestion_operation
from quant_platform.market_data.reconciliation import reconcile_historical_ingestion_operation
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.market_data.source_manifests import (
    RecordKind,
    SourceKind,
    compute_timestamp_policy_id,
    create_source_manifest,
)
from quant_platform.market_data.source_normalization import TimestampParsingPolicy

_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="XAUUSD", provider="in_memory")
_PARTITIONING = PartitioningSpec(granularity=PartitionGranularity.DAILY)
_TIMESTAMP_POLICY = TimestampParsingPolicy(formats=("%Y-%m-%dT%H:%M:%S%z",), source_timezone=None)
_TIMEZONE_POLICY_ID = compute_timestamp_policy_id(_TIMESTAMP_POLICY)
_INSTRUMENT_MAPPING = create_instrument_mapping_spec(mapping_version=1, entries=(InstrumentMappingEntry(source_symbol="XAUUSD", instrument_id="XAUUSD"),))
_ROWS = [
    {"timestamp": "2024-01-01T00:00:00Z", "symbol": "XAUUSD", "price": "2000.5"},
    {"timestamp": "2024-01-01T00:01:00Z", "symbol": "XAUUSD", "price": "bad"},
]


def _run_operation(tmp_path: Path, *, operation_id: str = "recon_op") -> MarketDataRepository:
    repository = MarketDataRepository.open(tmp_path)
    adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=_ROWS)
    manifest = create_source_manifest(
        source_name="test", source_kind=SourceKind.IN_MEMORY, source_schema_version=1, record_kind=RecordKind.TICK,
        source_label="fixture", content_digest=adapter.content_digest(), byte_size=adapter.byte_size(), encoding="utf-8",
        instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id, timezone_policy_id=_TIMEZONE_POLICY_ID, unit_normalization_version=1, creation_time=_T0,
    )
    plan = create_backfill_plan(
        source_manifest=manifest, target_dataset_key=_KEY, requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        requested_end=datetime(2024, 1, 2, tzinfo=timezone.utc), existing_covered_partition_keys=frozenset(), partitioning=_PARTITIONING,
        overlap_policy=OverlapPolicy.REJECT_ANY_OVERLAP, gap_policy=GapPolicy.ALLOW_AND_REPORT, instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id,
        timeframe_mapping_id=None, timezone_policy_id=_TIMEZONE_POLICY_ID, creation_time=_T0,
    )
    run_ingestion_operation(
        repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
        timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id=operation_id, operation_time=_T0,
    )
    return repository


class TestReconcileHistoricalIngestionOperation:
    def test_clean_completed_operation_has_no_criticals(self, tmp_path: Path) -> None:
        repository = _run_operation(tmp_path)
        report = reconcile_historical_ingestion_operation(repository=repository, dataset_key=_KEY, operation_id="recon_op", generated_at=_T0.isoformat())
        assert report.criticals == ()

    def test_unknown_operation_id_reports_operation_not_found(self, tmp_path: Path) -> None:
        repository = _run_operation(tmp_path)
        report = reconcile_historical_ingestion_operation(repository=repository, dataset_key=_KEY, operation_id="does_not_exist", generated_at=_T0.isoformat())
        assert len(report.criticals) == 1
        assert report.criticals[0].code == "operation_not_found"

    def test_row_counts_reconcile(self, tmp_path: Path) -> None:
        repository = _run_operation(tmp_path)
        report = reconcile_historical_ingestion_operation(repository=repository, dataset_key=_KEY, operation_id="recon_op", generated_at=_T0.isoformat())
        codes = {i.code for i in report.issues}
        assert "row_count_mismatch" not in codes
        assert "provenance_count_mismatch" not in codes
        assert "quarantine_count_mismatch" not in codes

    def test_resulting_dataset_id_is_in_manifest_history(self, tmp_path: Path) -> None:
        repository = _run_operation(tmp_path)
        report = reconcile_historical_ingestion_operation(repository=repository, dataset_key=_KEY, operation_id="recon_op", generated_at=_T0.isoformat())
        assert not any(i.code == "resulting_dataset_id_not_in_manifest_history" for i in report.issues)

    def test_no_row_appears_in_both_quarantine_and_provenance(self, tmp_path: Path) -> None:
        repository = _run_operation(tmp_path)
        report = reconcile_historical_ingestion_operation(repository=repository, dataset_key=_KEY, operation_id="recon_op", generated_at=_T0.isoformat())
        assert not any(i.code == "row_both_quarantined_and_ingested" for i in report.issues)

    def test_historical_checkpoint_exists_for_completed_operation(self, tmp_path: Path) -> None:
        repository = _run_operation(tmp_path)
        report = reconcile_historical_ingestion_operation(repository=repository, dataset_key=_KEY, operation_id="recon_op", generated_at=_T0.isoformat())
        assert not any(i.code == "missing_historical_checkpoint" for i in report.issues)

    def test_two_independent_non_overlapping_operations_reconcile_independently(self, tmp_path: Path) -> None:
        # Two GENUINELY independent operations -- different source content
        # (different day), non-overlapping partitions -- rather than two
        # operations reprocessing the identical rows: THAT scenario is
        # correctly expected to fail closed (`ProvenanceError`) since a
        # fresh `operation_id` durably pins its OWN new sequence range,
        # so the SAME economic row would otherwise be silently
        # duplicated under a second event_id; a caller wanting a true
        # retry of the same rows must reuse the SAME operation_id (see
        # `TestStageMachineProgression::test_exact_retry_is_idempotent...`
        # in `test_market_data_orchestration.py`).
        repository = MarketDataRepository.open(tmp_path)
        day1_rows = [{"timestamp": "2024-01-01T00:00:00Z", "symbol": "XAUUSD", "price": "2000.5"}]
        day2_rows = [{"timestamp": "2024-01-02T00:00:00Z", "symbol": "XAUUSD", "price": "2010.0"}]

        def _run_day(rows: list[dict[str, str]], operation_id: str, start: datetime, end: datetime) -> None:
            adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=rows)
            manifest = create_source_manifest(
                source_name="test", source_kind=SourceKind.IN_MEMORY, source_schema_version=1, record_kind=RecordKind.TICK,
                source_label=f"fixture-{operation_id}", content_digest=adapter.content_digest(), byte_size=adapter.byte_size(), encoding="utf-8",
                instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id, timezone_policy_id=_TIMEZONE_POLICY_ID, unit_normalization_version=1, creation_time=_T0,
            )
            existing = frozenset(repository.partition_store.list_partition_keys(_KEY))
            plan = create_backfill_plan(
                source_manifest=manifest, target_dataset_key=_KEY, requested_start=start, requested_end=end, existing_covered_partition_keys=existing,
                partitioning=_PARTITIONING, overlap_policy=OverlapPolicy.REJECT_ANY_OVERLAP, gap_policy=GapPolicy.ALLOW_AND_REPORT,
                instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id, timeframe_mapping_id=None, timezone_policy_id=_TIMEZONE_POLICY_ID, creation_time=_T0,
            )
            run_ingestion_operation(
                repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
                timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id=operation_id, operation_time=_T0,
            )

        _run_day(day1_rows, "first", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc))
        _run_day(day2_rows, "second", datetime(2024, 1, 2, tzinfo=timezone.utc), datetime(2024, 1, 3, tzinfo=timezone.utc))

        report_first = reconcile_historical_ingestion_operation(repository=repository, dataset_key=_KEY, operation_id="first", generated_at=_T0.isoformat())
        report_second = reconcile_historical_ingestion_operation(repository=repository, dataset_key=_KEY, operation_id="second", generated_at=_T0.isoformat())
        assert report_first.criticals == ()
        assert report_second.criticals == ()
        assert len(repository.event_store.read_events("in_memory", "XAUUSD")) == 2

    def test_reprocessing_the_same_rows_under_a_new_operation_id_fails_closed(self, tmp_path: Path) -> None:
        # The complementary case: NOT reusing the same operation_id for
        # a genuine retry of already-committed rows is correctly
        # rejected rather than silently duplicating the economic event.
        from quant_platform.core.exceptions import ProvenanceError

        repository = _run_operation(tmp_path, operation_id="first")
        adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=_ROWS)
        manifest = create_source_manifest(
            source_name="test", source_kind=SourceKind.IN_MEMORY, source_schema_version=1, record_kind=RecordKind.TICK,
            source_label="fixture", content_digest=adapter.content_digest(), byte_size=adapter.byte_size(), encoding="utf-8",
            instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id, timezone_policy_id=_TIMEZONE_POLICY_ID, unit_normalization_version=1, creation_time=_T0,
        )
        plan = create_backfill_plan(
            source_manifest=manifest, target_dataset_key=_KEY, requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            requested_end=datetime(2024, 1, 2, tzinfo=timezone.utc), existing_covered_partition_keys=frozenset(), partitioning=_PARTITIONING,
            overlap_policy=OverlapPolicy.ALLOW_LATE_ARRIVAL_NEW_VERSION, gap_policy=GapPolicy.ALLOW_AND_REPORT, instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id,
            timeframe_mapping_id=None, timezone_policy_id=_TIMEZONE_POLICY_ID, creation_time=_T0,
        )
        with pytest.raises(ProvenanceError):
            run_ingestion_operation(
                repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
                timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id="second", operation_time=_T0,
            )
        # No silent duplicate was created.
        assert len(repository.event_store.read_events("in_memory", "XAUUSD")) == 1
