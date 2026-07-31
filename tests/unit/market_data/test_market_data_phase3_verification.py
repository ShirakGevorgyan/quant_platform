"""Unit tests for the Milestone 10, Phase 3 additions to
`market_data.verification`: independent identity recomputation for
`SourceManifest`/`BackfillPlan`/`ProvenanceRecord`/`QuarantineRecord`/
`HistoricalIngestionCheckpoint`, and cross-record conflict detection."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

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
from quant_platform.market_data.provenance import ProvenanceStore
from quant_platform.market_data.quarantine import QuarantineStore
from quant_platform.market_data.repository import MarketDataRepository
from quant_platform.market_data.source_manifests import (
    RecordKind,
    SourceKind,
    compute_timestamp_policy_id,
    create_source_manifest,
)
from quant_platform.market_data.source_normalization import TimestampParsingPolicy
from quant_platform.market_data.verification import (
    verify_backfill_plan,
    verify_historical_ingestion_checkpoint,
    verify_provenance_store,
    verify_quarantine_store,
    verify_source_manifest,
)

_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="XAUUSD", provider="in_memory")
_PARTITIONING = PartitioningSpec(granularity=PartitionGranularity.DAILY)
_TIMESTAMP_POLICY = TimestampParsingPolicy(formats=("%Y-%m-%dT%H:%M:%S%z",), source_timezone=None)
_TIMEZONE_POLICY_ID = compute_timestamp_policy_id(_TIMESTAMP_POLICY)
_INSTRUMENT_MAPPING = create_instrument_mapping_spec(mapping_version=1, entries=(InstrumentMappingEntry(source_symbol="XAUUSD", instrument_id="XAUUSD"),))
_ROWS = [{"timestamp": "2024-01-01T00:00:00Z", "symbol": "XAUUSD", "price": "2000.5"}]


def _manifest() -> object:
    adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=_ROWS)
    return create_source_manifest(
        source_name="test", source_kind=SourceKind.IN_MEMORY, source_schema_version=1, record_kind=RecordKind.TICK,
        source_label="fixture", content_digest=adapter.content_digest(), byte_size=adapter.byte_size(), encoding="utf-8",
        instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id, timezone_policy_id=_TIMEZONE_POLICY_ID, unit_normalization_version=1, creation_time=_T0,
    )


def _plan(manifest: object) -> object:
    return create_backfill_plan(
        source_manifest=manifest, target_dataset_key=_KEY, requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        requested_end=datetime(2024, 1, 2, tzinfo=timezone.utc), existing_covered_partition_keys=frozenset(), partitioning=_PARTITIONING,
        overlap_policy=OverlapPolicy.REJECT_ANY_OVERLAP, gap_policy=GapPolicy.ALLOW_AND_REPORT, instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id,
        timeframe_mapping_id=None, timezone_policy_id=_TIMEZONE_POLICY_ID, creation_time=_T0,
    )


class TestVerifySourceManifest:
    def test_clean_manifest_has_no_criticals(self) -> None:
        report = verify_source_manifest(_manifest(), as_of=_T0)
        assert report.criticals == ()

    def test_forged_id_detected(self) -> None:
        manifest = _manifest()
        forged = replace(manifest, source_manifest_id="0" * 64)
        report = verify_source_manifest(forged, as_of=_T0)
        assert any(i.code == "forged_source_manifest_identity" for i in report.criticals)


class TestVerifyBackfillPlan:
    def test_clean_plan_has_no_criticals(self) -> None:
        plan = _plan(_manifest())
        report = verify_backfill_plan(plan, as_of=_T0)
        assert report.criticals == ()

    def test_forged_id_detected(self) -> None:
        plan = _plan(_manifest())
        forged = replace(plan, backfill_plan_id="0" * 64)
        report = verify_backfill_plan(forged, as_of=_T0)
        assert any(i.code == "forged_backfill_plan_identity" for i in report.criticals)


class TestVerifyProvenanceAndQuarantineStores:
    def _run_operation(self, tmp_path: Path) -> MarketDataRepository:
        repository = MarketDataRepository.open(tmp_path)
        manifest = _manifest()
        plan = _plan(manifest)
        adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=_ROWS)
        run_ingestion_operation(
            repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
            timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op1", operation_time=_T0,
        )
        return repository

    def test_clean_provenance_store_has_no_criticals(self, tmp_path: Path) -> None:
        repository = self._run_operation(tmp_path)
        report = verify_provenance_store(provenance_store=ProvenanceStore(tmp_path), dataset_key=_KEY, repository=repository, as_of=_T0)
        assert report.criticals == ()

    def test_forged_provenance_record_detected(self, tmp_path: Path) -> None:
        repository = self._run_operation(tmp_path)
        store = ProvenanceStore(tmp_path)
        records = store.read_all(_KEY)
        forged = replace(records[0], provenance_id="0" * 64)
        report = verify_provenance_store(provenance_store=_FakeProvenanceStore([forged]), dataset_key=_KEY, repository=repository, as_of=_T0)
        assert any(i.code == "forged_provenance_identity" for i in report.criticals)

    def test_provenance_referencing_missing_event_detected(self, tmp_path: Path) -> None:
        repository = self._run_operation(tmp_path)
        store = ProvenanceStore(tmp_path)
        records = store.read_all(_KEY)
        from quant_platform.market_data.provenance import create_provenance_record

        dangling = create_provenance_record(
            source_manifest_id=records[0].source_manifest_id, source_row_index=99, source_record_digest="f" * 64,
            original_timestamp_text="x", normalized_event_time=_T0, instrument_mapping_id=records[0].instrument_mapping_id,
            resolved_instrument_id="XAUUSD", timeframe_mapping_id=None, timezone_policy_id=_TIMEZONE_POLICY_ID,
            ingestion_batch_id="op1", event_id="event_that_does_not_exist", dataset_id=records[0].dataset_id, recorded_time=_T0,
        )
        report = verify_provenance_store(provenance_store=_FakeProvenanceStore([*records, dangling]), dataset_key=_KEY, repository=repository, as_of=_T0)
        assert any(i.code == "provenance_references_missing_event" for i in report.criticals)

    def test_clean_quarantine_store_has_no_criticals(self, tmp_path: Path) -> None:
        self._run_operation(tmp_path)
        report = verify_quarantine_store(quarantine_store=QuarantineStore(tmp_path), dataset_key=_KEY, as_of=_T0)
        assert report.criticals == ()


class TestVerifyHistoricalIngestionCheckpoint:
    def test_checkpoint_still_valid_immediately_after_its_own_operation(self, tmp_path: Path) -> None:
        from quant_platform.market_data.checkpoints import CheckpointStore, HistoricalIngestionCheckpoint

        repository = MarketDataRepository.open(tmp_path)
        manifest = _manifest()
        plan = _plan(manifest)
        adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=_ROWS)
        run_ingestion_operation(
            repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
            timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op1", operation_time=_T0,
        )
        checkpoint = next(c for c in CheckpointStore(tmp_path).read_history(_KEY) if isinstance(c, HistoricalIngestionCheckpoint))
        report = verify_historical_ingestion_checkpoint(checkpoint, repository=repository)
        assert report.criticals == ()

    def test_checkpoint_still_valid_after_unrelated_later_activity(self, tmp_path: Path) -> None:
        # Regression: a HistoricalIngestionCheckpoint references a
        # SPECIFIC point-in-time RawIngestionCheckpoint. Once a LATER,
        # unrelated operation advances the repository further, that
        # referenced checkpoint is legitimately "stale" relative to
        # CURRENT live state -- but that is expected, not a defect, and
        # must never be reported as one (the earlier, buggy
        # implementation called `verify_raw_ingestion_checkpoint`, which
        # checks against CURRENT state and would incorrectly flag this).
        from quant_platform.market_data.checkpoints import CheckpointStore, HistoricalIngestionCheckpoint

        repository = MarketDataRepository.open(tmp_path)
        manifest = _manifest()
        plan = _plan(manifest)
        adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=_ROWS)
        run_ingestion_operation(
            repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
            timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op1", operation_time=_T0,
        )
        checkpoint = next(c for c in CheckpointStore(tmp_path).read_history(_KEY) if isinstance(c, HistoricalIngestionCheckpoint))

        # Unrelated later activity against a DIFFERENT dataset_key does not
        # move THIS dataset's repository state -- use a second row set
        # against the SAME dataset_key via a non-overlapping day instead,
        # to genuinely advance the repository past this checkpoint.
        later_rows = [{"timestamp": "2024-01-02T00:00:00Z", "symbol": "XAUUSD", "price": "2010.0"}]
        later_adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.TICK, rows=later_rows)
        later_manifest = create_source_manifest(
            source_name="test", source_kind=SourceKind.IN_MEMORY, source_schema_version=1, record_kind=RecordKind.TICK,
            source_label="fixture-day2", content_digest=later_adapter.content_digest(), byte_size=later_adapter.byte_size(), encoding="utf-8",
            instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id, timezone_policy_id=_TIMEZONE_POLICY_ID, unit_normalization_version=1, creation_time=_T0,
        )
        later_plan = create_backfill_plan(
            source_manifest=later_manifest, target_dataset_key=_KEY, requested_start=datetime(2024, 1, 2, tzinfo=timezone.utc),
            requested_end=datetime(2024, 1, 3, tzinfo=timezone.utc), existing_covered_partition_keys=frozenset(repository.partition_store.list_partition_keys(_KEY)),
            partitioning=_PARTITIONING, overlap_policy=OverlapPolicy.REJECT_ANY_OVERLAP, gap_policy=GapPolicy.ALLOW_AND_REPORT,
            instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id, timeframe_mapping_id=None, timezone_policy_id=_TIMEZONE_POLICY_ID, creation_time=_T0,
        )
        run_ingestion_operation(
            repository=repository, adapter=later_adapter, source_manifest=later_manifest, backfill_plan=later_plan, instrument_mapping=_INSTRUMENT_MAPPING,
            timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op2_later", operation_time=_T0,
        )

        report = verify_historical_ingestion_checkpoint(checkpoint, repository=repository)
        assert report.criticals == ()
        assert not any(i.code == "stale_repository_checkpoint" for i in report.issues)


class _FakeProvenanceStore:
    """A read-only stand-in exposing exactly the `read_all` shape
    `verify_provenance_store` needs, so adversarial in-memory record
    sets (never actually durably written) can be verified in isolation."""

    def __init__(self, records: list) -> None:
        self._records = records

    def read_all(self, _dataset_key: DatasetKey) -> list:
        return self._records
