"""Unit tests for the Milestone 10, Phase 3 additions to
`market_data.reports`: source inspection, backfill plan, dry-run,
quarantine/provenance summaries, and replay comparison reports."""

from __future__ import annotations

import tempfile
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
from quant_platform.market_data.orchestration import replay_ingestion_operation, run_ingestion_operation
from quant_platform.market_data.provenance import ProvenanceStore
from quant_platform.market_data.quarantine import QuarantineStore
from quant_platform.market_data.reports import (
    generate_backfill_plan_report,
    generate_dry_run_report,
    generate_provenance_summary_report,
    generate_quarantine_summary_report,
    generate_replay_comparison_report,
    generate_source_inspection_report,
)
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


def _fixture() -> tuple:
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
    return adapter, manifest, plan


class TestSourceInspectionReport:
    def test_contains_identity_and_adapter_description(self) -> None:
        adapter, manifest, _plan = _fixture()
        report = generate_source_inspection_report(manifest, adapter_description=adapter.describe())
        assert report["source_manifest_id"] == manifest.source_manifest_id
        assert report["content_digest"] == manifest.content_digest


class TestBackfillPlanReport:
    def test_contains_plan_summary_fields(self) -> None:
        _adapter, _manifest, plan = _fixture()
        report = generate_backfill_plan_report(plan)
        assert report["backfill_plan_id"] == plan.backfill_plan_id
        assert report["is_admissible"] == plan.is_admissible
        assert report["batch_count"] == len(plan.batches)


class TestDryRunReport:
    def test_requires_dry_run_flag(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        adapter, manifest, plan = _fixture()
        real_report = run_ingestion_operation(
            repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
            timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op1", operation_time=_T0, dry_run=False,
        )
        with pytest.raises(ValueError, match="is_dry_run=True"):
            generate_dry_run_report(real_report)

    def test_dry_run_report_reflects_counts(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        adapter, manifest, plan = _fixture()
        dry_report = run_ingestion_operation(
            repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
            timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op1", operation_time=_T0, dry_run=True,
        )
        report = generate_dry_run_report(dry_report)
        assert report["valid_row_count"] == 1
        assert report["quarantined_row_count"] == 1


class TestQuarantineAndProvenanceSummaryReports:
    def _run(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        adapter, manifest, plan = _fixture()
        run_ingestion_operation(
            repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
            timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op1", operation_time=_T0,
        )

    def test_quarantine_summary_counts_issue_codes(self, tmp_path: Path) -> None:
        self._run(tmp_path)
        report = generate_quarantine_summary_report(quarantine_store=QuarantineStore(tmp_path), dataset_key=_KEY, operation_id="op1")
        assert report["total_quarantined"] == 1
        assert report["issue_counts"]["invalid_decimal"] == 1

    def test_provenance_summary_counts_records(self, tmp_path: Path) -> None:
        self._run(tmp_path)
        report = generate_provenance_summary_report(provenance_store=ProvenanceStore(tmp_path), dataset_key=_KEY, operation_id="op1")
        assert report["total_provenance_records"] == 1


class TestReplayComparisonReport:
    def test_identical_replay_reports_no_differences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            repo1 = MarketDataRepository.open(Path(tmp1))
            repo2 = MarketDataRepository.open(Path(tmp2))
            adapter, manifest, plan = _fixture()
            original = run_ingestion_operation(
                repository=repo1, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
                timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op1", operation_time=_T0,
            )
            replayed = replay_ingestion_operation(
                repository=repo2, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
                timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op1", operation_time=_T0,
            )
            comparison = generate_replay_comparison_report(original=original, replayed=replayed)
            assert comparison["identical"] is True
            assert comparison["differences"] == {}

    def test_different_operation_id_is_excluded_from_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            repo1 = MarketDataRepository.open(Path(tmp1))
            repo2 = MarketDataRepository.open(Path(tmp2))
            adapter, manifest, plan = _fixture()
            original = run_ingestion_operation(
                repository=repo1, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
                timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op_a", operation_time=_T0,
            )
            replayed = replay_ingestion_operation(
                repository=repo2, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
                timeframe_mapping=None, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op_b", operation_time=_T0,
            )
            # Different operation_id is legitimate (a fresh lineage) and
            # must not by itself register as a semantic difference.
            comparison = generate_replay_comparison_report(original=original, replayed=replayed)
            assert comparison["identical"] is True
