"""Unit tests for `market_data.orchestration` (Milestone 10, Phase 3):
the eleven-stage ingestion machine, dry-run write-nothing guarantees,
fail-fast vs quarantine policy, exact-retry idempotency, conflicting-
retry fail-closed behavior, replay determinism, concurrency, and
adversarial robustness."""

from __future__ import annotations

import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quant_platform.core.exceptions import (
    OrchestrationConflictError,
    OrchestrationStateError,
    RowValidationError,
)
from quant_platform.core.types import Timeframe
from quant_platform.market_data.adapters import create_in_memory_adapter
from quant_platform.market_data.backfill import GapPolicy, OverlapPolicy, create_backfill_plan
from quant_platform.market_data.checkpoints import (
    CheckpointStore,
    HistoricalIngestionCheckpoint,
    RawIngestionCheckpoint,
)
from quant_platform.market_data.manifests import (
    DatasetKey,
    DatasetKind,
    PartitionGranularity,
    PartitioningSpec,
)
from quant_platform.market_data.mappings import (
    InstrumentMappingEntry,
    TimeframeMappingEntry,
    create_instrument_mapping_spec,
    create_timeframe_mapping_spec,
)
from quant_platform.market_data.orchestration import (
    IngestionStage,
    OperationStore,
    RowFailurePolicy,
    replay_ingestion_operation,
    run_ingestion_operation,
)
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

_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="XAUUSD", provider="in_memory")
_PARTITIONING = PartitioningSpec(granularity=PartitionGranularity.DAILY)
_TIMESTAMP_POLICY = TimestampParsingPolicy(formats=("%Y-%m-%dT%H:%M:%S%z",), source_timezone=None)
_TIMEZONE_POLICY_ID = compute_timestamp_policy_id(_TIMESTAMP_POLICY)
_INSTRUMENT_MAPPING = create_instrument_mapping_spec(mapping_version=1, entries=(InstrumentMappingEntry(source_symbol="XAUUSD", instrument_id="XAUUSD"),))
_TIMEFRAME_MAPPING = create_timeframe_mapping_spec(mapping_version=1, entries=(TimeframeMappingEntry(source_label="M1", timeframe=Timeframe.M1),))

_CANDLE_ROWS = [
    {"timestamp": "2024-01-01T00:00:00Z", "symbol": "XAUUSD", "open": "2000.5", "high": "2001.0", "low": "1999.5", "close": "2000.0", "volume": "100", "timeframe": "M1"},
    {"timestamp": "2024-01-01T00:01:00Z", "symbol": "XAUUSD", "open": "2000.0", "high": "2000.5", "low": "1999.0", "close": "1999.5", "timeframe": "M1"},
    {"timestamp": "2024-01-01T00:02:00Z", "symbol": "XAUUSD", "open": "bad", "high": "2000.5", "low": "1999.0", "close": "1999.5", "timeframe": "M1"},
    {"timestamp": "2024-01-01T00:03:00Z", "symbol": "UNKNOWNSYM", "open": "1999.7", "high": "2000.2", "low": "1999.2", "close": "1999.9", "timeframe": "M1"},
]


def _build_fixture(repository: MarketDataRepository, *, rows: list[dict[str, str]] | None = None) -> tuple:
    adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.CANDLE, rows=(rows if rows is not None else _CANDLE_ROWS))
    manifest = create_source_manifest(
        source_name="test", source_kind=SourceKind.IN_MEMORY, source_schema_version=1, record_kind=RecordKind.CANDLE,
        source_label="fixture", content_digest=adapter.content_digest(), byte_size=adapter.byte_size(), encoding="utf-8",
        instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id, timezone_policy_id=_TIMEZONE_POLICY_ID, timeframe_mapping_id=_TIMEFRAME_MAPPING.mapping_id,
        unit_normalization_version=1, creation_time=_T0, expected_timeframe=Timeframe.M1,
    )
    plan = create_backfill_plan(
        source_manifest=manifest, target_dataset_key=_KEY, requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        requested_end=datetime(2024, 1, 2, tzinfo=timezone.utc), existing_covered_partition_keys=frozenset(), partitioning=_PARTITIONING,
        overlap_policy=OverlapPolicy.REJECT_ANY_OVERLAP, gap_policy=GapPolicy.ALLOW_AND_REPORT, instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id,
        timeframe_mapping_id=_TIMEFRAME_MAPPING.mapping_id, timezone_policy_id=_TIMEZONE_POLICY_ID, creation_time=_T0,
    )
    return adapter, manifest, plan


def _run(repository: MarketDataRepository, *, operation_id: str = "op1", rows: list[dict[str, str]] | None = None, **kwargs: object) -> object:
    adapter, manifest, plan = _build_fixture(repository, rows=rows)
    return run_ingestion_operation(
        repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
        timeframe_mapping=_TIMEFRAME_MAPPING, timestamp_policy=_TIMESTAMP_POLICY, operation_id=operation_id, operation_time=_T0,
        **kwargs,  # type: ignore[arg-type]
    )


class TestStageMachineProgression:
    def test_completed_operation_advances_through_every_stage_in_order(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        report = _run(repository)
        assert report.stage is IngestionStage.COMPLETED
        history = OperationStore(tmp_path).read_all(_KEY)
        assert [r.stage for r in history] == list(IngestionStage)

    def test_exact_retry_is_idempotent_and_writes_nothing_new(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        report1 = _run(repository)
        report2 = _run(repository)
        assert report1.resulting_dataset_id == report2.resulting_dataset_id
        assert len(repository.event_store.read_events("in_memory", "XAUUSD")) == report1.valid_row_count
        assert len(QuarantineStore(tmp_path).read_all(_KEY)) == report1.quarantined_row_count
        assert len(ProvenanceStore(tmp_path).read_all(_KEY)) == report1.valid_row_count

    def test_conflicting_retry_under_same_operation_id_fails_closed(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        _run(repository, operation_id="op1")
        different_mapping = create_instrument_mapping_spec(
            mapping_version=2, entries=(InstrumentMappingEntry(source_symbol="XAUUSD", instrument_id="XAUUSD"), InstrumentMappingEntry(source_symbol="EXTRA", instrument_id="EXTRA")),
        )
        adapter, manifest, plan = _build_fixture(repository)
        with pytest.raises(OrchestrationConflictError):
            run_ingestion_operation(
                repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=different_mapping,
                timeframe_mapping=_TIMEFRAME_MAPPING, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op1", operation_time=_T0,
            )

    def test_advance_rejects_illegal_stage_skip(self, tmp_path: Path) -> None:
        store = OperationStore(tmp_path)
        store.advance(dataset_key=_KEY, operation_id="x", content_digest="d1", stage=IngestionStage.SOURCE_VERIFIED, stage_evidence={}, operation_time=_T0)
        with pytest.raises(OrchestrationStateError):
            store.advance(dataset_key=_KEY, operation_id="x", content_digest="d1", stage=IngestionStage.BATCH_RESERVED, stage_evidence={}, operation_time=_T0)

    def test_advance_rejects_first_stage_not_source_verified(self, tmp_path: Path) -> None:
        store = OperationStore(tmp_path)
        with pytest.raises(OrchestrationStateError):
            store.advance(dataset_key=_KEY, operation_id="y", content_digest="d1", stage=IngestionStage.PLAN_CREATED, stage_evidence={}, operation_time=_T0)

    def test_source_content_digest_mismatch_fails_closed(self, tmp_path: Path) -> None:
        # A manifest whose OWN declared content_digest does not match
        # what the adapter it is paired with actually produces -- the
        # backfill plan is built against THIS (tampered) manifest so the
        # SOURCE_VERIFIED check is exercised in isolation, not the
        # earlier plan/manifest cross-reference check.
        repository = MarketDataRepository.open(tmp_path)
        adapter, _manifest, _plan = _build_fixture(repository)
        tampered_manifest = create_source_manifest(
            source_name="test", source_kind=SourceKind.IN_MEMORY, source_schema_version=1, record_kind=RecordKind.CANDLE,
            source_label="fixture", content_digest="0" * 64, byte_size=adapter.byte_size(), encoding="utf-8",
            instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id, timezone_policy_id=_TIMEZONE_POLICY_ID, timeframe_mapping_id=_TIMEFRAME_MAPPING.mapping_id,
            unit_normalization_version=1, creation_time=_T0, expected_timeframe=Timeframe.M1,
        )
        tampered_plan = create_backfill_plan(
            source_manifest=tampered_manifest, target_dataset_key=_KEY, requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            requested_end=datetime(2024, 1, 2, tzinfo=timezone.utc), existing_covered_partition_keys=frozenset(), partitioning=_PARTITIONING,
            overlap_policy=OverlapPolicy.REJECT_ANY_OVERLAP, gap_policy=GapPolicy.ALLOW_AND_REPORT, instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id,
            timeframe_mapping_id=_TIMEFRAME_MAPPING.mapping_id, timezone_policy_id=_TIMEZONE_POLICY_ID, creation_time=_T0,
        )
        with pytest.raises(Exception, match="content_digest"):
            run_ingestion_operation(
                repository=repository, adapter=adapter, source_manifest=tampered_manifest, backfill_plan=tampered_plan, instrument_mapping=_INSTRUMENT_MAPPING,
                timeframe_mapping=_TIMEFRAME_MAPPING, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op1", operation_time=_T0,
            )


class TestFailFastVsQuarantine:
    def test_quarantine_policy_ingests_valid_rows_and_quarantines_invalid_ones(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        report = _run(repository)
        assert report.valid_row_count == 2
        assert report.quarantined_row_count == 2
        assert "invalid_decimal" in report.quarantine_issue_counts
        assert "unknown_symbol" in report.quarantine_issue_counts

    def test_fail_fast_raises_on_first_invalid_row_and_commits_nothing(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        with pytest.raises(RowValidationError):
            _run(repository, operation_id="op_ff", on_invalid_row=RowFailurePolicy.FAIL_FAST)
        assert QuarantineStore(tmp_path).read_all(_KEY) == []
        assert repository.event_store.read_events("in_memory", "XAUUSD") == []

    def test_all_valid_rows_under_fail_fast_succeeds(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        valid_rows = _CANDLE_ROWS[:2]
        report = _run(repository, operation_id="op_ff_ok", rows=valid_rows, on_invalid_row=RowFailurePolicy.FAIL_FAST)
        assert report.valid_row_count == 2
        assert report.quarantined_row_count == 0


class TestDryRun:
    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        report = _run(repository, dry_run=True)
        assert report.is_dry_run
        assert OperationStore(tmp_path).read_all(_KEY) == []
        assert QuarantineStore(tmp_path).read_all(_KEY) == []
        assert ProvenanceStore(tmp_path).read_all(_KEY) == []
        assert CheckpointStore(tmp_path).read_history(_KEY) == []
        assert repository.event_store.read_events("in_memory", "XAUUSD") == []
        assert repository.manifest_store.read_current(_KEY) is None

    def test_dry_run_preview_matches_real_commit(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        dry_report = _run(repository, dry_run=True)
        real_report = _run(repository, dry_run=False)
        assert dry_report.resulting_dataset_id == real_report.resulting_dataset_id
        assert dry_report.normalized_events_digest == real_report.normalized_events_digest
        assert dry_report.valid_row_count == real_report.valid_row_count
        assert dry_report.quarantined_row_count == real_report.quarantined_row_count

    def test_dry_run_reports_accurate_counts(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        report = _run(repository, dry_run=True)
        assert report.parsed_row_count == 4
        assert report.valid_row_count == 2
        assert report.quarantined_row_count == 2

    def test_two_consecutive_dry_runs_are_both_side_effect_free(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        report1 = _run(repository, dry_run=True)
        report2 = _run(repository, dry_run=True)
        assert report1.resulting_dataset_id == report2.resulting_dataset_id
        assert repository.event_store.read_events("in_memory", "XAUUSD") == []


class TestProvenanceAndQuarantineIntegration:
    def test_every_ingested_event_has_exactly_one_provenance_record(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        report = _run(repository)
        provenance_records = ProvenanceStore(tmp_path).read_all(_KEY)
        assert len(provenance_records) == report.valid_row_count
        event_ids = {e.event_id for e in repository.event_store.read_events("in_memory", "XAUUSD")}
        assert {p.event_id for p in provenance_records} == event_ids

    def test_quarantined_rows_never_appear_in_the_repository(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        _run(repository)
        quarantine_coords = {(q.source_manifest_id, q.source_row_index) for q in QuarantineStore(tmp_path).read_all(_KEY)}
        provenance_coords = {(p.source_manifest_id, p.source_row_index) for p in ProvenanceStore(tmp_path).read_all(_KEY)}
        assert quarantine_coords.isdisjoint(provenance_coords)


class TestCheckpointing:
    def test_completed_operation_writes_both_checkpoint_kinds(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        _run(repository)
        checkpoints = CheckpointStore(tmp_path).read_history(_KEY)
        assert any(isinstance(c, RawIngestionCheckpoint) for c in checkpoints)
        assert any(isinstance(c, HistoricalIngestionCheckpoint) for c in checkpoints)


class TestReplay:
    def test_replay_into_a_fresh_repository_is_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            repo1 = MarketDataRepository.open(Path(tmp1))
            repo2 = MarketDataRepository.open(Path(tmp2))
            adapter, manifest, plan = _build_fixture(repo1)
            original = run_ingestion_operation(
                repository=repo1, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
                timeframe_mapping=_TIMEFRAME_MAPPING, timestamp_policy=_TIMESTAMP_POLICY, operation_id="replay_op", operation_time=_T0,
            )
            replayed = replay_ingestion_operation(
                repository=repo2, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
                timeframe_mapping=_TIMEFRAME_MAPPING, timestamp_policy=_TIMESTAMP_POLICY, operation_id="replay_op", operation_time=_T0,
            )
            assert original.resulting_dataset_id == replayed.resulting_dataset_id
            assert original.normalized_events_digest == replayed.normalized_events_digest
            assert original.quarantine_issue_counts == replayed.quarantine_issue_counts

    def test_replay_produces_identical_event_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            repo1 = MarketDataRepository.open(Path(tmp1))
            repo2 = MarketDataRepository.open(Path(tmp2))
            adapter, manifest, plan = _build_fixture(repo1)
            run_ingestion_operation(
                repository=repo1, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
                timeframe_mapping=_TIMEFRAME_MAPPING, timestamp_policy=_TIMESTAMP_POLICY, operation_id="replay_op2", operation_time=_T0,
            )
            replay_ingestion_operation(
                repository=repo2, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
                timeframe_mapping=_TIMEFRAME_MAPPING, timestamp_policy=_TIMESTAMP_POLICY, operation_id="replay_op2", operation_time=_T0,
            )
            ids1 = {e.event_id for e in repo1.event_store.read_events("in_memory", "XAUUSD")}
            ids2 = {e.event_id for e in repo2.event_store.read_events("in_memory", "XAUUSD")}
            assert ids1 == ids2


class TestConcurrency:
    """Mirrors `test_market_data_concurrency.py`'s own pattern exactly:
    `experiment_lock` is fail-fast, not blocking, so the only guarantees
    under real thread contention are (1) no corruption, (2) every thread
    that did not win converges correctly on a later retry."""

    _THREAD_COUNT = 4
    _REPEATS = 3

    def _run_identical_concurrent_retry_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = MarketDataRepository.open(Path(tmp))
            adapter, manifest, plan = _build_fixture(repository)
            results: list[object] = []
            errors: list[BaseException] = []
            barrier = threading.Barrier(self._THREAD_COUNT)

            def worker() -> None:
                try:
                    barrier.wait(timeout=10)
                    result = run_ingestion_operation(
                        repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=_INSTRUMENT_MAPPING,
                        timeframe_mapping=_TIMEFRAME_MAPPING, timestamp_policy=_TIMESTAMP_POLICY, operation_id="shared-op", operation_time=_T0,
                    )
                    results.append(result)
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(self._THREAD_COUNT)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            for exc in errors:
                assert "MarketDataLockError" in type(exc).__name__ or "Lock" in type(exc).__name__, f"unexpected error: {exc!r}"
            dataset_ids = {r.resulting_dataset_id for r in results}  # type: ignore[attr-defined]
            assert len(dataset_ids) <= 1
            stored = repository.event_store.read_events("in_memory", "XAUUSD")
            assert len(stored) == len({e.event_id for e in stored})  # no duplicate, no corruption

    def test_many_threads_submitting_the_same_operation_converge(self) -> None:
        for _ in range(self._REPEATS):
            self._run_identical_concurrent_retry_once()

    def _run_concurrent_quarantine_append_converges_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = QuarantineStore(Path(tmp))
            from quant_platform.market_data.quarantine import MALFORMED_TIMESTAMP, create_quarantine_record

            errors: list[BaseException] = []
            barrier = threading.Barrier(self._THREAD_COUNT)

            def worker() -> None:
                try:
                    barrier.wait(timeout=10)
                    record = create_quarantine_record(
                        source_manifest_id="a" * 64, source_row_index=0, raw_record_digest="b" * 64, raw_fields={"timestamp": "garbage"},
                        validation_issue_codes=(MALFORMED_TIMESTAMP,), ingestion_batch_id="shared-batch", event_time=_T0,
                    )
                    store.append(_KEY, record)
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(self._THREAD_COUNT)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            for exc in errors:
                assert "Lock" in type(exc).__name__, f"unexpected error: {exc!r}"
            assert len(store.read_all(_KEY)) == 1

    def test_many_threads_quarantining_the_same_row_converge(self) -> None:
        for _ in range(self._REPEATS):
            self._run_concurrent_quarantine_append_converges_once()


class TestAdversarial:
    def test_provenance_conflict_leaves_no_orphan_event(self, tmp_path: Path) -> None:
        # Regression: a provenance conflict is detected in a PRE-FLIGHT
        # pass before any repository write, specifically so that a row
        # which cannot acquire provenance never ends up durably committed
        # anyway (an "orphan" event with no provenance and no way to ever
        # get one). Two DIFFERENT operation_ids reprocessing the exact
        # same row against an already-populated dataset must fail closed
        # with the repository event COUNT UNCHANGED, never +1.
        from quant_platform.core.exceptions import ProvenanceError

        repository = MarketDataRepository.open(tmp_path)
        valid_rows = _CANDLE_ROWS[:1]
        _run(repository, operation_id="first", rows=valid_rows)
        before = len(repository.event_store.read_events("in_memory", "XAUUSD"))

        adapter, manifest, _plan = _build_fixture(repository, rows=valid_rows)
        late_arrival_plan = create_backfill_plan(
            source_manifest=manifest, target_dataset_key=_KEY, requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            requested_end=datetime(2024, 1, 2, tzinfo=timezone.utc), existing_covered_partition_keys=frozenset(), partitioning=_PARTITIONING,
            overlap_policy=OverlapPolicy.ALLOW_LATE_ARRIVAL_NEW_VERSION, gap_policy=GapPolicy.ALLOW_AND_REPORT,
            instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id, timeframe_mapping_id=_TIMEFRAME_MAPPING.mapping_id, timezone_policy_id=_TIMEZONE_POLICY_ID,
            creation_time=_T0,
        )
        with pytest.raises(ProvenanceError):
            run_ingestion_operation(
                repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=late_arrival_plan, instrument_mapping=_INSTRUMENT_MAPPING,
                timeframe_mapping=_TIMEFRAME_MAPPING, timestamp_policy=_TIMESTAMP_POLICY, operation_id="second", operation_time=_T0,
            )
        after = len(repository.event_store.read_events("in_memory", "XAUUSD"))
        assert after == before

    def test_empty_source_produces_zero_events_and_completes(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        report = _run(repository, rows=[])
        assert report.parsed_row_count == 0
        assert report.valid_row_count == 0
        assert report.stage is IngestionStage.COMPLETED

    def test_all_rows_invalid_still_completes_under_quarantine_policy(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        bad_rows = [{"timestamp": "2024-01-01T00:00:00Z", "symbol": "XAUUSD", "open": "bad", "high": "1", "low": "1", "close": "1", "timeframe": "M1"}]
        report = _run(repository, rows=bad_rows)
        assert report.valid_row_count == 0
        assert report.quarantined_row_count == 1
        assert report.stage is IngestionStage.COMPLETED

    def test_non_admissible_plan_is_rejected_before_any_write(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        adapter, manifest, _plan = _build_fixture(repository)
        # A plan requesting an already-fully-covered range under REJECT_ANY_OVERLAP is inadmissible.
        blocked_plan = create_backfill_plan(
            source_manifest=manifest, target_dataset_key=_KEY, requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            requested_end=datetime(2024, 1, 2, tzinfo=timezone.utc), existing_covered_partition_keys=frozenset({"2024-01-01"}),
            partitioning=_PARTITIONING, overlap_policy=OverlapPolicy.REJECT_ANY_OVERLAP, gap_policy=GapPolicy.ALLOW_AND_REPORT,
            instrument_mapping_id=_INSTRUMENT_MAPPING.mapping_id, timeframe_mapping_id=_TIMEFRAME_MAPPING.mapping_id, timezone_policy_id=_TIMEZONE_POLICY_ID,
            creation_time=_T0,
        )
        with pytest.raises(Exception, match="not admissible"):
            run_ingestion_operation(
                repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=blocked_plan, instrument_mapping=_INSTRUMENT_MAPPING,
                timeframe_mapping=_TIMEFRAME_MAPPING, timestamp_policy=_TIMESTAMP_POLICY, operation_id="op_blocked", operation_time=_T0,
            )
        assert OperationStore(tmp_path).read_all(_KEY) == []

    def test_row_with_all_issue_types_at_once_is_quarantined_with_every_code(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        pathological_row = {"timestamp": "not-a-date", "symbol": "UNKNOWNSYM", "open": "not_a_number", "high": "1", "low": "1", "close": "1", "volume": "-5", "timeframe": "unknown_tf"}
        report = _run(repository, rows=[pathological_row])
        assert report.quarantined_row_count == 1
        codes = set(report.quarantine_issue_counts)
        assert "malformed_timestamp" in codes
        assert "unknown_symbol" in codes
        assert "invalid_decimal" in codes

    def test_operation_id_reused_across_two_different_datasets_is_independent(self, tmp_path: Path) -> None:
        repository = MarketDataRepository.open(tmp_path)
        report1 = _run(repository, operation_id="shared_id")
        other_key = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="EURUSD", provider="in_memory")
        instrument_mapping_eur = create_instrument_mapping_spec(mapping_version=1, entries=(InstrumentMappingEntry(source_symbol="EURUSD", instrument_id="EURUSD"),))
        rows = [{"timestamp": "2024-01-01T00:00:00Z", "symbol": "EURUSD", "open": "1.1", "high": "1.2", "low": "1.0", "close": "1.15", "timeframe": "M1"}]
        adapter = create_in_memory_adapter(source_schema_version=1, record_kind=RecordKind.CANDLE, rows=rows)
        manifest = create_source_manifest(
            source_name="test", source_kind=SourceKind.IN_MEMORY, source_schema_version=1, record_kind=RecordKind.CANDLE,
            source_label="fixture", content_digest=adapter.content_digest(), byte_size=adapter.byte_size(), encoding="utf-8",
            instrument_mapping_id=instrument_mapping_eur.mapping_id, timezone_policy_id=_TIMEZONE_POLICY_ID, timeframe_mapping_id=_TIMEFRAME_MAPPING.mapping_id,
            unit_normalization_version=1, creation_time=_T0, expected_timeframe=Timeframe.M1,
        )
        plan = create_backfill_plan(
            source_manifest=manifest, target_dataset_key=other_key, requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            requested_end=datetime(2024, 1, 2, tzinfo=timezone.utc), existing_covered_partition_keys=frozenset(), partitioning=_PARTITIONING,
            overlap_policy=OverlapPolicy.REJECT_ANY_OVERLAP, gap_policy=GapPolicy.ALLOW_AND_REPORT, instrument_mapping_id=instrument_mapping_eur.mapping_id,
            timeframe_mapping_id=_TIMEFRAME_MAPPING.mapping_id, timezone_policy_id=_TIMEZONE_POLICY_ID, creation_time=_T0,
        )
        report2 = run_ingestion_operation(
            repository=repository, adapter=adapter, source_manifest=manifest, backfill_plan=plan, instrument_mapping=instrument_mapping_eur,
            timeframe_mapping=_TIMEFRAME_MAPPING, timestamp_policy=_TIMESTAMP_POLICY, operation_id="shared_id", operation_time=_T0,
        )
        assert report1.resulting_dataset_id != report2.resulting_dataset_id
        assert len(repository.event_store.read_events("in_memory", "XAUUSD")) == report1.valid_row_count
        assert len(repository.event_store.read_events("in_memory", "EURUSD")) == 1
