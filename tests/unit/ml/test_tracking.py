from __future__ import annotations

import threading
from pathlib import Path

import pytest

from quant_platform.core.exceptions import ArtifactCorruptionError
from quant_platform.ml.tracking import EventRecord, EventType, ExperimentEventStore

EID = "a" * 64


class TestAppendAndRead:
    def test_append_returns_sequential_records(self, tmp_path: Path) -> None:
        store = ExperimentEventStore(tmp_path)
        e1 = store.append(EID, EventType.EXPERIMENT_CREATED)
        e2 = store.append(EID, EventType.VALIDATION_STARTED)
        e3 = store.append(EID, EventType.VALIDATION_PASSED)
        assert (e1.sequence, e2.sequence, e3.sequence) == (1, 2, 3)

    def test_read_events_returns_in_order(self, tmp_path: Path) -> None:
        store = ExperimentEventStore(tmp_path)
        store.append(EID, EventType.EXPERIMENT_CREATED)
        store.append(EID, EventType.VALIDATION_STARTED)
        events = store.read_events(EID)
        assert [e.event_type for e in events] == [EventType.EXPERIMENT_CREATED, EventType.VALIDATION_STARTED]

    def test_read_events_empty_for_unknown_experiment(self, tmp_path: Path) -> None:
        store = ExperimentEventStore(tmp_path)
        assert store.read_events("b" * 64) == ()

    def test_details_round_trip(self, tmp_path: Path) -> None:
        store = ExperimentEventStore(tmp_path)
        record = store.append(EID, EventType.ARTIFACT_WRITTEN, details={"content_hash": "x", "size": 10})
        loaded = store.read_events(EID)[0]
        assert loaded.details == record.details

    def test_occurred_at_is_utc(self, tmp_path: Path) -> None:
        from quant_platform.ml.persistence import parse_utc_timestamp

        store = ExperimentEventStore(tmp_path)
        record = store.append(EID, EventType.EXPERIMENT_CREATED)
        parse_utc_timestamp(record.occurred_at)  # must not raise


class TestEventRecordValidation:
    def test_sequence_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="sequence"):
            EventRecord(schema_version=1, sequence=0, experiment_id=EID, event_type=EventType.EXPERIMENT_CREATED, occurred_at="2024-01-01T00:00:00+00:00")

    def test_invalid_experiment_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="experiment_id"):
            EventRecord(schema_version=1, sequence=1, experiment_id="bad", event_type=EventType.EXPERIMENT_CREATED, occurred_at="2024-01-01T00:00:00+00:00")

    def test_non_utc_occurred_at_rejected(self) -> None:
        with pytest.raises(ValueError, match="not timezone-aware"):
            EventRecord(schema_version=1, sequence=1, experiment_id=EID, event_type=EventType.EXPERIMENT_CREATED, occurred_at="2024-01-01T00:00:00")

    def test_non_primitive_details_rejected(self) -> None:
        with pytest.raises(ValueError):
            EventRecord(
                schema_version=1, sequence=1, experiment_id=EID, event_type=EventType.EXPERIMENT_CREATED,
                occurred_at="2024-01-01T00:00:00+00:00", details={"bad": [1, 2]},  # type: ignore[dict-item]
            )

    def test_round_trip(self) -> None:
        record = EventRecord(
            schema_version=1, sequence=1, experiment_id=EID, event_type=EventType.RUN_COMPLETED,
            occurred_at="2024-01-01T00:00:00+00:00", details={"k": "v"},
        )
        assert EventRecord.from_json_dict(record.to_json_dict()) == record


class TestCorruptionRecovery:
    def test_interrupted_final_write_is_repaired_on_read(self, tmp_path: Path) -> None:
        store = ExperimentEventStore(tmp_path)
        store.append(EID, EventType.EXPERIMENT_CREATED)
        store.append(EID, EventType.VALIDATION_STARTED)
        events_path = store._events_path(EID)
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write('{"schema_version": 1, "sequence": 3, "experiment_typ')  # truncated, no newline

        recovered = store.read_events(EID)
        assert len(recovered) == 2

    def test_append_after_repair_continues_sequence(self, tmp_path: Path) -> None:
        store = ExperimentEventStore(tmp_path)
        store.append(EID, EventType.EXPERIMENT_CREATED)
        events_path = store._events_path(EID)
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write('{"sequence": 2, "truncat')
        record = store.append(EID, EventType.VALIDATION_STARTED)
        assert record.sequence == 2

    def test_middle_line_corruption_raises(self, tmp_path: Path) -> None:
        store = ExperimentEventStore(tmp_path)
        store.append(EID, EventType.EXPERIMENT_CREATED)
        store.append(EID, EventType.VALIDATION_STARTED)
        store.append(EID, EventType.VALIDATION_PASSED)

        events_path = store._events_path(EID)
        lines = [line for line in events_path.read_text(encoding="utf-8").split("\n") if line]
        lines[1] = "NOT VALID JSON"
        events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(ArtifactCorruptionError):
            store.read_events(EID)

    def test_non_sequential_sequence_number_raises(self, tmp_path: Path) -> None:
        import json

        store = ExperimentEventStore(tmp_path)
        record1 = store.append(EID, EventType.EXPERIMENT_CREATED)
        events_path = store._events_path(EID)
        record2_broken = EventRecord(
            schema_version=1, sequence=5, experiment_id=EID, event_type=EventType.VALIDATION_STARTED,
            occurred_at=record1.occurred_at,
        )
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record2_broken.to_json_dict()) + "\n")

        with pytest.raises(ArtifactCorruptionError, match="non-sequential"):
            store.read_events(EID)


class TestConcurrency:
    def test_concurrent_appends_produce_no_duplicate_or_missing_sequence(self, tmp_path: Path) -> None:
        """`.events.lock` (via `ml.concurrency.experiment_lock`, a thin
        adapter over `historical.locking.DatasetLock`) is a fail-fast
        advisory lock, not a blocking queue -- a thread that loses the
        race for it gets `ExperimentLockError` immediately rather than
        waiting (this is `DatasetLock`'s own documented, intentional
        design, reused as-is here per this milestone's "do not duplicate
        locking logic" instruction; `experiment_lock` translates it at
        the ML boundary rather than leaking the underlying
        `DatasetLockError` type). The safety property under test is
        therefore not "every append succeeds" but "every append that DOES
        succeed gets a unique, gapless sequence number, and no append
        ever corrupts the log or silently loses another's event." Also
        tolerates `ExperimentLockError` arising from a rare, pre-existing
        Windows-specific race in `DatasetLock`'s own stale-lock reclaim
        path under many-way simultaneous FIRST lock acquisition
        (documented in `test_ml_manifests.py`'s identical concurrency
        test) -- out of scope to fix here since `DatasetLock` is reused
        Milestone 2 code; what matters is that it NEVER escapes as a raw
        `OSError`/`DatasetLockError` at this store's public boundary."""
        from quant_platform.core.exceptions import ExperimentLockError

        store = ExperimentEventStore(tmp_path)
        errors: list[BaseException] = []
        successes: list[int] = []
        lock = threading.Lock()

        def append() -> None:
            try:
                record = store.append(EID, EventType.ARTIFACT_WRITTEN)
                with lock:
                    successes.append(record.sequence)
            except ExperimentLockError as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=append) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(successes) + len(errors) == 4, "no unexpected exception type should have escaped"
        assert len(successes) == len(set(successes)), "no duplicate sequence numbers"
        events = store.read_events(EID)
        assert len(events) == len(successes)
        assert [e.sequence for e in events] == list(range(1, len(successes) + 1))

    def test_append_translates_contested_lock_deterministically(self, tmp_path: Path) -> None:
        """Non-flaky counterpart to the stress test above: the SAME
        thread holds the lock, guaranteeing (not just probabilistically
        risking) a contested acquisition, proving both the exception type
        and that the original `DatasetLockError` is preserved as the
        translated exception's cause."""
        from quant_platform.core.exceptions import DatasetLockError, ExperimentLockError
        from quant_platform.historical.locking import DatasetLock

        store = ExperimentEventStore(tmp_path)
        holder = DatasetLock(store._lock_path(EID))
        holder.acquire()
        try:
            with pytest.raises(ExperimentLockError) as exc_info:
                store.append(EID, EventType.ARTIFACT_WRITTEN)
            assert isinstance(exc_info.value.__cause__, DatasetLockError)
        finally:
            holder.release()
        # Lock released cleanly -- a subsequent append succeeds normally.
        record = store.append(EID, EventType.ARTIFACT_WRITTEN)
        assert record.sequence == 1
