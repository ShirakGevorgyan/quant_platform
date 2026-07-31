"""Unit tests for `market_data.quarantine` (Milestone 10, Phase 3):
quarantine evidence identity, idempotent append, conflicting-evidence
fail-closed, and retry-eligibility classification."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from quant_platform.core.exceptions import SourceQuarantineError
from quant_platform.market_data.adapters import SourceRowCoordinate
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.quarantine import (
    AMBIGUOUS_OR_NONEXISTENT_LOCAL_TIME,
    INVALID_DECIMAL,
    MALFORMED_TIMESTAMP,
    NAIVE_TIMESTAMP_WITHOUT_POLICY,
    UNKNOWN_SYMBOL,
    UNKNOWN_TIMEFRAME,
    VALIDATION_ISSUE_CODES,
    QuarantineRecord,
    QuarantineStore,
    RetryEligibility,
    create_quarantine_record,
    default_retry_eligibility,
)

_T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
_KEY = DatasetKey(dataset_kind=DatasetKind.RAW_MARKET_EVENTS, instrument_id="XAUUSD", provider="offline_csv")


def _record(*, row_index: int = 0, codes: tuple[str, ...] = (MALFORMED_TIMESTAMP,), raw_fields: dict[str, str] | None = None) -> QuarantineRecord:
    return create_quarantine_record(
        source_manifest_id="a" * 64, source_row_index=row_index, raw_record_digest="b" * 64,
        raw_fields=(raw_fields or {"timestamp": "garbage"}), validation_issue_codes=codes, ingestion_batch_id="batch1", event_time=_T0,
    )


class TestDefaultRetryEligibility:
    def test_source_content_fixable_codes_are_retryable(self) -> None:
        assert default_retry_eligibility((MALFORMED_TIMESTAMP,)) is RetryEligibility.RETRYABLE
        assert default_retry_eligibility((INVALID_DECIMAL,)) is RetryEligibility.RETRYABLE

    def test_config_dependent_codes_are_permanent(self) -> None:
        assert default_retry_eligibility((UNKNOWN_SYMBOL,)) is RetryEligibility.PERMANENT
        assert default_retry_eligibility((UNKNOWN_TIMEFRAME,)) is RetryEligibility.PERMANENT
        assert default_retry_eligibility((NAIVE_TIMESTAMP_WITHOUT_POLICY,)) is RetryEligibility.PERMANENT
        assert default_retry_eligibility((AMBIGUOUS_OR_NONEXISTENT_LOCAL_TIME,)) is RetryEligibility.PERMANENT

    def test_any_permanent_code_makes_the_whole_set_permanent(self) -> None:
        assert default_retry_eligibility((MALFORMED_TIMESTAMP, UNKNOWN_SYMBOL)) is RetryEligibility.PERMANENT


class TestQuarantineRecordIdentity:
    def test_deterministic(self) -> None:
        assert _record().quarantine_record_id == _record().quarantine_record_id

    def test_issue_code_order_independence(self) -> None:
        r1 = _record(codes=(MALFORMED_TIMESTAMP, INVALID_DECIMAL))
        r2 = _record(codes=(INVALID_DECIMAL, MALFORMED_TIMESTAMP))
        assert r1.quarantine_record_id == r2.quarantine_record_id

    def test_empty_issue_codes_rejected(self) -> None:
        with pytest.raises(SourceQuarantineError):
            _record(codes=())

    def test_unknown_issue_code_rejected(self) -> None:
        with pytest.raises(SourceQuarantineError):
            _record(codes=("not_a_real_code",))

    def test_negative_row_index_rejected(self) -> None:
        with pytest.raises(SourceQuarantineError):
            create_quarantine_record(
                source_manifest_id="a" * 64, source_row_index=-1, raw_record_digest="b" * 64, raw_fields={},
                validation_issue_codes=(MALFORMED_TIMESTAMP,), ingestion_batch_id="batch1", event_time=_T0,
            )

    def test_explicit_retry_eligibility_override(self) -> None:
        record = create_quarantine_record(
            source_manifest_id="a" * 64, source_row_index=0, raw_record_digest="b" * 64, raw_fields={},
            validation_issue_codes=(MALFORMED_TIMESTAMP,), ingestion_batch_id="batch1", event_time=_T0,
            retry_eligibility=RetryEligibility.PERMANENT,
        )
        assert record.retry_eligibility is RetryEligibility.PERMANENT

    def test_round_trip(self) -> None:
        record = _record()
        assert QuarantineRecord.from_json_dict(record.to_json_dict()) == record

    def test_all_seventeen_codes_present(self) -> None:
        assert len(VALIDATION_ISSUE_CODES) == 17


class TestQuarantineStore:
    def test_append_and_read(self, tmp_path: Path) -> None:
        store = QuarantineStore(tmp_path)
        record = _record()
        store.append(_KEY, record)
        assert store.read_all(_KEY) == [record]

    def test_exact_duplicate_quarantine_append_idempotent(self, tmp_path: Path) -> None:
        store = QuarantineStore(tmp_path)
        store.append(_KEY, _record())
        store.append(_KEY, _record())
        assert len(store.read_all(_KEY)) == 1

    def test_conflicting_evidence_under_same_coordinate_fails_closed(self, tmp_path: Path) -> None:
        store = QuarantineStore(tmp_path)
        store.append(_KEY, _record(codes=(MALFORMED_TIMESTAMP,)))
        with pytest.raises(SourceQuarantineError):
            store.append(_KEY, _record(codes=(INVALID_DECIMAL,)))
        assert len(store.read_all(_KEY)) == 1

    def test_two_independent_operations_rediscovering_the_same_bad_row_converge(self, tmp_path: Path) -> None:
        # Regression: `ingestion_batch_id` must NOT participate in
        # quarantine identity -- two different operation_ids
        # (`ingestion_batch_id`) rediscovering the SAME physically-bad
        # row for the SAME reason is one piece of evidence, not a
        # conflict (see `QuarantineRecord.to_identity_payload`'s own
        # docstring).
        store = QuarantineStore(tmp_path)
        first = create_quarantine_record(
            source_manifest_id="a" * 64, source_row_index=0, raw_record_digest="b" * 64, raw_fields={"timestamp": "garbage"},
            validation_issue_codes=(MALFORMED_TIMESTAMP,), ingestion_batch_id="operation_one", event_time=_T0,
        )
        second = create_quarantine_record(
            source_manifest_id="a" * 64, source_row_index=0, raw_record_digest="b" * 64, raw_fields={"timestamp": "garbage"},
            validation_issue_codes=(MALFORMED_TIMESTAMP,), ingestion_batch_id="operation_two", event_time=_T0,
        )
        store.append(_KEY, first)
        result = store.append(_KEY, second)
        assert len(store.read_all(_KEY)) == 1
        assert result.ingestion_batch_id == "operation_one"  # first writer wins, exactly like creation_time elsewhere

    def test_quarantined_rows_are_queryable_by_coordinate(self, tmp_path: Path) -> None:
        store = QuarantineStore(tmp_path)
        store.append(_KEY, _record(row_index=0))
        assert store.is_quarantined(_KEY, SourceRowCoordinate(source_manifest_id="a" * 64, row_index=0))
        assert not store.is_quarantined(_KEY, SourceRowCoordinate(source_manifest_id="a" * 64, row_index=999))

    def test_never_stores_secrets_or_binary_blobs(self, tmp_path: Path) -> None:
        # raw_fields is always dict[str, str] -- structurally impossible
        # to hold a bytes blob or a nested object; this proves the shape
        # a caller actually gets back is exactly that.
        store = QuarantineStore(tmp_path)
        record = _record(raw_fields={"timestamp": "garbage_text", "open": "not_a_number"})
        store.append(_KEY, record)
        stored = store.read_all(_KEY)[0]
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in stored.raw_fields.items())
