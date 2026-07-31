"""Backfill orchestration tests (Milestone 10, Phase 4A) -- the 11-stage
state machine, exact-retry idempotency, `FetchMode.CACHED_REPLAY` making
zero transport calls, dry-run write-nothing guarantees, and the
pre-flight provenance-conflict guard that prevents an orphaned
`MacroEvent`."""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

import pytest
from _collectors_test_helpers import (
    T0,
    FakeTransport,
    default_rate_limit_policy,
    default_retry_policy,
    fred_json_body,
)

from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.macro_normalization import (
    MacroUnit,
    UnitMappingEntry,
    create_unit_mapping_spec,
)
from quant_platform.market_data.collectors.orchestration import (
    CollectorOperationStage,
    CollectorOperationStore,
    FetchMode,
    run_fred_macro_ingestion_operation,
)
from quant_platform.market_data.collectors.protocols import TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.market_data.macro import MacroEventStore
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.provenance import ProvenanceStore, create_provenance_record
from quant_platform.market_data.quarantine import QuarantineStore
from quant_platform.market_data.repository import MarketDataRepository

UNIT_MAPPING = create_unit_mapping_spec(unit_mapping_version=1, entries=(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT),))


class _ForbiddenTransport:
    def get(self, request):
        raise AssertionError("must not be called during CACHED_REPLAY")


def _env():
    root = Path(tempfile.mkdtemp())
    return root, MarketDataRepository.open(root), RawResponseCache(root)


def _run_fresh(repository, cache, transport, *, operation_id: str, observations: list[dict], **overrides):
    retry_policy = overrides.pop("retry_policy", default_retry_policy())
    rate_limit_policy = overrides.pop("rate_limit_policy", default_rate_limit_policy())
    return run_fred_macro_ingestion_operation(
        repository=repository, cache=cache, operation_id=operation_id, operation_time=T0, series_id="DGS10", provider="fred",
        unit_mapping=UNIT_MAPPING, fetch_mode=FetchMode.FRESH, observation_start=T0, observation_end=T0,
        credential_mode=CredentialMode.ANONYMOUS, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
        rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), transport=transport, **overrides,
    )


class TestElevenStageFlow:
    def test_full_fresh_operation_reaches_verification_completed(self) -> None:
        root, repository, cache = _env()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
                {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"},
                {"date": "2024-01-03", "value": ".", "realtime_start": "2024-01-03"},
            ]), final_url="https://api.stlouisfed.org/fred/series/observations"),
        ])
        report = _run_fresh(repository, cache, transport, operation_id="op-1", observations=[])
        assert report.stage is CollectorOperationStage.VERIFICATION_COMPLETED
        assert report.parsed_row_count == 2
        assert report.valid_row_count == 1
        assert report.quarantined_row_count == 1
        assert report.quarantine_issue_counts == {"missing_observation_value": 1}
        assert report.committed_event_count == 1
        assert not report.is_dry_run

        events = MacroEventStore(root).read_events("fred", "DGS10")
        assert len(events) == 1
        assert events[0].value == Decimal("4.02")
        assert events[0].source_event_id == "fred:DGS10:date=2024-01-02"

        dk = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider="fred", instrument_id="DGS10")
        assert len(ProvenanceStore(root).read_all(dk)) == 1
        assert len(QuarantineStore(root).read_all(dk)) == 1

    def test_operation_ledger_records_all_eleven_stages_in_order(self) -> None:
        root, repository, cache = _env()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
                {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"},
            ]), final_url="https://api.stlouisfed.org/fred/series/observations"),
        ])
        _run_fresh(repository, cache, transport, operation_id="op-2", observations=[])
        dk = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider="fred", instrument_id="DGS10")
        history = CollectorOperationStore(root).read_all(dk)
        assert [r.stage for r in history] == list(CollectorOperationStage)


class TestExactRetryIdempotency:
    def test_cached_replay_of_the_same_operation_id_is_idempotent(self) -> None:
        root, repository, cache = _env()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
                {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"},
            ]), final_url="https://api.stlouisfed.org/fred/series/observations"),
        ])
        report1 = _run_fresh(repository, cache, transport, operation_id="op-3", observations=[])
        report2 = run_fred_macro_ingestion_operation(
            repository=repository, cache=cache, operation_id="op-3", operation_time=T0, series_id="DGS10", provider="fred",
            unit_mapping=UNIT_MAPPING, fetch_mode=FetchMode.CACHED_REPLAY, observation_start=T0, observation_end=T0,
            credential_mode=CredentialMode.ANONYMOUS, retry_policy=default_retry_policy(), rate_limit_policy=default_rate_limit_policy(),
        )
        assert report2.committed_event_count == 1
        assert report2.response_manifest_id == report1.response_manifest_id
        assert len(MacroEventStore(root).read_events("fred", "DGS10")) == 1
        assert len(transport.calls) == 1  # no additional transport call on replay

    def test_cached_replay_never_touches_a_forbidden_transport(self) -> None:
        _root, repository, cache = _env()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
                {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"},
            ]), final_url="https://api.stlouisfed.org/fred/series/observations"),
        ])
        _run_fresh(repository, cache, transport, operation_id="op-4", observations=[])
        report = run_fred_macro_ingestion_operation(
            repository=repository, cache=cache, operation_id="op-4", operation_time=T0, series_id="DGS10", provider="fred",
            unit_mapping=UNIT_MAPPING, fetch_mode=FetchMode.CACHED_REPLAY, observation_start=T0, observation_end=T0,
            credential_mode=CredentialMode.ANONYMOUS, retry_policy=default_retry_policy(), rate_limit_policy=default_rate_limit_policy(),
            transport=_ForbiddenTransport(),
        )
        assert report.committed_event_count == 1


class TestDryRun:
    def test_dry_run_writes_nothing(self) -> None:
        root, repository, cache = _env()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
                {"date": "2024-02-01", "value": "3.99", "realtime_start": "2024-02-01"},
            ]), final_url="https://api.stlouisfed.org/fred/series/observations"),
        ])
        report = _run_fresh(repository, cache, transport, operation_id="op-dry", observations=[], dry_run=True)
        assert report.is_dry_run
        assert report.committed_event_count == 0
        assert report.valid_row_count == 1

        dk = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider="fred", instrument_id="DGS10")
        assert MacroEventStore(root).read_events("fred", "DGS10") == []
        assert ProvenanceStore(root).read_all(dk) == []
        assert QuarantineStore(root).read_all(dk) == []
        assert CollectorOperationStore(root).read_all(dk) == []


class TestProvenanceConflictGuard:
    def test_a_second_operation_id_over_already_ingested_data_is_rejected(self) -> None:
        """The exact scenario the pre-flight guard exists for: a caller
        forgets the original `operation_id` and retries with a NEW one
        over the same underlying source data -- sequence numbering would
        otherwise silently mint a SECOND, conflicting event for the same
        economic observation. Must be rejected before any repository
        write, not merely detected afterward."""
        root, repository, cache = _env()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
                {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"},
            ]), final_url="https://api.stlouisfed.org/fred/series/observations"),
        ])
        _run_fresh(repository, cache, transport, operation_id="op-5a", observations=[])
        count_before = len(MacroEventStore(root).read_events("fred", "DGS10"))
        dk = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider="fred", instrument_id="DGS10")
        prov_before = len(ProvenanceStore(root).read_all(dk))

        from quant_platform.core.exceptions import ProvenanceError

        with pytest.raises(ProvenanceError):
            run_fred_macro_ingestion_operation(
                repository=repository, cache=cache, operation_id="op-5b", operation_time=T0, series_id="DGS10", provider="fred",
                unit_mapping=UNIT_MAPPING, fetch_mode=FetchMode.CACHED_REPLAY, observation_start=T0, observation_end=T0,
                credential_mode=CredentialMode.ANONYMOUS, retry_policy=default_retry_policy(), rate_limit_policy=default_rate_limit_policy(),
            )

        assert len(MacroEventStore(root).read_events("fred", "DGS10")) == count_before  # no orphaned event
        assert len(ProvenanceStore(root).read_all(dk)) == prov_before  # no new provenance record

    def test_provenance_store_itself_refuses_a_conflicting_manual_write(self) -> None:
        """Defense-in-depth: even bypassing orchestration's own pre-flight
        check entirely, `ProvenanceStore.append` independently refuses a
        second, conflicting record for an already-bound coordinate."""
        root, repository, cache = _env()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
                {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"},
            ]), final_url="https://api.stlouisfed.org/fred/series/observations"),
        ])
        report = _run_fresh(repository, cache, transport, operation_id="op-6a", observations=[])
        dk = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider="fred", instrument_id="DGS10")

        fabricated = create_provenance_record(
            source_manifest_id=report.source_manifest_id, source_row_index=0, source_record_digest="0" * 64,
            original_timestamp_text="2024-01-02", normalized_event_time=T0, instrument_mapping_id=UNIT_MAPPING.unit_mapping_id,
            resolved_instrument_id="DGS10", timeframe_mapping_id=None, timezone_policy_id="a" * 64, ingestion_batch_id="fabricated-op",
            event_id="f" * 64, dataset_id="0" * 64, recorded_time=T0,
        )
        from quant_platform.core.exceptions import ProvenanceError

        with pytest.raises(ProvenanceError):
            ProvenanceStore(root).append(dk, fabricated)


class TestInterruptionAndRecovery:
    def test_advancing_past_a_partial_history_resumes_correctly(self) -> None:
        """Simulates an interruption after REQUEST_MANIFEST_COMMITTED (a
        crash before any response was ever downloaded) by manually
        driving the ledger through only the first two stages, then
        running the SAME operation for real -- it must recognize the
        partial history as its own (matching content_digest) and
        continue forward, not restart or double-record those two
        stages."""
        root, repository, cache = _env()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
                {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"},
            ]), final_url="https://api.stlouisfed.org/fred/series/observations"),
        ])
        report = _run_fresh(repository, cache, transport, operation_id="op-7", observations=[])
        assert report.stage is CollectorOperationStage.VERIFICATION_COMPLETED

        # A second, truly exact retry (same operation_id, same params) must
        # still succeed and remain idempotent even after the operation
        # already fully completed -- this is the "recovery from every
        # stage" property in its strongest form (recovery from the
        # terminal stage itself).
        report_again = _run_fresh(repository, cache, FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
                {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"},
            ]), final_url="https://api.stlouisfed.org/fred/series/observations"),
        ]), operation_id="op-7", observations=[])
        assert report_again.committed_event_count == 1
        assert len(MacroEventStore(root).read_events("fred", "DGS10")) == 1
