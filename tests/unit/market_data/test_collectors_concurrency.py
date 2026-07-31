"""Concurrency tests (Milestone 10, Phase 4A) -- identical concurrent
requests, conflicting concurrent writes, and orchestration retry races
never produce duplicate repository records or corrupt manifests. The
underlying locks (`ml.concurrency.experiment_lock`) are FAIL-FAST (never
block/retry), matching every other store in this repository (see
`test_collectors_cache.py`'s own note on this) -- these tests assert the
SAFETY property (no corruption, no duplication) rather than "every
thread silently succeeds," which would be the wrong expectation for a
fail-fast lock."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from _collectors_test_helpers import (
    T0,
    FakeTransport,
    default_rate_limit_policy,
    default_retry_policy,
    fred_json_body,
)

from quant_platform.core.exceptions import CollectorOrchestrationConflictError, MarketDataLockError
from quant_platform.market_data.collectors.cache import RawResponseCache
from quant_platform.market_data.collectors.macro_normalization import (
    MacroUnit,
    UnitMappingEntry,
    create_unit_mapping_spec,
)
from quant_platform.market_data.collectors.orchestration import FetchMode, run_fred_macro_ingestion_operation
from quant_platform.market_data.collectors.protocols import TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.market_data.collectors.response_manifest import CompletionStatus, create_response_manifest
from quant_platform.market_data.macro import MacroEventStore
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.provenance import ProvenanceStore
from quant_platform.market_data.repository import MarketDataRepository

UNIT_MAPPING = create_unit_mapping_spec(unit_mapping_version=1, entries=(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT),))


def _env():
    root = Path(tempfile.mkdtemp())
    return root, MarketDataRepository.open(root), RawResponseCache(root)


class TestConcurrentIdenticalCacheStore:
    def test_many_threads_storing_the_same_response_never_corrupts(self) -> None:
        cache = RawResponseCache(Path(tempfile.mkdtemp()))
        body = b"identical-body"
        manifest = create_response_manifest(
            request_manifest_id="a" * 64, http_status=200, raw_headers={"Content-Type": "application/json"}, raw_bytes=body,
            content_type="application/json", encoding="utf-8", completion_status=CompletionStatus.COMPLETE, received_time=T0,
        )
        errors: list[BaseException] = []

        def _worker() -> None:
            try:
                cache.store(manifest, body)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(isinstance(e, MarketDataLockError) for e in errors)
        assert cache.read_bytes(manifest.response_manifest_id) == body
        assert len(cache.read_responses_for_request("a" * 64)) == 1


class TestConcurrentDifferentResponseBytesForSameRequest:
    def test_two_genuinely_different_responses_are_both_retained_safely(self) -> None:
        """Not a conflict at all -- DIFFERENT bytes produce DIFFERENT
        `response_manifest_id`s (the spec's own "same semantic request,
        different response over time" case), so concurrent stores of two
        distinct response versions must both land safely, with no
        corruption of either."""
        cache = RawResponseCache(Path(tempfile.mkdtemp()))
        manifest_1 = create_response_manifest(
            request_manifest_id="a" * 64, http_status=200, raw_headers={}, raw_bytes=b"version-1", content_type="application/json",
            encoding="utf-8", completion_status=CompletionStatus.COMPLETE, received_time=T0,
        )
        manifest_2 = create_response_manifest(
            request_manifest_id="a" * 64, http_status=200, raw_headers={}, raw_bytes=b"version-2", content_type="application/json",
            encoding="utf-8", completion_status=CompletionStatus.COMPLETE, received_time=T0,
        )
        errors: list[BaseException] = []

        def _store(manifest, body) -> None:
            try:
                cache.store(manifest, body)
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_store, args=(manifest_1, b"version-1")),
            threading.Thread(target=_store, args=(manifest_2, b"version-2")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # The two response BODIES have independent lock paths (keyed by
        # their own distinct response_manifest_id), but both share the
        # SAME request_manifest_id, so they contend on ONE shared
        # request-index lock -- a fail-fast loser there is expected and
        # safe (the index update is a small, retriable append; losing
        # the race does not lose or corrupt data, see the assertions
        # below).
        assert all(isinstance(e, MarketDataLockError) for e in errors)
        assert cache.read_bytes(manifest_1.response_manifest_id) == b"version-1"
        assert cache.read_bytes(manifest_2.response_manifest_id) == b"version-2"


class TestOrchestrationRetryRace:
    def test_concurrent_exact_retries_of_the_same_operation_never_duplicate_records(self) -> None:
        root, repository, cache = _env()
        retry_policy = default_retry_policy()
        rate_limit_policy = default_rate_limit_policy()
        transport = FakeTransport(responses=[
            TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
                {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"},
            ]), final_url="https://api.stlouisfed.org/fred/series/observations"),
        ])
        run_fred_macro_ingestion_operation(
            repository=repository, cache=cache, operation_id="op-race-1", operation_time=T0, series_id="DGS10", provider="fred",
            unit_mapping=UNIT_MAPPING, fetch_mode=FetchMode.FRESH, observation_start=T0, observation_end=T0,
            credential_mode=CredentialMode.ANONYMOUS, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
            rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), transport=transport,
        )

        errors: list[BaseException] = []

        def _replay() -> None:
            try:
                run_fred_macro_ingestion_operation(
                    repository=repository, cache=cache, operation_id="op-race-1", operation_time=T0, series_id="DGS10", provider="fred",
                    unit_mapping=UNIT_MAPPING, fetch_mode=FetchMode.CACHED_REPLAY, observation_start=T0, observation_end=T0,
                    credential_mode=CredentialMode.ANONYMOUS, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_replay) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Any errors must be clean lock-contention/idempotency-conflict
        # errors -- never a sign of duplicated or corrupted state.
        acceptable = (MarketDataLockError, CollectorOrchestrationConflictError)
        assert all(isinstance(e, acceptable) for e in errors), errors

        events = MacroEventStore(root).read_events("fred", "DGS10")
        assert len(events) == 1  # never duplicated, regardless of how many threads raced

        dk = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider="fred", instrument_id="DGS10")
        assert len(ProvenanceStore(root).read_all(dk)) == 1
