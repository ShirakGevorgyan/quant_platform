"""Offline replay determinism tests (Milestone 10, Phase 4A) -- given
only persisted request/response manifests, raw response bytes, and
mappings, `FetchMode.CACHED_REPLAY` reproduces identical ids, identical
repository records, identical provenance, and identical reports, with
ZERO network calls, across different temp roots and different
dict-insertion orderings (the practical analogue of "different
PYTHONHASHSEED" -- `compute_content_id` goes through sorted-key
canonical JSON, so it is insensitive to insertion/iteration order by
construction; changing PYTHONHASHSEED mid-process is not possible, so
this is the honest, in-process way to exercise that same property)."""

from __future__ import annotations

import tempfile
from pathlib import Path

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
from quant_platform.market_data.collectors.orchestration import FetchMode, run_fred_macro_ingestion_operation
from quant_platform.market_data.collectors.protocols import TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.reports import generate_replay_comparison_report
from quant_platform.market_data.collectors.request_manifest import CredentialMode
from quant_platform.market_data.macro import MacroEventStore
from quant_platform.market_data.manifests import DatasetKey, DatasetKind
from quant_platform.market_data.provenance import ProvenanceStore
from quant_platform.market_data.repository import MarketDataRepository


class _ForbiddenTransport:
    def get(self, request):
        raise AssertionError("offline replay must perform zero network calls")


def _env():
    root = Path(tempfile.mkdtemp())
    return root, MarketDataRepository.open(root), RawResponseCache(root)


def _unit_mapping() -> object:
    # Constructed with entries in a specific order each call; identity
    # must not depend on this.
    return create_unit_mapping_spec(unit_mapping_version=1, entries=(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT),))


def _seed_fresh_operation(repository, cache, operation_id: str):
    retry_policy = default_retry_policy()
    rate_limit_policy = default_rate_limit_policy()
    transport = FakeTransport(responses=[
        TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=fred_json_body([
            {"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02"},
        ]), final_url="https://api.stlouisfed.org/fred/series/observations"),
    ])
    return run_fred_macro_ingestion_operation(
        repository=repository, cache=cache, operation_id=operation_id, operation_time=T0, series_id="DGS10", provider="fred",
        unit_mapping=_unit_mapping(), fetch_mode=FetchMode.FRESH, observation_start=T0, observation_end=T0,
        credential_mode=CredentialMode.ANONYMOUS, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
        rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0), transport=transport,
    ), retry_policy, rate_limit_policy


class TestZeroNetworkCallsDuringReplay:
    def test_forbidden_transport_is_never_touched(self) -> None:
        _root, repository, cache = _env()
        original, retry_policy, rate_limit_policy = _seed_fresh_operation(repository, cache, "op-replay-1")
        replayed = run_fred_macro_ingestion_operation(
            repository=repository, cache=cache, operation_id="op-replay-1", operation_time=T0, series_id="DGS10", provider="fred",
            unit_mapping=_unit_mapping(), fetch_mode=FetchMode.CACHED_REPLAY, observation_start=T0, observation_end=T0,
            credential_mode=CredentialMode.ANONYMOUS, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
            transport=_ForbiddenTransport(),
        )
        assert replayed.committed_event_count == original.committed_event_count


class TestIdenticalIdsAndRecordsAcrossDifferentTempRoots:
    def test_two_independent_repositories_produce_identical_ids(self) -> None:
        """Two ENTIRELY SEPARATE temp roots, each independently fed the
        same FRESH fetch -- content-addressed identity must be identical
        despite the filesystem roots differing (operational, never part
        of identity)."""
        _root_a, repository_a, cache_a = _env()
        _root_b, repository_b, cache_b = _env()
        report_a, _rp_a, _rl_a = _seed_fresh_operation(repository_a, cache_a, "op-replay-2")
        report_b, _rp_b, _rl_b = _seed_fresh_operation(repository_b, cache_b, "op-replay-2")

        assert report_a.request_manifest_id == report_b.request_manifest_id
        assert report_a.response_manifest_id == report_b.response_manifest_id
        assert report_a.source_manifest_id == report_b.source_manifest_id
        assert report_a.normalized_events_digest == report_b.normalized_events_digest

        events_a = MacroEventStore(_root_a).read_events("fred", "DGS10")
        events_b = MacroEventStore(_root_b).read_events("fred", "DGS10")
        assert [e.event_id for e in events_a] == [e.event_id for e in events_b]

        dk = DatasetKey(dataset_kind=DatasetKind.MACRO_OBSERVATIONS, provider="fred", instrument_id="DGS10")
        prov_a = ProvenanceStore(_root_a).read_all(dk)
        prov_b = ProvenanceStore(_root_b).read_all(dk)
        assert [p.provenance_id for p in prov_a] == [p.provenance_id for p in prov_b]

    def test_replay_comparison_report_confirms_identical(self) -> None:
        _root, repository, cache = _env()
        original, retry_policy, rate_limit_policy = _seed_fresh_operation(repository, cache, "op-replay-3")
        replayed = run_fred_macro_ingestion_operation(
            repository=repository, cache=cache, operation_id="op-replay-3", operation_time=T0, series_id="DGS10", provider="fred",
            unit_mapping=_unit_mapping(), fetch_mode=FetchMode.CACHED_REPLAY, observation_start=T0, observation_end=T0,
            credential_mode=CredentialMode.ANONYMOUS, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy,
        )
        comparison = generate_replay_comparison_report(original=original, replayed=replayed)
        assert comparison["identical"] is True


class TestIdentityIsInsensitiveToDictInsertionOrder:
    """The practical, in-process analogue of "different PYTHONHASHSEED":
    `compute_content_id` always canonicalizes via SORTED keys before
    hashing, so two logically-identical payloads built with different
    dict-insertion orders must hash identically -- exactly what
    PYTHONHASHSEED-driven dict/set iteration-order variance would
    otherwise threaten if identity depended on it."""

    def test_unit_mapping_entries_built_in_different_order_hash_identically(self) -> None:
        a = create_unit_mapping_spec(
            unit_mapping_version=1,
            entries=(UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT), UnitMappingEntry(series_id="DFF", unit=MacroUnit.RATE)),
        )
        b = create_unit_mapping_spec(
            unit_mapping_version=1,
            entries=(UnitMappingEntry(series_id="DFF", unit=MacroUnit.RATE), UnitMappingEntry(series_id="DGS10", unit=MacroUnit.PERCENT)),
        )
        assert a.unit_mapping_id == b.unit_mapping_id

    def test_request_manifest_query_params_built_in_different_order_hash_identically(self) -> None:
        from quant_platform.market_data.collectors.request_manifest import create_request_manifest

        common = {
            "collector_name": "fred", "collector_version": "1.0.0", "endpoint_host": "api.stlouisfed.org", "endpoint_path": "/x",
            "canonical_headers": {}, "requested_series_or_dataset": "DGS10", "response_format": "json", "timeout_policy_id": "a" * 64,
            "retry_policy_id": "b" * 64, "rate_limit_policy_id": "c" * 64, "credential_mode": CredentialMode.ANONYMOUS, "request_time": T0,
        }
        a = create_request_manifest(canonical_query_params={"series_id": "DGS10", "file_type": "json"}, **common)
        b = create_request_manifest(canonical_query_params={"file_type": "json", "series_id": "DGS10"}, **common)
        assert a.request_manifest_id == b.request_manifest_id
