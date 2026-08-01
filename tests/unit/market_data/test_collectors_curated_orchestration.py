"""Multi-series curated backfill orchestration tests (Milestone 10,
Phase 4B) -- the 12-stage `CuratedOperationStage` state machine."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from _curated_test_helpers import (
    CORE_METADATA_BODIES,
    OBS_END,
    OBS_START,
    T0,
    default_availability_policies,
    default_core_registry,
    default_rate_limit_policy,
    default_retry_policy,
    default_revision_policy,
    fresh_repository_and_cache,
    observations_body,
)

from quant_platform.core.exceptions import (
    CollectorOrchestrationConflictError,
    CollectorOrchestrationStateError,
)
from quant_platform.market_data.collectors.curated.backfill import CachePolicy, create_curated_backfill_spec
from quant_platform.market_data.collectors.curated.datasets import (
    CombinedUniverseManifestStore,
    ComponentDatasetManifestStore,
)
from quant_platform.market_data.collectors.curated.macro_observation import CuratedObservationStore
from quant_platform.market_data.collectors.curated.orchestration import (
    CuratedOperationStage,
    run_curated_backfill_operation,
)
from quant_platform.market_data.collectors.protocols import TransportRequest, TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode

# Sorted processing order for the 4 core series is always this.
CORE_ORDER = ("CPIAUCSL", "DFF", "DFII10", "DGS10")

_DEFAULT_ROWS = {
    "CPIAUCSL": [
        {"date": "2024-01-01", "value": "308.417", "realtime_start": "2024-02-13", "realtime_end": "9999-12-31"},
        {"date": "2024-02-01", "value": "310.326", "realtime_start": "2024-03-12", "realtime_end": "9999-12-31"},
    ],
    "DFF": [
        {"date": "2024-01-02", "value": "5.33", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"},
        {"date": "2024-01-06", "value": ".", "realtime_start": "2024-01-06", "realtime_end": "9999-12-31"},
    ],
    "DFII10": [{"date": "2024-01-02", "value": "1.85", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"}],
    "DGS10": [{"date": "2024-01-02", "value": "4.02", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"}],
}


@dataclass
class FakeTransport:
    responses: list[object] = field(default_factory=list)
    calls: list[TransportRequest] = field(default_factory=list)

    def get(self, request: TransportRequest) -> TransportResponse:
        self.calls.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, TransportResponse)
        return item


def _resp(body: bytes) -> TransportResponse:
    return TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=body, final_url="x")


def _responses_for(series_ids: tuple[str, ...], rows_by_series: dict[str, list[dict]] | None = None, metadata_by_series: dict[str, bytes] | None = None) -> list[object]:
    rows = rows_by_series if rows_by_series is not None else _DEFAULT_ROWS
    metadata = metadata_by_series if metadata_by_series is not None else CORE_METADATA_BODIES
    out: list[object] = []
    for series_id in series_ids:
        out.append(_resp(metadata[series_id]))
        out.append(_resp(observations_body(rows[series_id])))
    return out


def _setup(series_ids: tuple[str, ...] = CORE_ORDER, namespace: str = "xauusd_macro_orch"):
    root, repository, cache = fresh_repository_and_cache()
    registry = default_core_registry()
    availability_policies = default_availability_policies()
    revision_policy = default_revision_policy()
    retry_policy = default_retry_policy()
    rate_limit_policy = default_rate_limit_policy()
    backfill_spec = create_curated_backfill_spec(
        registry=registry, selected_series_ids=series_ids, observation_start=OBS_START, observation_end=OBS_END,
        revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace=namespace,
    )
    return root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec


def _run(*, repository, cache, registry, backfill_spec, availability_policies, revision_policy, retry_policy, rate_limit_policy, transport, operation_id="op-1", dry_run=False, api_key=None):
    return run_curated_backfill_operation(
        repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies,
        revision_policy=revision_policy, operation_id=operation_id, operation_time=T0, transport=transport, api_key=api_key,
        retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0),
        credential_mode=CredentialMode.ANONYMOUS, dry_run=dry_run,
    )


class TestAllSeriesSucceed:
    def test_reaches_completed_with_complete_status(self) -> None:
        _root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup()
        transport = FakeTransport(responses=_responses_for(CORE_ORDER))
        report = _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport)
        assert report.stage is CuratedOperationStage.COMPLETED
        assert report.completeness_status == "complete"
        assert all(o.succeeded for o in report.series_outcomes)

    def test_missing_value_quarantined_under_default_policy(self) -> None:
        _root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup()
        transport = FakeTransport(responses=_responses_for(CORE_ORDER))
        report = _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport)
        dff = next(o for o in report.series_outcomes if o.series_id == "DFF")
        assert dff.quarantined_row_count == 1
        assert dff.valid_row_count == 1

    def test_daily_and_monthly_component_manifests_keep_own_frequency(self) -> None:
        root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup()
        transport = FakeTransport(responses=_responses_for(CORE_ORDER))
        _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport)
        component_store = ComponentDatasetManifestStore(root)
        assert component_store.read_current("fred", "DGS10").native_frequency == "D"
        assert component_store.read_current("fred", "CPIAUCSL").native_frequency == "M"

    def test_combined_manifest_binds_exact_component_versions(self) -> None:
        root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup()
        transport = FakeTransport(responses=_responses_for(CORE_ORDER))
        _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport)
        component_store = ComponentDatasetManifestStore(root)
        combined_store = CombinedUniverseManifestStore(root)
        combined = combined_store.read_current(backfill_spec.target_dataset_namespace)
        assert combined is not None
        for series_id in CORE_ORDER:
            assert combined.component_manifest_ids[series_id] == component_store.read_current("fred", series_id).component_manifest_id

    def test_point_in_time_availability_reflects_realtime_start_not_observation_month(self) -> None:
        root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup()
        transport = FakeTransport(responses=_responses_for(CORE_ORDER))
        _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport)
        obs_store = CuratedObservationStore(root)
        jan_cpi = next(o for o in obs_store.read_observations("fred", "CPIAUCSL") if o.observation_date == "2024-01-01")
        assert jan_cpi.availability_time.month == 2  # published mid-Feb, not the January observation month


class TestPartialSuccessAndFailFast:
    def _setup_with_one_bad_series(self):
        root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup()
        bad_metadata = dict(CORE_METADATA_BODIES)
        # Forge DFF's metadata to claim it's actually a different series id -- triggers fail-closed drift.
        import json

        forged = json.loads(CORE_METADATA_BODIES["DFF"])
        forged["seriess"][0]["id"] = "SOMETHING_ELSE"
        bad_metadata["DFF"] = json.dumps(forged).encode()
        responses = _responses_for(CORE_ORDER, metadata_by_series=bad_metadata)
        # DFF's observation response is never consumed since metadata verification fails first;
        # drop it so the fake transport's call sequence matches exactly what orchestration will issue.
        del responses[3]  # DFF observation body (index 2=meta, 3=obs in CPIAUCSL,DFF,... order)
        return root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec, responses

    def test_partial_success_when_fail_fast_false(self) -> None:
        _root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec, responses = self._setup_with_one_bad_series()
        backfill_spec2 = create_curated_backfill_spec(
            registry=registry, selected_series_ids=CORE_ORDER, observation_start=OBS_START, observation_end=OBS_END,
            revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace=backfill_spec.target_dataset_namespace, fail_fast=False,
        )
        transport = FakeTransport(responses=responses)
        report = _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec2, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport)
        assert report.completeness_status == "partial"
        dff_outcome = next(o for o in report.series_outcomes if o.series_id == "DFF")
        assert not dff_outcome.succeeded
        assert all(o.succeeded for o in report.series_outcomes if o.series_id != "DFF")

    def test_fail_fast_raises_and_commits_nothing(self) -> None:
        root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec, responses = self._setup_with_one_bad_series()
        transport = FakeTransport(responses=responses)  # backfill_spec default fail_fast=True
        with pytest.raises(CollectorOrchestrationStateError):
            _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport)
        combined_store = CombinedUniverseManifestStore(root)
        assert combined_store.read_current(backfill_spec.target_dataset_namespace) is None

    def test_zero_successes_raises(self) -> None:
        _root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup(series_ids=("DFF",))
        import json

        forged = json.loads(CORE_METADATA_BODIES["DFF"])
        forged["seriess"][0]["id"] = "SOMETHING_ELSE"
        responses = [_resp(json.dumps(forged).encode())]
        backfill_spec2 = create_curated_backfill_spec(
            registry=registry, selected_series_ids=("DFF",), observation_start=OBS_START, observation_end=OBS_END,
            revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace=backfill_spec.target_dataset_namespace, fail_fast=False,
        )
        transport = FakeTransport(responses=responses)
        with pytest.raises(CollectorOrchestrationStateError):
            _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec2, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport)


class TestCachedReplayAndIdempotency:
    def test_cached_replay_makes_zero_network_calls(self) -> None:
        """A cached-replay pass reuses the SAME `operation_id` as the
        original run (mirroring `acceptance.py`'s own real-vs-replay
        pattern) -- `ProvenanceRecord` identity is bound to
        `ingestion_batch_id=operation_id`, so a genuinely different
        operation_id reprocessing identical data is a DIFFERENT batch,
        not an idempotent retry, and is correctly rejected elsewhere
        (see `TestCachedReplayAndIdempotency.
        test_conflicting_content_digest_same_operation_id_raises` for the
        conflict path; a same-operation_id, same-content_digest replay is
        the only case guaranteed idempotent)."""
        _root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup()
        transport = FakeTransport(responses=_responses_for(CORE_ORDER))
        first = _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport, operation_id="op-a")

        class ForbiddenTransport:
            def get(self, _request: object) -> object:
                raise AssertionError("cached replay must not touch the network")

        second = _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=ForbiddenTransport(), operation_id="op-a")
        assert second.completeness_status == "complete"
        assert second.combined_manifest_id == first.combined_manifest_id

    def test_exact_retry_same_operation_id_is_idempotent(self) -> None:
        root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup()
        transport = FakeTransport(responses=_responses_for(CORE_ORDER))
        first = _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport, operation_id="op-fixed")
        transport2 = FakeTransport(responses=_responses_for(CORE_ORDER))
        second = _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport2, operation_id="op-fixed")
        assert first.combined_manifest_id == second.combined_manifest_id
        obs_store = CuratedObservationStore(root)
        assert len(obs_store.read_observations("fred", "DGS10")) == 1  # never duplicated

    def test_conflicting_content_digest_same_operation_id_raises(self) -> None:
        _root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup()
        transport = FakeTransport(responses=_responses_for(CORE_ORDER))
        _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport, operation_id="op-shared")

        other_spec = create_curated_backfill_spec(
            registry=registry, selected_series_ids=CORE_ORDER, observation_start=OBS_START, observation_end=OBS_END,
            revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace=backfill_spec.target_dataset_namespace, page_size=50,
        )
        assert other_spec.backfill_plan_id != backfill_spec.backfill_plan_id
        transport2 = FakeTransport(responses=_responses_for(CORE_ORDER))
        with pytest.raises(CollectorOrchestrationConflictError):
            _run(repository=repository, cache=cache, registry=registry, backfill_spec=other_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport2, operation_id="op-shared")

    def test_different_operation_id_reprocessing_same_rows_is_rejected_not_duplicated(self) -> None:
        """A DIFFERENT operation_id (a distinct ingestion batch) touching
        the exact same source rows a prior operation already committed
        must never silently duplicate or silently overwrite provenance --
        it fails loudly with `ProvenanceError` instead."""
        from quant_platform.core.exceptions import ProvenanceError

        _root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup()
        _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=FakeTransport(responses=_responses_for(CORE_ORDER)), operation_id="op-first")
        with pytest.raises(ProvenanceError):
            _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=FakeTransport(responses=_responses_for(CORE_ORDER)), operation_id="op-second")


class TestDryRun:
    def test_dry_run_commits_no_business_records(self) -> None:
        root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup()
        transport = FakeTransport(responses=_responses_for(CORE_ORDER))
        report = _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport, dry_run=True)
        assert report.is_dry_run
        assert report.stage is CuratedOperationStage.SERIES_DATASETS_COMMITTED
        obs_store = CuratedObservationStore(root)
        assert obs_store.read_observations("fred", "DGS10") == []
        combined_store = CombinedUniverseManifestStore(root)
        assert combined_store.read_current(backfill_spec.target_dataset_namespace) is None

    def test_dry_run_still_writes_cache_bytes(self) -> None:
        _root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup()
        transport = FakeTransport(responses=_responses_for(CORE_ORDER))
        _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=transport, dry_run=True)
        assert len(transport.calls) == 8  # 4 series x (metadata + observations), all fetched fresh


class TestRevisionCreatesNewVersion:
    def test_new_observation_appended_mints_new_component_and_combined_version(self) -> None:
        root, repository, cache, registry, availability_policies, revision_policy, retry_policy, rate_limit_policy, backfill_spec = _setup(series_ids=("DGS10",))
        responses1 = _responses_for(("DGS10",))
        _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=FakeTransport(responses=responses1), operation_id="op-v1")
        component_store = ComponentDatasetManifestStore(root)
        combined_store = CombinedUniverseManifestStore(root)
        v1_component = component_store.read_current("fred", "DGS10").component_manifest_id
        v1_combined = combined_store.read_current(backfill_spec.target_dataset_namespace).combined_manifest_id

        new_rows = {"DGS10": [*_DEFAULT_ROWS["DGS10"], {"date": "2024-01-03", "value": "4.05", "realtime_start": "2024-01-03", "realtime_end": "9999-12-31"}]}
        backfill_spec2 = create_curated_backfill_spec(
            registry=registry, selected_series_ids=("DGS10",), observation_start=OBS_START, observation_end=OBS_END, cache_policy=CachePolicy.FORCE_FRESH,
            revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace=backfill_spec.target_dataset_namespace,
        )
        responses2 = _responses_for(("DGS10",), rows_by_series=new_rows)
        _run(repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec2, availability_policies=availability_policies, revision_policy=revision_policy, retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, transport=FakeTransport(responses=responses2), operation_id="op-v2")
        assert component_store.read_current("fred", "DGS10").component_manifest_id != v1_component
        assert combined_store.read_current(backfill_spec.target_dataset_namespace).combined_manifest_id != v1_combined
        assert component_store.current_version("fred", "DGS10") == 2
