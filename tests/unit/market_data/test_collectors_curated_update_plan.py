"""Pure incremental update-plan tests (Milestone 10, Phase 4B) --
`create_curated_update_plan` never touches the network or the wall
clock; `planning_time`/`desired_observation_end` are always caller-
supplied."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

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

from quant_platform.market_data.collectors.curated.backfill import create_curated_backfill_spec
from quant_platform.market_data.collectors.curated.datasets import (
    CombinedUniverseManifestStore,
    ComponentDatasetManifestStore,
)
from quant_platform.market_data.collectors.curated.orchestration import run_curated_backfill_operation
from quant_platform.market_data.collectors.curated.revision_policy import (
    RevisionPolicyKind,
    create_revision_policy,
)
from quant_platform.market_data.collectors.curated.update_plan import (
    SeriesUpdateAction,
    create_curated_update_plan,
)
from quant_platform.market_data.collectors.protocols import TransportRequest, TransportResponse
from quant_platform.market_data.collectors.rate_limit import initial_bucket_state
from quant_platform.market_data.collectors.request_manifest import CredentialMode

CORE_ORDER = ("CPIAUCSL", "DFF", "DFII10", "DGS10")

_ROWS = {
    "CPIAUCSL": [{"date": "2024-01-01", "value": "308.417", "realtime_start": "2024-02-13", "realtime_end": "9999-12-31"}],
    "DFF": [{"date": "2024-01-02", "value": "5.33", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"}],
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
        assert isinstance(item, TransportResponse)
        return item


def _resp(body: bytes) -> TransportResponse:
    return TransportResponse(status_code=200, headers={"Content-Type": "application/json"}, body=body, final_url="x")


def _responses_for(series_ids: tuple[str, ...]) -> list[object]:
    out: list[object] = []
    for series_id in series_ids:
        out.append(_resp(CORE_METADATA_BODIES[series_id]))
        out.append(_resp(observations_body(_ROWS[series_id])))
    return out


def _seed_universe(namespace: str = "xauusd_macro_update_plan"):
    root, repository, cache = fresh_repository_and_cache()
    registry = default_core_registry()
    availability_policies = default_availability_policies()
    revision_policy = default_revision_policy()
    retry_policy = default_retry_policy()
    rate_limit_policy = default_rate_limit_policy()
    backfill_spec = create_curated_backfill_spec(
        registry=registry, selected_series_ids=CORE_ORDER, observation_start=OBS_START, observation_end=OBS_END,
        revision_policy_id=revision_policy.revision_policy_id, target_dataset_namespace=namespace,
    )
    run_curated_backfill_operation(
        repository=repository, cache=cache, registry=registry, backfill_spec=backfill_spec, availability_policies=availability_policies,
        revision_policy=revision_policy, operation_id="seed", operation_time=T0, transport=FakeTransport(responses=_responses_for(CORE_ORDER)),
        retry_policy=retry_policy, rate_limit_policy=rate_limit_policy, rate_limit_state=initial_bucket_state(rate_limit_policy, now=T0),
        credential_mode=CredentialMode.ANONYMOUS,
    )
    component_store = ComponentDatasetManifestStore(root)
    combined_store = CombinedUniverseManifestStore(root)
    combined = combined_store.read_current(namespace)
    assert combined is not None
    return registry, component_store, combined, revision_policy, namespace


class TestNoExistingManifest:
    def test_series_with_no_history_needs_append_from_default_start(self) -> None:
        root, _repository, _cache = fresh_repository_and_cache()
        registry = default_core_registry()
        component_store = ComponentDatasetManifestStore(root)
        revision_policy = default_revision_policy()
        plan = create_curated_update_plan(
            existing_combined_manifest=None, component_store=component_store, registry=registry, selected_series_ids=("DGS10",),
            target_dataset_namespace="fresh_ns", desired_observation_end=OBS_END, revision_policy=revision_policy, planning_time=T0,
        )
        entry = plan.entries[0]
        assert entry.action is SeriesUpdateAction.APPEND_OBSERVATIONS
        assert entry.requested_interval_start == registry.get("DGS10").default_observation_start
        assert entry.current_component_manifest_id is None


class TestNoUpdateNeeded:
    def test_desired_end_within_existing_coverage_is_no_op(self) -> None:
        registry, component_store, combined, revision_policy, namespace = _seed_universe()
        # CPIAUCSL only has one seeded row (2024-01-01), the earliest coverage_end among
        # the 4 core series -- use that as desired_end so EVERY series resolves NO_UPDATE_NEEDED.
        plan = create_curated_update_plan(
            existing_combined_manifest=combined, component_store=component_store, registry=registry, selected_series_ids=CORE_ORDER,
            target_dataset_namespace=namespace, desired_observation_end=datetime(2024, 1, 1, tzinfo=timezone.utc), revision_policy=revision_policy, planning_time=T0,
        )
        assert plan.is_exact_no_op()
        assert plan.series_requiring_update() == ()

    def test_no_op_plan_has_deterministic_identity(self) -> None:
        registry, component_store, combined, revision_policy, namespace = _seed_universe()
        desired_end = datetime(2024, 1, 2, tzinfo=timezone.utc)
        p1 = create_curated_update_plan(existing_combined_manifest=combined, component_store=component_store, registry=registry, selected_series_ids=CORE_ORDER, target_dataset_namespace=namespace, desired_observation_end=desired_end, revision_policy=revision_policy, planning_time=T0)
        p2 = create_curated_update_plan(existing_combined_manifest=combined, component_store=component_store, registry=registry, selected_series_ids=CORE_ORDER, target_dataset_namespace=namespace, desired_observation_end=desired_end, revision_policy=revision_policy, planning_time=T0 + timedelta(hours=3))
        assert p1.update_plan_id == p2.update_plan_id  # planning_time excluded from identity


class TestAppendNeeded:
    def test_desired_end_beyond_coverage_appends_from_day_after_coverage_end(self) -> None:
        registry, component_store, combined, revision_policy, namespace = _seed_universe()
        desired_end = datetime(2024, 2, 1, tzinfo=timezone.utc)
        plan = create_curated_update_plan(existing_combined_manifest=combined, component_store=component_store, registry=registry, selected_series_ids=("DGS10",), target_dataset_namespace=namespace, desired_observation_end=desired_end, revision_policy=revision_policy, planning_time=T0)
        entry = plan.entries[0]
        assert entry.action is SeriesUpdateAction.APPEND_OBSERVATIONS
        assert entry.requested_interval_start == datetime(2024, 1, 3, tzinfo=timezone.utc)
        assert entry.overlap_interval == ("2024-01-02", "2024-01-02")
        assert not plan.is_exact_no_op()
        assert plan.series_requiring_update() == ("DGS10",)


class TestRevisionRefresh:
    def test_changed_revision_policy_triggers_full_refresh(self) -> None:
        registry, component_store, combined, _old_policy, namespace = _seed_universe()
        new_policy = create_revision_policy(kind=RevisionPolicyKind.FIRST_RELEASE_ONLY)
        plan = create_curated_update_plan(existing_combined_manifest=combined, component_store=component_store, registry=registry, selected_series_ids=("DGS10",), target_dataset_namespace=namespace, desired_observation_end=datetime(2024, 1, 2, tzinfo=timezone.utc), revision_policy=new_policy, planning_time=T0)
        entry = plan.entries[0]
        assert entry.action is SeriesUpdateAction.REVISION_REFRESH
        assert entry.requested_interval_start == registry.get("DGS10").default_observation_start


class TestUnknownAndDisabledSelection:
    def test_unknown_series_reported_as_issue_not_entry(self) -> None:
        root, _repository, _cache = fresh_repository_and_cache()
        registry = default_core_registry()
        component_store = ComponentDatasetManifestStore(root)
        revision_policy = default_revision_policy()
        plan = create_curated_update_plan(existing_combined_manifest=None, component_store=component_store, registry=registry, selected_series_ids=("NOPE",), target_dataset_namespace="ns", desired_observation_end=OBS_END, revision_policy=revision_policy, planning_time=T0)
        assert plan.entries == ()
        assert "unknown_series:NOPE" in plan.plan_issues
        assert not plan.is_exact_no_op()  # issues present, not silently treated as no-op


class TestNoWallClockDependence:
    def test_desired_observation_end_and_planning_time_are_required_tz_aware(self) -> None:
        import inspect

        sig = inspect.signature(create_curated_update_plan)
        assert sig.parameters["desired_observation_end"].default is inspect.Parameter.empty
        assert sig.parameters["planning_time"].default is inspect.Parameter.empty

    def test_naive_desired_observation_end_rejected(self) -> None:
        root, _repository, _cache = fresh_repository_and_cache()
        registry = default_core_registry()
        component_store = ComponentDatasetManifestStore(root)
        revision_policy = default_revision_policy()
        with pytest.raises(Exception):  # noqa: B017 -- require_tz_aware's own exception type
            create_curated_update_plan(existing_combined_manifest=None, component_store=component_store, registry=registry, selected_series_ids=("DGS10",), target_dataset_namespace="ns", desired_observation_end=datetime(2024, 1, 1), revision_policy=revision_policy, planning_time=T0)


class TestEmptySelection:
    def test_empty_selected_series_ids_rejected(self) -> None:
        from quant_platform.core.exceptions import UpdatePlanError

        root, _repository, _cache = fresh_repository_and_cache()
        registry = default_core_registry()
        component_store = ComponentDatasetManifestStore(root)
        revision_policy = default_revision_policy()
        with pytest.raises(UpdatePlanError):
            create_curated_update_plan(existing_combined_manifest=None, component_store=component_store, registry=registry, selected_series_ids=(), target_dataset_namespace="ns", desired_observation_end=OBS_END, revision_policy=revision_policy, planning_time=T0)
